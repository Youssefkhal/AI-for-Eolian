"""
Generate 1000 synthetic scenarios for pile stiffness degradation.
Uses physics-based interpolation from the 19 existing scenarios in REAL DATA.xlsx.

Method:
1. Extract normalized KL degradation curves from existing data, indexed by PI
2. Fit KL_initial = f(Gmax, v, Tp, Lp) via log-linear regression on physics variables
3. For each new scenario: sample inputs -> predict KL_initial -> interpolate degradation curve
4. Derive KR and KLR from KL using Winkler beam relationship: KR = KL*Lp^2/3, KLR = -KL*Lp/2
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf
import os, shutil

np.random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    excel_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')

    # ==========================================================
    # STEP 1: Read and parse existing 19 scenarios
    # ==========================================================
    df_orig = pd.read_excel(excel_path)
    orig_cols = list(df_orig.columns)  # preserve exact names with spaces

    df = df_orig.copy()
    df.columns = df.columns.str.strip()

    gcols = ['PI', 'Gmax', 'v', 'Dp', 'Tp', 'Lp']
    groups = list(df.groupby(gcols, sort=False))

    pis, gmaxs, vs, tps, lps = [], [], [], [], []
    kl0s = []
    curves = []  # normalized KL degradation curves (19 x 44)

    for name, grp in groups:
        pi, gmax, v, dp, tp, lp = name
        kl = grp['KL'].values  # row 0 = initial, rows 1-43 = drops

        # Reconstruct actual KL at each step
        actual = np.zeros(44)
        actual[0] = kl[0]
        for j in range(1, 44):
            actual[j] = actual[j - 1] - kl[j]

        pis.append(pi)
        gmaxs.append(gmax)
        vs.append(v)
        tps.append(tp)
        lps.append(lp)
        kl0s.append(actual[0])
        curves.append(actual / actual[0])  # normalize by initial

    pis = np.array(pis, dtype=float)
    gmaxs = np.array(gmaxs, dtype=float)
    vs = np.array(vs, dtype=float)
    tps = np.array(tps, dtype=float)
    lps = np.array(lps, dtype=float)
    kl0s = np.array(kl0s)
    curves = np.array(curves)  # shape (19, 44)

    print(f"Parsed {len(pis)} existing scenarios")

    # ==========================================================
    # STEP 2: Build degradation curve interpolator (by PI)
    # ==========================================================
    order = np.argsort(pis)
    pi_sorted = pis[order]
    curves_sorted = curves[order]

    # For each of the 44 steps, interpolate the normalized value across PI
    step_interps = []
    for s in range(44):
        f = interp1d(pi_sorted, curves_sorted[:, s], kind='linear',
                     bounds_error=False,
                     fill_value=(curves_sorted[0, s], curves_sorted[-1, s]))
        step_interps.append(f)

    def get_norm_curve(pi_val):
        """Get 44-step normalized KL curve for any PI in [0, 200]."""
        c = np.array([step_interps[s](float(pi_val)) for s in range(44)])
        c[0] = 1.0
        c = np.clip(c, 0.001, 1.0)
        # Enforce monotonically decreasing
        for j in range(1, 44):
            c[j] = min(c[j], c[j - 1])
        return c

    # ==========================================================
    # STEP 3: Fit KL_initial as function of physics variables
    # ==========================================================
    # KL_initial depends on (Gmax, v, Tp, Lp) through:
    #   Es = 2 * Gmax * (1+v)           [soil Young's modulus at G/Gmax=1]
    #   EI = Ep * pi/64 * (Dp^4-(Dp-2Tp)^4)  [pile flexural rigidity]
    #   Lp                               [pile length, affects Winkler solution]
    #
    # Use RBF interpolation for better accuracy than log-linear regression

    Ep = 210e9  # steel Young's modulus
    es_arr = 2 * gmaxs * (1 + vs)
    ei_arr = Ep * (np.pi / 64) * (5**4 - (5 - 2 * tps)**4)

    log_es = np.log(es_arr)
    log_ei = np.log(ei_arr)
    log_lp = np.log(lps)
    log_kl0 = np.log(kl0s)

    # RBF interpolation in log-space (linear kernel for stable extrapolation)
    rbf_kl0 = Rbf(log_es, log_ei, log_lp, log_kl0,
                  function='linear', smooth=0.01)

    # Clamp range: allow modest extension beyond observed data
    kl0_min = kl0s.min() * 0.5
    kl0_max = kl0s.max() * 1.5

    def predict_kl0(gmax, v, tp, lp):
        es = 2 * gmax * (1 + v)
        ei = Ep * (np.pi / 64) * (5**4 - (5 - 2 * tp)**4)
        val = np.exp(float(rbf_kl0(np.log(es), np.log(ei), np.log(lp))))
        return np.clip(val, kl0_min, kl0_max)

    # Verify on existing data
    preds = np.array([predict_kl0(gmaxs[i], vs[i], tps[i], lps[i]) for i in range(19)])
    errs = np.abs(preds / kl0s - 1) * 100
    r2 = 1 - np.sum((np.log(preds) - np.log(kl0s))**2) / np.sum((np.log(kl0s) - np.mean(np.log(kl0s)))**2)
    print(f"KL_initial model (RBF): R²={r2:.4f}, max_error={errs.max():.1f}%, mean_error={errs.mean():.1f}%")

    # Verify on existing data
    print("\nVerification on existing 19 scenarios:")
    for i in range(len(pis)):
        pred = predict_kl0(gmaxs[i], vs[i], tps[i], lps[i])
        err = abs(pred / kl0s[i] - 1) * 100
        print(f"  PI={pis[i]:3.0f}: actual={kl0s[i]:.3e}, predicted={pred:.3e}, error={err:.1f}%")

    # ==========================================================
    # STEP 4: Generate 1000 new scenarios
    # ==========================================================
    N = 1000
    new_pi = np.random.uniform(0, 200, N)
    new_gmax = np.random.uniform(15e6, 40e6, N)
    new_v = np.random.uniform(0.3, 0.4, N)
    new_tp = np.random.uniform(0.05, 0.07, N)
    new_lp = np.random.uniform(20, 40, N)

    rows = []
    for i in range(N):
        pi = new_pi[i]
        gmax = new_gmax[i]
        v = new_v[i]
        tp = new_tp[i]
        lp = new_lp[i]
        dp = 5.0
        ip = (np.pi / 64) * (dp**4 - (dp - 2 * tp)**4)
        dp_lp = dp / lp

        # Predict initial KL
        kl0 = predict_kl0(gmax, v, tp, lp)

        # Get normalized degradation curve for this PI
        nc = get_norm_curve(pi)

        # Actual KL at each step
        kl_actual = kl0 * nc

        # Convert to REAL DATA format: row 0 = initial, rows 1-43 = drops
        kl_rows = np.zeros(44)
        kl_rows[0] = kl0
        for j in range(1, 44):
            kl_rows[j] = max(0.0, kl_actual[j - 1] - kl_actual[j])

        # Derive KR and KLR from KL using Winkler beam relationship
        kr_rows = kl_rows * (lp**2 / 3)
        klr_rows = kl_rows * (-lp / 2)

        # Append 44 rows for this scenario
        for j in range(44):
            rows.append([pi, gmax, v, dp, tp, lp, ip, dp_lp,
                         kl_rows[j], kr_rows[j], klr_rows[j]])

    new_df = pd.DataFrame(rows, columns=orig_cols)

    # ==========================================================
    # STEP 5: Validate generated data
    # ==========================================================
    print(f"\n{'='*50}")
    print("GENERATED DATA STATISTICS")
    print(f"{'='*50}")

    # Check a few generated scenarios
    new_scenarios = new_df.groupby(orig_cols[:6], sort=False)
    n_check = 5
    for idx, (name, grp) in enumerate(new_scenarios):
        if idx >= n_check:
            break
        kl = grp[orig_cols[8]].values  # KL column
        initial = kl[0]
        total_drop = kl[1:].sum()
        remaining_pct = (initial - total_drop) / initial * 100
        pi_val = grp[orig_cols[0]].iloc[0]
        print(f"  Scenario {idx+1}: PI={pi_val:.1f}, KL_init={initial:.2e}, remaining={remaining_pct:.1f}%")

    # Overall stats
    kl_col = orig_cols[8]
    initials = new_df.groupby(np.arange(len(new_df)) // 44)[kl_col].first()
    print(f"\n  KL_initial range: {initials.min():.2e} to {initials.max():.2e}")
    print(f"  Original range:   {kl0s.min():.2e} to {kl0s.max():.2e}")

    # ==========================================================
    # STEP 6: Save
    # ==========================================================
    # Backup original
    backup_path = os.path.join(SCRIPT_DIR, 'REAL DATA_backup.xlsx')
    shutil.copy2(excel_path, backup_path)
    print(f"\nBackup saved: REAL DATA_backup.xlsx")

    # Combine and save
    combined = pd.concat([df_orig, new_df], ignore_index=True)
    combined.to_excel(excel_path, index=False)

    n_orig = len(df_orig)
    n_new = len(new_df)
    print(f"Original: {n_orig} rows ({n_orig // 44} scenarios)")
    print(f"Added:    {n_new} rows ({n_new // 44} scenarios)")
    print(f"Total:    {len(combined)} rows ({len(combined) // 44} scenarios)")
    print(f"\nSaved to: {excel_path}")


if __name__ == "__main__":
    main()
