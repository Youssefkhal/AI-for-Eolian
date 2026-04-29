"""
Generate 1000 synthetic scenarios for PREM DATA.xlsx.
Uses the same scenario parameters as generate_scenarios.py (seed=42)
and computes ALL 23 columns of the PREM DATA physics chain.

Verified formulas:
  I  = pi/64 * (Dp^4 - (Dp-2*Tp)^4)
  EI = Ep * I
  Gdeg = G/Gmax * Gmax
  Es = 2 * Gdeg * (1+v)
  kh = C(v) * Es^(13/12) / ((1-v^2) * Dp^(2/3) * EI^(1/12))
       where C(v) = 1.371118 + 1.52984*v
  beta = (kh_init * Dp / (4*EI))^0.25  [constant per scenario, from initial kh]
  betaLp = beta * Lp
  If betaLp < 2.5 (rigid body):
    KL = kh*Dp*Lp, KR = kh*Dp*Lp^3/3, KLR = -kh*Dp*Lp^2/2
  Else (Hetenyi beam on elastic foundation):
    KL = 4*beta^3*EI * (sinh(2bL)+sin(2bL)) / (cosh(2bL)-cos(2bL))
    KR = 2*beta*EI   * (sinh(2bL)-sin(2bL)) / (cosh(2bL)-cos(2bL))
    KLR = -2*beta^2*EI * (sinh^2(bL)+sin^2(bL)) / (sinh^2(bL)-sin^2(bL))
    Then scale: KL_j = KL_init * (kh_j/kh_init), same for KR, KLR
  Drops: drop[0] = value[0], drop[i] = value[i-1] - value[i]
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os, shutil

np.random.seed(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Ep = 210e9  # steel pile Young's modulus
Dp = 5.0    # pile diameter (constant for all scenarios)

# Fixed 44 strain levels (%)
GAMMA_VALS = np.array([
    1.0e-04, 1.9e-04, 2.8e-04, 3.7e-04, 4.6e-04, 5.5e-04, 6.4e-04, 7.3e-04,
    8.2e-04, 9.1e-04, 1.0e-03, 1.9e-03, 2.8e-03, 3.7e-03, 4.6e-03, 5.5e-03,
    6.4e-03, 7.3e-03, 8.2e-03, 9.1e-03, 1.0e-02, 1.9e-02, 2.8e-02, 3.7e-02,
    4.6e-02, 5.5e-02, 6.4e-02, 7.3e-02, 8.2e-02, 9.1e-02, 1.0e-01, 1.9e-01,
    2.8e-01, 3.7e-01, 4.6e-01, 5.5e-01, 6.4e-01, 7.3e-01, 8.2e-01, 9.1e-01,
    1.0e+00, 1.9e+00, 2.8e+00, 3.7e+00
])


def compute_kh(Es, v, EI):
    """Compute subgrade reaction modulus kh (modified Vesic formula)."""
    C_v = 1.371118 + 1.52984 * v
    return C_v * Es**(13.0/12) / ((1 - v**2) * Dp**(2.0/3) * EI**(1.0/12))


def compute_hetenyi_KL(beta, EI, Lp):
    """Hetenyi formula for KL (lateral stiffness of pile head)."""
    bL = beta * Lp
    s2 = np.sinh(2*bL)
    c2 = np.cosh(2*bL)
    si2 = np.sin(2*bL)
    co2 = np.cos(2*bL)
    return 4 * beta**3 * EI * (s2 + si2) / (c2 - co2)


def compute_hetenyi_KR(beta, EI, Lp):
    """Hetenyi formula for KR (rotational stiffness)."""
    bL = beta * Lp
    s2 = np.sinh(2*bL)
    c2 = np.cosh(2*bL)
    si2 = np.sin(2*bL)
    co2 = np.cos(2*bL)
    return 2 * beta * EI * (s2 - si2) / (c2 - co2)


def compute_hetenyi_KLR(beta, EI, Lp):
    """Hetenyi formula for KLR (cross-coupling stiffness)."""
    bL = beta * Lp
    sh = np.sinh(bL)
    si = np.sin(bL)
    return -2 * beta**2 * EI * (sh**2 + si**2) / (sh**2 - si**2)


def main():
    prem_path = os.path.join(SCRIPT_DIR, 'data', 'PREM DATA.xlsx')

    # ==========================================================
    # STEP 1: Read existing PREM DATA and extract G/Gmax curves
    # ==========================================================
    df_prem = pd.read_excel(prem_path)
    orig_cols = list(df_prem.columns)  # preserve exact column names
    df = df_prem.copy()
    df.columns = df.columns.str.strip()

    print(f"Original PREM DATA: {len(df)} rows ({len(df)//44} scenarios)")
    print(f"Columns: {list(df.columns)}")

    # Extract G/Gmax curves by PI
    pi_values = []
    g_gmax_curves = []
    for sc_idx in range(19):
        sc = df.iloc[sc_idx*44:(sc_idx+1)*44]
        pi = sc['PI (%)'].iloc[0]
        g_gmax = sc['G/Gmax'].values
        pi_values.append(pi)
        g_gmax_curves.append(g_gmax)

    pi_arr = np.array(pi_values)
    curves_arr = np.array(g_gmax_curves)  # shape (19, 44)

    # Build G/Gmax interpolator (linear by PI, for each strain level)
    order = np.argsort(pi_arr)
    pi_sorted = pi_arr[order]
    curves_sorted = curves_arr[order]

    g_gmax_interps = []
    for s in range(44):
        f = interp1d(pi_sorted, curves_sorted[:, s], kind='linear',
                     bounds_error=False,
                     fill_value=(curves_sorted[0, s], curves_sorted[-1, s]))
        g_gmax_interps.append(f)

    def get_g_gmax_curve(pi_val):
        """Get 44-step G/Gmax curve for any PI in [0, 200]."""
        c = np.array([g_gmax_interps[s](float(pi_val)) for s in range(44)])
        c[0] = min(c[0], 1.0)  # ensure <= 1
        c = np.clip(c, 0.001, 1.0)
        # Enforce monotonically decreasing
        for j in range(1, 44):
            c[j] = min(c[j], c[j-1])
        return c

    # ==========================================================
    # STEP 2: Generate same 1000 scenario parameters (seed=42)
    # ==========================================================
    N = 1000
    new_pi = np.random.uniform(0, 200, N)
    new_gmax = np.random.uniform(15e6, 40e6, N)
    new_v = np.random.uniform(0.3, 0.4, N)
    new_tp = np.random.uniform(0.05, 0.07, N)
    new_lp = np.random.uniform(20, 40, N)

    print(f"\nGenerating {N} scenarios for PREM DATA...")

    # ==========================================================
    # STEP 3: Compute all 23 columns for each scenario
    # ==========================================================
    rows = []
    for i in range(N):
        pi = new_pi[i]
        gmax = new_gmax[i]
        v = new_v[i]
        tp = new_tp[i]
        lp = new_lp[i]

        # Pile properties
        I_p = np.pi/64 * (Dp**4 - (Dp - 2*tp)**4)
        EI = Ep * I_p

        # G/Gmax at 44 strain levels
        g_gmax = get_g_gmax_curve(pi)

        # Physics chain for each strain level
        Gdeg = g_gmax * gmax
        Es = 2 * Gdeg * (1 + v)

        # kh at each strain level
        kh = np.array([compute_kh(es_j, v, EI) for es_j in Es])

        # Beta from INITIAL kh (constant for scenario)
        kh_init = kh[0]
        beta = (kh_init * Dp / (4 * EI))**0.25
        betaLp = beta * lp

        # Compute KL, KR, KLR
        BETA_THRESHOLD = 2.5
        if betaLp < BETA_THRESHOLD:
            # Rigid body approximation
            KL = kh * Dp * lp
            KR = kh * Dp * lp**3 / 3
            KLR = -kh * Dp * lp**2 / 2
        else:
            # Hetenyi: compute Phi coefficients from initial state, then scale
            KL_init = compute_hetenyi_KL(beta, EI, lp)
            KR_init = compute_hetenyi_KR(beta, EI, lp)
            KLR_init = compute_hetenyi_KLR(beta, EI, lp)

            # Scale by kh ratio (Phi is constant within scenario)
            scale = kh / kh_init
            KL = KL_init * scale
            KR = KR_init * scale
            KLR = KLR_init * scale

        # Compute drops
        KL_drop = np.zeros(44)
        KR_drop = np.zeros(44)
        KLR_drop = np.zeros(44)
        KL_drop[0] = KL[0]
        KR_drop[0] = KR[0]
        KLR_drop[0] = KLR[0]
        for j in range(1, 44):
            KL_drop[j] = KL[j-1] - KL[j]
            KR_drop[j] = KR[j-1] - KR[j]
            KLR_drop[j] = KLR[j-1] - KLR[j]

        # Build 44 rows for this scenario
        for j in range(44):
            row = [
                pi,           # PI (%)
                g_gmax[j],    # G/Gmax
                Dp,           # Dp (m)
                lp,           # Lp (m)
                tp,           # Tp (m)
                I_p,          # I
                v,            # v
                gmax,         # Gmax
                Gdeg[j],      # Gdeg
                Ep,           # Ep
                beta,         # beta
                betaLp,       # betaLp
                kh[j],        # kh
                KL[j],        # KL
                KR[j],        # KR
                KLR[j],       # KLR
                KL_drop[j],   # KL drop
                KR_drop[j],   # KR drop
                KLR_drop[j],  # KLR drop
                np.nan,       # Unnamed: 19
                Es[j],        # Es
                EI,           # EI
                GAMMA_VALS[j] # gamma (%)
            ]
            rows.append(row)

        if (i+1) % 200 == 0:
            print(f"  Generated {i+1}/{N} scenarios...")

    new_df = pd.DataFrame(rows, columns=orig_cols)

    # ==========================================================
    # STEP 4: Validate
    # ==========================================================
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")

    # Check a few scenarios
    for sc_idx in [0, 100, 500, 999]:
        sc = new_df.iloc[sc_idx*44:(sc_idx+1)*44]
        sc.columns = [c.strip() for c in sc.columns]
        pi_v = sc['PI (%)'].iloc[0]
        gmax_v = sc['Gmax'].iloc[0]
        v_v = sc['v'].iloc[0]
        lp_v = sc['Lp (m)'].iloc[0]
        tp_v = sc['Tp (m)'].iloc[0]
        bl = sc[chr(946)+'Lp'].iloc[0] if chr(946)+'Lp' in sc.columns else sc.iloc[0, 11]
        kl_init = sc['KL'].iloc[0] if 'KL' in sc.columns else sc.iloc[0, 13]
        kl_last = sc['KL'].iloc[43] if 'KL' in sc.columns else sc.iloc[43, 13]
        gg_last = sc['G/Gmax'].iloc[43] if 'G/Gmax' in sc.columns else sc.iloc[43, 1]

        print(f"\n  Scenario {sc_idx}: PI={pi_v:.1f}, Gmax={gmax_v:.0f}, v={v_v:.2f}, Lp={lp_v:.1f}, Tp={tp_v:.3f}")
        print(f"    betaLp={bl:.4f}, regime={'rigid' if bl < 2.5 else 'Hetenyi'}")
        print(f"    KL_init={kl_init:.2e}, KL_last={kl_last:.2e}")
        print(f"    G/Gmax_last={gg_last:.4f}, KL_last/KL_init ratio={kl_last/kl_init:.4f}")

    # Verify drops consistency
    print("\n  Checking drop consistency...")
    all_ok = True
    for sc_idx in range(N):
        sc_vals = new_df.iloc[sc_idx*44:(sc_idx+1)*44]
        kl = sc_vals.iloc[:, 13].values  # KL column
        kl_drop = sc_vals.iloc[:, 16].values  # KL drop column

        # drop[0] should = KL[0]
        if abs(kl_drop[0] - kl[0]) > 1e-6:
            print(f"    FAIL: Sc {sc_idx} drop[0] != KL[0]")
            all_ok = False
            break

        # drop[i] should = KL[i-1] - KL[i]
        for j in range(1, 44):
            expected = kl[j-1] - kl[j]
            if abs(kl_drop[j] - expected) > 1e-6:
                print(f"    FAIL: Sc {sc_idx} drop[{j}] mismatch")
                all_ok = False
                break
        if not all_ok:
            break

    if all_ok:
        print("    All drops consistent!")

    # Compare with original PREM ranges
    df_clean = df.copy()
    print("\n  Column ranges (original vs generated):")
    for col in ['kh', 'KL', 'KR', 'KLR', 'Es', 'EI']:
        orig_min = df_clean[col].min()
        orig_max = df_clean[col].max()
        gen_min = new_df.iloc[:, list(df_clean.columns).index(col)].min()
        gen_max = new_df.iloc[:, list(df_clean.columns).index(col)].max()
        print(f"    {col:6s}: orig [{orig_min:.2e}, {orig_max:.2e}], gen [{gen_min:.2e}, {gen_max:.2e}]")

    # ==========================================================
    # STEP 5: Save
    # ==========================================================
    # Backup original
    backup_path = os.path.join(SCRIPT_DIR, 'data', 'PREM DATA_backup.xlsx')
    if not os.path.exists(backup_path):
        shutil.copy2(prem_path, backup_path)
        print(f"\nBackup saved: {backup_path}")
    else:
        print(f"\nBackup already exists: {backup_path}")

    # Combine and save
    combined = pd.concat([df_prem, new_df], ignore_index=True)
    combined.to_excel(prem_path, index=False)

    n_orig = len(df_prem)
    n_new = len(new_df)
    print(f"\nOriginal: {n_orig} rows ({n_orig // 44} scenarios)")
    print(f"Added:    {n_new} rows ({n_new // 44} scenarios)")
    print(f"Total:    {len(combined)} rows ({len(combined) // 44} scenarios)")
    print(f"\nSaved to: {prem_path}")


if __name__ == "__main__":
    main()
