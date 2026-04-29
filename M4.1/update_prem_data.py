"""
Update PREM DATA.xlsx with 1000 new scenarios from REAL DATA.xlsx.
Computes full physics chain: G/Gmax, Gdeg, Es, kh, beta, betaLp, KL, KR, KLR, drops.

Uses:
- DATA TR G/Gmax degradation curves (interpolated by PI)
- RBF interpolation for kh_initial from existing 19 scenarios
- Physics: kh(step) = kh_initial * G/Gmax^(13/12), beta from kh_initial
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, Rbf
import os, shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
Ep = 210e9  # steel Young's modulus
Dp = 5.0    # pile diameter (constant)

# 44 strain levels (gamma %)
GAMMA_VALUES = [
    0.0001, 0.00019, 0.00028, 0.00037, 0.00046, 0.00055, 0.00064, 0.00073, 0.00082, 0.00091,
    0.001, 0.0019, 0.0028, 0.0037, 0.0046, 0.0055, 0.0064, 0.0073, 0.0082, 0.0091,
    0.01, 0.019, 0.028, 0.037, 0.046, 0.055, 0.064, 0.073, 0.082, 0.091,
    0.1, 0.19, 0.28, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82, 0.91,
    1.0, 1.9, 2.8, 3.7
]


def load_ggmax_table():
    """Load G/Gmax degradation curves from DATA TR - Copy.xlsx."""
    df_tr = pd.read_excel(os.path.join(SCRIPT_DIR, 'data', 'DATA TR - Copy.xlsx'),
                          sheet_name='OCR 0', header=None)
    
    # Row 12 = headers, rows 13-56 = data (44 strain levels)
    headers = df_tr.iloc[12].tolist()
    data = df_tr.iloc[13:57].values  # 44 rows
    
    # Column 3 = strain values
    # Columns 4-9: PI 0, 15, 30, 50, 100, 200
    # Column 10: blank
    # Columns 11-30: PI 5, 10, 20, 25, 35, 40, 45, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160, 170, 180, 190
    
    pi_cols = {}
    for c in range(4, 31):
        h = headers[c]
        if isinstance(h, str) and h.startswith('PI'):
            pi_val = int(h.replace('PI ', '').replace('PI', ''))
            pi_cols[pi_val] = data[:, c].astype(float)
    
    # Sort by PI
    sorted_pis = sorted(pi_cols.keys())
    pi_array = np.array(sorted_pis, dtype=float)
    ggmax_matrix = np.column_stack([pi_cols[p] for p in sorted_pis])  # shape (44, n_pis)
    
    return pi_array, ggmax_matrix


def build_ggmax_interpolator(pi_array, ggmax_matrix):
    """Build interpolator: given PI, return 44-step G/Gmax curve."""
    step_interps = []
    for s in range(44):
        f = interp1d(pi_array, ggmax_matrix[s, :], kind='linear',
                     bounds_error=False,
                     fill_value=(ggmax_matrix[s, 0], ggmax_matrix[s, -1]))
        step_interps.append(f)
    
    def get_ggmax_curve(pi_val):
        curve = np.array([float(step_interps[s](pi_val)) for s in range(44)])
        curve[0] = 1.0
        curve = np.clip(curve, 0.0, 1.0)
        # Enforce monotonically decreasing
        for j in range(1, 44):
            curve[j] = min(curve[j], curve[j - 1])
        return curve
    
    return get_ggmax_curve


def build_kh_initial_model(df_prem):
    """Build RBF model for kh_initial from existing 19 scenarios."""
    # Extract first row of each scenario (G/Gmax=1 → initial values)
    scenario_starts = []
    for i in range(0, len(df_prem), 44):
        row = df_prem.iloc[i]
        scenario_starts.append({
            'Es': row['Es'],
            'EI': row['EI'],
            'Lp': row['Lp (m)'],
            'kh': row['kh'],
            'beta': row['β'],
        })
    
    es_arr = np.array([s['Es'] for s in scenario_starts])
    ei_arr = np.array([s['EI'] for s in scenario_starts])
    lp_arr = np.array([s['Lp'] for s in scenario_starts])
    kh_arr = np.array([s['kh'] for s in scenario_starts])
    
    log_es = np.log(es_arr)
    log_ei = np.log(ei_arr)
    log_lp = np.log(lp_arr)
    log_kh = np.log(kh_arr)
    
    rbf = Rbf(log_es, log_ei, log_lp, log_kh, function='linear', smooth=0.01)
    
    # Verify
    preds = np.exp(rbf(log_es, log_ei, log_lp))
    errs = np.abs(preds / kh_arr - 1) * 100
    r2 = 1 - np.sum((np.log(preds) - log_kh)**2) / np.sum((log_kh - log_kh.mean())**2)
    print(f"kh_initial RBF model: R²={r2:.4f}, max_error={errs.max():.1f}%, mean_error={errs.mean():.1f}%")
    
    kh_min = kh_arr.min() * 0.3
    kh_max = kh_arr.max() * 3.0
    
    def predict_kh_initial(es_init, ei, lp):
        val = np.exp(float(rbf(np.log(es_init), np.log(ei), np.log(lp))))
        return np.clip(val, kh_min, kh_max)
    
    return predict_kh_initial


def main():
    # ============================================================
    # Load existing PREM DATA
    # ============================================================
    prem_path = os.path.join(SCRIPT_DIR, 'data', 'PREM DATA.xlsx')
    df_prem = pd.read_excel(prem_path)
    prem_cols = list(df_prem.columns)
    print(f"Existing PREM DATA: {len(df_prem)} rows ({len(df_prem)//44} scenarios)")
    print(f"Columns: {prem_cols}")
    
    # ============================================================
    # Load REAL DATA (new scenarios start at row 836)
    # ============================================================
    real_path = os.path.join(SCRIPT_DIR, 'REAL DATA.xlsx')
    df_real = pd.read_excel(real_path)
    df_real.columns = df_real.columns.str.strip()
    
    n_orig_rows = 836  # 19 original scenarios × 44 steps
    df_new = df_real.iloc[n_orig_rows:].reset_index(drop=True)
    n_new_scenarios = len(df_new) // 44
    print(f"New scenarios to process: {n_new_scenarios} ({len(df_new)} rows)")
    
    # ============================================================
    # Build G/Gmax interpolator from DATA TR
    # ============================================================
    pi_array, ggmax_matrix = load_ggmax_table()
    get_ggmax_curve = build_ggmax_interpolator(pi_array, ggmax_matrix)
    print(f"G/Gmax table loaded: {len(pi_array)} PI values, 44 strain levels")
    
    # ============================================================
    # Build kh_initial RBF model from existing PREM DATA
    # ============================================================
    predict_kh_initial = build_kh_initial_model(df_prem)
    
    # ============================================================
    # Generate PREM DATA rows for 1000 new scenarios
    # ============================================================
    new_rows = []
    
    for sc in range(n_new_scenarios):
        idx0 = sc * 44
        grp = df_new.iloc[idx0:idx0+44]
        
        # Scenario parameters (constant within scenario)
        pi = grp['PI'].iloc[0]
        gmax = grp['Gmax'].iloc[0]
        v = grp['v'].iloc[0]
        dp = grp['Dp'].iloc[0]
        tp = grp['Tp'].iloc[0]
        lp = grp['Lp'].iloc[0]
        ip = grp['Ip'].iloc[0]
        
        # Derived constants
        ei = Ep * ip
        es_initial = 2 * gmax * (1 + v)
        
        # Reconstruct absolute KL, KR, KLR from drops in REAL DATA
        kl_drops = grp['KL'].values
        kr_drops = grp['KR'].values
        klr_drops = grp['KLR'].values
        
        kl_abs = np.zeros(44)
        kr_abs = np.zeros(44)
        klr_abs = np.zeros(44)
        
        kl_abs[0] = kl_drops[0]   # First value is absolute
        kr_abs[0] = kr_drops[0]
        klr_abs[0] = klr_drops[0]
        
        for j in range(1, 44):
            kl_abs[j] = kl_abs[j-1] - kl_drops[j]
            kr_abs[j] = kr_abs[j-1] - kr_drops[j]
            klr_abs[j] = klr_abs[j-1] - klr_drops[j]
        
        # G/Gmax from DATA TR interpolation
        ggmax_curve = get_ggmax_curve(pi)
        
        # kh_initial from RBF model
        kh_init = predict_kh_initial(es_initial, ei, lp)
        
        # beta from kh_initial (constant per scenario)
        beta_val = (kh_init / (4 * ei)) ** 0.25
        beta_lp = beta_val * lp
        
        # Per-step physics
        for j in range(44):
            gg = ggmax_curve[j]
            gdeg = gg * gmax
            es = 2 * gdeg * (1 + v)
            kh = kh_init * (gg ** (13.0/12.0))
            
            # Drops: step 0 = absolute value, step n>0 = previous - current
            if j == 0:
                kl_drop = kl_abs[0]
                kr_drop = kr_abs[0]
                klr_drop = klr_abs[0]
            else:
                kl_drop = kl_abs[j-1] - kl_abs[j]
                kr_drop = kr_abs[j-1] - kr_abs[j]
                klr_drop = klr_abs[j-1] - klr_abs[j]
            
            # Gamma (strain %) — fill for all scenarios
            gamma = GAMMA_VALUES[j]
            
            row = {
                'PI (%)': pi,
                'G/Gmax': gg,
                'Dp (m)': dp,
                'Lp (m)': lp,
                'Tp (m)': tp,
                'I': ip,
                'v': v,
                'Gmax': gmax,
                'Gdeg': gdeg,
                'Ep': Ep,
                'β': beta_val,
                'βLp': beta_lp,
                'kh': kh,
                'KL': kl_abs[j],
                'KR ': kr_abs[j],  # Note: trailing space to match original column name
                'KLR': klr_abs[j],
                'KL drop': kl_drop,
                'KR drop': kr_drop,
                'KLR drop': klr_drop,
                'Unnamed: 19': np.nan,
                'Es': es,
                'EI': ei,
                'ϒ (%)': gamma,
            }
            new_rows.append(row)
        
        if (sc + 1) % 200 == 0:
            print(f"  Processed {sc+1}/{n_new_scenarios} scenarios...")
    
    print(f"Generated {len(new_rows)} new rows")
    
    # ============================================================
    # Validate sample scenarios
    # ============================================================
    print(f"\n{'='*50}")
    print("VALIDATION")
    print(f"{'='*50}")
    
    new_df = pd.DataFrame(new_rows)
    
    # Check first 3 new scenarios
    for sc in range(min(3, n_new_scenarios)):
        idx0 = sc * 44
        grp = new_df.iloc[idx0:idx0+44]
        pi = grp['PI (%)'].iloc[0]
        lp = grp['Lp (m)'].iloc[0]
        tp = grp['Tp (m)'].iloc[0]
        gmax = grp['Gmax'].iloc[0]
        kl0 = grp['KL'].iloc[0]
        kl_last = grp['KL'].iloc[43]
        gg_last = grp['G/Gmax'].iloc[43]
        beta = grp['β'].iloc[0]
        blp = grp['βLp'].iloc[0]
        kh0 = grp['kh'].iloc[0]
        
        print(f"  Scenario {sc+1}: PI={pi:.1f}, Lp={lp:.1f}, Tp={tp:.4f}, Gmax={gmax:.0f}")
        print(f"    KL: {kl0:.3e} → {kl_last:.3e} ({kl_last/kl0*100:.1f}% remaining)")
        print(f"    G/Gmax final: {gg_last:.6f}, β={beta:.6f}, βLp={blp:.3f}")
        print(f"    kh_initial={kh0:.3e}")
    
    # Compare with original PREM DATA format
    print(f"\nColumn comparison:")
    print(f"  Original: {prem_cols}")
    print(f"  New:      {list(new_df.columns)}")
    
    # ============================================================
    # Backup and save
    # ============================================================
    backup_path = prem_path.replace('.xlsx', '_backup.xlsx')
    if not os.path.exists(backup_path):
        shutil.copy2(prem_path, backup_path)
        print(f"\nBackup saved: {backup_path}")
    
    combined = pd.concat([df_prem, new_df], ignore_index=True)
    combined.to_excel(prem_path, index=False)
    
    print(f"\nOriginal: {len(df_prem)} rows ({len(df_prem)//44} scenarios)")
    print(f"Added:    {len(new_df)} rows ({len(new_df)//44} scenarios)")
    print(f"Total:    {len(combined)} rows ({len(combined)//44} scenarios)")
    print(f"Saved to: {prem_path}")


if __name__ == "__main__":
    main()
