"""
Update DATA TR - Copy.xlsx with complete G/Gmax curves for all PI values (0-200, step 5).
Fills in 15 missing PI columns by interpolating from existing 26 columns.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import os, shutil
from openpyxl import load_workbook

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    tr_path = os.path.join(SCRIPT_DIR, 'data', 'DATA TR - Copy.xlsx')
    
    # Backup
    backup_path = tr_path.replace('.xlsx', '_backup.xlsx')
    if not os.path.exists(backup_path):
        shutil.copy2(tr_path, backup_path)
        print(f"Backup saved: {backup_path}")
    
    # Read OCR 0 sheet
    df = pd.read_excel(tr_path, sheet_name='OCR 0', header=None)
    print(f"Original shape: {df.shape}")
    
    # Extract existing G/Gmax data
    headers = df.iloc[12].tolist()
    data = df.iloc[13:57].values  # 44 strain rows
    gamma_values = data[:, 3].astype(float)  # Column 3 = strain values
    
    # Parse existing PI columns
    existing = {}
    for c in range(4, df.shape[1]):
        h = headers[c]
        if isinstance(h, str) and h.startswith('PI'):
            pi_val = int(h.replace('PI ', '').replace('PI', ''))
            existing[pi_val] = data[:, c].astype(float)
    
    print(f"Existing PI values ({len(existing)}): {sorted(existing.keys())}")
    
    # Target: every 5 from 0 to 200
    all_pis = list(range(0, 201, 5))  # 0, 5, 10, ..., 200 = 41 values
    missing = [p for p in all_pis if p not in existing]
    print(f"Missing PI values ({len(missing)}): {missing}")
    
    if not missing:
        print("No missing PI values — table is already complete!")
        return
    
    # Build interpolator from existing data
    sorted_pis = sorted(existing.keys())
    pi_arr = np.array(sorted_pis, dtype=float)
    
    # For each strain step, interpolate across PI
    complete = dict(existing)  # copy existing
    for pi_target in missing:
        values = np.zeros(44)
        for s in range(44):
            existing_vals = np.array([existing[p][s] for p in sorted_pis])
            f = interp1d(pi_arr, existing_vals, kind='linear',
                        bounds_error=False,
                        fill_value=(existing_vals[0], existing_vals[-1]))
            values[s] = float(f(pi_target))
        # Enforce constraints
        values[0] = 1.0
        values = np.clip(values, 0.0, 1.0)
        for j in range(1, 44):
            values[j] = min(values[j], values[j-1])
        complete[pi_target] = values
    
    # Build new dataframe with all PI values in order
    # Structure: columns 0-2 empty, col 3 = gamma, cols 4+ = PI values in order
    n_pis = len(all_pis)
    n_cols = 4 + n_pis  # 3 empty + gamma + PI columns
    n_rows = 13 + 44  # 13 header/blank rows + 44 data rows = 57
    
    new_data = np.full((n_rows, n_cols), np.nan)
    
    # Row 12 = headers (will be set as strings separately)
    # Rows 13-56 = data
    for j, pi_val in enumerate(all_pis):
        col = 4 + j
        for s in range(44):
            new_data[13 + s, col] = complete[pi_val][s]
    
    # Column 3 = gamma values
    for s in range(44):
        new_data[13 + s, 3] = gamma_values[s]
    
    new_df = pd.DataFrame(new_data)
    
    # Set header row
    new_df.iloc[12, 3] = 'ϒ (%)'
    for j, pi_val in enumerate(all_pis):
        new_df.iloc[12, 4 + j] = f'PI {pi_val}'
    
    # Save to OCR 0 sheet (preserve Sheet1)
    # Write using openpyxl to preserve Sheet1
    with pd.ExcelWriter(tr_path, engine='openpyxl', mode='w') as writer:
        new_df.to_excel(writer, sheet_name='OCR 0', index=False, header=False)
        # Recreate Sheet1 with headers
        sheet1_cols = ['Ip', 'Dp (m)', 'Tp (m)', 'Lp (m)', 'v', 'G/Gmax', 'Gdeg', 'kh', 'KL', 'KR', 'KRL']
        pd.DataFrame(columns=sheet1_cols).to_excel(writer, sheet_name='Sheet1', index=False)
    
    print(f"\nUpdated DATA TR:")
    print(f"  Shape: {new_df.shape}")
    print(f"  PI columns: {n_pis} (every 5 from 0 to 200)")
    print(f"  Added {len(missing)} new PI columns: {missing}")
    
    # Verify
    df_check = pd.read_excel(tr_path, sheet_name='OCR 0', header=None)
    print(f"  Verified shape: {df_check.shape}")
    print(f"  Row 12 (first 10): {df_check.iloc[12, :10].tolist()}")
    print(f"  Row 12 (last 5): {df_check.iloc[12, -5:].tolist()}")
    print(f"  First data row (4 entries): {df_check.iloc[13, 4:8].tolist()}")
    print(f"  Last data row (last 4): {df_check.iloc[56, -4:].tolist()}")
    print(f"\nSaved to: {tr_path}")


if __name__ == "__main__":
    main()
