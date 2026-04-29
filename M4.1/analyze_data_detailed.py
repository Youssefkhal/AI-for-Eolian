import pandas as pd
import numpy as np

# ===== DATA TR - Copy.xlsx =====
print("=" * 60)
print("DATA TR - Copy.xlsx - Full analysis")
print("=" * 60)

# Read OCR 0 sheet
df_tr = pd.read_excel('data/DATA TR - Copy.xlsx', sheet_name='OCR 0', header=None)
print(f"Shape: {df_tr.shape}")
print(f"\nRow 12 (headers):")
print(df_tr.iloc[12].tolist())
print(f"\nAll columns 15-30 of row 12:")
for c in range(15, 31):
    print(f"  Col {c}: {df_tr.iloc[12, c]}")

print(f"\nFirst data row (row 13), all 31 cols:")
for c in range(31):
    print(f"  Col {c}: {df_tr.iloc[13, c]}")

# Read Sheet1
try:
    df_tr2 = pd.read_excel('data/DATA TR - Copy.xlsx', sheet_name='Sheet1', header=None)
    print(f"\n\nSheet1 shape: {df_tr2.shape}")
    print(f"Sheet1 first 5 rows:")
    print(df_tr2.head())
    print(f"\nSheet1 non-null counts per column:")
    print(df_tr2.count())
except Exception as e:
    print(f"\nSheet1 error: {e}")

# ===== PREM DATA - Drop column analysis =====
print("\n" + "=" * 60)
print("PREM DATA.xlsx - Drop column analysis")
print("=" * 60)

df = pd.read_excel('data/PREM DATA.xlsx')

# Look at first scenario (rows 0-43)
print("\nFirst scenario (rows 0-10), key columns:")
cols = ['PI (%)', 'G/Gmax', 'KL', 'KR ', 'KLR', 'KL drop', 'KR drop', 'KLR drop', 'ϒ (%)']
print(df[cols].iloc[:10].to_string())

print("\nFirst scenario (rows 39-43):")
print(df[cols].iloc[39:44].to_string())

# Second scenario (rows 44-87)
print("\nSecond scenario (rows 44-55):")
print(df[cols].iloc[44:55].to_string())

# Verify drop = difference from previous row
print("\n\nVerifying KL drop formula:")
for i in range(1, 10):
    kl_curr = df['KL'].iloc[i]
    kl_prev = df['KL'].iloc[i-1]
    kl_drop = df['KL drop'].iloc[i]
    diff = kl_prev - kl_curr
    print(f"  Row {i}: KL={kl_curr:.3e}, KL_prev={kl_prev:.3e}, diff={diff:.3e}, KL_drop={kl_drop:.3e}, match={abs(diff-kl_drop)<1}")

# Check if drop is cumulative from initial
print("\nVerifying if KL drop = KL_initial - KL_current:")
kl_init = df['KL'].iloc[0]
for i in [0, 1, 20, 30, 40, 43]:
    kl_curr = df['KL'].iloc[i]
    kl_drop = df['KL drop'].iloc[i]
    cum_diff = kl_init - kl_curr
    print(f"  Row {i}: KL={kl_curr:.3e}, KL_init-KL={cum_diff:.3e}, KL_drop={kl_drop:.3e}")

# Check row 0 drop
print(f"\nRow 0: KL={df['KL'].iloc[0]:.3e}, KL_drop={df['KL drop'].iloc[0]:.3e}")
print(f"Equal? {abs(df['KL'].iloc[0] - df['KL drop'].iloc[0]) < 1}")

# Check gamma values
print(f"\nϒ (%) non-null rows: {df['ϒ (%)'].notna().sum()}")
gamma_vals = df['ϒ (%)'].dropna().values
print(f"Gamma values (first 10): {gamma_vals[:10]}")
print(f"Gamma values (last 10): {gamma_vals[-10:]}")

# What are the gamma values - are they the strain levels?
print(f"\nUnique gamma count: {len(np.unique(gamma_vals))}")
print(f"All unique gamma: {np.unique(gamma_vals)}")

# Check kh formula verification
print("\n" + "=" * 60)
print("Verifying kh formula")
print("=" * 60)
for idx in [0, 44, 100, 400, 800]:
    row = df.iloc[idx]
    Es = row['Es']
    EI = row['EI']
    Dp = row['Dp (m)']
    v = row['v']
    kh_actual = row['kh']
    
    vesic = 0.65 * Es / (1 - v**2) * (Es * Dp**4 / EI) ** (1/12)
    ratio = kh_actual / vesic
    
    # Try alternative: kh = Es * Dp / (EI) * something?
    simple = Es * Dp
    
    print(f"Row {idx}: Es={Es:.3e}, EI={EI:.3e}, kh={kh_actual:.3e}, Vesic={vesic:.3e}, ratio={ratio:.6f}")
