import pandas as pd
import numpy as np

df = pd.read_excel('data/PREM DATA.xlsx')

# Check if kh/beta change within a scenario
print('kh unique values for first scenario (rows 0-43):')
print(df['kh'].iloc[:44].unique())

print('\nbeta unique values for first scenario (rows 0-43):')
print(df['β'].iloc[:44].unique())

# Verify KL = kh * Dp * Lp for various rows
print('\nKL vs kh*Dp*Lp:')
for i in [0, 6, 20, 40, 43, 44, 100]:
    r = df.iloc[i]
    calc = r['kh'] * r['Dp (m)'] * r['Lp (m)']
    kl = r['KL']
    print(f"  Row {i}: KL={kl:.6e}, kh*Dp*Lp={calc:.6e}, match={abs(kl-calc)<1}")

# So kh is constant per scenario?
# Check scenario 2 (rows 44-87)
print('\nkh unique values for second scenario (rows 44-87):')
print(df['kh'].iloc[44:88].unique())

# Check if kh changes per STRAIN STEP
# For scenario 1 rows 40-43 where G/Gmax is very different
print('\nDetailed check: kh, beta, KL for rows with degradation:')
for i in [0, 10, 20, 30, 40, 43]:
    r = df.iloc[i]
    print(f"  Row {i}: G/Gmax={r['G/Gmax']:.6f}, kh={r['kh']:.3e}, beta={r['β']:.6f}, KL={r['KL']:.3e}")

# If kh is constant, then KL should NOT equal kh*Dp*Lp for degraded rows
# Let's check what the actual formula relating kh to KL is
print('\nFor degraded rows - how does KL relate to kh?')
for i in [0, 10, 20, 30, 40, 43]:
    r = df.iloc[i]
    kl = r['KL']
    kh_dp_lp = r['kh'] * r['Dp (m)'] * r['Lp (m)']
    ratio = kl / kh_dp_lp
    ggmax = r['G/Gmax']
    print(f"  Row {i}: G/Gmax={ggmax:.6f}, KL/(kh*Dp*Lp)={ratio:.6f}")
