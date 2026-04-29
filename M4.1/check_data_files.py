import pandas as pd
import numpy as np

# ======= DATA TR - Copy.xlsx =======
print("=" * 60)
print("DATA TR - Copy.xlsx")
print("=" * 60)
df_tr = pd.read_excel('data/DATA TR - Copy.xlsx', header=None)
print(f"Shape: {df_tr.shape}")
print(f"\nFirst 15 rows, first 15 cols:")
print(df_tr.iloc[:15, :15].to_string())
print(f"\nLast 5 rows, first 15 cols:")
print(df_tr.iloc[-5:, :15].to_string())

# Check all sheets
xls = pd.ExcelFile('data/DATA TR - Copy.xlsx')
print(f"\nSheets: {xls.sheet_names}")

print("\n\n")
print("=" * 60)
print("PREM DATA.xlsx")
print("=" * 60)
df_prem = pd.read_excel('data/PREM DATA.xlsx')
print(f"Shape: {df_prem.shape}")
print(f"Columns: {list(df_prem.columns)}")
print(f"\nFirst 5 rows:")
print(df_prem.head().to_string())
print(f"\nLast 5 rows:")
print(df_prem.tail().to_string())

# Check dtypes
print(f"\nDtypes:\n{df_prem.dtypes}")
print(f"\nNull counts:\n{df_prem.isnull().sum()}")
