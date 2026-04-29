import pandas as pd

# Path to the Excel file
file_path = r"c:\Users\youss\Downloads\PFE\M4.1\REAL DATA.xlsx"

print("=" * 60)
print("EXCEL FILE ANALYSIS: REAL DATA.xlsx")
print("=" * 60)

# Load the Excel file
xlsx = pd.ExcelFile(file_path)

# 1. Get all sheet names
print(f"\n1. SHEET NAMES ({len(xlsx.sheet_names)} sheets):")
print("-" * 40)
for i, name in enumerate(xlsx.sheet_names, 1):
    print(f"   {i}. {name}")

# 2 & 3 & 4. Read each sheet and show details
print("\n" + "=" * 60)
print("2. SHEET DETAILS")
print("=" * 60)

for sheet_name in xlsx.sheet_names:
    df = pd.read_excel(xlsx, sheet_name=sheet_name)
    
    print(f"\n{'='*60}")
    print(f"SHEET: '{sheet_name}'")
    print(f"{'='*60}")
    
    # Column names
    print(f"\nColumns ({len(df.columns)}):")
    print("-" * 40)
    for col in df.columns:
        print(f"   - {col} (dtype: {df[col].dtype})")
    
    # Total rows
    print(f"\nTotal rows: {len(df)}")
    
    # Sample data (first 15 rows)
    print(f"\nSample data (first 15 rows):")
    print("-" * 40)
    print(df.head(15).to_string())
    
    # Basic statistics for numeric columns
    print(f"\nNumeric column statistics:")
    print("-" * 40)
    print(df.describe().to_string())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
