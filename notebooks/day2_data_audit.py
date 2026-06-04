import pandas as pd

print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

df = pd.read_csv(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

print("\nDATASET LOADED SUCCESSFULLY\n")

print("=" * 60)
print("SHAPE")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("COLUMNS")
print("=" * 60)
print(df.columns.tolist())

print("\n")

print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATE RECORDS")
print("=" * 60)
print(df.duplicated().sum())

print("\n")

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())