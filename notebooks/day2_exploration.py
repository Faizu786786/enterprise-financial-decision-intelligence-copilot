import pandas as pd

print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

df = pd.read_csv(
    "data/raw/PS_20174392719_1491204439457_log.csv"
)

print("\nDATASET LOADED SUCCESSFULLY\n")

# --------------------------------------------------
# BASIC KPIs
# --------------------------------------------------

print("=" * 60)
print("BASIC KPIs")
print("=" * 60)

print(f"Total Transactions: {len(df):,}")
print(f"Unique Origin Accounts: {df['nameOrig'].nunique():,}")
print(f"Unique Destination Accounts: {df['nameDest'].nunique():,}")

# --------------------------------------------------
# FRAUD ANALYSIS
# --------------------------------------------------

fraud_count = df["isFraud"].sum()
fraud_percentage = (fraud_count / len(df)) * 100

print("\n" + "=" * 60)
print("FRAUD ANALYSIS")
print("=" * 60)

print(f"Fraud Transactions: {fraud_count:,}")
print(f"Fraud Percentage: {fraud_percentage:.4f}%")

# --------------------------------------------------
# TRANSACTION TYPES
# --------------------------------------------------

print("\n" + "=" * 60)
print("TRANSACTION TYPE DISTRIBUTION")
print("=" * 60)

print(df["type"].value_counts())

# --------------------------------------------------
# AMOUNT STATISTICS
# --------------------------------------------------

print("\n" + "=" * 60)
print("AMOUNT STATISTICS")
print("=" * 60)

print(df["amount"].describe())

# --------------------------------------------------
# MEMORY USAGE
# --------------------------------------------------

memory_mb = df.memory_usage(deep=True).sum() / (1024**2)

print("\n" + "=" * 60)
print("MEMORY USAGE")
print("=" * 60)

print(f"{memory_mb:.2f} MB")