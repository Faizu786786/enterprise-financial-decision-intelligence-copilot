def calculate_basic_kpis(df):

    total_transactions = len(df)

    fraud_transactions = df["isFraud"].sum()

    fraud_rate = (
        fraud_transactions / total_transactions
    ) * 100

    unique_origin_accounts = (
        df["nameOrig"].nunique()
    )

    unique_destination_accounts = (
        df["nameDest"].nunique()
    )

    transaction_distribution = (
        df["type"]
        .value_counts()
        .to_dict()
    )

    amount_stats = {
    "mean_amount": float(round(df["amount"].mean(), 2)),
    "median_amount": float(round(df["amount"].median(), 2)),
    "max_amount": float(round(df["amount"].max(), 2)),
    "min_amount": float(round(df["amount"].min(), 2)),
}
    fraud_by_type = (
    df[df["isFraud"] == 1]
    ["type"]
    .value_counts()
    .to_dict()
)

    return {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "fraud_rate": round(fraud_rate, 4),
        "unique_origin_accounts": unique_origin_accounts,
        "unique_destination_accounts": unique_destination_accounts,
        "transaction_distribution": transaction_distribution,
        "amount_statistics": amount_stats,
        "fraud_by_type": fraud_by_type,
    }