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

    return {
        "total_transactions": total_transactions,
        "fraud_transactions": fraud_transactions,
        "fraud_rate": fraud_rate,
        "unique_origin_accounts": unique_origin_accounts,
        "unique_destination_accounts": unique_destination_accounts,
    }