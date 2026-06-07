import sys
import os
import joblib

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(project_root)

import streamlit as st
import pandas as pd
import plotly.express as px

from backend.data_loader import load_data
from backend.kpi_calculator import calculate_basic_kpis

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Decision Intelligence Copilot",
    layout="wide"
)

st.title("🏦 Financial Decision Intelligence Copilot")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_data()

kpis = calculate_basic_kpis(df)

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Transactions",
    f"{kpis['total_transactions']:,}"
)

col2.metric(
    "Fraud Transactions",
    f"{kpis['fraud_transactions']:,}"
)

col3.metric(
    "Fraud Rate (%)",
    f"{kpis['fraud_rate']}"
)

col4.metric(
    "Origin Accounts",
    f"{kpis['unique_origin_accounts']:,}"
)

st.divider()

# --------------------------------------------------
# TRANSACTION DISTRIBUTION
# --------------------------------------------------

st.subheader("Transaction Type Distribution")

transaction_df = pd.DataFrame(
    list(
        kpis["transaction_distribution"].items()
    ),
    columns=["Transaction Type", "Count"]
)

fig = px.bar(
    transaction_df,
    x="Transaction Type",
    y="Count",
    title="Transaction Distribution"
)

st.plotly_chart(
    fig,
    width="stretch"
)

# --------------------------------------------------
# AMOUNT STATISTICS
# --------------------------------------------------

st.subheader("Amount Statistics")

amount_stats = kpis["amount_statistics"]

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Average Amount",
    f"{amount_stats['mean_amount']:,.2f}"
)

col2.metric(
    "Median Amount",
    f"{amount_stats['median_amount']:,.2f}"
)

col3.metric(
    "Maximum Amount",
    f"{amount_stats['max_amount']:,.2f}"
)

col4.metric(
    "Minimum Amount",
    f"{amount_stats['min_amount']:,.2f}"
)

st.divider()
# --------------------------------------------------
# FRAUD INTELLIGENCE
# --------------------------------------------------

st.divider()

st.subheader("Fraud Intelligence")
# Fraud vs Non-Fraud Pie Chart

fraud_count = kpis["fraud_transactions"]

non_fraud_count = (
    kpis["total_transactions"]
    - fraud_count
)

pie_df = pd.DataFrame(
    {
        "Category": [
            "Fraud",
            "Non-Fraud"
        ],
        "Count": [
            fraud_count,
            non_fraud_count
        ]
    }
)

pie_fig = px.pie(
    pie_df,
    names="Category",
    values="Count",
    title="Fraud vs Non-Fraud Transactions"
)

st.plotly_chart(
    pie_fig,
    width="stretch"
)

st.divider()

fraud_df = pd.DataFrame(
    list(kpis["fraud_by_type"].items()),
    columns=["Transaction Type", "Fraud Count"]
)

fraud_fig = px.bar(
    fraud_df,
    x="Transaction Type",
    y="Fraud Count",
    title="Fraud Transactions by Transaction Type"
)

st.plotly_chart(
    fraud_fig,
    width="stretch"
)

# --------------------------------------------------
# FRAUD PREDICTION
# --------------------------------------------------



st.divider()

st.header("AI Fraud Prediction")

model = joblib.load(
    "models/fraud_model.pkl"
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=50000.0
)

oldbalanceOrg = st.number_input(
    "Old Origin Balance",
    min_value=0.0,
    value=100000.0
)

newbalanceOrig = st.number_input(
    "New Origin Balance",
    min_value=0.0,
    value=50000.0
)

oldbalanceDest = st.number_input(
    "Old Destination Balance",
    min_value=0.0,
    value=0.0
)

newbalanceDest = st.number_input(
    "New Destination Balance",
    min_value=0.0,
    value=50000.0
)

if st.button("Predict Fraud Risk"):

    input_data = pd.DataFrame(
        [[
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]],
        columns=[
            "amount",
            "oldbalanceOrg",
            "newbalanceOrig",
            "oldbalanceDest",
            "newbalanceDest"
        ]
    )

    prediction = model.predict(
        input_data
    )

    probability = model.predict_proba(
        input_data
    )

    fraud_probability = (
        probability[0][1] * 100
    )

    st.metric(
        "Fraud Probability (%)",
        f"{fraud_probability:.2f}%"
    )

    if fraud_probability >= 80:
        st.error(
            "🚨 HIGH RISK TRANSACTION"
        )

    elif fraud_probability >= 40:
        st.warning(
            "⚠️ MEDIUM RISK TRANSACTION"
        )

    else:
        st.success(
            "✅ LOW RISK TRANSACTION"
        )