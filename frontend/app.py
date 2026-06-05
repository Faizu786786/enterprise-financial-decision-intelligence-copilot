import sys
import os

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
    use_container_width=True
)

# --------------------------------------------------
# FRAUD INTELLIGENCE
# --------------------------------------------------

st.divider()

st.subheader("Fraud Intelligence")

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