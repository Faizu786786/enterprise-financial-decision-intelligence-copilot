import json
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
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    return joblib.load(
        "models/fraud_model.pkl"
    )

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Decision Intelligence Copilot",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F8FAFC;
}

/* Hide Streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* KPI Cards */
.kpi-box{
    background: white;
    padding: 20px;
    border-radius: 18px;
    border-left: 6px solid #2563EB;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}

.kpi-title{
    color:#64748B;
    font-size:15px;
}

.kpi-value{
    color:#0F172A;
    font-size:32px;
    font-weight:700;
}

.amount-box{
    background:white;
    padding:18px;
    border-radius:18px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    text-align:center;
}

/* Download Buttons */
.stDownloadButton button{
    border-radius:10px;
    font-weight:bold;
}

/* Prediction Button */
.stButton button{
    background:#2563EB;
    color:white;
    border-radius:12px;
    font-weight:bold;
    width:100%;
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    font-size:16px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        background: linear-gradient(
            135deg,
            #1e3a8a,
            #0f172a
        );
        padding:40px;
        border-radius:20px;
        text-align:center;
        margin-bottom:20px;
    ">

    <h1 style="color:white;">
    🏦 Enterprise Financial Decision Intelligence Copilot
    </h1>

    <p style="
        color:#cbd5e1;
        font-size:18px;
    ">
    AI-Powered Fraud Detection • Financial Analytics • Business Intelligence
    </p>

    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

with st.sidebar:

    st.header("System Status")
    
    st.caption(
    "Financial Intelligence Platform"
)

    st.success(
        "✅ Fraud Model Loaded"
    )

    st.info(
        "📊 KPI Data Ready"
    )

    st.info(
        "🤖 AI Detection Active"
    )
    
    st.divider()

    st.header("Reports")

    st.divider()

    st.caption(
        "Enterprise Financial Intelligence Platform"
    )
tab1, tab2 = st.tabs(
    [
        "📊 Analytics Dashboard",
        "🤖 Fraud Prediction"
    ]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open(
    "data/processed/kpis.json",
    "r"
) as file:

    kpis = json.load(file)

with tab1:

# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-title'>Total Transactions</div>
            <div class='kpi-value'>{kpis['total_transactions']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-title'>Fraud Transactions</div>
            <div class='kpi-value'>{kpis['fraud_transactions']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-title'>Fraud Rate (%)</div>
            <div class='kpi-value'>{kpis['fraud_rate']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class='kpi-box'>
            <div class='kpi-title'>Origin Accounts</div>
            <div class='kpi-value'>{kpis['unique_origin_accounts']:,}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    report_df = pd.DataFrame(
        {
            "Metric": [
                "Total Transactions",
                "Fraud Transactions",
                "Fraud Rate (%)",
                "Unique Origin Accounts",
                "Unique Destination Accounts"
            ],
            "Value": [
                kpis["total_transactions"],
                kpis["fraud_transactions"],
                kpis["fraud_rate"],
                kpis["unique_origin_accounts"],
                kpis["unique_destination_accounts"]
            ]
        }
    )

    csv_report = report_df.to_csv(
        index=False
    )
    
    

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

    fig.update_layout(
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white"
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

    cards = [
        ("Average Amount", amount_stats["mean_amount"]),
        ("Median Amount", amount_stats["median_amount"]),
        ("Maximum Amount", amount_stats["max_amount"]),
        ("Minimum Amount", amount_stats["min_amount"])
    ]

    for col, (title, value) in zip(
        [col1, col2, col3, col4],
        cards
    ):
        with col:
            st.markdown(
                f"""
                <div class='amount-box'>
                    <div class='kpi-title'>{title}</div>
                    <div class='kpi-value'>
                        ₹ {value:,.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
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

    pie_fig.update_layout(
        title_x=0.5
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

    fraud_fig.update_layout(
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        fraud_fig,
        width="stretch"
    )

    fraud_report_df = pd.DataFrame(
        list(
            kpis["fraud_by_type"].items()
        ),
        columns=[
            "Transaction Type",
            "Fraud Count"
        ]
    )

    fraud_csv = fraud_report_df.to_csv(
        index=False
    )
    
    st.download_button(
        label="📥 Download Fraud Report",
        data=fraud_csv,
        file_name="fraud_report.csv",
        mime="text/csv"
    )

    # --------------------------------------------------
    # MODEL EXPLAINABILITY
    # --------------------------------------------------

    st.divider()

    st.header(
    "🔍 Fraud Detection Feature Importance"
)

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "oldbalanceOrg",
                "amount",
                "newbalanceDest",
                "oldbalanceDest",
                "newbalanceOrig"
            ],
            "Importance": [
                0.463251,
                0.218124,
                0.196146,
                0.065274,
                0.057205
            ]
        }
    )

    importance_fig = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importance in Fraud Detection"
    )

    importance_fig.update_layout(
        title_x=0.5,
        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    st.plotly_chart(
        importance_fig,
        width="stretch"
    )

with tab2:

    st.divider()

    st.markdown(
        """
        ## 🤖 AI Fraud Risk Assessment

        Enter transaction details and let the AI model
        estimate the fraud probability.
        """
    )

    model = load_model()

    with st.form("fraud_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=50000.0
            )

            newbalanceOrig = st.number_input(
                "New Origin Balance",
                min_value=0.0,
                value=50000.0
            )

            newbalanceDest = st.number_input(
                "New Destination Balance",
                min_value=0.0,
                value=50000.0
            )

        with col2:

            oldbalanceOrg = st.number_input(
                "Old Origin Balance",
                min_value=0.0,
                value=100000.0
            )

            oldbalanceDest = st.number_input(
                "Old Destination Balance",
                min_value=0.0,
                value=0.0
            )

        submitted = st.form_submit_button(
            "🚀 Predict Fraud Risk"
        )

    if submitted:

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

        probability = model.predict_proba(
            input_data
        )

        fraud_probability = (
            probability[0][1] * 100
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=fraud_probability,
                title={
                    "text": "Fraud Risk Score"
                },
                gauge={
                    "axis": {
                        "range": [0, 100]
                    },
                    "steps": [
                        {
                            "range": [0, 40],
                            "color": "green"
                        },
                        {
                            "range": [40, 80],
                            "color": "orange"
                        },
                        {
                            "range": [80, 100],
                            "color": "red"
                        }
                    ]
                }
            )
        )

        gauge.update_layout(
            height=350
        )

        st.plotly_chart(
            gauge,
            width="stretch",
            key="fraud_risk_gauge"
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