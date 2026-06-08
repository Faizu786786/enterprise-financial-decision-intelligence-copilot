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
:root{
  --bg0:#070A12;
  --bg1:#0B1222;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.09);
  --stroke: rgba(255,255,255,0.14);
  --stroke2: rgba(255,255,255,0.22);
  --text:#E5E7EB;
  --muted:#9CA3AF;
  --brand1:#22C55E;
  --brand2:#2563EB;
  --brand3:#7C3AED;
}

/* Main background */
.stApp {
  background: radial-gradient(1200px circle at 10% 10%, rgba(124,58,237,0.22), transparent 55%),
              radial-gradient(900px circle at 90% 20%, rgba(37,99,235,0.24), transparent 50%),
              radial-gradient(700px circle at 20% 90%, rgba(34,197,94,0.16), transparent 45%),
              linear-gradient(135deg, var(--bg0), var(--bg1));
}

/* Hide Streamlit menu */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Make controls feel modern */
.stTextInput>div>div,
.stNumberInput>div>div,
.stMultiselect>div>div,
.stSelectbox>div>div,
.stForm>div,
.stSidebar {
  background: transparent;
}

/* KPI Cards */
.kpi-box{
    background: var(--card);
    backdrop-filter: blur(14px);
    border: 1px solid var(--stroke);
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    transition: transform 0.18s ease, border-color 0.18s ease;
}

.kpi-box:hover{
    transform: translateY(-5px);
    border-color: var(--stroke2);
}

.kpi-title{
    color: rgba(229,231,235,0.75);
    font-size:15px;
    letter-spacing: 0.2px;
}

.kpi-value{
    color: #FFFFFF;
    font-size:32px;
    font-weight:800;
}

.amount-box{
    background: var(--card2);
    padding:18px;
    border-radius:18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.22);
    text-align:center;
    border: 1px solid rgba(255,255,255,0.12);
}

/* Buttons */
.stDownloadButton button{
    border-radius:12px;
    font-weight:800;
    border: 1px solid rgba(255,255,255,0.18);
    background: rgba(37,99,235,0.15);
    color: white;
}

.stButton button{
    background: linear-gradient(90deg, rgba(37,99,235,1), rgba(124,58,237,1));
    color:white;
    border-radius:14px;
    font-weight:900;
    width:100%;
    border: 1px solid rgba(255,255,255,0.16);
}

/* Tabs */
.stTabs [data-baseweb="tab"]{
    font-size:16px;
    font-weight:750;
    color: rgba(229,231,235,0.85);
    border-radius: 12px;
}

.stTabs [aria-selected="true"]{
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
}

/* Chart container tweak */
[data-testid="stPlotlyChart"]{
  background: rgba(255,255,255,0.02);
  border-radius: 18px;
  padding: 10px;
  border: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
background:linear-gradient(
135deg,
#1e3a8a,
#2563eb,
#0f172a
);
padding:45px;
border-radius:25px;
text-align:center;
box-shadow:0px 8px 30px rgba(0,0,0,0.15);
">

<h1 style="
color:white;
font-size:42px;
">
🏦 Enterprise Financial Intelligence Command Center
</h1>

<p style="
color:#dbeafe;
font-size:18px;
">
AI-Powered Fraud Detection • Explainable AI • Financial Analytics
</p>

</div>
""",
unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "📊 6.3M+ Transactions Analysed"
    )

with col2:
    st.info(
        "🤖 ML Fraud Detection Active"
    )

with col3:
    st.info(
        "⚡ Real-Time Risk Assessment"
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

    st.markdown(
        """
        <div style="
          background:rgba(255,255,255,0.06);
          border:1px solid rgba(255,255,255,0.14);
          border-radius:14px;
          padding:10px 14px;
          font-weight:900;
          letter-spacing:0.3px;
          color:#E5E7EB;
          margin-bottom:8px;
        ">
          📊 Transaction Type Distribution
        </div>
        """,
        unsafe_allow_html=True
    )

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
        template="plotly_dark",
        title_x=0.5,
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=20,r=20,t=60,b=10),
        hovermode="x unified",
    )

    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
        marker=dict(line=dict(width=0))
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    # --------------------------------------------------
    # AMOUNT STATISTICS
    # --------------------------------------------------
    
    st.markdown(
        """
        <div style="
          background:rgba(255,255,255,0.06);
          border:1px solid rgba(255,255,255,0.14);
          border-radius:14px;
          padding:10px 14px;
          font-weight:900;
          letter-spacing:0.3px;
          color:#E5E7EB;
          margin-bottom:8px;
        ">
          💠 Amount Statistics
        </div>
        """,
        unsafe_allow_html=True
    )

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

    st.markdown(
        """
        <div style="
          background:rgba(255,255,255,0.06);
          border:1px solid rgba(255,255,255,0.14);
          border-radius:14px;
          padding:10px 14px;
          font-weight:900;
          letter-spacing:0.3px;
          color:#E5E7EB;
          margin-bottom:8px;
        ">
          🛡️ Fraud Intelligence
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Interactive controls (precomputed KPI data) ---
    all_types = sorted(list(kpis["transaction_distribution"].keys()))
    selected_types = st.multiselect(
        "Filter by transaction type",
        options=all_types,
        default=all_types,
        help="Updates the fraud intelligence charts using the precomputed KPI breakdown."
    )

    # --- Aggregate fraud counts from precomputed breakdown ---
    fraud_by_type = kpis["fraud_by_type"]

    filtered_fraud_total = int(sum(
        fraud_by_type.get(t, 0) for t in selected_types
    ))

    # Approximate filtered non-fraud using global totals
    # (Since the precomputed file contains counts for fraud and all transactions by type,
    # but not non-fraud counts directly.)
    total_by_type = kpis["transaction_distribution"]
    filtered_total = int(sum(
        total_by_type.get(t, 0) for t in selected_types
    ))

    filtered_non_fraud_total = int(filtered_total - filtered_fraud_total)

    # Fraud vs Non-Fraud Pie Chart (within selected transaction types)

    fraud_count = filtered_fraud_total
    non_fraud_count = max(filtered_non_fraud_total, 0)

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
        title="Fraud vs Non-Fraud (Selected Types)"
    )

    st.plotly_chart(
        pie_fig,
        width="stretch"
    )


    st.divider()

    st.markdown(
        """
        <div style="
          background:rgba(255,255,255,0.06);
          border:1px solid rgba(255,255,255,0.14);
          border-radius:14px;
          padding:10px 14px;
          font-weight:900;
          letter-spacing:0.3px;
          color:#E5E7EB;
          margin-bottom:8px;
        ">
          🔎 Fraud Count by Transaction Type (Selected)
        </div>
        """,
        unsafe_allow_html=True
    )

    fraud_df = pd.DataFrame(
        [
            (t, v)
            for t, v in kpis["fraud_by_type"].items()
            if t in selected_types
        ],
        columns=["Transaction Type", "Fraud Count"],
    )


    fraud_fig = px.bar(
        fraud_df,
        x="Transaction Type",
        y="Fraud Count",
        title="Fraud by Transaction Type",
    )

    fraud_fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        height=500,
    )

    st.plotly_chart(
        fraud_fig,
        width="stretch",
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

    st.markdown(
        """
        <div style="
          background:linear-gradient(90deg, rgba(34,197,94,0.22), rgba(37,99,235,0.18), rgba(124,58,237,0.18));
          border:1px solid rgba(255,255,255,0.16);
          border-radius:16px;
          padding:12px 16px;
          font-weight:950;
          letter-spacing:0.2px;
          color:white;
          margin-bottom:10px;
        ">
          🔍 Fraud Detection Feature Importance
        </div>
        """,
        unsafe_allow_html=True
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
        <div style="
          background:linear-gradient(90deg, rgba(37,99,235,0.24), rgba(124,58,237,0.22));
          border:1px solid rgba(255,255,255,0.16);
          border-radius:16px;
          padding:12px 16px;
          font-weight:950;
          letter-spacing:0.2px;
          color:white;
          margin-bottom:10px;
        ">
          🤖 AI Fraud Risk Assessment
        </div>

        

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

        # Match model training feature order:
        # ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
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

            st.markdown(
f"""
<div style="
padding:20px;
border-radius:15px;
background:#fee2e2;
text-align:center;
font-size:24px;
font-weight:bold;
">
🚨 HIGH RISK TRANSACTION
</div>
""",
unsafe_allow_html=True
)

        elif fraud_probability >= 40:

            st.markdown(
f"""
<div style="
padding:20px;
border-radius:15px;
background:#fee2e2;
text-align:center;
font-size:24px;
font-weight:bold;
">
⚠️ MEDIUM RISK TRANSACTION
</div>
""",
unsafe_allow_html=True
)

        else:

            st.markdown(
f"""
<div style="
padding:20px;
border-radius:15px;
background:#fee2e2;
text-align:center;
font-size:24px;
font-weight:bold;
">
✅ LOW RISK TRANSACTION
</div>
""",
unsafe_allow_html=True
)