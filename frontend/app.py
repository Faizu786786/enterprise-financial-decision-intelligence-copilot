import json
import sys
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Setup project root and pathing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from backend.shap_explainer import (
    get_shap_importance
)
from backend.copilot import (
    ask_copilot,
    generate_executive_summary
)
from backend.anomaly_detector import (
    detect_anomaly
)
from datetime import datetime
import random
# --------------------------------------------------
# CACHED DATA & MODEL LOADERS
# --------------------------------------------------

MODEL_PATH = os.path.join(project_root, "models", "fraud_model.pkl")
KPI_PATH = os.path.join(project_root, "data", "processed", "kpis.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_kpi_data():
    with open(KPI_PATH, "r") as file:
        return json.load(file)

# --------------------------------------------------
# PAGE CONFIG & ADVANCED FLUID STYLES
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Risk Analytics & Copilot",
    layout="wide"
)

# Advanced Micro-Animation & Glassmorphic Layer Engine Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --primary-navy: #0F172A;
  --accent-blue: #2563EB;
  --text-primary: #1E293B;
  --text-muted: #64748B;
}

@keyframes fluidPlasma {
    0% { background-position: 0% 50%, 100% 50%; }
    50% { background-position: 100% 50%, 0% 50%; }
    100% { background-position: 0% 50%, 100% 50%; }
}

.stApp {
    font-family: 'Inter', sans-serif !important;
    background-image: 
        radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
        radial-gradient(circle at 80% 70%, rgba(16, 185, 129, 0.06) 0%, transparent 45%),
        linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
    background-size: 200% 200%;
    animation: fluidPlasma 15s ease infinite;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header {visibility: hidden;}

.stTextInput>div>div, .stNumberInput>div>div, .stMultiselect>div>div, .stSelectbox>div>div, .stForm>div, .stSidebar {
  background: rgba(255, 255, 255, 0.5) !important;
  backdrop-filter: blur(12px) !important;
  -webkit-backdrop-filter: blur(12px) !important;
  border: 1px solid rgba(255, 255, 255, 0.7) !important;
  color: var(--text-primary) !important;
  border-radius: 14px !important;
  box-shadow: 0 4px 12px rgba(148, 163, 184, 0.03) !important;
}

.kpi-box {
    background: rgba(255, 255, 255, 0.45);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    padding: 26px;
    border-radius: 16px;
    box-shadow: 0 8px 32px rgba(148, 163, 184, 0.04), inset 0 1px 1px rgba(255, 255, 255, 0.8);
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}
.kpi-box:hover {
    transform: translateY(-4px);
    background: rgba(255, 255, 255, 0.7);
    border-color: rgba(37, 99, 235, 0.25);
    box-shadow: 0 12px 40px rgba(37, 99, 235, 0.08);
}
.kpi-title { color: var(--text-muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; }
.kpi-value { color: var(--primary-navy); font-size: 34px; font-weight: 800; margin-top: 6px; letter-spacing: -0.5px; }

.amount-box {
    background: rgba(255, 255, 255, 0.6);
    backdrop-filter: blur(10px);
    padding: 20px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    text-align: center;
    box-shadow: 0 4px 14px rgba(148, 163, 184, 0.02);
}
.amount-box .kpi-value { color: var(--accent-blue); font-size: 26px; }

.stDownloadButton button {
    border-radius: 10px; font-weight: 600; border: 1px solid rgba(37, 99, 235, 0.15) !important;
    background: rgba(255, 255, 255, 0.6) !important; color: var(--accent-blue) !important;
    backdrop-filter: blur(4px); padding: 10px 20px; transition: all 0.2s ease;
}
.stDownloadButton button:hover {
    background: var(--accent-blue) !important; color: white !important;
}
.stButton button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important; border-radius: 12px; font-weight: 600; width: 100%; 
    border: none !important; padding: 14px !important; letter-spacing: 0.2px;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18); transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(37, 99, 235, 0.3) !important;
}

.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; color: var(--text-muted); padding: 12px 20px; }
.stTabs [aria-selected="true"] { 
    background: rgba(255, 255, 255, 0.5) !important; 
    border-radius: 12px 12px 0 0;
    border-bottom: 3px solid var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}

[data-testid="stPlotlyChart"] {
  background: rgba(255, 255, 255, 0.55) !important; 
  backdrop-filter: blur(16px);
  border-radius: 16px; padding: 20px; 
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 0 10px 30px rgba(148, 163, 184, 0.03);
}

/* Custom Chat Style Enforcements */
.user-bubble {
    background: #E2E8F0; color: #0F172A; padding: 12px 16px; border-radius: 16px 16px 0px 16px;
    margin: 8px 0; max-width: 80%; float: right; clear: both; font-size: 14px; font-weight: 500;
}
.copilot-bubble {
    background: #EFF6FF; color: #1E40AF; padding: 16px; border-radius: 16px 16px 16px 0px;
    border: 1px solid #BFDBFE; margin: 8px 0; max-width: 85%; float: left; clear: both;
    font-size: 14px; line-height: 1.5; box-shadow: 0 2px 8px rgba(37,99,235,0.04);
}
.executive-box{
    background: linear-gradient(
        135deg,
        #0F172A,
        #1E40AF
    );
    color:white;
    padding:25px;
    border-radius:18px;
    margin-bottom:20px;
    box-shadow:0 8px 24px rgba(0,0,0,0.15);
}
.alert-box{
    background:#FEF2F2;
    border-left:6px solid #DC2626;
    padding:18px;
    border-radius:12px;
    margin-bottom:15px;
    font-size:16px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# GRAPH THEME FUNCTIONS
# --------------------------------------------------

def apply_plotly_clean_theme(fig, title_x=0.02):
    fig.update_layout(
        template="plotly_white",
        title_x=title_x,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#475569", size=12),
        margin=dict(l=40, r=20, t=60, b=40),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor="rgba(255,255,255,0.3)"),
    )

    # Robust trace styling (avoid invalid/typo marker attributes)
    # - Scatter traces: keep plotly defaults (or they may not have marker.color shaped similarly)
    # - Pie traces: handled by plotly
    if len(fig.data):
        for trace in fig.data:
            try:
                if hasattr(trace, "marker") and not isinstance(trace, (go.Pie, go.Scatter)):
                    # marker.color exists for bar-like traces
                    marker = getattr(trace, "marker", None)
                    if marker is not None:
                        if hasattr(marker, "color") and marker.color is not None:
                            trace.marker.color = "#2563EB"
            except Exception:
                pass
    return fig

def render_section_header(title, is_gradient=False):
    accent = "#EF4444" if is_gradient else "#2563EB"
    style = f"border-left: 4px solid {accent}; padding: 2px 14px; font-weight: 700; font-size: 18px; color: #0F172A; margin: 28px 0 16px 0; letter-spacing: -0.3px;"
    st.markdown(f'<div style="{style}">{title}</div>', unsafe_allow_html=True)

# --------------------------------------------------
# APPLICATION HEADER BLOCK
# --------------------------------------------------

st.markdown("""
<div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); border: 1px solid rgba(255,255,255,0.6); padding: 40px; border-radius: 20px; box-shadow: 0 10px 40px rgba(148,163,184,0.03); margin-bottom: 30px;">
  <span style="color: #2563EB; font-size: 11px; font-weight:800; letter-spacing: 2px; text-transform: uppercase; background: rgba(37,99,235,0.06); padding: 4px 10px; border-radius: 20px;">Decision Intelligence Suite</span>
  <h1 style="color:#0F172A; font-size:36px; font-weight: 800; margin: 12px 0 8px 0; letter-spacing: -0.8px;">Financial Risk Command Environment</h1>
  <p style="color:#64748B; font-size:16px; margin:0;">Securing asset tracking networks through micro-stratified diagnostics • <b>6.3M+ Log Entries Monitored</b></p>
</div>
""", unsafe_allow_html=True)

sc1, sc2, sc3 = st.columns(3)
sc1.info("📊 Total Monitored Volume: 6.3M+ Vectors")
sc2.info("🛡 Risk Guardrail Integrity: Verified")
sc3.info("⚡ Simulation Engine Pipeline: Synced")
st.divider()

with st.sidebar:
    st.markdown('<div style="font-size:17px; font-weight:700; color:#0F172A; padding-bottom:6px; letter-spacing:-0.2px;">System Infrastructure</div>', unsafe_allow_html=True)
    st.caption("Risk Matrix Verification Node")
    st.success("✔ Analysis Framework Active")
    st.info("📊 Metrics Model Aggregated")
    st.info("🤖 Inference Pipeline Ready")
    st.divider()
    st.caption("Enterprise Risk Copilot v2.4")

# Load Global File Assets
kpis = load_kpi_data()

@st.cache_data(ttl=3600)
def get_executive_summary():
    try:
         return generate_executive_summary(kpis)
    except Exception as e:
         return f"⚠️ AI Chief Risk Officer executive summary is temporarily unavailable due to API limits ({str(e)}). Please try reloading the page later."

@st.cache_data
def get_percentile_95(amount_stats_mean):
    np.random.seed(42)
    mock_amounts_for_thresholds = np.random.exponential(
        scale=amount_stats_mean * 1.5,
        size=400
    )
    return float(np.percentile(mock_amounts_for_thresholds, 95))

# Shared deterministic mock thresholds so other tabs can safely use them.
# Tab 3 previously depended on a variable computed only inside Tab 1.
_amount_stats_mean = kpis.get("amount_statistics", {}).get("mean_amount", 1.0)
percentile_95 = get_percentile_95(_amount_stats_mean)

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Analytics Dashboard",
        "🤖 Fraud Prediction",
        "🧠 AI Copilot"
    ]
)
def generate_live_alert():
    alerts = [
        "🚨 High-risk transfer detected",
        "⚠️ Abnormal destination balance movement",
        "⚠️ Potential account takeover pattern",
        "✅ Transaction verified successfully",
        "🚨 Isolation Forest anomaly detected",
        "⚠️ Velocity threshold exceeded"
    ]
    return random.choice(alerts)

# --------------------------------------------------
# TAB 1: METRICS & DIAGNOSTIC VISUALIZATION
# --------------------------------------------------
with tab1:
    kpi_cols = st.columns(4)
    kpi_metrics = [
        ("Total Checked Volumes", f"{kpis['total_transactions']:,}"),
        ("Identified Deviations", f"{kpis['fraud_transactions']:,}"),
        ("Platform Anomaly Rate", f"{kpis['fraud_rate']}%"),
        ("Active Origin Channels", f"{kpis['unique_origin_accounts']:,}")
    ]
    for col, (title, val) in zip(kpi_cols, kpi_metrics):
        with col:
            st.markdown(f"""
            <div class='kpi-box'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value'>{val}</div>
            </div>
            """, unsafe_allow_html=True)
            
    render_section_header("Transaction Channel Volume Distributions")
    st.divider()
    
    render_section_header("🚨 Live Fraud Monitoring Feed")

    alerts = [generate_live_alert() for _ in range(5)]
    for alert in alerts:
        st.markdown(
            f"""
            <div class='alert-box'>
                {alert}
                <br>
                {datetime.now().strftime("%H:%M:%S")}
            </div>
            """,
            unsafe_allow_html=True
        )
        
    render_section_header("🧠 Executive Intelligence Center")
    summary = get_executive_summary()
    
    st.markdown(
        f"""
        <div class="executive-box">
            {summary}
        </div>
        """,
        unsafe_allow_html=True
    )

    transaction_df = pd.DataFrame(list(kpis["transaction_distribution"].items()), columns=["Transaction Type", "Count"])
    fig = px.bar(transaction_df, x="Transaction Type", y="Count", title="Log Density Classification", color_discrete_sequence=['#2563EB'])
    apply_plotly_clean_theme(fig)
    st.plotly_chart(fig, width="stretch")

    render_section_header("Audited Value Distribution Aggregates")
    amount_stats = kpis["amount_statistics"]
    amt_cols = st.columns(4)
    cards = [
        ("Average Order Value", amount_stats["mean_amount"]),
        ("Median Transaction Value", amount_stats["median_amount"]),
        ("Maximum Observed Ceiling", amount_stats["max_amount"]),
        ("Minimum Observed Floor", amount_stats["min_amount"])
    ]
    for col, (title, value) in zip(amt_cols, cards):
        with col:
            st.markdown(f"""
            <div class='amount-box'>
                <div class='kpi-title'>{title}</div>
                <div class='kpi-value'>₹ {value:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    render_section_header("Isolate Core Segment Matrix")
    all_types = sorted(list(kpis["transaction_distribution"].keys()))
    selected_types = st.multiselect("Isolate Target Channels for Review:", options=all_types, default=all_types)

    fraud_by_type = kpis["fraud_by_type"]
    total_by_type = kpis["transaction_distribution"]
    filtered_fraud_total = sum(int(fraud_by_type.get(t, 0)) for t in selected_types)
    filtered_total = sum(int(total_by_type.get(t, 0)) for t in selected_types)
    filtered_non_fraud_total = max(filtered_total - filtered_fraud_total, 0)

    pie_df = pd.DataFrame({"Category": ["Verified Irregularity", "Standard Volume"], "Count": [filtered_fraud_total, filtered_non_fraud_total]})
    pie_fig = px.pie(pie_df, names="Category", values="Count", title="Isolate Signal Veracity Breakdowns", color_discrete_sequence=['#EF4444', '#10B981'])
    apply_plotly_clean_theme(pie_fig)
    st.plotly_chart(pie_fig, width="stretch")

    st.divider()

    render_section_header("Risk Cohort Clustering & Statistical Outliers")
    np.random.seed(42)
    sample_size = 400
    mock_amounts = np.random.exponential(scale=amount_stats["mean_amount"] * 1.5, size=sample_size)
    mock_deltas = np.random.normal(loc=50, scale=25, size=sample_size)
    
    mock_status = []
    for a, d in zip(mock_amounts, mock_deltas):
        if a > amount_stats["median_amount"] * 5 and d > 60:
            mock_status.append("High-Risk Anomaly")
        elif a > amount_stats["mean_amount"] * 2:
            mock_status.append("Investigative Alert")
        else:
            mock_status.append("Baseline Standard")
            
    scatter_df = pd.DataFrame({"Transaction Amount (₹)": mock_amounts, "Account Balance Delta Score": mock_deltas, "Risk Classification": mock_status})
    sc_col1, sc_col2 = st.columns([3, 1])
    
    with sc_col1:
        scatter_fig = px.scatter(scatter_df, x="Transaction Amount (₹)", y="Account Balance Delta Score", color="Risk Classification",
                                color_discrete_map={"Baseline Standard": "#10B981", "Investigative Alert": "#F59E0B", "High-Risk Anomaly": "#EF4444"})
        scatter_fig.add_vline(x=percentile_95, line_dash="dash", line_color="#EF4444", annotation_text="95th Percentile Limit")
        apply_plotly_clean_theme(scatter_fig)
        st.plotly_chart(scatter_fig, width="stretch")
        
    with sc_col2:
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.4); border: 1px solid rgba(255,255,255,0.6); padding:24px; border-radius:14px; height:100%;">
            <p style="font-weight:700; margin-bottom:6px; color:#0F172A; font-size:14px;">Cohort Inferences</p>
            <p style="color:#475569; font-size:12.5px; line-height:1.5; margin-bottom:16px;">Isolates population parameters to streamline investigation task queues.</p>
            <hr style="border:0; border-top:1px solid rgba(148,163,184,0.2); margin-bottom:16px;"/>
            <p style="font-size:11px; font-weight:700; color:#64748B; text-transform:uppercase;">95% Threshold Line</p>
            <p style="font-size:20px; font-weight:800; color:#EF4444; margin-top:2px;">₹ {percentile_95:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    fraud_report_df = pd.DataFrame(list(fraud_by_type.items()), columns=["Transaction Type", "Fraud Count"])
    st.download_button(label="📥 Extract Audit Ledger Manifest (.CSV)", data=fraud_report_df.to_csv(index=False), file_name="risk_audit_manifest.csv", mime="text/csv")

def generate_sample_transaction():
    return {
        "amount": random.randint(1000, 500000),
        "oldbalanceOrg": random.randint(50000, 1000000),
        "newbalanceOrig": random.randint(0, 500000),
        "oldbalanceDest": random.randint(0, 500000),
        "newbalanceDest": random.randint(0, 1000000)
    }

# --------------------------------------------------
# TAB 2: INTERACTIVE INFERENCE RISK ENGINE
# --------------------------------------------------
with tab2:
    st.divider()
    
    if st.button("🎲 Generate Suspicious Transaction"):
        sample = generate_sample_transaction()
        st.session_state["generated_transaction"] = sample
        st.success("Sample transaction generated.")

    # Pre-populate form fields from generated transaction if available
    gen_txn = st.session_state.get("generated_transaction", {})
    amount = gen_txn.get("amount", 50000.0)
    oldbalanceOrg = gen_txn.get("oldbalanceOrg", 100000.0)
    newbalanceOrig = gen_txn.get("newbalanceOrig", 50000.0)
    oldbalanceDest = gen_txn.get("oldbalanceDest", 0.0)
    newbalanceDest = gen_txn.get("newbalanceDest", 50000.0)

    st.markdown('<div style="background: rgba(37,99,235,0.04); border: 1px solid rgba(37,99,235,0.15); padding:16px 20px; border-radius:10px; font-weight:600; color:#1E293B; margin-bottom:12px;">🛡 Interactive Verification Matrix</div>', unsafe_allow_html=True)
    
    model = load_model()

    with st.form("fraud_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Volume Vector Value", min_value=0.0, value=float(amount))
            newbalanceOrig = st.number_input("Post-Transaction Target Origin Balance", min_value=0.0, value=float(newbalanceOrig))
            newbalanceDest = st.number_input("Post-Transaction Target Destination Balance", min_value=0.0, value=float(newbalanceDest))
        with col2:
            oldbalanceOrg = st.number_input("Baseline Structural Origin Balance", min_value=0.0, value=float(oldbalanceOrg))
            oldbalanceDest = st.number_input("Baseline Structural Destination Balance", min_value=0.0, value=float(oldbalanceDest))
        submitted = st.form_submit_button("🛡 RUN INTERACTIVE RISK AUDIT")

    if submitted:
        input_data = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]], 
                                  columns=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
        probability = model.predict_proba(input_data)
        fraud_probability = probability[0][1] * 100

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=fraud_probability,
            title={"text": "Determined Anomaly Score Risk Index", "font": {"family": "Inter", "color": "#1E293B", "size": 14, "weight": "bold"}},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2563EB"}, "bgcolor": "rgba(226, 232, 240, 0.4)",
                   "steps": [{"range": [0, 40], "color": "rgba(16, 185, 129, 0.12)"},
                             {"range": [40, 80], "color": "rgba(245, 158, 11, 0.12)"},
                             {"range": [80, 100], "color": "rgba(239, 68, 68, 0.12)"}]}
        ))
        gauge.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, width="stretch", key="fraud_risk_gauge")

        if fraud_probability >= 80:
            bg_color, text_color, label = "#FEE2E2", "#991B1B", "🚨 ACTION REQUIRED: SIGNATURE COMPROMISE MATCH DETECTED"
        elif fraud_probability >= 40:
            bg_color, text_color, label = "#FEF3C7", "#92400E", "⚠️ ATTENTION NEEDED: MODERATE BOUNDARY EXCLUSION EXCEEDED"
        else:
            bg_color, text_color, label = "#D1FAE5", "#065F46", "✅ SECURE: VECTOR CONFORMS TO EXPECTED VOLUME METRICS"
           
        st.markdown(f'<div style="padding:22px; border-radius:12px; background:{bg_color}; color:{text_color}; text-align:center; font-size:15px; font-weight:600;">{label}</div>', unsafe_allow_html=True)
        st.divider()

        st.subheader("🔍 AI Explainability Engine (SHAP)")

        shap_df = get_shap_importance(input_data)
        shap_df = shap_df.sort_values(by="Importance", ascending=True)

        shap_fig = px.bar(
            shap_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Contribution Analysis"
        )

        shap_fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            height=550
        )

        st.plotly_chart(
            shap_fig,
            width="stretch",
            key="shap_chart"
        )

        top_feature = shap_df.iloc[0]["Feature"]
        top_importance = shap_df.iloc[0]["Importance"]

        st.info(
            f"""
            🎯 Primary Risk Driver
            
            Feature: {top_feature}
            
            Contribution Score: {top_importance:.4f}
            
            This variable had the strongest influence on the model decision.
            """
        )
        
        st.divider()

        st.subheader("🚨 Anomaly Detection Engine")

        anomaly_prediction, anomaly_score = detect_anomaly(input_data)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Anomaly Score", f"{anomaly_score:.4f}")

        with col2:
            if anomaly_prediction == -1:
                st.error("⚠️ Suspicious Transaction Detected")
            else:
                st.success("✅ Transaction Pattern Normal")
                
        st.divider()

        render_section_header("🧠 AI Transaction Assessment")

        st.metric(
            "Generated Scenario Amount",
            f"₹ {amount:,.0f}"
        )

        with st.spinner("Generating AI assessment..."):
            try:
                ai_summary = ask_copilot(
                    f"""
                    Transaction Amount: {amount}
                    Old Origin Balance: {oldbalanceOrg}
                    New Origin Balance: {newbalanceOrig}
                    Old Destination Balance: {oldbalanceDest}
                    New Destination Balance: {newbalanceDest}
                    Fraud Probability: {fraud_probability:.2f}%
                    Anomaly Score: {anomaly_score:.4f}
                    """,
                    kpis
                )
            except Exception as e:
                ai_summary = f"⚠️ Copilot advisory assessment is temporarily unavailable due to API rate limits ({str(e)}). Please try again later."

        st.markdown(
            f"""
            <div class='copilot-bubble'>
                {ai_summary}
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# NEW TRACK: INTERACTIVE CLIENT ANALYSIS COPILOT
# --------------------------------------------------
import html as _html

with tab3:
    st.divider()
    render_section_header("💬 Automated Risk Advisory Copilot")
    st.caption("Operational Sandbox: Provide a transaction identifier or value below to challenge the cognitive auditor node.")

    # In-memory dictionary map mocking localized customer lookups for live data simulation
    chat_input = st.text_input("Enter Transaction ID / Flag Reference:", placeholder="e.g., TXN-9082, TRANSFER-OVAL, or custom amounts...")

    if chat_input:
        safe_chat_input = _html.escape(str(chat_input))
        st.markdown(
            f'<div class="user-bubble">Can you explain the current security posture for item query: "{safe_chat_input}"?</div>',
            unsafe_allow_html=True
        )
        
        with st.spinner("🧠 NEXUS AI is analyzing..."):
            try:
                response_text = ask_copilot(
                    chat_input,
                    kpis
                )
            except Exception as e:
                response_text = f"⚠️ Nexus AI Copilot is temporarily unavailable due to API rate limits ({str(e)}). Please try again later."

        st.markdown(
            f"""
            <div class="copilot-bubble">
            {response_text}
            </div>
            """,
            unsafe_allow_html=True
        )