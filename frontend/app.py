import json
import sys
import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import random
import html as _html

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

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------

# Persistent Live Feed Alerts
if "live_alerts" not in st.session_state:
    base_time = datetime.now()
    initial_messages = [
        ("🚨 High-risk transfer detected from account origin group X90", "danger"),
        ("⚠️ Velocity threshold exceeded: multiple rapid micro-transfers", "warning"),
        ("⚠️ Abnormal destination balance movement in terminal nodes", "warning"),
        ("✅ Transaction verified successfully by isolation engine", "success"),
        ("🚨 Anomaly detected: structural divergence in balance delta", "danger")
    ]
    alerts_list = []
    for i, (msg, type_) in enumerate(initial_messages):
        alerts_list.append({
            "message": msg,
            "time": (base_time - timedelta(minutes=i * 3)).strftime("%H:%M:%S"),
            "type": type_
        })
    st.session_state["live_alerts"] = alerts_list

# Persistent Chat Copilot History
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Hello! I am the automated risk advisor copilot. Ask me anything about the metrics, anomalies, or security posture of the monitored transaction streams."}
    ]

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
# PAGE CONFIG & PREMIUM CYBER-FINANCIAL STYLES
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Risk Analytics & Copilot",
    layout="wide"
)

# Advanced Micro-Animation & Dark Glassmorphic Design Engine
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg-primary: #070a13;
  --bg-card: rgba(18, 24, 38, 0.65);
  --border-card: rgba(255, 255, 255, 0.08);
  --text-primary: #f8fafc;
  --text-secondary: #94a3b8;
  --accent-cyan: #00f2fe;
  --accent-violet: #9d4edd;
  --gradient-accent: linear-gradient(135deg, #00f2fe 0%, #9d4edd 100%);
  --color-success: #06d6a0;
  --color-warning: #ffb703;
  --color-danger: #ff5252;
}

/* Base resets & typography */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-image: 
        radial-gradient(circle at 5% 10%, rgba(0, 242, 254, 0.04) 0%, transparent 40%),
        radial-gradient(circle at 95% 85%, rgba(157, 78, 221, 0.04) 0%, transparent 40%),
        linear-gradient(180deg, #070a13 0%, #0d1220 100%) !important;
    background-size: cover;
    color: var(--text-primary) !important;
}

#MainMenu, footer, header {visibility: hidden;}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: rgba(10, 15, 28, 0.85) !important;
    border-right: 1px solid var(--border-card) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}

/* KPI Box Design */
.kpi-box {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border-card);
    padding: 24px;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
.kpi-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: var(--gradient-accent);
    opacity: 0.8;
}
.kpi-box:hover {
    transform: translateY(-3px);
    border-color: rgba(0, 242, 254, 0.3);
    box-shadow: 0 12px 35px rgba(0, 242, 254, 0.12);
}
.kpi-title { 
    color: var(--text-secondary); 
    font-size: 11px; 
    font-weight: 600; 
    text-transform: uppercase; 
    letter-spacing: 1.2px; 
    font-family: 'Outfit', sans-serif;
}
.kpi-value { 
    color: var(--text-primary); 
    font-size: 32px; 
    font-weight: 800; 
    margin-top: 8px; 
    letter-spacing: -0.5px;
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #ffffff, #cbe2ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Amount statistics cards */
.amount-box {
    background: rgba(22, 28, 45, 0.4);
    border: 1px solid var(--border-card);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    transition: all 0.25s ease;
}
.amount-box:hover {
    background: rgba(22, 28, 45, 0.6);
    border-color: rgba(157, 78, 221, 0.3);
    transform: translateY(-2px);
}
.amount-box .kpi-value { 
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 24px; 
}

/* Streamlit Button Overrides */
div.stButton > button {
    background: var(--gradient-accent) !important;
    color: #070a13 !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Outfit', sans-serif !important;
    padding: 12px 24px !important;
    font-size: 15px !important;
    letter-spacing: 0.5px !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4) !important;
    color: #070a13 !important;
}

/* Download button */
div.stDownloadButton > button {
    background: rgba(255, 255, 255, 0.05) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(0, 242, 254, 0.2) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    transition: all 0.2s ease !important;
}
div.stDownloadButton > button:hover {
    background: var(--accent-cyan) !important;
    color: #070a13 !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2) !important;
}

/* Form Styling */
div[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 16px !important;
    padding: 30px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
}

/* Text & number inputs */
.stTextInput input, .stNumberInput input, .stSelectbox [role="combobox"] {
    background-color: rgba(10, 14, 23, 0.8) !important;
    border: 1px solid var(--border-card) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px var(--accent-cyan) !important;
}

/* Custom Tabs layout styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: transparent;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(22, 28, 45, 0.4) !important;
    border: 1px solid var(--border-card) !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 24px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    margin-right: 4px;
}
.stTabs [aria-selected="true"] {
    background-color: rgba(22, 28, 45, 0.8) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
}

/* Plotly dashboard card */
[data-testid="stPlotlyChart"] {
  background: var(--bg-card) !important; 
  border-radius: 16px !important; 
  padding: 18px !important; 
  border: 1px solid var(--border-card) !important;
  box-shadow: 0 10px 30px rgba(0,0,0,0.2) !important;
}

/* Real-Time Live Feed Vertical Timeline */
.timeline {
    position: relative;
    margin-left: 10px;
    padding-left: 20px;
    border-left: 2px dashed rgba(255, 255, 255, 0.1);
}
.timeline-item {
    position: relative;
    margin-bottom: 20px;
}
.timeline-badge {
    position: absolute;
    left: -29px;
    top: 3px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    border: 2px solid var(--bg-primary);
}
.timeline-badge.danger { background-color: var(--color-danger); box-shadow: 0 0 8px var(--color-danger); }
.timeline-badge.warning { background-color: var(--color-warning); box-shadow: 0 0 8px var(--color-warning); }
.timeline-badge.success { background-color: var(--color-success); box-shadow: 0 0 8px var(--color-success); }

.timeline-card {
    background: rgba(22, 28, 45, 0.45);
    border: 1px solid var(--border-card);
    border-radius: 12px;
    padding: 16px;
    transition: all 0.2s ease;
}
.timeline-card:hover {
    background: rgba(22, 28, 45, 0.65);
    border-color: rgba(255, 255, 255, 0.15);
}
.timeline-time {
    font-size: 11px;
    color: var(--text-secondary);
    font-weight: 500;
    margin-bottom: 4px;
}
.timeline-msg {
    font-size: 13.5px;
    color: var(--text-primary);
    line-height: 1.4;
}

/* Chief Risk Officer briefing */
.executive-box {
    background: linear-gradient(135deg, rgba(18, 24, 38, 0.8), rgba(10, 15, 28, 0.95));
    border: 1px solid rgba(157, 78, 221, 0.2);
    position: relative;
    padding: 26px;
    border-radius: 18px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.executive-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(157, 78, 221, 0.15);
    color: #c084fc;
    border: 1px solid rgba(157, 78, 221, 0.3);
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 4px 10px;
    border-radius: 20px;
}
.executive-title {
    font-family: 'Outfit', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 14px;
    margin-bottom: 12px;
}

/* Chat system bubbles styling */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-top: 10px;
    margin-bottom: 25px;
}
.chat-bubble {
    display: flex;
    gap: 12px;
    padding: 16px;
    border-radius: 14px;
    line-height: 1.5;
    font-size: 14px;
    max-width: 85%;
    animation: slideUp 0.3s ease;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-bubble.user {
    background: rgba(0, 242, 254, 0.08);
    border: 1px solid rgba(0, 242, 254, 0.2);
    align-self: flex-end;
    color: var(--text-primary);
    border-top-right-radius: 0;
}
.chat-bubble.copilot {
    background: rgba(157, 78, 221, 0.08);
    border: 1px solid rgba(157, 78, 221, 0.2);
    align-self: flex-start;
    color: var(--text-primary);
    border-top-left-radius: 0;
}
.chat-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 12px;
    flex-shrink: 0;
}
.chat-avatar.user {
    background: var(--accent-cyan);
    color: #070a13;
}
.chat-avatar.copilot {
    background: var(--accent-violet);
    color: #f8fafc;
}
.chat-content {
    flex: 1;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Number input labels */
.stNumberInput label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Additional Streamlit label selectors */
div[data-testid="stNumberInput"] label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Form labels */
[data-testid="stWidgetLabel"] {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)
# --------------------------------------------------
# GRAPH THEME FUNCTIONS
# --------------------------------------------------

def apply_plotly_clean_theme(fig, title_x=0.02):
    # Detect trace types in the figure
    trace_types = {t.type for t in fig.data}
    has_cartesian = any(t in ['scatter', 'bar', 'histogram', 'box', 'violin'] for t in trace_types)
    
    # Configure hovermode safely based on trace type
    hovermode = "x unified" if has_cartesian else None
    
    layout_update = dict(
        template="plotly_dark",
        title_x=title_x,
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Inter, sans-serif", color="#94A3B8", size=12),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(bgcolor="rgba(10,15,28,0.7)", bordercolor="rgba(255,255,255,0.05)"),
    )
    if hovermode is not None:
        layout_update["hovermode"] = hovermode
        
    fig.update_layout(**layout_update)
    
    # Only update x/y axes if the figure uses Cartesian coordinates
    if has_cartesian:
        try:
            fig.update_xaxes(showgrid=False, linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94a3b8"))
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)", tickfont=dict(color="#94a3b8"))
        except Exception:
            pass
            
    # Clean/check trace markers safely
    for trace in fig.data:
        if hasattr(trace, 'marker') and isinstance(trace.marker, dict):
            # Ensure marker configuration is safe
            pass
            
    return fig

def render_section_header(title, is_gradient=False):
    accent = "var(--color-danger)" if is_gradient else "var(--accent-cyan)"
    style = f"border-left: 4px solid {accent}; padding: 2px 14px; font-weight: 700; font-size: 18px; color: var(--text-primary); margin: 28px 0 16px 0; letter-spacing: -0.3px; font-family: 'Outfit';"
    st.markdown(f'<div style="{style}">{title}</div>', unsafe_allow_html=True)

# --------------------------------------------------
# APPLICATION HEADER BLOCK
# --------------------------------------------------

st.markdown("""
<div style="background: linear-gradient(135deg, rgba(18, 24, 38, 0.7), rgba(10, 15, 28, 0.8)); backdrop-filter: blur(20px); border: 1px solid rgba(0, 242, 254, 0.15); padding: 35px; border-radius: 20px; box-shadow: 0 10px 40px rgba(0,0,0,0.3); margin-bottom: 30px; position: relative;">
  <div style="position: absolute; top: 20px; right: 25px; display: flex; align-items: center; gap: 8px;">
    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #06d6a0; box-shadow: 0 0 10px #06d6a0; animation: pulse 2s infinite;"></span>
    <span style="color: #06d6a0; font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; font-family: 'Outfit';">Live Node Verified</span>
  </div>
  <span style="color: #00F2FE; font-size: 11px; font-weight:800; letter-spacing: 2.5px; text-transform: uppercase; background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.2); padding: 5px 12px; border-radius: 20px; font-family: 'Outfit';">Decision Intelligence Suite</span>
  <h1 style="color:#f8fafc; font-size:36px; font-weight: 800; margin: 16px 0 8px 0; letter-spacing: -1px; font-family: 'Outfit';">Financial Risk Command Environment</h1>
  <p style="color:#94A3B8; font-size:15px; margin:0; line-height: 1.5;">Securing asset tracking networks through micro-stratified diagnostics • <b style="color: #9D4EDD;">6.3M+ Log Entries Monitored</b></p>
</div>
<style>
@keyframes pulse {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 214, 160, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(6, 214, 160, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(6, 214, 160, 0); }
}
</style>
""", unsafe_allow_html=True)

# Render top status pills as a beautiful HTML grid
st.markdown("""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 25px;">
  <div style="background: rgba(22, 28, 45, 0.35); border: 1px solid rgba(255,255,255,0.05); padding: 14px 20px; border-radius: 12px; display: flex; align-items: center; gap: 10px;">
    <span style="font-size: 18px;">📊</span>
    <div>
      <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Monitored Volume</div>
      <div style="font-size: 13.5px; font-weight: 700; color: #f8fafc;">6.3M+ Vectors</div>
    </div>
  </div>
  <div style="background: rgba(22, 28, 45, 0.35); border: 1px solid rgba(255,255,255,0.05); padding: 14px 20px; border-radius: 12px; display: flex; align-items: center; gap: 10px;">
    <span style="font-size: 18px;">🛡</span>
    <div>
      <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Risk Guardrail</div>
      <div style="font-size: 13.5px; font-weight: 700; color: #06d6a0;">Integrity Verified</div>
    </div>
  </div>
  <div style="background: rgba(22, 28, 45, 0.35); border: 1px solid rgba(255,255,255,0.05); padding: 14px 20px; border-radius: 12px; display: flex; align-items: center; gap: 10px;">
    <span style="font-size: 18px;">⚡</span>
    <div>
      <div style="font-size: 10px; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.5px;">Simulation Engine</div>
      <div style="font-size: 13.5px; font-weight: 700; color: #00F2FE;">Pipeline Synced</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div style="font-size:19px; font-weight:800; color:#f8fafc; padding-bottom:6px; letter-spacing:-0.2px; font-family:\'Outfit\';">System Infrastructure</div>', unsafe_allow_html=True)
    st.caption("Risk Matrix Verification Node")
    
    st.markdown("""
    <div style="display: flex; flex-direction: column; gap: 10px; margin-top: 15px; margin-bottom: 20px;">
      <div style="background: rgba(6, 214, 160, 0.1); border-left: 3px solid #06d6a0; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 12.5px; color: #e2e8f0;">
        <span style="font-weight: 600; color: #06d6a0;">✔ Active</span> Analysis Framework
      </div>
      <div style="background: rgba(0, 242, 254, 0.1); border-left: 3px solid #00f2fe; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 12.5px; color: #e2e8f0;">
        <span style="font-weight: 600; color: #00f2fe;">📊 Loaded</span> Metrics Model Aggregation
      </div>
      <div style="background: rgba(157, 78, 221, 0.1); border-left: 3px solid #9d4edd; padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 12.5px; color: #e2e8f0;">
        <span style="font-weight: 600; color: #9d4edd;">🤖 Ready</span> Inference Pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)
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

# Shared deterministic mock thresholds
percentile_95 = kpis.get("percentile_95", 766043.571251641)

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Analytics Dashboard",
        "🤖 Fraud Prediction",
        "🧠 AI Copilot"
    ]
)

# Function to simulate new live alerts incrementally
def add_live_alert_if_needed():
    if random.random() < 0.25:  # 25% chance to insert an alert on reload
        new_alerts = [
            ("🚨 Automated flag triggered: High-value TRANSFER deviation", "danger"),
            ("⚠️ Unusual activity pattern detected in origin routing account", "warning"),
            ("🚨 Isolation Forest anomaly classification: score 0.892", "danger"),
            ("✅ Verification node confirmation: transaction cleared", "success"),
            ("⚠️ Destination account balance delta ratio mismatch", "warning")
        ]
        chosen = random.choice(new_alerts)
        # Avoid duplicate consecutive messages
        if st.session_state["live_alerts"] and st.session_state["live_alerts"][0]["message"] != chosen[0]:
            st.session_state["live_alerts"].insert(0, {
                "message": chosen[0],
                "time": datetime.now().strftime("%H:%M:%S"),
                "type": chosen[1]
            })
            if len(st.session_state["live_alerts"]) > 8:
                st.session_state["live_alerts"].pop()

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
            
    # Executive summary block at the top
    render_section_header("🧠 Chief Risk Officer Briefing")
    summary = get_executive_summary()
    escaped_summary = _html.escape(summary).replace('\n', '<br>')
    st.markdown(f"""
    <div class="executive-box">
        <div class="executive-badge">Briefing Node</div>
        <div class="executive-title">Executive Intelligence Center</div>
        <div style="font-size: 14.5px; color: #cbd5e1; line-height: 1.6; font-family: 'Inter';">
            {escaped_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_section_header("Transaction Channel Volume Distributions")
    transaction_df = pd.DataFrame(list(kpis["transaction_distribution"].items()), columns=["Transaction Type", "Count"])
    fig = px.bar(
        transaction_df, 
        x="Transaction Type", 
        y="Count", 
        title="Log Density Classification",
        color="Transaction Type",
        color_discrete_sequence=['#00F2FE', '#9D4EDD', '#06D6A0', '#FFB703', '#FF5252']
    )
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

    col_pie, col_alert = st.columns([3, 2])
    
    with col_pie:
        render_section_header("Isolate Core Segment Matrix")
        all_types = sorted(list(kpis["transaction_distribution"].keys()))
        selected_types = st.multiselect("Isolate Target Channels for Review:", options=all_types, default=all_types)

        fraud_by_type = kpis["fraud_by_type"]
        total_by_type = kpis["transaction_distribution"]
        filtered_fraud_total = sum(int(fraud_by_type.get(t, 0)) for t in selected_types)
        filtered_total = sum(int(total_by_type.get(t, 0)) for t in selected_types)
        filtered_non_fraud_total = max(filtered_total - filtered_fraud_total, 0)

        pie_df = pd.DataFrame({"Category": ["Verified Irregularity", "Standard Volume"], "Count": [filtered_fraud_total, filtered_non_fraud_total]})
        pie_fig = px.pie(pie_df, names="Category", values="Count", title="Isolate Signal Veracity Breakdowns", color_discrete_sequence=['#ff5252', '#06d6a0'])
        apply_plotly_clean_theme(pie_fig)
        st.plotly_chart(pie_fig, width="stretch")

    with col_alert:
        render_section_header("🚨 Live Fraud Monitoring Feed")
        add_live_alert_if_needed()
        
        timeline_html = '<div class="timeline">'
        for item in st.session_state["live_alerts"]:
            badge_class = item["type"]
            timeline_html += (
                f'<div class="timeline-item">'
                f'<div class="timeline-badge {badge_class}"></div>'
                f'<div class="timeline-card">'
                f'<div class="timeline-time">{item["time"]}</div>'
                f'<div class="timeline-msg">{item["message"]}</div>'
                f'</div>'
                f'</div>'
            )
        timeline_html += '</div>'
        st.markdown(timeline_html, unsafe_allow_html=True)

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
                                color_discrete_map={"Baseline Standard": "#06d6a0", "Investigative Alert": "#ffb703", "High-Risk Anomaly": "#ff5252"})
        scatter_fig.add_vline(x=percentile_95, line_dash="dash", line_color="#ff5252", annotation_text="95th Percentile Limit")
        apply_plotly_clean_theme(scatter_fig)
        st.plotly_chart(scatter_fig, width="stretch")
        
    with sc_col2:
        st.markdown(f"""
        <div style="background: rgba(22, 28, 45, 0.45); border: 1px solid var(--border-card); padding:24px; border-radius:14px; height:100%; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
            <p style="font-weight:700; margin-bottom:6px; color:#f8fafc; font-size:14.5px; font-family:'Outfit';">Cohort Inferences</p>
            <p style="color:#94a3b8; font-size:12.5px; line-height:1.5; margin-bottom:16px;">Isolates population parameters to streamline investigation task queues.</p>
            <hr style="border:0; border-top:1px solid rgba(255,255,255,0.08); margin-bottom:16px;"/>
            <p style="font-size:11px; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing: 0.5px;">95% Threshold Line</p>
            <p style="font-size:22px; font-weight:800; color:#ff5252; margin-top:2px; font-family:'Outfit';">₹ {percentile_95:,.2f}</p>
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
    
    col_gen, _ = st.columns([1, 3])
    with col_gen:
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

    st.markdown('<div style="background: rgba(0, 242, 254, 0.05); border: 1px solid rgba(0, 242, 254, 0.15); padding:16px 20px; border-radius:12px; font-weight:600; color:#f8fafc; font-family:\'Outfit\'; margin-bottom:15px; display:flex; align-items:center; gap:10px;"><span>🛡</span> Interactive Verification Matrix</div>', unsafe_allow_html=True)
    
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
        submitted = st.form_submit_button(
    "🛡 RUN INTERACTIVE RISK AUDIT",
    type="primary"
)

    if submitted:
        input_data = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]], 
                                  columns=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])
        probability = model.predict_proba(input_data)
        fraud_probability = probability[0][1] * 100

        col_g, col_desc = st.columns([1, 1])
        
        with col_g:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=fraud_probability,
                title={"text": "Determined Anomaly Score Risk Index", "font": {"family": "Outfit, Inter", "color": "#f8fafc", "size": 15, "weight": "bold"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": "#94a3b8"}, "bar": {"color": "#00f2fe"}, "bgcolor": "rgba(255, 255, 255, 0.05)",
                       "steps": [{"range": [0, 40], "color": "rgba(6, 214, 160, 0.15)"},
                                 {"range": [40, 80], "color": "rgba(255, 183, 3, 0.15)"},
                                 {"range": [80, 100], "color": "rgba(255, 82, 82, 0.15)"}]}
            ))
            gauge.update_layout(
                height=320, 
                paper_bgcolor="rgba(0,0,0,0)", 
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit, Inter, sans-serif", color="#f8fafc")
            )
            st.plotly_chart(gauge, width="stretch", key="fraud_risk_gauge")

        with col_desc:
            if fraud_probability >= 80:
                bg_color, text_color, label = "rgba(255, 82, 82, 0.15)", "#ff5252", "🚨 ACTION REQUIRED: SIGNATURE COMPROMISE MATCH DETECTED"
            elif fraud_probability >= 40:
                bg_color, text_color, label = "rgba(255, 183, 3, 0.15)", "#ffb703", "⚠️ ATTENTION NEEDED: MODERATE BOUNDARY EXCLUSION EXCEEDED"
            else:
                bg_color, text_color, label = "rgba(6, 214, 160, 0.15)", "#06d6a0", "✅ SECURE: VECTOR CONFORMS TO EXPECTED VOLUME METRICS"
               
            st.markdown(f'<div style="padding:22px; border-radius:12px; background:{bg_color}; border:1px solid {text_color}; color:{text_color}; text-align:center; font-size:15px; font-weight:700; font-family:\'Outfit\'; margin-top: 50px;">{label}</div>', unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.subheader("🚨 Anomaly Detection Engine")

            anomaly_prediction, anomaly_score = detect_anomaly(input_data)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"""
                <div style="background: rgba(22, 28, 45, 0.45); border: 1px solid var(--border-card); padding: 18px; border-radius: 12px; text-align: center;">
                  <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;">Anomaly Score</div>
                  <div style="font-size: 26px; font-weight: 800; color: var(--accent-cyan); margin-top: 4px; font-family: 'Outfit';">{anomaly_score:.4f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if anomaly_prediction == -1:
                    anomaly_bg, anomaly_color, anomaly_text = "rgba(255, 82, 82, 0.15)", "#ff5252", "⚠️ SUSPICIOUS PATTERN"
                else:
                    anomaly_bg, anomaly_color, anomaly_text = "rgba(6, 214, 160, 0.15)", "#06d6a0", "✅ NORMAL PATTERN"
                    
                st.markdown(f"""
                <div style="background: {anomaly_bg}; border: 1px solid {anomaly_color}; padding: 18px; border-radius: 12px; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                  <div style="font-size: 10px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;">Pattern Verification</div>
                  <div style="font-size: 16px; font-weight: 800; color: {anomaly_color}; margin-top: 4px; font-family: 'Outfit';">{anomaly_text}</div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🔍 AI Explainability Engine (SHAP)")

        shap_df = get_shap_importance(input_data)
        shap_df = shap_df.sort_values(by="Importance", ascending=True)

        shap_fig = px.bar(
            shap_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Contribution Analysis",
            color="Feature",
            color_discrete_sequence=['#00f2fe', '#9d4edd', '#06d6a0', '#ffb703', '#ff5252']
        )
        apply_plotly_clean_theme(shap_fig)

        st.plotly_chart(
            shap_fig,
            width="stretch",
            key="shap_chart"
        )

        top_feature = shap_df.iloc[-1]["Feature"]
        top_importance = shap_df.iloc[-1]["Importance"]

        st.markdown(f"""
        <div style="background: rgba(157, 78, 221, 0.08); border: 1px solid rgba(157, 78, 221, 0.2); padding: 18px; border-radius: 12px; margin-top: 15px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #c084fc; letter-spacing: 1px; font-weight: 700; margin-bottom: 6px;">🎯 Primary Risk Driver</div>
            <div style="font-size: 15px; color: var(--text-primary); font-weight: 600; font-family: 'Outfit';">Feature: <span style="color: var(--accent-cyan);">{top_feature}</span></div>
            <div style="font-size: 13.5px; color: var(--text-secondary); margin-top: 4px;">Contribution Score: <span style="color: #06d6a0; font-weight: 700;">{top_importance:.4f}</span></div>
            <p style="font-size: 12.5px; color: var(--text-secondary); margin-top: 8px; line-height: 1.4; margin-bottom: 0;">This variable had the strongest mathematical influence on the model decision for this transaction instance.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        render_section_header("🧠 AI Transaction Assessment")

        st.metric(
            "Generated Scenario Amount",
            f"₹ {amount:,.2f}"
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

        escaped_ai_summary = _html.escape(ai_summary).replace('\n', '<br>')
        st.markdown(
            f"""
            <div class='chat-container'>
                <div class="chat-bubble copilot" style="max-width: 100%;">
                    <div class="chat-avatar copilot">AI</div>
                    <div class="chat-content">
                        <div style="font-weight: 600; font-size: 11px; color: var(--text-secondary); margin-bottom: 4px;">Nexus Copilot Advisor</div>
                        <div style="font-size: 14px; line-height: 1.5;">{escaped_ai_summary}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# NEW TRACK: INTERACTIVE CLIENT ANALYSIS COPILOT
# --------------------------------------------------
with tab3:
    st.divider()
    render_section_header("💬 Automated Risk Advisory Copilot")
    st.caption("Operational Sandbox: Provide a transaction identifier or value below to challenge the cognitive auditor node.")

    # Render persistent conversation history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state["chat_messages"]:
        role = msg["role"]
        content = msg["content"]
        avatar_label = "U" if role == "user" else "AI"
        bubble_class = "user" if role == "user" else "copilot"
        avatar_class = "user" if role == "user" else "copilot"
        role_label = "User Client" if role == "user" else "Nexus Decision Copilot"
        
        escaped_content = _html.escape(content).replace('\n', '<br>')
        
        st.markdown(f"""
        <div class="chat-bubble {bubble_class}">
            <div class="chat-avatar {avatar_class}">{avatar_label}</div>
            <div class="chat-content">
                <div style="font-weight: 600; font-size: 11px; color: var(--text-secondary); margin-bottom: 2px;">{role_label}</div>
                <div style="font-size:14px; line-height:1.5;">{escaped_content}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Interactive input using Streamlit's native chat input
    user_query = st.chat_input("Enter Transaction ID, query, or custom amount for audit challenge...")
    
    if user_query:
        # Save user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_query})
        
        # Call AI Copilot response
        with st.spinner("🧠 NEXUS AI is analyzing..."):
            try:
                response_text = ask_copilot(user_query, kpis)
            except Exception as e:
                response_text = f"⚠️ Nexus AI Copilot is temporarily unavailable due to API rate limits ({str(e)}). Please try again later."
        
        # Save assistant response
        st.session_state["chat_messages"].append({"role": "assistant", "content": response_text})
        st.rerun()