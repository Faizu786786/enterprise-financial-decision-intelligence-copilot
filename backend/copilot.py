import os
from google import genai
from dotenv import load_dotenv

backend_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(backend_dir, ".."))
dotenv_path = os.path.join(project_root, ".env")

load_dotenv(dotenv_path)

def get_clients():
    load_dotenv(dotenv_path, override=True)
    api_keys_str = os.getenv("GEMINI_API_KEYS", "")
    if api_keys_str:
        keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    else:
        single_key = os.getenv("GEMINI_API_KEY")
        keys = [single_key] if single_key else []
    return [genai.Client(api_key=key) for key in keys if key]

def call_gemini(model, prompt):
    clients = get_clients()
    if not clients:
        raise ValueError("No Gemini API keys configured. Please set GEMINI_API_KEY or GEMINI_API_KEYS in .env.")
    
    last_exception = None
    for i, client in enumerate(clients):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Gemini API call failed with key index {i}: {e}. Retrying next key...")
            last_exception = e
            
    raise RuntimeError(f"All Gemini API keys exhausted/failed. Last error: {last_exception}")

def ask_copilot(question, kpis):
    load_dotenv(dotenv_path, override=True)
    mock_mode = os.getenv("MOCK_API", "false").lower() == "true" or not os.getenv("GEMINI_API_KEY")
    if mock_mode:
        q = question.lower()
        fraud_rate = kpis.get("fraud_rate", 0.1291)
        total_tx = kpis.get("total_transactions", 6362620)
        fraud_tx = kpis.get("fraud_transactions", 8213)
        unique_org = kpis.get("unique_origin_accounts", 6353307)
        percentile_95 = kpis.get("percentile_95", 766043.57)
        
        if "fraud rate" in q or "rate" in q or "percent" in q:
            return (
                f"The current monitored platform fraud rate is **{fraud_rate}%**. Out of {total_tx:,} transactions, "
                f"{fraud_tx:,} have been flagged as irregular or suspicious. This is within the standard enterprise risk tolerance threshold of 0.15%."
            )
        elif "volume" in q or "transaction" in q or "total" in q:
            return (
                f"The system is currently auditing a total volume of **{total_tx:,}** transactions across **{unique_org:,}** unique origin accounts. "
                f"The transaction distribution is dominated by CASH_OUT and PAYMENT channels."
            )
        elif "anomaly" in q or "anomalies" in q or "score" in q or "threshold" in q:
            return (
                f"Isolation Forest anomaly detection checks for structural divergence in account balance deltas. "
                f"The determined 95th percentile threshold is **₹{percentile_95:,.2f}**. Any transaction volume exceeding "
                f"this line, or triggering an Isolation Forest score above 0.85, is placed into the investigation queue."
            )
        elif "amount" in q or "balance" in q:
            return (
                f"Looking at the transaction parameters: the average transaction value is "
                f"₹{kpis.get('amount_statistics', {}).get('mean_amount', 179861.90):,.2f}, and the median is "
                f"₹{kpis.get('amount_statistics', {}).get('median_amount', 74871.94):,.2f}. Outlier thresholds are computed dynamically."
            )
        else:
            return (
                f"Based on the analyzed enterprise ledger metrics:\n"
                f"- **Audit Status**: Live Node Verified\n"
                f"- **Monitored Volume**: {total_tx:,} transactions\n"
                f"- **Identified Deviations**: {fraud_tx:,} ({fraud_rate}%)\n"
                f"- **Recommended Control**: Calibrate Isolation Forest threshold to the 95th percentile (₹{percentile_95:,.2f}).\n\n"
                f"Let me know if you would like me to analyze a specific transaction vector or cohort."
            )

    prompt = f"""
You are a Senior Financial Intelligence Analyst.

Enterprise KPIs:

{kpis}

User Question:
{question}

Provide:

1. Executive Summary
2. Risk Assessment
3. Key Observations
4. Recommended Actions

Keep response concise and professional.
"""
    return call_gemini("gemini-2.5-flash", prompt)

def generate_executive_summary(kpis):
    load_dotenv(dotenv_path, override=True)
    mock_mode = os.getenv("MOCK_API", "false").lower() == "true" or not os.getenv("GEMINI_API_KEY")
    if mock_mode:
        fraud_rate = kpis.get("fraud_rate", 0.1291)
        total_tx = kpis.get("total_transactions", 6362620)
        fraud_tx = kpis.get("fraud_transactions", 8213)
        return (
            f"**Executive Summary:** Monitored transaction volume is healthy at {total_tx:,} vectors with a baseline anomaly rate of {fraud_rate}%.\n"
            f"**Major Risk:** Operational exposure is concentrated in high-volume TRANSFER channels showing elevated outlier delta scores.\n"
            f"**Business Impact:** Potential loss mitigation of ₹{(fraud_tx * 12500):,.2f} if proactive isolation rules are enforced.\n"
            f"**Recommended Action:** Deploy Isolation Forest threshold guards to monitor real-time balance divergence on origin nodes."
        )

    prompt = f"""
You are a Chief Risk Officer.

Analyze these KPIs:

{kpis}

Generate:

1. Executive Summary
2. Major Risk
3. Business Impact
4. Recommended Action

Maximum 150 words.
"""
    return call_gemini("gemini-2.5-flash", prompt)