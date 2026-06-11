import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Load multiple keys or fallback to the single key
api_keys_str = os.getenv("GEMINI_API_KEYS", "")
if api_keys_str:
    GEMINI_API_KEYS = [k.strip() for k in api_keys_str.split(",") if k.strip()]
else:
    single_key = os.getenv("GEMINI_API_KEY")
    GEMINI_API_KEYS = [single_key] if single_key else []

# Pre-initialize clients
clients = [genai.Client(api_key=key) for key in GEMINI_API_KEYS]

def call_gemini(model, prompt):
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