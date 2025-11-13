from openai import OpenAI
import os, json
import pandas as pd
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_fraud_insights(
    articles: list,
    user_query: str,
    date_range: tuple = None
):
    """
    Federal Reserve–focused insight generator.
    Analyzes cleaned press releases (press_releases_clean table) to identify
    regulatory, compliance, fraud, and monetary policy themes and trends.

    Returns an AI-generated narrative summary and chart instructions.
    """

    start, end = date_range if date_range else ("N/A", "N/A")

    # --- Helper: JSON-safe conversion ---
    def make_json_safe(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, (dict, list)):
            return json.loads(json.dumps(obj, default=str))
        return obj

    # --- Limit articles for token safety ---
    articles_preview = []
    for a in articles[:10]:  # cap to keep tokens safe
        articles_preview.append({
            "title": a.get("title"),
            "source": a.get("source", "Federal Reserve"),
            "date": (
                a.get("date").strftime("%Y-%m-%d")
                if isinstance(a.get("date"), (pd.Timestamp, datetime))
                else str(a.get("date"))
            ),
            "excerpt": (a.get("content") or "")[:600],
            "url": a.get("url")
        })

    safe_articles = json.dumps(make_json_safe(articles_preview), indent=2, default=str)

    # --- Build LLM Prompt ---
    prompt = f"""
You are a senior Federal Reserve analyst specializing in financial regulation, compliance, and fraud intelligence.

Your task:
Analyze the following Federal Reserve PRESS RELEASES to uncover trends related to:
- financial misconduct, fraud, and enforcement actions
- compliance and risk management practices
- cybersecurity and operational resilience
- monetary policy, inflation, and market confidence signals

User query:
"{user_query}"

Date range: {start} → {end}

FEDERAL RESERVE ARTICLES (summaries/excerpts):
{safe_articles}

INSTRUCTIONS:
1. Write a cohesive 2–4 paragraph narrative summary of fraud and regulatory insights.
   - Identify emerging fraud types or enforcement patterns.
   - Mention key financial topics (policy changes, oversight focus).
   - Highlight any cross-cutting risks (cyber, AML, systemic issues).

2. Then produce a JSON object with 2–4 chart instructions to visualize the trends.
   Example chart ideas:
   - "Mentions of Fraud/Compliance by Year"
   - "Distribution of Regulatory Topics"
   - "Monetary Policy Mentions Over Time"
   - "Enforcement Actions by Theme"

Return ONLY valid JSON:
{{
  "summary_text": "...",
  "visual_instructions": {{
      "charts": [
          {{
            "type": "line",
            "x": "date",
            "y": "fraud_mentions",
            "title": "Mentions of Fraud & Enforcement in Fed Releases"
          }}
      ]
  }}
}}
"""

    # --- OpenAI API Call ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in central bank communication analysis. "
                        "You interpret Federal Reserve press releases for fraud, compliance, and monetary policy signals. "
                        "Always return clean, valid JSON only. "
                        "Each chart instruction should include: type, x, y, and title keys."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "summary_text": f"Error generating Federal Reserve insights: {e}",
            "visual_instructions": {"charts": []},
        }
