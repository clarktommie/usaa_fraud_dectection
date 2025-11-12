from openai import OpenAI
import os, json
import pandas as pd
from datetime import datetime

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_fraud_insights(
    articles: list,
    complaints: dict,
    user_query: str,
    date_range: tuple = None
):
    """
    Unified fraud-intelligence engine.
    Blends patterns from ARTICLES + COMPLAINTS into one coherent insight output.
    Automatically trims data to stay within OpenAI token limits.
    """

    start, end = date_range if date_range else ("N/A", "N/A")

    # --- Helper: JSON-safe conversion ---
    def make_json_safe(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, (dict, list)):
            return json.loads(json.dumps(obj, default=str))
        return obj

    # --- Limit article size for token safety ---
    articles_preview = []
    for a in articles[:8]:  # limit number of articles
        articles_preview.append({
            "title": a.get("title"),
            "source": a.get("source"),
            "date": (
                a.get("date").strftime("%Y-%m-%d")
                if isinstance(a.get("date"), (pd.Timestamp, datetime))
                else str(a.get("date"))
            ),
            "excerpt": (a.get("content") or "")[:400],  # shorter excerpt
        })

    # --- Shrink complaints intelligently ---
    # If complaints is a list of rows, sample a small subset
    if isinstance(complaints, list):
        complaints_sample = complaints[:50]
    elif isinstance(complaints, dict):
        # Handle DataFrame-like dicts or aggregated complaint data
        if "rows" in complaints and isinstance(complaints["rows"], list):
            complaints_sample = complaints["rows"][:50]
        else:
            # reduce size if it's too long when stringified
            complaints_str = json.dumps(complaints, default=str)
            if len(complaints_str) > 8000:
                complaints_sample = {"summary": "complaint data truncated for brevity"}
            else:
                complaints_sample = complaints
    else:
        complaints_sample = {"summary": "no complaint data available"}

    safe_articles = json.dumps(articles_preview, indent=2, default=str)
    safe_complaints = json.dumps(make_json_safe(complaints_sample), indent=2, default=str)

    # --- Build model prompt ---
    prompt = f"""
You are a senior fraud-intelligence strategist.

Your job:
Blend insights from regulatory PRESS RELEASES and CONSUMER COMPLAINTS
to produce a unified, forward-looking fraud analysis.

User query:
"{user_query}"

Date range: {start} → {end}

REGULATORY ARTICLES (summaries/excerpts):
{safe_articles}

CONSUMER COMPLAINT PATTERNS:
{safe_complaints}

INSTRUCTIONS:
- Write a 2–4 paragraph narrative insight.
- Extract fraud themes, typologies, consumer harm areas, and regulatory signals.
- Connect complaint surges with enforcement or news cycles.
- Identify emerging fraud types or anomalies.
- Interpret *why* the pattern might exist (cybercrime trends, economic pressure, etc.).

Then output 2–4 chart instructions that visualize these findings.
Use ONLY these chart types:
["bar", "stacked_bar", "line", "area", "donut", "heatmap", "map"]

Return ONLY valid JSON:
{{
  "summary_text": "...",
  "visual_instructions": {{
      "charts": [
          {{
            "type": "line",
            "x": "date",
            "y": "identity_theft_count",
            "title": "Identity Theft Trend"
          }}
      ]
  }}
}}
"""

    # --- OpenAI API call ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in fraud analytics, financial crime, "
                        "and regulatory risk. Always return clean JSON. "
                        "Your insights must blend press-release content with "
                        "complaint-pattern analytics while staying concise."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "summary_text": f"Fraud insights error: {e}",
            "visual_instructions": {"charts": []},
        }
