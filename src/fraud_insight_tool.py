"""
fraud_insight_tool.py
-------------------------------------------------------
OpenAI Tool for narrative-driven fraud insights
based solely on article and press release content.

Produces:
- Analytical executive summary
- Context-aware, varied visualization recommendations
"""

import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple
from openai import OpenAI
from difflib import SequenceMatcher
import random

# ----------------------------
# Setup
# ----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================
# 1. Article Analysis
# ============================================================
def analyze_articles(df: pd.DataFrame, user_query: str, max_articles: int = 30) -> Dict[str, Any]:
    """Extract relevant article summaries for reasoning."""
    if df.empty:
        return {"error": "No article data provided."}

    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date", ascending=False)

    def similarity(a, b):
        return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

    # Relevance scoring
    df["relevance"] = df["title"].apply(lambda t: similarity(t, user_query))
    relevant = df.sort_values("relevance", ascending=False).head(max_articles)

    summaries = []
    for _, row in relevant.iterrows():
        title = row.get("title", "Untitled")
        content = str(row.get("content", ""))[:700]
        snippet = " ".join(content.split()[:120])
        date = str(row.get("date", ""))[:10]
        src = row.get("source", "Unknown Source")
        summaries.append(f"[{date}] {src}: {title} — {snippet}")

    topics = (
        pd.Series(" ".join(relevant["title"].fillna("")).lower().split())
        .value_counts()
        .head(15)
        .to_dict()
    )

    return {
        "total_articles": len(df),
        "relevant_articles": len(relevant),
        "topics_detected": topics,
        "summaries": summaries,
    }


# ============================================================
# 2. Integrated Reasoning & Visualization
# ============================================================
def generate_article_summary(
    article_insights: dict,
    user_query: str,
    date_range: Tuple[str, str] = None,
    model: str = "gpt-4o",
) -> Dict[str, Any]:
    """Use OpenAI reasoning to interpret article-based fraud and compliance narratives."""
    start, end = date_range if date_range else ("N/A", "N/A")

    # Variety pool of chart templates (no counts!)
    chart_templates = [
        {
            "type": "line",
            "x": "date",
            "y": "topic_strength",
            "title": "Shifts in Risk Emphasis Over Time",
        },
        {
            "type": "bar",
            "x": "source",
            "y": "risk_intensity",
            "title": "Fraud and Compliance Themes by Source",
        },
        {
            "type": "heatmap",
            "x": "date",
            "y": "theme",
            "value": "attention_level",
            "title": "Emerging Topics by Date and Theme",
        },
        {
            "type": "donut",
            "x": "theme",
            "y": "share",
            "title": "Distribution of Fraud-Related Focus Areas",
        },
        {
            "type": "stacked_bar",
            "x": "date",
            "y": "theme_strength",
            "color": "category",
            "title": "Evolving Regulatory Themes Across Time",
        },
        {
            "type": "map",
            "x": "region",
            "y": "risk_level",
            "title": "Regional Emphasis on Fraud and Policy Actions",
        },
    ]
    random.shuffle(chart_templates)
    visual_templates = chart_templates[:3]

    prompt = f"""
You are a senior fraud intelligence strategist analyzing *only* press releases and policy articles.

User query: "{user_query}"
Time range: {start} → {end}

=== Extracted Article Insights ===
{json.dumps(article_insights, indent=2)}

TASK:
- Write a 2–3 paragraph analytical narrative describing what the articles reveal about fraud, compliance, or risk themes.
- Discuss emerging policies, tone changes, or notable focus shifts.
- Base your reasoning entirely on article content — do NOT mention complaint counts or volume.
- Then, choose 2–3 visualizations that best illustrate your reasoning.
  (e.g., trends over time, focus by source, concentration of risk themes, policy emphasis, etc.)

Valid chart types: ["bar", "stacked_bar", "line", "area", "donut", "heatmap", "map"]

Return strictly valid JSON:
{{
  "summary_text": "Cohesive analytical narrative here...",
  "visual_instructions": {{
    "charts": {json.dumps(visual_templates, indent=2)}
  }}
}}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior financial intelligence analyst. "
                        "You interpret article content to extract risk narratives and strategy-level insights. "
                        "Never reference complaint data or counts. Always output valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.65,  # encourage variation
            max_tokens=1800,
            response_format={"type": "json_object"},
        )

        output = response.choices[0].message.content.strip()
        parsed = json.loads(output)

        # Sanitize chart definitions for Streamlit compatibility
        cleaned = []
        for chart in parsed.get("visual_instructions", {}).get("charts", []):
            valid_chart = {
                k: (v if isinstance(v, (str, int, float)) else str(v))
                for k, v in chart.items()
            }
            cleaned.append(valid_chart)
        parsed["visual_instructions"]["charts"] = cleaned

        return parsed

    except Exception as e:
        return {
            "summary_text": f"Error generating article insight: {e}",
            "visual_instructions": {"charts": []},
        }


# ============================================================
# 3. TOOL WRAPPER
# ============================================================
def fraud_insight_tool(
    complaints_df: pd.DataFrame,  # kept for API signature
    articles_df: pd.DataFrame,
    user_query: str,
    date_range: Tuple[str, str] = None,
) -> Dict[str, Any]:
    """Main callable tool — analyzes articles only."""
    article_insights = analyze_articles(articles_df, user_query)
    return generate_article_summary(article_insights, user_query, date_range)


# ============================================================
# 4. OPENAI TOOL REGISTRATION
# ============================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "fraud_insight_tool",
            "description": (
                "Analyze regulatory and financial press releases to uncover fraud, compliance, "
                "and risk communication patterns. Generates summaries and visualization suggestions "
                "without using complaint data or counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "Analytical question about fraud, compliance, or regulatory tone.",
                    },
                    "date_range": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional [start_date, end_date] for article analysis.",
                    },
                },
                "required": ["user_query"],
            },
        },
    }
]


# ============================================================
# 5. TEST HARNESS
# ============================================================
if __name__ == "__main__":
    articles_data = {
        "date": pd.date_range("2024-01-01", periods=10, freq="M"),
        "source": [
            "CFPB", "Federal Reserve", "BPI", "CFPB", "Federal Reserve",
            "BPI", "CFPB", "Federal Reserve", "CFPB", "BPI",
        ],
        "title": [
            "CFPB Warns of Synthetic Identity Fraud Surge",
            "Federal Reserve Updates AML Oversight Guidance",
            "BPI Responds to Increased Enforcement Actions",
            "CFPB Targets Misleading Digital Lending Ads",
            "Federal Reserve Expands Cyber Risk Framework",
            "BPI Calls for Regulatory Coordination on AI",
            "CFPB Highlights Fraud Awareness Campaigns",
            "Federal Reserve Introduces Operational Resilience Standards",
            "CFPB Expands Focus on Financial Data Privacy",
            "BPI Comments on AML Reporting Requirements",
        ],
        "content": [
            "CFPB warns banks of increasing cases of synthetic identity fraud impacting consumer credit...",
            "The Federal Reserve issued updated AML guidance aimed at modernizing oversight...",
            "The Bank Policy Institute addressed rising enforcement actions from federal agencies...",
            "CFPB launched new rules targeting deceptive digital lending advertisements...",
            "The Federal Reserve expanded its cybersecurity risk management framework...",
            "BPI called for increased coordination between regulators on artificial intelligence oversight...",
            "CFPB launched national campaigns to increase consumer awareness about fraud...",
            "Federal Reserve emphasized resilience standards to counter evolving threats...",
            "CFPB highlighted data privacy and protection as a top 2024 priority...",
            "BPI commented on new AML reporting standards to improve transparency...",
        ],
    }

    df = pd.DataFrame(articles_data)
    query = "What fraud and AML communication patterns emerged in 2024?"
    result = fraud_insight_tool(pd.DataFrame(), df, query, ("2024-01-01", "2024-12-31"))
    print(json.dumps(result, indent=2))
