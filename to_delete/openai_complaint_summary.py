"""
openai_complaint_summary.py
---------------------------------
Generates intelligent, query-driven summaries and visualization guidance
from CFPB consumer complaint data, with semantic reasoning tied to user intent.
"""

import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_complaint_data(stats_dict: dict, user_query: str, date_range: tuple = None):
    """
    Summarize CFPB complaint data in context of a semantic query.

    Args:
        stats_dict (dict): Aggregated or filtered complaint data.
        user_query (str): The user's search or analytical question.
        date_range (tuple): (start_date, end_date)

    Returns:
        dict: {"summary_text": str, "visual_instructions": dict}
    """
    start, end = date_range if date_range else ("N/A", "N/A")

    # --- Build prompt ---
    user_prompt = f"""
You are a senior fraud analytics strategist preparing a data-driven briefing for an executive audience.

User query: "{user_query}"

Structured CFPB complaint data ({start} to {end}):
{json.dumps(stats_dict, indent=2)}

INSTRUCTIONS:
- Write a 2–3 paragraph *narrative analysis* focused on interpreting trends and answering the user's question.
- Identify patterns, correlations, or anomalies across complaint categories, states, or time periods.
- Explain possible causes (e.g., regulation changes, cyber incidents, economic shifts).
- Highlight any emerging or declining fraud types and their consumer or institutional impact.
- Avoid restating raw totals or obvious facts.

Then, suggest 2–3 specific data visuals **directly tied** to your analysis (e.g., “line chart showing surge in identity theft after 2022”).
Choose only from: bar, stacked_bar, line, area, donut, heatmap, map.

Return ONLY a valid JSON object with this schema:
{{
  "summary_text": "Analytical narrative here...",
  "visual_instructions": {{
    "charts": [
      {{
        "type": "heatmap",
        "x": "state",
        "y": "fraud_category",
        "value": "complaint_density",
        "title": "Fraud Intensity by State and Category"
      }},
      {{
        "type": "line",
        "x": "date_received",
        "y": "complaint_count",
        "color": "product",
        "title": "Trend of Complaints Over Time"
      }}
    ]
  }}
}}
    """

    # --- API call ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert financial analyst specializing in consumer fraud trends. "
                        "You interpret structured complaint data to reveal cause–effect patterns, "
                        "not just surface-level summaries. Focus on correlations, risk shifts, "
                        "and meaningful insights for leadership briefings. Always output valid JSON."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.45,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )

        raw_output = response.choices[0].message.content.strip()

        parsed = json.loads(raw_output)
        if "summary_text" not in parsed:
            parsed = {"summary_text": raw_output, "visual_instructions": {"charts": []}}

        return parsed

    except json.JSONDecodeError:
        # Fallback if the model outputs non-JSON text
        return {
            "summary_text": raw_output if 'raw_output' in locals() else "Model returned invalid JSON.",
            "visual_instructions": {"charts": []},
        }
    except Exception as e:
        return {
            "summary_text": f"Complaint summary error: {e}",
            "visual_instructions": {"charts": []},
        }
