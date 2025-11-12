"""
article_preprocessing.py
---------------------------------
Transforms raw article rows into AI-ready structured dictionaries.
This keeps fraud_insights.py clean and ensures each module
has one clear job.
"""

import re
from datetime import datetime


def clean_text(text: str) -> str:
    """
    Light cleaning only.
    Removes excessive whitespace, boilerplate line breaks, etc.
    Does NOT remove important signals (fraud terms, dates, names).
    """
    if not text:
        return ""

    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_articles_for_ai(articles: list) -> list:
    """
    Convert raw Supabase article rows into a consistent AI-ready format.

    Input each article should contain:
      id, title, content, source, date, embedding

    Output format:
    [
      {
        "id": "...",
        "title": "...",
        "source": "...",
        "date": "YYYY-MM-DD",
        "summary_input": "cleaned text (truncated)",
        "embedding": [...],
      }
    ]

    Notes:
    - We do NOT generate summaries here.
    - We do NOT generate fraud insights here.
    - This is ONLY a formatting + cleaning step.
    """

    processed = []

    for art in articles:
        text = clean_text(art.get("content", ""))
        date_val = art.get("date")

        # Standardize date
        if isinstance(date_val, str):
            try:
                date_val = date_val.split("T")[0]
            except Exception:
                date_val = None
        else:
            date_val = None

        processed.append({
            "id": art.get("id"),
            "title": art.get("title"),
            "source": art.get("source"),
            "date": date_val,
            "summary_input": text[:4000],  # limit what we feed the model
            "embedding": art.get("embedding"),
        })

    return processed
