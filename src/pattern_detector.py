"""
pattern_detector.py
---------------------------------
Detects recurring fraud-related patterns from financial articles.
"""

import pandas as pd
import re
from collections import Counter
from datetime import datetime


def detect_patterns(df: pd.DataFrame, focus_term: str = "fraud"):
    """
    Analyze article content for recurring fraud-related patterns and trends.
    """
    if df.empty or "content" not in df:
        return {"yearly_trend": pd.DataFrame(), "common_phrases": [], "keyword_counts": {}}

    df = df.copy()
    df["content"] = df["content"].astype(str).str.lower()

    # Extract publication year
    df["year"] = df.get("year") or df.get("date", "").apply(
        lambda x: int(re.search(r"\b(20\d{2}|19\d{2})\b", str(x)).group(0))
        if re.search(r"\b(20\d{2}|19\d{2})\b", str(x))
        else datetime.now().year
    )

    # Keyword matching
    focus_mask = df["content"].str.contains(focus_term, na=False)
    focused = df[focus_mask]
    text = " ".join(focused["content"]) if not focused.empty else ""

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    keyword_counts = dict(Counter(words).most_common(20))

    # Phrase detection
    phrases = re.findall(r"\b\w+\s+\w+\b", text)
    common_phrases = [p for p, _ in Counter(phrases).most_common(15)]

    # Yearly frequency
    yearly_trend = (
        focused.groupby("year", as_index=False)
        .size()
        .rename(columns={"size": "mentions"})
        .sort_values("year")
    )

    return {
        "yearly_trend": yearly_trend,
        "common_phrases": common_phrases,
        "keyword_counts": keyword_counts,
    }
