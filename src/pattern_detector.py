# src/AI/auto_visuals.py
"""
auto_visuals.py
---------------------------------
Automatically generates meaningful fraud-focused visualizations
from filtered press releases that match the user query.
"""

import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords", quiet=True)


def generate_auto_visuals(filtered_articles: pd.DataFrame, user_query: str):
    """
    Generate visual instructions from filtered articles
    that connect directly to the user's fraud-related question.
    """
    if filtered_articles.empty or "content" not in filtered_articles:
        return {
            "summary": "No article content available for visualization.",
            "visual_instructions": {"charts": []}
        }

    df = filtered_articles.copy()
    df["content"] = df["content"].astype(str).str.lower()

    # -----------------------------
    # 1. Year extraction
    # -----------------------------
    years = []
    for _, row in df.iterrows():
        date_field = str(row.get("date", "")) or str(row.get("year", ""))
        found = re.findall(r"\b(20\d{2}|19\d{2})\b", date_field)
        years.append(int(found[0]) if found else None)

    df["year"] = years
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # -----------------------------
    # 2. Trend detection for query keywords
    # -----------------------------
    query_terms = [t.strip() for t in user_query.lower().split() if len(t) > 3]
    query_pattern = "|".join(map(re.escape, query_terms))
    df["match"] = df["content"].apply(lambda t: len(re.findall(query_pattern, t)))

    yearly_trend = (
        df.groupby("year", as_index=False)["match"]
        .sum()
        .sort_values("year")
        .rename(columns={"match": "mentions"})
    )

    if yearly_trend["mentions"].sum() == 0:
        yearly_trend = (
            df.groupby("year", as_index=False)["content"]
            .count()
            .rename(columns={"content": "mentions"})
            .sort_values("year")
        )

    # -----------------------------
    # 3. Keyword + phrase extraction
    # -----------------------------
    text = " ".join(df["content"])
    stop = set(stopwords.words("english")).union({
        "federal", "reserve", "board", "press", "release", "bank",
        "financial", "system", "statement", "policy", "committee",
        "public", "meeting", "minutes", "update", "news", "report",
        "https", "gov", "page", "joint", "agency", "agencies", "statement"
    })

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text)
    filtered = [w for w in words if w not in stop]
    keyword_counts = dict(Counter(filtered).most_common(20))

    vec = CountVectorizer(stop_words=list(stop), ngram_range=(2, 4), max_features=300)
    X = vec.fit_transform([text])
    phrase_freq = list(zip(vec.get_feature_names_out(), X.toarray()[0]))
    top_phrases = sorted(phrase_freq, key=lambda x: x[1], reverse=True)[:10]

    # -----------------------------
    # 4. Structured visual plan
    # -----------------------------
    visuals = [
        {
            "type": "line",
            "x": "year",
            "y": "mentions",
            "title": f"Trend of '{user_query}' Mentions Over Time"
        },
        {
            "type": "bar",
            "x": "keyword",
            "y": "count",
            "title": "Top Keywords in Fraud-Related Articles"
        },
        {
            "type": "donut",
            "x": "phrase",
            "y": "count",
            "title": "Most Common Fraud-Related Phrases"
        }
    ]

    return {
        "summary": f"Auto-detected trends and keyword patterns related to '{user_query}'.",
        "visual_instructions": {"charts": visuals},
        "data": {
            "yearly_trend": yearly_trend,
            "keyword_counts": keyword_counts,
            "top_phrases": top_phrases
        }
    }
