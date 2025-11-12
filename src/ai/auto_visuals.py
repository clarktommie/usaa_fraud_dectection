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


def generate_auto_visuals(filtered_articles: pd.DataFrame, user_query: str):
    """Generate clean, query-driven visuals from filtered press releases."""
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
    def extract_year(val):
        match = re.search(r"(19|20)\d{2}", str(val))
        return int(match.group(0)) if match else None

    df["year"] = df["date"].apply(extract_year) if "date" in df else None
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # -----------------------------
    # 2. Trend detection for query keywords
    # -----------------------------
    query_terms = [t.strip() for t in user_query.lower().split() if len(t) > 3]
    pattern = "|".join(map(re.escape, query_terms)) if query_terms else "fraud"
    df["match"] = df["content"].apply(lambda t: len(re.findall(pattern, t)))

    yearly_trend = (
        df.groupby("year", as_index=False)["match"]
        .sum()
        .rename(columns={"match": "mentions"})
        .sort_values("year")
    )

    # -----------------------------
    # 3. Keyword + phrase extraction
    # -----------------------------
    text = " ".join(df["content"])

    # Clean text
    text = re.sub(r"http\S+|www\S+|[^a-z\s]", " ", text)
    words = [w for w in text.split() if len(w) > 3]

    # Custom stopword list (expanded and tuned for financial/regulatory text)
    stopwords = {
        "the", "and", "for", "with", "that", "from", "this", "have", "will", "are",
        "not", "was", "were", "been", "their", "they", "them", "into", "which",
        "about", "also", "more", "such", "other", "than", "like", "only", "each",
        "over", "under", "because", "through", "being", "while", "between",
        "federal", "reserve", "board", "press", "release", "bank", "financial",
        "statement", "policy", "committee", "public", "meeting", "minutes",
        "update", "report", "system", "page", "joint", "agency", "agencies",
        "office", "contact", "information", "learn", "consumer", "protection",
        "bureau", "cfpb", "newsroom", "subscribe", "email", "today", "issued",
        "available", "announced", "including", "within", "may", "june", "july",
        "august", "september", "october", "november", "december"
    }

    filtered_words = [w for w in words if w not in stopwords]
    keyword_counts = dict(Counter(filtered_words).most_common(20))

    # Phrases (2–4 words)
    vec = CountVectorizer(stop_words=list(stopwords), ngram_range=(2, 4), max_features=200)
    X = vec.fit_transform([text])
    phrase_freq = list(zip(vec.get_feature_names_out(), X.toarray()[0]))
    top_phrases = sorted(phrase_freq, key=lambda x: x[1], reverse=True)[:10]

    # -----------------------------
    # 4. Structured output for Streamlit
    # -----------------------------
    visuals = [
        {"type": "line", "x": "year", "y": "mentions",
         "title": f"Trend of '{user_query}' Mentions Over Time"},
        {"type": "bar", "x": "keyword", "y": "count",
         "title": "Top Keywords in Relevant Articles"},
        {"type": "donut", "x": "phrase", "y": "count",
         "title": "Most Common Multiword Phrases"}
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
