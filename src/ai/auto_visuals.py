"""
auto_visuals.py
---------------------------------
Generates fraud-focused visual insights from Federal Reserve press releases.
Extracts fraud-related terms, identifies top trends, and prepares visualization data.
"""

import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_auto_visuals(filtered_articles: pd.DataFrame, user_query: str):
    """
    Extracts fraud-related intelligence from Federal Reserve press releases.
    Returns top 5 fraud terms and top 3 fraud trends for visualization.
    """

    if filtered_articles.empty or "content" not in filtered_articles:
        return {
            "summary": "No article content available for visualization.",
            "visual_instructions": {"charts": []}
        }

    df = filtered_articles.copy()
    df["content"] = df["content"].astype(str).str.lower()

    # ------------------------------------
    # 1. Fraud-related keyword space
    # ------------------------------------
    fraud_keywords = [
        "fraud", "scam", "money laundering", "bribery", "embezzlement",
        "sanctions", "compliance", "cybercrime", "identity theft",
        "enforcement", "misconduct", "audit", "oversight"
    ]

    # Precompile for efficiency
    compiled_pattern = re.compile("|".join(map(re.escape, fraud_keywords)))

    # ------------------------------------
    # 2. Fast fraud-related text extraction
    # ------------------------------------
    def extract_snippets_fast(text: str):
        matches = []
        for kw in fraud_keywords:
            if kw in text:
                matches.append(kw)
        return " ".join(matches)

    df["fraud_snippets"] = df["content"].apply(extract_snippets_fast)
    df = df[df["fraud_snippets"].str.strip().astype(bool)]

    if df.empty:
        return {
            "summary": f"No fraud-related content found for '{user_query}'.",
            "visual_instructions": {"charts": []}
        }

    # ------------------------------------
    # 3. TF-IDF on fraud-specific segments
    # ------------------------------------
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50)
    X = tfidf.fit_transform(df["fraud_snippets"])
    tfidf_df = pd.DataFrame({
        "term": tfidf.get_feature_names_out(),
        "score": X.sum(axis=0).A1
    }).sort_values("score", ascending=False)

    # Keep only relevant fraud-related terms
    tfidf_df = tfidf_df[tfidf_df["term"].str.contains(compiled_pattern, na=False)]
    top_terms = tfidf_df.head(5).reset_index(drop=True)

    # ------------------------------------
    # 4. Trend extraction (semantic grouping)
    # ------------------------------------
    trend_patterns = {
        "Cyber & Digital Fraud": r"cyber|ransomware|phish|data breach|hacking",
        "Financial Crime & AML": r"launder|sanction|transaction|suspicious|aml",
        "Corporate Misconduct": r"bribery|embezzlement|misconduct|governance",
        "Consumer & Identity Fraud": r"scam|identity|consumer|impersonation",
        "Internal Controls & Compliance": r"compliance|audit|oversight|reporting",
    }

    def detect_trend(text):
        for trend, pat in trend_patterns.items():
            if re.search(pat, text):
                return trend
        return None

    df["fraud_trend"] = df["fraud_snippets"].apply(detect_trend)
    fraud_trends = (
        df["fraud_trend"]
        .dropna()
        .value_counts()
        .head(3)
        .reset_index()
    )

    fraud_trends.columns = ["trend", "mentions"]

    # ------------------------------------
    # 5. Structured visual output
    # ------------------------------------
    visuals = [
        {
            "type": "bar",
            "x": "term",
            "y": "score",
            "title": "Top 5 Fraud-Related Terms in Federal Reserve Releases"
        },
        {
            "type": "bar",
            "x": "trend",
            "y": "mentions",
            "title": "Top 3 Emerging Fraud Trends"
        }
    ]

    return {
        "summary": (
            f"Identified {len(top_terms)} key fraud-related terms and "
            f"{len(fraud_trends)} dominant fraud trends from Federal Reserve releases."
        ),
        "visual_instructions": {"charts": visuals},
        "data": {
            "top_terms": top_terms,
            "fraud_trends": fraud_trends
        }
    }
