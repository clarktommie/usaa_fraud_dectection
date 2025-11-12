"""
semantic_storytelling.py
---------------------------------
Performs sentence-level semantic search using OpenAI embeddings,
displays the most relevant articles, and visualizes detected fraud/
compliance patterns such as top keywords, phrases, and yearly trends.
"""

import os
import re
import pandas as pd
import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from src.pattern_detector import detect_patterns
import nltk

nltk.download("stopwords", quiet=True)

# -----------------
# Domain keywords for focus extraction
# -----------------
DOMAIN_KEYWORDS = {
    "fraud", "aml", "money", "laundering", "cyber", "identity", "sanctions",
    "risk", "scam", "bribery", "enforcement", "compliance", "bsa", "fincen",
    "payments", "scheme", "audit", "reporting", "bank", "consumer"
}

# -----------------
# OpenAI Client Setup
# -----------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embeddings(texts, model="text-embedding-3-large"):
    """Generate embeddings using OpenAI's embedding API."""
    response = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in response.data]


# -----------------
# Focus Word Extraction
# -----------------
def detect_focus_word(query_sentence: str) -> str:
    """Extracts the most domain-relevant focus word from the user query."""
    tokens = re.findall(r"[A-Za-z]+", query_sentence.lower())
    for token in tokens:
        if token in DOMAIN_KEYWORDS:
            return token
    if tokens:
        return max(tokens, key=len)
    return "fraud"


# -----------------
# Core Function
# -----------------
def run_semantic_storytelling(all_articles: pd.DataFrame, query_sentence: str):
    """Run semantic search, focus detection, visualization, and trend analysis."""
    if all_articles.empty:
        st.warning("No article data available.")
        return {}

    st.info("🔍 Using OpenAI embeddings for semantic matching...")

    # -----------------
    # Step 1: Prepare corpus for embedding
    # -----------------
    all_articles["combined"] = (
        all_articles["title"].fillna("") + ". " + all_articles["content"].fillna("")
    )

    # -----------------
    # Step 2: Compute embeddings via OpenAI
    # -----------------
    try:
        corpus_embeddings = get_openai_embeddings(all_articles["combined"].tolist())
        query_embedding = get_openai_embeddings([query_sentence])[0]
    except Exception as e:
        st.error(f"⚠️ OpenAI embedding error: {e}")
        return {}

    # -----------------
    # Step 3: Compute cosine similarity
    # -----------------
    import numpy as np
    corpus_embeddings = np.array(corpus_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Retrieve top 200 most relevant
    top_idx = similarities.argsort()[::-1][:200]
    top_articles = all_articles.iloc[top_idx]

    # -----------------
    # Step 4: Display top matches
    # -----------------
    st.subheader("📄 Top Semantically Related Articles")
    for idx in top_idx[:10]:  # Display top 10 only
        st.write(f"**{all_articles.iloc[idx]['title']}**")
        st.caption(f"Similarity: {similarities[idx]:.3f}")
        st.write(all_articles.iloc[idx]["content"][:400] + "…")
        st.divider()

    # -----------------
    # Step 5: Detect focus term from query
    # -----------------
    focus_word = detect_focus_word(query_sentence)
    st.markdown(f"### 🎯 Focus Word: **{focus_word}**")

    # -----------------
    # Step 6: Detect keyword, phrase, and yearly trends
    # -----------------
    patterns = detect_patterns(top_articles, focus_word)

    # --- Visualization ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Keywords")
        if patterns.get("keyword_counts"):
            st.bar_chart(pd.Series(patterns["keyword_counts"]))
        else:
            st.info("No significant keywords found.")

    with col2:
        st.subheader("Common Phrases")
        if patterns.get("top_phrases"):
            st.dataframe(pd.DataFrame(patterns["top_phrases"], columns=["Phrase", "Mentions"]))
        else:
            st.info("No frequent phrases found.")

    # -----------------
    # Step 7: Yearly Trend Visualization
    # -----------------
    trend_data = patterns.get("yearly_trend")
    if trend_data is not None and not trend_data.empty:
        trend_data = trend_data.rename(columns={
            "Year": "year",
            "Mentions": "mentions",
            f"Mentions of '{focus_word}'": "mentions"
        })

        if "year" in trend_data.columns and "mentions" in trend_data.columns:
            trend_data = trend_data.set_index("year")
            st.markdown(f"### 📊 Yearly Trend for **'{focus_word}'** Mentions")
            st.line_chart(trend_data["mentions"])
        else:
            st.warning("⚠️ Trend data found but missing required columns.")
    else:
        st.info(f"No yearly trend data found for '{focus_word}'.")

    # Return detected patterns for further analysis (e.g., OpenAI summary)
    return patterns
