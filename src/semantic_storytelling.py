"""
semantic_storytelling.py
---------------------------------
Handles sentence-level semantic search, article display,
pattern detection, and visualization for the USAA Fraud Research project.
"""

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pattern_detector import detect_patterns
import nltk
import re

nltk.download("stopwords", quiet=True)

# Common fraud / compliance keywords for focus detection
DOMAIN_KEYWORDS = {
    "fraud", "aml", "money", "laundering", "cyber", "identity", "sanctions",
    "risk", "scam", "bribery", "enforcement", "compliance", "bsa", "finCEN",
    "payments", "scheme", "audit", "reporting", "bank", "consumer"
}


# -----------------
# Model Loader
# -----------------
@st.cache_resource
def load_semantic_model():
    """Load and cache the lightweight sentence transformer model."""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading semantic model: {e}")
        raise


# -----------------
# Core Function
# -----------------
def detect_focus_word(query_sentence: str) -> str:
    """Extract the most domain-relevant focus word from the user's query."""
    tokens = re.findall(r"[A-Za-z]+", query_sentence.lower())
    # Prefer first domain term found in the query
    for token in tokens:
        if token in DOMAIN_KEYWORDS:
            return token
    # Fallback: take most meaningful (longest) token
    if tokens:
        return max(tokens, key=len)
    return "fraud"  # safe default


def run_semantic_storytelling(all_articles: pd.DataFrame, query_sentence: str):
    """Run semantic search, display top results, detect focus word, and visualize patterns."""
    if all_articles.empty:
        st.warning("No article data available.")
        return {}

    model = load_semantic_model()

    # Combine title and content for embedding
    all_articles["combined"] = (
        all_articles["title"].fillna("") + ". " + all_articles["content"].fillna("")
    )

    # Compute embeddings
    corpus_embeddings = model.encode(
        all_articles["combined"].tolist(), normalize_embeddings=True
    )
    query_embedding = model.encode([query_sentence], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Retrieve up to 200 most relevant articles
    top_idx = similarities.argsort()[::-1][:200]
    top_articles = all_articles.iloc[top_idx]

    # --- Display Top Articles ---
    st.subheader("🔍 Top Semantically Related Articles")
    for idx in top_idx[:10]:  # Display top 10 only
        st.write(f"**{all_articles.iloc[idx]['title']}**")
        st.caption(f"Similarity: {similarities[idx]:.3f}")
        st.write(all_articles.iloc[idx]["content"][:400] + "…")
        st.divider()

    # --- Focus Word Detection ---
    focus_word = detect_focus_word(query_sentence)
    st.markdown(f"### 🎯 Focus Word: **{focus_word}**")

    # --- Pattern Detection based on Focus Word ---
    patterns = detect_patterns(top_articles, focus_word)

    # --- Top Keywords + Phrases ---
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

    # --- Yearly Trend for Focus Word ---
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

    return patterns
