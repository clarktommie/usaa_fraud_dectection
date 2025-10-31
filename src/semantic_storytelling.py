"""
semantic_storytelling.py
---------------------------------
Handles sentence-level semantic search and pattern detection
for the USAA Fraud Research project.
"""

import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.pattern_detector import detect_patterns
import nltk

nltk.download("stopwords", quiet=True)


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
def run_semantic_storytelling(all_articles: pd.DataFrame, query_sentence: str):
    """
    Run semantic similarity search + pattern detection.
    Returns a dictionary with keyword counts, top phrases, and yearly trend data.
    """
    if all_articles.empty:
        st.warning("No article data available.")
        return {}

    # Load model
    model = load_semantic_model()

    # Combine title and content
    all_articles["combined"] = (
        all_articles["title"].fillna("") + ". " + all_articles["content"].fillna("")
    )

    # Compute embeddings
    corpus_embeddings = model.encode(
        all_articles["combined"].tolist(), normalize_embeddings=True
    )
    query_embedding = model.encode([query_sentence], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Retrieve top N matches
    top_idx = similarities.argsort()[::-1][:10]
    top_articles = all_articles.iloc[top_idx]

    # --- Display ---
    st.subheader("🔍 Top Semantically Related Articles")
    for idx in top_idx:
        st.write(f"**{all_articles.iloc[idx]['title']}**")
        st.caption(f"Similarity: {similarities[idx]:.3f}")
        st.write(all_articles.iloc[idx]["content"][:400] + "…")
        st.divider()

    # --- Pattern Detection ---
    patterns = detect_patterns(top_articles, query_sentence)

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
            st.dataframe(
                pd.DataFrame(patterns["top_phrases"], columns=["Phrase", "Mentions"])
            )
        else:
            st.info("No frequent phrases found.")

    # --- Yearly Trend ---
    trend_df = patterns.get("yearly_trend")
    if trend_df is not None and not trend_df.empty:
        st.subheader("Trend of Mentions Over Time")
        st.line_chart(trend_df.set_index("Year"))
    else:
        st.info("No yearly trend data available.")

    # ✅ Return patterns only (for use by OpenAI summary)
    return patterns
