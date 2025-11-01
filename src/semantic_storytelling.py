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
from src.openai_summary import summarize_text  # ✅ added OpenAI summarizer for focus detection
import nltk
import re

nltk.download("stopwords", quiet=True)

# -----------------
# Domain Keywords (EXACTLY as provided)
# -----------------
DOMAIN_KEYWORDS = {
    # --- Core Fraud & Compliance ---
    "fraud", "scam", "aml", "money", "laundering", "cyber", "identity",
    "sanctions", "risk", "bribery", "enforcement", "compliance", "bsa",
    "fincen", "reporting", "audit", "settlement", "penalty", "investigation",
    "oversight", "governance", "policy", "supervision", "regulation",

    # --- Financial & Banking ---
    "bank", "consumer", "payments", "transaction", "account", "credit",
    "debit", "loan", "wire", "funds", "transfer", "mortgage", "foreclosure",

    # --- Technology & AI ---
    "ai", "artificial", "machine", "learning", "automation", "algorithm",
    "model", "analytics", "system", "technology", "data", "monitoring",

    # --- Fraud & Scam Specific Terms (added from CFPB list) ---
    "elder", "exploitation", "elder financial exploitation",
    "foreclosure relief scam", "mortgage loan modification", "loan modification",
    "fraud alert", "fraud alerts", "fraud by fiduciaries", "fiduciary fraud",
    "identity theft", "impostor scam", "imposter scam", "imposter scams",
    "mail fraud", "phishing", "spoofing", "wire transfer fraud",
    "wire fraud", "money transfer fraud", "security freeze",
    "credit freeze", "grandparent scam", "caller id spoofing",
    "military fraud alert", "active duty alert", "financial exploitation",
    "fake charity", "fake government call", "investment scam",
    "romance scam", "debt relief scam", "lottery scam",
    "check scam", "check fraud", "online scam", "phone scam",
    "credit card fraud", "mortgage scam", "foreclosure scam",
    "identity scam", "phishing email", "fraudulent website",
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
# Local Focus Detection (backup if OpenAI fails)
# -----------------
def detect_focus_word(query_sentence: str) -> str:
    """Fallback local focus detection."""
    q = query_sentence.strip().lower()
    if not q:
        return "fraud"

    tokens = re.findall(r"[A-Za-z]+", q)

    # --- Match full multi-word phrases first ---
    for phrase in sorted([k for k in DOMAIN_KEYWORDS if " " in k], key=len, reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, q):
            return phrase

    # --- Single keyword fallback ---
    for token in tokens:
        if token in DOMAIN_KEYWORDS:
            return token

    if tokens:
        return max(tokens, key=len)

    return "fraud"


# -----------------
# NEW: OpenAI-Assisted Focus Detection
# -----------------
def determine_focus_with_openai(query_sentence: str) -> str:
    """Ask OpenAI which focus word best represents the query."""
    try:
        prompt = f"""
        The user asked: "{query_sentence}"

        Choose ONE domain keyword or phrase from the following list that best captures the main topic:
        {', '.join(sorted(DOMAIN_KEYWORDS))}

        Respond with ONLY the single keyword or phrase that best represents the query.
        """
        response = summarize_text(prompt).strip().lower()

        for k in DOMAIN_KEYWORDS:
            if k.lower() == response:
                return k
        return detect_focus_word(query_sentence)  # fallback
    except Exception as e:
        st.warning(f"⚠️ OpenAI focus detection failed: {e}")
        return detect_focus_word(query_sentence)


# -----------------
# Core Semantic Storytelling Function
# -----------------
def run_semantic_storytelling(all_articles: pd.DataFrame, query_sentence: str):
    """
    Run semantic search, display top results, detect focus word (OpenAI or fallback),
    and visualize patterns.
    Returns: (patterns: dict, top_articles: pd.DataFrame)
    """
    if all_articles.empty:
        st.warning("No article data available.")
        return {}, pd.DataFrame()

    model = load_semantic_model()

    # Combine title + content for embeddings
    all_articles = all_articles.copy()
    all_articles["combined"] = (
        all_articles["title"].fillna("") + ". " + all_articles["content"].fillna("")
    )

    # Compute embeddings
    corpus_embeddings = model.encode(
        all_articles["combined"].tolist(),
        normalize_embeddings=True
    )
    query_embedding = model.encode([query_sentence], normalize_embeddings=True)
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]

    # Retrieve up to 200 most relevant articles
    top_idx = similarities.argsort()[::-1][:200]
    top_articles = all_articles.iloc[top_idx].copy()
    top_articles["similarity"] = similarities[top_idx]

    # --- Display Top Articles ---
    st.subheader("🔍 Top Semantically Related Articles")
    for idx in top_idx[:3]:
        st.write(f"**{all_articles.iloc[idx]['title']}**")
        st.caption(f"Similarity: {similarities[idx]:.3f}")
        st.write(all_articles.iloc[idx]['content'][:400] + "…")
        st.divider()

    # --- Focus Word Detection (OpenAI first, local fallback) ---
    focus_word = determine_focus_with_openai(query_sentence)
    st.markdown(f"### 🎯 Focus Word: **{focus_word}**")

    # --- Pattern Detection ---
    patterns = detect_patterns(top_articles, focus_word)

    # --- Visualizations ---
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

    # --- Yearly Trend Visualization ---
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

    # ✅ Return both for downstream use (e.g., OpenAI summary)
    return patterns, top_articles
