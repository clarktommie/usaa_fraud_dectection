"""
semantic_storytelling.py
---------------------------------
Handles sentence-level semantic search, article display,
pattern detection, and visualization for the USAA Fraud Research project.
Uses precomputed embeddings from Supabase ('press_releases_embed') and
joins with 'press_releases_clean' to retrieve full article text.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from src.pattern_detector import detect_patterns
from src.openai_summary import summarize_text
from supabase import create_client
from dotenv import load_dotenv
import nltk
import re

nltk.download("stopwords", quiet=True)

# -----------------
# Supabase client setup
# -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------
# Domain Keywords
# -----------------
DOMAIN_KEYWORDS = {
    "fraud_detection": {
        "label": "Fraud Detection",
        "context": (
            "Fraud detection involves identifying, preventing, and responding to deceptive or illegal financial activity. "
            "It includes scams, phishing, identity theft, wire fraud, and fraudulent transactions. "
            "Modern fraud detection also monitors impersonation scams, social engineering tactics, "
            "account takeovers, business email compromise, and synthetic identity fraud. "
            "The focus is on detecting suspicious behavior patterns and protecting both consumers and institutions from financial loss."
        )
    },

    "aml_compliance": {
        "label": "AML Compliance",
        "context": (
            "Anti-money laundering (AML) compliance refers to the processes and regulations that financial institutions follow "
            "to detect and prevent the movement of illicit funds. It includes adherence to BSA and FinCEN guidelines, "
            "performing Know Your Customer (KYC) checks, monitoring transactions for suspicious activity, "
            "reporting suspicious activity reports (SARs), and managing sanctions and risk exposure. "
            "AML programs are essential to reducing money laundering and terrorism financing risks."
        )
    },

    "financial_crime": {
        "label": "Financial Crime",
        "context": (
            "Financial crime encompasses a wide range of offenses such as bribery, corruption, embezzlement, "
            "fraud by fiduciaries, money laundering, insider trading, and investment scams. "
            "It often involves the exploitation of financial systems to gain illicit profits "
            "or conceal illegal activities through complex transactions and shell entities."
        )
    },

    "regulatory": {
        "label": "Regulatory Enforcement",
        "context": (
            "Regulatory enforcement focuses on how authorities and oversight bodies govern, supervise, and penalize organizations "
            "for misconduct or policy violations. It includes compliance audits, enforcement actions, settlements, "
            "penalties, and updates to governance and oversight policies. Regulators often release press statements "
            "or guidance documents describing how they address risks like AML deficiencies, consumer protection, "
            "and emerging fraud threats."
        )
    },

    "technology_context": {
        "label": "Technology in Fraud Prevention",
        "context": (
            "Technology plays a crucial role in combating financial misconduct. "
            "This includes artificial intelligence, machine learning, and analytics systems used for real-time fraud detection, "
            "pattern recognition, and anomaly detection. "
            "Automated transaction monitoring tools, natural language models for alert triage, and predictive algorithms "
            "are increasingly used to identify evolving threats such as phishing campaigns and digital impersonations."
        )
    },

    "cyber_fraud": {
        "label": "Cyber Fraud and Impersonation",
        "context": (
            "Cyber fraud involves digital deception, where attackers use online methods to steal sensitive data or assets. "
            "This includes phishing emails, fake websites, social media impersonation, malware-based credential theft, "
            "and ransomware extortion. Cyber fraud prevention focuses on authentication controls, user awareness, "
            "and cybersecurity integration with financial monitoring systems."
        )
    },

    "consumer_protection": {
        "label": "Consumer Protection and Awareness",
        "context": (
            "Consumer protection focuses on educating and defending individuals against deceptive financial practices. "
            "This includes awareness of online scams, fraudulent investment offers, fake charities, and impersonation calls. "
            "It emphasizes transparency, financial literacy, and proactive fraud alerts to help consumers recognize red flags."
        )
    }
}




# -----------------
# Focus Detection
# -----------------
def detect_focus_word(query_sentence: str) -> str:
    q = query_sentence.strip().lower()
    if not q:
        return "fraud"
    tokens = re.findall(r"[A-Za-z]+", q)
    for phrase in sorted([k for k in DOMAIN_KEYWORDS if " " in k], key=len, reverse=True):
        pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(pattern, q):
            return phrase
    for token in tokens:
        if token in DOMAIN_KEYWORDS:
            return token
    if tokens:
        return max(tokens, key=len)
    return "fraud"


def determine_focus_with_openai(query_sentence: str) -> str:
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
        return detect_focus_word(query_sentence)
    except Exception as e:
        st.warning(f"⚠️ OpenAI focus detection failed: {e}")
        return detect_focus_word(query_sentence)
    


# -----------------
# Core Semantic Storytelling Function
# -----------------
def run_semantic_storytelling(all_articles: pd.DataFrame, query_sentence: str):
    """
    Run semantic search using embeddings from Supabase (press_releases_embed),
    join with press_releases_clean to retrieve full text,
    and visualize patterns.
    Returns: (patterns: dict, top_articles: pd.DataFrame)
    """
    if all_articles.empty:
        st.warning("No article data available.")
        return {}, pd.DataFrame()

    if "embedding" not in all_articles.columns:
        st.error("⚠️ Missing 'embedding' column in DataFrame from Supabase.")
        return {}, pd.DataFrame()

    # -----------------
    # Step 1: Convert stored embeddings
    # -----------------
    st.info("🔍 Using stored Supabase embeddings for semantic similarity...")
    all_articles["embedding"] = all_articles["embedding"].apply(
        lambda e: np.array(e, dtype=float) if isinstance(e, list) else np.array([])
    )
    all_articles = all_articles[all_articles["embedding"].apply(lambda x: x.size > 0)].copy()

    if all_articles.empty:
        st.error("⚠️ No valid embeddings found in Supabase data.")
        return {}, pd.DataFrame()

    corpus_embeddings = np.vstack(all_articles["embedding"].values)

    # -----------------
    # Step 2: Query embedding (FastEmbed version to match Supabase)
    # -----------------
    from fastembed import TextEmbedding
    try:
        embedder = TextEmbedding("BAAI/bge-large-en-v1.5")
        query_embedding = np.array(list(embedder.embed([query_sentence])))
    except Exception as e:
        st.error(f"Error computing query embedding: {e}")
        return {}, pd.DataFrame()

    # -----------------
    # Step 3: Similarity computation
    # -----------------
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_idx = similarities.argsort()[::-1][:200]
    top_articles = all_articles.iloc[top_idx].copy()
    top_articles["similarity"] = similarities[top_idx]

    # -----------------
    # Step 4: Fetch full article text from clean table
    # -----------------
    st.info("📥 Fetching full article content for matched results...")
    top_ids = top_articles["id"].tolist()
    full_data = []
    try:
        if top_ids:
            response = (
                supabase.table("press_releases_clean")
                .select("id, title, content, date_standard, author, url")
                .in_("id", top_ids)
                .execute()
            )
            full_data = response.data or []
    except Exception as e:
        st.error(f"⚠️ Error fetching full-text data: {e}")

    full_df = pd.DataFrame(full_data)
    if not full_df.empty:
        top_articles = top_articles.merge(full_df, on="id", suffixes=("_embed", ""))
    else:
        st.warning("⚠️ No matching full-text records found in clean table.")

    # -----------------
    # Step 5: Display top matches
    # -----------------
    st.subheader("🔍 Top Semantically Related Articles")
    if "content" in top_articles.columns:
        for _, row in top_articles.head(3).iterrows():
            st.write(f"**{row['title']}**")
            st.caption(f"Similarity: {row['similarity']:.3f}")
            st.write((row.get("content") or "")[:400] + "…")
            st.divider()
    else:
        st.info("No content available to display for top matches.")

    # -----------------
    # Step 6: Focus Word Detection
    # -----------------
    focus_word = determine_focus_with_openai(query_sentence)
    st.markdown(f"### 🎯 Focus Word: **{focus_word}**")

    # -----------------
    # Step 7: Pattern Detection
    # -----------------
    patterns = detect_patterns(top_articles, focus_word)

    # -----------------
    # Step 8: Visualization
    # -----------------
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
    # Yearly Trend
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

    # ✅ Return both for downstream use
    return patterns, top_articles
