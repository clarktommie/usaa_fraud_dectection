"""
USAA Fraud Research Dashboard
---------------------------------
Shows semantic relationships between financial compliance articles,
extracts patterns, and summarizes insights using OpenAI.
"""

import streamlit as st
import pandas as pd
from src.data_loader import fetch_articles
from src.sidebar_controls import sidebar_controls
from src.semantic_storytelling import run_semantic_storytelling  # ✅ brings back article list + charts
from src.openai_summary import summarize_text  # ✅ only one summary engine

# -----------------
# Initialization
# -----------------
st.set_page_config(page_title="USAA Semantic Search", layout="wide")
st.title("Financial Compliance Insight System")

# Sidebar controls
preset, year, keyword, threshold, top_k = sidebar_controls()
default_query = "" if preset == "— none —" else preset
st.info("Enter a query (or choose a preset in the sidebar), then click **Search**.")

# Fetch articles
all_articles = fetch_articles()

if all_articles.empty:
    st.warning("⚠️ No articles found in Supabase.")
else:
    st.success(f"✅ Loaded {len(all_articles)} articles from Supabase.")

# -----------------
# Semantic Storytelling + Charts + OpenAI Summary
# -----------------
st.markdown("### ⚠️ Semantic Exploration & Storytelling")

query_sentence = st.text_input(
    "Enter a question or sentence to explore:",
    placeholder="e.g., How are banks addressing AML and third-party risks?"
)

run_semantic = st.button("Run Semantic Search")

if run_semantic and query_sentence.strip():
    with st.spinner(f"Running semantic analysis for: '{query_sentence}'"):
        # Step 1: Run semantic search + patterns + charts
        patterns = run_semantic_storytelling(all_articles, query_sentence)

        # Step 2: Only run summary if patterns are valid
        if isinstance(patterns, dict) and any(patterns.values()):
            st.divider()
            st.markdown("### 🧠 AI-Generated Summary (OpenAI)")
            keyword_text = " ".join(list(patterns.get("keyword_counts", {}).keys()))
            phrase_text = " ".join([p for p, _ in patterns.get("top_phrases", [])])

            trend_data = patterns.get("yearly_trend")
            if trend_data is not None and not trend_data.empty:
                recent_trends = trend_data.tail(3).to_dict(orient="records")
                trend_summary = f"Recent yearly trends in mentions: {recent_trends}."
            else:
                trend_summary = "No clear yearly trend data detected."

            # Combine content for OpenAI summary
            summary_input = (
                f"Analyze the following detected fraud and compliance patterns related to '{query_sentence}'. "
                f"Keywords: {keyword_text}. Phrases: {phrase_text}. {trend_summary}"
            )

            summary_text = summarize_text(summary_input)
            st.write(summary_text)
        else:
            st.warning("No meaningful patterns detected for summarization.")
else:
    st.info("Enter a full sentence or question above, then click **Run Semantic Search**.")
