"""
This Streamlit app provides an interactive interface for exploring and analyzing 
federal financial compliance and fraud-related press releases. It integrates 
semantic search powered by embeddings to identify relevant articles based on 
user-defined queries, filters, and similarity thresholds. The app also generates 
visual insights—such as fraud trend heatmaps, rolling trends, and word clouds—
and allows users to download summarized PDF reports.
"""

import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from transformers import pipeline

from src.visualizations import (
    plot_keyword_momentum,
    plot_trend_summary,
    plot_thematic_wordcloud,
    plot_keyword_network,
    plot_top_terms,
)
from src.data_loader import fetch_articles
from src.pdf_generator import generate_pdf
from src.embedding_search import run_search, get_embedder
from src.sidebar_controls import sidebar_controls
from src.search_section import search_section
from src.article_analysis import analyze_articles
from src.pattern_detector import detect_patterns

# -----------------
# Initialization
# -----------------
st.set_page_config(page_title="USAA Semantic Search", layout="wide")
st.title("Financial Compliance Insight System")

# Sidebar
preset, year, keyword, threshold, top_k = sidebar_controls()
default_query = "" if preset == "— none —" else preset
st.info("Enter a query (or choose a preset in the sidebar), then click **Search**.")

# Fetch all articles
all_articles = fetch_articles()

# -----------------
# Pattern Detection Section
# -----------------
st.markdown("### ⚠️ Emerging Fraud Patterns")
patterns = detect_patterns(all_articles)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Top Keywords")
    if patterns.get("keyword_counts"):
        st.bar_chart(pd.Series(patterns["keyword_counts"]))
    else:
        st.info("No keywords found in recent articles.")

with col2:
    st.subheader("Common Phrases")
    if patterns.get("top_phrases"):
        st.dataframe(pd.DataFrame(patterns["top_phrases"], columns=["Phrase", "Mentions"]))
    else:
        st.info("No frequent phrases found in recent text data.")

st.subheader("Activity by Hour (if available)")
if patterns.get("hourly_pattern"):
    st.bar_chart(pd.Series(patterns["hourly_pattern"]))
else:
    st.info("No hourly pattern detected.")

# -----------------
# NLM Summarization
# -----------------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_patterns(patterns):
    if not patterns.get("keyword_counts") or not patterns.get("top_phrases"):
        return "Not enough data available to summarize recent fraud patterns."
    text = f"Top keywords: {', '.join(patterns['keyword_counts'].keys())}. " \
           f"Frequent phrases include: {', '.join(p for p, _ in patterns['top_phrases'][:5])}."
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    return summary

st.markdown("**🧠 Summary Insight:**")
st.write(summarize_patterns(patterns))
st.divider()

# -----------------
# Embedding Model Init
# -----------------
embedder = get_embedder()

# -----------------
# Global Trends Section
# -----------------
st.markdown("### 🌐 Global Fraud Trends Overview")

try:
    fraud_keywords = ["fraud", "scam", "aml", "launder", "bribe", "cyber", "identity", "enforcement"]
    plot_keyword_momentum(all_articles, fraud_keywords)
    plot_trend_summary(all_articles)
    plot_top_terms(all_articles, top_n=15, focus_terms=fraud_keywords)
    plot_keyword_network(all_articles)
except Exception as e:
    st.warning(f"Could not load visualizations: {e}")

# -----------------
# Search Section
# -----------------
hits = search_section(default_query, year, keyword, top_k, threshold)

if hits:
    analysis = analyze_articles(hits)
    if analysis:
        st.divider()
        st.subheader("📊 Article Insights")
        st.metric("Total Articles Retrieved", analysis["total_articles"])
