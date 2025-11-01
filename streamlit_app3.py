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
from src.semantic_storytelling import run_semantic_storytelling
from src.openai_summary import summarize_text
from src.library_viewer import render_library_viewer
from src.usaa_logo import display_usaa_logo




# -----------------
# Initialization
# -----------------
st.set_page_config(page_title="USAA Semantic Search", layout="wide")
display_usaa_logo()
st.title("Financial Compliance Insight System")

# Sidebar controls
preset, year, keyword, insight_choice = sidebar_controls()
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

# Auto-fill query from sidebar preset
query_sentence = st.text_input(
    "Enter a question or sentence to explore:",
    value=default_query,
    placeholder="e.g., How are banks addressing AML and third-party risks?"
)

run_semantic = st.button("Run Semantic Search")

if run_semantic and query_sentence.strip():
    # --- Filter articles before semantic search ---
    df = all_articles.copy()

    if year.strip():
        try:
            y = int(year.strip())
            df = df[df["year"] == y]
        except ValueError:
            st.warning("Year must be numeric (e.g., 2024). Ignoring year filter.")

    if keyword.strip():
        kw = keyword.lower()
        df = df[
            df["title"].astype(str).str.lower().str.contains(kw, na=False)
            | df["content"].astype(str).str.lower().str.contains(kw, na=False)
        ]

    if df.empty:
        st.warning("No articles match the selected filters.")
    else:
        with st.spinner(f"Running semantic analysis for: '{query_sentence}'"):
            # Step 1: Semantic + pattern detection (returns articles + charts)
            patterns, top_articles = run_semantic_storytelling(df, query_sentence)

            # Step 2: Summarize patterns using OpenAI (from actual article text)
            if isinstance(patterns, dict) and any(patterns.values()):
                st.divider()
                st.markdown("### 🧠 AI-Generated Summary (OpenAI)")

                # --- Extract top article text for richer summary ---
                top_text = " ".join(
                    top_articles["content"].fillna("").tolist()[:5]
                )[:8000]  # limit for token safety

                keyword_text = " ".join(list(patterns.get("keyword_counts", {}).keys()))
                phrase_text = " ".join([p for p, _ in patterns.get("top_phrases", [])])

                trend_data = patterns.get("yearly_trend")
                if trend_data is not None and not trend_data.empty:
                    recent_trends = trend_data.tail(3).to_dict(orient="records")
                    trend_summary = f"Recent yearly trends in mentions: {recent_trends}."
                else:
                    trend_summary = "No clear yearly trend data detected."

                focus_map = {
                    "Trends and patterns over time": "Focus on trend shifts and anomalies.",
                    "Emerging risk areas": "Highlight newly emerging fraud risks or tactics.",
                    "Policy and regulatory tone": "Emphasize regulatory tone and enforcement sentiment.",
                    "Consumer or institutional impact": "Focus on who is affected and operational implications.",
                }
                focus_note = focus_map.get(insight_choice, "")

                # --- Combine everything for OpenAI ---
                summary_input = (
                    f"User question: {query_sentence}\n"
                    f"{focus_note}\n\n"
                    f"--- Article Excerpts ---\n{top_text}\n\n"
                    f"--- Keywords ---\n{keyword_text}\n\n"
                    f"--- Common Phrases ---\n{phrase_text}\n\n"
                    f"--- Trend Summary ---\n{trend_summary}\n\n"
                    f"Provide a cohesive, paragraph-style analytical summary (no bullet points)."
                )

                summary_text = summarize_text(summary_input)
                st.write(summary_text)
            else:
                st.warning("No meaningful patterns detected for summarization.")
else:
    st.info("Enter a question or choose a preset, then click **Run Semantic Search**.")

# -----------------
# Library Viewer Integration
# -----------------
if st.sidebar.button("📚 Open Library Viewer"):
    st.session_state["view_library"] = True

if st.session_state.get("view_library"):
    render_library_viewer()
