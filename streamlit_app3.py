"""
USAA Fraud Research Dashboard
---------------------------------
Analyzes financial compliance and fraud-related content from Federal Reserve
and CFPB press releases. Combines semantic storytelling, domain tagging,
and OpenAI reasoning to provide insight into fraud, compliance, and risk trends.
"""

import os
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# -------------------------------
# Imports from your project
# -------------------------------
from src.data_loader import fetch_articles
from src.sidebar_controls import sidebar_controls
from src.usaa_logo import display_usaa_logo
from src.ai.fraud_insights import generate_fraud_insights
from src.ai.article_preprocessing import prepare_articles_for_ai
from src.cfpb_loader import fetch_complaints
from src.pattern_detector import detect_patterns
from src.library_viewer import render_library_viewer
from src.topic_dashboard import render_topic_dashboard
from src.ai.auto_visuals import generate_auto_visuals
from src.semantic_fraud_filter import semantic_filter  # ✅ using your existing semantic layer

# -----------------
# Setup
# -----------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------
# Initialization
# -----------------
st.set_page_config(page_title="USAA Semantic Search", layout="wide")
display_usaa_logo()
st.title("Financial Compliance Insight System")

preset, year, keyword, insight_choice = sidebar_controls()
default_query = "" if preset == "— none —" else preset
st.info("Enter a query (or choose a preset in the sidebar), then click **Search**.")

# -----------------
# Sidebar: Topic Dashboard Button
# -----------------
st.sidebar.markdown("### Analysis Modules")
if st.sidebar.button("📘 Topic Dashboard"):
    st.session_state["show_topic_dashboard"] = True
if st.session_state.get("show_topic_dashboard"):
    render_topic_dashboard()

# -----------------
# Load Complaints
# -----------------
complaints_df = fetch_complaints()
if complaints_df.empty:
    st.warning("⚠️ No complaints found in Supabase.")
else:
    st.success(f"✅ Loaded {len(complaints_df)} classified CFPB complaints.")

# -----------------
# SEMANTIC + INSIGHT ENGINE
# -----------------
st.markdown("### ⚙️ Semantic Exploration & Insight Engine")

query_sentence = st.text_input(
    "Enter a question or sentence to explore:",
    value=default_query,
    placeholder="e.g., How are banks addressing AML and third-party risks?"
)
run_semantic = st.button("Run Semantic Search")

if run_semantic and query_sentence.strip():
    # -----------------------------
    # Step 1: Semantic Retrieval (live from Supabase)
    # -----------------------------
    with st.spinner(f"🔎 Running semantic search for '{query_sentence}'..."):
        df = semantic_filter(query_sentence, top_n=50)

    if df.empty:
        st.warning("⚠️ No semantically relevant articles found.")
        st.stop()

    st.success(f"✅ Retrieved {len(df)} semantically relevant press releases.")

    # -----------------------------
    # Step 2: Apply optional filters
    # -----------------------------
    if year.strip():
        try:
            y = int(year.strip())
            df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
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
        st.warning("⚠️ No results after filtering.")
        st.stop()

    # -----------------------------
    # Step 3: Run AI Insight Generation
    # -----------------------------
    with st.spinner(f"Analyzing {len(df)} articles with AI reasoning..."):
        clean_articles = prepare_articles_for_ai(df.to_dict(orient="records"))
        complaints_data = complaints_df.to_dict(orient="records")

        result = generate_fraud_insights(
            articles=clean_articles,
            complaints=complaints_data,
            user_query=query_sentence,
            date_range=("N/A", "N/A")
        )

        auto = detect_patterns(df, focus_term="fraud")

    # -----------------------------
    # Step 4: Display Unified AI Summary
    # -----------------------------
    st.divider()
    st.markdown("### 🧠 AI-Generated Unified Fraud Insight")
    st.write(result.get("summary_text", "No insight generated."))
    st.divider()

    # -----------------------------
    # Step 5: Article-Based Yearly Trend
    # -----------------------------
    st.markdown("### 📊 Fraud Mentions Over Time (Articles)")
    trend = auto.get("yearly_trend", pd.DataFrame())

    if not trend.empty:
        trend.columns = [c.strip() for c in trend.columns]
        if "Year" not in trend.columns:
            trend["Year"] = range(1, len(trend) + 1)

        trend["Year"] = pd.to_numeric(trend["Year"], errors="coerce").fillna(0).astype(int)
        y_cols = [c for c in trend.columns if c.lower() != "year"]
        y_col = y_cols[0] if y_cols else "Mentions"

        trend = trend.dropna(subset=["Year"])
        trend = trend[trend["Year"] > 0]

        chart = (
            alt.Chart(trend)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(format="d")),
                y=alt.Y(f"{y_col}:Q", title="Mentions"),
                tooltip=["Year", y_col],
            )
            .properties(title="Fraud Mentions in Articles by Year", height=360)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No yearly trend data available.")

    # -----------------------------
    # Step 6: Auto-Generated Visuals (AI Suggested)
    # -----------------------------
    st.markdown("### 🤖 Auto-Generated Visualizations (Articles)")

    try:
        auto_viz = generate_auto_visuals(df, query_sentence)
        st.write(auto_viz["summary"])

        # Trend
        if "data" in auto_viz and "yearly_trend" in auto_viz["data"]:
            trend_df = auto_viz["data"]["yearly_trend"]
            if not trend_df.empty:
                st.markdown("#### 📈 Query Keyword Trend Over Time")
                chart = (
                    alt.Chart(trend_df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("year:O", title="Year", axis=alt.Axis(format="d")),
                        y=alt.Y("mentions:Q", title="Mentions"),
                        tooltip=["year", "mentions"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        # Keywords
        if "data" in auto_viz and "keyword_counts" in auto_viz["data"]:
            keyword_data = pd.DataFrame(
                list(auto_viz["data"]["keyword_counts"].items()),
                columns=["keyword", "count"]
            )
            if not keyword_data.empty:
                st.markdown("#### 🔑 Top Keywords in Relevant Articles")
                chart = (
                    alt.Chart(keyword_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("keyword:N", sort="-y", title="Keyword"),
                        y=alt.Y("count:Q", title="Frequency"),
                        tooltip=["keyword", "count"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

        # Phrases
        if "data" in auto_viz and "top_phrases" in auto_viz["data"]:
            phrase_data = pd.DataFrame(
                auto_viz["data"]["top_phrases"],
                columns=["phrase", "count"]
            )
            if not phrase_data.empty:
                st.markdown("#### 💬 Most Common Phrases")
                chart = (
                    alt.Chart(phrase_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("phrase:N", sort="-y", title="Phrase"),
                        y=alt.Y("count:Q", title="Frequency"),
                        tooltip=["phrase", "count"],
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.warning(f"⚠️ Visualization error: {e}")

else:
    st.info("Enter a query and click **Run Semantic Search** to begin.")

# -----------------
# Library Viewer Integration
# -----------------
if st.sidebar.button("📚 Open Library Viewer"):
    st.session_state["view_library"] = True
if st.session_state.get("view_library"):
    render_library_viewer()
