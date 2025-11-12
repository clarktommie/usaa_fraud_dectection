# streamlit_app3.py
"""
USAA Fraud Research Dashboard
---------------------------------
Analyzes financial compliance and fraud-related content from Federal Reserve
and CFPB press releases. Combines semantic storytelling, domain tagging,
and OpenAI reasoning to provide insight into fraud, compliance, and risk trends.
"""

import os
import json
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
from src.library_viewer import render_library_viewer
from src.usaa_logo import display_usaa_logo
from src.topic_dashboard import render_topic_dashboard
from src.AI.fraud_insights import generate_fraud_insights
from src.AI.article_preprocessing import prepare_articles_for_ai
from src.cfpb_loader import fetch_complaints
from src.pattern_detector import detect_patterns

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
# Fetch Articles
# -----------------
all_articles = fetch_articles()
if all_articles.empty:
    st.warning("⚠️ No articles found in Supabase.")
else:
    st.success(f"✅ Loaded {len(all_articles)} press releases and policy articles.")

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
    df = all_articles.copy()

    # ---- Filters ----
    if year.strip():
        try:
            y = int(year.strip())
            if "year" in df.columns:
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
        with st.spinner(f"Analyzing '{query_sentence}' across articles and complaints..."):

            # Step 1: Prepare articles for AI
            clean_articles = prepare_articles_for_ai(df.to_dict(orient="records"))

            # Step 2: Use real complaints dataset
            complaints_data = complaints_df.to_dict(orient="records")

            # Step 3: Generate AI insights
            result = generate_fraud_insights(
                articles=clean_articles,
                complaints=complaints_data,
                user_query=query_sentence,
                date_range=("N/A", "N/A")
            )

            # Step 4: Detect fraud patterns from articles
            auto = detect_patterns(df, focus_term="fraud")

        # -----------------------------
        # DISPLAY UNIFIED AI SUMMARY
        # -----------------------------
        st.divider()
        st.markdown("### 🧠 AI-Generated Unified Fraud Insight")
        st.write(result.get("summary_text", "No insight generated."))
        st.divider()

        # -----------------------------
        # ARTICLE-BASED YEARLY TREND
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
                    x=alt.X("Year:Q", title="Year"),
                    y=alt.Y(f"{y_col}:Q", title="Mentions"),
                    tooltip=["Year", y_col],
                )
                .properties(title="Fraud Mentions in Articles by Year", height=360)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No yearly trend data available.")

        # -----------------------------
        # AI-SUGGESTED VISUALIZATIONS
        # -----------------------------
        st.markdown("### 🤖 AI-Suggested Visualizations (Articles + Complaints)")
        visuals = result.get("visual_instructions", {}).get("charts", [])

        if not visuals:
            st.info("No AI-generated visual instructions available for this query.")
        else:
            for viz in visuals:
                try:
                    vtype = viz.get("type", "").lower()
                    title = viz.get("title", "Chart")
                    x = viz.get("x", "")
                    y = viz.get("y", "")
                    color = viz.get("color", None)

                    df_viz = complaints_df.copy()

                    if "date" in x.lower():
                        df_viz["date_received"] = pd.to_datetime(df_viz["date_received"], errors="coerce")
                        df_viz = (
                            df_viz.groupby(df_viz["date_received"].dt.to_period("M"))
                            .size()
                            .reset_index(name="complaint_count")
                        )
                        df_viz["date_received"] = df_viz["date_received"].astype(str)
                        x, y = "date_received", "complaint_count"

                    elif "state" in x.lower():
                        df_viz = df_viz.groupby("state").size().reset_index(name="complaint_count")
                        x, y = "state", "complaint_count"

                    elif "issue" in x.lower():
                        df_viz = df_viz.groupby("issue").size().reset_index(name="complaint_count")
                        x, y = "issue", "complaint_count"

                    elif "company" in x.lower():
                        df_viz = (
                            df_viz.groupby("company").size().reset_index(name="complaint_count")
                        ).sort_values("complaint_count", ascending=False).head(15)
                        x, y = "company", "complaint_count"

                    else:
                        df_viz = (
                            df_viz.groupby("domain_label").size().reset_index(name="complaint_count")
                        )
                        x, y = "domain_label", "complaint_count"

                    base = alt.Chart(df_viz).properties(title=title, height=380)

                    if vtype in ("bar", "stacked_bar"):
                        chart = base.mark_bar().encode(
                            x=alt.X(f"{x}:N", title=x),
                            y=alt.Y(f"{y}:Q", title=y),
                            color=alt.Color(color or x, title=color or x),
                            tooltip=[x, y],
                        )
                    elif vtype == "line":
                        chart = base.mark_line(point=True).encode(
                            x=alt.X(f"{x}:N", title=x),
                            y=alt.Y(f"{y}:Q", title=y),
                            tooltip=[x, y],
                        )
                    elif vtype == "heatmap":
                        chart = base.mark_rect().encode(
                            x=alt.X(f"{x}:N", title=x),
                            y=alt.Y(f"{y}:N", title=y),
                            color=alt.Color(f"{y}:Q", title="Intensity"),
                            tooltip=[x, y],
                        )
                    elif vtype == "donut":
                        donut_df = df_viz.copy()
                        chart = (
                            alt.Chart(donut_df)
                            .mark_arc(innerRadius=60)
                            .encode(
                                theta=alt.Theta(f"{y}:Q", stack=True),
                                color=alt.Color(f"{x}:N", title=x),
                                tooltip=[x, y],
                            )
                        )
                    else:
                        st.info(f"Chart type '{vtype}' not supported yet.")
                        continue

                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.warning(f"⚠️ Could not render '{viz.get('title', '')}': {e}")

else:
    st.info("Enter a query and click **Run Semantic Search** to begin.")

# -----------------
# Library Viewer Integration
# -----------------
if st.sidebar.button("📚 Open Library Viewer"):
    st.session_state["view_library"] = True
if st.session_state.get("view_library"):
    render_library_viewer()
