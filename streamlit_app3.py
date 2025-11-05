"""
USAA Fraud Research Dashboard
---------------------------------
Analyzes financial compliance and fraud-related content from Federal Reserve
articles and CFPB consumer complaints. Combines semantic storytelling, 
domain tagging, and OpenAI summarization to provide insight into 
fraud, compliance, and consumer protection risks.
"""

import streamlit as st
import pandas as pd
import altair as alt
from supabase import create_client
from dotenv import load_dotenv
import os
import json

from src.data_loader import fetch_articles
from src.sidebar_controls import sidebar_controls
from src.semantic_storytelling import run_semantic_storytelling
from src.openai_summary import summarize_text
from src.openai_complaint_summary import summarize_complaint_data
from src.library_viewer import render_library_viewer
from src.usaa_logo import display_usaa_logo


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
# Fetch Federal Reserve Articles
# -----------------
all_articles = fetch_articles()
if all_articles.empty:
    st.warning("⚠️ No articles found in Supabase.")
else:
    st.success(f"✅ Loaded {len(all_articles)} Federal Reserve articles.")


# -----------------
# Semantic Storytelling + OpenAI Summary
# -----------------
st.markdown("### ⚙️ Semantic Exploration & Storytelling")

query_sentence = st.text_input(
    "Enter a question or sentence to explore:",
    value=default_query,
    placeholder="e.g., How are banks addressing AML and third-party risks?"
)
run_semantic = st.button("Run Semantic Search")

if run_semantic and query_sentence.strip():
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
            patterns, top_articles = run_semantic_storytelling(df, query_sentence)
            if isinstance(patterns, dict) and any(patterns.values()):
                st.divider()
                st.markdown("### 🧠 AI-Generated Summary (OpenAI)")
                top_text = " ".join(top_articles["content"].fillna("").tolist()[:5])[:8000]
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
                summary_input = (
                    f"User question: {query_sentence}\n"
                    f"{focus_note}\n\n"
                    f"--- Article Excerpts ---\n{top_text}\n\n"
                    f"--- Keywords ---\n{keyword_text}\n\n"
                    f"--- Common Phrases ---\n{phrase_text}\n\n"
                    f"--- Trend Summary ---\n{trend_summary}\n\n"
                    "Provide a cohesive, paragraph-style analytical summary "
                    "focusing only on fraud, compliance, or financial misconduct. "
                    "Include statistics directly tied to the topic."
                )
                summary_text = summarize_text(summary_input)
                st.write(summary_text)
            else:
                st.warning("No meaningful patterns detected for summarization.")
else:
    st.info("Enter a question or choose a preset, then click **Run Semantic Search**.")


# -----------------
# CFPB Complaint Data Integration (AI-driven)
# -----------------
st.divider()
st.markdown("### 🧭 Consumer Complaint Trends (CFPB Integration)")

try:
    cfpb_resp = supabase.table("cfpb_complaints").select("*").execute()
    complaints = pd.DataFrame(cfpb_resp.data or [])

    if complaints.empty:
        st.info("No CFPB complaint data found in Supabase.")
    else:
        st.success(f"✅ Loaded {len(complaints)} complaints from CFPB dataset.")

        for col in ["domain_label", "state", "product"]:
            if col in complaints.columns:
                complaints[col] = complaints[col].astype(str).fillna("Unknown")

        stats = {
            "total_complaints": len(complaints),
            "by_domain": complaints["domain_label"].value_counts().to_dict()
            if "domain_label" in complaints.columns else {},
            "by_state": complaints["state"].value_counts().head(10).to_dict()
            if "state" in complaints.columns else {},
            "by_product": complaints["product"].value_counts().head(10).to_dict()
            if "product" in complaints.columns else {},
        }

        date_range = (
            complaints["date_received"].min() if "date_received" in complaints.columns else None,
            complaints["date_received"].max() if "date_received" in complaints.columns else None,
        )

        with st.spinner("Generating AI-driven CFPB summary and visuals..."):
            ai_output = summarize_complaint_data(stats, date_range)
            st.markdown("### 🧠 AI-Generated Complaint Summary")
            st.write(ai_output.get("summary_text", "No summary generated."))

        # -----------------
        # Visualization Section (Static)
        # -----------------
        st.markdown("### 📊 CFPB Complaint Visuals")

        if "domain_label" in complaints.columns:
            domain_chart = (
                alt.Chart(complaints)
                .mark_bar()
                .encode(
                    x=alt.X("domain_label:N", sort="-y", title="Fraud/Compliance Domain"),
                    y=alt.Y("count():Q", title="Number of Complaints"),
                    color=alt.Color("domain_label:N", title="Domain"),
                    tooltip=["domain_label", "count()"],
                )
                .properties(title="Complaint Volume by Domain", height=400)
            )
            st.altair_chart(domain_chart, use_container_width=True)

        if "state" in complaints.columns:
            geo_chart = (
                alt.Chart(complaints)
                .mark_bar()
                .encode(
                    y=alt.Y("state:N", sort="-x", title="State"),
                    x=alt.X("count():Q", title="Complaint Count"),
                    color=alt.Color("state:N", title="State"),
                    tooltip=["state", "count()"],
                )
                .properties(title="Complaints by State", height=500)
            )
            st.altair_chart(geo_chart, use_container_width=True)

        if "date_received" in complaints.columns:
            complaints["date_received"] = pd.to_datetime(complaints["date_received"], errors="coerce")
            time_chart = (
                alt.Chart(complaints)
                .mark_line(point=True)
                .encode(
                    x=alt.X("yearmonth(date_received):T", title="Date"),
                    y=alt.Y("count():Q", title="Complaint Count"),
                    color=alt.Color("domain_label:N", title="Domain"),
                    tooltip=["yearmonth(date_received)", "domain_label", "count()"],
                )
                .properties(title="Complaint Trends Over Time", height=350)
            )
            st.altair_chart(time_chart, use_container_width=True)

except Exception as e:
    st.warning(f"⚠️ Unable to load CFPB complaint data: {e}")


# -----------------
# Library Viewer Integration
# -----------------
if st.sidebar.button("📚 Open Library Viewer"):
    st.session_state["view_library"] = True

if st.session_state.get("view_library"):
    render_library_viewer()
