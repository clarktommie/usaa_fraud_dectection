# src/topic_dashboard.py

"""
Topic Dashboard
----------------------------------------
Visualizes BERTopic output stored in:
    - topics
    - doc_topics
    - topic_trends
Adds:
    ✅ Topic explorer
    ✅ Heatmaps + trends
    ✅ Forecasting (Prophet/ARIMA)
"""

import pandas as pd
import altair as alt
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv
import os

# from src.topic_forecasting import forecast_topic, forecast_all_topics

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =========================
# Load tables
# =========================
def load_table(name: str) -> pd.DataFrame:
    try:
        resp = supabase.table(name).select("*").execute()
        df = pd.DataFrame(resp.data or [])
        return df
    except Exception:
        return pd.DataFrame()


# =========================
# Render Dashboard
# =========================
def render_topic_dashboard():
    st.title("📘 Topic Modeling Dashboard")
    st.markdown("Explore BERTopic clusters, trends, and forecasts.")

    # -------------------------
    # Load data
    # -------------------------
    topics_df = load_table("topics")
    docs_df = load_table("doc_topics")
    trends_df = load_table("topic_trends")

    if topics_df.empty:
        st.warning("No topics found in Supabase.")
        return

    # Fix timestamp
    if "timestamp" in trends_df.columns:
        trends_df["timestamp"] = pd.to_datetime(trends_df["timestamp"], errors="coerce")

    # =========================
    # 1. Topic Overview Table
    # =========================
    st.subheader("📊 Topic Overview")
    show_cols = ["topic_id", "name", "count", "top_words"]
    for c in show_cols:
        if c not in topics_df.columns:
            topics_df[c] = None

    st.dataframe(
        topics_df[show_cols].sort_values("topic_id").reset_index(drop=True),
        use_container_width=True
    )

    # =========================
    # 2. Topic Selection
    # =========================
    st.subheader("🔍 Explore a Topic")
    topic_list = sorted(topics_df["topic_id"].unique())
    selected_topic = st.selectbox("Choose a topic:", topic_list)

    topic_name = topics_df[topics_df["topic_id"] == selected_topic]["name"].values[0]
    st.markdown(f"### Topic {selected_topic}: **{topic_name}**")

    # Show top words
    words = topics_df[topics_df["topic_id"] == selected_topic]["top_words"].values[0]
    st.markdown("**Top Words:** " + ", ".join(words))

    # =========================
    # 3. Topic Trend Chart
    # =========================
    st.subheader("📈 Topic Frequency Over Time")

    tdf = trends_df[trends_df["topic_id"] == selected_topic].copy()
    if tdf.empty:
        st.info("No trend data available.")
    else:
        tdf = tdf.sort_values("timestamp")

        chart = (
            alt.Chart(tdf)
            .mark_line(point=True)
            .encode(
                x=alt.X("timestamp:T", title="Month"),
                y=alt.Y("count:Q", title="Frequency"),
                tooltip=["timestamp", "count"],
            )
            .properties(height=350)
        )
        st.altair_chart(chart, use_container_width=True)

    # =========================
    # 4. Forecasting UI
    # =========================
    st.subheader("🔮 Topic Trend Forecasting (Next 6 Months)")

    if st.button("Generate Forecast"):
        with st.spinner("Forecasting topic trend..."):
            forecast_df = forecast_topic(topic_id=selected_topic, periods=6)

        if forecast_df.empty:
            st.warning("Forecast failed or insufficient data.")
        else:
            # Clean
            forecast_df = forecast_df[["ds", "yhat"]].rename(columns={"ds": "timestamp"})

            # Merge actual + forecast
            tdf2 = tdf.rename(columns={"count": "y"})
            tdf2["type"] = "Actual"

            fdf = forecast_df.copy()
            fdf["y"] = fdf["yhat"]
            fdf["type"] = "Forecast"

            merged = pd.concat([tdf2[["timestamp", "y", "type"]], fdf[["timestamp", "y", "type"]]])

            fc_chart = (
                alt.Chart(merged)
                .mark_line(point=True)
                .encode(
                    x=alt.X("timestamp:T", title="Month"),
                    y=alt.Y("y:Q", title="Frequency"),
                    color=alt.Color("type:N", title=""),
                    tooltip=["timestamp", "y", "type"],
                )
                .properties(height=380)
            )

            st.altair_chart(fc_chart, use_container_width=True)

    # =========================
    # 5. Topic → Article Explorer
    # =========================
    st.subheader("📄 Top Articles for This Topic")

    if "topic" in docs_df.columns:
        df = docs_df[docs_df["topic"] == selected_topic].copy()

        if df.empty:
            st.info("No articles assigned to this topic.")
        else:
            show = df[["date", "title", "source", "url", "probability"]].sort_values(
                "probability", ascending=False
            )

            st.dataframe(show.reset_index(drop=True), use_container_width=True)

    st.divider()
    st.markdown("✅ Topic Dashboard Ready")
