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

import ast
import os
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from supabase import create_client

from src.cfpb_loader import fetch_complaints

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


def _ensure_word_list(value) -> List[str]:
    """Normalize Supabase 'top_words' column to a clean list[str]."""
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v]
            except Exception:
                pass
        if "," in text:
            return [w.strip() for w in text.split(",") if w.strip()]
        if text:
            return [text]
    return []


def prepare_topic_space(topics_df: pd.DataFrame) -> pd.DataFrame:
    """Project topic keywords into 3D space via TF-IDF + TruncatedSVD."""
    df = topics_df.copy()
    df["keyword_text"] = df["top_words"].apply(_ensure_word_list).apply(lambda words: " ".join(words))
    text_series = df["keyword_text"].fillna("")

    if text_series.str.strip().eq("").all():
        df["dim_x"] = df["dim_y"] = df["dim_z"] = 0.0
        return df

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(text_series)

    n_components = min(3, matrix.shape[1] - 1 if matrix.shape[1] > 1 else 1)
    if n_components <= 0:
        reduced = np.zeros((matrix.shape[0], 3))
    else:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(matrix)
        if n_components < 3:
            pad = np.zeros((matrix.shape[0], 3 - n_components))
            reduced = np.hstack([reduced, pad])

    df["dim_x"] = reduced[:, 0]
    df["dim_y"] = reduced[:, 1] if reduced.shape[1] > 1 else 0.0
    df["dim_z"] = reduced[:, 2] if reduced.shape[1] > 2 else 0.0
    return df


def render_topic_scatter(space_df: pd.DataFrame, selected_topic: int, keyword: str):
    """Render a Plotly 3D scatter plot highlighting the keyword/topic."""
    if space_df.empty:
        st.info("Topic data unavailable for scatter plot.")
        return

    keyword_lower = keyword.lower()
    space_df = space_df.copy()
    space_df["has_keyword"] = space_df["keyword_text"].str.contains(keyword_lower, case=False, na=False)
    space_df["is_selected"] = space_df["topic_id"] == selected_topic
    space_df["display_label"] = space_df.apply(
        lambda row: f"Topic {row['topic_id']}: {row.get('name', 'N/A')}", axis=1
    )

    fig = px.scatter_3d(
        space_df,
        x="dim_x",
        y="dim_y",
        z="dim_z",
        color="has_keyword",
        symbol="is_selected",
        hover_name="display_label",
        hover_data={
            "count": True,
            "keyword_text": True,
            "dim_x": ":.2f",
            "dim_y": ":.2f",
            "dim_z": ":.2f",
            "has_keyword": False,
            "is_selected": False,
        },
        color_discrete_map={True: "#c43d4d", False: "#9ecae1"},
        symbol_map={True: "diamond", False: "circle"},
        title=f"3D Topic Keyword Space — Highlighting '{keyword}'",
        height=500,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)


def render_keyword_choropleth(complaints_df: pd.DataFrame, keyword: str):
    """Render a USA choropleth map showing where the keyword appears in complaints."""
    if complaints_df.empty:
        st.info("Complaint data unavailable for choropleth.")
        return

    keyword = keyword.strip()
    if not keyword:
        st.info("Enter a keyword to map its geographic footprint.")
        return

    df = complaints_df.copy()
    search_cols = [col for col in ["issue", "product", "company", "domain_label"] if col in df.columns]
    if not search_cols:
        st.info("Complaint dataset lacks searchable keyword columns.")
        return

    mask = False
    for col in search_cols:
        mask = mask | df[col].astype(str).str.contains(keyword, case=False, na=False)

    keyword_df = df[mask].copy()
    if keyword_df.empty:
        st.warning(f"No complaints mention '{keyword}'.")
        return

    keyword_df["state"] = keyword_df["state"].str.upper()
    state_counts = (
        keyword_df[keyword_df["state"].str.len() == 2]
        .groupby("state")
        .size()
        .reset_index(name="mentions")
    )

    if state_counts.empty:
        st.warning("Keyword matches lack state information.")
        return

    fig = px.choropleth(
        state_counts,
        locations="state",
        color="mentions",
        color_continuous_scale="Reds",
        locationmode="USA-states",
        scope="usa",
        labels={"mentions": "Matches"},
        title=f"Choropleth of '{keyword}' Mentions in CFPB Complaints",
    )
    st.plotly_chart(fig, use_container_width=True)


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
    complaints_df = fetch_complaints()

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

    topics_preview = topics_df.copy()
    topics_preview["top_words"] = topics_preview["top_words"].apply(_ensure_word_list)
    st.dataframe(
        topics_preview[show_cols].sort_values("topic_id").reset_index(drop=True),
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
    words = _ensure_word_list(
        topics_df[topics_df["topic_id"] == selected_topic]["top_words"].values[0]
    )
    st.markdown("**Top Words:** " + ", ".join(words) if words else "No keywords detected.")

    # =========================
    # 2b. Keyword Visual Explorer
    # =========================
    st.subheader("🧭 Topic Keyword Visual Explorer")
    topic_space_df = prepare_topic_space(topics_df)
    keyword_options = words.copy()

    if not keyword_options:
        fallback = (
            topics_df["top_words"]
            .apply(_ensure_word_list)
            .explode()
            .dropna()
            .unique()
            .tolist()
        )
        keyword_options = sorted(set(fallback))

    keyword_choice = None
    if keyword_options:
        keyword_choice = st.selectbox(
            "Select a keyword or phrase from this topic:",
            keyword_options,
            index=0,
        )
    custom_keyword = st.text_input(
        "Or enter another keyword/phrase to analyze:",
        value=keyword_choice or "",
        placeholder="e.g., money laundering",
    )
    focus_keyword = custom_keyword.strip() or (keyword_choice or "")

    if focus_keyword:
        render_topic_scatter(topic_space_df, selected_topic, focus_keyword)
        render_keyword_choropleth(complaints_df, focus_keyword)
        st.caption("The 3D scatter shows how all topics relate to this keyword; the choropleth maps complaint hotspots.")
    else:
        st.info("Add a keyword to unlock the 3D scatter and choropleth visuals.")

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
