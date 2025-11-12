# src/topic_trends.py
"""
topic_trends.py
-------------------------------------------------------
Trend analytics + future forecasting for BERTopic topics.

Inputs:
  - topics_df:     from topic_model.py
  - doc_topics_df: document → topic assignments
  - trends_df:     monthly aggregated topic trends

Outputs:
  - trend_summary: rising, declining, cyclical topics
  - forecast_df:   predicted future topic weights (next 6–12 months)
  - chart helpers: Streamlit-safe visual encoding

Guaranteed fields for plots:
  ['year_month','topic_id','docs','avg_prob','weight','topic_name']

This module avoids external heavy libs (Prophet) and uses a smooth
trend+seasonal decomposition + linear regression for forecasting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


# ============================================================
# Helpers
# ============================================================
def _ym_to_numeric(ym: str) -> int:
    """Convert YYYY-MM → integer index, e.g., 2024-05 → 202405."""
    try:
        y, m = ym.split("-")
        return int(y) * 100 + int(m)
    except Exception:
        return None


def _smooth(series: np.ndarray, w: int = 3) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(series) < w:
        return series
    return np.convolve(series, np.ones(w) / w, mode="same")


def _linear_forecast(x: np.ndarray, y: np.ndarray, horizon: int = 6) -> np.ndarray:
    """Linear regression future values."""
    if len(x) < 3:
        # Not enough history → flat projection
        return np.array([y[-1]] * horizon)

    # Fit y = ax + b
    coef = np.polyfit(x, y, 1)
    a, b = coef[0], coef[1]

    future_x = np.arange(x[-1] + 1, x[-1] + horizon + 1)
    pred = a * future_x + b
    return pred


# ============================================================
# Data structure used everywhere
# ============================================================
@dataclass
class TopicTrendOutputs:
    rising: pd.DataFrame
    declining: pd.DataFrame
    cyclical: pd.DataFrame
    forecast_df: pd.DataFrame
    enriched_trends: pd.DataFrame


# ============================================================
# Enrichment
# ============================================================
def enrich_trends(trends_df: pd.DataFrame, topics_df: pd.DataFrame) -> pd.DataFrame:
    df = trends_df.copy()

    # Ensure expected fields exist
    for col in ["year_month", "topic_id", "docs", "avg_prob", "weight"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Map topic names
    name_map = dict(zip(topics_df["topic_id"], topics_df["name"]))
    df["topic_name"] = df["topic_id"].map(name_map).fillna("Unknown Topic")

    # numeric month index
    df["ym_num"] = df["year_month"].apply(_ym_to_numeric)

    return df.sort_values(["topic_id", "ym_num"])


# ============================================================
# Trend Classification
# ============================================================
def classify_trends(enriched: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rising = []
    declining = []
    cyclical = []

    # For each topic, inspect slope of weight over time
    for tid, grp in enriched.groupby("topic_id"):
        grp = grp.sort_values("ym_num")
        if len(grp) < 4:
            continue

        y = grp["weight"].values
        x = np.arange(len(y))

        # Smooth
        y_smooth = _smooth(y, w=3)

        # Compute slope
        slope = np.polyfit(x, y_smooth, 1)[0]

        if slope > 0.02:
            rising.append({"topic_id": tid, "topic_name": grp["topic_name"].iloc[0], "slope": slope})
        elif slope < -0.02:
            declining.append({"topic_id": tid, "topic_name": grp["topic_name"].iloc[0], "slope": slope})
        else:
            cyclical.append({"topic_id": tid, "topic_name": grp["topic_name"].iloc[0], "slope": slope})

    return (
        pd.DataFrame(rising).sort_values("slope", ascending=False),
        pd.DataFrame(declining).sort_values("slope"),
        pd.DataFrame(cyclical).sort_values("topic_name"),
    )


# ============================================================
# Forecasting
# ============================================================
def forecast_topics(enriched: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    fc_list = []

    for tid, grp in enriched.groupby("topic_id"):
        grp = grp.sort_values("ym_num")
        if len(grp) < 3:
            continue

        x = grp["ym_num"].values
        y = grp["weight"].values
        y_smooth = _smooth(y)

        pred = _linear_forecast(x, y_smooth, horizon=horizon)

        # Create future YM labels
        last = x[-1]
        fut_ym = []
        for i in range(horizon):
            next_val = last + (i + 1)
            # Convert e.g. 202412 → wrap to 2024-12 → next is 2025-01
            y = next_val // 100
            m = next_val % 100
            if m > 12:
                y += m // 12
                m = m % 12
                if m == 0:
                    m = 12
            fut_ym.append(f"{y:04d}-{m:02d}")

        for ym, v in zip(fut_ym, pred):
            fc_list.append({
                "topic_id": tid,
                "topic_name": grp["topic_name"].iloc[0],
                "year_month": ym,
                "predicted_weight": float(v)
            })

    return pd.DataFrame(fc_list)


# ============================================================
# Unified API
# ============================================================
def analyze_topic_trends(topics_df: pd.DataFrame,
                         doc_topics_df: pd.DataFrame,
                         trends_df: pd.DataFrame,
                         forecast_horizon: int = 6) -> TopicTrendOutputs:

    enriched = enrich_trends(trends_df, topics_df)

    rising, declining, cyclical = classify_trends(enriched)
    forecast_df = forecast_topics(enriched, horizon=forecast_horizon)

    return TopicTrendOutputs(
        rising=rising,
        declining=declining,
        cyclical=cyclical,
        forecast_df=forecast_df,
        enriched_trends=enriched,
    )


# ============================================================
# Streamlit-ready charts
# ============================================================
def build_topic_line_chart(enriched: pd.DataFrame, topic_id: int):
    import altair as alt
    sub = enriched[enriched["topic_id"] == topic_id].copy()
    if sub.empty:
        return None

    return (
        alt.Chart(sub)
        .mark_line(point=True)
        .encode(
            x=alt.X("year_month:N", sort="ascending"),
            y=alt.Y("weight:Q", title="Topic Weight"),
            tooltip=["topic_name", "year_month", "weight"],
            color=alt.Color("topic_name:N"),
        )
        .properties(title=f"Topic {topic_id} – Trend Over Time", height=350)
    )


def build_forecast_chart(forecast_df: pd.DataFrame, topic_id: int):
    import altair as alt
    sub = forecast_df[forecast_df["topic_id"] == topic_id].copy()
    if sub.empty:
        return None

    return (
        alt.Chart(sub)
        .mark_line(point=True, strokeDash=[4,2])
        .encode(
            x=alt.X("year_month:N", sort="ascending"),
            y=alt.Y("predicted_weight:Q", title="Predicted Weight"),
            tooltip=["topic_name", "year_month", "predicted_weight"],
            color=alt.Color("topic_name:N"),
        )
        .properties(title=f"Topic {topic_id} – Forecast Next Periods", height=350)
    )
