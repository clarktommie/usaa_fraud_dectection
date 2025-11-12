# src/AI/complaint_preprocessing.py

"""
complaint_preprocessing.py
---------------------------------
Transforms raw CFPB consumer complaint records into an AI-ready
structured dictionary for downstream insight generation.

This module extracts:
  - overall complaint counts
  - complaint types / categories
  - states involved
  - date trends
  - any fraud-related product/sub-product patterns
"""

import pandas as pd


def prepare_complaints_for_ai(df: pd.DataFrame) -> dict:
    """
    Convert raw complaint rows into statistical features that the LLM
    can reason about.

    Expected columns (typical CFPB schema):
      - complaint_id
      - date_received
      - product
      - sub_product
      - issue
      - sub_issue
      - state
      - company
      - complaint_what_happened (free text)

    Returns a dictionary like:
    {
        "total_complaints": 1234,
        "top_products": {...},
        "top_states": {...},
        "monthly_trend": [...],
        "fraud_keywords_found": {...}
    }
    """

    out = {}

    if df.empty:
        return {
            "total_complaints": 0,
            "top_products": {},
            "top_states": {},
            "monthly_trend": [],
            "fraud_keywords_found": {},
        }

    # Normalize dates
    df = df.copy()
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    # --- total count ---
    out["total_complaints"] = len(df)

    # --- product distribution ---
    if "product" in df.columns:
        out["top_products"] = (
            df["product"]
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        out["top_products"] = {}

    # --- state distribution ---
    if "state" in df.columns:
        out["top_states"] = (
            df["state"]
            .value_counts()
            .head(10)
            .to_dict()
        )
    else:
        out["top_states"] = {}

    # --- monthly trend ---
    if "date_received" in df.columns:
        trend = (
            df.set_index("date_received")
            .resample("M")
            .size()
            .reset_index(name="count")
        )
        out["monthly_trend"] = [
            {
                "date": str(row["date_received"].date()),
                "count": int(row["count"]),
            }
            for _, row in trend.iterrows()
        ]
    else:
        out["monthly_trend"] = []

    # --- fraud keyword scan ---
    FRAUD_TERMS = ["fraud", "scam", "identity theft", "unauthorized", "fake", "phishing"]
    text_col = "complaint_what_happened"

    if text_col in df.columns:
        combined = " ".join(df[text_col].dropna().astype(str)).lower()
        out["fraud_keywords_found"] = {
            term: combined.count(term)
            for term in FRAUD_TERMS
        }
    else:
        out["fraud_keywords_found"] = {}

    return out
