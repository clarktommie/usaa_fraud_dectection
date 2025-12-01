"""
cfpb_loader.py
---------------------------------
Loads pre-classified CFPB consumer complaints from Supabase.
Used by Streamlit dashboard and fraud_insights.py to merge with articles.

"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import streamlit as st

# ---------------------------
# Environment & Supabase Client
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------
# Fetch CFPB Complaints (Paginated)
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_complaints(limit: int = 30000, batch_size: int = 1000) -> pd.DataFrame:
    """
    Fetch complaint metadata and domain classification with pagination.

    Supabase/PostgREST responses are typically capped at ~1000 rows per request,
    so we page using .range(start, end) until:
      - we hit `limit`, or
      - a page returns fewer than `batch_size` rows.
    """
    all_rows = []
    start = 0
    fetched = 0

    while True:
        # Respect overall limit if provided
        if limit is not None and fetched >= limit:
            break

        end = start + batch_size - 1
        if limit is not None:
            remaining = limit - fetched
            if remaining < batch_size:
                end = start + remaining - 1

        try:
            response = (
                supabase
                .table("cfpb_complaints")
                .select(
                    "complaint_id, date_received, product, issue, state, "
                    "company, domain_label, similarity_score"
                )
                .range(start, end)
                .execute()
            )
        except Exception as e:
            st.error(f"⚠️ Error fetching complaints: {e}")
            break

        data = response.data or []
        if not data:
            # No more rows available
            break

        all_rows.extend(data)
        batch_len = len(data)
        fetched += batch_len

        # If we got fewer than batch_size, we've reached the end of the table
        if batch_len < batch_size:
            break

        # Move to next page
        start += batch_size

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    return df
