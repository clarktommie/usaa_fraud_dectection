"""
data_loader.py
---------------------------------
Fetches press releases from Supabase in paginated batches
to avoid timeouts and large single-query loads.
"""

import os
import pandas as pd
import streamlit as st
from supabase import create_client
from dotenv import load_dotenv

# -------------------
# Setup
# -------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# -------------------
# Batched Fetch
# -------------------
@st.cache_data(show_spinner=True)
def fetch_articles(limit=1000, max_rows=10000):
    """
    Safely fetch articles from Supabase in small batches to avoid statement timeouts.
    Returns a pandas DataFrame of all retrieved articles.
    """
    all_data = []
    offset = 0

    try:
        while True:
            response = (
                supabase.table("press_releases_clean")
                .select("id, title, url, date_standard, content, author")
                .range(offset, offset + limit - 1)
                .execute()
            )

            batch = response.data or []
            if not batch:
                break

            all_data.extend(batch)
            offset += limit

            # stop after reaching max_rows safety cap
            if offset >= max_rows:
                break

        if not all_data:
            st.warning("⚠️ No data returned from Supabase.")
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # Normalize and standardize date
        if "date_standard" in df.columns:
            df["date"] = pd.to_datetime(df["date_standard"], errors="coerce")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Drop empty content rows
        df = df.dropna(subset=["content"])
        st.success(f"✅ Loaded {len(df)} articles from Supabase in batches of {limit}.")
        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching articles: {e}")
        return pd.DataFrame()
