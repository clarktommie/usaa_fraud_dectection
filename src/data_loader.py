import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

# -------------------
# Setup
# -------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------
# Cached Safe Fetch
# -------------------
@st.cache_data(show_spinner=False)
def fetch_articles():
    """Fetch up to 10,000 articles from Supabase (press_releases_clean table)."""
    try:
        table = "press_releases_clean"
        all_rows = []
        batch_size = 1000
        offset = 0

        while True:
            response = (
                supabase.table(table)
                .select("id, title, url, date, content, author")
                .range(offset, offset + batch_size - 1)
                .execute()
            )
            data = response.data or []
            all_rows.extend(data)
            if len(data) < batch_size:
                break
            offset += batch_size

        if not all_rows:
            st.warning("⚠️ No data returned from Supabase.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = df.dropna(subset=["content"])

        # --- Use best date column available ---
        if "date_standard" in df.columns:
            df["date"] = pd.to_datetime(df["date_standard"], errors="coerce", utc=True)
        else:
            df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

        df["year"] = df["date"].dt.year.astype("Int64")

        st.success(f"✅ Loaded {len(df)} articles from Supabase.")
        return df

    except Exception as e:
        st.error(f"Error fetching articles: {e}")
        return pd.DataFrame()
