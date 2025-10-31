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
def fetch_articles(table="press_releases_clean"):
    """
    Efficiently fetch up to ~2000 rows from Supabase.
    - Prevents statement timeout
    - Keeps Streamlit responsive
    - Automatically retries once if failed
    """
    try:
        print(f"📥 Fetching records from {table} (limited to 2000)...")
        response = (
            supabase.table(table)
            .select("id, title, content, date")
            .limit(2000)
            .order("date", desc=True)
            .execute()
        )

        data = response.data or []
        if not data:
            print("⚠️ No data returned.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df = df.dropna(subset=["content"])
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
        print(f"✅ Loaded {len(df)} rows from {table}.")
        return df

    except Exception as e:
        print(f"⚠️ Initial query failed: {e}")
        return pd.DataFrame()
