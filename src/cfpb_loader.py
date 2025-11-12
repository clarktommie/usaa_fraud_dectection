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

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_data(show_spinner=False)
def fetch_complaints(limit: int = 5000) -> pd.DataFrame:
    """Fetch complaint metadata and domain classification."""
    try:
        response = (
            supabase.table("cfpb_complaints")
            .select("complaint_id, date_received, product, issue, state, company, domain_label, similarity_score")
            .limit(limit)
            .execute()
        )

        data = response.data or []
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "date_received" in df.columns:
            df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching complaints: {e}")
        return pd.DataFrame()
