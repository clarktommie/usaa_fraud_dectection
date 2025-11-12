"""
data_loader.py
---------------------------------
Fetches embedded press release metadata from Supabase.
The full article text is fetched later by semantic_storytelling.py.
"""

import os
import time
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import streamlit as st

# -----------------------------
# Environment setup
# -----------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# Fetch articles with pagination
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_articles():

    """Fetch all press release embeddings from Supabase with pagination."""
    try:
        all_data = []
        batch_size = 500
        start = 0

        print("📥 Fetching press release embeddings from Supabase...")

        while True:
            end = start + batch_size - 1
            response = (
                supabase.table("press_releases_clean")
                .select("id, title, content, source, date, scraped_at, updated_at, embedding")
                .range(start, end)
                .execute()
            )

            data = response.data or []
            all_data.extend(data)

            print(f"  • Retrieved rows {start}–{end} (total: {len(all_data)})")

            if len(data) < batch_size:
                break  # no more data to fetch

            start += batch_size
            time.sleep(0.1)

        print(f"✅ Loaded {len(all_data)} total press releases.")

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)

        # ✅ Ensure embedding column is parsed correctly
        # if "embedding" in df.columns:
        #     df["embedding"] = df["embedding"].apply(
        #         lambda x: eval(x) if isinstance(x, str) else x
        #     )

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching articles: {e}")
        return pd.DataFrame()
