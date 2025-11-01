"""
data_loader.py
---------------------------------
Fetches embedded press release metadata from Supabase.
The full article text is fetched later by semantic_storytelling.py.
"""

import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_articles():
    """Fetch press release embeddings from Supabase."""
    try:
        response = (
            supabase.table("press_releases_embed")
            .select("id, title, embedding, url, created_at, updated_at")
            .execute()
        )
        data = response.data or []
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)

        # ✅ Ensure embedding column is parsed correctly
        if "embedding" in df.columns:
            df["embedding"] = df["embedding"].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )

        return df

    except Exception as e:
        st.error(f"⚠️ Error fetching articles: {e}")
        return pd.DataFrame()
