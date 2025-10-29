import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_articles(table="press_releases_clean"):
    response = supabase.table(table).select("title, content, date").execute()
    df = pd.DataFrame(response.data).dropna(subset=["content"])
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    return df

