import os
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Initialize Supabase client ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Pull from raw table and insert into clean table ---
def migrate_press_releases():
    response = supabase.table("press_releases").select("*").execute()
    if not response.data:
        return
    df = pd.DataFrame(response.data)
    records = df.to_dict(orient="records")
    supabase.table("press_releases_clean").upsert(records).execute()

if __name__ == "__main__":
    migrate_press_releases()
