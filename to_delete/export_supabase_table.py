import os
import time
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

# --- Load environment ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Constants ---
BATCH_SIZE = 1000

def migrate_press_releases():
    print("📥 Fetching from 'press_releases' and inserting into 'press_releases_clean'...")
    all_data = []
    start = 0

    while True:
        end = start + BATCH_SIZE - 1
        response = (
            supabase.table("press_releases")
            .select("*")
            .order("id")
            .range(start, end)
            .execute()
        )

        data = response.data or []
        all_data.extend(data)
        print(f"Fetched rows {start}–{end} (total so far: {len(all_data)})")

        if len(data) < BATCH_SIZE:
            break

        start += BATCH_SIZE
        time.sleep(0.3)

    if not all_data:
        print("⚠️ No data found in 'press_releases'.")
        return

    df = pd.DataFrame(all_data)
    print(f"✅ Total fetched: {len(df)} rows.")

    # Upload to clean table
    records = df.to_dict(orient="records")
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("press_releases_clean").upsert(batch).execute()
        print(f"⬆️ Uploaded rows {i}–{i+len(batch)-1}")

    print(f"🎯 Migration complete: {len(df)} rows inserted into 'press_releases_clean'.")

if __name__ == "__main__":
    migrate_press_releases()
