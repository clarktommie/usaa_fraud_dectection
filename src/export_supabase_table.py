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

# --- Export all records from press_releases table ---
def export_press_releases():
    print("📤 Fetching data from Supabase...")
    response = supabase.table("press_releases").select("*").execute()

    if not response.data:
        print("⚠️ No data found in table.")
        return

    df = pd.DataFrame(response.data)

    # Define your target export path
    output_path = "/home/tclark/Data Science/usaa_fraud_dection/usaa_fraud_dectection/data/press_releases_dump.csv"

    # Create the directory if it doesn’t exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df)} records to {output_path}")

if __name__ == "__main__":
    export_press_releases()
    print("Finished.")