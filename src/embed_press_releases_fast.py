import os
import time
import numpy as np
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from fastembed import TextEmbedding
from tqdm import tqdm

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Initialize model ---
print("🧠 Loading FastEmbed model (BAAI/bge-small-en-v1.5)...")
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

# --- Parameters ---
FETCH_SIZE = 1000
BATCH_SIZE = 100

def fetch_all_rows():
    print("📥 Fetching data from Supabase (paged)...")
    all_data = []
    start = 0

    while True:
        end = start + FETCH_SIZE - 1
        response = (
            supabase.table("press_releases_clean")
            .select("id, title, content")
            .order("id", desc=False)        # <-- ensures stable paging
            .range(start, end)
            .execute()
        )


        data = response.data or []
        all_data.extend(data)
        print(f"Fetched rows {start}–{end} (total so far: {len(all_data)})")

        # Stop if fewer than FETCH_SIZE rows returned
        if len(data) < FETCH_SIZE:
            break

        start += FETCH_SIZE
        time.sleep(0.5)

    print(f"✅ Total fetched: {len(all_data)} rows.")
    return pd.DataFrame(all_data)

# --- Main process ---
df = fetch_all_rows()
df = df.dropna(subset=["content"])
print(f"Embedding {len(df)} valid rows...")

# --- Embed and upload ---
for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[i:i+BATCH_SIZE]
    texts = batch["content"].tolist()
    ids = batch["id"].tolist()

    embeddings = list(embedder.embed(texts))
    records = [{"id": ids[j], "embedding": embeddings[j].tolist()} for j in range(len(ids))]

    supabase.table("press_releases_clean").upsert(records).execute()
    time.sleep(0.5)

print("✅ All embeddings uploaded successfully.")
