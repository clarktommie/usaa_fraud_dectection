import os
import time
import numpy as np
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
from fastembed import TextEmbedding
from tqdm import tqdm

# ---------------------
# Setup
# ---------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------
# Model
# ---------------------
print("🧠 Loading FastEmbed model (BAAI/bge-large-en-v1.5)...")
embedder = TextEmbedding("BAAI/bge-large-en-v1.5")

# ---------------------
# Parameters
# ---------------------
FETCH_SIZE = 500
BATCH_SIZE = 100
SOURCE_TABLE = "press_releases_clean"   # ✅ embed from clean text
TARGET_TABLE = "press_releases_embed"   # ✅ store embeddings here

# ---------------------
# Helpers
# ---------------------
def clean_text(text: str) -> str:
    """Preprocess text for embedding."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.strip()

def fetch_all_rows():
    """Fetch all cleaned records from Supabase in pages."""
    print("📥 Fetching data from Supabase (paged)...")
    all_data = []
    start = 0

    while True:
        end = start + FETCH_SIZE - 1
        try:
            response = (
                supabase.table(SOURCE_TABLE)
                .select("id, title, content")
                .order("id", desc=False)
                .range(start, end)
                .execute()
            )
        except Exception as e:
            print(f"⚠️ Retry due to Supabase error at rows {start}–{end}: {e}")
            time.sleep(3)
            continue

        data = response.data or []
        all_data.extend(data)
        print(f"Fetched rows {start}–{end} (total so far: {len(all_data)})")

        if len(data) < FETCH_SIZE:
            break

        start += FETCH_SIZE
        time.sleep(0.5)

    print(f"✅ Total fetched: {len(all_data)} rows.")
    return pd.DataFrame(all_data)

def normalize(vectors):
    """L2 normalize embeddings for cosine similarity."""
    arr = np.array(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-12)

def upload_batch(records):
    """Upload a batch of embeddings with retry logic."""
    for attempt in range(2):
        try:
            supabase.table(TARGET_TABLE).upsert(records).execute()
            return
        except Exception as e:
            print(f"⚠️ Upload failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
    print("❌ Batch permanently failed after retries.")

# ---------------------
# Main Process
# ---------------------
df = fetch_all_rows()
df = df.dropna(subset=["content"])
print(f"🧾 Preparing {len(df)} valid rows for embedding...")

# Combine title + content
df["full_text"] = df.apply(
    lambda x: clean_text(f"{x['title']} — {x['content']}" if x["title"] else x["content"]),
    axis=1
)

# Batch embed
for i in tqdm(range(0, len(df), BATCH_SIZE), desc="Embedding batches"):
    batch = df.iloc[i : i + BATCH_SIZE]
    ids = batch["id"].tolist()
    texts = batch["full_text"].tolist()

    try:
        embeddings = list(embedder.embed(texts))
        embeddings = normalize(embeddings)
    except Exception as e:
        print(f"⚠️ Embedding error on batch {i // BATCH_SIZE}: {e}")
        continue

    records = [
        {
            "id": ids[j],
            "embedding": embeddings[j].tolist(),
            "title": batch.iloc[j]["title"],
        }
        for j in range(len(ids))
    ]

    upload_batch(records)
    time.sleep(0.3)

print(f"✅ All {len(df)} embeddings created and uploaded to '{TARGET_TABLE}'.")
