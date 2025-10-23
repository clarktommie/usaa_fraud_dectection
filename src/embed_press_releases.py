import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm

# --- Load environment ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load local data (or pull from Supabase if you prefer) ---
print("📥 Fetching data from Supabase...")
response = supabase.table("press_releases_clean").select("id, content").execute()
df = pd.DataFrame(response.data)

# --- Initialize embedding model ---
print("🧠 Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Generate embeddings ---
embeddings = []
print("⚙️ Generating embeddings...")
for text in tqdm(df["content"].fillna("").tolist()):
    emb = model.encode(text, normalize_embeddings=True)
    embeddings.append(emb.tolist())

df["embedding"] = embeddings

# --- Upload embeddings back to Supabase ---
print("⬆️ Uploading embeddings to Supabase...")
for i, row in tqdm(df.iterrows(), total=len(df)):
    supabase.table("press_releases_clean").update({
        "embedding": row["embedding"]
    }).eq("id", row["id"]).execute()

print("✅ Embeddings successfully uploaded.")
