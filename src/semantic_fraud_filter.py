# src/semantic_fraud_filter.py

import os
import time
import pandas as pd
from datetime import datetime, UTC
from supabase import create_client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SOURCE_TABLE = "press_releases_clean"

def semantic_filter(user_query: str, top_n: int = 50) -> pd.DataFrame:
    """
    Run a live semantic similarity search between the user's query
    and all press releases stored in Supabase.
    Returns the top-N most relevant articles.
    """
    print(f"🔎 Running live semantic filter for query: '{user_query}'")

    rows = []
    start = 0
    batch = 500
    while True:
        resp = (
            supabase.table(SOURCE_TABLE)
            .select("id,title,content,url,date,source")
            .range(start, start + batch - 1)
            .execute()
        )
        data = resp.data or []
        rows.extend(data)
        if len(data) < batch:
            break
        start += batch
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No articles found in Supabase.")
        return pd.DataFrame()

    df["text"] = (df["title"].fillna("") + ". " + df["content"].fillna(""))

    print("🧠 Loading embedding model (BAAI/bge-small-en-v1.5)…")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    docs = df["text"].tolist()
    print("📈 Computing semantic similarities…")

    query_emb = model.encode(user_query, normalize_embeddings=True)
    doc_emb = model.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    scores = util.cos_sim(query_emb, doc_emb)[0].cpu().tolist()
    df["semantic_score"] = scores

    df_sorted = df.sort_values("semantic_score", ascending=False).head(top_n)
    print(f"✅ Retrieved top {len(df_sorted)} semantically relevant articles.")
    return df_sorted
