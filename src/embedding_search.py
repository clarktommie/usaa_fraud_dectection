import os
import numpy as np
import streamlit as st
from supabase import create_client
from fastembed import TextEmbedding
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Cached Embedder ---
@st.cache_resource(show_spinner=False)
def get_embedder():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

def run_search(query, year_filter=None, keyword_filter=None, top_k=20, threshold=0.6):
    """Perform semantic search through Supabase RPC."""
    if not query.strip():
        return []

    embedder = get_embedder()
    query_vector = list(embedder.embed([query]))[0]

    # Step 1: call RPC for embedding match
    resp = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": query_vector,
            "match_threshold": float(threshold),
            "match_count": int(max(top_k, 50)),
        },
    ).execute()

    results = resp.data or []
    if not results:
        return []

    # Step 2: fetch article metadata for matched IDs
    ids = [r["id"] for r in results]
    meta = supabase.table("press_releases_clean").select(
        "id, title, date, content, url"
    ).in_("id", ids).execute()

    meta_map = {m["id"]: m for m in meta.data or []}
    enriched = [{**r, **meta_map.get(r["id"], {})} for r in results]

    # Step 3: apply optional filters
    if year_filter:
        enriched = [r for r in enriched if r.get("date") and str(year_filter) in str(r["date"])]
    if keyword_filter:
        enriched = [r for r in enriched if keyword_filter.lower() in (r.get("content") or "").lower()]

    # Step 4: sort and return
    enriched = sorted(enriched, key=lambda x: x.get("similarity", 0.0), reverse=True)[:top_k]
    return enriched
