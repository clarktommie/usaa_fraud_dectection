import os
import numpy as np
import streamlit as st
from supabase import create_client
from fastembed import TextEmbedding
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@st.cache_resource(show_spinner=False)
def get_embedder():
    return TextEmbedding("BAAI/bge-small-en-v1.5")

def run_search(q, year_filter, keyword_filter, top_k, threshold):
    embedder = get_embedder()

    q_vec = np.array(list(embedder.embed([q]))[0])

    resp = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": q_vec.tolist(),
            "match_threshold": float(threshold),
            "match_count": int(max(top_k, 50)),
        },
    ).execute()

    results = resp.data or []

    if year_filter:
        results = [r for r in results if r.get("date") and str(year_filter) in str(r["date"])]
    if keyword_filter:
        kw = keyword_filter.lower()
        results = [r for r in results if kw in (r.get("content") or "").lower()]

    results = sorted(results, key=lambda x: x.get("similarity", 0.0), reverse=True)[:top_k]
    return results

