# app.py
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client
from fastembed import TextEmbedding

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

st.set_page_config(page_title="USAA Fraud Semantic Search", layout="wide")
st.title("🔎 USAA Fraud Semantic Search (MVP)")

with st.sidebar:
    st.header("Filters")
    year = st.text_input("Year (optional)", value="")
    keyword = st.text_input("Keyword in content (optional)", value="")
    threshold = st.slider("Match threshold", 0.0, 1.0, 0.6, 0.05)
    top_k = st.slider("Results to return", 5, 50, 20, 5)

query = st.text_input("Type a query (e.g., 'bank fraud', 'AML enforcement action')", "")
search_btn = st.button("Search")

def run_search(q, year_filter, keyword_filter, top_k, threshold):
    # Embed query
    q_vec = np.array(list(embedder.embed([q]))[0])

    # Fetch semantic matches
    resp = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": q_vec.tolist(),
            "match_threshold": float(threshold),
            "match_count": int(max(top_k, 50)),  # get extra to filter locally
        },
    ).execute()

    results = resp.data or []

    # Local filters
    if year_filter:
        results = [r for r in results if r.get("date") and str(year_filter) in str(r["date"])]
    if keyword_filter:
        kw = keyword_filter.lower()
        results = [r for r in results if kw in (r.get("content") or "").lower()]

    # Sort and trim
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
    return results

if search_btn and query.strip():
    with st.spinner("Searching…"):
        hits = run_search(query.strip(), year, keyword, top_k, threshold)

    if not hits:
        st.warning("No results found. Try lowering the threshold or broadening your query.")
    else:
        st.subheader(f"Top {len(hits)} matches")
        for i, r in enumerate(hits, 1):
            st.markdown(f"**{i}. {r.get('title','(no title)')}**  \n"
                        f"Score: `{r.get('similarity',0):.3f}` · Date: `{r.get('date','N/A')}`  \n"
                        f"[Open]({r.get('url')})")
            with st.expander("Preview"):
                st.write((r.get("content") or "")[:1500] + ("..." if r.get("content") and len(r["content"]) > 1500 else ""))
