import os
from supabase import create_client
from dotenv import load_dotenv
from fastembed import TextEmbedding
import numpy as np

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Load FastEmbed model ---
embedder = TextEmbedding("BAAI/bge-small-en-v1.5")

def semantic_search(query, year=None, keyword=None, top_k=20):
    """Perform semantic search with optional year/keyword filters."""
    query_vector = list(embedder.embed([query]))[0]

    # Step 1: Call Supabase vector RPC
    response = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": query_vector,
            "match_threshold": 0.6,
            "match_count": 100
        },
    ).execute()

    results = response.data or []
    if not results:
        print("⚠️ No semantic matches found.")
        return []

    # Step 2: Collect IDs and fetch metadata
    ids = [r["id"] for r in results]
    meta_resp = (
        supabase.table("press_releases_clean")
        .select("id, title, url, content, date")
        .in_("id", ids)
        .execute()
    )
    meta_df = {m["id"]: m for m in meta_resp.data or []}

    # Step 3: Merge metadata
    enriched = []
    for r in results:
        meta = meta_df.get(r["id"], {})
        enriched.append({**r, **meta})

    # Step 4: Optional filters
    if year:
        enriched = [r for r in enriched if r.get("date") and str(year) in str(r["date"])]
    if keyword:
        enriched = [r for r in enriched if keyword.lower() in (r.get("content") or "").lower()]

    # Step 5: Sort and show top_k
    enriched = sorted(enriched, key=lambda x: x["similarity"], reverse=True)[:top_k]

    print(f"\nTop {len(enriched)} matches for: '{query}'")
    if year:
        print(f"📅 Year filter: {year}")
    if keyword:
        print(f"🔍 Keyword filter: {keyword}")
    print()

    for i, r in enumerate(enriched, 1):
        print(f"{i}. {r.get('title', 'Untitled')}")
        print(f"   Score: {r['similarity']:.3f}")
        print(f"   Date: {r.get('date', 'N/A')}")
        print(f"   URL: {r.get('url', 'N/A')}\n")

    return enriched


if __name__ == "__main__":
    semantic_search("check fraud")
    # semantic_search("offshore payments", keyword="fraud")
    # semantic_search("fraud enforcement action", year=2024)
    # semantic_search("money laundering", keyword="AML")
