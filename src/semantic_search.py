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
    """Perform semantic + optional keyword/year search."""
    query_vector = np.array(list(embedder.embed([query]))[0])

    # Step 1: Call the vector similarity function in Supabase
    response = supabase.rpc(
        "match_press_releases",
        {
            "query_embedding": query_vector.tolist(),
            "match_threshold": 0.6,
            "match_count": 100  # get more results to filter later
        },
    ).execute()

    results = response.data or []
    if not results:
        print("⚠️ No semantic matches found.")
        return

    # Step 2: Apply keyword/year filters locally
    filtered = results
    if year:
        filtered = [r for r in filtered if r.get("date") and str(year) in str(r["date"])]
    if keyword:
        filtered = [r for r in filtered if keyword.lower() in (r.get("content") or "").lower()]

    # Step 3: Sort by similarity and show top_k
    filtered = sorted(filtered, key=lambda x: x["similarity"], reverse=True)[:top_k]

    print(f"\nTop {len(filtered)} semantic matches for: '{query}'")
    if year:
        print(f"📅 Filter: Year = {year}")
    if keyword:
        print(f"🔍 Filter: Keyword = '{keyword}'")

    print()
    for i, r in enumerate(filtered, 1):
        print(f"{i}. {r['title']}")
        print(f"   Score: {r['similarity']:.3f}")
        print(f"   Date: {r.get('date', 'N/A')}")
        print(f"   URL: {r['url']}\n")

if __name__ == "__main__":
    # Example queries
    semantic_search("debit card fraud")
    # semantic_search("fraud enforcement action", year=2024)
    # semantic_search("money laundering", keyword="AML")
    # semantic_search("financial scams", year=2022, keyword="bank")
