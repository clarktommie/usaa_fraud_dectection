# src/semantic_library.py
"""
semantic_library.py
---------------------------------
Save and retrieve topic-based semantic collections in Supabase.
Targets the 'public.library' table.
"""

import os
from datetime import datetime
from typing import Iterable, Any

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------
# Helpers
# ---------------------------
def _to_float(x: Any) -> float:
    """Coerce various similarity scalar/list/np types into a float."""
    try:
        # If list/tuple like [0.9123], use first element
        if isinstance(x, (list, tuple)) and x:
            return float(x[0])
        # NumPy scalars or plain numbers
        return float(x)
    except Exception:
        return 0.0


def _existing_article_ids_for_topic(topic: str) -> set:
    """Return set of article_ids already saved for this topic (to avoid duplicates)."""
    try:
        res = (
            supabase.table("library")
            .select("article_id")
            .ilike("topic", topic.lower())
            .execute()
        )
        rows = res.data or []
        return {r["article_id"] for r in rows if r.get("article_id")}
    except Exception:
        return set()


# ---------------------------
# Public API
# ---------------------------
def save_semantic_library(
    topic: str,
    query: str,
    articles_df: pd.DataFrame,
    similarities: Iterable[Any],
    avoid_duplicates: bool = False,
) -> int:
    """
    Save related articles into public.library.
    - Coerces similarity values to floats (handles lists/tuples/np scalars).
    - Truncates to the aligned length of (articles_df, similarities).
    - Skips rows with missing id/title/url; optionally avoids duplicates per topic.

    Args:
        topic: focus topic (e.g., 'phishing', 'aml')
        query: the user's original query
        articles_df: DataFrame with at least ['id','title','url','date']
        similarities: iterable of similarity scores aligned to articles_df rows
        avoid_duplicates: when True, avoid inserting existing article_ids for topic

    Returns:
        Number of inserted rows.
    """
    if articles_df is None or articles_df.empty:
        return 0

    # Materialize and coerce similarities -> floats
    sims_list = list(similarities)
    sims_list = [_to_float(x) for x in sims_list]

    # Align lengths safely
    n = min(len(articles_df), len(sims_list))
    if n == 0:
        return 0

    # Deduplicate by topic+article_id (optional)
    existing_ids = _existing_article_ids_for_topic(topic) if avoid_duplicates else set()
    payload = []

    # Use .iloc to preserve ordering; iterate aligned pairs
    for i in range(n):
        row = articles_df.iloc[i]
        aid = row.get("id")
        title = row.get("title")
        url = row.get("url")

        if not aid or not title or not url:
            continue
        if avoid_duplicates and aid in existing_ids:
            continue

        payload.append(
            {
                "topic": topic.lower(),
                "query": query,
                "article_id": aid,
                "title": title,
                "url": url,
                "summary": None,  # optional: populate later if you store summaries
                "created_at": datetime.utcnow().isoformat(),
            }
        )

    if not payload:
        return 0

    supabase.table("library").insert(payload).execute()
    return len(payload)


def get_topic_library(topic: str) -> pd.DataFrame:
    """Retrieve saved rows for a topic from public.library."""
    res = (
        supabase.table("library")
        .select("*")
        .ilike("topic", topic.lower())
        .order("created_at", desc=True)
        .execute()
    )
    return pd.DataFrame(res.data or [])
