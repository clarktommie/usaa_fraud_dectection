# src/semantic_library.py
"""
semantic_library.py
---------------------------------
Save and retrieve topic-based semantic collections in Supabase.
Targets the 'public.library' table.
"""

import os
from datetime import datetime
from typing import Iterable, Any, List, Set

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Curated synonym map so semantic topic queries catch common variants (e.g., AML).
_TOPIC_SYNONYMS = {
    "anti money laundering": ["aml", "money laundering", "anti-money laundering"],
    "aml": ["anti money laundering", "money laundering", "anti-money laundering"],
    "money laundering": ["aml", "anti money laundering", "anti-money laundering"],
    "know your customer": ["kyc", "customer due diligence"],
    "kyc": ["know your customer", "customer due diligence"],
}


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


def _normalize_topic(topic: str) -> str:
    """Lowercase, strip, and collapse whitespace/hyphens for consistent matching."""
    norm = (topic or "").strip().lower().replace("-", " ")
    return " ".join(norm.split())


def _topic_variants(topic: str) -> List[str]:
    """
    Build semantic variants for a topic:
    - normalized phrase
    - acronym (e.g., anti money laundering -> aml)
    - curated synonyms/aliases
    """
    base = _normalize_topic(topic)
    if not base:
        return []

    variants: Set[str] = {base}

    tokens = base.split()
    if len(tokens) >= 2:
        acronym = "".join(word[0] for word in tokens if word)
        if acronym:
            variants.add(acronym)

    for alt in _TOPIC_SYNONYMS.get(base, []):
        variants.add(_normalize_topic(alt))

    return sorted(variants)


def topic_variants(topic: str) -> List[str]:
    """Public helper to expose the normalized + synonym variants used for querying."""
    return _topic_variants(topic)


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
    """Retrieve saved rows for a topic from public.library, including common variants."""
    variants = _topic_variants(topic)

    query = supabase.table("library").select("*")
    if variants:
        # Use OR across variant patterns; wildcard allows partial topic strings.
        filters = [f"topic.ilike.%{v}%" for v in variants]
        query = query.or_(",".join(filters))

    res = query.order("created_at", desc=True).execute()
    return pd.DataFrame(res.data or [])


def get_library_by_article_ids(article_ids: Iterable[Any]) -> pd.DataFrame:
    """
    Retrieve library rows whose article_id is in the provided list.
    Uses chunking to avoid Supabase filter size limits.
    """
    ids = [aid for aid in article_ids if aid is not None]
    if not ids:
        return pd.DataFrame()

    unique_ids = list(dict.fromkeys(ids))
    chunk_size = 200
    frames = []

    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start : start + chunk_size]
        res = (
            supabase.table("library")
            .select("*")
            .in_("article_id", chunk)
            .order("created_at", desc=True)
            .execute()
        )
        frames.append(pd.DataFrame(res.data or []))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False)
    return df.reset_index(drop=True)
