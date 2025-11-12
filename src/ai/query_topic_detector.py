"""
query_filter.py
---------------------------------
Filters press release and complaint data based on the user query
before visualization or semantic reasoning.
"""

import pandas as pd
import re


def filter_by_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Return only rows whose title or content match the query keywords."""
    if df.empty or "content" not in df:
        return pd.DataFrame()

    query = query.strip().lower()
    if not query:
        return df

    keywords = [re.escape(w) for w in query.split() if len(w) > 3]
    pattern = "|".join(keywords)
    mask = df["content"].astype(str).str.lower().str.contains(pattern, na=False) | \
           df["title"].astype(str).str.lower().str.contains(pattern, na=False)

    return df[mask].reset_index(drop=True)
