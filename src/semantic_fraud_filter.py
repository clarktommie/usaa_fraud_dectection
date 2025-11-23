# src/semantic_fraud_filter.py

import pandas as pd

from src.article_index import ArticleIndex


def semantic_filter(user_query: str, top_n: int = 50, force_refresh: bool = False) -> pd.DataFrame:
    """
    Run a cached semantic similarity search between the user's query and
    all indexed press releases. Uses the locally cached OpenAI embedding
    index for speed; set force_refresh=True to rebuild the cache on demand.
    """
    index = ArticleIndex.load(force_refresh=force_refresh)
    return index.search(user_query, top_k=top_n)
