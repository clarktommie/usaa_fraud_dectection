"\"\"\"Dynamic topic-focused visualization prep based on user query.\"\"\""

from __future__ import annotations

import os
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set for topic visuals.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"


def _parse_embedding(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return parsed
    return None


def _normalize(vec: np.ndarray) -> Optional[np.ndarray]:
    if vec is None:
        return None
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0 or np.isnan(norm):
        return None
    return arr / norm


def _embed_phrase(phrase: str) -> Optional[np.ndarray]:
    if not phrase:
        return None
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=phrase)
    return _normalize(resp.data[0].embedding)


def generate_topic_visual_data(
    articles_df: pd.DataFrame,
    focus_phrase: str,
    top_n: int = 200,
) -> Dict[str, pd.DataFrame]:
    """Return similarity-scored article subset and derived aggregates."""
    if articles_df.empty or "embedding" not in articles_df.columns:
        return {
            "articles": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "source_breakdown": pd.DataFrame(),
            "keyword_scores": pd.DataFrame(),
            "embedding_projection": pd.DataFrame(),
        }

    df = articles_df.copy()
    df["embedding_vector"] = df["embedding"].apply(_parse_embedding)
    df = df[df["embedding_vector"].notna()]
    if df.empty:
        return {
            "articles": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "source_breakdown": pd.DataFrame(),
            "keyword_scores": pd.DataFrame(),
            "embedding_projection": pd.DataFrame(),
        }

    query_vec = _embed_phrase(focus_phrase)
    if query_vec is None:
        return {
            "articles": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "source_breakdown": pd.DataFrame(),
            "keyword_scores": pd.DataFrame(),
            "embedding_projection": pd.DataFrame(),
        }

    df["embedding_norm"] = df["embedding_vector"].apply(_normalize)
    df = df[df["embedding_norm"].notna()]
    if df.empty:
        return {
            "articles": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "source_breakdown": pd.DataFrame(),
            "keyword_scores": pd.DataFrame(),
            "embedding_projection": pd.DataFrame(),
        }

    df["similarity"] = df["embedding_norm"].apply(lambda vec: float(np.dot(vec, query_vec)))
    df = df.sort_values("similarity", ascending=False).head(top_n).copy()
    if df.empty:
        return {
            "articles": pd.DataFrame(),
            "time_series": pd.DataFrame(),
            "source_breakdown": pd.DataFrame(),
            "keyword_scores": pd.DataFrame(),
            "embedding_projection": pd.DataFrame(),
        }

    matrix = np.vstack(df["embedding_norm"].to_list())
    n_components = min(3, matrix.shape[1])
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced = svd.fit_transform(matrix)
    if n_components < 3:
        pad = np.zeros((reduced.shape[0], 3 - n_components))
        reduced = np.hstack([reduced, pad])

    projection = pd.DataFrame(
        {
            "dim_1": reduced[:, 0],
            "dim_2": reduced[:, 1],
            "dim_3": reduced[:, 2],
            "title": df.get("title").fillna("Untitled"),
            "similarity": df["similarity"],
            "date": df.get("date"),
        }
    )

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    time_series = (
        df.dropna(subset=["date"])
        .assign(month=lambda d: d["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")
        .agg(
            avg_similarity=("similarity", "mean"),
            article_count=("id", "count"),
        )
        .reset_index()
        .rename(columns={"month": "timestamp"})
    )

    source_breakdown = (
        df.groupby(df.get("source", pd.Series(index=df.index, dtype=str)).fillna("Unknown"))
        .agg(
            article_count=("id", "count"),
            avg_similarity=("similarity", "mean"),
        )
        .reset_index()
        .rename(columns={"index": "source"})
        .sort_values("article_count", ascending=False)
        .head(10)
    )

    keyword_scores = pd.DataFrame()
    contents = df.get("content")
    if contents is not None and contents.notna().any():
        corpus = contents.fillna("").tolist()
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=40)
        matrix = vectorizer.fit_transform(corpus)
        scores = matrix.mean(axis=0).A1
        keyword_scores = (
            pd.DataFrame({"term": vectorizer.get_feature_names_out(), "score": scores})
            .sort_values("score", ascending=False)
            .head(20)
        )

    stats = {
        "article_count": int(len(df)),
        "avg_similarity": float(df["similarity"].mean()),
        "max_similarity": float(df["similarity"].max()),
        "unique_sources": int(df.get("source").nunique() if "source" in df else 0),
        "earliest_date": pd.to_datetime(df.get("date"), errors="coerce").min(),
        "latest_date": pd.to_datetime(df.get("date"), errors="coerce").max(),
    }

    return {
        "articles": df.reset_index(drop=True),
        "time_series": time_series,
        "source_breakdown": source_breakdown,
        "keyword_scores": keyword_scores,
        "embedding_projection": projection.reset_index(drop=True),
        "stats": stats,
    }
