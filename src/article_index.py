"""
article_index.py
---------------------------------
Builds and caches a local semantic index of press releases for fast lookup.
Uses OpenAI embeddings (text-embedding-3-small) and stores normalized vectors
alongside article metadata on disk so repeated searches avoid re-querying Supabase.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.data_loader import fetch_articles

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE = CACHE_DIR / "article_embeddings.npz"
META_CACHE = CACHE_DIR / "article_metadata.pkl"
TTL_HOURS = float(os.getenv("ARTICLE_INDEX_TTL_HOURS", "12"))


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _parse_embedding(value) -> Optional[Sequence[float]]:
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


def _embed_text(text: str) -> np.ndarray:
    """Embed and normalize a query string for similarity search."""
    if not text:
        return np.zeros(1536)
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    vec = np.array(response.data[0].embedding, dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


class ArticleIndex:
    _instance: Optional["ArticleIndex"] = None

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame, built_at: float):
        self.embeddings = embeddings
        self.metadata = metadata.reset_index(drop=True)
        self.built_at = built_at

    # ---------------------------
    # Cache helpers
    # ---------------------------
    @staticmethod
    def _cache_is_fresh() -> bool:
        if not EMBED_CACHE.exists() or not META_CACHE.exists():
            return False
        age_seconds = time.time() - EMBED_CACHE.stat().st_mtime
        return age_seconds < TTL_HOURS * 3600

    @classmethod
    def _load_from_disk(cls) -> "ArticleIndex":
        embeddings = np.load(EMBED_CACHE)["embeddings"]
        metadata = pd.read_pickle(META_CACHE)
        built_at = EMBED_CACHE.stat().st_mtime
        return cls(embeddings, metadata, built_at)

    @classmethod
    def _build_from_source(cls) -> "ArticleIndex":
        df = fetch_articles()
        if df.empty:
            raise RuntimeError("No articles returned from Supabase; cannot build index.")

        if "embedding" not in df.columns:
            raise RuntimeError("Supabase rows missing 'embedding' column.")

        df = df.copy()
        df["embedding"] = df["embedding"].apply(_parse_embedding)
        df = df[df["embedding"].notna()]

        if df.empty:
            raise RuntimeError(
                "No stored embeddings available to build index. Run the embedding pipeline first."
            )

        matrix = np.vstack(df["embedding"].to_list()).astype(float)
        normalized = _normalize_matrix(matrix)

        meta_cols = ["id", "title", "source", "date", "url", "content"]
        for col in meta_cols:
            if col not in df.columns:
                df[col] = None
        metadata = df[meta_cols].reset_index(drop=True)

        np.savez_compressed(EMBED_CACHE, embeddings=normalized)
        metadata.to_pickle(META_CACHE)
        return cls(normalized, metadata, time.time())

    @classmethod
    def load(cls, force_refresh: bool = False) -> "ArticleIndex":
        if cls._instance and not force_refresh:
            return cls._instance

        if force_refresh or not cls._cache_is_fresh():
            cls._instance = cls._build_from_source()
        else:
            cls._instance = cls._load_from_disk()
        return cls._instance

    # ---------------------------
    # Search
    # ---------------------------
    def search(self, query: str, top_k: int = 50) -> pd.DataFrame:
        if not query or not query.strip():
            return pd.DataFrame()
        query_vec = _embed_text(query.strip())
        scores = self.embeddings @ query_vec
        top_k = max(1, min(top_k, len(scores)))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = self.metadata.iloc[top_idx].copy()
        results["semantic_score"] = scores[top_idx]
        return results.reset_index(drop=True)

    def to_dataframe(self, include_embeddings: bool = True) -> pd.DataFrame:
        """Return cached metadata (optionally with embedding vectors)."""
        df = self.metadata.copy()
        if include_embeddings:
            df["embedding"] = [vec.tolist() for vec in self.embeddings]
        return df

    @property
    def article_count(self) -> int:
        return len(self.metadata)


def rebuild_article_index() -> ArticleIndex:
    """Force rebuild of the cached article index."""
    ArticleIndex._instance = ArticleIndex._build_from_source()
    return ArticleIndex._instance


if __name__ == "__main__":
    print("🔁 Rebuilding article index...")
    index = rebuild_article_index()
    print(f"✅ Cached {len(index.metadata)} articles at {time.ctime(index.built_at)}")
