"""
Simple agentic retrieval helper.
--------------------------------
Decides how to retrieve context for the RAG workflow by examining the user
query, sidebar focus word, and the cached ArticleIndex. The agent expands the
search window or retries with a focus-derived query when initial results are
sparse, ensuring Streamlit always receives enough context for downstream
summaries and visuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from src.article_index import ArticleIndex


@dataclass
class AgentStep:
    action: str
    detail: str


@dataclass
class AgenticResult:
    query_used: str
    top_k: int
    articles: pd.DataFrame
    steps: List[AgentStep] = field(default_factory=list)


class AgenticRetriever:
    """Lightweight agent that plans retrieval actions for the Streamlit app."""

    def __init__(self, index: ArticleIndex, min_results: int = 15):
        self.index = index
        self.min_results = min_results

    def retrieve(self, user_query: str, focus_hint: Optional[str] = "") -> AgenticResult:
        steps: List[AgentStep] = []
        query = (user_query or "").strip()
        focus_hint = (focus_hint or "").strip()

        if not query and focus_hint:
            query = focus_hint
            steps.append(
                AgentStep("focus_hint", "User query empty; used sidebar focus word.")
            )

        if not query:
            query = "financial fraud compliance risk"
            steps.append(
                AgentStep("default_seed", "Falling back to default fraud monitoring query.")
            )

        top_k = 30
        if len(query.split()) < 3:
            top_k = 60
            steps.append(
                AgentStep("expand_scope", "Short query detected; expanding retrieval window.")
            )

        primary_results = self.index.search(query, top_k=top_k)
        steps.append(
            AgentStep(
                "semantic_search",
                f"Semantic search on '{query}' returned {len(primary_results)} articles.",
            )
        )

        best_results = primary_results

        if (
            len(primary_results) < self.min_results
            and focus_hint
            and focus_hint.lower() not in query.lower()
        ):
            alt_query = f"{focus_hint} enforcement trend"
            focus_results = self.index.search(alt_query, top_k=60)
            steps.append(
                AgentStep(
                    "focus_retry",
                    f"Retrying with focus word '{focus_hint}' yielded {len(focus_results)} articles.",
                )
            )
            if len(focus_results) > len(best_results):
                best_results = focus_results
                query = alt_query
                top_k = 60

        if len(best_results) < self.min_results:
            fallback_rows = self.index.metadata.head(self.min_results).copy()
            steps.append(
                AgentStep(
                    "cache_backfill",
                    "Backfilled with cached articles to guarantee minimum context.",
                )
            )
            best_results = fallback_rows

        return AgenticResult(
            query_used=query,
            top_k=top_k,
            articles=best_results.reset_index(drop=True),
            steps=steps,
        )
