"""Focus phrase inference and state-level keyword aggregation."""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd

STOP_WORDS = {
    "how",
    "are",
    "the",
    "what",
    "when",
    "where",
    "who",
    "which",
    "and",
    "or",
    "for",
    "from",
    "into",
    "about",
    "that",
    "this",
    "with",
    "their",
    "your",
    "does",
    "will",
    "can",
    "could",
    "should",
    "would",
    "why",
    "do",
    "did",
    "using",
    "use",
    "focus",
    "been",
    "being",
    "have",
    "has",
    "had",
    "over",
    "more",
    "than",
    "any",
    "our",
    "its",
    "also",
    "such",
    "trend",
    "trends",
    "addressing",
    "improving",
    "emerging",
    "responding",
    "managing",
}

STATE_COORDS = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DC": (38.897438, -77.026817),
    "DE": (39.318523, -75.507141),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "IA": (42.011539, -93.210526),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "KS": (38.526600, -96.726486),
    "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805),
    "MA": (42.230171, -71.530106),
    "MD": (39.063946, -76.802101),
    "ME": (44.693947, -69.381927),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MO": (38.456085, -92.288368),
    "MS": (32.741646, -89.678696),
    "MT": (46.921925, -110.454353),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "NE": (41.125370, -98.268082),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NV": (38.313515, -117.055374),
    "NY": (42.165726, -74.948051),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "PR": (18.220833, -66.590149),
    "RI": (41.680893, -71.511780),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VA": (37.769337, -78.169968),
    "VT": (44.045876, -72.710686),
    "WA": (47.400902, -121.490494),
    "WI": (44.268543, -89.616508),
    "WV": (38.491226, -80.954453),
    "WY": (42.755966, -107.302490),
}


def infer_focus_phrase(
    query_sentence: str,
    keyword_input: str = "",
    candidate_terms: Optional[List[str]] = None,
    articles_df: Optional[pd.DataFrame] = None,
) -> str:
    """Heuristically derive a keyword/phrase from user input or query sentence."""
    if keyword_input and keyword_input.strip():
        return keyword_input.strip()

    query_sentence = (query_sentence or "").strip()
    if not query_sentence and candidate_terms:
        for term in candidate_terms:
            if term:
                return str(term)
        return ""

    query_lower = query_sentence.lower()
    if candidate_terms:
        for term in candidate_terms:
            if term and term.lower() in query_lower:
                return term

    quote_match = re.search(r'["“](.+?)["”]', query_sentence)
    if quote_match:
        return quote_match.group(1).strip()

    tokens = [
        tok.lower()
        for tok in re.findall(r"[A-Za-z0-9\-]+", query_sentence)
        if tok.lower() not in STOP_WORDS and len(tok) > 2
    ]

    candidates: List[tuple[str, int]] = []
    max_len = min(4, len(tokens))
    for n in range(max_len, 1, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i : i + n])
            near_tail = 1 if i >= len(tokens) - n - 1 else 0
            candidates.append((phrase, near_tail))
    for idx, token in enumerate(tokens):
        near_tail = 1 if idx >= len(tokens) - 2 else 0
        candidates.append((token, near_tail))

    def score_phrase(entry: tuple[str, int]) -> float:
        phrase, tail_bonus = entry
        score = len(phrase) * 0.1 + len(phrase.split()) + tail_bonus * 2
        if phrase in query_lower:
            score += 1.0
        if articles_df is not None and not articles_df.empty:
            content = articles_df.get("content")
            if content is not None:
                try:
                    matches = content.astype(str).str.lower().str.contains(re.escape(phrase), na=False)
                    score += matches.sum() * 2
                except Exception:
                    pass
        return score

    if candidates:
        best_phrase = max(candidates, key=score_phrase)[0]
        return best_phrase.strip()

    if candidate_terms:
        for term in candidate_terms:
            if term:
                return str(term)

    return query_sentence


def build_state_heatmap_data(complaints_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """Return state-level complaint counts for the keyword with coordinates."""
    if complaints_df.empty or not keyword.strip():
        return pd.DataFrame()

    keyword = keyword.strip().lower()
    search_cols = [col for col in ["issue", "product", "company", "domain_label", "complaint_id"] if col in complaints_df.columns]
    if not search_cols:
        return pd.DataFrame()

    df = complaints_df.copy()
    mask = False
    for col in search_cols:
        mask = mask | df[col].astype(str).str.lower().str.contains(keyword, na=False)
    df = df[mask].copy()
    if df.empty:
        return pd.DataFrame()

    df["state"] = df["state"].astype(str).str.upper()
    state_counts = (
        df[df["state"].isin(STATE_COORDS.keys())]
        .groupby("state")
        .size()
        .reset_index(name="mentions")
    )

    if state_counts.empty:
        return pd.DataFrame()

    state_counts["latitude"] = state_counts["state"].map(lambda s: STATE_COORDS[s][0])
    state_counts["longitude"] = state_counts["state"].map(lambda s: STATE_COORDS[s][1])
    return state_counts
