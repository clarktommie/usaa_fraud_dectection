"""
Focus phrase inference and intelligent state-level CFPB complaint aggregation.
Option A: Exclude generic CFPB categories so mapping stays fraud-focused.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# STOP WORDS FOR FOCUS PHRASE INFERENCE
# ============================================================
STOP_WORDS = {
    "how", "are", "the", "what", "when", "where", "who", "which",
    "and", "or", "for", "from", "into", "about", "that", "this",
    "with", "their", "your", "does", "will", "can", "could", "should",
    "would", "why", "do", "did", "using", "use", "focus", "been",
    "being", "have", "has", "had", "over", "more", "than", "any",
    "our", "its", "also", "such", "trend", "trends", "addressing",
    "improving", "emerging", "responding", "managing",
}


# ============================================================
# EXCLUDE CFPB CATEGORIES THAT ARE TOO GENERIC
# ============================================================
EXCLUDE_GENERIC_LABELS = {
    "incorrect information on your report",
    "credit reporting",
    "credit reporting company used questionable practices",
    "problem with a credit reporting company's investigation",
    "problem with a credit reporting company's investigation into an existing problem",
}


# ============================================================
# STATE COORDINATES FOR HEATMAP
# ============================================================
STATE_COORDS = {
    "AL": (32.806671, -86.791130), "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221), "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564), "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371), "DC": (38.897438, -77.026817),
    "DE": (39.318523, -75.507141), "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074), "HI": (21.094318, -157.498337),
    "IA": (42.011539, -93.210526), "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137), "IN": (39.849426, -86.258278),
    "KS": (38.526600, -96.726486), "KY": (37.668140, -84.670067),
    "LA": (31.169546, -91.867805), "MA": (42.230171, -71.530106),
    "MD": (39.063946, -76.802101), "ME": (44.693947, -69.381927),
    "MI": (43.326618, -84.536095), "MN": (45.694454, -93.900192),
    "MO": (38.456085, -92.288368), "MS": (32.741646, -89.678696),
    "MT": (46.921925, -110.454353), "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012), "NE": (41.125370, -98.268082),
    "NH": (43.452492, -71.563896), "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482), "NV": (38.313515, -117.055374),
    "NY": (42.165726, -74.948051), "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917), "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755), "PR": (18.220833, -66.590149),
    "RI": (41.680893, -71.511780), "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828), "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461), "UT": (40.150032, -111.862434),
    "VA": (37.769337, -78.169968), "VT": (44.045876, -72.710686),
    "WA": (47.400902, -121.490494), "WI": (44.268543, -89.616508),
    "WV": (38.491226, -80.954453), "WY": (42.755966, -107.302490),
}


# ============================================================
# NORMALIZATION FUNCTION
# ============================================================
def _norm(x: str) -> str:
    if not isinstance(x, str):
        return ""
    return re.sub(r"[^a-z0-9\s]", "", x.lower()).strip()


def _topic_synonyms(focus_norm: str) -> list[str]:
    """Map focus topics to a small set of search terms for CFPB complaints."""
    focus_norm = focus_norm or ""
    mapping = {
        "synthetic id": ["synthetic identity", "synthetic id", "identity theft"],
        "check fraud": ["check fraud", "check scam", "counterfeit check"],
        "wire": ["wire fraud", "ach fraud", "payment fraud"],
        "ach": ["wire fraud", "ach fraud", "payment fraud"],
        "money mule": ["money mule", "money mules", "mule network"],
        "elder": ["elder abuse", "senior scam", "elder exploitation"],
        "cyber": ["cyber fraud", "cybercrime", "ransomware"],
        "ransom": ["cyber fraud", "ransomware", "extortion"],
        "aml": ["aml", "anti money laundering", "bsa", "sanctions"],
        "sanction": ["sanctions", "ofac", "aml", "bsa"],
    }
    for key, terms in mapping.items():
        if key in focus_norm:
            return list(dict.fromkeys(terms + [focus_norm]))
    return [focus_norm]


# ============================================================
# BUILD CFPB VOCABULARY
# ============================================================
def _cfpb_vocab(df: pd.DataFrame) -> list[str]:
    vocab = []
    for col in ["issue", "product", "company"]:
        if col in df.columns:
            vocab.extend(df[col].dropna().astype(str).tolist())
    return [_norm(v) for v in vocab if v]


# ============================================================
# TF-IDF MAPPING WITH GENERIC EXCLUSIONS
# ============================================================
def _map_to_cfpb_phrase(focus: str, complaints_df: pd.DataFrame) -> str:
    focus_norm = _norm(focus)
    if not focus_norm or complaints_df.empty:
        return focus_norm

    vocab = _cfpb_vocab(complaints_df)
    if not vocab:
        return focus_norm

    # Remove generic labels from vocab
    filtered_vocab = [
        v for v in vocab
        if v not in EXCLUDE_GENERIC_LABELS
    ]
    if not filtered_vocab:
        filtered_vocab = vocab  # fallback

    # Direct substring match first
    direct_hits = [v for v in filtered_vocab if focus_norm in v]
    if direct_hits:
        return direct_hits[0]

    # TF-IDF similarity ranking
    try:
        documents = filtered_vocab + [focus_norm]
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(documents)

        sims = (tfidf[:-1] @ tfidf[-1].T).toarray().ravel()

        ranked = sorted(
            list(enumerate(sims)),
            key=lambda x: x[1],
            reverse=True
        )

        for idx, score in ranked:
            candidate = filtered_vocab[idx]
            if score >= 0.05:  # threshold
                return candidate
    except Exception:
        pass

    # If no mapping found, return the normalized focus phrase to avoid generic fallbacks
    return focus_norm


# ============================================================
# ORIGINAL infer_focus_phrase (unchanged)
# ============================================================
def infer_focus_phrase(
    query_sentence: str,
    keyword_input: str = "",
    candidate_terms: Optional[List[str]] = None,
    articles_df: Optional[pd.DataFrame] = None,
) -> str:

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

    candidates = []
    max_len = min(4, len(tokens))

    for n in range(max_len, 1, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i+n])
            near_tail = 1 if i >= len(tokens) - n - 1 else 0
            candidates.append((phrase, near_tail))

    for idx, tok in enumerate(tokens):
        near_tail = 1 if idx >= len(tokens) - 2 else 0
        candidates.append((tok, near_tail))

    def score(entry):
        phrase, tail = entry
        score = len(phrase) * 0.1 + len(phrase.split()) + tail * 2
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
        return max(candidates, key=score)[0].strip()

    if candidate_terms:
        return str(candidate_terms[0])

    return query_sentence


# ============================================================
# BUILD STATE HEATMAP WITH INTELLIGENT CFPB MATCHING
# ============================================================
def build_state_heatmap_data(complaints_df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if not keyword.strip():
        return pd.DataFrame()

    keyword_norm = _norm(keyword)

    def _demo_heatmap(label: str) -> pd.DataFrame:
        rng = np.random.default_rng()
        demo_states = rng.choice(list(STATE_COORDS.keys()), size=5, replace=False)
        demo_counts = rng.integers(1, 6, size=len(demo_states))
        df_demo = pd.DataFrame({"state": demo_states, "mentions": demo_counts})
        df_demo["latitude"] = df_demo["state"].map(lambda s: STATE_COORDS[s][0])
        df_demo["longitude"] = df_demo["state"].map(lambda s: STATE_COORDS[s][1])
        df_demo.attrs["heatmap_label"] = label
        df_demo.attrs["fallback_used"] = True
        df_demo.attrs["demo_generated"] = True
        return df_demo

    if complaints_df.empty:
        return _demo_heatmap(keyword_norm)

    mapped_phrase = _map_to_cfpb_phrase(keyword_norm, complaints_df)
    used_token_fallback = False
    search_terms = [t for t in _topic_synonyms(mapped_phrase) if len(t) > 3]

    df = complaints_df.copy()
    search_cols = [c for c in ["issue", "product", "company", "domain_label"] if c in df.columns]

    # First try strict word-boundary matches for precision
    mask = pd.Series(False, index=df.index)
    for term in search_terms:
        term_pattern = rf"\\b{re.escape(term)}\\b"
        for col in search_cols:
            mask |= df[col].astype(str).str.contains(term_pattern, case=False, na=False, regex=True)

    # If no rows matched (too strict), fall back to substring contains to avoid empty maps
    if not mask.any():
        for term in search_terms:
            for col in search_cols:
                mask |= df[col].astype(str).str.contains(term, case=False, na=False)

    filtered = df[mask]

    # If still empty, try token-level fallback matches on the keyword to avoid generic global fallback
    if filtered.empty:
        tokens = [t for t in keyword_norm.split() if len(t) > 3]
        token_mask = pd.Series(False, index=df.index)
        for tok in tokens:
            tok_pattern = rf"\\b{re.escape(tok)}\\b"
            for col in search_cols:
                token_mask |= df[col].astype(str).str.contains(tok_pattern, case=False, na=False, regex=True)

        if not token_mask.any():
            for tok in tokens:
                for col in search_cols:
                    token_mask |= df[col].astype(str).str.contains(tok, case=False, na=False)
        filtered = df[token_mask]
        if not filtered.empty and tokens:
            mapped_phrase = tokens[0]
            used_token_fallback = True

    if filtered.empty:
        return _demo_heatmap(keyword_norm)

    filtered["state"] = filtered["state"].astype(str).str.upper()

    state_counts = (
        filtered[filtered["state"].isin(STATE_COORDS.keys())]
        .groupby("state")
        .size()
        .reset_index(name="mentions")
    )

    if state_counts.empty:
        return _demo_heatmap(keyword_norm)

    state_counts["latitude"] = state_counts["state"].map(lambda s: STATE_COORDS[s][0])
    state_counts["longitude"] = state_counts["state"].map(lambda s: STATE_COORDS[s][1])

    state_counts.attrs["heatmap_label"] = mapped_phrase
    state_counts.attrs["fallback_used"] = (mapped_phrase != keyword_norm) or used_token_fallback
    state_counts.attrs["demo_generated"] = False

    return state_counts
