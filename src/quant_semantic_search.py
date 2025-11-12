"""
quant_semantic_search.py
-------------------------------------------------------
Detects and ranks *quantitative* articles and extracts chart-ready facts.

This module adds a second semantic layer that focuses specifically on
quantitative signals in articles (percentages, currency, counts, YoY changes).
It returns:
  1) A ranked subset of quantitatively rich, query-relevant articles
  2) A normalized "facts" dataframe with parsed numbers and context
  3) Visualization suggestions that reference fields guaranteed to exist

Expected input columns in `articles_df`:
  - id (int/str)                 optional but recommended
  - title (str)                  recommended
  - content (str)                required
  - url (str)                    optional
  - date (str/datetime)          optional
  - source (str)                 optional

Public functions:
  - search_quantitative_articles(articles_df, user_query, top_k=25, min_quant_score=1.5)
  - suggest_visuals_from_facts(facts_df, user_query)
"""

from __future__ import annotations

import re
import math
import json
import itertools
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# Optional semantic embeddings
# -----------------------------
_EMBEDDER = None
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMBEDDER = SentenceTransformer("BAAI/bge-small-en-v1.5")
except Exception:
    _EMBEDDER = None


# =========================
# Regex Patterns
# =========================
PCT_RE = re.compile(r"(?P<val>\d{1,3}(?:\.\d{1,2})?)\s?%", re.IGNORECASE)

# ✅ FIXED CURRENCY REGEX — full valid pattern
CURR_RE = re.compile(
    r"(?P<cur>\$|USD|US\$)\s?"
    r"(?P<num>\d{1,3}(?:[,\d]{3})*(?:\.\d{1,2})?)\s?"
    r"(?P<suf>bn|billion|mm|m|million|k|thousand)?",
    re.IGNORECASE,
)

KMB_RE = re.compile(
    r"(?P<num>\d{1,3}(?:[,\d]{3})*(?:\.\d{1,2})?)\s?(?P<suf>bn|billion|mm|m|million|k|thousand)\b",
    re.IGNORECASE,
)

# large raw integers
PLAIN_NUM_RE = re.compile(r"\b(?P<num>\d{4,})\b")

# year
YEAR_RE = re.compile(r"\b(20[0-4]\d|19\d{2})\b")

# quarter markers
QTR_RE = re.compile(r"\b(Q[1-4])\s*(20[0-4]\d)\b", re.IGNORECASE)

# change direction verbs
CHANGE_VERBS_RE = re.compile(
    r"\b(increase(?:d)?|decrease(?:d)?|rose|declined|grew|fell|drop(?:ped)?|"
    r"surged|spiked|plunged|improved|worsened)\b",
    re.IGNORECASE,
)

# time context words
TIME_WORDS_RE = re.compile(
    r"\b(YoY|MoM|quarter|monthly|annual|year(?:ly)?|trend|since|compared to)\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+")

US_STATE_ABBR = {
    "AL","AK","AZ","AR","CA","CO","CT","DC","DE","FL","GA","HI","IA","ID","IL","IN","KS","KY",
    "LA","MA","MD","ME","MI","MN","MO","MS","MT","NC","ND","NE","NH","NJ","NM","NV","NY","OH",
    "OK","OR","PA","PR","RI","SC","SD","TN","TX","UT","VA","VT","WA","WI","WV","WY"
}


# =========================
# Utility Functions
# =========================
def _to_year(dt: Any) -> Optional[int]:
    if pd.isna(dt):
        return None
    if isinstance(dt, (pd.Timestamp, datetime)):
        return int(dt.year)
    try:
        d = pd.to_datetime(dt, errors="coerce")
        return int(d.year) if not pd.isna(d) else None
    except Exception:
        return None


def _safe_str(x: Any) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def _normalize_num_with_suffix(num_str: str, suf: Optional[str]) -> float:
    base = float(num_str.replace(",", ""))
    if not suf:
        return base
    s = suf.lower()
    if s in ("bn", "billion", "b"):
        return base * 1_000_000_000
    if s in ("mm", "m", "million"):
        return base * 1_000_000
    if s in ("k", "thousand"):
        return base * 1_000
    return base


def _tokenize_lower(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _keyword_relevance(text: str, query: str) -> float:
    tset = set(_tokenize_lower(text))
    qset = set(_tokenize_lower(query))
    if not tset or not qset:
        return 0.0
    inter = len(tset & qset)
    return inter / (len(qset) ** 0.5)


def _embed_relevance(text: str, query: str) -> float:
    if _EMBEDDER is None:
        return _keyword_relevance(text, query)
    try:
        vq = _EMBEDDER.encode([query], normalize_embeddings=True)
        vt = _EMBEDDER.encode([text], normalize_embeddings=True)
        sim = float(np.dot(vq, vt.T)[0][0])
        return (sim + 1.0) / 2.0
    except Exception:
        return _keyword_relevance(text, query)


# =========================
# Extraction Logic
# =========================
def _split_sentences(text: str) -> List[str]:
    text = text.replace("\n", " ").replace("\r", " ")
    return [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]


def _extract_facts_from_sentence(sent: str) -> List[Dict[str, Any]]:
    facts = []
    lowered = sent.lower()

    direction = None
    m_dir = CHANGE_VERBS_RE.search(sent)
    if m_dir:
        dv = m_dir.group(1).lower()
        if dv in ("increase", "increased", "rose", "grew", "surged", "spiked", "improved"):
            direction = "increase"
        elif dv in ("decrease", "decreased", "declined", "fell", "drop", "dropped", "plunged", "worsened"):
            direction = "decrease"
        else:
            direction = "change"

    has_timeword = bool(TIME_WORDS_RE.search(sent))
    years = [int(y) for y in YEAR_RE.findall(sent)]
    year = years[0] if years else None

    # percentages
    for m in PCT_RE.finditer(sent):
        facts.append({
            "metric": "percentage",
            "value": float(m.group("val")),
            "unit": "pct",
            "direction": direction,
            "year": year,
            "time_flag": has_timeword,
            "context": sent,
        })

    # currency values
    for m in CURR_RE.finditer(sent):
        amount = _normalize_num_with_suffix(m.group("num"), m.group("suf"))
        facts.append({
            "metric": "amount",
            "value": amount,
            "unit": "usd",
            "direction": direction,
            "year": year,
            "time_flag": has_timeword,
            "context": sent,
        })

    # K/M/B suffix numbers
    for m in KMB_RE.finditer(sent):
        amount = _normalize_num_with_suffix(m.group("num"), m.group("suf"))
        facts.append({
            "metric": "amount",
            "value": amount,
            "unit": "unit",
            "direction": direction,
            "year": year,
            "time_flag": has_timeword,
            "context": sent,
        })

    # large integers
    for m in PLAIN_NUM_RE.finditer(sent):
        try:
            v = float(m.group("num").replace(",", ""))
        except Exception:
            continue
        if year and abs(v - year) < 1e-6:
            continue
        facts.append({
            "metric": "count",
            "value": v,
            "unit": "count",
            "direction": direction,
            "year": year,
            "time_flag": has_timeword,
            "context": sent,
        })

    # quarter markers
    for m in QTR_RE.finditer(sent):
        facts.append({
            "metric": "quarter_marker",
            "value": {"qtr": m.group(1).upper(), "year": int(m.group(2))},
            "unit": "marker",
            "direction": direction,
            "year": int(m.group(2)),
            "time_flag": True,
            "context": sent,
        })

    # state mentions
    states = [s for s in US_STATE_ABBR if re.search(rf"\b{s}\b", sent)]
    if states:
        facts.append({
            "metric": "state_mentions",
            "value": states,
            "unit": "list",
            "direction": direction,
            "year": year,
            "time_flag": has_timeword,
            "context": sent,
        })

    return facts


def extract_quant_facts_from_text(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []
    sentences = _split_sentences(text)
    facts = list(itertools.chain.from_iterable(_extract_facts_from_sentence(s) for s in sentences))
    cleaned = []

    for f in facts:
        if f["metric"] == "count" and f.get("year") and abs(f["value"] - f["year"]) < 1e-6:
            continue
        cleaned.append(f)
    return cleaned


# =========================
# Scoring & Ranking
# =========================
def score_quant_signals(facts: List[Dict[str, Any]]) -> float:
    if not facts:
        return 0.0

    w = {
        "percentage": 0.9,
        "amount": 1.0,
        "count": 0.6,
        "state_mentions": 0.3,
        "quarter_marker": 0.2,
    }

    score = 0.0
    metrics_present = set()
    time_bonus = 0.0
    dir_bonus = 0.0

    for f in facts:
        m = f.get("metric")
        metrics_present.add(m)
        score += w.get(m, 0.1)
        if f.get("time_flag"):
            time_bonus += 0.05
        if f.get("direction"):
            dir_bonus += 0.05

    diversity_bonus = 0.15 * len(metrics_present)
    return score + min(time_bonus, 0.4) + min(dir_bonus, 0.4) + diversity_bonus


def _normalize_articles_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        df["id"] = np.arange(1, len(df) + 1)
    if "title" not in df.columns:
        df["title"] = ""
    if "url" not in df.columns:
        df["url"] = ""
    if "source" not in df.columns:
        df["source"] = ""
    if "content" not in df.columns:
        df["content"] = ""
    if "date" not in df.columns:
        df["date"] = None

    df["title"] = df["title"].astype(str)
    df["url"] = df["url"].astype(str)
    df["source"] = df["source"].astype(str)
    df["content"] = df["content"].astype(str)
    return df


# =========================
# Main Search Function
# =========================
def search_quantitative_articles(
    articles_df: pd.DataFrame,
    user_query: str,
    top_k: int = 25,
    min_quant_score: float = 1.5,
) -> Dict[str, Any]:

    df = _normalize_articles_df(articles_df)
    if df.empty:
        return {"articles": pd.DataFrame(), "facts": pd.DataFrame(), "visual_suggestions": []}

    rows = []
    all_facts = []

    for _, row in df.iterrows():
        text = _safe_str(row.get("content", ""))
        title = _safe_str(row.get("title", ""))

        if len(text) < 150:
            continue

        facts = extract_quant_facts_from_text(text)
        qscore = score_quant_signals(facts)
        if qscore < min_quant_score:
            continue

        rel = _embed_relevance(f"{title}\n\n{text[:2000]}", user_query)

        rows.append({
            "id": row["id"],
            "title": title,
            "url": _safe_str(row.get("url", "")),
            "source": _safe_str(row.get("source", "")),
            "date": row.get("date", None),
            "year": _to_year(row.get("date", None)),
            "quant_score": qscore,
            "relevance": rel,
        })

        for f in facts:
            all_facts.append({
                "article_id": row["id"],
                "title": title,
                "url": _safe_str(row.get("url", "")),
                "date": row.get("date", None),
                "year": _to_year(row.get("date", None)) if f.get("year") is None else f.get("year"),
                "metric": f.get("metric"),
                "value": f.get("value"),
                "unit": f.get("unit"),
                "direction": f.get("direction"),
                "topic": None,
                "context": f.get("context"),
                "source": _safe_str(row.get("source", "")),
            })

    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return {"articles": pd.DataFrame(), "facts": pd.DataFrame(), "visual_suggestions": []}

    ranked["score"] = 0.65 * ranked["relevance"] + 0.35 * np.tanh(ranked["quant_score"] / 3.0)
    ranked = ranked.sort_values(["score", "quant_score", "relevance"], ascending=False).head(top_k)

    facts_df = pd.DataFrame(all_facts)
    if not facts_df.empty:
        keep_ids = set(ranked["id"].tolist())
        facts_df = facts_df[facts_df["article_id"].isin(keep_ids)].copy()

        facts_df["year"] = pd.to_numeric(facts_df["year"], errors="coerce").astype("Int64")

        mask_states = facts_df["metric"] == "state_mentions"
        if mask_states.any():
            exploded = facts_df[mask_states].explode("value")
            exploded = exploded.rename(columns={"value": "state"})
            exploded["metric"] = "state_mention"
            exploded["value"] = 1.0
            exploded["unit"] = "flag"
            facts_df = pd.concat([facts_df[~mask_states], exploded], ignore_index=True)
    else:
        facts_df = pd.DataFrame(columns=[
            "article_id","title","url","date","year","metric","value","unit","direction","topic","context","source"
        ])

    visuals = suggest_visuals_from_facts(facts_df, user_query)
    return {"articles": ranked, "facts": facts_df, "visual_suggestions": visuals}


# =========================
# Visualization Suggestion Logic
# =========================
def _has_timeseries(facts_df: pd.DataFrame) -> bool:
    return not facts_df.empty and facts_df["year"].notna().sum() > 3


def _has_states(facts_df: pd.DataFrame) -> bool:
    return "state" in facts_df.columns


def _has_percentages(facts_df: pd.DataFrame) -> bool:
    return (facts_df["metric"] == "percentage").any()


def _has_amounts(facts_df: pd.DataFrame) -> bool:
    return (facts_df["metric"] == "amount").any()


def suggest_visuals_from_facts(facts_df: pd.DataFrame, user_query: str) -> List[Dict[str, Any]]:
    if facts_df.empty:
        return []

    visuals = []

    # ---------- 1. Time-series visual ----------
    if _has_timeseries(facts_df):
        visuals.append({
            "dataset": "facts",
            "type": "line",
            "x": "year",
            "y": "value",
            "color": "metric",
            "title": "Year-over-Year Quantitative Signals",
        })

    # ---------- 2. State Frequency Heatmap ----------
    if _has_states(facts_df):
        visuals.append({
            "dataset": "facts",
            "type": "heatmap",
            "x": "state",
            "y": "metric",
            "value": "value",
            "title": "Geographic Distribution of Quantitative Mentions",
        })

    # ---------- 3. Percentages Summary ----------
    if _has_percentages(facts_df):
        pct_df = facts_df[facts_df["metric"] == "percentage"]
        if not pct_df.empty:
            visuals.append({
                "dataset": "facts",
                "type": "bar",
                "x": "title",
                "y": "value",
                "color": "direction",
                "title": "Percentage-Based Signals by Article",
            })

    # ---------- 4. Amounts (financial) ----------
    if _has_amounts(facts_df):
        visuals.append({
            "dataset": "facts",
            "type": "bar",
            "x": "title",
            "y": "value",
            "color": "metric",
            "title": "Financial Magnitudes Mentioned Across Articles",
        })

    if not visuals:
        return [{"message": "No quantitative signals strong enough for charting were found."}]

    return visuals
