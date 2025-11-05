"""
cfpb_data_loader.py
---------------------------------
Fetch CFPB Consumer Complaint data (5-year window by default),
semantically classify complaints into fraud/compliance domains,
and upsert filtered rows into Supabase.

Run:
  uv run python src/cfpb_data_loader.py
"""

from __future__ import annotations

import os
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any

from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


# =========================
# DOMAIN CONTEXTS (Provided)
# =========================
DOMAIN_KEYWORDS: Dict[str, Dict[str, str]] = {
    "fraud_detection": {
        "label": "Fraud Detection",
        "context": (
            "Fraud detection involves identifying, preventing, and responding to deceptive or illegal financial activity. "
            "It includes scams, phishing, identity theft, wire fraud, fraudulent transactions, account takeovers, business "
            "email compromise, synthetic identity fraud, and social engineering tactics. Focus is detecting suspicious "
            "behavior patterns and protecting consumers and institutions from financial loss."
        ),
    },
    "aml_compliance": {
        "label": "AML Compliance",
        "context": (
            "Anti-money laundering (AML) compliance covers processes and regulations to detect and prevent illicit funds. "
            "Includes BSA and FinCEN guidelines, KYC, transaction monitoring for suspicious activity, SARs, sanctions screening, "
            "and risk management. AML programs mitigate money laundering and terrorism financing risks."
        ),
    },
    "financial_crime": {
        "label": "Financial Crime",
        "context": (
            "Financial crime includes bribery, corruption, embezzlement, fraud by fiduciaries, money laundering, insider trading, "
            "and investment scams. It exploits financial systems to gain illicit profits or conceal illegal activities "
            "through complex transactions and shell entities."
        ),
    },
    "regulatory": {
        "label": "Regulatory Enforcement",
        "context": (
            "Regulatory enforcement covers oversight, supervision, penalties, and guidance shaping organizational conduct. "
            "Includes audits, enforcement actions, settlements, penalties, and governance/oversight policy changes addressing "
            "AML deficiencies, consumer protection, and emerging fraud threats."
        ),
    },
    "technology_context": {
        "label": "Technology in Fraud Prevention",
        "context": (
            "Technology combats misconduct using AI, machine learning, analytics, and automation. Real-time monitoring, "
            "pattern/anomaly detection, and NLP-driven alert triage support detection of phishing, digital impersonation, "
            "and evolving fraud tactics."
        ),
    },
    "cyber_fraud": {
        "label": "Cyber Fraud and Impersonation",
        "context": (
            "Cyber fraud uses digital deception to steal data or assets: phishing emails, fake websites, social media "
            "impersonation, malware credential theft, and ransomware. Prevention emphasizes authentication controls, "
            "user awareness, and cybersecurity integrated with financial monitoring."
        ),
    },
    "consumer_protection": {
        "label": "Consumer Protection and Awareness",
        "context": (
            "Consumer protection defends individuals from deceptive practices: online scams, fraudulent investments, fake charities, "
            "and impersonation calls. Emphasizes transparency, financial literacy, and proactive fraud alerts to help identify "
            "red flags and reduce harm."
        ),
    },
}


# =========================
# Setup
# =========================
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

CFPB_API_BASE = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
TARGET_TABLE = "cfpb_complaints"


# =========================
# Helpers
# =========================
def utc_date_str(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).date().isoformat()


def default_date_window(months: int) -> tuple[str, str]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(months * 30.4375))
    return utc_date_str(start), utc_date_str(end)


def robust_get(url: str, params: Dict[str, Any], tries: int = 3, timeout: int = 30) -> Dict[str, Any]:
    last = None
    for attempt in range(1, tries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            last = e
            print(f"⚠️ GET {url} failed (attempt {attempt}/{tries}): {e}")
            time.sleep(min(5 * attempt, 15))
    raise RuntimeError(f"Failed to fetch after {tries} attempts: {url} (last: {last})")


def parse_hits(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not payload:
        return rows
    if isinstance(payload.get("hits"), dict) and isinstance(payload["hits"].get("hits"), list):
        hits = payload["hits"]["hits"]
        for h in hits:
            src = h.get("_source", h)
            rows.append(src)
    elif isinstance(payload.get("hits"), list):
        for h in payload["hits"]:
            src = h.get("_source", h)
            rows.append(src)
    return rows


def normalize_df(raw: List[Dict[str, Any]]) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()

    df = pd.json_normalize(raw)
    if df.empty:
        return df

    possible_text_cols = [
        c for c in df.columns if any(k in c.lower() for k in ["narrative", "complaint", "description", "what_happened"])
    ]
    text_col = possible_text_cols[0] if possible_text_cols else None
    if text_col and text_col != "consumer_complaint_narrative":
        df = df.rename(columns={text_col: "consumer_complaint_narrative"})
    elif "consumer_complaint_narrative" not in df.columns:
        df["consumer_complaint_narrative"] = ""

    # Try to normalize date
    if "date_received" in df.columns:
        df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")

    # ID column
    if "complaint_id" not in df.columns:
        for c in df.columns:
            if "complaint" in c.lower() and "id" in c.lower():
                df = df.rename(columns={c: "complaint_id"})
                break
    if "complaint_id" not in df.columns:
        df["complaint_id"] = range(1, len(df) + 1)

    return df


def fetch_complaints(date_min: str, date_max: str, page_size: int, max_pages: int) -> pd.DataFrame:
    total_rows: List[Dict[str, Any]] = []
    offset = 0
    print(f"⬇️ Fetching complaints {date_min} → {date_max} …")

    for page in range(1, max_pages + 1):
        params = {
            "size": page_size,
            "from": offset,
            "date_received_min": date_min,
            "date_received_max": date_max,
        }
        payload = robust_get(CFPB_API_BASE, params)
        rows = parse_hits(payload)
        if not rows:
            break
        total_rows.extend(rows)
        print(f"✅ Retrieved page {page} with {len(rows)} complaints.")
        offset += page_size
        time.sleep(0.3)

    print(f"📊 Total complaints fetched: {len(total_rows)}")
    return normalize_df(total_rows)


# =========================
# Classification
# =========================
def classify_complaints(df: pd.DataFrame, threshold: float = 0.15) -> pd.DataFrame:
    if df.empty:
        print("⚠️ No complaint data available for classification.")
        return df

    # Ensure text column exists
    if "consumer_complaint_narrative" not in df.columns:
        print("⚠️ 'consumer_complaint_narrative' column missing, detecting possible alternative.")
        text_col = None
        for c in df.columns:
            if any(k in c.lower() for k in ["narrative", "complaint", "description", "what_happened"]):
                text_col = c
                break
        if text_col:
            df = df.rename(columns={text_col: "consumer_complaint_narrative"})
        else:
            print("❌ No text column found in CFPB dataset.")
            return pd.DataFrame()

    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].fillna("").astype(str)
    df = df[df["consumer_complaint_narrative"].str.split().str.len() >= 10]
    if df.empty:
        print("⚠️ All complaints were empty or too short.")
        return df

    print("🧠 Loading semantic embedder (BAAI/bge-large-en-v1.5)…")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    domain_keys = list(DOMAIN_KEYWORDS.keys())
    domain_labels = [DOMAIN_KEYWORDS[k]["label"] for k in domain_keys]
    domain_contexts = [DOMAIN_KEYWORDS[k]["context"] for k in domain_keys]
    domain_embs = model.encode(domain_contexts, convert_to_tensor=True, normalize_embeddings=True)

    tagged: List[Dict[str, Any]] = []
    iterator = tqdm(df.itertuples(index=False), total=len(df), desc="🔍 Semantic Classification")

    for row in iterator:
        row_dict = row._asdict()
        text = row_dict.get("consumer_complaint_narrative", "").strip()
        if not text:
            continue

        emb = model.encode(text, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(emb, domain_embs)[0].tolist()
        best_idx = int(max(range(len(sims)), key=sims.__getitem__))
        best_score = float(sims[best_idx])

        if best_score < threshold:
            low = text.lower()
            fraud_keywords = [
                "fraud", "scam", "identity", "phishing", "aml", "wire",
                "stolen", "unauthorized", "chargeback", "account takeover",
                "ransomware", "impersonation", "breach", "theft",
                "money laundering", "sanction"
            ]
            if any(k in low for k in fraud_keywords):
                best_score = threshold
            else:
                continue

        row_dict["domain_key"] = domain_keys[best_idx]
        row_dict["domain_label"] = domain_labels[best_idx]
        row_dict["similarity_score"] = round(best_score, 3)
        tagged.append(row_dict)

    filtered = pd.DataFrame(tagged)
    print(f"🧠 Retained {len(filtered)} complaints with domain classification.")
    return filtered


# =========================
# Upload to Supabase
# =========================
def upsert_supabase(df: pd.DataFrame) -> None:
    if df.empty:
        print("⚠️ No complaint data to upload.")
        return

    expected = {"complaint_id", "date_received", "consumer_complaint_narrative",
                "product", "issue", "state", "company",
                "domain_key", "domain_label", "similarity_score"}

    for m in expected - set(df.columns):
        df[m] = None

    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce").dt.strftime("%Y-%m-%d")
    records = df.to_dict(orient="records")

    batch_size = 1000
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        for attempt in range(2):
            try:
                supabase.table(TARGET_TABLE).upsert(chunk).execute()
                break
            except Exception as e:
                print(f"⚠️ Upload failed (attempt {attempt+1}): {e}")
                time.sleep(2)
        else:
            print("❌ Batch permanently failed.")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=80)
    args = parser.parse_args()

    date_min, date_max = default_date_window(args.months)
    print(f"📅 Date window: {date_min} → {date_max}")
    print(f"🎚️ Threshold: {args.threshold}")

    df_raw = fetch_complaints(date_min, date_max, args.page_size, args.max_pages)
    df_tagged = classify_complaints(df_raw, args.threshold)
    upsert_supabase(df_tagged)


if __name__ == "__main__":
    main()
