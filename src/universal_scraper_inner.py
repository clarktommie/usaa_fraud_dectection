import json
import os
import re
import time
import urllib.robotparser
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from requests import Response
from textblob import TextBlob
from supabase import Client, create_client

# ---------------------------
# Environment Setup
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
USER_AGENT = "Tommie-Clark-Fraud-Scraper/2.0 (+your-email@example.com)"
REQUEST_TIMEOUT = 25
PDF_MIN_WORDS = 40
MIN_ACCEPTABLE_WORDS = 25
PDF_FALLBACK_PREVIEW = 1500
BATCH_SIZE = 100
SUMMARY_SENTENCES = 3
MAX_ENTITIES = 5
MAX_TAGS = 5
DIGEST_LIMIT = 10
DIGEST_PATH = Path("data") / "latest_scrape_digest.md"
METRICS_PATH = Path("data") / "latest_scrape_metrics.json"
STOPWORDS = {
    "the",
    "and",
    "with",
    "from",
    "that",
    "this",
    "have",
    "will",
    "about",
    "into",
    "your",
    "they",
    "their",
    "been",
    "which",
    "because",
    "after",
    "before",
    "over",
    "such",
    "including",
    "between",
    "while",
    "where",
    "when",
    "press",
    "release",
    "news",
    "company",
    "business",
    "service",
    "services",
    "financial",
    "information",
    "industry",
    "customers",
    "report",
    "reports",
    "update",
    "updates",
    "fraud",
    "security",
    "compliance",
}
ENTITY_STOPWORDS = {"Inc", "LLC", "Corp", "Company", "Group", "US", "U.S.", "USA"}
NOISE_TAGS = {
    "press",
    "release",
    "fraud",
    "financial",
    "company",
    "business",
    "services",
    "security",
    "information",
    "report",
    "update",
    "industry",
    "global",
    "market",
    "customers",
}

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


# ---------------------------
# Helper Functions
# ---------------------------
def word_count(text: Optional[str]) -> int:
    """Return the number of words in a string-like value."""
    if not text:
        return 0
    return len(text.split())


def can_fetch(url: str) -> bool:
    """Check robots.txt permissions for the configured user-agent."""
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = f"{base}/robots.txt"
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        print(f"robots.txt check for {base}: {'✅ allowed' if allowed else '🚫 disallowed'}")
        return allowed
    except Exception as exc:
        print(f"⚠️ Could not read robots.txt for {base} — assuming allowed. ({exc})")
        return True


def get_html(url: str) -> Optional[Response]:
    """Fetch HTML (or PDF) content with shared headers."""
    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None


def extract_text_from_pdf(content: bytes) -> str:
    """Extract text safely from PDFs, with a latin-1 fallback."""
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
        if not text:
            print("⚠️ PDF had no extractable text.")
        return text
    except Exception as e:
        print(f"⚠️ PDF extraction error: {e} — fallback decoding...")
        try:
            decoded = content.decode("latin-1", errors="ignore")
            return decoded[:PDF_FALLBACK_PREVIEW]
        except Exception:
            return ""


def find_pdf_fallback(soup: BeautifulSoup, base_url: str, current_words: int) -> Optional[str]:
    """Scan anchor tags for the strongest PDF fallback."""
    return _find_pdf_fallback(soup, base_url, current_words)[0]


def _find_pdf_fallback(
    soup: BeautifulSoup, base_url: str, current_words: int
) -> Tuple[Optional[str], Optional[str]]:
    """Internal helper to surface best PDF text and its URL."""
    best_content: Optional[str] = None
    best_words = current_words
    best_url: Optional[str] = None

    for link in soup.find_all("a", href=lambda href: href and href.lower().endswith(".pdf")):
        pdf_url = urljoin(base_url, link["href"])
        print(f"📎 Found PDF: {pdf_url}")
        pdf_resp = get_html(pdf_url)
        if not pdf_resp:
            continue

        content_type = pdf_resp.headers.get("Content-Type", "")
        if "application/pdf" not in content_type and not pdf_url.endswith(".pdf"):
            continue

        pdf_text = extract_text_from_pdf(pdf_resp.content)
        pdf_words = word_count(pdf_text)
        if pdf_words > best_words:
            best_content = pdf_text
            best_words = pdf_words
            best_url = pdf_url

    return best_content, best_url


def parse_html(resp: Response, base_url: str) -> Dict[str, Optional[str]]:
    """Extract metadata and body content from an HTML response."""
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    main = (
        soup.select_one(
            "div.o-post-content, div.entry-content, div.article__body, article, main, div.col-sm-8"
        )
        or soup
    )
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    content = "\n\n".join(t for t in paragraphs if word_count(t) > 5).strip()

    if not content:
        alt = soup.select_one("div.m-text-block, section.o-feature, div.o-page-content") or soup
        alt_paragraphs = [p.get_text(" ", strip=True) for p in alt.find_all("p")]
        content = "\n\n".join(t for t in alt_paragraphs if word_count(t) > 5).strip()

    tag = soup.select_one("time[datetime], time, meta[property='article:published_time']")
    date = (
        tag.get("datetime")
        if tag and tag.has_attr("datetime")
        else tag.get("content")
        if tag and tag.has_attr("content")
        else tag.get_text(strip=True)
        if tag
        else None
    )

    by = soup.select_one(".author-name, .byline, meta[name='author']")
    author = (
        by.get("content")
        if by and by.has_attr("content")
        else by.get_text(" ", strip=True)
        if by
        else None
    )

    pdf_used = False
    fallback_pdf_url = None

    if word_count(content) < PDF_MIN_WORDS:
        pdf_text, pdf_url = _find_pdf_fallback(soup, base_url, word_count(content))
        if pdf_text:
            content = pdf_text
            pdf_used = True
            fallback_pdf_url = pdf_url

    return {
        "title": title,
        "content": content,
        "date": date,
        "author": author,
        "pdf_used": pdf_used,
        "fallback_pdf_url": fallback_pdf_url,
    }


def generate_summary(content: Optional[str], max_sentences: int = SUMMARY_SENTENCES) -> Optional[str]:
    """Create a short multi-sentence summary from the leading sentences."""
    if not content:
        return None

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
    if not sentences:
        return None

    summary = " ".join(sentences[:max_sentences]).strip()
    return summary or None


def analyze_sentiment(content: Optional[str]) -> Dict[str, Optional[float]]:
    """Return a light sentiment analysis payload using TextBlob polarity."""
    if not content:
        return {"label": None, "score": None}

    blob = TextBlob(content)
    polarity = round(blob.sentiment.polarity, 4)
    if polarity > 0.15:
        label = "positive"
    elif polarity < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "score": polarity}


def extract_entities(content: Optional[str], max_entities: int = MAX_ENTITIES) -> List[str]:
    """Pull proper-noun and currency entities using lightweight heuristics."""
    if not content:
        return []

    names = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", content)
    filtered_names = [
        name
        for name in names
        if name not in ENTITY_STOPWORDS and not any(stop in name for stop in ENTITY_STOPWORDS)
    ]
    currency = re.findall(r"(?:USD|US\$|\$)\s?\d[\d,]*(?:\.\d+)?", content)

    counts = Counter(filtered_names + currency)
    top = [name for name, _ in counts.most_common(max_entities)]
    return top


def generate_tags(content: Optional[str], max_tags: int = MAX_TAGS) -> List[str]:
    """Generate quick keyword tags from the document body."""
    if not content:
        return []

    tokens = re.findall(r"[A-Za-z]{4,}", content.lower())
    weighted = Counter(token for token in tokens if token not in STOPWORDS)
    if not weighted:
        return []

    primary = [
        (token, count)
        for token, count in weighted.most_common()
        if token not in NOISE_TAGS and count > 1
    ]

    selected: List[Tuple[str, int]] = []
    for token, count in primary:
        selected.append((token, count))
        if len(selected) >= max_tags:
            break

    if len(selected) < max_tags:
        for token, count in weighted.most_common():
            if token in NOISE_TAGS or any(token == word for word, _ in selected):
                continue
            selected.append((token, count))
            if len(selected) >= max_tags:
                break

    return [word for word, _ in selected[:max_tags]]


def enrich_content(content: Optional[str]) -> Dict[str, object]:
    """Bundle summary, tags, entities, and confidence helpers."""
    text = content or ""
    words = word_count(text)
    summary = generate_summary(text)
    sentiment = analyze_sentiment(text)
    entities = extract_entities(text)
    tags = generate_tags(text)

    return {
        "summary": summary,
        "sentiment": sentiment,
        "entities": entities,
        "tags": tags,
        "word_count": words,
        "snippet": text[:400].strip() or None,
    }


def build_metadata(
    url: str,
    resp: Response,
    parsed_payload: Dict[str, Optional[str]],
    robots_allowed: bool,
    fetch_duration: float,
    parse_duration: float,
) -> Dict[str, object]:
    """Construct a metadata payload for observability dashboards."""
    parsed_url = urlparse(url)
    timestamp = datetime.now(UTC).isoformat()
    word_total = word_count(parsed_payload.get("content"))
    pdf_used = bool(parsed_payload.get("pdf_used"))

    confidence = min(1.0, max(0.1, word_total / 500))
    metrics = {
        "word_count": word_total,
        "html_bytes": len(resp.content or b""),
        "fetch_duration_sec": round(fetch_duration, 4),
        "parse_duration_sec": round(parse_duration, 4),
        "pdf_used": pdf_used,
    }

    return {
        "source_domain": parsed_url.netloc,
        "fetched_at": timestamp,
        "robots_allowed": robots_allowed,
        "status_code": resp.status_code,
        "content_type": resp.headers.get("Content-Type"),
        "pdf_used": pdf_used,
        "fallback_pdf_url": parsed_payload.get("fallback_pdf_url"),
        "confidence": round(confidence, 3),
        "metrics": metrics,
    }


def write_digest_report(records: List[Dict[str, object]]) -> None:
    """Persist a concise markdown digest and companion metrics for stakeholders."""
    if not records:
        return

    DIGEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(UTC).isoformat()
    domain_counts = Counter(entry["domain"] for entry in records if entry.get("domain"))
    avg_sent: Optional[float] = None
    sent_scores = [
        entry["sentiment"]["score"]
        for entry in records
        if entry.get("sentiment") and entry["sentiment"].get("score") is not None
    ]
    if sent_scores:
        avg_sent = sum(sent_scores) / len(sent_scores)

    entity_counter = Counter()
    tag_counter = Counter()
    sentiment_mix = Counter()
    for entry in records:
        entity_counter.update(entry.get("entities") or [])
        tag_counter.update(entry.get("tags") or [])
        label = (entry.get("sentiment") or {}).get("label")
        if label:
            sentiment_mix.update([label])

    lines = [
        "# Universal Inner Scraper Digest",
        f"*Generated at {generated_at}*",
        "",
        "## Snapshot",
        f"- Total releases: **{len(records)}**",
        f"- Domains: {', '.join(f'{dom} ({cnt})' for dom, cnt in domain_counts.most_common(5)) or 'N/A'}",
        f"- Avg sentiment: {avg_sent:+.2f if avg_sent is not None else 'N/A'}",
        f"- Sentiment mix: {', '.join(f'{label} ({count})' for label, count in sentiment_mix.items()) or 'N/A'}",
        f"- Top entities: {', '.join(name for name, _ in entity_counter.most_common(MAX_ENTITIES)) or 'N/A'}",
        f"- Top tags: {', '.join(tag for tag, _ in tag_counter.most_common(MAX_TAGS)) or 'N/A'}",
        "",
        "## Highlights",
    ]

    featured = records[:DIGEST_LIMIT]
    for idx, entry in enumerate(featured, 1):
        sentiment = entry["sentiment"]
        sentiment_line = "N/A"
        if sentiment and sentiment.get("score") is not None:
            sentiment_line = f"{sentiment.get('label')} ({sentiment.get('score'):+.2f})"

        entities_line = ", ".join(entry.get("entities") or []) or "N/A"
        tags_line = ", ".join(entry.get("tags") or []) or "N/A"
        summary_text = entry.get("summary") or entry.get("snippet") or "No summary available."
        word_ct = entry.get("word_count") or "N/A"
        domain = entry.get("domain") or "Unknown"

        lines.extend(
            [
                f"### {idx}. {entry['title'] or 'Untitled Release'}",
                f"- Source: **{domain}**",
                f"- Sentiment: {sentiment_line}",
                f"- Word Count: {word_ct}",
                f"- Entities: {entities_line}",
                f"- Tags: {tags_line}",
                f"- URL: {entry['url']}",
                "",
                "Summary:",
                f"> {summary_text}",
                "",
                "---",
                "",
            ]
        )

    DIGEST_PATH.write_text("\n".join(lines), encoding="utf-8")

    metrics_payload = {
        "generated_at": generated_at,
        "total_records": len(records),
        "domains": dict(domain_counts),
        "avg_sentiment": avg_sent,
        "top_entities": entity_counter.most_common(MAX_ENTITIES),
        "top_tags": tag_counter.most_common(MAX_TAGS),
        "sentiment_mix": dict(sentiment_mix),
    }
    METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")


# ---------------------------
# Fetch Pending
# ---------------------------
def fetch_pending_articles(batch_size: int = 1000) -> List[Dict[str, Optional[str]]]:
    """Retrieve all pending records from press_releases."""
    print("📥 Fetching pending records...")
    all_data: List[Dict[str, Optional[str]]] = []
    start = 0

    while True:
        resp = (
            supabase.table("press_releases")
            .select("id, url, title, author, date")
            .eq("status", "pending")
            .range(start, start + batch_size - 1)
            .execute()
        )
        data = resp.data or []
        all_data.extend(data)
        print(f"➡️ Retrieved rows {start}-{start+batch_size-1} (total {len(all_data)})")

        if len(data) < batch_size:
            break

        start += batch_size
        time.sleep(0.2)

    print(f"✅ Total pending: {len(all_data)}")
    return all_data


def build_clean_record(
    row: Dict[str, Optional[str]],
    parsed: Dict[str, Optional[str]],
    url: str,
    enrichment: Dict[str, object],
    metadata: Dict[str, object],
) -> Dict[str, object]:
    """Assemble the clean payload for press_releases_clean."""
    timestamp = datetime.now(UTC).isoformat()
    return {
        "id": row["id"],
        "title": parsed.get("title") or row.get("title"),
        "url": url,
        "date": parsed.get("date") or row.get("date"),
        "author": parsed.get("author") or row.get("author"),
        "content": parsed.get("content"),
        "status": "complete",
        "summary": enrichment["summary"],
        "keywords": enrichment["tags"],
        "entities": enrichment["entities"],
        "sentiment": enrichment["sentiment"],
        "snippet": enrichment["snippet"],
        "metadata": metadata,
        "updated_at": timestamp,
        "scraped_at": timestamp,
    }


def flush_batches(
    cleaned_batch: List[Dict[str, object]], complete_ids: List[Dict[str, object]]
) -> None:
    """Write accumulated batches to Supabase."""
    if not cleaned_batch:
        return

    print("\n📦 Writing batch to Supabase...")
    supabase.table("press_releases_clean").upsert(cleaned_batch).execute()
    supabase.table("press_releases").upsert(complete_ids).execute()
    cleaned_batch.clear()
    complete_ids.clear()
    time.sleep(0.5)


def mark_failed_records(failed_ids: List[Dict[str, object]]) -> None:
    """Persist failed record IDs to Supabase."""
    if not failed_ids:
        return

    print("\n⚠️ Marking failed IDs...")
    supabase.table("press_releases").upsert(failed_ids).execute()


def process_pending_records() -> None:
    """Main orchestration routine for scraping and persisting results."""
    pending = fetch_pending_articles()
    if not pending:
        print("⚠️ No pending records — exiting.")
        return

    cleaned_batch: List[Dict[str, object]] = []
    complete_ids: List[Dict[str, object]] = []
    failed_ids: List[Dict[str, object]] = []
    digest_records: List[Dict[str, object]] = []

    for row in pending:
        url = row.get("url")
        if not url:
            failed_ids.append({"id": row["id"], "status": "failed"})
            print("❌ missing URL")
            continue

        print(f"\n🔍 Scraping: {url}")

        robots_allowed = can_fetch(url)
        if not robots_allowed:
            failed_ids.append({"id": row["id"], "status": "failed"})
            print("❌ robots.txt disallowed")
            continue

        fetch_started = time.perf_counter()
        resp = get_html(url)
        fetch_duration = time.perf_counter() - fetch_started
        if not resp:
            failed_ids.append({"id": row["id"], "status": "failed"})
            print("❌ failed to fetch HTML")
            continue

        parse_started = time.perf_counter()
        parsed = parse_html(resp, url)
        parse_duration = time.perf_counter() - parse_started
        if word_count(parsed.get("content")) < MIN_ACCEPTABLE_WORDS:
            failed_ids.append({"id": row["id"], "status": "failed"})
            print("❌ empty or too short")
            continue

        enrichment = enrich_content(parsed.get("content"))
        metadata = build_metadata(url, resp, parsed, robots_allowed, fetch_duration, parse_duration)
        cleaned_record = build_clean_record(row, parsed, url, enrichment, metadata)

        cleaned_batch.append(cleaned_record)
        complete_ids.append({"id": row["id"], "status": "complete"})
        digest_records.append(
            {
                "title": cleaned_record["title"],
                "url": url,
                "summary": enrichment["summary"],
                "sentiment": enrichment["sentiment"],
                "tags": enrichment["tags"],
                "domain": metadata.get("source_domain"),
                "word_count": metadata.get("metrics", {}).get("word_count"),
                "entities": enrichment["entities"],
            }
        )

        if len(cleaned_batch) >= BATCH_SIZE:
            flush_batches(cleaned_batch, complete_ids)

    if cleaned_batch:
        flush_batches(cleaned_batch, complete_ids)

    mark_failed_records(failed_ids)
    write_digest_report(digest_records)
    print("\n🏁 Universal Inner Scraper Complete (Batch Safe Mode).")


# ---------------------------
# Main Scraper
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Universal Inner Scraper (BATCH SAFE MODE)...\n")
    process_pending_records()
