import os
import time
from datetime import datetime, UTC
from io import BytesIO
from urllib.parse import urlparse, urljoin
import urllib.robotparser
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from supabase import create_client
from dotenv import load_dotenv

# ---------------------------
# Environment Setup
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_AGENT = "Tommie-Clark-Fraud-Scraper/2.0 (+your-email@example.com)"

# ---------------------------
# Helper Functions
# ---------------------------
def can_fetch(url):
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    robots_url = f"{base}/robots.txt"
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        print(f"robots.txt check for {base}: {'✅ allowed' if allowed else '🚫 disallowed'}")
        return allowed
    except Exception:
        print(f"⚠️ Could not read robots.txt for {base} — assuming allowed.")
        return True


def get_html(url):
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None


def extract_text_from_pdf(content):
    """Extract text from PDF content safely."""
    try:
        reader = PdfReader(BytesIO(content))
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
        if not text:
            print("⚠️ PDF has no extractable text (may be scanned or malformed).")
        return text
    except Exception as e:
        # Try a fallback for malformed PDFs
        print(f"⚠️ PDF extraction error: {e} — trying fallback decode...")
        try:
            # fallback: decode visible text bytes (works for some broken PDFs)
            decoded = content.decode("latin-1", errors="ignore")
            snippet = decoded[:1000]
            if "Copyright" in snippet or "United States" in snippet:
                print("🩹 Partial recovery via fallback decode.")
                return snippet
        except Exception:
            pass
        return ""



def parse_html(resp, base_url):
    """Extract title, content, author, and date — with fallback to embedded PDF."""
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Standard article text
    main = soup.select_one("div.entry-content, div.article__body, article, main, div.col-sm-8") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    content = "\n\n".join([t for t in paragraphs if len(t.split()) > 5]).strip()

    # Legacy fallback
    if not content:
        legacy_blocks = soup.find_all(["font", "td", "pre", "div"], text=True)
        content = "\n\n".join(
            [b.get_text(" ", strip=True) for b in legacy_blocks if len(b.get_text(strip=True).split()) > 5]
        ).strip()

    # Date extraction
    tag = soup.select_one("time[datetime], time, meta[property='article:published_time']")
    date = tag.get("datetime") if tag and tag.has_attr("datetime") else (
        tag.get("content") if tag and tag.has_attr("content") else (
            tag.get_text(strip=True) if tag else None
        )
    )

    # Author extraction
    by = soup.select_one(".author-name, .byline, meta[name='author']")
    author = by.get("content") if by and by.has_attr("content") else (
        by.get_text(" ", strip=True) if by else None
    )

    # PDF fallback if short or empty
    if not content or len(content.split()) < 40:
        pdf_link = soup.find("a", href=lambda h: h and h.lower().endswith(".pdf"))
        if pdf_link:
            pdf_url = urljoin(base_url, pdf_link["href"])
            print(f"📎 Found embedded PDF: {pdf_url}")
            pdf_resp = get_html(pdf_url)
            if pdf_resp and ("application/pdf" in pdf_resp.headers.get("Content-Type", "") or pdf_url.endswith(".pdf")):
                pdf_text = extract_text_from_pdf(pdf_resp.content)
                if len(pdf_text.split()) > len(content.split()):
                    content = pdf_text
    # After the current pdf_link check:
    if not content or len(content.split()) < 40:
        pdf_links = soup.find_all("a", href=lambda h: h and h.lower().endswith(".pdf"))
        for link in pdf_links:
            pdf_url = urljoin(base_url, link["href"])
            print(f"📎 Found embedded PDF: {pdf_url}")
            pdf_resp = get_html(pdf_url)
            if pdf_resp and ("application/pdf" in pdf_resp.headers.get("Content-Type", "") or pdf_url.endswith(".pdf")):
                pdf_text = extract_text_from_pdf(pdf_resp.content)
                if pdf_text:
                    content += "\n\n" + pdf_text


    return {"title": title, "content": content, "date": date, "author": author}


# ---------------------------
# Supabase Functions
# ---------------------------
def fetch_pending_articles():
    print("📥 Fetching all pending records from press_releases...")
    all_data = []
    batch_size = 1000
    start = 0
    while True:
        end = start + batch_size - 1
        resp = (
            supabase.table("press_releases")
            .select("id, url, title, author, date")
            .eq("status", "pending")
            .range(start, end)
            .execute()
        )
        data = resp.data or []
        all_data.extend(data)
        print(f"  • Retrieved rows {start}–{end} (total: {len(all_data)})")
        if len(data) < batch_size:
            break
        start += batch_size
        time.sleep(0.2)
    print(f"✅ Total pending records fetched: {len(all_data)}")
    return all_data


def upload_clean_record(data):
    data["updated_at"] = datetime.now(UTC).isoformat()
    data["scraped_at"] = datetime.now(UTC).isoformat()
    supabase.table("press_releases_clean").upsert(data, on_conflict="url").execute()


def mark_complete(record_id):
    supabase.table("press_releases").update({"status": "complete"}).eq("id", record_id).execute()


def mark_failed(record_id, reason=None):
    supabase.table("press_releases").update({"status": "failed"}).eq("id", record_id).execute()
    print(f"❌ Marked as failed: ID {record_id} ({reason or 'no content'})")


# ---------------------------
# Main Process
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Universal Inner Scraper (no embeddings)...")
    pending = fetch_pending_articles()
    print(f"Found {len(pending)} pending articles.\n")

    for row in pending:
        url = row["url"]
        print(f"🔍 Scraping: {url}")

        if not can_fetch(url):
            mark_failed(row["id"], "robots.txt disallowed")
            continue

        resp = get_html(url)
        if not resp:
            mark_failed(row["id"], "failed to fetch HTML")
            continue

        parsed = parse_html(resp, url)

        if not parsed.get("content") or len(parsed["content"].split()) < 20:
            mark_failed(row["id"], "empty or too short")
            continue

        clean_record = {
            "id": row["id"],
            "title": parsed.get("title") or row.get("title"),
            "url": url,
            "date": parsed.get("date") or row.get("date"),
            "author": parsed.get("author") or row.get("author"),
            "content": parsed.get("content"),
            "status": "complete",
        }

        upload_clean_record(clean_record)
        mark_complete(row["id"])
        print(f"✅ Scraped and saved: {url}")
        time.sleep(1)

    print("\n🏁 Universal Inner Scraper Complete.")
