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
    """Check robots.txt permissions."""
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
    """Fetch HTML content with standard headers."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"⚠️ Failed to fetch {url}: {e}")
        return None


def extract_text_from_pdf(content):
    """Extract text safely from PDFs."""
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
        if not text:
            print("⚠️ PDF had no extractable text.")
        return text
    except Exception as e:
        print(f"⚠️ PDF extraction error: {e} — fallback decoding...")
        try:
            decoded = content.decode("latin-1", errors="ignore")
            return decoded[:1500]
        except Exception:
            return ""


def parse_html(resp, base_url):
    """Extract title, date, author, and main content (with PDF fallback)."""
    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Standard content
    main = (
        soup.select_one(
            "div.o-post-content, div.entry-content, div.article__body, article, main, div.col-sm-8"
        )
        or soup
    )
    paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    content = "\n\n".join([t for t in paragraphs if len(t.split()) > 5]).strip()

    # CFPB fallback (different structure)
    if not content:
        alt = soup.select_one("div.m-text-block, section.o-feature, div.o-page-content") or soup
        alt_paragraphs = [p.get_text(" ", strip=True) for p in alt.find_all("p")]
        content = "\n\n".join([t for t in alt_paragraphs if len(t.split()) > 5]).strip()

    # Date
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

    # Author
    by = soup.select_one(".author-name, .byline, meta[name='author']")
    author = (
        by.get("content")
        if by and by.has_attr("content")
        else by.get_text(" ", strip=True)
        if by
        else None
    )

    # PDF fallback
    if not content or len(content.split()) < 40:
        for link in soup.find_all("a", href=lambda h: h and h.lower().endswith(".pdf")):
            pdf_url = urljoin(base_url, link["href"])
            print(f"📎 Found PDF: {pdf_url}")
            pdf_resp = get_html(pdf_url)
            if pdf_resp and (
                "application/pdf" in pdf_resp.headers.get("Content-Type", "")
                or pdf_url.endswith(".pdf")
            ):
                pdf_text = extract_text_from_pdf(pdf_resp.content)
                if len(pdf_text.split()) > len(content.split()):
                    content = pdf_text

    return {"title": title, "content": content, "date": date, "author": author}


# ---------------------------
# Supabase Functions
# ---------------------------
def fetch_pending_articles():
    """Retrieve all pending records from Supabase press_releases table."""
    print("📥 Fetching pending records from press_releases...")
    all_data = []
    start = 0
    batch_size = 1000
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
        print(f"  • Retrieved rows {start}-{start + batch_size - 1} (total: {len(all_data)})")
        if len(data) < batch_size:
            break
        start += batch_size
        time.sleep(0.25)
    print(f"✅ Total pending records: {len(all_data)}")
    return all_data


def upload_clean_record(data):
    """Insert cleaned article into press_releases_clean."""
    data["updated_at"] = datetime.now(UTC).isoformat()
    data["scraped_at"] = datetime.now(UTC).isoformat()
    supabase.table("press_releases_clean").upsert(data, on_conflict="url").execute()


def mark_complete(record_id):
    supabase.table("press_releases").update({"status": "complete"}).eq("id", record_id).execute()


def mark_failed(record_id, reason=None):
    supabase.table("press_releases").update({"status": "failed"}).eq("id", record_id).execute()
    print(f"❌ Marked failed ID {record_id} ({reason or 'no content'})")


# ---------------------------
# Main Process
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting Universal Inner Scraper (no embeddings)...")

    pending = fetch_pending_articles()
    print(f"Found {len(pending)} pending articles.\n")

    if not pending:
        print("⚠️ No pending records found — exiting.")
        exit()

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

        # Handle empty or too short content
        if not parsed.get("content") or len(parsed["content"].split()) < 25:
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
