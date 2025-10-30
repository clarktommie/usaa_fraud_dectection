import os
import time
from datetime import datetime, UTC
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader
from supabase import create_client
from dotenv import load_dotenv


# ---------------------------
# Setup
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_URL = "https://bpi.com"


# ---------------------------
# Extract text, date, and author
# ---------------------------
def extract_article_text(url):
    headers = {"User-Agent": "Tommie-Clark-Fraud-Scraper/1.0 (+your-email@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        # --- Handle PDF files ---
        if "application/pdf" in resp.headers.get("Content-Type", "") or url.endswith(".pdf"):
            print(f"📄 Extracting PDF content: {url}")
            try:
                pdf_reader = PdfReader(BytesIO(resp.content))
                text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
                if not text.strip():
                    print(f"⚠️ No text found in PDF: {url}")
                    return None
                return {"content": text.strip(), "date": None, "author": None}
            except Exception as e:
                print(f"⚠️ Error reading PDF: {e}")
                return None

        # --- Parse HTML page ---
        soup = BeautifulSoup(resp.text, "html.parser")

        # Date (BPI posts often have <time> or meta property)
        date_tag = (
            soup.select_one("time") or
            soup.find("meta", {"property": "article:published_time"})
        )
        date_text = None
        if date_tag:
            date_text = (
                date_tag.get("datetime")
                or date_tag.get("content")
                or date_tag.get_text(strip=True)
            )

        # Author (meta tag or under a byline)
        author_tag = (
            soup.select_one(".author-name, .byline, meta[name='author']")
            or soup.find("meta", {"name": "author"})
        )
        author_text = (
            author_tag.get("content") if author_tag and author_tag.has_attr("content")
            else author_tag.get_text(strip=True) if author_tag
            else None
        )

        # Main content (div.entry-content or article tag)
        main = (
            soup.select_one("div.entry-content")
            or soup.select_one("article")
            or soup.find("main")
            or soup
        )

        paragraphs = []
        for p in main.find_all("p", recursive=True):
            text = p.get_text(" ", strip=True)
            if len(text.split()) > 5:
                paragraphs.append(text)

        content = "\n\n".join(paragraphs).strip()
        if not content:
            print(f"⚠️ No readable content found at: {url}")
            return None

        return {
            "content": content,
            "date": date_text,
            "author": author_text
        }

    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return None


# ---------------------------
# Fetch pending records from Supabase
# ---------------------------
def fetch_pending_articles():
    resp = supabase.table("press_releases").select("id,url").eq("status", "pending").execute()
    return resp.data


# ---------------------------
# Update record in Supabase
# ---------------------------
def update_article(id, article):
    supabase.table("press_releases").update({
        "content": article["content"],
        "author": article["author"],
        "date": article["date"],
        "status": "complete",
        "updated_at": datetime.now(UTC).isoformat()
    }).eq("id", id).execute()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting BPI inner scrape...")
    pending = fetch_pending_articles()
    print(f"Found {len(pending)} pending articles.\n")

    for row in pending:
        url = row["url"]
        print(f"🔍 Scraping: {url}")

        # Skip unrelated or external URLs
        if not url.startswith(BASE_URL):
            print(f"Skipping external URL: {url}")
            continue

        article = extract_article_text(url)
        if article and article["content"]:
            update_article(row["id"], article)
            print(f"✅ Updated record {row['id']}")
        else:
            print(f"⚠️ Skipped {url}")

        time.sleep(1)  # polite delay

    print("\n🏁 Inner scrape complete.")
