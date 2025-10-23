import os
import time
from datetime import datetime, UTC
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from PyPDF2 import PdfReader   # ← added for PDF text extraction

from supabase import create_client
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_URL = "https://www.federalreserve.gov"

# --- Helper to extract text, date, and author ---
def extract_article_text(url):
    headers = {"User-Agent": "Tommie-Clark-Fraud-Scraper/1.0 (+your-email@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()

        # --- Handle PDFs ---
        if "application/pdf" in resp.headers.get("Content-Type", "") or url.endswith(".pdf"):
            print(f"Extracting PDF content: {url}")
            try:
                pdf_reader = PdfReader(BytesIO(resp.content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
                if not text.strip():
                    print(f"No text found in PDF: {url}")
                    return None
                return {
                    "content": text.strip(),
                    "date": None,
                    "author": None
                }
            except Exception as e:
                print(f"Error extracting PDF text: {e}")
                return None

        # --- Handle HTML pages ---
        soup = BeautifulSoup(resp.text, "html.parser")

        date_tag = soup.select_one(".article__time, .release-date, .pubdate, time")
        date_text = date_tag.get_text(strip=True) if date_tag else None

        author_tag = None
        for p in soup.find_all("p"):
            if any(k in p.text for k in ["Contact", "Public Affairs", "Media", "Press"]):
                author_tag = p
                break
        author_text = author_tag.get_text(strip=True) if author_tag else None

        main = soup.select_one(".col-xs-12.col-sm-8.col-md-8") or soup.select_one(".col-xs-12.col-sm-12.col-md-8")
        if not main:
            main = soup

        paragraphs = [p.get_text(" ", strip=True) for p in main.find_all("p")]
        if not paragraphs:
            paragraphs = [div.get_text(" ", strip=True) for div in main.find_all("div") if div.get_text(strip=True)]
        content = "\n\n".join(paragraphs)

        return {
            "content": content.strip(),
            "date": date_text,
            "author": author_text
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


# --- Pull all pending rows ---
def fetch_pending_articles():
    resp = supabase.table("press_releases").select("id,url").eq("status", "pending").execute()
    return resp.data


# --- Update record ---
def update_article(id, article):
    supabase.table("press_releases").update({
        "content": article["content"],
        "author": article["author"],
        "date": article["date"],
        "status": "complete",
        "updated_at": datetime.now(UTC).isoformat()
    }).eq("id", id).execute()


# --- Main ---
if __name__ == "__main__":
    print("Starting inner scrape...")
    pending = fetch_pending_articles()
    print(f"Found {len(pending)} pending articles.")

    for row in pending:
        url = row["url"]
        print(f"Scraping: {url}")

        # Skip non-press-release links
        if not url.startswith("https://www.federalreserve.gov/newsevents/pressreleases/"):
            print(f"Skipping non-press-release URL: {url}")
            continue

        article = extract_article_text(url)
        if article and article["content"]:
            update_article(row["id"], article)
            print(f"Updated {row['id']} ✓")
        else:
            print(f"Skipped {url}")
        time.sleep(1)  # polite delay between requests

    print("Inner scrape complete.")

