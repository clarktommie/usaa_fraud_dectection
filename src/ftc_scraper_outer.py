import os
import time
from datetime import datetime
import datetime as dt
import requests
from bs4 import BeautifulSoup
from supabase import create_client
from dotenv import load_dotenv
import urllib.robotparser


# ---------------------------
# Setup
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_URL = "https://bpi.com"
START_URL = f"{BASE_URL}/issues-and-topics/"
ROBOTS_URL = f"{BASE_URL}/robots.txt"

USER_AGENT = "Tommie-Clark-Fraud-Scraper/1.0 (+your-email@example.com)"


# ---------------------------
# Check robots.txt
# ---------------------------
def can_scrape(url=START_URL, user_agent=USER_AGENT):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(ROBOTS_URL)
    try:
        rp.read()
        allowed = rp.can_fetch(user_agent, url)
        print(f"robots.txt check for {url}: {'✅ allowed' if allowed else '🚫 disallowed'}")
        return allowed
    except Exception:
        print("⚠️ Could not read robots.txt — proceeding cautiously.")
        return True


# ---------------------------
# Scrape BPI "Issues & Topics" links
# ---------------------------
def scrape_bpi_topics():
    headers = {"User-Agent": USER_AGENT}
    all_data = []

    print(f"🌐 Fetching: {START_URL}")
    try:
        resp = requests.get(START_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        all_links = soup.select("a[href]")
        topic_links = []

        for a in all_links:
            href = a.get("href", "").strip()
            text = a.get_text(strip=True)

            if not text or href == "/issues-and-topics/":
                continue

            # match both absolute and relative topic links
            if any(x in href for x in [
                "issues-and-topics",
                "/issues/",
                "/topics/",
                "bank-capital",
                "liquidity",
                "cybersecurity",
                "consumer-affairs",
                "stress-testing"
            ]):
                topic_links.append((text, href))

        print(f"Found {len(topic_links)} topic links.")

        for title, href in topic_links:
            full_url = href if href.startswith("http") else f"{BASE_URL}/{href.lstrip('/')}"
            all_data.append({
                "title": title,
                "url": full_url,
                "date": None,
                "status": "pending",
                "scraped_at": datetime.now(dt.UTC).isoformat()
            })

        print(f"✓ Collected {len(all_data)} topics successfully.")

    except Exception as e:
        print(f"⚠️ Error scraping {START_URL}: {e}")

    return all_data


# ---------------------------
# Upload to Supabase
# ---------------------------
def save_to_supabase(records):
    if not records:
        print("No records found to upload.")
        return

    unique_records = list({r["url"]: r for r in records}.values())
    print(f"📤 Uploading {len(unique_records)} unique records to Supabase...")

    for r in unique_records:
        if isinstance(r.get("date"), dt.date):
            r["date"] = r["date"].isoformat()

    batch_size = 200
    total_uploaded = 0
    for i in range(0, len(unique_records), batch_size):
        batch = unique_records[i:i + batch_size]
        supabase.table("press_releases").upsert(batch, on_conflict="url").execute()
        total_uploaded += len(batch)
        print(f"Batch {i // batch_size + 1}: Uploaded {len(batch)} records.")
        time.sleep(0.5)

    print(f"✅ Uploaded {total_uploaded} total records to Supabase.")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting BPI Issues & Topics scrape...")

    if not can_scrape():
        print("🚫 Scraping not allowed per robots.txt. Exiting.")
        exit(0)

    records = scrape_bpi_topics()
    print(f"Scraped {len(records)} total entries.")
    save_to_supabase(records)
    print("🏁 Finished.")
