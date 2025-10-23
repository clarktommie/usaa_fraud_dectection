import os
import time
from datetime import datetime
import datetime as dt

import requests
from bs4 import BeautifulSoup
from supabase import create_client
from dotenv import load_dotenv
import urllib.robotparser



# --- Load environment variables ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# --- Initialize Supabase client ---
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Define base URLs ---
BASE_URL = "https://www.federalreserve.gov"
PRESS_ARCHIVE_URL = f"{BASE_URL}/newsevents/pressreleases/"
ROBOTS_URL = f"{BASE_URL}/robots.txt"

# --- Check robots.txt before scraping ---
def can_scrape(url=PRESS_ARCHIVE_URL, user_agent="Tommie-Clark-Fraud-Scraper"):
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(ROBOTS_URL)
    rp.read()
    return rp.can_fetch(user_agent, url)

if not can_scrape():
    print("Scraping not allowed per robots.txt. Exiting.")
    exit(0)

# --- Scrape individual releases from each year's index page ---
def scrape_all_releases(start_year=2006, end_year=2025):
    headers = {"User-Agent": "Tommie-Clark-Fraud-Scraper/1.0 (+your-email@example.com)"}
    all_data = []

    for year in range(start_year, end_year + 1):
        index_url = f"{PRESS_ARCHIVE_URL}{year}-press.htm"
        print(f"\nScraping index: {index_url}")

        if not can_scrape(index_url):
            print(f"⚠️ Skipping {index_url} (disallowed by robots.txt)")
            continue

        try:
            response = requests.get(index_url, headers=headers, timeout=20)
            if response.status_code == 404:
                print(f"⚠️ No index page found for {year}. Skipping.")
                continue
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = soup.select("a[href]")
            count = 0

            for a in links:
                href = a.get("href")
                if not href:
                    continue

                # Build full URL
                full_url = href if href.startswith("http") else f"{BASE_URL}{href}"

                # Filter only valid press-release paths
                if "/newsevents/pressreleases/" not in full_url:
                    continue
                if full_url.endswith("-press.htm"):
                    continue  # skip the yearly summary page itself

                title = a.get_text(strip=True)
                if not title:
                    continue

                # Optional date parsing (some links include year/month in URL)
                pub_date = None
                try:
                    if str(year) in href:
                        pub_date = datetime(year, 1, 1).date()
                except Exception:
                    pub_date = None

                all_data.append({
                    "title": title,
                    "url": full_url,
                    "date": pub_date,
                    "status": "pending",
                    "scraped_at": datetime.utcnow().isoformat()
                })
                count += 1

            print(f"✓ Found {count} release links for {year}")
            time.sleep(1)

        except Exception as e:
            print(f"⚠️ Error scraping {index_url}: {e}")
            continue

    return all_data

# --- Save to Supabase ---
def save_to_supabase(records):
    if not records:
        print("No records found.")
        return

    # --- Deduplicate by URL ---
    unique_records = list({r["url"]: r for r in records}.values())
    print(f"\nUploading {len(unique_records)} unique records to Supabase...")

    # --- Convert date objects to strings ---
    for r in unique_records:
        if isinstance(r.get("date"), dt.date):
            r["date"] = r["date"].isoformat()

    # --- Batch upload to avoid Supabase payload limit ---
    batch_size = 500
    total_uploaded = 0
    for i in range(0, len(unique_records), batch_size):
        batch = unique_records[i:i + batch_size]
        response = supabase.table("press_releases").upsert(batch, on_conflict="url").execute()
        total_uploaded += len(batch)
        print(f"Batch {i // batch_size + 1}: Uploaded {len(batch)} records.")
        time.sleep(0.5)

    print(f"\n✅ Uploaded {total_uploaded} total records to Supabase.")


# --- Main ---
if __name__ == "__main__":
    print("Starting deep scrape (2006–2025)...")
    records = scrape_all_releases(2006, 2025)
    print(f"\nScraped {len(records)} total press releases.")
    save_to_supabase(records)
    print("Finished.")
