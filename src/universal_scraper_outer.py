import os
import time
from datetime import datetime, UTC
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from supabase import create_client
from dotenv import load_dotenv
import urllib.robotparser

# ---------------------------
# Environment Setup
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_AGENT = "Tommie-Clark-Fraud-Scraper/2.0 (+your-email@example.com)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------
# Helpers
# ---------------------------
def can_scrape(url: str) -> bool:
    """Check robots.txt before scraping."""
    base = "/".join(url.split("/")[:3])
    robots_url = urljoin(base, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch(USER_AGENT, url)
        print(f"robots.txt check for {base}: {'✅ allowed' if allowed else '🚫 disallowed'}")
        return allowed
    except Exception:
        print(f"⚠️ Could not read robots.txt for {base} — assuming allowed.")
        return True


# ---------------------------
# Federal Reserve Recursive Scraper
# ---------------------------
def scrape_federal_reserve():
    """Scrape all Federal Reserve press releases (recursive through yearly pages)."""
    base = "https://www.federalreserve.gov"
    start_url = f"{base}/newsevents/pressreleases.htm"
    all_links = set()

    print("🌐 Scraping Federal Reserve (recursive)...")

    if not can_scrape(start_url):
        print("🚫 Skipping Federal Reserve (robots.txt blocked)")
        return []

    resp = requests.get(start_url, headers=HEADERS, timeout=25)
    soup = BeautifulSoup(resp.text, "html.parser")

    index_links = [
        urljoin(base, a["href"])
        for a in soup.select("a[href*='/newsevents/pressreleases/']")
        if a["href"].endswith(".htm")
    ]
    print(f"📄 Found {len(index_links)} yearly/topic index pages")

    for link in index_links:
        try:
            sub_resp = requests.get(link, headers=HEADERS, timeout=25)
            sub_soup = BeautifulSoup(sub_resp.text, "html.parser")

            sub_links = [
                urljoin(base, a["href"])
                for a in sub_soup.select("a[href*='/newsevents/pressreleases/']")
                if any(a["href"].endswith(ext) for ext in [".htm", ".pdf"])
            ]
            all_links.update(sub_links)
            print(f"  • {link} → {len(sub_links)} releases")
            time.sleep(0.5)
        except Exception as e:
            print(f"⚠️ Failed to scrape {link}: {e}")

    print(f"📰 Total unique Federal Reserve release URLs: {len(all_links)}")

    records = []
    for u in sorted(all_links):
        records.append({
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "FederalReserve",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        })
    return records


# ---------------------------
# Bank Policy Institute Scraper
# ---------------------------
def scrape_bpi():
    base = "https://bpi.com"
    start_url = f"{base}/issues-and-topics/"
    all_links = set()

    print("🌐 Scraping Bank Policy Institute...")

    if not can_scrape(start_url):
        print("🚫 Skipping BPI (robots.txt blocked)")
        return []

    try:
        resp = requests.get(start_url, headers=HEADERS, timeout=25)
        soup = BeautifulSoup(resp.text, "html.parser")
        links = [
            urljoin(base, a["href"])
            for a in soup.select("a[href]")
            if "category" in a["href"] or "issues" in a["href"]
        ]
        all_links.update(links)
    except Exception as e:
        print(f"⚠️ BPI scrape error: {e}")

    print(f"📰 Total BPI links collected: {len(all_links)}")

    records = []
    for u in sorted(all_links):
        records.append({
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "BPI",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        })
    return records


# ---------------------------
# CFPB Activity Log Scraper
# ---------------------------
def scrape_cfpb_activity_log():
    base = "https://www.consumerfinance.gov"
    start_url = f"{base}/activity-log/"

    print("🌐 Scraping CFPB Activity Log...")

    if not can_scrape(start_url):
        print("🚫 Skipping CFPB (robots.txt blocked)")
        return []

    all_links = set()
    try:
        resp = requests.get(start_url, headers=HEADERS, timeout=25)
        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select("li.o-post-list__item a[href]"):
            href = item["href"]
            full = urljoin(base, href)
            all_links.add(full)
        print(f"📰 Total CFPB activity links collected: {len(all_links)}")

    except Exception as e:
        print(f"⚠️ CFPB scrape error: {e}")

    records = []
    for u in sorted(all_links):
        records.append({
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "ConsumerFinanceGov",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        })
    return records


# ---------------------------
# Upload to Supabase
# ---------------------------
def upload_to_supabase(records):
    if not records:
        print("⚠️ No records to upload.")
        return
    print(f"📤 Uploading {len(records)} records to Supabase...")
    batch_size = 200
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("press_releases").upsert(batch, on_conflict="url").execute()
        print(f"Uploaded batch {i // batch_size + 1}")
        time.sleep(0.5)
    print("✅ All records uploaded successfully.")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🚀 Universal Outer Scraper Starting...")

    all_records = []

    # Federal Reserve
    fed_records = scrape_federal_reserve()
    print(f"✅ Federal Reserve: {len(fed_records)} records scraped.\n")
    all_records.extend(fed_records)

    # Bank Policy Institute
    bpi_records = scrape_bpi()
    print(f"✅ BPI: {len(bpi_records)} records scraped.\n")
    all_records.extend(bpi_records)

    # CFPB Activity Log
    cfpb_records = scrape_cfpb_activity_log()
    print(f"✅ CFPB: {len(cfpb_records)} records scraped.\n")
    all_records.extend(cfpb_records)

    # Upload all
    print(f"🧾 Total scraped across all sources: {len(all_records)}")
    upload_to_supabase(all_records)
    print("🏁 Universal Outer Scraper Complete.")
