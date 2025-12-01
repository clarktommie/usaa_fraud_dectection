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

USER_AGENT = "Tommie-Clark-Fraud-Scraper/2.7 (+your-email@example.com)"
HEADERS = {"User-Agent": USER_AGENT}

# ---------------------------
# Helper Functions
# ---------------------------
def can_scrape(url: str) -> bool:
    base = "/".join(url.split("/")[:3])
    robots_url = urljoin(base, "/robots.txt")
    rp = urllib.robotparser.RobotFileParser()
    try:
        # Manually fetch so we can treat 404 as "no robots" instead of disallowing everything.
        resp = requests.get(robots_url, headers=HEADERS, timeout=15)
        if resp.status_code == 404:
            print(f"robots.txt check for {base}: ⚠️ 404 returned — treating as allowed")
            return True
        resp.raise_for_status()
        rp.parse(resp.text.splitlines())
        allowed = rp.can_fetch(USER_AGENT, url)
        print(f"robots.txt check for {base}: {'✅ allowed' if allowed else '🚫 disallowed'}")
        return allowed
    except Exception:
        print(f"⚠️ Could not read robots.txt for {base} — assuming allowed.")
        return True

def safe_get(url: str, timeout: int = 25):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"⚠️ GET failed for {url}: {e}")
        return None

# ---------------------------
# Federal Reserve Scraper
# ---------------------------
def scrape_federal_reserve():
    base = "https://www.federalreserve.gov"
    start_url = f"{base}/newsevents/pressreleases.htm"
    all_links = set()

    print("🌐 Scraping Federal Reserve (recursive)...")

    if not can_scrape(start_url):
        print("🚫 Skipping Federal Reserve (robots.txt blocked)")
        return []

    resp = safe_get(start_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    index_links = [
        urljoin(base, a["href"])
        for a in soup.select("a[href*='/newsevents/pressreleases/']")
        if a.get("href", "").endswith(".htm")
    ]
    print(f"📄 Found {len(index_links)} yearly/topic index pages")

    for link in index_links:
        sub_resp = safe_get(link)
        if not sub_resp:
            continue
        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
        sub_links = [
            urljoin(base, a["href"])
            for a in sub_soup.select("a[href*='/newsevents/pressreleases/']")
            if any(a.get("href", "").endswith(ext) for ext in [".htm", ".pdf"])
        ]
        all_links.update(sub_links)
        print(f"  • {link} → {len(sub_links)} releases")
        time.sleep(0.15)

    print(f"📰 Total unique Federal Reserve URLs: {len(all_links)}")

    return [
        {
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "FederalReserve",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        for u in sorted(all_links)
    ]

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

    resp = safe_get(start_url)
    if not resp:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")

    links = [
        urljoin(base, a["href"])
        for a in soup.select("a[href]")
        if ("category" in (a.get("href") or "")) or ("issues" in (a.get("href") or ""))
    ]
    all_links.update(links)

    print(f"📰 Total BPI links collected: {len(all_links)}")

    return [
        {
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "BPI",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        for u in sorted(all_links)
    ]

# ---------------------------
# CFPB Press Releases Scraper (fixed)
# ---------------------------
def scrape_cfpb_press_releases():
    base = "https://www.consumerfinance.gov"
    root = f"{base}/about-us/newsroom"
    all_links = set()

    print("🌐 Scraping CFPB Press Releases (Full Pagination)...")

    if not can_scrape(f"{root}/?categories=press-release"):
        print("🚫 Skipping CFPB (robots.txt blocked)")
        return []

    page = 1
    max_pages = 1000
    while page <= max_pages:
        if page == 1:
            url = f"{root}/?categories=press-release"
        else:
            url = f"{root}/?page={page}&categories=press-release"

        print(f"🔄 CFPB page {page}: {url}")
        resp = safe_get(url, timeout=30)
        if not resp:
            print(f"❌ Could not fetch page {page} — stopping.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        # extract links for press-release cards
        cards = soup.select("a.o-card__link[href]")
        if not cards:
            cards = soup.select("a[href*='/about-us/newsroom/']")

        page_links = []
        for a in cards:
            href = a.get("href")
            if not href:
                continue
            full = urljoin(base, href.split("?")[0])
            if "/about-us/newsroom/" in full and "/about-us/newsroom/?categories=" not in full:
                page_links.append(full)
        # dedupe
        page_links = list(dict.fromkeys(page_links))
        if not page_links:
            print(f"⚠️ No press-release links found on page {page} — stopping.")
            break

        all_links.update(page_links)
        print(f"  • Found {len(page_links)} links on page {page}")
        page += 1
        time.sleep(0.3)

    print(f"📰 Total CFPB press-release links collected: {len(all_links)}")

    return [
        {
            "title": None,
            "url": u,
            "date": None,
            "scraped_at": datetime.now(UTC).isoformat(),
            "author": None,
            "source": "ConsumerFinanceGov",
            "status": "pending",
            "updated_at": datetime.now(UTC).isoformat(),
        }
        for u in sorted(all_links)
    ]

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
        batch = records[i:i + batch_size]
        supabase.table("press_releases").upsert(batch, on_conflict="url").execute()
        print(f"✅ Uploaded batch {i // batch_size + 1} ({len(batch)} records)")
        time.sleep(0.15)
    print("✅ All records uploaded successfully.")

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🚀 Universal Outer Scraper Starting...")

    all_records = []

    fed_records = scrape_federal_reserve()
    print(f"✅ Federal Reserve: {len(fed_records)} records scraped.\n")
    all_records.extend(fed_records)

    bpi_records = scrape_bpi()
    print(f"✅ BPI: {len(bpi_records)} records scraped.\n")
    all_records.extend(bpi_records)

    cfpb_records = scrape_cfpb_press_releases()
    print(f"✅ CFPB: {len(cfpb_records)} records scraped.\n")
    all_records.extend(cfpb_records)

    print(f"🧾 Total scraped across all sources: {len(all_records)}")
    upload_to_supabase(all_records)
    print("🏁 Universal Outer Scraper Complete.")
