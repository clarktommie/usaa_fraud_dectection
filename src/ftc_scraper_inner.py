import os
import time
from datetime import datetime, UTC
from urllib.parse import urlparse, urljoin
from io import BytesIO
import urllib.robotparser
import re

import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from supabase import create_client
from dotenv import load_dotenv

# ---------------------------
# Config
# ---------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_ORIGIN = "https://bpi.com"
BASE_DOMAIN = "bpi.com"
ROBOT_URL = f"{BASE_ORIGIN}/robots.txt"
UA = "Tommie-Clark-Fraud-Scraper/1.0 (+your-email@example.com)"

MAX_PAGES_PER_TOPIC = 5
REQUEST_TIMEOUT = 25
SLEEP_BETWEEN = 1.0

# ---------------------------
# Helpers
# ---------------------------
def can_fetch(url: str) -> bool:
    try:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(ROBOT_URL)
        rp.read()
        return rp.can_fetch(UA, url)
    except Exception:
        return True  # be permissive if robots fetch fails (path was allowed earlier)

def _get(url, timeout=REQUEST_TIMEOUT):
    return requests.get(url, headers={"User-Agent": UA}, timeout=timeout)

def _is_bpi(url: str) -> bool:
    try:
        return urlparse(url).netloc.endswith(BASE_DOMAIN)
    except Exception:
        return False

def _abs(url: str, base: str = BASE_ORIGIN) -> str:
    return url if url.startswith("http") else urljoin(base, url)

def _looks_like_post_url(href: str) -> bool:
    """
    WP permalinks often look like /YYYY/MM/slug/ or /YYYY/slug/.
    Also accept links that contain '/20xx' anywhere.
    """
    path = urlparse(href).path
    return bool(re.search(r"/20\d{2}(/|-[0-9])", path))

# ---------------------------
# Single post extraction
# ---------------------------
def extract_single_post(url: str):
    if not can_fetch(url):
        print(f"🚫 Disallowed by robots.txt: {url}")
        return None
    try:
        resp = _get(url)
        resp.raise_for_status()

        if "application/pdf" in resp.headers.get("Content-Type", "") or url.lower().endswith(".pdf"):
            try:
                reader = PdfReader(BytesIO(resp.content))
                text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
                if not text:
                    return None
                return {"content": text, "date": None, "author": None}
            except Exception:
                return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Date
        date_text = None
        t = soup.select_one("time[datetime]") or soup.find("time") or soup.find("meta", {"property": "article:published_time"})
        if t:
            date_text = t.get("datetime") or t.get("content") or t.get_text(strip=True)

        # Author
        author = None
        by = soup.select_one(".author-name, .byline, meta[name='author']")
        if by:
            author = by.get("content") if by.has_attr("content") else by.get_text(" ", strip=True)

        # Content
        main = (
            soup.select_one("div.entry-content")
            or soup.select_one("article")
            or soup.find("main")
            or soup
        )
        parts = []
        for el in main.find_all(["p", "li"], recursive=True):
            txt = el.get_text(" ", strip=True)
            if len(txt.split()) > 5:
                parts.append(txt)
        content = "\n\n".join(parts).strip()
        if not content:
            return None

        return {"content": content, "date": date_text, "author": author}
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return None

# ---------------------------
# Page type detection
# ---------------------------
def is_single_post_page(soup: BeautifulSoup) -> bool:
    body = soup.find("body")
    if body and body.has_attr("class"):
        classes = " ".join(body.get("class", []))
        if "single" in classes or "single-post" in classes:
            return True
    entry = soup.select_one("div.entry-content")
    if entry and len(entry.get_text(strip=True)) > 400:
        return True
    return False

# ---------------------------
# Archive extraction
# ---------------------------
def extract_archive_links(page_url: str, soup: BeautifulSoup):
    """
    Return list of candidate post links from category/topic/archive pages.
    """
    links = set()

    selectors = [
        "h1.entry-title a[href]",          # sometimes home loop uses h1
        "h2.entry-title a[href]",
        "h3.entry-title a[href]",
        "h2.elementor-post__title a[href]",
        "h3.elementor-post__title a[href]",
        "article.post a[href]",
        "article.elementor-post a[href]",
        "div.elementor-posts-container a[href]",
        "a[rel='bookmark'][href]",
        "a.entry-title-link[href]",
        "a[href*='/202']",                 # year-based permalinks
    ]

    for sel in selectors:
        for a in soup.select(sel):
            href = a.get("href", "").strip()
            if not href:
                continue
            href_abs = _abs(href, page_url)
            if not _is_bpi(href_abs):
                continue
            # Filter out obvious non-post lists
            path = urlparse(href_abs).path.rstrip("/")
            if path.endswith("/category") or "/category/" in path and not _looks_like_post_url(href_abs):
                continue
            # Many BPI posts live at /YYYY/... -> keep anything that looks like a post
            if _looks_like_post_url(href_abs) or "/blog/" in path or "/insights/" in path:
                links.add(href_abs)

    # Fallback: if still empty, accept any internal link that isn't a category/tag/home
    if not links:
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            href_abs = _abs(href, page_url)
            if not _is_bpi(href_abs):
                continue
            p = urlparse(href_abs).path.lower()
            if any(seg in p for seg in ["/category/", "/tag/", "/issues-and-topics/"]):
                continue
            if p.count("/") >= 2 and len(p.split("/")[-1]) > 3:  # looks like a slug
                links.add(href_abs)

    return list(links)

def find_next_page(soup: BeautifulSoup, current_url: str):
    for sel in ['a[rel="next"]', ".nav-links a.next", "a.next.page-numbers", "a.page-numbers.next", "a.next"]:
        a = soup.select_one(sel)
        if a and a.get("href"):
            return _abs(a["href"], current_url)
    return None

# ---------------------------
# Supabase helpers
# ---------------------------
def fetch_topic_candidates():
    """
    Get BPI rows that are topics/archives to expand:
    - status pending OR complete but content is NULL (from prior incorrect completion)
    - URL under bpi.com and looks like a category/topic/index
    """
    resp = supabase.table("press_releases").select("id,url,title,status,content").in_("status", ["pending", "complete"]).execute()
    rows = resp.data or []
    candidates = []
    for r in rows:
        u = (r.get("url") or "").rstrip("/")
        if not _is_bpi(u):
            continue
        path = urlparse(u).path.lower()
        looks_topic = (
            u == f"{BASE_ORIGIN}/issues-and-topics" or
            "/issues-and-topics" in path or
            "/category/" in path or
            path.count("/") <= 2  # top-level BPI sections like /cybersecurity
        )
        if looks_topic and (r.get("content") in (None, "")):
            candidates.append({"id": r["id"], "url": u, "title": r.get("title", ""), "status": r.get("status", "pending")})
    return candidates

def upsert_new_posts(posts):
    """
    posts: list of {"url": ..., "title": ...}
    Insert as pending (upsert on url).
    """
    if not posts:
        return 0
    dedup = {}
    for p in posts:
        u = p.get("url")
        if not u:
            continue
        dedup[u] = {
            "title": p.get("title") or u,
            "url": u,
            "date": None,
            "status": "pending",
            "scraped_at": datetime.now(UTC).isoformat()
        }
    payload = list(dedup.values())
    if not payload:
        return 0
    supabase.table("press_releases").upsert(payload, on_conflict="url").execute()
    return len(payload)

def update_row_status(id_, fields: dict):
    supabase.table("press_releases").update({
        **fields,
        "updated_at": datetime.now(UTC).isoformat()
    }).eq("id", id_).execute()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("🚀 Starting BPI inner scrape...")
    topics = fetch_topic_candidates()
    print(f"Found {len(topics)} topic/archive pages to expand.\n")

    for row in topics:
        topic_url = row["url"]
        print(f"🔍 Topic/archive: {topic_url}")

        if not can_fetch(topic_url):
            print(f"🚫 Disallowed by robots.txt: {topic_url}")
            continue

        try:
            page = 1
            page_url = topic_url
            collected = []

            while page_url and page <= MAX_PAGES_PER_TOPIC:
                print(f"  • Parsing archive page {page}: {page_url}")
                r = _get(page_url)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")

                # If this is actually a single post, extract content & finish
                if is_single_post_page(soup):
                    article = extract_single_post(page_url)
                    if article and article.get("content"):
                        update_row_status(row["id"], {
                            "content": article["content"],
                            "author": article.get("author"),
                            "date": article.get("date"),
                            "status": "complete"
                        })
                        print("  ✅ Single post detected; updated content and marked complete.")
                        break

                # Otherwise treat as archive and extract post links
                links = extract_archive_links(page_url, soup)
                if links:
                    # Best-effort fetch titles
                    for u in links:
                        title = None
                        try:
                            ar = _get(u, timeout=12)
                            if ar.status_code == 200:
                                sp = BeautifulSoup(ar.text, "html.parser")
                                h = sp.select_one("h1.entry-title, h1")
                                title = h.get_text(strip=True) if h else None
                        except Exception:
                            pass
                        collected.append({"url": u, "title": title or u})

                next_url = find_next_page(soup, page_url)
                page_url = next_url
                page += 1
                time.sleep(SLEEP_BETWEEN)

            inserted = upsert_new_posts(collected)
            if inserted > 0:
                update_row_status(row["id"], {"status": "complete"})
                print(f"✅ Expanded archive: inserted {inserted} posts; marked topic complete.")
            else:
                # leave as pending to revisit / debug rather than losing it
                print(f"⚠️ No posts found under archive: {topic_url} (left status as '{row['status']}').")

        except Exception as e:
            print(f"⚠️ Error processing {topic_url}: {e}")

        time.sleep(SLEEP_BETWEEN)

    print("\n🏁 BPI inner scrape complete.")
