#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
OUTPUT_FILE = "catalog.json"

# Required by prompt: K/A/B/P/S/E/C/MS
VALID_TEST_TYPES = {"K", "A", "B", "P", "S", "E", "C", "MS"}

# Lightweight politeness to reduce transient blocks
REQUEST_TIMEOUT = 25
REQUEST_DELAY_SEC = 0.2
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@dataclass
class CatalogItem:
    name: str
    url: str
    description: str
    test_type: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "test_type": self.test_type,
        }


def get_soup(session: requests.Session, url: str) -> Optional[BeautifulSoup]:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        response.raise_for_status()
        time.sleep(REQUEST_DELAY_SEC)
        return BeautifulSoup(response.text, "html.parser")
    except requests.RequestException:
        return None


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip("/") + "/"
    return f"{parsed.scheme}://{parsed.netloc}{clean_path}"


def is_shl_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith("shl.com")


def text_or_empty(node) -> str:
    if not node:
        return ""
    return " ".join(node.get_text(" ", strip=True).split())


def is_prepackaged_job_solution(text_blob: str) -> bool:
    text = text_blob.lower()
    return "pre-packaged job solutions" in text or "job solution" in text


def get_test_type(text_blob: str) -> str:
    text = text_blob.lower()

    if any(x in text for x in ["coding", "technical knowledge", "language test", "knowledge test"]):
        return "K"
    if any(x in text for x in ["ability", "aptitude", "reasoning", "cognitive"]):
        return "A"
    if any(x in text for x in ["biodata"]):
        return "B"
    if any(x in text for x in ["personality", "opq", "motivation questionnaire"]):
        return "P"
    if any(x in text for x in ["situational judgement", "situational judgment", "sjt"]):
        return "S"
    if any(x in text for x in ["simulation", "assessment exercise", "in-box", "inbox"]):
        return "E"
    if any(x in text for x in ["competency", "competencies"]):
        return "C"
    if any(x in text for x in ["manager simulation", "management simulation", "ms"]):
        return "MS"

    # Fallback to a valid class to satisfy schema.
    return "C"


def looks_like_individual_test_link(url: str) -> bool:
    path = urlparse(url).path.lower()
    if "/solutions/products/product-catalog/" not in path:
        return False
    if "job-solutions" in path:
        return False
    return True


def extract_description(soup: BeautifulSoup) -> str:
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return " ".join(meta["content"].split())

    og = soup.find("meta", attrs={"property": "og:description"})
    if og and og.get("content"):
        return " ".join(og["content"].split())

    main = soup.find("main") or soup.find("article") or soup.body
    if not main:
        return ""
    paragraphs = main.find_all("p")
    best = [text_or_empty(p) for p in paragraphs if text_or_empty(p)]
    return best[0][:1000] if best else ""


def extract_name(soup: BeautifulSoup) -> str:
    h1 = soup.find("h1")
    if h1:
        return text_or_empty(h1)
    title = soup.find("title")
    if title:
        return text_or_empty(title).replace(" | SHL", "").strip()
    return ""


def collect_catalog_links(catalog_soup: BeautifulSoup) -> Set[str]:
    links: Set[str] = set()
    for a in catalog_soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = normalize_url(urljoin(CATALOG_URL, href))
        if not is_shl_url(full):
            continue
        if looks_like_individual_test_link(full):
            links.add(full)
    return links


def crawl_related_test_pages(
    session: requests.Session, seed_urls: Iterable[str], limit: int = 500
) -> Set[str]:
    visited: Set[str] = set()
    discovered: Set[str] = set()
    queue = deque(seed_urls)

    while queue and len(visited) < limit:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        soup = get_soup(session, url)
        if soup is None:
            continue

        discovered.add(url)
        for a in soup.select("a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            nxt = normalize_url(urljoin(url, href))
            if not is_shl_url(nxt):
                continue
            if not looks_like_individual_test_link(nxt):
                continue
            if nxt not in visited:
                queue.append(nxt)

    return discovered


def scrape_item(session: requests.Session, url: str) -> Optional[CatalogItem]:
    soup = get_soup(session, url)
    if soup is None:
        return None

    name = extract_name(soup)
    description = extract_description(soup)
    page_text = " ".join(soup.get_text(" ", strip=True).split())[:5000]

    combined = f"{name}\n{description}\n{page_text}"
    if is_prepackaged_job_solution(combined):
        return None

    if not name:
        return None

    test_type = get_test_type(combined)
    if test_type not in VALID_TEST_TYPES:
        test_type = "C"

    return CatalogItem(name=name, url=url, description=description, test_type=test_type)


def dedupe_by_name_and_url(items: List[CatalogItem]) -> List[CatalogItem]:
    seen: Set[str] = set()
    out: List[CatalogItem] = []
    for item in items:
        key = f"{item.name.lower()}|{item.url}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main() -> None:
    with requests.Session() as session:
        catalog_soup = get_soup(session, CATALOG_URL)
        if catalog_soup is None:
            raise RuntimeError(f"Failed to fetch catalog page: {CATALOG_URL}")

        seed_links = collect_catalog_links(catalog_soup)
        all_links = crawl_related_test_pages(session, seed_links)

        items: List[CatalogItem] = []
        for url in sorted(all_links):
            item = scrape_item(session, url)
            if item:
                items.append(item)

    items = dedupe_by_name_and_url(items)
    payload = [x.to_dict() for x in items]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(payload)} items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
