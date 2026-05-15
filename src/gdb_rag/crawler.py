from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlunparse, urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from gdb_rag.config import Settings


@dataclass(frozen=True)
class CachedPage:
    url: str
    path: Path
    html: str


def canonical_url(url: str) -> str:
    clean_url, _fragment = urldefrag(url)
    parsed = urlparse(clean_url)
    path = parsed.path
    if path.endswith(".html/"):
        path = path[:-1]
    return urlunparse(parsed._replace(path=path))


def is_manual_page(url: str, source_url: str) -> bool:
    parsed = urlparse(url)
    source = urlparse(source_url)
    source_path = source.path[:-1] if source.path.endswith(".html/") else source.path
    split_dir = source_path[:-5] + "/"
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == source.netloc
        and (parsed.path == source_path or parsed.path.startswith(split_dir))
        and parsed.path.endswith(".html")
    )


def resolve_manual_link(page_url: str, href: str) -> str:
    parsed_href = urlparse(href)
    if parsed_href.scheme or href.startswith("#"):
        return urljoin(page_url, href)

    parsed_page = urlparse(page_url)
    if parsed_page.path.endswith("/gdb.html"):
        # The source TOC is at onlinedocs/gdb.html, but split pages live under
        # onlinedocs/gdb/. Resolve Texinfo menu links against that directory.
        split_base = urlunparse(parsed_page._replace(path=parsed_page.path[:-5] + "/index.html"))
        return urljoin(split_base, href)

    return urljoin(page_url, href)


def extract_manual_links(html: str, page_url: str, source_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    for tag in soup.find_all("a", href=True):
        absolute = canonical_url(resolve_manual_link(page_url, tag["href"]))
        if is_manual_page(absolute, source_url):
            links.add(absolute)
    return sorted(links)


def cache_path_for_url(raw_dir: Path, url: str) -> Path:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or "index"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return raw_dir / f"{stem}-{digest}.html"


def manifest_path(raw_dir: Path) -> Path:
    return raw_dir / "manifest.json"


def load_manifest(raw_dir: Path) -> dict[str, str]:
    path = manifest_path(raw_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(raw_dir: Path, manifest: dict[str, str]) -> None:
    manifest_path(raw_dir).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def fetch_page(url: str, settings: Settings) -> str:
    response = requests.get(
        url,
        headers={"User-Agent": settings.user_agent},
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    return response.text


def read_or_fetch_page(
    url: str,
    settings: Settings,
    manifest: dict[str, str],
    refresh_cache: bool = False,
) -> CachedPage:
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_url(settings.raw_dir, url)
    if refresh_cache or not path.exists():
        html = fetch_page(url, settings)
        path.write_text(html, encoding="utf-8")
    else:
        html = path.read_text(encoding="utf-8")
    manifest[url] = path.name
    return CachedPage(url=url, path=path, html=html)


def crawl_manual(
    settings: Settings,
    limit: int | None = None,
    refresh_cache: bool = False,
) -> list[CachedPage]:
    start_url = canonical_url(settings.source_url)
    manifest = load_manifest(settings.raw_dir)
    queue: deque[str] = deque([start_url])
    queued: set[str] = {start_url}
    visited: set[str] = set()
    pages: list[CachedPage] = []

    with tqdm(total=limit, desc="Crawling GDB manual", unit="page") as progress:
        while queue and (limit is None or len(pages) < limit):
            url = queue.popleft()
            if url in visited:
                continue

            try:
                page = read_or_fetch_page(url, settings, manifest, refresh_cache)
            except requests.RequestException as exc:
                tqdm.write(f"Skipping {url}: {exc}")
                visited.add(url)
                continue
            pages.append(page)
            visited.add(url)
            progress.update(1)

            for link in extract_manual_links(page.html, page.url, settings.source_url):
                if link not in queued and link not in visited:
                    queued.add(link)
                    queue.append(link)

    save_manifest(settings.raw_dir, manifest)
    return pages


def iter_cached_pages(settings: Settings) -> Iterable[CachedPage]:
    manifest = load_manifest(settings.raw_dir)
    for url, filename in sorted(manifest.items()):
        path = settings.raw_dir / filename
        if path.exists():
            yield CachedPage(url=url, path=path, html=path.read_text(encoding="utf-8"))
