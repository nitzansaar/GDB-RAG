from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class Section:
    source_url: str
    page_title: str
    heading_path: tuple[str, ...]
    anchor: str | None
    text: str
    gdb_version: str | None = None


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
CONTENT_TAGS = {"p", "pre", "ul", "ol", "dl", "table", "blockquote"}
BLOCK_TAGS = tuple(sorted(HEADING_TAGS | CONTENT_TAGS))
VERSION_RE = re.compile(r"GDB\)?\s+Version\s+([^\s.]+(?:\.[^\s.]+)*)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                normalized.append("")
            previous_blank = True
            continue
        normalized.append(line)
        previous_blank = False
    return "\n".join(normalized).strip()


def page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return normalize_text(soup.title.string)
    heading = soup.find(HEADING_TAGS)
    if isinstance(heading, Tag):
        return normalize_text(heading.get_text(" ", strip=True))
    return "GDB Manual"


def gdb_version(soup: BeautifulSoup) -> str | None:
    match = VERSION_RE.search(soup.get_text(" ", strip=True))
    return match.group(1) if match else None


def clean_soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "html.parser")
    for selector in [
        "script",
        "style",
        "noscript",
        "header",
        "footer",
        "nav",
        ".header",
        ".footnote",
        ".contents",
    ]:
        for tag in soup.select(selector):
            tag.decompose()
    return soup


def tag_anchor(tag: Tag, source_url: str) -> str | None:
    candidate = tag.get("id") or tag.get("name")
    if not candidate and tag.parent and isinstance(tag.parent, Tag):
        candidate = tag.parent.get("id") or tag.parent.get("name")
    if not candidate:
        anchor = tag.find("a", attrs={"id": True})
        if isinstance(anchor, Tag):
            candidate = anchor.get("id")
    return urljoin(source_url, f"#{candidate}") if candidate else None


def block_text(tag: Tag) -> str:
    if tag.name == "pre":
        return tag.get_text("\n").rstrip()
    if tag.name in {"ul", "ol"}:
        items = [
            normalize_text(item.get_text(" ", strip=True))
            for item in tag.find_all("li", recursive=False)
        ]
        return "\n".join(f"- {item}" for item in items if item)
    if tag.name == "dl":
        parts: list[str] = []
        for child in tag.find_all(["dt", "dd"], recursive=False):
            text = normalize_text(child.get_text(" ", strip=True))
            if text:
                prefix = "- " if child.name == "dt" else "  "
                parts.append(f"{prefix}{text}")
        return "\n".join(parts)
    if tag.name == "table":
        rows: list[str] = []
        for row in tag.find_all("tr"):
            cells = [normalize_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["th", "td"])]
            cells = [cell for cell in cells if cell]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    return normalize_text(tag.get_text(" ", strip=True))


def parse_page(html: str, source_url: str) -> list[Section]:
    soup = clean_soup(html)
    title = page_title(soup)
    version = gdb_version(soup)
    body = soup.body or soup

    sections: list[Section] = []
    heading_stack: list[str] = []
    current_anchor: str | None = None
    current_blocks: list[str] = []

    def flush() -> None:
        nonlocal current_blocks
        text = normalize_text("\n\n".join(block for block in current_blocks if block))
        if text:
            sections.append(
                Section(
                    source_url=source_url,
                    page_title=title,
                    heading_path=tuple(heading_stack or [title]),
                    anchor=current_anchor or source_url,
                    text=text,
                    gdb_version=version,
                )
            )
        current_blocks = []

    for tag in body.find_all(BLOCK_TAGS):
        if not isinstance(tag, Tag):
            continue
        if tag.find_parent(BLOCK_TAGS):
            continue

        if tag.name in HEADING_TAGS:
            flush()
            level = int(tag.name[1])
            text = normalize_text(tag.get_text(" ", strip=True))
            if not text:
                continue
            heading_stack = heading_stack[: max(level - 1, 0)]
            heading_stack.append(text)
            current_anchor = tag_anchor(tag, source_url)
            continue

        text = block_text(tag)
        if text:
            current_blocks.append(text)

    flush()
    return sections
