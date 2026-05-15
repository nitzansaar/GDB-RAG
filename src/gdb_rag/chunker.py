from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

from gdb_rag.parser import Section


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, str | int | None]


def token_count(text: str) -> int:
    return len(text.split())


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def chunk_id(section: Section, chunk_index: int, text: str) -> str:
    source = "|".join(
        [
            section.source_url,
            section.anchor or "",
            " / ".join(section.heading_path),
            str(chunk_index),
            content_hash(text)[:16],
        ]
    )
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


def split_text(text: str, token_limit: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= token_limit:
        return [text]

    chunks: list[str] = []
    start = 0
    step = max(token_limit - overlap, 1)
    while start < len(words):
        end = min(start + token_limit, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_section(section: Section, token_limit: int, overlap: int) -> list[Chunk]:
    heading = " > ".join(section.heading_path)
    prefix = f"{heading}\n\n" if heading else ""
    raw_chunks = split_text(section.text, token_limit=token_limit, overlap=overlap)
    chunks: list[Chunk] = []

    for index, raw_text in enumerate(raw_chunks):
        text = f"{prefix}{raw_text}".strip()
        chunks.append(
            Chunk(
                id=chunk_id(section, index, text),
                text=text,
                metadata={
                    "source_url": section.source_url,
                    "page_title": section.page_title,
                    "heading_path": heading,
                    "anchor": section.anchor,
                    "chunk_index": index,
                    "content_hash": content_hash(text),
                    "gdb_version": section.gdb_version,
                    "token_count": token_count(text),
                },
            )
        )
    return chunks


def chunk_sections(
    sections: Iterable[Section],
    token_limit: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        chunks.extend(chunk_section(section, token_limit=token_limit, overlap=overlap))
    return chunks
