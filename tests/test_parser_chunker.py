from pathlib import Path

from gdb_rag.chunker import chunk_sections
from gdb_rag.parser import parse_page


FIXTURE = Path(__file__).parent / "fixtures" / "sample_gdb_page.html"
SOURCE_URL = "https://sourceware.org/gdb/current/onlinedocs/Breakpoints.html"


def test_parse_page_preserves_headings_examples_and_metadata() -> None:
    sections = parse_page(FIXTURE.read_text(encoding="utf-8"), SOURCE_URL)

    assert len(sections) == 2
    assert sections[0].page_title == "Breakpoints (Debugging with GDB)"
    assert sections[0].heading_path == ("5.1 Breakpoints, Watchpoints, and Catchpoints",)
    assert sections[0].anchor == f"{SOURCE_URL}#Breakpoints"
    assert sections[0].gdb_version == "18.0.50.20260514-git"

    breakpoint_section = sections[1]
    assert breakpoint_section.heading_path == (
        "5.1 Breakpoints, Watchpoints, and Catchpoints",
        "5.1.1 Setting Breakpoints",
    )
    assert "(gdb) break main" in breakpoint_section.text
    assert "- Use condition for conditional breakpoints." in breakpoint_section.text


def test_chunk_sections_adds_stable_metadata() -> None:
    sections = parse_page(FIXTURE.read_text(encoding="utf-8"), SOURCE_URL)
    chunks = chunk_sections(sections, token_limit=20, overlap=5)

    assert chunks
    assert chunks[0].id
    assert chunks[0].metadata["source_url"] == SOURCE_URL
    assert chunks[0].metadata["heading_path"]
    assert chunks[0].metadata["content_hash"]
    assert chunks[0].metadata["token_count"] > 0
