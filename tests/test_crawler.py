from gdb_rag.crawler import canonical_url, extract_manual_links, is_manual_page


SOURCE_URL = "https://sourceware.org/gdb/current/onlinedocs/gdb.html/#SEC_Contents"


def test_canonical_url_removes_fragment() -> None:
    assert canonical_url(SOURCE_URL) == "https://sourceware.org/gdb/current/onlinedocs/gdb.html"


def test_is_manual_page_rejects_external_links() -> None:
    assert is_manual_page("https://sourceware.org/gdb/current/onlinedocs/gdb/Running.html", SOURCE_URL)
    assert not is_manual_page("https://example.com/gdb/current/onlinedocs/Running.html", SOURCE_URL)
    assert not is_manual_page("https://sourceware.org/gdb/current/onlinedocs/gdb.pdf", SOURCE_URL)
    assert not is_manual_page("https://sourceware.org/gdb/current/onlinedocs/index.html", SOURCE_URL)


def test_extract_manual_links_normalizes_relative_links() -> None:
    html = """
    <a href="Running.html#Running">Running</a>
    <a href="../index.html">Outside manual</a>
    <a href="https://example.com/Remote.html">External</a>
    """
    links = extract_manual_links(
        html,
        "https://sourceware.org/gdb/current/onlinedocs/gdb.html",
        SOURCE_URL,
    )
    assert links == ["https://sourceware.org/gdb/current/onlinedocs/gdb/Running.html"]
