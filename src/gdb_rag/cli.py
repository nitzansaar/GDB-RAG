from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from textwrap import shorten

from gdb_rag.chunker import chunk_sections
from gdb_rag.config import DEFAULT_SETTINGS, Settings
from gdb_rag.crawler import crawl_manual, iter_cached_pages
from gdb_rag.index import build_index, get_collection, query_index, save_chunks
from gdb_rag.llm import generate_answer
from gdb_rag.parser import parse_page


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = DEFAULT_SETTINGS
    updates = {}
    for name in ["source_url", "collection_name", "embedding_model"]:
        value = getattr(args, name, None)
        if value:
            updates[name] = value
    if getattr(args, "data_dir", None):
        updates["data_dir"] = Path(args.data_dir)
    if getattr(args, "chunk_token_limit", None):
        updates["chunk_token_limit"] = args.chunk_token_limit
    if getattr(args, "chunk_overlap", None) is not None:
        updates["chunk_overlap"] = args.chunk_overlap
    return replace(settings, **updates)


def ingest(args: argparse.Namespace) -> None:
    settings = settings_from_args(args)
    if args.cached_only:
        pages = list(iter_cached_pages(settings))
    else:
        pages = crawl_manual(
            settings=settings,
            limit=args.limit,
            refresh_cache=args.refresh_cache,
        )

    sections = []
    for page in pages:
        sections.extend(parse_page(page.html, page.url))

    chunks = chunk_sections(
        sections,
        token_limit=settings.chunk_token_limit,
        overlap=settings.chunk_overlap,
    )
    save_chunks(chunks, settings.chunks_path)
    build_index(chunks, settings=settings, reset=args.reset, batch_size=args.batch_size)

    print(f"Pages: {len(pages)}")
    print(f"Sections: {len(sections)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chroma collection: {settings.collection_name}")
    print(f"Chroma path: {settings.chroma_dir}")


def query(args: argparse.Namespace) -> None:
    settings = settings_from_args(args)
    results = query_index(args.question, settings=settings, top_k=args.top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=False),
        start=1,
    ):
        title = metadata.get("page_title") or "Untitled"
        heading = metadata.get("heading_path") or title
        url = metadata.get("anchor") or metadata.get("source_url") or ""
        snippet = shorten(" ".join(document.split()), width=args.snippet_chars, placeholder="...")
        print(f"\n{index}. {title}")
        print(f"   Heading: {heading}")
        print(f"   Distance: {distance:.4f}")
        print(f"   URL: {url}")
        print(f"   {snippet}")


def ask(args: argparse.Namespace) -> None:
    settings = settings_from_args(args)
    model = args.model or settings.ollama_model
    results = query_index(args.question, settings=settings, top_k=args.top_k)
    chunks = results.get("documents", [[]])[0]
    if not chunks:
        print("No relevant chunks found.")
        return
    answer = generate_answer(args.question, chunks, model=model)
    print(answer)


def stats(args: argparse.Namespace) -> None:
    settings = settings_from_args(args)
    collection = get_collection(settings, reset=False)
    print(f"Collection: {settings.collection_name}")
    print(f"Items: {collection.count()}")
    print(f"Chroma path: {settings.chroma_dir}")
    print(f"Chunks file: {settings.chunks_path}")


def serve(args: argparse.Namespace) -> None:
    from gdb_rag.server import app  # noqa: PLC0415
    host = args.host
    port = args.port
    print(f"Starting GDB RAG server on http://{host}:{port}")
    app.run(host=host, port=port, debug=args.debug, threaded=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and query a GDB manual vector database.")
    parser.add_argument("--data-dir", default=None, help="Directory for generated raw/chroma data.")
    parser.add_argument("--source-url", default=None, help="GDB manual contents URL.")
    parser.add_argument("--collection-name", default=None, help="ChromaDB collection name.")
    parser.add_argument("--embedding-model", default=None, help="SentenceTransformers model name.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Crawl, chunk, embed, and index the manual.")
    ingest_parser.add_argument("--limit", type=int, default=None, help="Maximum pages to crawl.")
    ingest_parser.add_argument("--reset", action="store_true", help="Reset the Chroma collection before indexing.")
    ingest_parser.add_argument("--refresh-cache", action="store_true", help="Re-download pages even if cached.")
    ingest_parser.add_argument("--cached-only", action="store_true", help="Index already cached pages after crawling.")
    ingest_parser.add_argument("--batch-size", type=int, default=64, help="Embedding/indexing batch size.")
    ingest_parser.add_argument("--chunk-token-limit", type=int, default=None, help="Approximate tokens per chunk.")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=None, help="Approximate token overlap.")
    ingest_parser.set_defaults(func=ingest)

    query_parser = subparsers.add_parser("query", help="Query the local vector database.")
    query_parser.add_argument("question", help="Question to retrieve GDB manual chunks for.")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    query_parser.add_argument("--snippet-chars", type=int, default=700, help="Maximum snippet length.")
    query_parser.set_defaults(func=query)

    ask_parser = subparsers.add_parser("ask", help="Ask a question; retrieves chunks and generates an LLM answer.")
    ask_parser.add_argument("question", help="Question to answer using the GDB manual.")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    ask_parser.add_argument("--model", default=None, help="Ollama model name (default: llama3.2).")
    ask_parser.set_defaults(func=ask)

    stats_parser = subparsers.add_parser("stats", help="Show ChromaDB collection stats.")
    stats_parser.set_defaults(func=stats)

    serve_parser = subparsers.add_parser("serve", help="Start the web chatbot server.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=5000)
    serve_parser.add_argument("--debug", action="store_true")
    serve_parser.set_defaults(func=serve)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
