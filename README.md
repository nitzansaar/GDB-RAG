# GDB RAG Vector Database

Build a local vector database from the GNU GDB online manual for retrieval-augmented generation experiments.

The pipeline crawls the manual, cleans each page into section-aware text, chunks by headings, embeds chunks locally with SentenceTransformers, and stores them in a persistent ChromaDB collection.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The first ingest downloads the embedding model configured in `src/gdb_rag/config.py`.

## Ingest

Run a small crawl first:

```bash
gdb-rag ingest --limit 10 --reset
```

Then build the full database:

```bash
gdb-rag ingest --reset
```

Generated files are written to:

- `data/raw/`: cached GDB manual HTML pages
- `data/chunks.jsonl`: parsed chunks and metadata
- `data/chroma/`: persistent ChromaDB collection
- `data/models/`: local SentenceTransformers model cache

## Query

```bash
gdb-rag query "How do I set a conditional breakpoint?"
gdb-rag query "How does reverse execution work?" --top-k 5
gdb-rag query "How do I connect to a remote target?"
```

Each result prints the source title, manual URL, heading path, distance score, and a text snippet.

## Refresh

Use `--refresh-cache` to re-download HTML and `--reset` to recreate the Chroma collection:

```bash
gdb-rag ingest --refresh-cache --reset
```

## Tests

```bash
pytest
```
# GDB-RAG
