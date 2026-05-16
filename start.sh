#!/bin/bash
set -e

if [ ! -f "$DATA_DIR/bm25_index.json" ]; then
    echo "==> Index not found. Running ingest (this may take 10-15 min)..."
    gdb-rag ingest --reset
    echo "==> Ingest complete."
fi

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --worker-class gthread \
    --threads 4 \
    --timeout 120 \
    "gdb_rag.server:app"
