#!/bin/bash
set -e

if [ ! -f "$DATA_DIR/bm25_index.json" ]; then
    echo "==> Index not found. Starting background ingest..."
    gdb-rag ingest --reset &
fi

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --worker-class gthread \
    --threads 4 \
    --timeout 120 \
    "gdb_rag.server:app"
