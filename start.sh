#!/bin/bash
set -e

# On first run, copy the pre-built index from the image to the persistent volume.
# Subsequent restarts skip this (index already on volume).
if [ ! -f "$DATA_DIR/bm25_index.json" ]; then
    echo "==> Copying index from image to volume..."
    mkdir -p "$DATA_DIR"
    cp /app/index/bm25_index.json "$DATA_DIR/"
    cp -r /app/index/chroma "$DATA_DIR/"
    echo "==> Done."
fi

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --worker-class gthread \
    --threads 4 \
    --timeout 120 \
    "gdb_rag.server:app"
