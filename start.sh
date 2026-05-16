#!/bin/bash
set -e

exec gunicorn \
    --bind "0.0.0.0:${PORT:-8080}" \
    --workers 1 \
    --worker-class gthread \
    --threads 4 \
    --timeout 120 \
    "gdb_rag.server:app"
