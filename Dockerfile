FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e .

ENV MODEL_CACHE_DIR=/app/model_cache

# Bake the embedding model into the image
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer
SentenceTransformer("BAAI/bge-small-en-v1.5", cache_folder="/app/model_cache")
EOF

# Run the full ingest during build (build machine has 8 GB RAM — no OOM risk).
# Only the index files are kept; raw HTML cache is discarded to limit image size.
RUN DATA_DIR=/tmp/build_data gdb-rag ingest --reset && \
    mkdir -p /app/index && \
    cp /tmp/build_data/bm25_index.json /app/index/ && \
    cp -r /tmp/build_data/chroma /app/index/ && \
    rm -rf /tmp/build_data

ENV DATA_DIR=/data
ENV PORT=8080

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8080

CMD ["./start.sh"]
