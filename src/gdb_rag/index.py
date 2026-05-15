from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.errors import NotFoundError
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from gdb_rag.chunker import Chunk
from gdb_rag.config import Settings


def batch(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    return sanitized


def configure_model_cache(settings: Settings) -> None:
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(settings.model_cache_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(settings.model_cache_dir / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(settings.model_cache_dir / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_embedding_model(settings: Settings, local_files_only: bool = False) -> SentenceTransformer:
    configure_model_cache(settings)
    return SentenceTransformer(
        settings.embedding_model,
        cache_folder=str(settings.model_cache_dir),
        local_files_only=local_files_only,
    )


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for chunk in chunks:
            record = {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Chunk(
                    id=record["id"],
                    text=record["text"],
                    metadata=record["metadata"],
                )
            )
    return chunks


def get_collection(settings: Settings, reset: bool = False) -> Collection:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except (ValueError, NotFoundError):
            pass
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine", "source": settings.source_url},
    )


def build_index(
    chunks: list[Chunk],
    settings: Settings,
    reset: bool = False,
    batch_size: int = 64,
) -> Collection:
    model = load_embedding_model(settings)
    collection = get_collection(settings, reset=reset)

    for chunk_batch in tqdm(list(batch(chunks, batch_size)), desc="Embedding chunks", unit="batch"):
        documents = [chunk.text for chunk in chunk_batch]
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()
        collection.upsert(
            ids=[chunk.id for chunk in chunk_batch],
            documents=documents,
            embeddings=embeddings,
            metadatas=[sanitize_metadata(chunk.metadata) for chunk in chunk_batch],
        )
    return collection


def query_index(
    question: str,
    settings: Settings,
    top_k: int = 5,
) -> dict[str, Any]:
    model = load_embedding_model(settings, local_files_only=True)
    collection = get_collection(settings, reset=False)
    embedding = model.encode([question], normalize_embeddings=True).tolist()[0]
    return collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
