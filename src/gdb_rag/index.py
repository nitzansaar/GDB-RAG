from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable

import chromadb
from chromadb.errors import NotFoundError
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from gdb_rag.chunker import Chunk
from gdb_rag.config import Settings

_BGE_MODELS = frozenset({
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
})

_bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}
_bm25_lock = threading.Lock()

_model_cache: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()

_chroma_clients: dict[str, chromadb.PersistentClient] = {}
_chroma_lock = threading.Lock()


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
    key = settings.embedding_model
    with _model_lock:
        if key not in _model_cache:
            _model_cache[key] = SentenceTransformer(
                settings.embedding_model,
                cache_folder=str(settings.model_cache_dir),
                local_files_only=local_files_only,
            )
        return _model_cache[key]


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
            chunks.append(Chunk(id=record["id"], text=record["text"], metadata=record["metadata"]))
    return chunks


def _get_chroma_client(settings: Settings) -> chromadb.PersistentClient:
    key = str(settings.chroma_dir)
    with _chroma_lock:
        if key not in _chroma_clients:
            settings.chroma_dir.mkdir(parents=True, exist_ok=True)
            _chroma_clients[key] = chromadb.PersistentClient(path=str(settings.chroma_dir))
        return _chroma_clients[key]


def get_collection(settings: Settings, reset: bool = False) -> Collection:
    client = _get_chroma_client(settings)
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except (ValueError, NotFoundError):
            pass
    return client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine", "source": settings.source_url},
    )


def tokenize_for_bm25(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z][a-z0-9_]*", text.lower()) if len(t) > 1]


def _serialize_bm25(bm25: BM25Okapi) -> dict:
    return {
        "k1": bm25.k1,
        "b": bm25.b,
        "epsilon": bm25.epsilon,
        "corpus_size": bm25.corpus_size,
        "avgdl": bm25.avgdl,
        "doc_freqs": bm25.doc_freqs,
        "idf": bm25.idf,
        "doc_len": bm25.doc_len,
    }


def _deserialize_bm25(data: dict) -> BM25Okapi:
    bm25: BM25Okapi = object.__new__(BM25Okapi)
    bm25.k1 = data["k1"]
    bm25.b = data["b"]
    bm25.epsilon = data["epsilon"]
    bm25.corpus_size = data["corpus_size"]
    bm25.avgdl = data["avgdl"]
    bm25.doc_freqs = data["doc_freqs"]
    bm25.idf = data["idf"]
    bm25.doc_len = data["doc_len"]
    bm25.tokenizer = None
    return bm25


def build_bm25_index(chunks: list[Chunk], settings: Settings) -> None:
    tokenized = [tokenize_for_bm25(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    chunk_ids = [chunk.id for chunk in chunks]
    settings.bm25_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"bm25": _serialize_bm25(bm25), "chunk_ids": chunk_ids}
    with settings.bm25_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    cache_key = str(settings.bm25_path)
    with _bm25_lock:
        _bm25_cache.pop(cache_key, None)


def load_bm25_index(settings: Settings) -> tuple[BM25Okapi, list[str]]:
    cache_key = str(settings.bm25_path)
    with _bm25_lock:
        if cache_key not in _bm25_cache:
            with settings.bm25_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            _bm25_cache[cache_key] = (_deserialize_bm25(payload["bm25"]), payload["chunk_ids"])
        return _bm25_cache[cache_key]


def _apply_query_prefix(question: str, settings: Settings) -> str:
    if settings.embedding_model in _BGE_MODELS:
        return settings.bge_query_prefix + question
    return question


def _reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


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

    build_bm25_index(chunks, settings)
    return collection


def query_index(
    question: str,
    settings: Settings,
    top_k: int = 5,
) -> dict[str, Any]:
    overfetch = top_k * 4

    model = load_embedding_model(settings, local_files_only=True)
    collection = get_collection(settings, reset=False)
    prefixed = _apply_query_prefix(question, settings)
    embedding = model.encode([prefixed], normalize_embeddings=True).tolist()[0]

    vector_results = collection.query(
        query_embeddings=[embedding],
        n_results=overfetch,
        include=["documents", "metadatas", "distances"],
    )
    vector_ids: list[str] = vector_results["ids"][0]
    id_to_doc: dict[str, str] = dict(zip(vector_ids, vector_results["documents"][0]))
    id_to_meta: dict[str, dict] = dict(zip(vector_ids, vector_results["metadatas"][0]))
    id_to_dist: dict[str, float] = dict(zip(vector_ids, vector_results["distances"][0]))

    bm25, bm25_chunk_ids = load_bm25_index(settings)
    query_tokens = tokenize_for_bm25(question)
    bm25_scores = bm25.get_scores(query_tokens).tolist()
    bm25_top_ids = [
        cid for cid, _ in sorted(zip(bm25_chunk_ids, bm25_scores), key=lambda x: x[1], reverse=True)[:overfetch]
    ]

    fused = _reciprocal_rank_fusion([vector_ids, bm25_top_ids])
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    missing = [cid for cid in top_ids if cid not in id_to_doc]
    if missing:
        got = collection.get(ids=missing, include=["documents", "metadatas"])
        for cid, doc, meta in zip(got["ids"], got["documents"] or [], got["metadatas"] or []):
            id_to_doc[cid] = doc
            id_to_meta[cid] = meta
            id_to_dist[cid] = 0.0

    return {
        "ids": [top_ids],
        "documents": [[id_to_doc.get(cid, "") for cid in top_ids]],
        "metadatas": [[id_to_meta.get(cid, {}) for cid in top_ids]],
        "distances": [[id_to_dist.get(cid, 0.0) for cid in top_ids]],
    }
