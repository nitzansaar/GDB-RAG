from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    source_url: str = "https://sourceware.org/gdb/current/onlinedocs/gdb.html/#SEC_Contents"
    collection_name: str = "gdb_manual"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_token_limit: int = 900
    chunk_overlap: int = 120
    request_timeout: int = 30
    llm_model: str = "llama-3.1-8b-instant"
    top_k: int = 3
    user_agent: str = "gdb-rag-ingester/0.1"
    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def model_cache_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def bm25_path(self) -> Path:
        return self.data_dir / "bm25_index.json"


DEFAULT_SETTINGS = Settings()
