"""Build indexes once per (split, text source, chunking, embedding) and reuse them.

Purpose: Embedding ~15k chunks takes minutes; experiments that only change retrieval or
LLM settings must not pay that again. The cache key hashes exactly the settings that
change the index, nothing else.
Flow: papers + AppConfig -> build_or_load() -> IndexBundle(chunks, dense, sparse)
-> retrieval/retriever.py.
Params: data.split, data.text_source, chunking.*, embedding.model_name,
embedding.passage_prefix, embedding.max_seq_length, retrieval.bm25_k1, retrieval.bm25_b.
Try: Run two configs that differ only in retrieval.top_k; the second logs 'index cache hit'.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import structlog
from qdrant_client import QdrantClient

from paperlens.config.models import AppConfig
from paperlens.domain import Chunk, Paper
from paperlens.index.dense import DenseIndex
from paperlens.index.embedder import Embedder
from paperlens.index.sparse import SparseIndex
from paperlens.ingestion.chunkers import chunk_paper

log = structlog.get_logger(__name__)

COLLECTION = "chunks"
CHUNKS_FILE = "chunks.jsonl"  # written last: its presence marks a complete build
KEY_LENGTH = 12


def index_key(cfg: AppConfig) -> str:
    payload = {
        "split": cfg.data.split,
        "source": cfg.data.text_source,
        "chunking": cfg.chunking.model_dump(mode="json"),
        "embedding": {
            "model": cfg.embedding.model_name,
            "passage_prefix": cfg.embedding.passage_prefix,
            "max_seq_length": cfg.embedding.max_seq_length,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:KEY_LENGTH]


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    dense: DenseIndex
    sparse: SparseIndex
    cache_hit: bool


def build_or_load(cfg: AppConfig, papers: Sequence[Paper], embedder: Embedder) -> IndexBundle:
    key = index_key(cfg)
    root = cfg.data.data_dir / "index" / key
    root.mkdir(parents=True, exist_ok=True)
    chunks_file = root / CHUNKS_FILE
    dense = DenseIndex(QdrantClient(path=str(root / "qdrant")), COLLECTION, embedder)

    if chunks_file.exists():
        lines = chunks_file.read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line]
        if dense.count() == len(chunks):
            log.info("index cache hit", key=key, chunks=len(chunks))
            sparse = SparseIndex(chunks, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
            return IndexBundle(chunks, dense, sparse, cache_hit=True)
        log.warning("index cache incomplete, rebuilding", key=key)

    chunks = [chunk for paper in papers for chunk in chunk_paper(paper, cfg.chunking)]
    log.info("building index", key=key, papers=len(papers), chunks=len(chunks))
    dense.build(chunks)
    chunks_file.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    sparse = SparseIndex(chunks, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
    return IndexBundle(chunks, dense, sparse, cache_hit=False)
