"""Build indexes once per (split, text source, chunking, embedding) and reuse them.

Purpose: Embedding ~7k chunks (validation split at fixed 200 words) takes minutes;
experiments that only change retrieval or LLM settings must not pay that again. The cache
key hashes exactly the settings that change the index, nothing else: semantic_* chunking
fields only count for strategy 'semantic', and for 'pdf_docling' a digest of the parsed
papers' content is added (a re-parse or a newly parsed paper builds a new index).
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
SEMANTIC_PREFIX = "semantic_"  # chunking fields read only by the 'semantic' chunker


def _papers_digest(papers: Sequence[Paper]) -> str:
    """sha256 over the papers' content, so a different parsed set or parse is a new index."""
    digest = hashlib.sha256()
    for paper in sorted(papers, key=lambda p: p.paper_id):
        digest.update(paper.model_dump_json().encode())
    return digest.hexdigest()


def index_key(cfg: AppConfig, papers: Sequence[Paper]) -> str:
    chunking = cfg.chunking.model_dump(mode="json")
    if cfg.chunking.strategy != "semantic":  # keeps fixed/section keys stable across new fields
        chunking = {k: v for k, v in chunking.items() if not k.startswith(SEMANTIC_PREFIX)}
    payload: dict[str, object] = {
        "split": cfg.data.split,
        "source": cfg.data.text_source,
        "chunking": chunking,
        "embedding": {
            "model": cfg.embedding.model_name,
            "passage_prefix": cfg.embedding.passage_prefix,
            "max_seq_length": cfg.embedding.max_seq_length,
        },
    }
    # QASPER text is fixed per split; parsed PDFs change when prepare-pdfs runs again
    if cfg.data.text_source == "pdf_docling":
        payload["papers"] = _papers_digest(papers)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:KEY_LENGTH]


@dataclass
class IndexBundle:
    chunks: list[Chunk]
    dense: DenseIndex
    sparse: SparseIndex
    cache_hit: bool


def build_or_load(cfg: AppConfig, papers: Sequence[Paper], embedder: Embedder) -> IndexBundle:
    key = index_key(cfg, papers)
    root = cfg.data.data_dir / "index" / key
    root.mkdir(parents=True, exist_ok=True)
    chunks_file = root / CHUNKS_FILE
    dense = DenseIndex(QdrantClient(path=str(root / "qdrant")), COLLECTION, embedder)

    if chunks_file.exists():
        lines = chunks_file.read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line]
        if any(c.paragraph_ids and not c.segments for c in chunks):
            log.warning("index cache predates segments, rebuilding", key=key)
        elif dense.count() == len(chunks):
            log.info("index cache hit", key=key, chunks=len(chunks))
            sparse = SparseIndex(chunks, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
            return IndexBundle(chunks, dense, sparse, cache_hit=True)
        else:
            log.warning("index cache incomplete, rebuilding", key=key)

    chunks = [c for paper in papers for c in chunk_paper(paper, cfg.chunking, embedder)]
    log.info("building index", key=key, papers=len(papers), chunks=len(chunks))
    dense.build(chunks)
    chunks_file.write_text("".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8")
    sparse = SparseIndex(chunks, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
    return IndexBundle(chunks, dense, sparse, cache_hit=False)
