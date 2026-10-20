"""Second-stage reranking: rescore a candidate pool with a cross-encoder and keep the best.

Purpose: A bi-encoder (dense) or BM25 scores query and chunk separately; a cross-encoder
reads them together, which is slower but much sharper. It is run only on the small pool the
first stage returns. Off by default.
Flow: Retriever pool (candidate_k chunks) -> Reranker.rerank(query, pool, top_n) -> top_n
ScoredChunk, best first, with score = reranker score. Equal scores keep first-stage order.
Params: rerank.enabled, rerank.model_name ('lexical' = test fake), rerank.batch_size,
rerank.max_length, rerank.device; the pool size is retrieval.candidate_k.
Try: Compare retrieval.method dense vs hybrid with rerank on (experiments/configs/m1b):
does a hybrid pool hand the reranker better candidates than a dense one?
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

from paperlens.config.models import RerankConfig
from paperlens.domain import ScoredChunk

LEXICAL_MODEL_NAME = "lexical"
_TOKEN = re.compile(r"\w+")


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...


def _best_first(
    candidates: Sequence[ScoredChunk], scores: Sequence[float], top_n: int
) -> list[ScoredChunk]:
    """Replace each score by the reranker's and sort best first; sorted() is stable, so ties
    keep the first-stage order."""
    rescored = [
        ScoredChunk(chunk=c.chunk, score=float(s)) for c, s in zip(candidates, scores, strict=True)
    ]
    return sorted(rescored, key=lambda sc: sc.score, reverse=True)[:top_n]


class LexicalReranker:
    """Score = number of distinct words shared by query and chunk. Tests/CI only."""

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        words = set(_TOKEN.findall(query.lower()))
        scores = [len(words & set(_TOKEN.findall(c.chunk.text.lower()))) for c in candidates]
        return _best_first(candidates, scores, top_n)


class CrossEncoderReranker:
    def __init__(self, cfg: RerankConfig) -> None:
        from sentence_transformers import CrossEncoder  # heavy; only when really used

        self._cfg = cfg
        self._model: Any = CrossEncoder(
            cfg.model_name, max_length=cfg.max_length, device=cfg.device
        )
        if cfg.device.startswith("cuda"):
            self._model.model.half()  # fp16 halves VRAM, same reasoning as the embedder

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        if not candidates:
            return []
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self._cfg.batch_size)
        return _best_first(candidates, [float(s) for s in scores], top_n)


def build_reranker(cfg: RerankConfig) -> Reranker | None:
    if not cfg.enabled:
        return None
    if cfg.model_name == LEXICAL_MODEL_NAME:
        return LexicalReranker()
    return CrossEncoderReranker(cfg)
