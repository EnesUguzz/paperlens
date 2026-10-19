"""BM25 keyword index (bm25s), built per paper on demand plus one global index.

Purpose: Exact-term retrieval for dataset names, metric names and acronyms that dense
embeddings blur. Per-paper indexes make single-paper search exact and cheap.
Flow: chunks -> SparseIndex -> search(query, k, paper_id) -> ScoredChunk list.
Params: retrieval.bm25_k1, retrieval.bm25_b; retrieval.top_k / candidate_k arrive as k.
Try: With chunking.strategy='section' (uneven chunk lengths) raise bm25_b to 0.9 and see
whether long chunks stop dominating the top ranks.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import bm25s

from paperlens.domain import Chunk, ScoredChunk

STOPWORDS = "en"


class SparseIndex:
    def __init__(self, chunks: Sequence[Chunk], k1: float, b: float) -> None:
        self._k1 = k1
        self._b = b
        self._all = list(chunks)
        self._by_paper: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            self._by_paper.setdefault(chunk.paper_id, []).append(chunk)
        self._cache: dict[str | None, Any] = {}

    def _docs(self, paper_id: str | None) -> list[Chunk]:
        return self._all if paper_id is None else self._by_paper.get(paper_id, [])

    def _retriever(self, paper_id: str | None) -> Any:
        if paper_id not in self._cache:
            retriever = bm25s.BM25(k1=self._k1, b=self._b)
            corpus = bm25s.tokenize(
                [c.text for c in self._docs(paper_id)], stopwords=STOPWORDS, show_progress=False
            )
            retriever.index(corpus, show_progress=False)
            self._cache[paper_id] = retriever
        return self._cache[paper_id]

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        docs = self._docs(paper_id)
        if not docs:
            return []
        retriever = self._retriever(paper_id)
        tokens = bm25s.tokenize([query], stopwords=STOPWORDS, return_ids=False, show_progress=False)
        known = [t for t in tokens[0] if t in retriever.vocab_dict]
        if not known:  # only stopwords or words absent from these chunks
            return []
        indices, scores = retriever.retrieve([known], k=min(k, len(docs)), show_progress=False)
        return [
            ScoredChunk(chunk=docs[int(i)], score=float(s))
            for i, s in zip(indices[0], scores[0], strict=True)
            if s > 0
        ]
