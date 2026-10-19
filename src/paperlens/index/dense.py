"""Dense vector index on Qdrant (embedded local mode; no server needed in M1).

Purpose: Stores one vector per chunk with the chunk itself as payload, and searches with
an optional paper_id filter (single-paper mode).
Flow: chunks + Embedder -> build() -> Qdrant collection; query -> search() -> ScoredChunk.
Params: embedding.* (through the Embedder); retrieval.top_k / candidate_k arrive as k.
Try: Search one question with and without paper_id: the gap shows how much of corpus
mode's difficulty is just finding the right paper (that is what M2's router fixes).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient, models

from paperlens.domain import Chunk, ScoredChunk
from paperlens.index.embedder import Embedder

# Qdrant point ids must be ints or UUIDs; uuid5 turns chunk ids into stable UUIDs.
_ID_NAMESPACE = uuid.UUID("6f1c2a1e-0000-4000-8000-000000000000")
UPSERT_BATCH = 256


class DenseIndex:
    def __init__(self, client: QdrantClient, collection: str, embedder: Embedder) -> None:
        self._client = client
        self._collection = collection
        self._embedder = embedder

    def build(self, chunks: Sequence[Chunk]) -> None:
        if self._client.collection_exists(self._collection):
            self._client.delete_collection(self._collection)
        self._client.create_collection(
            self._collection,
            vectors_config=models.VectorParams(
                size=self._embedder.dim, distance=models.Distance.COSINE
            ),
        )
        for start in range(0, len(chunks), UPSERT_BATCH):
            batch = chunks[start : start + UPSERT_BATCH]
            vectors = self._embedder.embed_passages([c.text for c in batch])
            self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid5(_ID_NAMESPACE, c.chunk_id)),
                        vector=v.tolist(),
                        payload=c.model_dump(),
                    )
                    for c, v in zip(batch, vectors, strict=True)
                ],
            )

    def count(self) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(self._collection, exact=True).count

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        query_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(key="paper_id", match=models.MatchValue(value=paper_id))
                ]
            )
            if paper_id
            else None
        )
        hits = self._client.query_points(
            self._collection,
            query=self._embedder.embed_query(query).tolist(),
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        ).points
        return [
            ScoredChunk(chunk=Chunk.model_validate(h.payload or {}), score=float(h.score))
            for h in hits
        ]

    def close(self) -> None:
        self._client.close()
