"""Turn text into vectors: a real GPU embedder and a tiny deterministic one for tests.

Purpose: Hides the embedding model behind a small protocol, so the index never cares
which model (or fake) is in use.
Flow: chunk texts / queries -> Embedder -> L2-normalised float32 vectors -> index/dense.py.
Params: embedding.model_name, embedding.batch_size, embedding.device,
embedding.max_seq_length, embedding.query_prefix, embedding.passage_prefix.
Try: Swap bge-m3 for 'intfloat/e5-large-v2' with prefixes 'query: ' / 'passage: ' and
compare Recall@5; then run e5 without prefixes to see what they are worth.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from paperlens.config.models import EmbeddingConfig

HASHING_MODEL_NAME = "hashing"
HASHING_DIM = 256
_TOKEN = re.compile(r"\w+")


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, text: str) -> NDArray[np.float32]: ...


class HashingEmbedder:
    """Bag-of-words hashed into `dim` buckets: shared words -> similar vectors. Tests/CI only."""

    def __init__(self, dim: int = HASHING_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN.findall(text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0] = 1.0  # cosine distance is undefined for a zero vector
            return vec
        return vec / norm

    def embed_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._vector(text)


class SentenceTransformerEmbedder:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer  # heavy; only when really used

        self._cfg = cfg
        self._model: Any = SentenceTransformer(cfg.model_name, device=cfg.device)
        if cfg.device.startswith("cuda"):
            self._model.half()  # fp16 halves VRAM, no measurable quality loss for retrieval
        self._model.max_seq_length = cfg.max_seq_length
        # sentence-transformers renamed this method; support both old and new releases
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        dim = get_dim() if get_dim else self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(f"{cfg.model_name} does not report an embedding dimension")
        self.dim = int(dim)

    def _encode(self, texts: list[str], show_progress: bool) -> NDArray[np.float32]:
        vectors = self._model.encode(
            texts,
            batch_size=self._cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: Sequence[str]) -> NDArray[np.float32]:
        prefixed = [self._cfg.passage_prefix + t for t in texts]
        return self._encode(prefixed, show_progress=len(prefixed) > self._cfg.batch_size)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self._encode([self._cfg.query_prefix + text], False)[0]
        return vector


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.model_name == HASHING_MODEL_NAME:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(cfg)
