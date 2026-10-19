import numpy as np
import pytest

from paperlens.config.models import EmbeddingConfig
from paperlens.index.embedder import build_embedder


@pytest.mark.gpu
def test_bge_m3_on_gpu() -> None:
    emb = build_embedder(EmbeddingConfig())
    vecs = emb.embed_passages(["retrieval for question answering", "a cat on a mat"])
    q = emb.embed_query("question answering retrieval")
    assert vecs.shape == (2, emb.dim)
    assert float(vecs[0] @ q) > float(vecs[1] @ q)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-2)
