# retrieval

| File | Role |
|---|---|
| `retriever.py` | picks dense / bm25 / hybrid from config; the only thing pipelines call |
| `fusion.py` | Reciprocal Rank Fusion (rank-based, scale-free) |

M1b adds `reranker.py` (cross-encoder) and `query_transforms.py` (multi-query, HyDE) here.
Experiments: `retrieval.method`, `top_k`, `candidate_k`, `rrf_k`.
