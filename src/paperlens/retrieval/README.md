# retrieval

| File | Role |
|---|---|
| `retriever.py` | picks dense / bm25 / hybrid from config; the only thing pipelines call |
| `fusion.py` | Reciprocal Rank Fusion (rank-based, scale-free) |
| `query.py` | `QueryPlan` (texts per channel + cost/error) and the no-op `IdentityTransformer` |
| `reranker.py` | `Reranker` protocol, `CrossEncoderReranker` (bge-reranker-v2-m3, GPU), `LexicalReranker` (tests/CI), `build_reranker` (None when `rerank.enabled` is false) |

M1b adds query transforms (multi-query, HyDE) here.
Experiments: `retrieval.method`, `top_k`, `candidate_k`, `rrf_k`.
