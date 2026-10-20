# retrieval

| File | Role |
|---|---|
| `retriever.py` | picks dense / bm25 / hybrid from config; the only thing pipelines call |
| `fusion.py` | Reciprocal Rank Fusion (rank-based, scale-free) |
| `query.py` | `QueryPlan` (texts per channel + cost/error) and the no-op `IdentityTransformer` |
| `reranker.py` | `Reranker` protocol (second-stage scorer hook; implementation in M1b) |

M1b adds the cross-encoder reranker and query transforms (multi-query, HyDE) here.
Experiments: `retrieval.method`, `top_k`, `candidate_k`, `rrf_k`.
