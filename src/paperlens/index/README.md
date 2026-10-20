# index

| File | Role |
|---|---|
| `embedder.py` | `SentenceTransformerEmbedder` (GPU, bge-m3) and `HashingEmbedder` (tests/CI) |
| `dense.py` | Qdrant collection: build, count, search with optional `paper_id` filter |
| `sparse.py` | BM25 (bm25s), one index per paper built on demand + a global one |
| `store.py` | builds once per (split, text source, chunking, embedding) and caches under `data/index/<key>`; `semantic_*` fields only key semantic indexes, `pdf_docling` keys include a digest of the parsed papers; a cache whose chunks lack paragraph `segments` is rebuilt in place |

Experiments: embedding model / prefixes, BM25 k1 and b. Changing chunking or embedding
settings creates a new cache key (a new index); retrieval/LLM settings reuse the cache.
