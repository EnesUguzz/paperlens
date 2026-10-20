# notebooks

| Notebook | What it shows |
|---|---|
| `rag_walkthrough_tr.ipynb` | One QASPER question through every RAG step with real outputs: chunking, embeddings and cosine similarity, the Qdrant point, BM25 tokens, RRF by hand, a retrieval-only Recall@k eval, the exact prompt, the raw and structured LLM answer, and Answer-F1 (Turkish). |

Run `make notebook` and open http://localhost:8888. Needs the GPU (bge-m3) and, for the LLM cells, a running Ollama.
