# notebooks

| Notebook | What it shows |
|---|---|
| `rag_walkthrough_tr.ipynb` | One QASPER question through every RAG step with real outputs: chunking, embeddings and cosine similarity, the Qdrant point, BM25 tokens, RRF by hand, a retrieval-only Recall@k eval, the exact prompt, the raw and structured LLM answer, and Answer-F1 (Turkish). |
| `m1b_walkthrough_tr.ipynb` | Every M1b addition on the same question with real outputs: the results table with paired-bootstrap CIs for Answer-F1 and Evidence-F1, fixed vs section vs semantic chunks, chunk vs paragraph citations and the Evidence-F1 ceiling, the reranker pool, multi-query / HyDE query plans, Docling evidence alignment, LLM-as-judge rules, and per-question baseline vs candidate changes (Turkish; summary in `docs/m1b_ozet_tr.md`). |
| `m2_walkthrough_tr.ipynb` | The M2 LangGraph agent: the graph drawn from code, the synthetic conversations, the paper vote by hand, one conversation streamed node by node, an interrupt/resume by hand, the follow-up demo, the three-system results by turn type, and reading MLflow traces (Turkish; summary in `docs/m2_ozet_tr.md`). |

Run `make notebook` and open http://localhost:8888. Needs the GPU (bge-m3) and, for the LLM cells, a running Ollama.
