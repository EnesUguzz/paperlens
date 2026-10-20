# pipelines

| File | Role |
|---|---|
| `baseline.py` | naive RAG: retrieve -> numbered context -> one structured LLM call |
| `factory.py` | `build_pipeline(cfg, bundle, llm, titles, reranker=None)` (builds the reranker from `rerank.*` when not given): the one place evals and trace build a pipeline |

Every pipeline returns `PipelineResult(answer, retrieved, retrieval_ms, query_error)`, so evals can
score any pipeline the same way. The prompt lives in `prompts/` and is a parameter
(`generation.prompt_template`) - edit it like code and compare reports.
