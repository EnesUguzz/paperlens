# pipelines

| File | Role |
|---|---|
| `baseline.py` | naive RAG: retrieve -> numbered context -> one structured LLM call |

Every pipeline returns `PipelineResult(answer, retrieved, retrieval_ms)`, so evals can
score any pipeline the same way. The prompt lives in `prompts/` and is a parameter
(`generation.prompt_template`) - edit it like code and compare reports.
