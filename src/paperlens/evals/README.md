# evals

| File | Role |
|---|---|
| `answer_metrics.py` | official-style QASPER Answer-F1 / Evidence-F1, refusal helpers |
| `retrieval_metrics.py` | Recall@k, MRR, nDCG@k against gold paragraph ids |
| `records.py` | one `EvalRecord` per question (scores + what was retrieved/cited) |
| `aggregate.py` | records -> flat metric dict (means, per-type, refusal, latency, tokens) |
| `report.py` | Markdown report: changed params vs reference, metric deltas, worst examples |
| `summary.py` | one table across all runs (the ablation table for the README) |
| `stats.py` | paired bootstrap CI for run-vs-run answer-F1 differences |
| `provenance.py` | meta.* params: prompt/data hashes, git sha, model digest |
| `rescore.py` | recompute metrics/reports of saved runs without the LLM (`paperlens rescore`) |
| `tracking.py` | MLflow logging |
| `runner.py` | wires everything for one YAML config |

Metrics have no parameters on purpose: if the ruler changes, runs stop being comparable.
