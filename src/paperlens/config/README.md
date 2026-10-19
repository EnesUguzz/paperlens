# config

Every tunable value lives here; nothing else in the codebase hard-codes a parameter.

| File | Role |
|---|---|
| `models.py` | pydantic models; each field says *what* it does, its *typical* range and its *effect* on metrics |
| `loader.py` | reads experiment YAML; `extends: parent.yaml` lets a file state only its change |
| `explain.py` | renders all fields as a table (`paperlens explain-config`, `docs/CONFIG.md`) |

Workflow: copy a YAML in `experiments/configs/`, change one value, run `make eval CONFIG=...`,
read `evals/results/<run_name>/report.md`.
