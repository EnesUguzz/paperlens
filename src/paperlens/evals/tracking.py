"""Log a finished run to MLflow (params, metrics, and the run folder as artifacts).

Purpose: MLflow answers "which config was better?" across many runs with sortable
tables and charts; the Markdown report answers "why?" for one run.
Flow: runner -> log_to_mlflow() -> MLflow (SQLite) -> `uv run mlflow ui`.
Params: eval.mlflow_tracking_uri, eval.experiment_name, eval.run_name.
Try: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`, select runs, "Compare",
plot answer_f1 against latency_p95_ms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paperlens.config.models import AppConfig

MAX_PARAM_CHARS = 500


def log_to_mlflow(
    cfg: AppConfig, params: dict[str, Any], metrics: dict[str, float], run_dir: Path
) -> str:
    import mlflow  # heavy import, only when a run is logged

    mlflow.set_tracking_uri(cfg.eval.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.eval.experiment_name)
    with mlflow.start_run(run_name=cfg.eval.run_name) as run:
        mlflow.log_params({k: str(v)[:MAX_PARAM_CHARS] for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir))
        return str(run.info.run_id)
