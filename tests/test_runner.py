import json
from pathlib import Path

from paperlens.config.models import AppConfig
from paperlens.domain import Paper, Question
from paperlens.evals.runner import flatten, run_experiment
from paperlens.index.embedder import HashingEmbedder
from paperlens.llm.fake import FakeLLM

REPO = Path(__file__).resolve().parents[1]


def test_flatten() -> None:
    assert flatten({"a": {"b": 1, "c": {}}, "d": 2}) == {"a.b": 1, "a.c": {}, "d": 2}


def test_run_experiment_end_to_end(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = AppConfig(
        data={"data_dir": tmp_path / "data", "sample_size": None},
        chunking={"strategy": "section", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"model_name": "hashing"},
        retrieval={"method": "hybrid", "top_k": 2, "candidate_k": 4},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={
            "run_name": "smoke",
            "reference_run": None,
            "results_dir": tmp_path / "results",
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "k_values": [1, 2, 5],
        },
    )
    llm = FakeLLM(
        [
            '{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}',
            '{"answer_type": "unanswerable", "answer": "Unanswerable", "evidence": []}',
            '{"answer_type": "abstractive", "answer": "81.2 F1", "evidence": [1]}',
        ]
    )
    result = run_experiment(
        cfg, dataset=([sample_paper], sample_questions), embedder=HashingEmbedder(), llm=llm
    )
    run_dir = tmp_path / "results" / "smoke"
    assert (run_dir / "report.md").exists()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_questions"] == 3 and metrics["answer_f1"] == 1.0
    assert "retrieval.recall_at_5" not in metrics  # 5 > top_k is skipped
    assert result.params["retrieval.method"] == "hybrid"
