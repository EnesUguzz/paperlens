import copy
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import TypeAdapter

from paperlens.config.models import AppConfig
from paperlens.domain import Paper, Question
from paperlens.evals.runner import flatten, run_experiment
from paperlens.index.embedder import HashingEmbedder
from paperlens.index.store import build_or_load
from paperlens.ingestion.docling_parser import Block
from paperlens.ingestion.pdf import parsed_path
from paperlens.ingestion.qasper import raw_path
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.baseline import format_context, render_prompt
from paperlens.pipelines.factory import build_pipeline
from paperlens.retrieval.query import QueryPlan
from paperlens.retrieval.retriever import Retriever

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


def test_factory_baseline_prompt_is_unchanged(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = AppConfig(
        data={"data_dir": tmp_path / "data", "sample_size": None},
        chunking={"strategy": "section", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"model_name": "hashing"},
        retrieval={"method": "bm25", "top_k": 2},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
    )
    bundle = build_or_load(cfg, [sample_paper], HashingEmbedder())
    try:
        titles = {sample_paper.paper_id: sample_paper.title}
        pipe = build_pipeline(cfg, bundle, FakeLLM([]), titles)
        question = sample_questions[0]
        retrieved = Retriever(cfg.retrieval, bundle.dense, bundle.sparse).retrieve(
            question.question, paper_id=question.paper_id
        )
        expected = render_prompt(
            cfg.generation.prompt_template.read_text(encoding="utf-8"),
            title=sample_paper.title,
            context=format_context(retrieved),
            question=question.question,
        )
        assert pipe.build_prompt(question, retrieved) == expected
        plan = pipe.plan(question)
        assert plan == QueryPlan.identity(question.question)
        assert pipe.retrieve(question, plan) == retrieved
    finally:
        bundle.dense.close()


def test_run_experiment_pdf_docling_drops_papers_without_pdf(
    tmp_path: Path, sample_row: dict[str, Any]
) -> None:
    other = copy.deepcopy(sample_row)
    other["id"] = "9999.0001"
    other["qas"]["question_id"] = ["o1", "o2", "o3"]
    cfg = AppConfig(
        data={
            "data_dir": tmp_path / "data",
            "sample_size": None,
            "text_source": "pdf_docling",
            "align_min_words": 3,
        },
        chunking={"strategy": "section", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"model_name": "hashing"},
        retrieval={"method": "bm25", "top_k": 2},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={
            "run_name": "pdf_smoke",
            "reference_run": None,
            "results_dir": tmp_path / "results",
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "k_values": [1, 2],
        },
    )
    raw = raw_path(cfg.data, cfg.data.split)
    raw.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([sample_row, other]), raw)
    parsed = parsed_path(cfg.data, sample_row["id"])
    parsed.parent.mkdir(parents=True)
    blocks = [
        Block(kind="text", text="We use the SQuAD dataset for evaluation."),
        Block(kind="text", text="Our model reaches 81.2 F1 on the test set."),
    ]
    parsed.write_bytes(TypeAdapter(list[Block]).dump_json(blocks))
    llm = FakeLLM(['{"answer_type": "unanswerable", "answer": "Unanswerable", "evidence": []}'] * 3)
    result = run_experiment(cfg, embedder=HashingEmbedder(), llm=llm)
    assert result.metrics["n_questions"] == 3
    assert result.metrics["data.questions_dropped"] == 3
    assert result.metrics["data.papers_missing"] == 1
    assert "data.evidence_aligned_rate" in result.metrics
