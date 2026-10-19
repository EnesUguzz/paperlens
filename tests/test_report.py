from pathlib import Path

from paperlens.evals.records import EvalRecord
from paperlens.evals.report import RunResult, changed_params, load_run, render_report, save_run
from paperlens.evals.summary import summarize


def rec(qid: str, f1: float, prompt_tokens: int = 100) -> EvalRecord:
    return EvalRecord(
        question_id=qid,
        paper_id="P",
        question=f"question {qid}?",
        gold_texts=["SQuAD"],
        gold_unanswerable=False,
        gold_type="extractive",
        prediction="pred",
        predicted_type="extractive",
        refused=False,
        error=None,
        answer_f1=f1,
        evidence_f1=f1,
        evidence_f1_ceiling=1.0,
        best_ref_type="extractive",
        retrieval=None,
        retrieved_chunk_ids=[],
        cited_chunk_ids=[],
        top_context_preview="ctx",
        context_words=10,
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        latency_ms=50.0,
    )


BASE = RunResult(
    run_name="baseline",
    params={"retrieval.top_k": 5, "llm.context_window": 16384, "eval.run_name": "baseline"},
    metrics={"answer_f1": 0.40, "prompt_tokens_max": 1000.0},
    records=[rec("q1", 1.0), rec("q2", 0.5)],
)
NEW = RunResult(
    run_name="topk10",
    params={"retrieval.top_k": 10, "llm.context_window": 16384, "eval.run_name": "topk10"},
    metrics={"answer_f1": 0.35, "prompt_tokens_max": 16000.0},
    records=[rec("q1", 0.0), rec("q2", 0.5)],
)


def test_changed_params_ignore_eval_section() -> None:
    assert changed_params(NEW.params, BASE.params) == [("retrieval.top_k", 10, 5)]


def test_report_shows_change_delta_regression_and_truncation_warning() -> None:
    text = render_report(NEW, BASE, worst_n=3)
    assert "`retrieval.top_k` | 5 | 10" in text
    assert "| answer_f1 | 0.350 | 0.400 | -0.050 |" in text
    assert "question q1?" in text  # dropped from 1.0 to 0.0
    assert "question q2?" not in text  # unchanged
    assert "context_window" in text and "WARNING" in text


def test_report_without_reference() -> None:
    text = render_report(BASE, None, worst_n=1)
    assert "No reference run" in text
    assert "question q2?" in text  # lowest score listed


def test_save_load_roundtrip_and_missing(tmp_path: Path) -> None:
    save_run(BASE, tmp_path / "baseline")
    loaded = load_run(tmp_path / "baseline")
    assert loaded is not None and loaded.metrics == BASE.metrics and len(loaded.records) == 2
    assert load_run(tmp_path / "missing") is None


def test_summary_table_shows_varying_params(tmp_path: Path) -> None:
    save_run(BASE, tmp_path / "baseline")
    save_run(NEW, tmp_path / "topk10")
    table = summarize(tmp_path)
    assert "retrieval.top_k" in table and "llm.context_window" not in table
    assert "| topk10 | 10 |" in table
