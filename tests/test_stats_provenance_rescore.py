import json
from pathlib import Path

import pytest

from paperlens.config.models import AppConfig
from paperlens.domain import Chunk, Paper, Question
from paperlens.evals.provenance import collect_meta
from paperlens.evals.report import changed_params
from paperlens.evals.rescore import rescore_run, unflatten
from paperlens.evals.runner import run_experiment
from paperlens.evals.stats import paired_bootstrap
from paperlens.index.embedder import HashingEmbedder
from paperlens.llm.fake import FakeLLM

REPO = Path(__file__).resolve().parents[1]


def test_paired_bootstrap_identical_and_shifted() -> None:
    a = {"q1": 0.2, "q2": 0.8, "q3": 0.5}
    same = paired_bootstrap(a, a)
    assert (same.mean_diff, same.low, same.high) == (0.0, 0.0, 0.0)
    shifted = paired_bootstrap({k: v + 0.1 for k, v in a.items()}, a)
    assert shifted.mean_diff == pytest.approx(0.1)
    assert shifted.low == pytest.approx(0.1) and shifted.high == pytest.approx(0.1)
    assert shifted.n == 3


def test_paired_bootstrap_is_deterministic_and_brackets_mean() -> None:
    a = {f"q{i}": float(i % 3) / 2 for i in range(40)}
    b = {f"q{i}": float(i % 5) / 4 for i in range(40)}
    first, second = paired_bootstrap(a, b), paired_bootstrap(a, b)
    assert first == second
    assert first.low <= first.mean_diff <= first.high


def test_collect_meta_hashes_prompt_and_skips_server(tmp_path: Path) -> None:
    prompt = tmp_path / "p.md"
    prompt.write_text("{{title}} {{context}} {{question}}")
    cfg = AppConfig(generation={"prompt_template": prompt}, data={"data_dir": tmp_path})
    meta = collect_meta(cfg, query_server=False)
    prompt.write_text("{{title}} {{context}} {{question}} be brief")
    changed = collect_meta(cfg, query_server=False)
    assert meta["meta.prompt_sha256"] != changed["meta.prompt_sha256"]
    assert meta["meta.llm_digest"] == "not queried"
    assert meta["meta.data_sha256"] == "missing"


def test_prompt_change_shows_in_diff_but_git_sha_does_not() -> None:
    old = {"meta.prompt_sha256": "a", "meta.git_sha": "1", "retrieval.top_k": 5}
    new = {"meta.prompt_sha256": "b", "meta.git_sha": "2", "retrieval.top_k": 5}
    assert changed_params(new, old) == [("meta.prompt_sha256", "b", "a")]


def test_unflatten() -> None:
    assert unflatten({"a.b": 1, "a.c.d": 2, "e": {}}) == {"a": {"b": 1, "c": {"d": 2}}, "e": {}}


def test_rescore_reproduces_metrics_without_llm(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = AppConfig(
        data={"data_dir": tmp_path / "data", "sample_size": None},
        chunking={"strategy": "fixed", "chunk_size": 10, "chunk_overlap": 2},
        embedding={"model_name": "hashing"},
        retrieval={"method": "dense", "top_k": 3},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={
            "run_name": "orig",
            "reference_run": None,
            "results_dir": tmp_path / "results",
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "k_values": [1, 3],
        },
    )
    llm = FakeLLM(
        [
            '{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1, 2]}',
            '{"answer_type": "yes_no", "answer": "No", "evidence": []}',
            '{"answer_type": "abstractive", "answer": "81.2", "evidence": [1]}',
        ]
    )
    dataset = ([sample_paper], sample_questions)
    original = run_experiment(cfg, dataset=dataset, embedder=HashingEmbedder(), llm=llm)
    run_dir = tmp_path / "results" / "orig"
    (run_dir / "metrics.json").write_text("{}")  # pretend the metric code changed

    rescored = rescore_run(run_dir, dataset=dataset)
    saved = json.loads((run_dir / "metrics.json").read_text())
    assert saved == rescored.metrics
    for key, value in original.metrics.items():
        assert rescored.metrics[key] == pytest.approx(value), key
    assert (run_dir / "report.md").exists()


def _rescore_single_citation(
    tmp_path: Path,
    sample_paper: Paper,
    sample_questions: list[Question],
    cited_paragraph_ids: list[str] | None,
    cite_unit: str = "chunk",
) -> float:
    """Run once, then rewrite the run so q1 cites one 2-paragraph chunk; return rescored evidence_f1."""
    cfg = AppConfig(
        data={"data_dir": tmp_path / "data", "sample_size": None},
        chunking={"strategy": "fixed", "chunk_size": 10, "chunk_overlap": 2},
        embedding={"model_name": "hashing"},
        retrieval={"method": "dense", "top_k": 3},
        generation={
            "prompt_template": REPO / "prompts" / "baseline_answer.md",
            "cite_unit": cite_unit,
        },
        eval={
            "run_name": "orig",
            "reference_run": None,
            "results_dir": tmp_path / "results",
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "k_values": [1, 3],
        },
    )
    llm = FakeLLM(['{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}'] * 3)
    dataset = ([sample_paper], sample_questions)
    run_experiment(cfg, dataset=dataset, embedder=HashingEmbedder(), llm=llm)
    run_dir = tmp_path / "results" / "orig"
    wide = Chunk(
        chunk_id="P::c1",
        paper_id="1234.5678",
        section="S",
        text="wide chunk",
        paragraph_ids=["1234.5678::s0p1", "1234.5678::s0p0"],
    )
    (chunk_file,) = (tmp_path / "data" / "index").glob("*/chunks.jsonl")
    chunk_file.write_text(wide.model_dump_json() + "\n", encoding="utf-8")
    records_file = run_dir / "records.jsonl"
    raw = json.loads(records_file.read_text(encoding="utf-8").splitlines()[0])
    raw.update(retrieved_chunk_ids=["P::c1"], cited_chunk_ids=["P::c1"])
    assert "cited_paragraph_ids" in raw  # new code saves it, so the old format must be forced
    if cited_paragraph_ids is None:
        del raw["cited_paragraph_ids"]  # an old record: only chunk ids were saved
    else:
        raw["cited_paragraph_ids"] = cited_paragraph_ids
    records_file.write_text(json.dumps(raw) + "\n", encoding="utf-8")
    run = rescore_run(run_dir, dataset=dataset)
    return run.records[0].evidence_f1


def test_rescore_uses_cited_paragraph_ids_when_present(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    f1 = _rescore_single_citation(tmp_path, sample_paper, sample_questions, ["1234.5678::s0p1"])
    assert f1 == 1.0


def test_rescore_expands_chunks_for_old_records_without_the_field(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    f1 = _rescore_single_citation(tmp_path, sample_paper, sample_questions, None)
    assert f1 == pytest.approx(2 / 3)


def test_rescore_keeps_judge_metrics(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    from paperlens.config.models import JudgeConfig
    from paperlens.evals.judge import judge_run

    cfg = AppConfig(
        data={"data_dir": tmp_path / "data", "sample_size": None},
        chunking={"strategy": "fixed", "chunk_size": 10, "chunk_overlap": 2},
        embedding={"model_name": "hashing"},
        retrieval={"method": "dense", "top_k": 3},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={
            "run_name": "orig",
            "reference_run": None,
            "results_dir": tmp_path / "results",
            "mlflow_tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "k_values": [1, 3],
        },
    )
    llm = FakeLLM(
        [
            '{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1, 2]}',
            '{"answer_type": "yes_no", "answer": "No", "evidence": []}',
            '{"answer_type": "abstractive", "answer": "81.2", "evidence": [1]}',
        ]
    )
    dataset = ([sample_paper], sample_questions)
    run_experiment(cfg, dataset=dataset, embedder=HashingEmbedder(), llm=llm)
    run_dir = tmp_path / "results" / "orig"
    verdicts = FakeLLM(['{"reason": "ok", "correct": true}'] * 3)
    judged = judge_run(
        run_dir, JudgeConfig(prompt_template=REPO / "prompts" / "judge.md"), verdicts
    )
    assert "judge.accuracy" in judged
    rescored = rescore_run(run_dir, dataset=dataset)
    assert rescored.metrics["judge.accuracy"] == judged["judge.accuracy"]
    assert (
        json.loads((run_dir / "metrics.json").read_text())["judge.accuracy"]
        == judged["judge.accuracy"]
    )


def test_rescore_of_paragraph_run_keeps_evidence_f1(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    # paragraph mode: the ceiling must be recomputed per paragraph, and the model's exact
    # paragraph citation kept, so a single cited paragraph of a 2-paragraph chunk scores 1.0
    f1 = _rescore_single_citation(
        tmp_path, sample_paper, sample_questions, ["1234.5678::s0p1"], cite_unit="paragraph"
    )
    assert f1 == 1.0
