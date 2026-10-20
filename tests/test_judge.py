import json
from pathlib import Path

from paperlens.config.models import JudgeConfig
from paperlens.evals.aggregate import aggregate
from paperlens.evals.judge import (
    JUDGE_FILE,
    JudgeVerdict,
    judge_metrics,
    judge_record,
    judge_run,
    rescored_verdicts,
)
from paperlens.evals.records import EvalRecord
from paperlens.evals.report import METRICS_FILE, RECORDS_FILE
from paperlens.llm.client import LLMError
from paperlens.llm.fake import FakeLLM

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = (REPO / "prompts" / "judge.md").read_text(encoding="utf-8")
CORRECT = '{"reason": "same", "correct": true}'


def make_record(qid: str = "q1", **overrides: object) -> EvalRecord:
    fields: dict[str, object] = {
        "question_id": qid,
        "paper_id": "p",
        "question": "Which dataset?",
        "gold_texts": ["SQuAD"],
        "gold_unanswerable": False,
        "gold_type": "abstractive",
        "prediction": "The SQuAD dataset",
        "predicted_type": "abstractive",
        "refused": False,
        "error": None,
        "answer_f1": 0.5,
        "evidence_f1": 0.0,
        "evidence_f1_ceiling": 0.0,
        "best_ref_type": "abstractive",
        "retrieval": None,
        "retrieved_chunk_ids": [],
        "cited_chunk_ids": [],
        "top_context_preview": "",
        "context_words": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
    }
    fields.update(overrides)
    return EvalRecord(**fields)  # type: ignore[arg-type]


def verdict(record: EvalRecord, llm: FakeLLM) -> JudgeVerdict:
    return judge_record(record, llm, TEMPLATE, "judge-model", "sha")


def test_rules_decide_without_llm() -> None:
    llm = FakeLLM([])
    refused_ok = make_record(gold_unanswerable=True, gold_type="unanswerable", refused=True)
    answered_bad = make_record("q2", gold_unanswerable=True, gold_type="unanswerable")
    refused_bad = make_record("q3", refused=True)
    assert verdict(refused_ok, llm).correct is True
    assert verdict(answered_bad, llm).correct is False
    assert verdict(refused_bad, llm).correct is False
    assert verdict(refused_ok, llm).method == "rule"
    assert llm.calls == []


def test_yes_no_uses_exact_match() -> None:
    llm = FakeLLM([])
    assert verdict(make_record(gold_type="yes_no", answer_f1=1.0), llm).correct is True
    assert verdict(make_record(gold_type="yes_no", answer_f1=0.0), llm).correct is False
    assert llm.calls == []


def test_llm_verdict_for_abstractive() -> None:
    llm = FakeLLM([CORRECT])
    result = verdict(make_record(gold_texts=["SQuAD", "Unanswerable", "NewsQA"]), llm)
    assert (result.correct, result.method, result.reason) == (True, "llm", "same")
    prompt = llm.calls[0][0][0]["content"]
    assert "- SQuAD\n- NewsQA" in prompt and "Unanswerable" not in prompt
    assert "The SQuAD dataset" in prompt and "Which dataset?" in prompt


def test_llm_error_is_excluded_from_accuracy() -> None:
    records = [make_record("a"), make_record("b")]
    llm = FakeLLM([LLMError("down"), CORRECT])
    verdicts = {r.question_id: verdict(r, llm) for r in records}
    assert verdicts["a"].correct is None and verdicts["a"].method == "error"
    metrics = judge_metrics(records, verdicts)
    assert metrics["judge.accuracy"] == 1.0
    assert metrics["judge.error_rate"] == 0.5
    assert metrics["judge.llm_calls"] == 2.0


def test_exact_match_with_minority_reference_is_correct_without_llm() -> None:
    llm = FakeLLM([])
    refused = make_record(gold_unanswerable=True, gold_type="unanswerable", answer_f1=1.0)
    answered = make_record("q2", gold_texts=["Unanswerable", "Yes"], answer_f1=1.0, refused=False)
    assert verdict(refused, llm).correct is True
    assert verdict(answered, llm).correct is True
    assert llm.calls == []


def test_judge_metrics_agreement_on_high_f1() -> None:
    records = [make_record("a", answer_f1=0.9), make_record("b", answer_f1=0.9)]
    llm = FakeLLM([CORRECT, '{"reason": "no", "correct": false}'])
    verdicts = {r.question_id: verdict(r, llm) for r in records}
    assert judge_metrics(records, verdicts)["judge.agreement_on_high_f1"] == 0.5
    llm = FakeLLM([CORRECT, CORRECT])
    verdicts = {r.question_id: verdict(r, llm) for r in records}
    metrics = judge_metrics(records, verdicts)
    assert metrics["judge.agreement_on_high_f1"] == 1.0
    assert metrics["judge.accuracy.gold.abstractive"] == 1.0


def test_no_high_f1_llm_records_omits_agreement() -> None:
    records = [make_record("a", answer_f1=0.2), make_record("b", answer_f1=1.0)]
    llm = FakeLLM([CORRECT])
    verdicts = {r.question_id: verdict(r, llm) for r in records}
    assert "judge.agreement_on_high_f1" not in judge_metrics(records, verdicts)


def write_run(run_dir: Path, records: list[EvalRecord]) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / RECORDS_FILE).write_text("".join(r.model_dump_json() + "\n" for r in records))
    (run_dir / METRICS_FILE).write_text(json.dumps(aggregate(records, [1])))


def test_judge_run_reuses_cached_verdicts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, [make_record("a"), make_record("b")])
    cfg = JudgeConfig(prompt_template=REPO / "prompts" / "judge.md")
    first = judge_run(run_dir, cfg, FakeLLM([CORRECT, CORRECT]))
    llm = FakeLLM([])
    second = judge_run(run_dir, cfg, llm)
    assert llm.calls == [] and second == first
    saved = json.loads((run_dir / METRICS_FILE).read_text())
    assert saved["judge.accuracy"] == 1.0 and "answer_f1" in saved
    assert len((run_dir / JUDGE_FILE).read_text().splitlines()) == 2


def test_judge_run_rejudges_when_prediction_changed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, [make_record("a")])
    cfg = JudgeConfig(prompt_template=REPO / "prompts" / "judge.md")
    judge_run(run_dir, cfg, FakeLLM([CORRECT]))
    write_run_again = make_record("a", prediction="something else")
    (run_dir / RECORDS_FILE).write_text(write_run_again.model_dump_json() + "\n")
    llm = FakeLLM(['{"reason": "no", "correct": false}'])
    metrics = judge_run(run_dir, cfg, llm)
    assert len(llm.calls) == 1 and metrics["judge.accuracy"] == 0.0


def test_judge_config_extra_body_is_not_shared() -> None:
    a, b = JudgeConfig(), JudgeConfig()
    a.extra_body["x"] = 1
    assert b.extra_body == {"reasoning_effort": "none"}


def test_errored_record_is_incorrect_without_llm() -> None:
    llm = FakeLLM([])
    result = verdict(make_record(error="timeout", prediction="Unanswerable"), llm)
    assert (result.correct, result.method) == (False, "rule") and llm.calls == []


def test_judge_run_rejudges_when_model_or_prompt_changed(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    write_run(run_dir, [make_record("a")])
    cfg = JudgeConfig(prompt_template=REPO / "prompts" / "judge.md")
    judge_run(run_dir, cfg, FakeLLM([CORRECT]))
    other_model = FakeLLM([CORRECT])
    judge_run(run_dir, cfg.model_copy(update={"model": "other"}), other_model)
    assert len(other_model.calls) == 1
    prompt = tmp_path / "judge2.md"
    prompt.write_text(TEMPLATE + "\nBe strict.\n")
    other_prompt = FakeLLM([CORRECT])
    judge_run(run_dir, cfg.model_copy(update={"prompt_template": prompt}), other_prompt)
    assert len(other_prompt.calls) == 1


def test_rescored_verdicts_redo_rules_and_keep_llm_verdicts() -> None:
    stale = verdict(make_record("a", answer_f1=1.0), FakeLLM([]))  # rule: exact -> True
    llm_old = verdict(make_record("b"), FakeLLM([CORRECT]))
    records = [
        make_record("a", answer_f1=0.4, gold_type="yes_no"),  # now a plain yes/no miss
        make_record("b"),
        make_record("c"),  # needs the LLM, never judged
    ]
    result = rescored_verdicts(records, {"a": stale, "b": llm_old})
    assert result["a"].correct is False
    assert result["b"] is llm_old
    assert "c" not in result
