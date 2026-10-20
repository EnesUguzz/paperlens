"""LLM-as-judge: a second opinion on answer correctness, applied to saved runs.

Purpose: Answer-F1 counts word overlap, so a correct but long answer scores low. The judge
asks "is this the same fact as a reference answer?". It is reported next to F1 as extra
information only (judge.* metrics); the deterministic metrics stay the headline.
Flow: evals/results/<run>/records.jsonl -> judge_run() -> per question a rule verdict
(unanswerable, refusal, yes/no) or an LLM verdict -> judge.jsonl + judge.* keys in
metrics.json. rescore_run() keeps the judge.* metrics for answers that did not change.
Params: judge.model, judge.extra_body, judge.prompt_template (llm.base_url/api_key/timeouts
are reused). Temperature is fixed to 0 so verdicts are repeatable.
Try: Open judge.jsonl and compare the reasons with answer_f1 for long answers; check that
judge.agreement_on_high_f1 is >= 0.95 (a judge that rejects near-exact answers is not trusted).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel

from paperlens.config.models import JudgeConfig
from paperlens.domain import UNANSWERABLE_TEXT, AnswerType
from paperlens.evals.provenance import file_sha
from paperlens.evals.records import EvalRecord
from paperlens.evals.report import METRICS_FILE, RECORDS_FILE
from paperlens.llm.client import LLMClient, LLMError
from paperlens.llm.structured import StructuredOutputError, structured_call

JUDGE_FILE = "judge.jsonl"
JUDGE_PREFIX = "judge."
EXACT_F1 = 1.0  # an answer with this F1 matched a reference word for word
# LLM-judged answers with at least this F1 should almost always be judged correct; the share
# that is (judge.agreement_on_high_f1) is the sanity check of the judge
HIGH_F1 = 0.8

Method = Literal["rule", "llm", "error"]


class Verdict(BaseModel):
    """What the judge model must return; the reason comes first so it explains the decision."""

    reason: str
    correct: bool


class JudgeVerdict(BaseModel):
    question_id: str
    correct: bool | None
    method: Method
    reason: str
    prediction: str
    judge_model: str
    prompt_sha256: str


def render_judge_prompt(template: str, record: EvalRecord) -> str:
    gold = "\n".join(f"- {t}" for t in record.gold_texts if t != UNANSWERABLE_TEXT)
    return (
        template.replace("{{question}}", record.question)
        .replace("{{gold}}", gold)
        .replace("{{prediction}}", record.prediction)
    )


def _verdict(
    record: EvalRecord,
    correct: bool | None,
    method: Method,
    reason: str,
    judge_model: str,
    prompt_sha: str,
) -> JudgeVerdict:
    return JudgeVerdict(
        question_id=record.question_id,
        correct=correct,
        method=method,
        reason=reason,
        prediction=record.prediction,
        judge_model=judge_model,
        prompt_sha256=prompt_sha,
    )


def rule_verdict(
    record: EvalRecord, judge_model: str = "", prompt_sha: str = ""
) -> JudgeVerdict | None:
    """The verdict the rules give without an LLM, or None when the LLM must decide."""

    def make(correct: bool, reason: str) -> JudgeVerdict:
        return _verdict(record, correct, "rule", reason, judge_model, prompt_sha)

    # QASPER scores the best annotator, so matching any reference exactly is correct
    if record.answer_f1 == EXACT_F1:
        return make(True, "exact match with a reference answer")
    if record.error:
        return make(False, "the pipeline failed on this question")
    # only when every reference is unanswerable: with a mixed set an answer can still match an
    # answerable reference (any reference counts), so it goes on to the LLM; a refusal there
    # already matched the 'Unanswerable' reference exactly (rule above)
    if all(t == UNANSWERABLE_TEXT for t in record.gold_texts):
        return make(record.refused, "gold is unanswerable: correct iff the model refused")
    if record.refused:
        return make(False, "the model refused a question that has an answer")
    if record.gold_type == "yes_no":
        return make(False, "yes/no without an exact match")
    return None


def judge_record(
    record: EvalRecord, llm: LLMClient, template: str, judge_model: str, prompt_sha: str
) -> JudgeVerdict:
    ruled = rule_verdict(record, judge_model, prompt_sha)
    if ruled is not None:
        return ruled
    prompt = render_judge_prompt(template, record)
    try:
        result, _ = structured_call(llm, [{"role": "user", "content": prompt}], Verdict)
    except (LLMError, StructuredOutputError) as err:
        return _verdict(record, None, "error", str(err), judge_model, prompt_sha)
    return _verdict(record, result.correct, "llm", result.reason, judge_model, prompt_sha)


def _accuracy(verdicts: Sequence[JudgeVerdict]) -> float:
    return sum(bool(v.correct) for v in verdicts) / len(verdicts)


def judge_metrics(
    records: Sequence[EvalRecord], verdicts: Mapping[str, JudgeVerdict]
) -> dict[str, float]:
    """judge.* metrics over the records that have a verdict; errors are not in accuracy."""
    judged = [(r, verdicts[r.question_id]) for r in records if r.question_id in verdicts]
    if not judged:
        return {}
    decided = [(r, v) for r, v in judged if v.correct is not None]
    metrics = {
        "judge.error_rate": (len(judged) - len(decided)) / len(judged),
        "judge.llm_calls": float(sum(v.method != "rule" for _, v in judged)),
    }
    if decided:
        metrics["judge.accuracy"] = _accuracy([v for _, v in decided])
    for gold in get_args(AnswerType):
        subset = [v for r, v in decided if r.gold_type == gold]
        if subset:
            metrics[f"judge.accuracy.gold.{gold}"] = _accuracy(subset)
    high = [v for r, v in decided if v.method == "llm" and r.answer_f1 >= HIGH_F1]
    if high:
        metrics["judge.agreement_on_high_f1"] = _accuracy(high)
    return metrics


def load_verdicts(run_dir: Path) -> dict[str, JudgeVerdict]:
    path = run_dir / JUDGE_FILE
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    verdicts = (JudgeVerdict.model_validate_json(line) for line in lines if line)
    return {v.question_id: v for v in verdicts}


def current_verdicts(
    records: Sequence[EvalRecord], saved: Mapping[str, JudgeVerdict]
) -> dict[str, JudgeVerdict]:
    """Saved verdicts whose answer is unchanged (so they still describe this record)."""
    return {
        r.question_id: saved[r.question_id]
        for r in records
        if r.question_id in saved and saved[r.question_id].prediction == r.prediction
    }


def rescored_verdicts(
    records: Sequence[EvalRecord], saved: Mapping[str, JudgeVerdict]
) -> dict[str, JudgeVerdict]:
    """Verdicts for rescored records without calling the LLM.

    Rule verdicts are re-derived from the fresh records (rescoring can change answer_f1 or
    refusals); LLM verdicts are taken from judge.jsonl only if the answer is unchanged.
    Records that would need the LLM and have no such verdict are left out (not judged)."""
    verdicts: dict[str, JudgeVerdict] = {}
    for r in records:
        old = saved.get(r.question_id)
        ruled = rule_verdict(r, old.judge_model if old else "", old.prompt_sha256 if old else "")
        if ruled is not None:
            verdicts[r.question_id] = ruled
        elif old is not None and old.method == "llm" and old.prediction == r.prediction:
            verdicts[r.question_id] = old
    return verdicts


def judge_run(run_dir: Path, cfg: JudgeConfig, llm: LLMClient) -> dict[str, float]:
    lines = (run_dir / RECORDS_FILE).read_text(encoding="utf-8").splitlines()
    records = [EvalRecord.model_validate_json(line) for line in lines if line]
    template = cfg.prompt_template.read_text(encoding="utf-8")
    prompt_sha = file_sha(cfg.prompt_template)
    reusable = {
        qid: v
        for qid, v in current_verdicts(records, load_verdicts(run_dir)).items()
        if v.method == "llm" and v.judge_model == cfg.model and v.prompt_sha256 == prompt_sha
    }
    # rules come first (as in rescored_verdicts): a cached LLM verdict for a record the rules
    # now decide is dropped, so judge and rescore always report the same judge.* metrics
    ruled = {
        r.question_id: v
        for r in records
        if (v := rule_verdict(r, cfg.model, prompt_sha)) is not None
    }
    reusable = {qid: v for qid, v in reusable.items() if qid not in ruled}
    verdicts: dict[str, JudgeVerdict] = {}
    # only LLM verdicts are cached; rule verdicts are free and follow the current rules
    # the file is rewritten one verdict at a time (cached ones first come back in order), so
    # a killed run keeps what it has judged and the next call only asks for the rest
    cached_lines = [reusable[r.question_id] for r in records if r.question_id in reusable]
    with (run_dir / JUDGE_FILE).open("w", encoding="utf-8") as out:
        for v in cached_lines:
            out.write(v.model_dump_json() + "\n")
        out.flush()
        for r in records:
            if r.question_id in reusable:
                verdicts[r.question_id] = reusable[r.question_id]
                continue
            verdicts[r.question_id] = ruled.get(r.question_id) or judge_record(
                r, llm, template, cfg.model, prompt_sha
            )
            out.write(verdicts[r.question_id].model_dump_json() + "\n")
            out.flush()
    metrics = judge_metrics(records, verdicts)
    path = run_dir / METRICS_FILE
    existing = {
        k: v
        for k, v in json.loads(path.read_text(encoding="utf-8")).items()
        if not k.startswith(JUDGE_PREFIX)
    }
    path.write_text(json.dumps({**existing, **metrics}, indent=2), encoding="utf-8")
    return metrics
