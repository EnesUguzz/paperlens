import json
from pathlib import Path

from test_graph import ANSWER, TITLES, chunk, make, question

from paperlens.domain import Answer, PipelineResult, Question
from paperlens.evals.conversation_runner import run_conversations
from paperlens.evals.conversations import Turn
from paperlens.evals.rescore import is_conversation_run
from paperlens.evals.routing_metrics import routing_metrics


def t(cid: str, i: int, qid: str, paper: str, text: str, kind: str, template=None) -> Turn:
    return Turn(
        conversation_id=cid,
        turn_index=i,
        question_id=qid,
        paper_id=paper,
        text=text,
        turn_type=kind,
        template=template,
    )


CONVS = [
    [
        t("c0", 0, "q1", "A", "Which fruit?", "first"),
        t("c0", 1, "q2", "A", "Which fruit again?", "same"),
        t("c0", 2, "q3", "B", "About the Beta paper: which fruit?", "switch_signaled", 2),
    ],
    [t("c1", 0, "q4", "B", "Which fruit?", "first")],
]
QUESTIONS = {
    qid: question(qid, paper) for qid, paper in [("q1", "A"), ("q2", "A"), ("q3", "B"), ("q4", "B")]
}
SAME = '{"question": "Which fruit again?"}'
NAMED = '{"question": "About the Beta paper: which fruit?"}'


def test_agent_conversation_records() -> None:
    # c0: clear vote A, then a split vote (stay on A), then B named; c1: split vote -> clarify
    votes = [["A"] * 10, ["A", "B"] * 5, ["A"] * 10, ["A", "B"] * 5]
    graph, _, _, _ = make(votes, [ANSWER, SAME, ANSWER, NAMED, ANSWER, ANSWER])
    records = run_conversations(CONVS, QUESTIONS, TITLES, [1], "paragraph", graph=graph)
    assert [r.question_id for r in records] == ["q1", "q2", "q3", "q4"]
    assert [r.decision for r in records] == ["vote", "stay", "named", "clarify"]
    assert [r.resolved_paper_id for r in records] == ["A", "A", "B", "B"]
    assert records[3].clarified and not records[0].clarified
    assert records[2].shown_question == "About the Beta paper: which fruit?"
    m = routing_metrics(records)
    assert m["route.paper_accuracy"] == 1.0
    assert m["route.signaled_catch"] == 1.0 and m["route.signaled_catch.t2"] == 1.0
    assert m["route.clarify_rate"] == 0.25
    assert m["route.wrong_paper_rate"] == 0.0
    assert "answer_f1.no_clarify" in m and "answer_f1.turn.same" in m


class FlakyPipeline:
    """corpus_rag stand-in whose first turn crashes."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, question: Question, plan=None, text: str | None = None) -> PipelineResult:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        answer = Answer(question_id=question.question_id, answer_type="extractive", text="apples")
        return PipelineResult(answer=answer, retrieved=[chunk("B")], retrieval_ms=1.0)


def test_turn_error_continues() -> None:
    records = run_conversations(
        CONVS, QUESTIONS, TITLES, [1], "paragraph", pipeline=FlakyPipeline()
    )
    assert len(records) == 4
    assert records[0].error and "boom" in records[0].error and records[0].refused
    assert records[1].prediction == "apples" and records[1].resolved_paper_id == "B"
    assert routing_metrics(records)["route.wrong_paper_rate"] == 0.25  # q2 (gold A) answered from B


def test_rescore_skips_conversation_runs(tmp_path: Path) -> None:
    (tmp_path / "params.json").write_text(json.dumps({"eval.system": "agent"}))
    assert is_conversation_run(tmp_path)
    (tmp_path / "params.json").write_text(json.dumps({"eval.system": "single_paper"}))
    assert not is_conversation_run(tmp_path)
