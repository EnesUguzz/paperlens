import itertools
import json
from pathlib import Path

from paperlens.config.models import ConversationsConfig
from paperlens.domain import Question, Reference
from paperlens.evals.conversations import (
    build_conversations,
    load_conversations,
    save_conversations,
    short_title,
)


def _questions() -> tuple[list[Question], dict[str, str]]:
    """250 questions over 171 papers, the real m1b_best sample shape (110x1, 43x2, 18x3)."""
    sizes = [1] * 110 + [2] * 43 + [3] * 18
    questions: list[Question] = []
    for p, size in enumerate(sizes):
        for i in range(size):
            ref = Reference(answer_type="extractive", answer_text="x", evidence_ids=[])
            questions.append(
                Question(
                    question_id=f"q{p}_{i}",
                    paper_id=f"P{p}",
                    question=f"Which dataset {i}?",
                    references=[ref],
                )
            )
    return questions, {f"P{p}": f"Paper {p}: a study" for p in range(len(sizes))}


def test_every_question_once() -> None:
    questions, titles = _questions()
    convs = build_conversations(questions, titles, ConversationsConfig())
    used = sorted(t.question_id for conv in convs for t in conv)
    assert used == sorted(q.question_id for q in questions)


def test_deterministic(tmp_path: Path) -> None:
    questions, titles = _questions()
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_conversations(build_conversations(questions, titles, ConversationsConfig()), a)
    save_conversations(build_conversations(questions, titles, ConversationsConfig()), b)
    assert a.read_bytes() == b.read_bytes()
    assert load_conversations(a) == build_conversations(questions, titles, ConversationsConfig())


def test_turn_types() -> None:
    questions, titles = _questions()
    for conv in build_conversations(questions, titles, ConversationsConfig()):
        assert conv[0].turn_type == "first"
        for prev, turn in itertools.pairwise(conv):
            if turn.turn_type == "same":
                assert turn.paper_id == prev.paper_id
            else:
                assert turn.turn_type.startswith("switch_")
                assert turn.paper_id != prev.paper_id
            assert (turn.template is not None) == (turn.turn_type == "switch_signaled")


def test_signal_text() -> None:
    ref = Reference(answer_type="extractive", answer_text="x", evidence_ids=[])
    q = Question(question_id="q", paper_id="P", question="Which dataset?", references=[ref])
    cfg = ConversationsConfig(
        signal_prob=1.0, signal_templates=['In the paper "{title}", {question}']
    )
    [[turn]] = build_conversations([q], {"P": "T"}, cfg)
    assert turn.text == 'In the paper "T", which dataset?'
    assert turn.template == 0


def test_short_title() -> None:
    assert short_title("BERT: Pre-training of deep") == "BERT"
    assert short_title("A tiny paper about retrieval") == "A tiny paper about"


def test_returns_happen() -> None:
    questions, titles = _questions()
    cfg = ConversationsConfig(return_prob=1.0)
    returned = 0
    for conv in build_conversations(questions, titles, cfg):
        blocks = [
            t.paper_id for i, t in enumerate(conv) if i == 0 or t.paper_id != conv[i - 1].paper_id
        ]
        returned += len(blocks) - len(set(blocks))
    assert returned > 0


def test_jsonl_one_turn_per_line(tmp_path: Path) -> None:
    questions, titles = _questions()
    path = tmp_path / "c.jsonl"
    convs = build_conversations(questions, titles, ConversationsConfig())
    save_conversations(convs, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 250
    assert {"conversation_id", "turn_type", "text"} <= set(json.loads(lines[0]))
