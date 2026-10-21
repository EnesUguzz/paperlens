"""Synthetic multi-paper conversations built from the eval sample (M2).

Purpose: QASPER asks every question next to its paper; a chat user does not. This mixes
the sampled questions into conversations that hop between papers, so the agent has to
remember the active paper, notice switches and ask when it cannot tell.
Flow: sampled Questions + titles + ConversationsConfig -> build_conversations() ->
list[list[Turn]] -> save_conversations() (evals/conversations/m2.jsonl, committed) ->
evals/conversation_runner.py.
Params: conversations.papers_per_conversation, conversations.return_prob,
conversations.signal_prob, conversations.seed, conversations.signal_templates.
Try: Set signal_prob to 0 and 1 and compare route.paper_accuracy: the gap is how much the
agent relies on the user naming the paper.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from paperlens.config.models import ConversationsConfig
from paperlens.domain import Question

TurnType = Literal["first", "same", "switch_signaled", "switch_unsignaled"]
SHORT_TITLE_WORDS = 4


class Turn(BaseModel):
    conversation_id: str
    turn_index: int
    question_id: str
    paper_id: str  # gold paper: for scoring and the simulated user only, never the agent
    text: str  # what the user types (the question, maybe wrapped in a signal template)
    turn_type: TurnType
    template: int | None = None  # index into signal_templates when the turn names its paper


def short_title(title: str) -> str:
    """'BERT: Pre-training ...' -> 'BERT'; titles without ':' -> their first 4 words."""
    head, sep, _ = title.partition(":")
    if sep and head.strip():
        return head.strip()
    return " ".join(title.split()[:SHORT_TITLE_WORDS])


def _signal(template: str, title: str, question: str) -> str:
    lowered = question[:1].lower() + question[1:]
    return template.format(title=title, short_title=short_title(title), question=lowered)


def _order(blocks: list[list[Question]], rng: random.Random, return_prob: float) -> list[Question]:
    """Blocks in order; a 2+ question paper may send its last question to the end."""
    moved: list[Question] = []
    ordered: list[Question] = []
    for i, block in enumerate(blocks):
        has_later_block = i < len(blocks) - 1
        if len(block) >= 2 and has_later_block and rng.random() < return_prob:
            ordered.extend(block[:-1])
            moved.append(block[-1])
        else:
            ordered.extend(block)
    return ordered + moved


def build_conversations(
    questions: Sequence[Question], titles: Mapping[str, str], cfg: ConversationsConfig
) -> list[list[Turn]]:
    rng = random.Random(cfg.seed)
    by_paper: dict[str, list[Question]] = {}
    for q in questions:
        by_paper.setdefault(q.paper_id, []).append(q)
    papers = list(by_paper)
    rng.shuffle(papers)
    size = cfg.papers_per_conversation
    conversations: list[list[Turn]] = []
    for c, start in enumerate(range(0, len(papers), size)):
        blocks = [by_paper[p] for p in papers[start : start + size]]
        cid = f"c{c:03d}"
        turns: list[Turn] = []
        for i, q in enumerate(_order(blocks, rng, cfg.return_prob)):
            switched = i == 0 or q.paper_id != turns[-1].paper_id
            template = None
            if switched and rng.random() < cfg.signal_prob:
                template = rng.randrange(len(cfg.signal_templates))
            if i == 0:
                turn_type: TurnType = "first"
            elif not switched:
                turn_type = "same"
            else:
                turn_type = "switch_signaled" if template is not None else "switch_unsignaled"
            text = (
                q.question
                if template is None
                else _signal(cfg.signal_templates[template], titles.get(q.paper_id, ""), q.question)
            )
            turns.append(
                Turn(
                    conversation_id=cid,
                    turn_index=i,
                    question_id=q.question_id,
                    paper_id=q.paper_id,
                    text=text,
                    turn_type=turn_type,
                    template=template,
                )
            )
        conversations.append(turns)
    return conversations


def save_conversations(conversations: Sequence[Sequence[Turn]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (t.model_dump_json() + "\n" for conv in conversations for t in conv)
    path.write_text("".join(lines), encoding="utf-8")


def load_conversations(path: Path) -> list[list[Turn]]:
    conversations: dict[str, list[Turn]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            turn = Turn.model_validate(json.loads(line))
            conversations.setdefault(turn.conversation_id, []).append(turn)
    return list(conversations.values())
