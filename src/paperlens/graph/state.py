"""What the M2 agent remembers within a turn and across the turns of one conversation.

Purpose: One typed dict that every node reads and returns updates to. LangGraph stores it
per thread (one thread = one conversation) in the checkpointer, so `history` and
`active_paper_id` survive from one user message to the next.
Flow: eval loop / CLI -> graph.invoke({"question", "text"}, thread) -> nodes update this
state -> checkpointer -> next turn starts from it.
Params: none.
Try: After a turn, call graph.get_state(config).values in the notebook and look at
`candidates`: those shares are why the agent stayed, switched or asked.
"""

from __future__ import annotations

from typing import TypedDict

from paperlens.domain import Answer, Question, ScoredChunk, Usage


class AgentState(TypedDict, total=False):
    # input of every turn
    question: Question  # carries question_id for scoring; nodes never read its paper_id
    text: str  # what the user typed
    # kept across turns (the conversation's memory)
    history: list[tuple[str, str]]  # (user text, answer text) per finished turn
    active_paper_id: str | None
    # set during the turn
    standalone: str  # condense output: the text searched and answered
    condense_error: str | None
    decision: str  # named | vote | stay | clarify
    candidates: list[tuple[str, float]]  # (paper_id, share of the corpus top-k)
    resolved_paper_id: str | None
    clarified: bool
    clarify_message: str
    retrieved: list[ScoredChunk]
    answer: Answer
    usage: Usage  # LLM cost of the turn so far
    llm_calls: int
