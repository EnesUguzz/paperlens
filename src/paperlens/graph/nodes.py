"""The M2 agent's steps: condense, resolve_paper, clarify, retrieve, generate, refuse.

Purpose: Each node is a small function of the state that returns only what it changes,
so a trace shows one decision per span. Retrieval and answering reuse the M1 code
(Retriever, BaselinePipeline.generate); routing is graph/routing.py.
Flow: condense -> resolve_paper -> (clarify: interrupt, the user names the paper) ->
retrieve -> generate | refuse. Wired in graph/build.py.
Params: agent.condense.*, agent.route.*, agent.clarify.max_candidates, retrieval.*,
generation.*.
Try: Turn agent.condense.enabled off and run the follow-up demo (`paperlens converse`):
'and how big is it?' is then searched as typed and usually gets refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from langgraph.types import interrupt
from pydantic import BaseModel

from paperlens.config.models import AgentConfig
from paperlens.domain import UNANSWERABLE_TEXT, Answer, ScoredChunk, Usage
from paperlens.graph.routing import decide, match_title, vote
from paperlens.graph.state import AgentState
from paperlens.llm.client import LLMClient, LLMError
from paperlens.llm.structured import StructuredOutputError, structured_call
from paperlens.pipelines.baseline import BaselinePipeline, RetrieverLike

CLARIFY_PROMPT = "Which paper do you mean?"
NO_HISTORY = "(no earlier messages)"


class CorpusSearch(Protocol):
    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]: ...


class Condensed(BaseModel):
    question: str


@dataclass
class AgentDeps:
    retriever: RetrieverLike  # per-paper retrieval (m1b_best settings)
    corpus: CorpusSearch  # dense index searched without a filter, for the paper vote
    generator: BaselinePipeline  # only .generate() is used
    llm: LLMClient
    titles: Mapping[str, str]
    cfg: AgentConfig
    condense_template: str


def _history_text(history: list[tuple[str, str]], turns: int) -> str:
    lines = [f"User: {q}\nAssistant: {a}" for q, a in history[-turns:]]
    return "\n".join(lines) or NO_HISTORY


class Nodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.d = deps

    def condense(self, state: AgentState) -> dict[str, Any]:
        text = state["text"]
        history = state.get("history", [])
        # per-turn fields start fresh: the checkpointer keeps last turn's values otherwise
        fresh: dict[str, Any] = {
            "standalone": text,
            "condense_error": None,
            "clarified": False,
            "clarify_message": "",
            "resolved_paper_id": None,
            "retrieved": [],
            "usage": Usage(),
            "llm_calls": 0,
        }
        cfg = self.d.cfg.condense
        if not cfg.enabled or not history:
            return fresh
        prompt = self.d.condense_template.replace(
            "{{history}}", _history_text(history, cfg.history_turns)
        ).replace("{{question}}", text)
        try:
            out, usage = structured_call(
                self.d.llm, [{"role": "user", "content": prompt}], Condensed
            )
        except (LLMError, StructuredOutputError) as err:
            used = err.usage if isinstance(err, StructuredOutputError) else Usage()
            return fresh | {
                "condense_error": f"{type(err).__name__}: {err}",
                "usage": used,
                "llm_calls": 1,
            }
        standalone = out.question.strip() or text
        return fresh | {"standalone": standalone, "usage": usage, "llm_calls": 1}

    def resolve_paper(self, state: AgentState) -> dict[str, Any]:
        route = self.d.cfg.route
        votes = vote(self.d.corpus.search(state["standalone"], route.top_k))
        # a paper is named in what the user typed; the vote uses the condensed question
        decision, paper_id = decide(
            state["text"], state.get("active_paper_id"), votes, self.d.titles, route
        )
        return {"decision": decision, "candidates": votes, "resolved_paper_id": paper_id}

    def clarify(self, state: AgentState) -> dict[str, Any]:
        top = [pid for pid, _ in state.get("candidates", [])][: self.d.cfg.clarify.max_candidates]
        options = " ".join(f"{i}) {self.d.titles.get(pid, pid)}" for i, pid in enumerate(top, 1))
        message = f"{CLARIFY_PROMPT} {options}".strip()
        # pauses the graph; the caller resumes with Command(resume=<user reply>)
        reply = interrupt({"message": message, "candidates": top})
        paper_id = match_title(
            str(reply), self.d.titles, self.d.cfg.route.title_match_min, candidates=top
        )
        return {"clarified": True, "clarify_message": message, "resolved_paper_id": paper_id}

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        paper_id = state["resolved_paper_id"]
        return {"retrieved": self.d.retriever.retrieve(state["standalone"], paper_id=paper_id)}

    def generate(self, state: AgentState) -> dict[str, Any]:
        paper_id = state["resolved_paper_id"]
        answer = self.d.generator.generate(
            state["question"],
            state["standalone"],
            self.d.titles.get(paper_id or "", ""),
            state.get("retrieved", []),
            state.get("usage"),
        )
        return {
            "answer": answer,
            "active_paper_id": paper_id,
            "history": [*state.get("history", []), (state["text"], answer.text)],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def refuse(self, state: AgentState) -> dict[str, Any]:
        answer = Answer(
            question_id=state["question"].question_id,
            answer_type="unanswerable",
            text=UNANSWERABLE_TEXT,
            refused=True,
            usage=state.get("usage", Usage()),
        )
        return {
            "answer": answer,
            "active_paper_id": None,  # the paper could not be resolved: ask again next turn
            "history": [*state.get("history", []), (state["text"], answer.text)],
        }
