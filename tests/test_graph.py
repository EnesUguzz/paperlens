from pathlib import Path

from langgraph.types import Command

from paperlens.config.models import AgentConfig
from paperlens.domain import Chunk, Question, Reference, ScoredChunk, Segment
from paperlens.graph.build import AgentDeps, build_graph, make_checkpointer
from paperlens.llm.client import LLMError
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.baseline import BaselinePipeline
from paperlens.retrieval.query import QueryPlan

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "prompts" / "baseline_answer.md").read_text()
CONDENSE = (ROOT / "prompts" / "condense.md").read_text()
TITLES = {"A": "Alpha: a paper about apples", "B": "Beta: a paper about bananas"}
ANSWER = '{"answer_type": "extractive", "answer": "apples", "evidence": [1]}'


def chunk(paper_id: str, i: int = 0) -> ScoredChunk:
    pid = f"{paper_id}::s0p{i}"
    c = Chunk(
        chunk_id=f"{paper_id}::c{i}",
        paper_id=paper_id,
        section="s",
        text=f"text of {paper_id}",
        paragraph_ids=[pid],
        segments=[Segment(paragraph_id=pid, text=f"text of {paper_id}")],
    )
    return ScoredChunk(chunk=c, score=1.0)


class StubCorpus:
    """Dense index stand-in: the corpus vote returns whatever the test queues."""

    def __init__(self, votes: list[list[str]]) -> None:
        self.votes = votes
        self.filters: list[str | None] = []
        self.queries: list[str] = []

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        self.filters.append(paper_id)
        self.queries.append(query)
        return [chunk(p, i) for i, p in enumerate(self.votes.pop(0))]


class StubRetriever:
    def __init__(self) -> None:
        self.filters: list[str | None] = []

    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]:
        self.filters.append(paper_id)
        return [chunk(paper_id or "?")]


def question(qid: str = "q1", paper_id: str = "GOLD") -> Question:
    ref = Reference(answer_type="extractive", answer_text="apples", evidence_ids=[])
    return Question(question_id=qid, paper_id=paper_id, question="Which fruit?", references=[ref])


def make(votes: list[list[str]], replies: list[str | Exception]):
    llm = FakeLLM(replies)
    retriever = StubRetriever()
    corpus = StubCorpus(votes)
    generator = BaselinePipeline(retriever, llm, TEMPLATE, TITLES, cite_unit="paragraph")
    deps = AgentDeps(
        retriever=retriever,
        corpus=corpus,
        generator=generator,
        llm=llm,
        titles=TITLES,
        cfg=AgentConfig(tracing=False),
        condense_template=CONDENSE,
    )
    return build_graph(deps, make_checkpointer()), retriever, corpus, llm


def turn(graph, thread: str, text: str, qid: str = "q1"):
    config = {"configurable": {"thread_id": thread}}
    return graph.invoke({"question": question(qid), "text": text}, config), config


def test_first_turn_confident() -> None:
    graph, retriever, _, _ = make([["A"] * 10], [ANSWER])
    state, _ = turn(graph, "t", "Which fruit?")
    assert "__interrupt__" not in state
    assert state["decision"] == "vote" and state["active_paper_id"] == "A"
    assert retriever.filters == ["A"]
    assert state["answer"].text == "apples"


def test_first_turn_clarify_resume() -> None:
    graph, retriever, _, _ = make([["A", "B"] * 5], [ANSWER])
    state, config = turn(graph, "t", "Which fruit?")
    assert "__interrupt__" in state
    payload = state["__interrupt__"][0].value
    assert TITLES["A"] in payload["message"] and set(payload["candidates"]) == {"A", "B"}
    state = graph.invoke(Command(resume=TITLES["B"]), config)
    assert state["clarified"] and state["active_paper_id"] == "B"
    assert retriever.filters == ["B"]


def test_clarify_no_match_refuses() -> None:
    graph, retriever, _, _ = make([["A", "B"] * 5], [])
    _, config = turn(graph, "t", "Which fruit?")
    state = graph.invoke(Command(resume="no idea, sorry"), config)
    assert state["answer"].refused and state["active_paper_id"] is None
    assert retriever.filters == []


def test_stay_on_active() -> None:
    split = ["A", "B"] * 5
    condensed = '{"question": "Which colour is the fruit?"}'
    graph, retriever, _, _ = make([["A"] * 10, split], [ANSWER, condensed, ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "And its colour?", qid="q2")
    assert "__interrupt__" not in state
    assert state["decision"] == "stay" and retriever.filters == ["A", "A"]
    assert state["standalone"] == "Which colour is the fruit?"
    assert len(state["history"]) == 2


def test_named_switch() -> None:
    # no referring word -> condense makes no LLM call
    graph, retriever, _, _ = make([["A"] * 10, ["A"] * 10], [ANSWER, ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "About the Beta paper: which fruit?", qid="q2")
    assert state["decision"] == "named" and state["active_paper_id"] == "B"
    assert retriever.filters == ["A", "B"]


def test_condense_fallback() -> None:
    graph, _, _, _ = make([["A"] * 10, ["A"] * 10], [ANSWER, LLMError("down"), ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "And its colour?", qid="q2")
    assert state["standalone"] == "And its colour?"
    assert state["condense_error"] and "LLMError" in state["condense_error"]
    assert state["answer"].text == "apples"


def test_no_gold_paper_id_read() -> None:
    graph, retriever, corpus, _ = make([["A"] * 10], [ANSWER])
    turn(graph, "t", "Which fruit?")
    assert "GOLD" not in retriever.filters and corpus.filters == [None]


def test_vote_uses_typed_text_not_condensed() -> None:
    # the condensed question drags in the old topic; switches are detected on what was typed
    condensed = '{"question": "Which colour is the fruit in the apples paper?"}'
    graph, _, corpus, _ = make([["A"] * 10, ["B"] * 10], [ANSWER, condensed, ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "Which colour is it?", qid="q2")
    assert corpus.queries == ["Which fruit?", "Which colour is it?"]
    assert state["standalone"] == "Which colour is the fruit in the apples paper?"


def test_condense_rejects_added_paper_title() -> None:
    added = '{"question": "In Alpha: a paper about apples, which colour?"}'
    graph, _, _, _ = make([["A"] * 10, ["A"] * 10], [ANSWER, added, ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "Which colour is it?", qid="q2")
    assert state["standalone"] == "Which colour is it?"
    assert state["condense_error"] == "rejected: added a paper title"


def test_condense_only_on_referring_words() -> None:
    graph, _, _, llm = make([["A"] * 10, ["A"] * 10], [ANSWER, ANSWER])
    turn(graph, "t", "Which fruit?")
    state, _ = turn(graph, "t", "What baselines do they compare to?", qid="q2")
    assert state["standalone"] == "What baselines do they compare to?"
    assert state["llm_calls"] == 1 and len(llm.calls) == 2  # two generate calls, no condense
