"""Wires the M2 nodes into a LangGraph StateGraph with one interrupt.

Purpose: The whole agent in one picture: condense -> resolve_paper -> (clarify) ->
retrieve -> generate, with refuse when a clarification names no known paper. Conditional
edges read the node outputs; nothing loops, so a turn always ends.
Flow: AgentDeps + checkpointer -> build_graph() -> CompiledStateGraph ->
evals/conversation_runner.py and `paperlens converse`. build_agent_deps() makes the deps
from an AppConfig and a loaded index.
Params: everything under agent.*, plus retrieval.*, generation.*, llm.* through the deps.
Try: print(build_graph(deps, InMemorySaver()).get_graph().draw_mermaid()) - the diagram in
the README is generated this way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from paperlens.config.models import AppConfig
from paperlens.graph.nodes import AgentDeps, Nodes
from paperlens.graph.state import AgentState
from paperlens.index.store import IndexBundle
from paperlens.llm.client import LLMClient
from paperlens.pipelines.factory import build_pipeline
from paperlens.retrieval.retriever import Retriever

__all__ = ["AgentDeps", "build_agent_deps", "build_graph"]


def _after_resolve(state: AgentState) -> str:
    return "clarify" if state["decision"] == "clarify" else "retrieve"


def _after_clarify(state: AgentState) -> str:
    return "retrieve" if state.get("resolved_paper_id") else "refuse"


def build_graph(deps: AgentDeps, checkpointer: Any) -> Any:
    nodes = Nodes(deps)
    graph = StateGraph(AgentState)
    graph.add_node("condense", nodes.condense)
    graph.add_node("resolve_paper", nodes.resolve_paper)
    graph.add_node("clarify", nodes.clarify)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("generate", nodes.generate)
    graph.add_node("refuse", nodes.refuse)
    graph.add_edge(START, "condense")
    graph.add_edge("condense", "resolve_paper")
    graph.add_conditional_edges("resolve_paper", _after_resolve, ["clarify", "retrieve"])
    graph.add_conditional_edges("clarify", _after_clarify, ["retrieve", "refuse"])
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)
    return graph.compile(checkpointer=checkpointer)


def build_agent_deps(
    cfg: AppConfig, bundle: IndexBundle, llm: LLMClient, titles: Mapping[str, str]
) -> AgentDeps:
    generator = build_pipeline(cfg, bundle, llm, titles)  # same prompt/citations as m1b_best
    return AgentDeps(
        retriever=Retriever(cfg.retrieval, bundle.dense, bundle.sparse),
        corpus=bundle.dense,
        generator=generator,
        llm=llm,
        titles=titles,
        cfg=cfg.agent,
        condense_template=cfg.agent.condense.prompt_template.read_text(encoding="utf-8"),
    )
