"""Sweep agent.route.min_lead without the LLM: which paper would the agent use on each turn?

Purpose: Routing is deterministic and the corpus vote runs on the typed text, so the paper
the agent picks on every turn depends only on min_lead, the corpus search and the earlier
decisions in the conversation. That makes the whole threshold curve a seconds-long
replay instead of one 10-minute LLM run per value; only the few values worth it need a
real `paperlens eval`. The replay of the chosen value matches the real run's
route.paper_accuracy exactly.
Flow: conversations (evals/conversations/m2.jsonl) + corpus votes per turn (dense search,
top_k) -> replay_routing() per value -> sweep() -> render_sweep() -> Markdown table
(`paperlens route-sweep`).
Params: agent.route.top_k, agent.route.title_match_min, agent.clarify.max_candidates; the
values to try are a CLI option.
Try: Run it, pick the plateau, then run only its ends with `paperlens eval`: the answer-F1
differences inside a flat plateau are usually smaller than run-to-run noise.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from paperlens.config.models import RouteConfig
from paperlens.evals.conversations import Turn
from paperlens.evals.routing_metrics import TURN_TYPES
from paperlens.graph.routing import decide, match_title

Votes = Mapping[str, Sequence[tuple[str, float]]]  # question_id -> (paper_id, share) best first


def replay_routing(
    conversations: Sequence[Sequence[Turn]],
    votes: Votes,
    titles: Mapping[str, str],
    route: RouteConfig,
    max_candidates: int,
) -> dict[str, float]:
    """The agent's paper decisions for one threshold, as in graph/nodes.py, without answers."""
    hits: dict[str, list[bool]] = defaultdict(list)
    asked = 0
    for conversation in conversations:
        active: str | None = None
        for turn in conversation:
            turn_votes = list(votes[turn.question_id])
            decision, paper_id = decide(turn.text, active, turn_votes, titles, route)
            if decision == "clarify":
                asked += 1
                top = [pid for pid, _ in turn_votes][:max_candidates]
                # simulated user: replies with the gold title, as in the eval
                paper_id = match_title(
                    titles.get(turn.paper_id, ""), titles, route.title_match_min, top
                )
            active = paper_id  # generate keeps it; refuse (no paper) clears it
            hits[turn.turn_type].append(paper_id == turn.paper_id)
    every = [h for kind in hits.values() for h in kind]
    out = {
        "paper_accuracy": sum(every) / len(every) if every else 0.0,
        "clarify_rate": asked / len(every) if every else 0.0,
    }
    for kind in TURN_TYPES:
        if hits[kind]:
            out[f"paper_accuracy.{kind}"] = sum(hits[kind]) / len(hits[kind])
    return out


def sweep(
    conversations: Sequence[Sequence[Turn]],
    votes: Votes,
    titles: Mapping[str, str],
    route: RouteConfig,
    max_candidates: int,
    values: Sequence[float],
) -> list[dict[str, float]]:
    rows = []
    for value in values:
        cfg = route.model_copy(update={"min_lead": value})
        row = replay_routing(conversations, votes, titles, cfg, max_candidates)
        rows.append({"min_lead": value, **row})
    return rows


def render_sweep(rows: Sequence[Mapping[str, float]]) -> str:
    cols = [
        ("paper_accuracy", "paper accuracy"),
        ("clarify_rate", "clarify rate"),
        *[(f"paper_accuracy.{k}", k) for k in TURN_TYPES],
    ]
    lines = [
        "| min_lead | " + " | ".join(label for _, label in cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for row in rows:
        cells = [f"{row[key]:.3f}" if key in row else "-" for key, _ in cols]
        lines.append(f"| {row['min_lead']:g} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
