"""Which paper is this turn about? Pure, deterministic rules (no LLM call).

Purpose: The agent's one real decision, kept out of the graph so it can be unit-tested and
tuned with a single threshold: a named paper wins, then a clear corpus vote, then the
active paper; with none of those the agent asks.
Flow: turn text + active paper + corpus top-k chunks -> vote() -> decide() ->
(decision, paper_id) -> graph/nodes.py. match_title() also resolves the user's reply to a
clarifying question.
Params: agent.route.min_lead, agent.route.title_match_min (top_k is used by the node).
Try: Raise agent.route.min_lead from 0.3 to 0.5 and compare route.clarify_rate with
route.wrong_paper_rate in the report: that pair is the whole trade-off.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from rapidfuzz import fuzz

from paperlens.config.models import RouteConfig
from paperlens.domain import ScoredChunk
from paperlens.evals.conversations import short_title

Decision = Literal["named", "vote", "stay", "clarify"]
MIN_SHORT_TITLE_CHARS = 4  # "QA" or "NLP" would match far too many questions
EXACT = 100.0


def vote(chunks: Sequence[ScoredChunk]) -> list[tuple[str, float]]:
    """(paper_id, share of the chunks) best first; ties keep the paper seen first."""
    if not chunks:
        return []
    counts: dict[str, int] = {}
    for sc in chunks:
        counts[sc.chunk.paper_id] = counts.get(sc.chunk.paper_id, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])  # stable: first seen wins ties
    return [(pid, n / len(chunks)) for pid, n in ranked]


def _score(text: str, title: str) -> float:
    full = fuzz.partial_ratio(title.casefold(), text.casefold())
    short = short_title(title)
    # Short names count only as a case-sensitive whole word: "In BERT, ..." names the
    # paper, "how does attention help?" does not name "Attention: a survey".
    if len(short) >= MIN_SHORT_TITLE_CHARS and re.search(rf"\b{re.escape(short)}\b", text):
        return EXACT
    return float(full)


def _best(text: str, titles: Mapping[str, str], ids: Sequence[str], min_score: float) -> str | None:
    scored = [(_score(text, titles[pid]), pid) for pid in ids if titles.get(pid)]
    if not scored:
        return None
    top = max(s for s, _ in scored)
    winners = [pid for s, pid in scored if s == top]
    return winners[0] if top >= min_score and len(winners) == 1 else None


def match_title(
    text: str, titles: Mapping[str, str], min_score: float, candidates: Sequence[str] = ()
) -> str | None:
    """The one paper the text names (full or short title), or None if none or ambiguous."""
    if candidates:
        hit = _best(text, titles, list(candidates), min_score)
        if hit is not None:
            return hit
    return _best(text, titles, list(titles), min_score)


def decide(
    text: str,
    active: str | None,
    votes: Sequence[tuple[str, float]],
    titles: Mapping[str, str],
    cfg: RouteConfig,
) -> tuple[Decision, str | None]:
    named = match_title(text, titles, cfg.title_match_min)
    if named is not None:
        return "named", named
    if votes:
        lead = votes[0][1] - (votes[1][1] if len(votes) > 1 else 0.0)
        if lead >= cfg.min_lead:
            return "vote", votes[0][0]
    if active is not None:
        return "stay", active  # no clear signal: a person would assume the same paper
    return "clarify", None
