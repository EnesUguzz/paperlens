from paperlens.config.models import RouteConfig
from paperlens.evals.conversations import Turn
from paperlens.evals.route_sweep import render_sweep, replay_routing, sweep

TITLES = {"A": "Alpha: apples", "B": "Beta: bananas"}


def t(i: int, qid: str, paper: str, kind: str, text: str = "Which fruit?") -> Turn:
    return Turn(
        conversation_id="c0",
        turn_index=i,
        question_id=qid,
        paper_id=paper,
        text=text,
        turn_type=kind,
    )


CONV = [[t(0, "q1", "A", "first"), t(1, "q2", "A", "same"), t(2, "q3", "B", "switch_unsignaled")]]
VOTES = {
    "q1": [("A", 0.4), ("B", 0.3)],  # lead 0.1: asks unless the threshold is 0.1 or lower
    "q2": [("B", 0.6), ("A", 0.2)],  # lead 0.4: pulls a 'same' turn to the wrong paper
    "q3": [("B", 0.9), ("A", 0.1)],  # lead 0.8: catches the silent switch
}


def test_replay_low_threshold_follows_every_vote() -> None:
    m = replay_routing(CONV, VOTES, TITLES, RouteConfig(min_lead=0.0), max_candidates=3)
    assert m["clarify_rate"] == 0.0
    assert m["paper_accuracy.first"] == 1.0
    assert m["paper_accuracy.same"] == 0.0  # q2 voted away to B
    assert m["paper_accuracy.switch_unsignaled"] == 1.0


def test_replay_mid_threshold_asks_first_and_stays() -> None:
    m = replay_routing(CONV, VOTES, TITLES, RouteConfig(min_lead=0.5), max_candidates=3)
    assert m["clarify_rate"] == 1 / 3  # q1 asks; the simulated user names A
    assert m["paper_accuracy.same"] == 1.0  # q2 lead 0.4 < 0.5 -> stays on A
    assert m["paper_accuracy.switch_unsignaled"] == 1.0  # q3 lead 0.8 >= 0.5


def test_sweep_rows_and_table() -> None:
    rows = sweep(CONV, VOTES, TITLES, RouteConfig(), 3, [0.0, 0.5, 0.9])
    assert [r["min_lead"] for r in rows] == [0.0, 0.5, 0.9]
    assert rows[2]["paper_accuracy.switch_unsignaled"] == 0.0  # 0.9 never follows the vote
    table = render_sweep(rows)
    assert table.splitlines()[0].startswith("| min_lead | paper accuracy |")
    assert "| 0.5 |" in table
