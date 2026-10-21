from pathlib import Path

from test_graph import ANSWER, TITLES, make

from paperlens.graph.converse import converse, load_demo


def test_converse_transcript(tmp_path: Path) -> None:
    demo = tmp_path / "demo.jsonl"
    demo.write_text(
        '{"conversation_id": "d1", "text": "Which fruit?", "reply": "Beta: a paper about bananas"}\n'
        '{"conversation_id": "d1", "text": "And its colour?"}\n'
    )
    condensed = '{"question": "Which colour is the banana?"}'
    graph, retriever, _, _ = make([["A", "B"] * 5, ["A", "B"] * 5], [ANSWER, condensed, ANSWER])
    text = converse(graph, load_demo(demo), TITLES)
    assert "agent asks: Which paper do you mean?" in text
    assert "decision: `clarify` -> Beta: a paper about bananas" in text
    assert "standalone question: Which colour is the banana?" in text
    assert "decision: `stay`" in text and "**answer:** apples" in text
    assert retriever.filters == ["B", "B"]
