from paperlens.config.models import RouteConfig
from paperlens.domain import Chunk, ScoredChunk
from paperlens.graph.routing import decide, match_title, vote

TITLES = {
    "A": "BERT: Pre-training of Deep Bidirectional Transformers",
    "B": "Neural attention for question answering",
    "C": "Attention: a survey",
}


def chunk(paper_id: str, i: int) -> ScoredChunk:
    c = Chunk(
        chunk_id=f"{paper_id}::c{i}", paper_id=paper_id, section="s", text="t", paragraph_ids=[]
    )
    return ScoredChunk(chunk=c, score=1.0 - i / 100)


def test_vote_shares() -> None:
    papers = ["A"] * 6 + ["B"] * 3 + ["C"]
    votes = vote([chunk(p, i) for i, p in enumerate(papers)])
    assert [(p, round(s, 2)) for p, s in votes] == [("A", 0.6), ("B", 0.3), ("C", 0.1)]


def test_vote_tie_keeps_first_seen() -> None:
    votes = vote([chunk("B", 0), chunk("A", 1)])
    assert [p for p, _ in votes] == ["B", "A"]


def test_named_full_and_short() -> None:
    full = f'In the paper "{TITLES["A"]}", which dataset do they use?'
    assert match_title(full, TITLES, 90) == "A"
    assert match_title("About the BERT paper: which dataset?", TITLES, 90) == "A"
    assert match_title("In BERT, what is the baseline?", TITLES, 90) == "A"


def test_short_word_no_false_match() -> None:
    assert match_title("In Attention, what is the baseline?", TITLES, 90) == "C"
    # a common word in lower case is not a mention of the paper "Attention: a survey"
    assert match_title("how does attention help the model?", TITLES, 90) is None


def test_clarify_reply_prefers_candidates() -> None:
    assert match_title(TITLES["B"], TITLES, 90, candidates=["B", "C"]) == "B"
    assert match_title("nonsense reply", TITLES, 90) is None


def test_decide_rules() -> None:
    cfg = RouteConfig()
    clear = [("A", 0.7), ("B", 0.2)]
    split = [("A", 0.4), ("B", 0.3), ("C", 0.3)]
    assert decide("Which dataset?", None, clear, TITLES, cfg) == ("vote", "A")
    assert decide("Which dataset?", "B", clear, TITLES, cfg) == ("vote", "A")
    assert decide("Which dataset?", "B", split, TITLES, cfg) == ("stay", "B")
    assert decide("Which dataset?", None, split, TITLES, cfg) == ("clarify", None)
    assert decide("Which dataset?", None, [], TITLES, cfg) == ("clarify", None)
    assert decide("In Attention, which dataset?", "A", clear, TITLES, cfg) == ("named", "C")


def test_full_title_beats_other_short_title() -> None:
    # 'Attention' (short title of C) appears inside B's title-cased full title
    titles = {"B": "Neural Attention for Question Answering", "C": "Attention: a survey"}
    asked = 'In the paper "Neural Attention for Question Answering", what is the baseline?'
    assert match_title(asked, titles, 90) == "B"
    assert match_title(titles["B"], titles, 90, candidates=["B", "C"]) == "B"
