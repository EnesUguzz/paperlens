import pytest

from paperlens.domain import Reference
from paperlens.evals.retrieval_metrics import best_over_references, mrr, ndcg_at_k, recall_at_k

RANKED = [["p1", "p2"], ["p3"], ["p4"]]
GOLD = {"p3", "p4"}


def test_recall_at_k() -> None:
    assert recall_at_k(RANKED, GOLD, 1) == 0.0
    assert recall_at_k(RANKED, GOLD, 2) == 0.5
    assert recall_at_k(RANKED, GOLD, 3) == 1.0


def test_mrr() -> None:
    assert mrr(RANKED, GOLD) == 0.5
    assert mrr(RANKED, {"zz"}) == 0.0


def test_ndcg() -> None:
    assert ndcg_at_k(RANKED, GOLD, 3) == pytest.approx(0.6934, abs=1e-4)


def test_best_over_references_skips_evidence_free_refs() -> None:
    refs = [
        Reference(answer_type="extractive", answer_text="a", evidence_ids=["p3", "p4"]),
        Reference(answer_type="unanswerable", answer_text="Unanswerable", evidence_ids=[]),
    ]
    out = best_over_references(RANKED, refs, [2])
    assert out == {"recall_at_2": 0.5, "ndcg_at_2": pytest.approx(0.3869, abs=1e-4), "mrr": 0.5}
    assert best_over_references(RANKED, refs[1:], [2]) is None
