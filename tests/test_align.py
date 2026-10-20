from paperlens.domain import Paper, Paragraph, Question, Reference
from paperlens.evals.answer_metrics import score_answer
from paperlens.ingestion.align import align_paragraph, align_questions

PID = "P"
SENTENCE_A = "We evaluate the proposed encoder on three public benchmarks and report accuracy"
SENTENCE_B = "and the results show consistent gains over the strong baselines in every setting"
MIN_SCORE = 85.0
MIN_WORDS = 8


def _para(pid: str, text: str, caption: bool = False) -> Paragraph:
    return Paragraph(paragraph_id=pid, section="S", text=text, is_caption=caption)


def _paper(paragraphs: list[Paragraph]) -> Paper:
    return Paper(paper_id=PID, title="T", abstract="", paragraphs=paragraphs)


def _question(evidence: list[str]) -> Question:
    ref = Reference(answer_type="extractive", answer_text="x", evidence_ids=evidence)
    return Question(question_id="q", paper_id=PID, question="?", references=[ref])


def test_align_exact_paragraph() -> None:
    candidates = [_para("P::d0", SENTENCE_A), _para("P::d1", "Completely different text " * 5)]
    assert align_paragraph(SENTENCE_A, candidates, MIN_SCORE, MIN_WORDS) == ["P::d0"]


def test_align_split_paragraph_maps_to_both_halves() -> None:
    qasper = _paper([_para("P::s0p0", f"{SENTENCE_A} {SENTENCE_B}")])
    pdf = _paper(
        [
            _para("P::d0", SENTENCE_A),
            _para("P::d1", "Unrelated filler sentence about something else entirely here."),
            _para("P::d2", SENTENCE_B),
        ]
    )
    questions, stats = align_questions(qasper, pdf, [_question(["P::s0p0"])], MIN_SCORE, MIN_WORDS)
    assert questions[0].references[0].evidence_ids == ["P::d0", "P::d2"]
    assert (stats.evidence, stats.aligned, stats.split) == (1, 1, 1)


def test_short_fragments_never_match() -> None:
    candidates = [_para("P::d0", "Results"), _para("P::d1", SENTENCE_A)]
    assert align_paragraph(f"Results {SENTENCE_A}", candidates, MIN_SCORE, MIN_WORDS) == ["P::d1"]


def test_short_captions_are_exempt_from_min_words() -> None:
    candidates = [_para("P::d0", "Table 2: Results", caption=True)]
    assert align_paragraph("Table 2: Results", candidates, MIN_SCORE, MIN_WORDS) == ["P::d0"]


def test_unaligned_evidence_becomes_sentinel() -> None:
    qasper = _paper([_para("P::s0p0", SENTENCE_A)])
    pdf = _paper([_para("P::d0", "Nothing alike appears in this other paragraph at all today.")])
    questions, stats = align_questions(qasper, pdf, [_question(["P::s0p0"])], MIN_SCORE, MIN_WORDS)
    ref = questions[0].references[0]
    assert ref.evidence_ids == ["P::unaligned0"]
    assert ref.unmatched_evidence == 1
    assert (stats.evidence, stats.aligned) == (1, 0)


def test_unaligned_reference_cannot_score_empty_evidence() -> None:
    qasper = _paper([_para("P::s0p0", SENTENCE_A)])
    pdf = _paper([_para("P::d0", "Nothing alike appears in this other paragraph at all today.")])
    questions, _ = align_questions(qasper, pdf, [_question(["P::s0p0"])], MIN_SCORE, MIN_WORDS)
    best = score_answer("x", [], questions[0].references)
    assert best.evidence_f1 == 0.0


def test_references_without_evidence_stay_untouched() -> None:
    qasper = _paper([_para("P::s0p0", SENTENCE_A)])
    pdf = _paper([_para("P::d0", SENTENCE_A)])
    questions, stats = align_questions(qasper, pdf, [_question([])], MIN_SCORE, MIN_WORDS)
    assert questions[0].references[0].evidence_ids == [] and stats.evidence == 0
