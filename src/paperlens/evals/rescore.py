"""Recompute a saved run's records, metrics and report without calling the LLM again.

Purpose: When a metric is fixed or added (e.g. the nDCG fix, Evidence-F1 ceiling), every
past run can be re-scored from what it already saved: the model's answers and citations
(records.jsonl) plus the cached chunks. Minutes instead of hours of GPU time.
Flow: evals/results/<run>/{params.json, records.jsonl} + data/index/<key>/chunks.jsonl +
QASPER questions -> rescore_run() -> rewritten records.jsonl, metrics.json, report.md.
Params: everything is read back from the run's params.json (meta.* keys are kept as-is).
Try: Change a metric in evals/, run `paperlens rescore`, and diff SUMMARY.md before/after:
only the metric moved, because the answers are the same.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from paperlens.config.models import AppConfig
from paperlens.domain import Answer, Chunk, Paper, PipelineResult, Question, ScoredChunk, Usage
from paperlens.evals.aggregate import aggregate
from paperlens.evals.judge import judge_metrics, load_verdicts, rescored_verdicts
from paperlens.evals.provenance import collect_meta
from paperlens.evals.records import build_record
from paperlens.evals.report import (
    PARAMS_FILE,
    RECORDS_FILE,
    RunResult,
    load_run,
    render_report,
    save_run,
)
from paperlens.index.store import CHUNKS_FILE, index_key
from paperlens.ingestion.qasper import download_split, load_split

META_PREFIX = "meta."
REPORT_FILE = "report.md"
# provenance a pre-provenance run cannot reconstruct honestly
UNRECORDED = {
    "meta.git_sha": "pre-provenance",
    "meta.git_dirty": "pre-provenance",
    "meta.llm_digest": "not recorded",
    "meta.server_version": "not recorded",
}


def unflatten(flat: Mapping[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        node = nested
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return nested


def _answer_from_record(raw: dict[str, Any], chunks: Mapping[str, Chunk]) -> Answer:
    cited = [chunks[c] for c in raw["cited_chunk_ids"]]
    if "cited_paragraph_ids" in raw:  # the model's own paragraph citations, kept exactly
        evidence_ids = list(raw["cited_paragraph_ids"])
    else:  # old record: a cited chunk counts as all of its paragraphs
        evidence_ids = list(dict.fromkeys(p for c in cited for p in c.paragraph_ids))
    return Answer(
        question_id=raw["question_id"],
        answer_type=raw["predicted_type"],
        text=raw["prediction"],
        citations=raw["cited_chunk_ids"],
        evidence_ids=evidence_ids,
        refused=raw["refused"],
        error=raw["error"],
        # total latency was stored; keep it whole so latency metrics are unchanged
        usage=Usage(
            prompt_tokens=raw["prompt_tokens"],
            completion_tokens=raw["completion_tokens"],
            latency_ms=raw["latency_ms"],
        ),
    )


def rescore_run(
    run_dir: Path, *, dataset: tuple[list[Paper], list[Question]] | None = None
) -> RunResult:
    params: dict[str, Any] = json.loads((run_dir / PARAMS_FILE).read_text(encoding="utf-8"))
    cfg = AppConfig(**unflatten({k: v for k, v in params.items() if not k.startswith(META_PREFIX)}))
    _, questions = dataset or load_split(download_split(cfg.data, cfg.data.split))
    by_id = {q.question_id: q for q in questions}
    chunk_file = cfg.data.data_dir / "index" / index_key(cfg) / CHUNKS_FILE
    lines = chunk_file.read_text(encoding="utf-8").splitlines()
    chunks = {c.chunk_id: c for c in (Chunk.model_validate_json(line) for line in lines if line)}
    k_values = [k for k in cfg.eval.k_values if k <= cfg.retrieval.top_k]

    records = []
    for line in (run_dir / RECORDS_FILE).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        result = PipelineResult(
            answer=_answer_from_record(raw, chunks),
            retrieved=[ScoredChunk(chunk=chunks[c], score=0.0) for c in raw["retrieved_chunk_ids"]],
            retrieval_ms=0.0,
        )
        records.append(
            build_record(by_id[raw["question_id"]], result, k_values, cfg.generation.cite_unit)
        )

    metrics = aggregate(records, k_values)
    metrics["index.num_chunks"] = float(len(chunks))
    # LLM verdicts cannot be recomputed: keep those whose answer is unchanged, redo the rules
    metrics.update(judge_metrics(records, rescored_verdicts(records, load_verdicts(run_dir))))
    if "meta.prompt_sha256" not in params:
        params.update(collect_meta(cfg, query_server=False))
        params.update(UNRECORDED)
    run = RunResult(run_name=run_dir.name, params=params, metrics=metrics, records=records)
    save_run(run, run_dir)
    ref_name = cfg.eval.reference_run
    reference = None
    if ref_name and ref_name != run_dir.name:
        reference = load_run(run_dir.parent / ref_name)
    (run_dir / REPORT_FILE).write_text(
        render_report(run, reference, cfg.eval.worst_examples), encoding="utf-8"
    )
    return run
