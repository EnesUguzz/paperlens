"""Run one experiment end to end from a single AppConfig.

Purpose: The experiment loop: data -> index (cached) -> pipeline -> per-question records
-> metrics -> saved run + report + MLflow. Everything it does is decided by the config.
Flow: AppConfig -> run_experiment() -> evals/results/<run_name>/{params.json,
metrics.json, records.jsonl, report.md} + MLflow run.
Params: every section; notably data.sample_size, eval.k_values, eval.reference_run,
eval.system (single_paper = M1 loop; corpus_rag / agent = M2 conversations).
Provenance (prompt/data/git/model hashes) is added as meta.* params by evals/provenance.py.
Try: Run the same config twice: metrics should match exactly (temperature 0, fixed seed).
If they do not, the server is not deterministic - worth knowing before comparing runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import structlog

from paperlens.config.models import AppConfig
from paperlens.domain import Paper, Question
from paperlens.evals.aggregate import aggregate
from paperlens.evals.provenance import collect_meta
from paperlens.evals.records import EvalRecord, build_record
from paperlens.evals.report import RunResult, load_run, render_report, save_run
from paperlens.evals.routing_metrics import routing_metrics
from paperlens.evals.tracking import log_to_mlflow
from paperlens.index.embedder import Embedder, build_embedder
from paperlens.index.store import IndexBundle, build_or_load, index_key
from paperlens.ingestion.sources import resolve_dataset, sample_dataset
from paperlens.llm.client import LLMClient, OpenAICompatClient
from paperlens.pipelines.baseline import Pipeline
from paperlens.pipelines.factory import build_pipeline

log = structlog.get_logger(__name__)
REPORT_FILE = "report.md"


def flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, prefix=f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def evaluate(
    questions: Sequence[Question],
    pipeline: Pipeline,
    k_values: Sequence[int],
    log_every: int,
    cite_unit: Literal["chunk", "paragraph"] = "chunk",
) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for i, question in enumerate(questions, start=1):
        records.append(build_record(question, pipeline.run(question), k_values, cite_unit))
        if i % log_every == 0 or i == len(questions):
            mean_f1 = sum(r.answer_f1 for r in records) / len(records)
            log.info("progress", done=i, total=len(questions), answer_f1=round(mean_f1, 3))
    return records


def _run_conversations(
    cfg: AppConfig,
    sample: Sequence[Question],
    bundle: IndexBundle,
    llm: LLMClient,
    titles: Mapping[str, str],
    k_values: Sequence[int],
) -> list[EvalRecord]:
    """M2: mix the sample into conversations, then play them through corpus_rag or the agent."""
    from langgraph.checkpoint.memory import InMemorySaver

    from paperlens.evals.conversation_runner import enable_tracing, run_conversations
    from paperlens.evals.conversations import build_conversations, save_conversations
    from paperlens.graph.build import build_agent_deps, build_graph

    conversations = build_conversations(sample, titles, cfg.conversations)
    save_conversations(conversations, cfg.conversations.path)
    questions = {q.question_id: q for q in sample}
    if cfg.eval.system == "corpus_rag":
        pipeline = build_pipeline(cfg, bundle, llm, titles, scope="corpus")
        return run_conversations(
            conversations,
            questions,
            titles,
            k_values,
            cfg.generation.cite_unit,
            pipeline=pipeline,
            log_every=cfg.eval.log_every,
        )
    tracing = cfg.agent.tracing and enable_tracing(cfg)
    graph = build_graph(build_agent_deps(cfg, bundle, llm, titles), InMemorySaver())
    return run_conversations(
        conversations,
        questions,
        titles,
        k_values,
        cfg.generation.cite_unit,
        graph=graph,
        tracing=tracing,
        log_every=cfg.eval.log_every,
    )


def run_experiment(
    cfg: AppConfig,
    *,
    dataset: tuple[list[Paper], list[Question]] | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> RunResult:
    loaded = resolve_dataset(cfg.data, dataset)
    papers = loaded.papers
    sample, dropped = sample_dataset(loaded, cfg.data.sample_size, cfg.data.seed)
    bundle = build_or_load(cfg, papers, embedder or build_embedder(cfg.embedding))
    k_values = [k for k in cfg.eval.k_values if k <= cfg.retrieval.top_k]
    titles = {p.paper_id: p.title for p in papers}
    client = llm or OpenAICompatClient(cfg.llm)
    try:
        log.info(
            "run start",
            run=cfg.eval.run_name,
            system=cfg.eval.system,
            questions=len(sample),
            index=index_key(cfg, papers),
        )
        if cfg.eval.system == "single_paper":
            pipeline = build_pipeline(cfg, bundle, client, titles)
            records = evaluate(
                sample, pipeline, k_values, cfg.eval.log_every, cfg.generation.cite_unit
            )
        else:
            records = _run_conversations(cfg, sample, bundle, client, titles, k_values)
    finally:
        bundle.dense.close()

    metrics = aggregate(records, k_values)
    metrics.update(routing_metrics(records))  # empty for single-paper runs
    metrics["index.num_chunks"] = float(len(bundle.chunks))
    metrics["data.questions_dropped"] = float(dropped)  # sampled questions without a parsed PDF
    metrics.update(loaded.metrics)
    params = flatten(cfg.model_dump(mode="json"))
    params.update(collect_meta(cfg, query_server=llm is None))  # a fake LLM has no server
    result = RunResult(
        run_name=cfg.eval.run_name,
        params=params,
        metrics=metrics,
        records=records,
    )
    run_dir = cfg.eval.results_dir / cfg.eval.run_name
    save_run(result, run_dir)
    reference = None
    if cfg.eval.reference_run and cfg.eval.reference_run != cfg.eval.run_name:
        reference = load_run(cfg.eval.results_dir / cfg.eval.reference_run)
        if reference is None:
            log.warning("reference run not found", reference=cfg.eval.reference_run)
    report = render_report(result, reference, cfg.eval.worst_examples)
    (run_dir / REPORT_FILE).write_text(report, encoding="utf-8")
    log_to_mlflow(cfg, result.params, metrics, run_dir)
    log.info("run done", run=cfg.eval.run_name, report=str(run_dir / REPORT_FILE))
    return result
