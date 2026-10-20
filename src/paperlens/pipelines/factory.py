"""Builds the configured pipeline from an AppConfig and an index.

Purpose: One place that wires Retriever + BaselinePipeline, so evals and trace cannot
drift apart when M1b adds a reranker and query transforms.
Flow: AppConfig + IndexBundle + LLMClient + titles -> (optional Reranker from rerank.*) ->
Retriever -> BaselinePipeline.
Params: retrieval.*, rerank.*, query.* (multi-query / HyDE transformer), generation.prompt_template, generation.cite_unit.
Try: Call build_pipeline() in a notebook and run pipeline.plan(q) / pipeline.retrieve(q,
plan) separately to inspect each stage.
"""

from __future__ import annotations

from collections.abc import Mapping

from paperlens.config.models import AppConfig
from paperlens.index.store import IndexBundle
from paperlens.llm.client import LLMClient
from paperlens.pipelines.baseline import BaselinePipeline
from paperlens.retrieval.query import build_transformer
from paperlens.retrieval.reranker import Reranker, build_reranker
from paperlens.retrieval.retriever import Retriever


def build_pipeline(
    cfg: AppConfig,
    bundle: IndexBundle,
    llm: LLMClient,
    titles: Mapping[str, str],
    reranker: Reranker | None = None,
) -> BaselinePipeline:
    if reranker is None:
        reranker = build_reranker(cfg.rerank)  # None when rerank.enabled is false
    retriever = Retriever(cfg.retrieval, bundle.dense, bundle.sparse, reranker)
    template = cfg.generation.prompt_template.read_text(encoding="utf-8")
    transformer = build_transformer(cfg.query, llm)  # identity when query.transform is none
    return BaselinePipeline(retriever, llm, template, titles, transformer, cfg.generation.cite_unit)
