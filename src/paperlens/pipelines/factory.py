"""Builds the configured pipeline from an AppConfig and an index.

Purpose: One place that wires Retriever + BaselinePipeline, so evals and trace cannot
drift apart when M1b adds a reranker and query transforms.
Flow: AppConfig + IndexBundle + LLMClient + titles -> Retriever -> BaselinePipeline.
Params: retrieval.*, generation.prompt_template (reranker and query transform arrive in
later M1b tasks; both are off here).
Try: Call build_pipeline() in a notebook and run pipeline.plan(q) / pipeline.retrieve(q,
plan) separately to inspect each stage.
"""

from __future__ import annotations

from collections.abc import Mapping

from paperlens.config.models import AppConfig
from paperlens.index.store import IndexBundle
from paperlens.llm.client import LLMClient
from paperlens.pipelines.baseline import BaselinePipeline
from paperlens.retrieval.reranker import Reranker
from paperlens.retrieval.retriever import Retriever


def build_pipeline(
    cfg: AppConfig,
    bundle: IndexBundle,
    llm: LLMClient,
    titles: Mapping[str, str],
    reranker: Reranker | None = None,
) -> BaselinePipeline:
    retriever = Retriever(cfg.retrieval, bundle.dense, bundle.sparse, reranker)
    template = cfg.generation.prompt_template.read_text(encoding="utf-8")
    return BaselinePipeline(retriever, llm, template, titles)
