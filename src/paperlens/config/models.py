"""Typed configuration for every tunable value in paperlens.

Purpose: Single source of truth for parameters. Every field documents what it does, its
typical range and what happens to the metrics when it changes, so experiments can be
reasoned about without reading the implementation.
Flow: experiments/configs/*.yaml -> config/loader.py -> AppConfig -> every component.
Params: all of them; run `paperlens explain-config` (or read docs/CONFIG.md) for the table.
Try: Copy experiments/configs/baseline.yaml, change one field, `make eval CONFIG=<file>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Upper bound used to check that a chunk (in words) fits the embedder window (in tokens).
# English scientific text is ~1.3 tokens/word; 1.5 leaves room for the section prefix.
TOKENS_PER_WORD_UPPER = 1.5


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    split: Literal["train", "validation", "test"] = Field(
        "validation",
        description=(
            "What: QASPER split to evaluate on. "
            "Effect: tune everything on 'validation'; run 'test' once for the final table, "
            "otherwise test scores become optimistic (eval contamination)."
        ),
    )
    sample_size: int | None = Field(
        250,
        ge=1,
        description=(
            "What: number of questions sampled from the split (null = all). "
            "Typical: 100-300 while iterating, null for final runs. "
            "Effect: higher = less noisy metrics but slower runs; below ~100 questions, "
            "differences of 1-2 F1 points are mostly noise."
        ),
    )
    seed: int = Field(
        13,
        description=(
            "What: random seed for question sampling. Typical: any fixed integer. "
            "Effect: a different seed draws a different subset; keep it fixed so runs are "
            "comparable."
        ),
    )
    text_source: Literal["qasper_text", "pdf_docling"] = Field(
        "qasper_text",
        description=(
            "What: where paper text comes from; 'qasper_text' = the dataset's pre-parsed "
            "paragraphs, 'pdf_docling' = the arXiv PDF parsed by Docling (run `prepare-pdfs` "
            "first; gold evidence is fuzzy-aligned to PDF paragraphs). Effect: measures how "
            "parser quality changes retrieval; papers without a parsed PDF are dropped from "
            "the evaluation."
        ),
    )
    data_dir: Path = Field(
        Path("data"),
        description="What: folder for downloads and indexes (git-ignored). Effect: none on metrics.",
    )
    parquet_url_template: str = Field(
        "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet",
        description="What: download URL of a QASPER split; '{split}' is replaced. Effect: none.",
    )
    pdf_url_template: str = Field(
        "https://arxiv.org/pdf/{paper_id}",
        description=(
            "What: download URL of a paper's PDF; '{paper_id}' is replaced by the arXiv id. "
            "Effect: none on metrics; only changes where `prepare-pdfs` fetches from."
        ),
    )
    pdf_download_delay_s: float = Field(
        3.0,
        ge=0,
        description=(
            "What: seconds to wait between two PDF requests. Typical: 3-5. "
            "Effect: arXiv's automated-access rule asks for at least 3 s; lower values risk "
            "an IP block, higher values only make `prepare-pdfs` slower (cache hits never wait)."
        ),
    )
    align_min_score: float = Field(
        85.0,
        ge=0,
        le=100,
        description=(
            "What: minimum rapidfuzz partial_ratio score (0-100) for a PDF paragraph to count "
            "as holding a gold evidence paragraph (text_source 'pdf_docling' only). "
            "Typical: 80-95. "
            "Effect: lower values accept wrong matches (evidence points at the wrong paragraph); "
            "higher values leave more evidence unaligned, which then scores as never found."
        ),
    )
    align_min_words: int = Field(
        8,
        ge=1,
        description=(
            "What: PDF paragraphs shorter than this many words cannot be alignment candidates "
            "(captions are exempt; text_source 'pdf_docling' only). Typical: 5-15. "
            "Effect: too low and a fragment such as 'Results' matches every paragraph that "
            "contains it; too high and genuinely short paragraphs lose their evidence label."
        ),
    )
    align_score_margin: float = Field(
        5.0,
        ge=0,
        le=100,
        description=(
            "What: a PDF paragraph only counts as a match if its score is within this many "
            "points of the best score for that gold paragraph (text_source 'pdf_docling' "
            "only). Typical: 3-10. "
            "Effect: lower values drop near-duplicate sibling paragraphs (fewer wrong ids) but "
            "can lose the second half of a genuinely split paragraph; higher values keep both."
        ),
    )


class ChunkingConfig(_Section):
    strategy: Literal["fixed", "section", "semantic"] = Field(
        "fixed",
        description=(
            "What: how papers are cut. 'fixed' = sliding window of chunk_size words that "
            "ignores structure; 'section' = whole paragraphs packed up to chunk_size words, "
            "never crossing a section; 'semantic' = sentences packed up to chunk_size words, "
            "split where neighbouring sentence embeddings diverge (never crossing a section). "
            "Effect: 'section' keeps chunks aligned with paragraphs (usually higher "
            "Evidence-F1); 'fixed' can split an answer across two chunks; 'semantic' follows "
            "topic shifts but costs one embedding pass over all sentences when indexing."
        ),
    )
    chunk_size: int = Field(
        200,
        ge=5,
        description=(
            "What: target chunk length in words (~1.3 tokens per word). Typical: 100-400. "
            "Effect: higher -> more context per chunk, Recall@k rises (M1a: 0.70 -> 0.86 at "
            "400) and prompt tokens/latency grow; Evidence-F1 drops mechanically because a cited "
            "chunk counts all its paragraphs (compare evidence_f1 to evidence_f1_ceiling). "
            "Lower -> sharper matches, but answers may be split across chunks."
        ),
    )
    chunk_overlap: int = Field(
        40,
        ge=0,
        description=(
            "What: words shared by consecutive chunks ('fixed' only). Typical: 10-25% of "
            "chunk_size. Effect: higher -> fewer answers cut at a boundary, but more chunks "
            "and duplicate text in the prompt. 0 -> smallest index, more boundary misses."
        ),
    )
    semantic_breakpoint_percentile: float = Field(
        90.0,
        gt=0,
        lt=100,
        description=(
            "What: 'semantic' only. A chunk is split where the distance between neighbouring "
            "sentences exceeds this percentile of the paper's distances. Typical: 80-95. "
            "Effect: lower -> more splits and smaller chunks; higher -> only the sharpest "
            "topic shifts split (chunks then mostly end at chunk_size)."
        ),
    )
    semantic_min_words: int = Field(
        50,
        ge=1,
        description=(
            "What: 'semantic' only. A chunk shorter than this is not split at a semantic "
            "breakpoint. Typical: 30-80. Effect: higher -> fewer tiny chunks but breakpoints "
            "are ignored more often; must stay below chunk_size (checked for 'semantic')."
        ),
    )
    include_section_title: bool = Field(
        True,
        description=(
            "What: prefix each chunk with '[Section name]'. Effect: gives the embedder and "
            "the LLM structural context ('Results' vs 'Related Work'); usually helps."
        ),
    )
    include_captions: bool = Field(
        True,
        description=(
            "What: index figure/table captions as paragraphs. Effect: QASPER evidence often "
            "points at tables; without captions those questions can never be retrieved."
        ),
    )

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkingConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.strategy == "semantic" and self.semantic_min_words >= self.chunk_size:
            raise ValueError("semantic_min_words must be smaller than chunk_size")
        return self


class EmbeddingConfig(_Section):
    model_name: str = Field(
        "BAAI/bge-m3",
        description=(
            "What: sentence-transformers model for dense vectors; 'hashing' = tiny fake for "
            "tests/CI. Effect: stronger models raise Recall@k; bigger ones cost VRAM and "
            "indexing time. Changing it rebuilds the index."
        ),
    )
    batch_size: int = Field(
        32,
        ge=1,
        description=(
            "What: passages embedded per GPU batch. Typical: 16-128. "
            "Effect: higher = faster indexing until VRAM runs out; no effect on metrics."
        ),
    )
    device: str = Field(
        "cuda",
        description=(
            "What: torch device ('cuda' or 'cpu'). Effect: 'cpu' works without a GPU but "
            "indexing is 20-50x slower; no effect on metrics."
        ),
    )
    max_seq_length: int = Field(
        1024,
        ge=32,
        description=(
            "What: max tokens the embedder reads per text; the rest is silently truncated. "
            "Typical: 512-1024 (bge-m3 supports 8192). Effect: must exceed chunk_size * 1.5 "
            "(checked) or chunk endings are ignored; higher costs memory."
        ),
    )
    query_prefix: str = Field(
        "",
        description=(
            "What: text prepended to queries (e5 models need 'query: '). "
            "Effect: a wrong prefix for the chosen model can cost several Recall points."
        ),
    )
    passage_prefix: str = Field(
        "",
        description=(
            "What: text prepended to passages (e5 models need 'passage: '). "
            "Effect: same as query_prefix; changing it rebuilds the index."
        ),
    )


class RetrievalConfig(_Section):
    method: Literal["dense", "bm25", "hybrid"] = Field(
        "dense",
        description=(
            "What: 'dense' = embedding similarity, 'bm25' = keyword match, 'hybrid' = both "
            "fused with Reciprocal Rank Fusion. Effect: bm25 wins on exact terms (dataset "
            "names, metrics, acronyms), dense on paraphrases; hybrid is usually most robust."
        ),
    )
    top_k: int = Field(
        5,
        ge=1,
        description=(
            "What: chunks passed to the LLM. Typical: 3-10. Effect: higher -> Recall@k up, "
            "but more prompt tokens, latency and distracting text (small LLMs may answer "
            "worse). Lower -> cheaper, misses multi-part evidence."
        ),
    )
    candidate_k: int = Field(
        30,
        ge=1,
        description=(
            "What: chunks each retriever returns before fusion ('hybrid'), and the pool size "
            "handed to the reranker when rerank.enabled. Typical: 20-100. Effect: higher lets "
            "RRF or the reranker rescue items the first stage ranked low; the reranker's cost "
            "grows linearly with it."
        ),
    )
    rrf_k: int = Field(
        60,
        ge=1,
        description=(
            "What: RRF damping constant; score = sum of 1/(rrf_k + rank). Typical: 10-100 "
            "(60 is the original paper's value). Effect: lower -> each retriever's top ranks "
            "dominate; higher -> rankings are blended more evenly."
        ),
    )
    bm25_k1: float = Field(
        1.5,
        ge=0,
        description=(
            "What: BM25 term-frequency saturation. Typical: 1.2-2.0. Effect: higher -> "
            "repeated query words keep adding score; lower -> one mention counts almost as "
            "much as many."
        ),
    )
    bm25_b: float = Field(
        0.75,
        ge=0,
        le=1,
        description=(
            "What: BM25 length normalisation. Typical: 0.5-0.9. Effect: higher penalises "
            "long chunks more; 0 ignores length."
        ),
    )

    @model_validator(mode="after")
    def _candidates_cover_top_k(self) -> RetrievalConfig:
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        return self


class RerankConfig(_Section):
    enabled: bool = Field(
        False,
        description=(
            "What: rescore the first-stage pool (retrieval.candidate_k chunks) with a "
            "cross-encoder and keep the best retrieval.top_k. Effect: usually raises "
            "Recall@k/evidence F1 over plain retrieval; costs GPU time per question. "
            "Does not change the index."
        ),
    )
    model_name: str = Field(
        "BAAI/bge-reranker-v2-m3",
        description=(
            "What: sentence-transformers CrossEncoder model; 'lexical' = fake that scores "
            "shared words, for tests/CI. Effect: stronger models rank better; bigger ones "
            "cost VRAM and latency."
        ),
    )
    batch_size: int = Field(
        16,
        ge=1,
        description=(
            "What: (query, chunk) pairs scored per GPU batch. Typical: 8-64. "
            "Effect: higher = faster until VRAM runs out; no effect on metrics."
        ),
    )
    max_length: int = Field(
        512,
        ge=32,
        description=(
            "What: max tokens of a (query, chunk) pair; the rest is truncated. "
            "Typical: 512-1024. Effect: should cover chunk_size * 1.5 tokens plus the query "
            "or chunk endings are ignored; higher costs memory."
        ),
    )
    device: str = Field(
        "cuda",
        description=(
            "What: torch device ('cuda' or 'cpu'). Effect: 'cpu' works without a GPU but "
            "is far slower; no effect on metrics."
        ),
    )


class QueryConfig(_Section):
    transform: Literal["none", "multi_query", "hyde"] = Field(
        "none",
        description=(
            "What: how the question is rewritten before retrieval. 'none' searches with the "
            "question itself; 'multi_query' asks the LLM for n_queries paraphrases and fuses "
            "their results (RRF); 'hyde' asks the LLM for a hypothetical answer paragraph "
            "that the dense search uses (BM25 keeps the plain question). Effect: can raise "
            "recall on vocabulary mismatch; costs one extra LLM call per question (latency "
            "and tokens are counted in the run); on failure the question itself is used and "
            "query_error_rate rises."
        ),
    )
    n_queries: int = Field(
        3,
        ge=1,
        description=(
            "What: paraphrases kept for multi_query (extra ones are dropped). Typical: 2-5. "
            "Effect: more phrasings widen recall but add retrieval work and can dilute the "
            "fused ranking; the LLM call count does not change."
        ),
    )
    include_original: bool = Field(
        True,
        description=(
            "What: keep the original question among the search texts (multi_query: in both "
            "channels; hyde: in the dense channel next to the hypothetical paragraph). "
            "Effect: True is the safe choice, False tests whether the rewrite alone is enough."
        ),
    )
    multi_query_prompt: Path = Field(
        Path("prompts/multi_query.md"),
        description=(
            "What: prompt file with {{question}}, {{title}} and {{n}} placeholders. "
            "Effect: wording decides how varied and paper-specific the paraphrases are."
        ),
    )
    hyde_prompt: Path = Field(
        Path("prompts/hyde.md"),
        description=(
            "What: prompt file with {{question}} and {{title}} placeholders. Effect: wording "
            "decides how paper-like the hypothetical paragraph is (and so how close it lands "
            "to the real passages in embedding space)."
        ),
    )


class LLMConfig(_Section):
    base_url: str = Field(
        "http://localhost:11434/v1",
        description="What: OpenAI-compatible endpoint (Ollama now, vLLM later). Effect: none.",
    )
    api_key: str = Field(
        "ollama",
        description="What: key sent to the endpoint; local servers ignore it. Effect: none.",
    )
    model: str = Field(
        "qwen3:8b",
        description=(
            "What: model name as the server knows it. Effect: the biggest single lever on "
            "answer F1, latency and VRAM; compare candidates with the model_* configs."
        ),
    )
    temperature: float = Field(
        0.0,
        ge=0,
        le=2,
        description=(
            "What: sampling randomness. Typical: 0-0.3 for QA. Effect: 0 makes runs "
            "repeatable; higher adds variety and run-to-run noise in metrics."
        ),
    )
    max_tokens: int = Field(
        512,
        ge=16,
        description=(
            "What: max tokens the model may generate. Typical: 256-1024. Effect: too low "
            "truncates the JSON (error_rate rises); higher only matters if the model rambles."
        ),
    )
    context_window: int = Field(
        16384,
        ge=1024,
        description=(
            "What: context length the server runs with (Ollama: OLLAMA_CONTEXT_LENGTH). Only "
            "used for a warning. Typical: 8192-32768. Effect: prompts longer than the real "
            "window are silently cut by Ollama and scores collapse; the report warns at 95%."
        ),
    )
    timeout_s: float = Field(
        120.0,
        gt=0,
        description=(
            "What: seconds before one request is abandoned. Typical: 60-300. Effect: too low "
            "turns slow answers into errors; too high hides a stuck server."
        ),
    )
    max_retries: int = Field(
        3,
        ge=1,
        description=(
            "What: attempts per request on connection/timeout/5xx errors. Typical: 2-5. "
            "Effect: higher survives a flaky server but fails slower; bad answers are not retried."
        ),
    )
    retry_backoff_s: float = Field(
        1.0,
        gt=0,
        description="What: base wait between retries, doubled each time. Typical: 0.5-2. Effect: none.",
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What: extra JSON fields sent with every request (server-specific switches, e.g. "
            "turning off a model's thinking mode: {'reasoning_effort': 'none'} on Ollama's "
            "OpenAI endpoint, which ignores 'think'). Effect: qwen3 with thinking on used "
            "~280 extra completion tokens for a one-line JSON answer (~50x)."
        ),
    )


class GenerationConfig(_Section):
    prompt_template: Path = Field(
        Path("prompts/baseline_answer.md"),
        description=(
            "What: prompt file with {{title}}, {{context}} and {{question}} placeholders. "
            "Effect: wording changes how often the model refuses, cites and answers "
            "extractively; version prompts like code."
        ),
    )
    cite_unit: Literal["chunk", "paragraph"] = Field(
        "chunk",
        description=(
            "What: what the model cites: 'chunk' numbers whole chunks, 'paragraph' numbers the "
            "paragraph pieces inside each chunk. Effect: 'paragraph' removes the chunk-width "
            "limit on Evidence-F1 (the ceiling is bounded by retrieval only), but the prompt "
            "holds many more numbers and the model must pick among them; the index must be "
            "built with segments (rebuilt automatically)."
        ),
    )


class JudgeConfig(_Section):
    model: str = Field(
        "qwen3:14b",
        description=(
            "What: Ollama model that grades answers as correct/incorrect (`paperlens judge`). "
            "Effect: pick a different model family than the answering model (llm.model), "
            "because models tend to favour their own answers; a weak judge disagrees with "
            "near-exact answers (judge.agreement_on_high_f1 drops)."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=lambda: {"reasoning_effort": "none"},
        description=(
            "What: extra JSON fields sent with every judge request (same meaning as "
            "llm.extra_body). Effect: the default turns off qwen3 thinking, which would "
            "multiply judge time for a one-sentence verdict."
        ),
    )
    prompt_template: Path = Field(
        Path("prompts/judge.md"),
        description=(
            "What: judge prompt with {{question}}, {{gold}} and {{prediction}} placeholders. "
            "Effect: wording changes how strict the judge is; its hash is stored with every "
            "verdict, so editing it re-judges all answers."
        ),
    )


class CondenseConfig(_Section):
    enabled: bool = Field(
        True,
        description=(
            "What: rewrite a follow-up into a standalone question using the chat history "
            "before routing and retrieval (M2 agent). Effect: off = every turn is searched as "
            "typed, so 'and how big is it?' finds nothing; on = one extra LLM call per turn "
            "after the first."
        ),
    )
    prompt_template: Path = Field(
        Path("prompts/condense.md"),
        description=(
            "What: condense prompt with {{history}} and {{question}} placeholders. "
            "Effect: wording decides how often standalone questions get rewritten anyway "
            "(condense.changed_rate in the report)."
        ),
    )
    triggers: list[str] = Field(
        default_factory=lambda: [
            r"\bit\b",
            r"\bits\b",
            r"\bthem\b",
            r"\bthat\b",
            r"\bthose\b",
            r"\bthere\b",
            r"\bsame\b",
            r"\b(former|latter)\b",
            r"^\s*(and|what about|how about)\b",
        ],
        description=(
            "What: regexes (case-insensitive) that mark a message as a follow-up; only then is "
            "the condense LLM called. 'they'/'their' are left out on purpose: in papers they "
            "mean the authors. Effect: a 14B model rewrites most standalone questions anyway "
            "(adds titles, fixes typos), so the gate keeps condense.changed_rate low; a missing "
            "pattern leaves that kind of follow-up unresolved."
        ),
    )
    history_turns: int = Field(
        3,
        ge=1,
        description=(
            "What: how many previous turns (question + answer) the condense prompt sees. "
            "Typical: 2-5. Effect: more = resolves older references but longer prompts and "
            "more chances to drag in the wrong topic."
        ),
    )


class RouteConfig(_Section):
    top_k: int = Field(
        10,
        ge=1,
        description=(
            "What: chunks retrieved over the whole corpus to vote on which paper a question "
            "is about. Typical: 5-20. Effect: more = smoother paper shares, but generic "
            "questions spread over more papers."
        ),
    )
    min_lead: float = Field(
        0.2,
        ge=0.0,
        le=1.0,
        description=(
            "What: share of the top-voted paper minus the runner-up needed to pick it without "
            "asking (shares move in steps of 1/top_k, so 0.2 = two chunks ahead). Typical: "
            "0.2-0.4 (a plateau in the M2 sweep, `paperlens route-sweep`). Effect: higher = "
            "more clarifying questions and fewer caught unsignaled switches; lower (0.0-0.1) "
            "= generic questions pull the agent off the active paper."
        ),
    )
    title_match_min: float = Field(
        90.0,
        ge=0.0,
        le=100.0,
        description=(
            "What: rapidfuzz partial_ratio needed to say a question names a paper (full or "
            "short title). Typical: 85-95. Effect: lower = catches sloppier mentions but may "
            "switch on a title word used in passing."
        ),
    )


class ClarifyConfig(_Section):
    max_candidates: int = Field(
        3,
        ge=1,
        description=(
            "What: candidate paper titles listed when the agent asks 'which paper?'. "
            "Typical: 2-5. Effect: none on scores with the simulated user; for a real user "
            "more = longer question, fewer misses."
        ),
    )


class AgentConfig(_Section):
    condense: CondenseConfig = Field(default_factory=CondenseConfig)
    route: RouteConfig = Field(default_factory=RouteConfig)
    clarify: ClarifyConfig = Field(default_factory=ClarifyConfig)
    tracing: bool = Field(
        True,
        description=(
            "What: record every agent conversation as an MLflow trace (one span per graph "
            "node). Effect: none on scores; a little slower, and traces appear in `mlflow ui`."
        ),
    )


class ConversationsConfig(_Section):
    papers_per_conversation: int = Field(
        3,
        ge=1,
        description=(
            "What: papers mixed into one synthetic conversation. Typical: 2-4. Effect: more = "
            "more paper switches per conversation, harder routing."
        ),
    )
    return_prob: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "What: chance that a paper with 2+ questions has its last question moved after a "
            "later paper (a return to an earlier paper). Typical: 0.3-0.7. Effect: more "
            "returns = more switches to a paper the agent saw before."
        ),
    )
    signal_prob: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description=(
            "What: chance that a first or switching turn names its paper (via a signal "
            "template). Typical: 0.3-0.7. Effect: higher = easier routing; the rest are "
            "unsignaled switches."
        ),
    )
    seed: int = Field(
        13,
        description=(
            "What: seed for building conversations. Typical: any fixed integer. Effect: a "
            "different seed gives different conversations; keep it fixed to compare runs."
        ),
    )
    signal_templates: list[str] = Field(
        default_factory=lambda: [
            'In the paper "{title}", {question}',
            "In {short_title}, {question}",
            "About the {short_title} paper: {question}",
        ],
        description=(
            "What: how a turn names its paper; {title}, {short_title} (title before ':' or "
            "first 4 words) and {question}. Effect: full titles are easy to catch, short ones "
            "test real name matching (route.signaled_catch.t<i> per template)."
        ),
    )
    path: Path = Field(
        Path("evals/conversations/m2.jsonl"),
        description="What: where the built conversations are written. Effect: none.",
    )


class EvalConfig(_Section):
    experiment_name: str = Field("m1a", description="What: MLflow experiment name. Effect: none.")
    system: Literal["single_paper", "corpus_rag", "agent"] = Field(
        "single_paper",
        description=(
            "What: what is evaluated. 'single_paper' = each question with its paper (M1); "
            "'corpus_rag' = conversations, every turn searched over all papers with no "
            "history; 'agent' = conversations through the M2 LangGraph agent. Effect: the "
            "two conversation systems add route.* metrics and need no gold paper."
        ),
    )
    run_name: str = Field(
        "baseline",
        description=(
            "What: unique run name; results go to results_dir/run_name. "
            "Effect: none on metrics; reusing a name overwrites that run's files."
        ),
    )
    reference_run: str | None = Field(
        "baseline",
        description=(
            "What: run the report compares against (null = none). "
            "Effect: none on metrics; changes the deltas shown in report.md."
        ),
    )
    results_dir: Path = Field(
        Path("evals/results"), description="What: where run folders are written. Effect: none."
    )
    mlflow_tracking_uri: str = Field(
        "sqlite:///mlflow.db",
        description="What: MLflow backend (`uv run mlflow ui --backend-store-uri <this>`). Effect: none.",
    )
    k_values: list[int] = Field(
        default_factory=lambda: [1, 3, 5, 10],
        description=(
            "What: cut-offs for Recall@k and nDCG@k; values above retrieval.top_k are skipped. "
            "Typical: [1, 3, 5, 10]. Effect: none on the system, only on what is reported."
        ),
    )
    worst_examples: int = Field(
        5,
        ge=0,
        description="What: examples listed in the report. Typical: 5-10. Effect: none on metrics.",
    )
    log_every: int = Field(
        25,
        ge=1,
        description="What: progress log interval. Typical: 10-50. Effect: none on metrics.",
    )


class AppConfig(BaseSettings):
    """Root config. Environment variables like PAPERLENS_LLM__BASE_URL fill gaps in YAML."""

    model_config = SettingsConfigDict(
        env_prefix="PAPERLENS_", env_nested_delimiter="__", extra="forbid"
    )

    data: DataConfig = Field(default_factory=DataConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    conversations: ConversationsConfig = Field(default_factory=ConversationsConfig)

    @model_validator(mode="after")
    def _chunks_fit_embedder(self) -> AppConfig:
        needed = self.chunking.chunk_size * TOKENS_PER_WORD_UPPER
        if needed > self.embedding.max_seq_length:
            raise ValueError(
                f"chunking.chunk_size={self.chunking.chunk_size} words needs ~{needed:.0f} "
                f"tokens but embedding.max_seq_length={self.embedding.max_seq_length}; "
                "raise max_seq_length or the embedder silently drops chunk endings"
            )
        return self
