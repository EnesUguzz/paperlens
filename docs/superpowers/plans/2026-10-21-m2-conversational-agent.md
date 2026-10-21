# M2 Conversational Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph agent that answers multi-paper QASPER conversations: it condenses
follow-ups, resolves the paper (title, corpus vote or a clarifying question) and answers
with paragraph citations. It is evaluated against `m1b_best` and a stateless `corpus_rag`.

**Architecture:**
- Pure routing functions (`graph/routing.py`) are wrapped by LangGraph nodes
  (`graph/nodes.py`) and wired in `graph/build.py`.
- Turns persist through a checkpointer. A clarification is a LangGraph `interrupt()` that the
  eval loop resumes with the simulated user's reply.
- The existing runner gains a conversation mode that writes ordinary `EvalRecord`s plus turn
  fields. The existing report, summary and judge keep working.

**Tech Stack:** Python 3.14, LangGraph 1.x, MLflow 3.17 tracing, existing paperlens stack
(Qdrant, bge-m3, Ollama `ministral-3:14b`).

**Spec:** `docs/superpowers/specs/2026-10-21-m2-conversational-agent-design.md`

## Global Constraints

- Local models only: generator/condense `ministral-3:14b`, judge `qwen3:14b`, embedding `bge-m3`.
- Reuse the `m1b_best` index and settings. The M2 configs `extends: m1b_best.yaml` and change
  no chunking field.
- Every new config field needs `What:`/`Effect:` (+`Typical:` if numeric). The test
  `test_docs_contract` enforces this.
- Every new module docstring has `Purpose:`/`Flow:`/`Params:`/`Try:`. Every new package has a
  `README.md`.
- No gold leak: neither the agent nor `corpus_rag` reads `Question.paper_id` or the gold title.
  The only exception is the simulated user reply.
- Commits use the user's git identity, with no Co-Authored-By or "Generated with" lines. No
  push without the user's go-ahead.
- Spec defaults:
  - `route.top_k` 10, `route.min_lead` 0.3, `title_match_min` 90
  - `clarify.max_candidates` 3, one clarification per turn
  - `papers_per_conversation` 3, `return_prob` 0.5, `signal_prob` 0.5, seed 13
  - three signal templates: `In the paper "{title}", {question}`, `In {short_title}, {question}`,
    `About the {short_title} paper: {question}`
  - `short_title` = the text before ":" in the title, otherwise its first 4 words

## Review Focus

1. A question that names a paper whose short title is a common word (e.g. "In BERT, ...")
   must not match another title that merely contains it. Pin with a test where two titles
   share a word.
2. A clarification reply that matches no title must refuse and clear the active paper, not
   crash or loop. Pin in the graph test.
3. LLM failure in `condense` must fall back to the raw question. Pin in the node test.
4. A conversation whose turn throws (LLM error in generate) must still yield a record and
   continue the next turn. The existing `BaselinePipeline` error path covers generate; pin
   one runner test.
5. Re-running `build_conversations` with the same seed must give byte-identical JSONL, and
   every sampled question must appear exactly once. Pin in the conversations test.

---

### Task 1: Config sections and dependency

**Files:**
- Modify: `pyproject.toml` (add `langgraph>=1.0`)
- Modify: `src/paperlens/config/models.py` (new sections, add `eval.system`)
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `ConversationsConfig(papers_per_conversation=3, return_prob=0.5, signal_prob=0.5, seed=13, signal_templates: list[str], path=Path("evals/conversations/m2.jsonl"))`
  - `AgentConfig(condense: CondenseConfig, route: RouteConfig, clarify: ClarifyConfig, tracing: bool=True)`
  - `CondenseConfig(enabled=True, prompt_template=Path("prompts/condense.md"), history_turns=3)`
  - `RouteConfig(top_k=10, min_lead=0.3, title_match_min=90.0)`
  - `ClarifyConfig(max_candidates=3)`
  - `EvalConfig.system: Literal["single_paper","corpus_rag","agent"] = "single_paper"`
  - `AppConfig.agent` and `AppConfig.conversations`

- [ ] **Step 1:** Write `test_m2_defaults` asserting the defaults above and `AppConfig().eval.system == "single_paper"`.
- [ ] **Step 2:** Run `uv run pytest tests/test_config.py -q`. Expected: FAIL (no attribute).
- [ ] **Step 3:** Add the sections with documented fields. Run `uv add langgraph`.
- [ ] **Step 4:** Run `uv run pytest tests/test_config.py tests/test_docs_contract.py -q`. Expected: PASS.
- [ ] **Step 5:** Commit `feat(config): agent and conversations sections for M2`.

### Task 2: Conversation builder

**Files:**
- Create: `src/paperlens/evals/conversations.py`
- Test: `tests/test_conversations.py`

**Interfaces:**
- Produces:
  - `TurnType = Literal["first","same","switch_signaled","switch_unsignaled"]`
  - `Turn(conversation_id: str, turn_index: int, question_id: str, paper_id: str, text: str, turn_type: TurnType, template: int | None)`
    - `text` is what the user types; the original question stays in the Question object.
  - `short_title(title: str) -> str`
  - `build_conversations(questions: Sequence[Question], titles: Mapping[str,str], cfg: ConversationsConfig) -> list[list[Turn]]`
  - `save_conversations(convs, path)` / `load_conversations(path) -> list[list[Turn]]` (one JSON line per turn)

Algorithm (the spec leaves details open; this is the decision):
1. Group the questions by paper in sample order. Shuffle the papers with `random.Random(seed)`.
2. Chunk the papers into conversations of `papers_per_conversation`.
3. Within a conversation, the blocks follow paper order. For each paper with ≥2 questions, with
   `return_prob`, move its last question to the end of the conversation if another block
   follows it.
4. Label turns: index 0 → `first`; same paper as previous → `same`; otherwise switch.
5. On `first` and switch turns, with `signal_prob`, pick a template index with the same RNG
   and render it. A non-first turn becomes `switch_signaled` (first turns stay `first` with
   `template` set). Lower-case the first character of the question.

- [ ] **Step 1:** Write the tests:
  - `test_every_question_once` (250 fake questions over 171 papers, using the real distribution 110×1, 43×2, 18×3): the multiset of question_ids is equal.
  - `test_deterministic`: two builds have equal JSON.
  - `test_turn_types`: a `same` turn has the same paper as the previous one; every `switch_*` turn changes paper.
  - `test_signal_text`: template 0 renders `In the paper "T", which dataset?` for title `T` and question `Which dataset?`.
  - `test_short_title`: `"BERT: Pre-training of deep"` → `"BERT"`; `"A tiny paper about retrieval"` → `"A tiny paper about"`.
  - `test_returns_happen`: with `return_prob=1`, some paper appears in two non-adjacent blocks.
- [ ] **Step 2:** Run `uv run pytest tests/test_conversations.py -q`. Expected: FAIL (import).
- [ ] **Step 3:** Implement the module.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(evals): deterministic multi-paper conversations`.

### Task 3: Routing functions

**Files:**
- Create: `src/paperlens/graph/__init__.py`, `src/paperlens/graph/README.md`, `src/paperlens/graph/routing.py`
- Test: `tests/test_routing.py`

**Interfaces:**
- Produces:
  - `vote(chunks: Sequence[ScoredChunk]) -> list[tuple[str, float]]`: `(paper_id, share)` sorted by share desc, ties broken by first appearance.
  - `match_title(text: str, titles: Mapping[str,str], min_score: float, candidates: Sequence[str] = ()) -> str | None`:
    - Score each title by `max(partial_ratio(full title, text), partial_ratio(short_title, text))`, case-folded.
    - A short-title match only counts if the short title has ≥ 4 characters and appears as a whole word.
    - Candidates are tried first.
    - Return the best paper_id ≥ `min_score`. Return None on a tie between different papers.
  - `Decision = Literal["named","vote","stay","clarify"]`
  - `decide(text: str, active: str | None, votes: list[tuple[str,float]], titles, cfg: RouteConfig) -> tuple[Decision, str | None]`: implements spec 4.3. A `vote` win with an active paper equal to the winner returns `("vote", winner)`.

- [ ] **Step 1:** Write the tests:
  - `test_vote_shares`: 10 chunks (6 A, 3 B, 1 C) → `[("A",.6),("B",.3),("C",.1)]`.
  - `test_named_full_and_short`: a text containing the full title matches; `About the BERT paper:` matches the title `BERT: Pre-training...`.
  - `test_short_word_no_false_match` (Review Focus 1): titles `"Attention: a survey"` and `"Neural attention for QA"` with the text `In Attention, what...` → the first paper, not the second.
  - `test_decide_rules`:
    - lead ≥ .3 → vote
    - lead < .3 with an active paper → stay
    - lead < .3 with no active paper → clarify
    - a named paper wins over the vote
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement with `rapidfuzz.fuzz.partial_ratio`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(graph): deterministic paper routing`.

### Task 4: Generation reuse and corpus_rag

**Files:**
- Modify: `src/paperlens/pipelines/baseline.py`, `src/paperlens/pipelines/factory.py`
- Test: `tests/test_baseline.py`

**Interfaces:**
- Produces:
  - `BaselinePipeline.generate(question: Question, text: str, title: str, retrieved: list[ScoredChunk], usage: Usage = Usage()) -> Answer`. This is the existing post-retrieval body of `run`. `text` is the question shown to the LLM.
  - `BaselinePipeline(..., scope: Literal["paper","corpus"]="paper")`. When `scope == "corpus"`:
    - `retrieve` passes `paper_id=None`.
    - The title is `titles[retrieved[0].chunk.paper_id]` ("" if nothing was retrieved).
    - `plan()` uses the title "".
  - `run(question, plan=None, text=None)`: `text` defaults to `question.question`.
  - `build_pipeline(..., scope="paper")`.

- [ ] **Step 1:** Write the tests:
  - `test_corpus_scope_no_gold_leak`: StubRetriever records `paper_id=None`. The prompt contains the top chunk paper's title, not the gold paper's.
  - `test_generate_matches_run`: same answer for the same chunks.
- [ ] **Step 2:** Run `uv run pytest tests/test_baseline.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement as a refactor. `run` calls `generate`.
- [ ] **Step 4:** Run the full `uv run pytest -q`. Expected: all PASS (no M1 regression).
- [ ] **Step 5:** Commit `refactor(pipelines): reusable generate step and corpus scope`.

### Task 5: Graph nodes and build

**Files:**
- Create: `src/paperlens/graph/state.py`, `src/paperlens/graph/nodes.py`, `src/paperlens/graph/build.py`, `prompts/condense.md`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `vote`, `match_title`, `decide` (Task 3); `BaselinePipeline.generate` (Task 4); `Retriever.retrieve`.
- Produces:
  - `AgentState(TypedDict, total=False)`. Fields:
    - `question: Question` (carries the id; the agent must not read its `paper_id`)
    - `text: str`, `history: list[tuple[str,str]]` (user text, answer text), `standalone: str`
    - `active_paper_id: str | None`, `decision: str`, `candidates: list[tuple[str,float]]`
    - `resolved_paper_id: str | None`, `clarified: bool`, `clarify_message: str`
    - `retrieved: list[ScoredChunk]`, `answer: Answer`, `usage: Usage`, `condense_error: str | None`
  - `AgentDeps(dataclass)`: `retriever`, `llm`, `generator: BaselinePipeline`, `titles`, `cfg: AppConfig`, `condense_template: str`
  - Nodes, each `(state) -> dict`: `condense`, `resolve_paper`, `clarify`, `retrieve`, `generate`, `refuse`
    - `clarify` calls `interrupt({"message": ..., "candidates": [...]})` and then runs `match_title` on the reply itself. This folds the spec's `match_title` node into `clarify` (the graph stays 6 nodes).
  - `build_graph(deps: AgentDeps, checkpointer) -> CompiledStateGraph`
  - Persisted across turns: `history`, `active_paper_id`. The input of each turn is `{"question", "text"}`.

Condense prompt: placeholders `{{history}}` and `{{question}}`. The model returns JSON `{"question": "..."}`. If the question already stands alone it returns it unchanged. It never adds a paper title.

- [ ] **Step 1:** Write the tests (FakeLLM, a stub retriever keyed by `paper_id` filter, `InMemorySaver`):
  - `test_first_turn_confident`: the vote is all A → no interrupt, the answer comes from A, `active_paper_id == "A"`.
  - `test_first_turn_clarify_resume`:
    - The vote is split → the result has `__interrupt__`.
    - `Command(resume="Title of A")` → answered from A, `clarified` true.
  - `test_clarify_no_match_refuses` (Review Focus 2): resume with "nonsense" → `answer.refused` and `active_paper_id` None.
  - `test_stay_on_active`: second turn with a split vote → stays on A, no interrupt.
  - `test_named_switch`: the second turn names B's title → switches to B.
  - `test_condense_fallback` (Review Focus 3): the FakeLLM raises LLMError on condense → `standalone == text`, `condense_error` set.
  - `test_no_gold_paper_id_read`: the Question's `paper_id` is `"GOLD"`, which is not in the titles; the agent still works and never filters by "GOLD".
- [ ] **Step 2:** Run `uv run pytest tests/test_graph.py -q`. Expected: FAIL.
- [ ] **Step 3:** Implement the nodes and the graph, with conditional edges per the spec diagram.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat(graph): LangGraph conversational agent`.

### Task 6: Conversation runner, records and routing metrics

**Files:**
- Modify: `src/paperlens/evals/records.py` (optional turn fields), `src/paperlens/evals/runner.py`, `src/paperlens/evals/rescore.py` (skip non-single_paper runs), `src/paperlens/cli.py` (`rescore` prints the skip)
- Create: `src/paperlens/evals/conversation_runner.py`, `src/paperlens/evals/routing_metrics.py`
- Test: `tests/test_conversation_runner.py`

**Interfaces:**
- Produces:
  - New `EvalRecord` fields, all defaulted: `conversation_id: str | None`, `turn_index: int | None`, `turn_type: str | None`, `shown_question: str | None`, `decision: str | None`, `resolved_paper_id: str | None`, `clarified: bool = False`, `standalone_question: str | None`.
  - `run_conversations(convs, questions_by_id, system, deps_or_pipeline, cfg, k_values) -> list[EvalRecord]`
    - `corpus_rag`: `pipeline.run(q, text=turn.text)` per turn; `resolved_paper_id` = the top chunk's paper.
    - `agent`: one `thread_id` per conversation. On an interrupt, resume once with `turn` title (simulated user, `titles[turn.paper_id]`). Every turn is wrapped in try/except so it always yields a record (Review Focus 4).
    - With `cfg.agent.tracing`: `mlflow.langchain.autolog()` before the loop, and each conversation inside `mlflow.start_span(name=conversation_id)`.
  - `routing_metrics(records) -> dict[str,float]`. Keys:
    - `route.paper_accuracy`, `route.paper_accuracy.<turn_type>`
    - `route.signaled_catch.t<i>` (per template) and `route.signaled_catch`
    - `route.clarify_rate`, `route.wrong_paper_rate`, `route.unsignaled_catch`
    - `answer_f1.<turn_type>`, `answer_f1.no_clarify`, `evidence_f1.no_clarify`
    - `condense.changed_rate`, `llm_calls_mean`
  - `run_experiment`: when `cfg.eval.system != "single_paper"`, build the conversations (and save to `cfg.conversations.path`), choose the pipeline/graph, merge `routing_metrics` into metrics, and keep everything else as before.
  - The report has a "## Routing" table rendered from the `route.*` metrics (`render_report` lists them already as metrics; the existing table is enough).

- [ ] **Step 1:** Write the tests:
  - `test_agent_conversation_records` (2 conversations, fake LLM/retriever): one record per turn, turn fields filled, `routing_metrics` keys present.
  - `test_turn_error_continues`: generate raises → the record has an `error` and the next turn still runs.
  - `test_rescore_skips_conversation_runs`.
- [ ] **Step 2:** Run the tests. Expected: FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run `uv run pytest -q`. Expected: all PASS.
- [ ] **Step 5:** Commit `feat(evals): conversation runner and routing metrics`.

### Task 7: Configs, demo command, real runs

**Files:**
- Create: `experiments/configs/m2_corpus_rag.yaml`, `experiments/configs/m2_agent.yaml` (both `extends: m1b_best.yaml`, `eval.experiment_name: m2`, `reference_run: m1b_best`)
- Create: `evals/conversations/followups_demo.jsonl` (10 conversations hand-written over real validation papers; each turn is `{conversation_id, text, reply?}`)
- Modify: `src/paperlens/cli.py` (`converse --config --file --out`: runs the agent on a conversation file and writes a Markdown transcript showing the decision, standalone question, answer and citations per turn), `Makefile` (`m2` target)

- [ ] **Step 1:** Write `test_converse_cli` (fake deps via an injectable builder): the transcript contains the decision and the answer.
- [ ] **Step 2:** Implement `converse` and the configs. Run `uv run paperlens explain-config --out docs/CONFIG.md`.
- [ ] **Step 3:** Run `uv run paperlens eval -c experiments/configs/m2_corpus_rag.yaml`, then the same for `m2_agent.yaml`. Expected: `evals/results/m2_*/report.md` exist with `route.*` metrics.
- [ ] **Step 4:** Run `uv run paperlens judge --run m2_corpus_rag` and `--run m2_agent`, `uv run paperlens summarize --out evals/results/SUMMARY.md`, then `uv run paperlens converse -c experiments/configs/m2_agent.yaml --file evals/conversations/followups_demo.jsonl --out docs/m2_demo.md`.
- [ ] **Step 5:** Commit `chore: M2 runs, conversations and follow-up demo`.

### Task 8: Docs and notebook

**Files:**
- Create: `docs/decisions/003-m2-agent.md`, `docs/m2_ozet_tr.md`, `notebooks/m2_walkthrough_tr.ipynb`
- Modify: `README.md` (M2 section, diagram, results table, demo excerpt), `src/paperlens/graph/README.md`, `src/paperlens/evals/README.md`

Notebook (Turkish, like M1b's) walks:
1. the conversation file,
2. one conversation turn by turn: condense → votes → decision → retrieved → answer,
3. the graph diagram (`graph.get_graph().draw_mermaid()`),
4. an interrupt/resume by hand,
5. the results tables across the three systems and turn types,
6. how to open the MLflow traces.

- [ ] **Step 1:** Write the docs from the real numbers.
- [ ] **Step 2:** Run the notebook with `uv run --extra ml --group notebook jupyter nbconvert --to notebook --execute --inplace notebooks/m2_walkthrough_tr.ipynb`. Expected: 0 errors.
- [ ] **Step 3:** Run `make lint` and `make test`. Expected: clean, all pass.
- [ ] **Step 4:** Commit `docs: M2 decision record, Turkish summary and walkthrough notebook`.
