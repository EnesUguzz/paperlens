# 003: M2 conversational agent

Date: 2026-10-21 · Status: accepted (validation split; test split not run yet)
Design: `docs/superpowers/specs/2026-10-21-m2-conversational-agent-design.md`

## Context

In M1 every question came with its paper (QASPER's `single_paper` setting). M2 asks the same
250 validation questions (`m1b_best` sample, seed 13) inside conversations, without saying
which paper they are about. The agent has to find the paper, remember it across turns,
notice switches and ask when it cannot tell.

## Decision

The agent is a LangGraph graph with fixed nodes:

```
condense -> resolve_paper -> (clarify: interrupt) -> retrieve -> generate | refuse
```

- **Retrieval and generation are `m1b_best`'s.** Section chunks of 200 words, dense top-5
  inside the paper, paragraph citations.
- **Routing is deterministic** (`graph/routing.py`), in this order:
  1. A paper named in the typed text wins.
  2. Otherwise, a corpus vote over the top-10 dense chunks picks a paper when
     `share1 - share2 >= min_lead` (0.3).
  3. Otherwise, the agent stays on the active paper.
  4. With no active paper, it asks.
- **Clarification is a LangGraph `interrupt()`.** In the eval, a simulated user answers with
  the gold title.
- **condense only runs on follow-ups.** It is called only when the typed text matches
  `agent.condense.triggers`, and a rewrite that adds a paper title is rejected.
- **Every conversation is one MLflow trace**, with one span per node.

## Results (250 questions, 57 conversations)

| System | answer F1 | evidence F1 | judge | right paper | clarify | wrong paper (unasked) | LLM calls / turn | p95 |
|---|---|---|---|---|---|---|---|---|
| m1b_best (gold paper) | 0.440 | 0.491 | 0.568 | 1.000 | – | – | 1.00 | 1.6 s |
| m2_corpus_rag | 0.216 | 0.181 | 0.248 | 0.456 | 0.000 | 0.492 | 1.00 | 1.9 s |
| m2_agent | 0.293 | 0.295 | 0.336 | 0.676 | 0.076 | 0.216 | 1.02 | 1.7 s |

Paired bootstrap, 95% CI:

| Comparison | answer F1 | evidence F1 |
|---|---|---|
| agent − corpus_rag | +0.077 [+0.039, +0.113] | +0.114 [+0.070, +0.159] |
| agent − m1b_best | −0.148 [−0.195, −0.100] | −0.196 [−0.253, −0.139] |

Paper accuracy by turn type (corpus_rag → agent):

| Turn type | n | corpus_rag | agent |
|---|---|---|---|
| `first` | 57 | 0.60 | 0.98 |
| `same` | 54 | 0.26 | 0.63 |
| `switch_signaled` | 67 | 0.69 | 1.00 |
| `switch_unsignaled` | 72 | 0.28 | 0.17 |

The agent's answers without the 19 clarified turns score answer F1 0.281 and evidence F1
0.284, so asking is not what carries the result.

Agent decisions per turn type:

| Turn type | Decisions (right paper / wrong paper) |
|---|---|
| `first` | named 32, clarify 19, vote 6 (5 right) |
| `same` | stay 39 (25 / 14), vote 14 (8 / 6), named 1 |
| `switch_signaled` | named 66, vote 1 |
| `switch_unsignaled` | stay 58 (2 / 56), vote 14 (10 / 4) |

## What changed after the first run

The first agent run scored answer F1 0.266. Two problems showed up:
- `condense.changed_rate` was 0.74.
- `route.unsignaled_catch` was 0.07, below the stateless corpus_rag's 0.28.

ministral-3:14b rewrote most standalone questions. It fixed typos, added details and often
appended the previous paper's title. The vote then ran on that text and stayed on the old
paper. A stricter prompt with examples still changed 30 of 40 questions in a spot check
(`prompts/condense.md`). Three code changes fixed it:

1. The corpus vote runs on the typed text. The condensed text is only used for retrieval
   and the answer.
2. The condense LLM only runs when a referring pattern matches. The defaults are `it`, `its`,
   `them`, `that`, `those`, `there`, `same`, `former`/`latter`, and a leading
   `and`/`what about`. `they` and `their` are excluded because they usually mean the authors.
3. A rewrite that names a paper the typed text does not name is rejected.

With these changes the rewrite rate dropped to 0.012 and answer F1 rose to 0.293. LLM calls
per turn went from 1.77 to 1.02.

## Limitations

- **Silent switches are mostly missed, by design.** With an active paper, an unclear
  question means "same paper". 56 of the 72 unsignaled switches stay on the old paper, and
  the error carries over to later `same` turns (14 of 54). Lowering `min_lead` catches more
  switches but also moves more `same` turns away from the right paper.
- **The signal templates are easy.** The short name is either the text before ":" or the
  first 4 words of the title, and it appears verbatim in the question, so a catch rate of
  1.00 for all three templates was expected. Real users abbreviate and paraphrase; that is
  not measured here.
- **The simulated user always knows the title.** That makes clarification free in the eval;
  `answer_f1.no_clarify` is reported for this reason.
- **The condense benefit is shown, not measured.** QASPER questions already stand alone, so
  the benefit appears only in the hand-written follow-up demo (`docs/m2_demo.md`).
- **false_refusal_rate is 0.155 (m1b_best: 0.061).** On the wrong paper the model mostly
  refuses, which is the right behaviour, but it is counted as a false refusal against the
  gold answer.
- **Validation only.** The final test-split run waits for the user's go-ahead (see the spec,
  section 8).

## Follow-ups

- Sweep `agent.route.min_lead` (for example 0.2 / 0.3 / 0.5) and read
  `route.paper_accuracy.same` against `route.unsignaled_catch`.
- Add a harder signal template (a paraphrased or abbreviated name).
- Run the test split once the thresholds are frozen.
