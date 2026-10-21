# 003: M2 conversational agent

Date: 2026-10-22 · Status: accepted (validation split; test split not run yet)
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
  1. A paper named in the typed text wins. A full title outranks another paper's short name.
  2. Otherwise, a corpus vote over the top-10 dense chunks picks a paper when
     `share1 - share2 >= min_lead`. The default is 0.2, chosen by the sweep below.
  3. Otherwise, the agent stays on the active paper.
  4. With no active paper, it asks.
- **Clarification is a LangGraph `interrupt()`.** In the eval, a simulated user answers with
  the gold title.
- **condense only runs on follow-ups.** It is called only when the typed text matches
  `agent.condense.triggers`, and a rewrite that adds a paper title is rejected.
- **Every conversation is one MLflow trace**, with one span per node.

## Choosing `min_lead`

Routing does not depend on the answering LLM, because the vote runs on the typed text. So the
whole threshold curve can be replayed in seconds (`paperlens route-sweep`, table in
`evals/results/route_sweep.md`). The replay of 0.3 matches the real 0.3 run exactly.

| min_lead | right paper | asked "which?" | first | same | silent switch caught |
|---|---|---|---|---|---|
| 0.0 | 0.564 | 0.0 % | 0.67 | 0.28 | 0.29 |
| 0.1 | 0.632 | 4.4 % | 0.86 | 0.50 | 0.21 |
| **0.2** | **0.676** | 6.8 % | 0.97 | 0.61 | 0.19 |
| 0.3 | 0.676 | 7.6 % | 0.98 | 0.63 | 0.17 |
| 0.4 | 0.672 | 8.4 % | 1.00 | 0.63 | 0.14 |
| 0.5 | 0.664 | 8.4 % | 1.00 | 0.61 | 0.13 |
| 0.6 | 0.652 | 8.8 % | 1.00 | 0.59 | 0.10 |
| 0.7–0.8 | 0.644 | 9.2 % | 1.00 | 0.59 | 0.07 |
| 0.9–1.0 | 0.632 | 9.2 % | 1.00 | 0.56 | 0.06 |

Named switches are caught at every value (1.00).

**Reading the curve:**
- Shares move in steps of 0.1, so 0.2 means "two chunks ahead".
- Below 0.2, generic questions pull the agent off the active paper.
- Above 0.4, it mainly asks more and misses more unannounced switches.
- 0.2–0.4 is a plateau, and any value in it is defensible. 0.2 is the low end: it is the
  least intrusive (fewest clarifying questions) and catches the most silent switches.
- 0.8 was considered as a "safe" choice and rejected, because it is worse on every measure
  except first turns.

**Real LLM runs at three values:**

| min_lead | answer F1 | evidence F1 | judge |
|---|---|---|---|
| 0.2 | 0.304 | 0.294 | 0.356 |
| 0.3 | 0.290 | 0.293 | 0.344 |
| 0.5 | 0.292 | 0.306 | 0.356 |

These answer-F1 values are not treated as a ranking:
- The paired bootstrap for 0.2 vs 0.3 is +0.014 [+0.004, +0.028]. Most of it comes from the
  17 turns whose routing changed (3.37 of 3.60 summed points), so it is not pure noise.
- But 17 turns is little evidence.
- The bootstrap ignores run-to-run LLM variation: two runs with identical routing differed
  by 0.003.
- 0.2 was picked after looking at several values.

Choosing on validation and then reporting one test-split run keeps the final number honest.
Because the curve is flat, little selection bias should carry over.

## Results (250 questions, 57 conversations, `min_lead` 0.2)

| System | answer F1 | evidence F1 | judge | right paper | clarify | wrong paper (unasked) | LLM calls / turn | p95 |
|---|---|---|---|---|---|---|---|---|
| m1b_best (gold paper) | 0.440 | 0.491 | 0.568 | 1.000 | – | – | 1.00 | 1.6 s |
| m2_corpus_rag | 0.216 | 0.181 | 0.248 | 0.456 | 0.000 | 0.492 | 1.00 | 1.9 s |
| m2_agent | 0.304 | 0.294 | 0.356 | 0.676 | 0.068 | 0.236 | 1.02 | 1.7 s* |

\* The 0.2 and 0.5 runs shared the GPU with another load for part of the run (from
conversation c041 in the 0.2 run, throughout in the 0.5 run). Their p95 latencies, 8.7 s and
11.2 s, measure that load, not the agent. The p95 shown is from the 0.3 run, which uses the
same code path and has no slow tail. Answers and routing are unaffected.

Paired bootstrap, 95% CI:

| Comparison | answer F1 | evidence F1 |
|---|---|---|
| agent − corpus_rag | +0.088 [+0.052, +0.127] | +0.113 [+0.070, +0.157] |
| agent − m1b_best | −0.136 [−0.185, −0.085] | −0.197 [−0.256, −0.139] |

Paper accuracy by turn type:

| Turn type | n | corpus_rag | agent |
|---|---|---|---|
| `first` | 57 | 0.60 | 0.96 |
| `same` | 54 | 0.26 | 0.61 |
| `switch_signaled` | 67 | 0.69 | 1.00 |
| `switch_unsignaled` | 72 | 0.28 | 0.19 |

Without the 17 clarified turns, the agent's answers score answer F1 0.294 and evidence F1
0.283, so asking is not what carries the result.

Agent decisions per turn type:

| Turn type | Decisions (right paper / wrong paper) |
|---|---|
| `first` | named 32, clarify 17, vote 8 (6 / 2) |
| `same` | stay 37 (24 / 13), vote 16 (8 / 8), named 1 |
| `switch_signaled` | named 67 |
| `switch_unsignaled` | stay 50 (1 / 49), vote 22 (13 / 9) |

## What changed after the first run

The first agent run (`min_lead` 0.3) scored answer F1 0.266. Two problems showed up:
- `condense.changed_rate` was 0.74.
- `route.unsignaled_catch` was 0.07, below the stateless corpus_rag's 0.28.

ministral-3:14b rewrote most standalone questions. It fixed typos, added details, and often
appended the previous paper's title. The vote then ran on that text and stayed on the old
paper. A stricter prompt with examples still changed 30 of 40 questions in a spot check
(`prompts/condense.md`). Three code changes fixed it:

1. The corpus vote runs on the typed text. The condensed text is only used for retrieval
   and the answer.
2. The condense LLM only runs when a referring pattern matches. The defaults are `it`, `its`,
   `them`, `that`, `those`, `there`, `same`, `former`/`latter`, and a leading
   `and`/`what about`. `they` and `their` are excluded because they usually mean the authors.
3. A rewrite that names a paper the typed text does not name is rejected.

With these changes the rewrite rate dropped to 0.012, answer F1 rose to 0.290 (still at 0.3),
and LLM calls per turn went from 1.77 to 1.02. A code review then found that a full title
could tie with another paper's short name in title matching. The fix changed the resolved
paper for 5 of 500 replayed texts and did not change the routing metrics.

## Limitations

- **Silent switches are the main failure.** With an active paper, an unclear question
  means "same paper". Only 19% of unsignaled switches are caught, and the error carries over
  to later `same` turns (13 of 54). No threshold fixes this: the best value in the sweep is
  0.29, at 0.0, and that costs half of the `same` turns.
- **The signal templates are easy.** The short name is either the text before ":" or the
  first 4 words of the title, and it appears verbatim in the question, so a catch rate of
  1.00 for all three templates was expected. Real users abbreviate and paraphrase; that is
  not measured here.
- **The simulated user always knows the title.** That makes clarification free in the eval
  and flatters higher thresholds. `answer_f1.no_clarify` is reported for this reason, but it
  compares a different set of turns at each threshold, so it was not used to choose one.
- **The condense benefit is shown, not measured.** QASPER questions already stand alone, so
  the benefit appears only in the hand-written follow-up demo (`docs/m2_demo.md`).
- **false_refusal_rate is 0.131 (m1b_best: 0.061).** On the wrong paper the model mostly
  refuses, which is the right behaviour, but it is counted as a false refusal against the
  gold answer.
- **Validation only.** The final test-split run waits for the user's go-ahead (see the spec,
  section 8).

## Follow-ups

- Add a harder signal template (a paraphrased or abbreviated name).
- Detect silent switches with something other than the vote. For example, a low similarity
  between the question and the active paper's best chunk.
- Run the test split with `min_lead` 0.2 frozen.
