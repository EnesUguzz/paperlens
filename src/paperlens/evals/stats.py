"""Paired bootstrap confidence intervals for "is run A really better than run B?".

Purpose: Two runs answer the same questions, so the per-question difference removes the
question-difficulty noise; resampling those differences gives a CI for the mean gain.
A gain whose CI includes 0 is not evidence of improvement at this sample size.
Flow: two {question_id: answer_f1} maps (from records) -> paired_bootstrap() ->
BootstrapResult -> evals/summary.py (Delta column) and README tables.
Params: none on purpose (fixed seed and resample count keep CIs reproducible).
Try: Compare a run against itself (CI is exactly 0), then against a run that only changed
the seed of question sampling: the CI width is the noise floor of a 250-question eval.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

BOOTSTRAP_RESAMPLES = 2000
CONFIDENCE = 0.95
SEED = 0


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    low: float
    high: float
    n: int


def paired_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = SEED,
    confidence: float = CONFIDENCE,
) -> BootstrapResult:
    """CI of mean(a - b) over the questions both runs answered."""
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("the two runs share no questions")
    diffs = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), size=(resamples, len(diffs)))].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return BootstrapResult(float(diffs.mean()), float(low), float(high), len(keys))
