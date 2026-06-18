"""
allocation.py — Lead-allocation strategies for LeadFund.

Given a fixed contact budget (we can only afford to call/email a fraction of
leads), which leads do we pick? Each strategy takes the scored-leads DataFrame
and a `budget` fraction (0.2 = contact the top 20% of leads) and returns the
selected lead indices (labels from `leads.index`).

We rank on `prob_gradient_boosting` — the gradient-boosting conversion
probability — because that model's probabilities are well-calibrated, i.e. a
lead it scores 0.7 really does convert ~70% of the time. The allocation math
below treats those probabilities as real, so calibration is what makes this
trustworthy.

Strategies:
    allocate_random   — random pick (lower-bound baseline)
    allocate_sort     — greedy: take the highest-probability leads (the naive
                        baseline almost everyone builds first)
    allocate_thompson — uncertainty-aware Thompson Sampling

Compatible with Python 3.11 / numpy 1.26.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Column carrying the (calibrated) gradient-boosting conversion probability.
PROB_COL = "prob_gradient_boosting"
# Column carrying the per-lead expected-value proxy (see value.py). If absent,
# every lead is treated as equal value (1.0), which makes the value-aware
# strategies collapse back onto the probability-only ones.
VALUE_COL = "expected_value"


def _n_select(n_leads: int, budget: float) -> int:
    """
    How many leads a `budget` fraction buys. We round to the nearest lead and
    always contact at least one, so every strategy at a given budget selects
    the SAME count — that keeps the backtest an apples-to-apples comparison.
    """
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be in (0, 1], got {budget}")
    return max(1, int(round(budget * n_leads)))


def _probs(leads: pd.DataFrame) -> np.ndarray:
    if PROB_COL not in leads.columns:
        raise KeyError(f"'{PROB_COL}' column not found in leads.")
    return leads[PROB_COL].to_numpy(dtype=float)


def _values(leads: pd.DataFrame) -> np.ndarray:
    """
    Per-lead expected value. Falls back to all-ones if the column is missing,
    so the value-aware strategies degrade gracefully into the prob-only ones
    rather than crashing.
    """
    if VALUE_COL not in leads.columns:
        return np.ones(len(leads), dtype=float)
    return leads[VALUE_COL].to_numpy(dtype=float)


# --------------------------------------------------------------------------- #
# 1) Random — lower-bound baseline
# --------------------------------------------------------------------------- #

def allocate_random(leads: pd.DataFrame, budget: float,
                    random_state: int | None = 42) -> np.ndarray:
    """
    Pick `budget` fraction of leads uniformly at random. This is the floor:
    with no information, you expect to capture roughly `budget` of all
    conversions. Any useful strategy must beat this.
    """
    n = len(leads)
    k = _n_select(n, budget)
    rng = np.random.default_rng(random_state)
    positions = rng.choice(n, size=k, replace=False)
    return leads.index.to_numpy()[positions]


# --------------------------------------------------------------------------- #
# 2) Sort — greedy top-k by probability (naive baseline)
# --------------------------------------------------------------------------- #

def allocate_sort(leads: pd.DataFrame, budget: float) -> np.ndarray:
    """
    Sort by predicted probability (descending) and take the top `budget`
    fraction. This is the obvious, strong baseline: if the probabilities are
    perfectly accurate, this is provably optimal for one-shot batch selection.
    Its blind spot is that it treats every probability as exact truth and
    ignores how *uncertain* each estimate is.
    """
    p = _probs(leads)
    k = _n_select(len(leads), budget)
    # Stable sort so ties resolve deterministically (by original order).
    order = np.argsort(-p, kind="stable")
    positions = order[:k]
    return leads.index.to_numpy()[positions]


# --------------------------------------------------------------------------- #
# 3) Thompson Sampling — uncertainty-aware selection
# --------------------------------------------------------------------------- #

def allocate_thompson(leads: pd.DataFrame, budget: float,
                      n_samples: int = 50, random_state: int | None = 42,
                      k: float = 20.0) -> np.ndarray:
    """
    Thompson Sampling for batch lead selection.

    WHY THIS DIFFERS FROM allocate_sort
    -----------------------------------
    `allocate_sort` treats each predicted probability `p` as exact truth. But
    `p` is an *estimate*; the model is more sure about some leads than others.
    Here we treat each lead's true conversion rate as a random variable with a
    Beta distribution centred near `p`, then SAMPLE from it. On any given draw,
    a lead with a slightly lower point estimate but more uncertainty can sample
    HIGHER than a lead with a marginally better but confident estimate — so it
    gets a chance to be selected. Sort can never do that. Averaged over many
    draws, the leads that are *reliably* near the cut-off rise to the top.

    THE PRIOR (and what `k` does)
    -----------------------------
    We turn the point estimate `p` into a Beta(alpha, beta) using a pseudo-count
    prior of strength `k`:

        alpha = p * k + 1
        beta  = (1 - p) * k + 1

    Read `k` as "how many prior observations is this probability worth":
      * LARGE k  -> alpha/beta are large -> the Beta is tight around `p`
                    -> we strongly trust the point estimate -> behaves like sort.
      * SMALL k  -> the Beta is wide -> we admit we're unsure -> more exploration,
                    lower-`p` leads outrank higher-`p` ones more often.
      * k = 0    -> Beta(1, 1) = Uniform for every lead -> pure random.
    Default k=20 trusts the model but still allows meaningful uncertainty.
    The `+1` keeps both parameters >= 1 (a smooth, well-behaved unimodal prior)
    even at p=0 or p=1.

    SELECTION
    ---------
    For each of `n_samples` draws we sample a rate for every lead, rank, and
    mark the top-`budget` leads. We then return the leads selected MOST OFTEN
    across draws — averaging gives a stable, reproducible choice rather than the
    luck of a single sample. Ties are broken by the point estimate `p`.
    """
    p = _probs(leads)
    n = len(leads)
    n_select = _n_select(n, budget)

    if k < 0:
        raise ValueError(f"prior strength k must be >= 0, got {k}")

    # Beta parameters per lead (vectorised). Shapes: (n,)
    alpha = p * k + 1.0
    beta = (1.0 - p) * k + 1.0

    rng = np.random.default_rng(random_state)
    selection_counts = np.zeros(n, dtype=np.int64)

    for _ in range(n_samples):
        # One sampled conversion rate per lead from its own Beta posterior.
        sampled = rng.beta(alpha, beta)
        # argpartition gets the top-n_select positions in O(n) without a full
        # sort — cheap even when this runs 50x over thousands of leads.
        top = np.argpartition(-sampled, n_select - 1)[:n_select]
        selection_counts[top] += 1

    # Rank by how often each lead was chosen; break ties by point estimate so
    # the result is deterministic. lexsort's LAST key is primary.
    order = np.lexsort((-p, -selection_counts))
    positions = order[:n_select]
    return leads.index.to_numpy()[positions]


# --------------------------------------------------------------------------- #
# 4) Value-aware sort — greedy top-k by EXPECTED VALUE (probability * value)
# --------------------------------------------------------------------------- #

def allocate_sort_value(leads: pd.DataFrame, budget: float) -> np.ndarray:
    """
    Sort by expected value = prob_gradient_boosting * expected_value, descending,
    and take the top `budget` fraction. This is the strong baseline once leads
    differ in value: it prefers a 0.5-prob lead worth 3x over a 0.6-prob lead
    worth 1x (expected value 1.5 vs 0.6), which a prob-only sort would get
    backwards. Optimal for one-shot batch selection IF both prob and value are
    exact.
    """
    score = _probs(leads) * _values(leads)
    k = _n_select(len(leads), budget)
    order = np.argsort(-score, kind="stable")
    positions = order[:k]
    return leads.index.to_numpy()[positions]


# --------------------------------------------------------------------------- #
# 5) Value-aware Thompson — uncertainty on the rate, weighted by value
# --------------------------------------------------------------------------- #

def allocate_thompson_value(leads: pd.DataFrame, budget: float,
                            n_samples: int = 50, random_state: int | None = 42,
                            k: float = 20.0) -> np.ndarray:
    """
    Like allocate_thompson, but ranks by SAMPLED rate * expected_value instead
    of the sampled rate alone. We still draw the conversion rate from each
    lead's Beta(p*k+1, (1-p)*k+1) posterior (uncertainty lives in the rate, not
    the value, which we treat as a known weight), then multiply by value before
    ranking. Averaged over `n_samples` draws; returns the leads selected most
    often. See allocate_thompson for the full explanation of the Beta prior and
    what `k` controls.
    """
    p = _probs(leads)
    v = _values(leads)
    n = len(leads)
    n_select = _n_select(n, budget)

    if k < 0:
        raise ValueError(f"prior strength k must be >= 0, got {k}")

    alpha = p * k + 1.0
    beta = (1.0 - p) * k + 1.0

    rng = np.random.default_rng(random_state)
    selection_counts = np.zeros(n, dtype=np.int64)

    for _ in range(n_samples):
        sampled_value = rng.beta(alpha, beta) * v  # expected value this draw
        top = np.argpartition(-sampled_value, n_select - 1)[:n_select]
        selection_counts[top] += 1

    # Tie-break by point expected value (p*v) so the result is deterministic.
    order = np.lexsort((-(p * v), -selection_counts))
    positions = order[:n_select]
    return leads.index.to_numpy()[positions]


# Registry so the backtest can iterate strategies by name. Ordered to show the
# full progression: floor -> prob-only -> value-aware.
STRATEGIES = {
    "random": allocate_random,
    "sort": allocate_sort,
    "thompson": allocate_thompson,
    "sort_value": allocate_sort_value,
    "thompson_value": allocate_thompson_value,
}
