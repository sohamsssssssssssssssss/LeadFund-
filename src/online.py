"""
online.py — REAL online feedback loop (segment-level sequential learning).

This turns the static Thompson Sampling allocator into genuine sequential
learning: leads arrive over time in rounds, we allocate a contact budget,
OBSERVE the real `Converted` outcomes of the leads we contacted, and UPDATE our
beliefs so later rounds allocate better. The update is a real Bayesian posterior
update from observed data — not a cosmetic loop.

WHAT IS LEARNED, AND AT WHAT GRANULARITY (read this — it's the honest framing)
------------------------------------------------------------------------------
Each individual lead is contacted at most once, so we can NOT learn a per-lead
rate from repeated trials. Instead we learn at the SEGMENT level: we maintain a
posterior over each Lead-Source segment's calibration (how its real conversion
rate compares to what the model predicted) and update it with the outcomes of
contacted leads in that segment. The model's per-lead probability still ranks
leads WITHIN a segment; the learned posterior corrects how attractive each
segment is relative to what the model originally predicted. So this is "the model
said Olark Chat converts at 24%, but as real outcomes come in we learn it
actually converts ~10% MORE than predicted, so we shift budget toward Olark."

WHAT WE LEARN: a per-segment CALIBRATION RATIO (not the raw rate)
----------------------------------------------------------------
A naive version would put a Beta on each segment's conversion rate and update
alpha += conversions, beta += non-conversions. That is real learning, but it is
CONFOUNDED by selection: we only ever observe the high-scoring leads we chose to
contact, so the learned rate is the (inflated) contacted-rate, not the segment's
true rate — and a blend that divides by the segment average then over-weights
low-probability segments for the wrong reason. It makes the learning-vs-static
test unfair.

So instead we learn each segment's CALIBRATION RATIO R_s = (actual conversions) /
(conversions the MODEL expected), measured on the SAME contacted leads. This is
selection-robust: if the model is well-calibrated on the leads we contact,
R_s = 1 and learning changes nothing; R_s > 1 means the model under-predicts that
segment (contact it more), R_s < 1 means it over-predicts (contact it less).

We use a Gamma-Poisson posterior (the conjugate, ratio-valued analogue of a Beta):

    R_s ~ Gamma(a_s, b_s),   mean = a_s / b_s
    update (learning): a_s += observed_conversions,  b_s += model_expected (sum p_i)

The informed prior is a_s = b_s = k0, i.e. mean 1 ("trust the model's own
probabilities"), with strength k0 expected-conversions — this is how the model's
predictions seed the prior.

HOW WE BLEND it with the per-lead model probability + value
-----------------------------------------------------------
For a lead i in segment s the allocation score each round is:

    score_i = p_i  *  R_s  *  value_i

  - p_i      : the model's per-lead probability (individual signal + ranking).
  - R_s      : a THOMPSON DRAW from the segment's calibration posterior. At init
               R_s ~ mean 1, so score_i ~ p_i * value_i — IDENTICAL to value-aware
               allocation; it only diverges as outcomes reveal miscalibration.
  - value_i  : the value proxy, so we stay value-aware.

THE BASELINE
------------
The non-learning baseline runs the exact same sequential process but draws R_s
from the FROZEN prior (mean 1) every round and never updates it — it Thompson-
samples, it just never learns. Because both policies are centred at R_s = 1, the
comparison is symmetric and fair. The key question: does updating beliefs from
real outcomes capture more cumulative value than never updating? We answer it
straight — including if the answer is "no".

Compatible with Python 3.11.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from allocation import PROB_COL, VALUE_COL, TARGET_COL

SCORED_PATH = "data/processed/scored_leads.csv"
RAW_DIR = "data/raw"
OUT_PATH = "data/processed/learning_curve.csv"

SEGMENT_RAW_COL = "Lead Source"
MIN_SEGMENT_N = 30          # Lead Sources rarer than this (in test) -> "Other"
N_ROUNDS = 10               # leads arrive in 10 ordered batches over "time"
ROUND_BUDGET = 0.20         # contact 20% of each round's leads
CAL_PRIOR_STRENGTH = 5.0    # k0: prior weight (in expected-conversions) on "model is right"
N_SEEDS = 60                # average over seeds for a stable, fair comparison
MIN_PRIOR_MEAN = 0.01       # floor for reported segment model means


# --------------------------------------------------------------------------- #
# Data + segments
# --------------------------------------------------------------------------- #

def load_leads() -> pd.DataFrame:
    """Load scored test leads and attach a bucketed Lead-Source segment."""
    scored = pd.read_csv(SCORED_PATH)
    raw = pd.read_csv(sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))[0])
    raw.columns = [c.strip() for c in raw.columns]

    src = raw.loc[scored["row_index"].to_numpy(), SEGMENT_RAW_COL]
    src.index = scored.index
    src = src.fillna("Unknown").astype(str)

    # Bucket rare sources into "Other" so every segment has enough leads to learn.
    counts = src.value_counts()
    keep = set(counts[counts >= MIN_SEGMENT_N].index)
    scored["segment"] = src.where(src.isin(keep), "Other")
    return scored


def segment_priors(leads: pd.DataFrame) -> dict[str, float]:
    """
    Informed prior mean per segment = the model's AVERAGE predicted probability
    for that segment. Uses model predictions only (no outcomes) — no leakage.
    """
    means = leads.groupby("segment")[PROB_COL].mean()
    return {s: float(max(m, MIN_PRIOR_MEAN)) for s, m in means.items()}


# --------------------------------------------------------------------------- #
# One sequential simulation
# --------------------------------------------------------------------------- #

def run_simulation(leads: pd.DataFrame, priors: dict[str, float],
                   learning: bool, seed: int,
                   n_rounds: int = N_ROUNDS, round_budget: float = ROUND_BUDGET,
                   k0: float = CAL_PRIOR_STRENGTH):
    """
    Run one ordered-rounds simulation.

    Returns
    -------
    per_round_value : np.ndarray, shape (n_rounds,)
        Value captured in each round (value of contacted leads that converted).
    post_ratio : np.ndarray, shape (n_rounds, n_segments)
        Posterior MEAN CALIBRATION RATIO per segment AFTER each round (R_s = a/b).
        This is the engine's belief track: how much higher/lower a segment really
        converts versus what the model predicted (1.0 = perfectly calibrated).
    contacted_conv, contacted_pred : np.ndarray, shape (n_segments,)
        Total observed conversions and total MODEL-EXPECTED conversions (sum of
        p_i) per segment over the run. Their ratio is the realised calibration
        ratio the posterior is estimating.
    segments : list[str]   (column order)
    """
    rng = np.random.default_rng(seed)
    segments = sorted(priors.keys())
    seg_idx = {s: i for i, s in enumerate(segments)}
    S = len(segments)

    # Gamma(a, b) calibration-ratio prior, mean a/b = 1 ("trust the model"),
    # strength k0 expected-conversions. This is how the model's own predictions
    # seed the informed prior.
    a = np.full(S, k0)
    b = np.full(S, k0)
    a0, b0 = a.copy(), b.copy()  # frozen prior the static baseline samples from

    # Pre-extract arrays; map each lead to its segment index.
    p = leads[PROB_COL].to_numpy(dtype=float)
    v = leads[VALUE_COL].to_numpy(dtype=float)
    y = leads[TARGET_COL].to_numpy(dtype=int)
    seg_of_lead = leads["segment"].map(seg_idx).to_numpy()

    # Order leads, then split into n_rounds ordered batches ("arriving over time").
    order = rng.permutation(len(leads))
    batches = np.array_split(order, n_rounds)

    per_round_value = np.zeros(n_rounds)
    post_ratio = np.zeros((n_rounds, S))
    contacted_conv = np.zeros(S)   # observed conversions per segment
    contacted_pred = np.zeros(S)   # model-expected conversions (sum p_i) per segment

    for r, batch in enumerate(batches):
        b_seg = seg_of_lead[batch]

        # THOMPSON DRAW of each segment's calibration ratio from the CURRENT
        # posterior (learning) or the frozen prior (static). Gamma(shape=a,
        # scale=1/b). One sampled ratio per segment this round.
        if learning:
            R = rng.gamma(shape=a, scale=1.0 / b)
        else:
            R = rng.gamma(shape=a0, scale=1.0 / b0)

        # Per-lead score = model prob * sampled calibration ratio * value.
        score = p[batch] * R[b_seg] * v[batch]

        # Contact the top round_budget fraction of this round's leads.
        n_select = max(1, int(round(round_budget * len(batch))))
        top_local = np.argpartition(-score, n_select - 1)[:n_select]
        contacted = batch[top_local]

        # OBSERVE real outcomes and bank value realised on actual conversions.
        out = y[contacted]
        per_round_value[r] = float((v[contacted] * out).sum())

        c_seg = seg_of_lead[contacted]
        conv = np.bincount(c_seg, weights=out, minlength=S)            # observed
        pred = np.bincount(c_seg, weights=p[contacted], minlength=S)  # model-expected
        contacted_conv += conv
        contacted_pred += pred

        # ---- THE LEARNING STEP --------------------------------------------- #
        # Gamma-Poisson update of each contacted segment's calibration posterior
        # with the REAL outcomes: a += observed conversions, b += expected
        # conversions. This is the actual Bayesian update; static skips it.
        if learning:
            a += conv
            b += pred

        post_ratio[r] = a / b

    return per_round_value, post_ratio, contacted_conv, contacted_pred, segments


# --------------------------------------------------------------------------- #
# Compare learning vs static across many seeds
# --------------------------------------------------------------------------- #

def run_experiment(leads: pd.DataFrame, priors: dict[str, float],
                   n_seeds: int = N_SEEDS):
    """Average per-round value over seeds for both policies; track beliefs."""
    segments = sorted(priors.keys())
    S = len(segments)
    learn_vals = np.zeros((n_seeds, N_ROUNDS))
    static_vals = np.zeros((n_seeds, N_ROUNDS))
    learn_post = np.zeros((n_seeds, N_ROUNDS, S))
    conv_tot = np.zeros(S)   # conversions observed by the learning policy
    pred_tot = np.zeros(S)   # model-expected conversions for those contacts

    for s in range(n_seeds):
        # Same seed => same lead arrival order for both policies (fair head-to-head).
        lv, lp, cc, cp, _ = run_simulation(leads, priors, learning=True, seed=s)
        sv, _, _, _, _ = run_simulation(leads, priors, learning=False, seed=s)
        learn_vals[s] = lv
        static_vals[s] = sv
        learn_post[s] = lp
        conv_tot += cc
        pred_tot += cp

    # Realised calibration ratio per segment = observed / model-expected. This is
    # the quantity the posterior is estimating (1.0 = model was perfectly right).
    realised_ratio = np.divide(conv_tot, pred_tot, out=np.ones(S), where=pred_tot > 0)

    return {
        "segments": segments,
        "learn_round": learn_vals.mean(axis=0),
        "static_round": static_vals.mean(axis=0),
        "learn_post_ratio": learn_post.mean(axis=0),  # avg belief (ratio) track
        "realised_ratio": realised_ratio,
    }


# --------------------------------------------------------------------------- #
# Report + save
# --------------------------------------------------------------------------- #

def print_and_save(leads: pd.DataFrame, priors: dict[str, float], exp: dict) -> pd.DataFrame:
    segments = exp["segments"]
    learn_round = exp["learn_round"]
    static_round = exp["static_round"]
    learn_cum = np.cumsum(learn_round)
    static_cum = np.cumsum(static_round)
    total_value = float((leads[VALUE_COL] * leads[TARGET_COL]).sum())

    print("=" * 78)
    print("ONLINE FEEDBACK LOOP — segment-level sequential learning")
    print(f"{len(leads)} leads | {N_ROUNDS} rounds | {ROUND_BUDGET:.0%} contacted/round "
          f"| segment = Lead Source | {N_SEEDS} seeds averaged")
    print("Learning updates per-segment CALIBRATION-RATIO posteriors from REAL outcomes;")
    print("the static baseline runs the same process with frozen priors (never learns).")
    print("=" * 78)

    print("\nPER-ROUND and CUMULATIVE value captured (relative units):")
    print(f"{'round':>5} | {'learn/rd':>9} | {'static/rd':>9} | "
          f"{'learn cum':>9} | {'static cum':>10} | {'delta cum':>9}")
    print("-" * 70)
    for r in range(N_ROUNDS):
        print(f"{r + 1:>5} | {learn_round[r]:>9.2f} | {static_round[r]:>9.2f} | "
              f"{learn_cum[r]:>9.2f} | {static_cum[r]:>10.2f} | "
              f"{learn_cum[r] - static_cum[r]:>+9.2f}")

    final_delta = learn_cum[-1] - static_cum[-1]
    rel = final_delta / static_cum[-1] * 100 if static_cum[-1] else 0.0
    print("-" * 70)
    print(f"FINAL cumulative value — learning {learn_cum[-1]:.2f} vs "
          f"static {static_cum[-1]:.2f}  (of {total_value:.0f} total realisable)")
    if abs(final_delta) < 1e-9:
        verdict = "DEAD HEAT — learning made no difference here."
    elif final_delta > 0:
        verdict = (f"LEARNING WINS by {final_delta:+.2f} units ({rel:+.1f}%) "
                   f"cumulative.")
    else:
        verdict = (f"LEARNING LOST by {final_delta:+.2f} units ({rel:+.1f}%) — "
                   f"reported straight.")
    print(f"VERDICT: {verdict}")

    # Belief evolution: the engine starts believing every segment is perfectly
    # calibrated (ratio 1.0) and learns each segment's TRUE calibration ratio
    # (observed conversions / model-expected) from real outcomes. ratio > 1 means
    # the model under-predicts the segment (contact it more); < 1 over-predicts.
    realised = exp["realised_ratio"]
    model_avg = priors
    print("\nSEGMENT BELIEFS — calibration ratio R = actual / model-expected.")
    print("Prior = 1.0 (trust the model); learned should move toward realised ratio.")
    print(f"{'segment':>16} | {'n':>5} | {'model p':>7} | {'prior R':>7} | "
          f"{'learned R':>9} | {'realised':>8} | moved?")
    print("-" * 80)
    for j, seg in enumerate(segments):
        n = int((leads['segment'] == seg).sum())
        learned = exp["learn_post_ratio"][-1, j]
        target = float(realised[j])                  # what the posterior estimates
        moved = "toward" if abs(learned - target) < abs(1.0 - target) else "away"
        print(f"{seg:>16} | {n:>5} | {model_avg[seg]:>7.3f} | {1.0:>7.3f} | "
              f"{learned:>9.3f} | {target:>8.3f} | {moved}")

    # ---- Save tidy per-round CSV (+ per-segment learned posterior track) --- #
    df = pd.DataFrame({
        "round": np.arange(1, N_ROUNDS + 1),
        "learning_value": learn_round,
        "static_value": static_round,
        "learning_cumulative": learn_cum,
        "static_cumulative": static_cum,
        "learning_cum_pct": learn_cum / total_value,
        "static_cum_pct": static_cum / total_value,
        "delta_cumulative": learn_cum - static_cum,
    })
    for j, seg in enumerate(segments):
        df[f"ratio__{seg}"] = exp["learn_post_ratio"][:, j]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved learning curve -> {OUT_PATH}")
    print("=" * 78)
    return df


def main() -> pd.DataFrame:
    leads = load_leads()
    priors = segment_priors(leads)
    exp = run_experiment(leads, priors)
    return print_and_save(leads, priors, exp)


if __name__ == "__main__":
    main()
