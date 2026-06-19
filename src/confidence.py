"""
confidence.py — 95% confidence intervals on the headline backtest results.

THE QUESTION
------------
The backtest reports a single number per budget (e.g. at 5% budget LeadFund
captures +4.27pp more value than naive sort). Is that edge REAL, or a fluke of
the particular 1,848-lead test sample we happened to draw? This module answers
that with confidence intervals.

METHOD: paired percentile bootstrap of the test set (stated clearly)
--------------------------------------------------------------------
We resample the scored test leads WITH REPLACEMENT B times. On each resample we
re-run every allocator and recompute the metrics, then take the 2.5th/97.5th
percentiles of the resampled distribution as the 95% CI.

  - PAIRED: on each resample all strategies see the SAME leads, so we take the
    edge (thompson_value - sort) per resample. This correctly accounts for the
    strong correlation between strategies and gives a tight, honest CI on the
    DIFFERENCE — which is the quantity we actually care about.
  - PERCENTILE bootstrap: non-parametric, makes no normality assumption about
    the edge distribution.

WHY this method (and its honest limitation)
-------------------------------------------
Bootstrapping the test set captures the variability we care about here: "if we
had drawn a different sample of leads from the same population, would the edge
survive?" It is cheap (no model retraining) so we can afford 500+ trials for
stable intervals. LIMITATION, stated plainly: it holds the trained model fixed,
so it does NOT capture variance from retraining on a different train/test split.
It answers "is the edge robust to the lead sample," not "is it robust to a
different model fit." We say so rather than overclaiming.

Compatible with Python 3.11.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from allocation import (
    VALUE_COL,
    TARGET_COL,
    allocate_oracle,
    allocate_sort,
    allocate_thompson_value,
)

SCORED_PATH = "data/processed/scored_leads.csv"
OUT_PATH = "data/processed/confidence_intervals.csv"

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
N_BOOT = 600           # bootstrap resamples (enough for stable 95% CIs)
MASTER_SEED = 42
# Fixed Thompson seed inside each resample, so the edge's variability comes from
# LEAD RESAMPLING (the thing we're testing), not Thompson's own MC noise.
TV_SEED = 0


def _value_captured(resample: pd.DataFrame, selected: np.ndarray) -> float:
    """Value realised on actual conversions among the selected leads."""
    chosen = resample.loc[selected]
    return float((chosen[VALUE_COL] * chosen[TARGET_COL]).sum())


def bootstrap(leads: pd.DataFrame, budgets=BUDGETS,
              n_boot: int = N_BOOT, seed: int = MASTER_SEED) -> dict:
    """
    Run the paired bootstrap. Returns, per budget, arrays of length n_boot for
    each metric: LeadFund value-capture rate, the edge over naive sort (pp), and
    % of oracle captured.
    """
    rng = np.random.default_rng(seed)
    n = len(leads)

    tv_rate = {b: np.empty(n_boot) for b in budgets}   # LeadFund value-capture %
    edge_pp = {b: np.empty(n_boot) for b in budgets}   # thompson_value - sort (pp)
    pct_oracle = {b: np.empty(n_boot) for b in budgets}  # % of perfect

    for t in range(n_boot):
        # Resample leads with replacement; reset index so labels are unique
        # (bootstrap creates duplicate original indices, which would break .loc).
        idx = rng.integers(0, n, size=n)
        rs = leads.iloc[idx].reset_index(drop=True)
        total_value = float((rs[VALUE_COL] * rs[TARGET_COL]).sum())
        if total_value <= 0:  # degenerate resample (no converters); skip safely
            for b in budgets:
                tv_rate[b][t] = edge_pp[b][t] = pct_oracle[b][t] = np.nan
            continue

        for b in budgets:
            sort_v = _value_captured(rs, allocate_sort(rs, b))
            tv_v = _value_captured(rs, allocate_thompson_value(rs, b, random_state=TV_SEED))
            orc_v = _value_captured(rs, allocate_oracle(rs, b))

            tv_rate[b][t] = tv_v / total_value
            edge_pp[b][t] = (tv_v - sort_v) / total_value * 100.0
            pct_oracle[b][t] = tv_v / orc_v if orc_v > 0 else np.nan

    return {"tv_rate": tv_rate, "edge_pp": edge_pp, "pct_oracle": pct_oracle}


def _summarise(values: np.ndarray) -> tuple[float, float, float]:
    """mean, 2.5th pct, 97.5th pct (ignoring any NaNs from degenerate resamples)."""
    v = values[~np.isnan(values)]
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def build_table(boot: dict, budgets=BUDGETS) -> pd.DataFrame:
    rows = []
    for b in budgets:
        tv_mean, tv_lo, tv_hi = _summarise(boot["tv_rate"][b])
        e_mean, e_lo, e_hi = _summarise(boot["edge_pp"][b])
        o_mean, o_lo, o_hi = _summarise(boot["pct_oracle"][b])
        # Significant iff the 95% CI for the edge excludes zero.
        significant = bool(e_lo > 0 or e_hi < 0)
        rows.append({
            "budget": b,
            "n_bootstrap": int(np.sum(~np.isnan(boot["edge_pp"][b]))),
            "tv_value_capture_mean": tv_mean,
            "tv_value_capture_ci_low": tv_lo,
            "tv_value_capture_ci_high": tv_hi,
            "edge_pp_mean": e_mean,
            "edge_pp_ci_low": e_lo,
            "edge_pp_ci_high": e_hi,
            "edge_significant": significant,
            "pct_of_oracle_mean": o_mean,
            "pct_of_oracle_ci_low": o_lo,
            "pct_of_oracle_ci_high": o_hi,
        })
    return pd.DataFrame(rows)


def print_report(table: pd.DataFrame) -> None:
    print("=" * 86)
    print("CONFIDENCE INTERVALS — paired percentile bootstrap of the test set")
    print(f"{int(table['n_bootstrap'].iloc[0])} resamples | 95% CI (2.5–97.5 pct) | "
          f"model held fixed (lead-sampling variability only)")
    print("=" * 86)

    print(f"\n{'budget':>7} | {'LeadFund value %':>22} | {'edge vs naive (pp)':>26} | "
          f"{'% of oracle':>20}")
    print("-" * 86)
    for _, r in table.iterrows():
        tv = f"{r.tv_value_capture_mean:.1%} [{r.tv_value_capture_ci_low:.1%}, {r.tv_value_capture_ci_high:.1%}]"
        ed = f"{r.edge_pp_mean:+.2f} [{r.edge_pp_ci_low:+.2f}, {r.edge_pp_ci_high:+.2f}]"
        oc = f"{r.pct_of_oracle_mean:.1%} [{r.pct_of_oracle_ci_low:.1%}, {r.pct_of_oracle_ci_high:.1%}]"
        print(f"{r.budget:>6.0%} | {tv:>22} | {ed:>26} | {oc:>20}")

    print("\nIS THE EDGE OVER NAIVE SORT STATISTICALLY SIGNIFICANT? (95% CI excludes 0)")
    print("-" * 86)
    for _, r in table.iterrows():
        if r.edge_significant:
            verdict = (f"SIGNIFICANT — edge {r.edge_pp_mean:+.2f}pp, "
                       f"95% CI [{r.edge_pp_ci_low:+.2f}, {r.edge_pp_ci_high:+.2f}] excludes 0")
        else:
            verdict = (f"NOT significant — 95% CI [{r.edge_pp_ci_low:+.2f}, "
                       f"{r.edge_pp_ci_high:+.2f}] crosses 0; not distinguishable from naive sort")
        print(f"  {r.budget:>5.0%}: {verdict}")
    print("=" * 86)


def main() -> pd.DataFrame:
    leads = pd.read_csv(SCORED_PATH)
    boot = bootstrap(leads)
    table = build_table(boot)
    print_report(table)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    table.to_csv(OUT_PATH, index=False)
    print(f"\nSaved -> {OUT_PATH}")
    return table


if __name__ == "__main__":
    main()
