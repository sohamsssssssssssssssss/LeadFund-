"""
backtest.py — Evaluate the allocation strategies on REAL outcomes.

We replay the strategies over the scored test leads, where every lead has a
true `Converted` label and an `expected_value` proxy. For each contact budget
we measure TWO things:

  (a) CONVERSION CAPTURE: of all conversions in the data, what fraction did the
      selected leads capture?  conversions_captured / total_conversions
  (b) VALUE CAPTURE:      of the total expected value across all CONVERTING
      leads, what fraction did the selected set capture?
      value_captured / total_value
      where value is realised only when a lead actually converts:
      value_captured = sum( expected_value * Converted ) over selected leads.

We also report efficiency = conversions per lead contacted.

Strategies, in progression order:
  random          — floor
  sort            — prob-only greedy (old naive baseline)
  thompson        — prob-only, uncertainty-aware
  sort_value      — value-aware greedy (prob * value)  <- strong baseline now
  thompson_value  — value-aware, uncertainty-aware

----------------------------------------------------------------------------
HONESTY NOTES
----------------------------------------------------------------------------
1. With well-calibrated probabilities and ONE-SHOT batch selection, the
   uncertainty-aware (thompson) variants are expected to land very close to
   their greedy (sort) counterparts — sometimes losing. That is correct, not a
   bug: sort is optimal when the point estimates are trustworthy.

2. The REAL question this version answers: does VALUE-AWARE allocation capture
   more VALUE than naive probability-sorting? We flag value_sort and
   value_thompson against the old prob-only `sort` on the value metric. If
   value-awareness does NOT help, the output says LOST — numbers are not
   massaged either way.

Compatible with Python 3.11.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from allocation import (
    PROB_COL,
    VALUE_COL,
    allocate_oracle,
    allocate_random,
    allocate_sort,
    allocate_sort_value,
    allocate_thompson,
    allocate_thompson_value,
)

SCORED_PATH = "data/processed/scored_leads.csv"
RESULTS_PATH = "data/processed/backtest_results.csv"
TARGET = "Converted"

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
RANDOM_STATE = 42

# Two rates are a TIE if they differ by less than this (i.e. identical capture).
TIE_EPS = 1e-9

# Strategy display order for the tables. "oracle" is the hindsight ceiling
# (NOT a real strategy — see allocation.allocate_oracle); it appears last.
STRATEGY_ORDER = ["random", "sort", "thompson", "sort_value", "thompson_value", "oracle"]
# Columns for the "% of oracle" table: the real strategies, plus oracle (=100%).
ORACLE_PCT_ORDER = ["random", "sort", "sort_value", "thompson_value", "oracle"]


def load_scored(path: str = SCORED_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (PROB_COL, TARGET):
        if col not in df.columns:
            raise KeyError(f"'{col}' missing from {path}")
    if VALUE_COL not in df.columns:
        raise KeyError(
            f"'{VALUE_COL}' missing from {path}. Run value.py (or re-run "
            f"scoring.py) to add the expected_value column first."
        )
    return df


def _metrics(leads: pd.DataFrame, selected: np.ndarray,
             total_conversions: int, total_value: float) -> dict:
    """Compute capture/value metrics for one selection."""
    chosen = leads.loc[selected]
    n_contacted = len(chosen)
    captured = int(chosen[TARGET].sum())
    # Value is realised only on actual conversions.
    value_captured = float((chosen[VALUE_COL] * chosen[TARGET]).sum())
    return {
        "n_contacted": n_contacted,
        "conversions_captured": captured,
        "total_conversions": total_conversions,
        "capture_rate": captured / total_conversions if total_conversions else 0.0,
        "value_captured": value_captured,
        "total_value": total_value,
        "value_capture_rate": value_captured / total_value if total_value else 0.0,
        "efficiency": captured / n_contacted if n_contacted else 0.0,
    }


def run_backtest(leads: pd.DataFrame,
                 budgets: list[float] = BUDGETS,
                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Run every strategy at every budget; return tidy results."""
    total_conversions = int(leads[TARGET].sum())
    # Total realisable value = expected_value summed over leads that converted.
    total_value = float((leads[VALUE_COL] * leads[TARGET]).sum())
    n_leads = len(leads)
    rows = []

    for budget in budgets:
        selections = {
            "random": allocate_random(leads, budget, random_state=random_state),
            "sort": allocate_sort(leads, budget),
            "thompson": allocate_thompson(leads, budget, random_state=random_state),
            "sort_value": allocate_sort_value(leads, budget),
            "thompson_value": allocate_thompson_value(leads, budget,
                                                      random_state=random_state),
            # Hindsight ceiling — uses real outcomes; not a deployable strategy.
            "oracle": allocate_oracle(leads, budget),
        }
        budget_rows = []
        for strategy, selected in selections.items():
            m = _metrics(leads, selected, total_conversions, total_value)
            m.update({"budget": budget, "strategy": strategy})
            budget_rows.append(m)

        # "% of oracle" = how close each strategy gets to the perfect (oracle)
        # value capture at this budget. Oracle is the best possible, so 100%.
        oracle_value = next(r["value_captured"] for r in budget_rows
                            if r["strategy"] == "oracle")
        for r in budget_rows:
            r["oracle_value_captured"] = oracle_value
            r["pct_of_oracle"] = (r["value_captured"] / oracle_value
                                  if oracle_value else 0.0)
        rows.extend(budget_rows)

    results = pd.DataFrame(rows)
    results.attrs["n_leads"] = n_leads
    results.attrs["total_conversions"] = total_conversions
    results.attrs["total_value"] = total_value
    return results


def _verdict(challenger: float, baseline: float) -> tuple[str, float]:
    """BEAT / TIED / LOST for challenger vs baseline, with gap in pp."""
    delta_pp = (challenger - baseline) * 100
    if abs(challenger - baseline) < TIE_EPS:
        return "TIED", delta_pp
    return ("BEAT", delta_pp) if delta_pp > 0 else ("LOST", delta_pp)


def _print_pivot(results: pd.DataFrame, value_col: str, title: str,
                 order: list[str] = STRATEGY_ORDER) -> pd.DataFrame:
    pivot = results.pivot(index="budget", columns="strategy", values=value_col)
    pivot = pivot[order]
    cols = "".join(f" | {s:>14}" for s in order)
    header = f"{'budget':>7}{cols}"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for budget, row in pivot.iterrows():
        cells = "".join(f" | {row[s]:>13.1%}" for s in order)
        print(f"{budget:>6.0%}{cells}")
    return pivot


def print_report(results: pd.DataFrame) -> None:
    n_leads = results.attrs["n_leads"]
    total_conv = results.attrs["total_conversions"]
    total_value = results.attrs["total_value"]

    print("=" * 96)
    print("ALLOCATION BACKTEST")
    print(f"test set: {n_leads} leads | {total_conv} conversions "
          f"(base rate {total_conv / n_leads:.1%}) | "
          f"total realisable value {total_value:,.1f} (relative units)")
    print("=" * 96)

    # (a) Conversion capture.
    _print_pivot(results, "capture_rate", "(a) % of REAL conversions captured")

    # (b) Value capture. (Rightmost "oracle" column = hindsight ceiling.)
    vpivot = _print_pivot(results, "value_capture_rate",
                          "(b) % of total VALUE captured  (oracle = hindsight ceiling)")

    # (c) % of oracle — how close each REAL strategy gets to perfect at each
    #     budget. Oracle itself is 100% by definition (it IS the ceiling).
    opivot = _print_pivot(
        results, "pct_of_oracle",
        "(c) % of ORACLE captured  (oracle = theoretical best, hindsight only)",
        order=ORACLE_PCT_ORDER,
    )

    # ---- Honest verdicts on the VALUE metric ---------------------------- #
    print("\nValue-aware vs. baselines (VALUE capture, percentage points):")
    print(f"{'budget':>7} | {'val_thompson vs val_sort':>26} | "
          f"{'val_sort vs prob_sort':>24} | {'val_thompson vs prob_sort':>26}")
    print("-" * 92)
    for budget, row in vpivot.iterrows():
        v_t, v_s, p_s = row["thompson_value"], row["sort_value"], row["sort"]
        for_disp = []
        for challenger, baseline in [(v_t, v_s), (v_s, p_s), (v_t, p_s)]:
            verdict, delta = _verdict(challenger, baseline)
            sign = "+" if delta >= 0 else ""
            for_disp.append(f"{verdict:4s} ({sign}{delta:.2f} pp)")
        print(f"{budget:>6.0%} | {for_disp[0]:>26} | {for_disp[1]:>24} | "
              f"{for_disp[2]:>26}")

    # ---- Overall honest summary ----------------------------------------- #
    budgets = list(vpivot.index)
    vs_naive = sum(1 for b in budgets
                   if _verdict(vpivot.loc[b, "sort_value"],
                               vpivot.loc[b, "sort"])[0] == "BEAT")
    t_vs_s = sum(1 for b in budgets
                 if _verdict(vpivot.loc[b, "thompson_value"],
                             vpivot.loc[b, "sort_value"])[0] == "BEAT")
    print("\n" + "-" * 96)
    print(f"SUMMARY (value metric):")
    print(f"  value-aware sort BEAT naive prob-sort at {vs_naive}/{len(budgets)} budgets.")
    print(f"  value-aware thompson BEAT value-aware sort at {t_vs_s}/{len(budgets)} budgets.")
    if vs_naive == 0:
        print("  Value-awareness did NOT increase value capture here — reported straight.")
    else:
        print("  -> Value-aware allocation captures more VALUE than naive prob-sorting.")

    # ---- Headline: how close LeadFund gets to the theoretical maximum ---- #
    print("\nHEADLINE — % of the theoretical maximum (oracle, hindsight) that")
    print("LeadFund (thompson_value) captures at each budget:")
    for budget in opivot.index:
        lf = opivot.loc[budget, "thompson_value"]
        naive = opivot.loc[budget, "sort"]
        print(f"  {budget:>5.0%}: LeadFund = {lf:>6.1%} of perfect   "
              f"(naive sort = {naive:.1%})")
    print("  Reminder: oracle uses real outcomes (perfect hindsight) and is NOT")
    print("  a deployable strategy — it is only the ceiling for comparison.")
    print("=" * 96)


def main() -> pd.DataFrame:
    leads = load_scored()
    results = run_backtest(leads)
    print_report(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved backtest results -> {RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
