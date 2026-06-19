"""
export_dashboard.py — Emit the REAL backtest numbers as JSON for the frontend.

This is the single bridge between the Python analysis and the React dashboard.
The dashboard NEVER hardcodes numbers; it imports the JSON this script writes.
Everything here is derived from the same artifacts the backtest used:
    data/processed/backtest_results.csv   (all strategies x budgets, both metrics)
    data/processed/scored_leads.csv       (per-lead prob + expected_value)
    data/raw/<X Education>.csv            (occupation label for the "why" story)

Output: dashboard/src/data/backtest.json

Run (from project root, venv active):
    PYTHONPATH=src python -m export_dashboard
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from allocation import (
    PROB_COL,
    VALUE_COL,
    allocate_sort,
    allocate_sort_value,
    allocate_thompson_value,
)
from value import OCCUPATION_COL, _normalise

BACKTEST_CSV = "data/processed/backtest_results.csv"
SCORED_CSV = "data/processed/scored_leads.csv"
CONFIDENCE_CSV = "data/processed/confidence_intervals.csv"
RAW_DIR = "data/raw"
OUT_JSON = "dashboard/src/data/backtest.json"

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
RANDOM_STATE = 42
QUEUE_TOP_N = 60  # how many of LeadFund's picks to expose for the lead-queue UI


def _occupation_label(raw: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    """Per-lead occupation string (display 'Unknown' for missing/Select)."""
    occ = raw.loc[scored["row_index"].to_numpy(), OCCUPATION_COL]
    occ.index = scored.index
    cleaned = occ.map(_normalise)
    return cleaned.fillna("Unknown")


def build_payload() -> dict:
    # --- backtest summary (the race + results table read this) ------------- #
    results = pd.read_csv(BACKTEST_CSV)

    # nLeads / base rate come straight from scored_leads (no derived arithmetic).
    scored = pd.read_csv(SCORED_CSV)
    n_leads = len(scored)
    total_conversions = int(scored["Converted"].sum())
    total_value = float((scored[VALUE_COL] * scored["Converted"]).sum())

    meta = {
        "nLeads": n_leads,
        "totalConversions": total_conversions,
        "baseRate": total_conversions / n_leads,
        "totalValue": round(total_value, 1),
    }

    # Tidy results list (round for display; keep raw rates as floats). Includes
    # the "oracle" rows (hindsight ceiling) and each strategy's pctOfOracle.
    result_rows = [
        {
            "budget": float(r["budget"]),
            "strategy": str(r["strategy"]),
            "nContacted": int(r["n_contacted"]),
            "conversionsCaptured": int(r["conversions_captured"]),
            "captureRate": float(r["capture_rate"]),
            "valueCaptured": float(r["value_captured"]),
            "valueCaptureRate": float(r["value_capture_rate"]),
            "efficiency": float(r["efficiency"]),
            # oracle ceiling fields (present for every row)
            "oracleValueCaptured": float(r["oracle_value_captured"]),
            "pctOfOracle": float(r["pct_of_oracle"]),
        }
        for _, r in results.iterrows()
    ]

    # --- lead queue: LeadFund's (thompson_value) top picks per budget ------ #
    raw_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    raw = pd.read_csv(raw_files[0])
    raw.columns = [c.strip() for c in raw.columns]

    occ_label = _occupation_label(raw, scored)
    scored = scored.copy()
    scored["occupation"] = occ_label.values
    scored["expected_score"] = scored[PROB_COL] * scored[VALUE_COL]

    lead_queue = {}
    for budget in BUDGETS:
        selected = allocate_thompson_value(scored, budget, random_state=RANDOM_STATE)
        picks = scored.loc[selected].sort_values("expected_score", ascending=False)
        picks = picks.head(QUEUE_TOP_N)
        lead_queue[f"{budget}"] = [
            {
                "leadNumber": int(row["Lead Number"]),
                "prob": round(float(row[PROB_COL]), 4),
                "occupation": str(row["occupation"]),
                "value": round(float(row[VALUE_COL]), 2),
                "expectedScore": round(float(row["expected_score"]), 4),
                "converted": int(row["Converted"]),
            }
            for _, row in picks.iterrows()
        ]

    return {
        "meta": meta,
        "budgets": BUDGETS,
        "results": result_rows,
        "leadQueue": lead_queue,
        # surface a couple of headline gaps so the hero never recomputes them
        "valueGap": {  # value_capture_rate: leadfund(thompson_value) - naive sort
            f"{b}": _gap(results, b) for b in BUDGETS
        },
        # oracle ceiling headline: how close each gets to perfect (hindsight).
        "oracle": {f"{b}": _oracle(results, b) for b in BUDGETS},
        # 95% bootstrap confidence intervals on the headline metrics (if computed).
        "confidence": _confidence(),
    }


def _confidence() -> dict:
    """Load bootstrap CIs (per budget) if present; else empty. Keyed by budget."""
    if not os.path.exists(CONFIDENCE_CSV):
        return {}
    ci = pd.read_csv(CONFIDENCE_CSV)
    out = {}
    for _, r in ci.iterrows():
        out[f"{float(r['budget'])}"] = {
            "nBootstrap": int(r["n_bootstrap"]),
            "valueCapture": {"mean": float(r["tv_value_capture_mean"]),
                             "ciLow": float(r["tv_value_capture_ci_low"]),
                             "ciHigh": float(r["tv_value_capture_ci_high"])},
            "edgePp": {"mean": float(r["edge_pp_mean"]),
                       "ciLow": float(r["edge_pp_ci_low"]),
                       "ciHigh": float(r["edge_pp_ci_high"]),
                       "significant": bool(r["edge_significant"])},
            "pctOfOracle": {"mean": float(r["pct_of_oracle_mean"]),
                            "ciLow": float(r["pct_of_oracle_ci_low"]),
                            "ciHigh": float(r["pct_of_oracle_ci_high"])},
        }
    return out


def _oracle(results: pd.DataFrame, budget: float) -> dict:
    """LeadFund / naive % of the oracle (hindsight) ceiling at a budget."""
    at = results[results["budget"] == budget]
    def pof(strategy):
        return float(at[at["strategy"] == strategy]["pct_of_oracle"].iloc[0])
    return {
        "leadfund": pof("thompson_value"),   # % of perfect
        "naiveSort": pof("sort"),
        "oracleValue": float(at["oracle_value_captured"].iloc[0]),
    }


def _gap(results: pd.DataFrame, budget: float) -> dict:
    """LeadFund vs naive prob-sort value-capture gap at a budget (real numbers)."""
    at = results[results["budget"] == budget]
    lf = float(at[at["strategy"] == "thompson_value"]["value_capture_rate"].iloc[0])
    naive = float(at[at["strategy"] == "sort"]["value_capture_rate"].iloc[0])
    return {"leadfund": lf, "naiveSort": naive, "gapPp": round((lf - naive) * 100, 2)}


def main() -> None:
    payload = build_payload()
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    print(f"  meta: {payload['meta']}")
    print(f"  results rows: {len(payload['results'])}")
    print(f"  lead-queue budgets: {list(payload['leadQueue'].keys())}")
    print(f"  value gap @5%: {payload['valueGap']['0.05']}")


if __name__ == "__main__":
    main()
