"""
value.py — Assign a per-lead "expected_value" proxy.

WHY THIS EXISTS
---------------
The allocation engine should maximise *value captured*, not just *conversions
captured*. But the X Education dataset has NO deal sizes / revenue — every lead
is, on paper, worth the same. With equal values, expected value = probability
for every lead, so the value-aware strategies collapse back into a plain
probability sort and tie with it. To make value-awareness meaningful (and
testable), we need leads to differ in value.

THE PROXY (a STATED MODELING ASSUMPTION — not real revenue)
-----------------------------------------------------------
We derive a *relative* value from the lead's occupation. This is a defensible
signal: a working professional buying a paid course is plausibly a
higher-value conversion than an unemployed lead or a student, both in ability
to pay and in likelihood of completing/upselling. The numbers below are
RELATIVE WEIGHTS we are choosing, not measured dollars — there are no dollars
in this dataset. They are intentionally simple and easy to audit/change.

(For reference, observed conversion rates back up the ordering qualitatively:
 Working Professional ~92%, Businessman ~63%, Unemployed ~44%, Student ~37%.
 But value here is about worth-per-conversion, which is a separate assumption
 from likelihood-of-conversion — that is exactly why value * probability is
 more informative than probability alone.)

Compatible with Python 3.11.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Column in the raw data that carries the occupation segment.
OCCUPATION_COL = "What is your current occupation"

# STATED ASSUMPTION: relative value-per-conversion by occupation segment.
# These are modeling weights, NOT revenue. Edit freely — the whole point is
# that they are explicit and honest.
VALUE_MAP: dict[str, float] = {
    "Working Professional": 3.0,   # highest ability/intent to pay
    "Businessman": 2.5,
    "Housewife": 1.5,
    "Other": 1.5,
    "Student": 1.0,                # low current income
    "Unemployed": 1.0,             # low current income
}

# Fallback for missing / "Select" / unseen occupations. We use a mid value
# (not the minimum) so that "we don't know" is treated as average, not penalised.
DEFAULT_VALUE = 1.5

# Placeholder strings that really mean "missing" in this dataset.
_MISSING = {"select", "", "nan"}

EXPECTED_VALUE_COL = "expected_value"


def _normalise(occ: object) -> str | float:
    """Trim whitespace; map blanks/"Select" to NaN so they hit DEFAULT_VALUE."""
    if occ is None or (isinstance(occ, float) and np.isnan(occ)):
        return np.nan
    s = str(occ).strip()
    if s.lower() in _MISSING:
        return np.nan
    return s


def expected_value_from_occupation(occupation: pd.Series) -> pd.Series:
    """
    Map an occupation Series to relative expected values. Unknown/missing
    segments get DEFAULT_VALUE. Returns float Series aligned to the input index.
    """
    cleaned = occupation.map(_normalise)
    values = cleaned.map(VALUE_MAP)          # known segments -> weight
    values = values.fillna(DEFAULT_VALUE)    # missing/unseen -> mid value
    return values.astype(float)


def attach_expected_value(scored: pd.DataFrame, raw: pd.DataFrame,
                          index_col: str = "row_index") -> pd.DataFrame:
    """
    Add an `expected_value` column to a scored-leads frame by looking up each
    lead's occupation in the raw data via the original row index.

    `scored[index_col]` holds the original raw-CSV row position, so we align on
    it directly. Returns a copy with the new column.
    """
    out = scored.copy()
    if OCCUPATION_COL not in raw.columns:
        raise KeyError(f"'{OCCUPATION_COL}' not found in raw data.")
    occ = raw.loc[out[index_col].to_numpy(), OCCUPATION_COL]
    occ.index = out.index  # realign to scored's index for assignment
    out[EXPECTED_VALUE_COL] = expected_value_from_occupation(occ)
    return out


# --------------------------------------------------------------------------- #
# CLI: patch an existing scored_leads.csv in place with expected_value
# --------------------------------------------------------------------------- #

def main(scored_path: str = "data/processed/scored_leads.csv",
         raw_dir: str = "data/raw") -> pd.DataFrame:
    import glob
    raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not raw_files:
        raise FileNotFoundError(f"No raw CSV under {raw_dir}/")
    raw = pd.read_csv(raw_files[0])
    raw.columns = [c.strip() for c in raw.columns]

    scored = pd.read_csv(scored_path)
    scored = attach_expected_value(scored, raw)
    scored.to_csv(scored_path, index=False)

    dist = scored[EXPECTED_VALUE_COL].value_counts().sort_index()
    print(f"Added '{EXPECTED_VALUE_COL}' to {scored_path}")
    print("Value distribution (relative weights — a STATED proxy, not revenue):")
    print(dist.to_string())
    return scored


if __name__ == "__main__":
    main()
