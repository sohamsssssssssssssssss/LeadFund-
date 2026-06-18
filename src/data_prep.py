"""
data_prep.py — Load and clean the X Education lead-scoring dataset.

The goal of this module is to turn the raw, messy CSV into a *leak-free*,
model-ready dataset: a cleaned feature frame `X`, a target `y` (Converted),
a stratified train/test split, and a sklearn preprocessor that does
imputation + encoding.

DESIGN NOTE — why preprocessing lives in a sklearn ColumnTransformer instead
of being baked into X here:
    Imputation (e.g. "median of a numeric column") and one-hot encoding learn
    parameters FROM the data. If we computed those over the whole dataset
    before splitting, information from the test rows would leak into training.
    That would quietly inflate our metrics — and, critically for LeadFund,
    it would make the calibration check (do the predicted probabilities mean
    what they say?) dishonest, because the allocation engine trusts those
    probabilities. So we do the structural cleaning here (which is decided by
    domain knowledge, not learned from values), and hand back a preprocessor
    that scoring.py fits on the TRAINING FOLD ONLY.

Compatible with Python 3.11 / scikit-learn 1.5.x.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuration — kept as module-level constants so they're easy to audit/tune
# --------------------------------------------------------------------------- #

TARGET = "Converted"

# Drop a column if MORE than this fraction of its values are missing.
# WHY 40%: above this, imputation is mostly inventing data — the column is
# more "absence pattern" than signal, and the user asked us to drop these.
NULL_DROP_THRESHOLD = 0.40

# Drop a column if a single value covers AT LEAST this fraction of rows.
# WHY: the X Education data has many yes/no flags that are ~100% "No"
# (Magazine, Newspaper Article, "I agree to pay by cheque", ...). A
# near-constant column carries no discriminative signal and just adds noise
# / one-hot columns. 0.98 keeps genuinely useful imbalanced features while
# dropping the dead ones.
NEAR_CONSTANT_THRESHOLD = 0.98

# Pure identifiers — unique per row, zero predictive value, and a classic
# source of overfitting if accidentally encoded.
ID_COLS = ["Prospect ID", "Lead Number"]

# LEAKAGE columns: information that would NOT exist at lead-intake time.
# These are populated by sales reps AFTER they engage the lead, so training
# on them gives unrealistically good scores that collapse in production.
#   - Tags / Lead Quality      : a rep's subjective notes/rating post-contact
#   - Last Activity / Last Notable Activity : describe engagement that happens
#                                 during the sales process, not at intake
# We keep this as an explicit, documented list so the choice is reviewable.
LEAKAGE_COLS = [
    "Tags",
    "Lead Quality",
    "Last Activity",
    "Last Notable Activity",
]

# In this dataset, un-filled dropdowns are stored as the literal string
# "Select" rather than as blank — a notorious gotcha. It is really a missing
# value, so we convert it to NaN up front; otherwise "Select" would survive
# as a bogus category and dodge the null-threshold filter.
PLACEHOLDER_MISSING = ["Select", "select", ""]

CATEGORICAL_UNKNOWN = "Unknown"  # fill value for missing categoricals


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #

@dataclass
class PrepReport:
    """Lightweight record of what cleaning did, for the data-quality printout."""
    n_rows: int = 0
    n_cols_raw: int = 0
    kept_cols: list[str] = field(default_factory=list)
    dropped: dict[str, list[str]] = field(default_factory=dict)  # reason -> cols
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)

    def print_report(self) -> None:
        print("=" * 64)
        print("DATA-QUALITY REPORT")
        print("=" * 64)
        print(f"Rows kept            : {self.n_rows:,}")
        print(f"Columns in raw CSV   : {self.n_cols_raw}")
        print(f"Feature columns kept : {len(self.kept_cols)} "
              f"({len(self.numeric_cols)} numeric, "
              f"{len(self.categorical_cols)} categorical)")
        total_dropped = sum(len(v) for v in self.dropped.values())
        print(f"Columns dropped      : {total_dropped}")
        for reason, cols in self.dropped.items():
            if cols:
                print(f"  - {reason} ({len(cols)}): {', '.join(cols)}")
        print("-" * 64)
        total = sum(self.class_counts.values()) or 1
        for label, count in sorted(self.class_counts.items()):
            print(f"Class {label}: {count:,} ({count / total:.1%})")
        print("=" * 64)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def find_dataset(data_dir: str = "data") -> str:
    """
    Locate the raw CSV. We auto-detect rather than hard-code a filename so the
    pipeline works whatever the file is called — we pick the first CSV that
    actually contains the `Converted` target column.
    """
    candidates = sorted(
        glob.glob(os.path.join(data_dir, "*.csv"))
        + glob.glob(os.path.join(data_dir, "raw", "*.csv"))
    )
    # Never pick our own output file.
    candidates = [c for c in candidates if os.path.basename(c) != "scored_leads.csv"]
    for path in candidates:
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        if TARGET in header.columns:
            return path
    raise FileNotFoundError(
        f"No CSV containing a '{TARGET}' column found under '{data_dir}/' "
        f"or '{data_dir}/raw/'. Place the X Education lead CSV there.\n"
        f"(Checked: {candidates or 'none found'})"
    )


def load_raw(data_dir: str = "data") -> pd.DataFrame:
    """Load the raw CSV and normalise placeholder-missing values to NaN."""
    path = find_dataset(data_dir)
    df = pd.read_csv(path)
    # Strip stray whitespace in column names (real exports often have it).
    df.columns = [c.strip() for c in df.columns]
    # Convert the "Select" placeholder (and blanks) to real NaN so the
    # downstream null analysis sees them as missing.
    df = df.replace(PLACEHOLDER_MISSING, np.nan)
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #

def _near_constant_cols(df: pd.DataFrame, threshold: float) -> list[str]:
    """Columns where one value dominates >= threshold of NON-null rows."""
    out = []
    for col in df.columns:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:  # all-null column
            out.append(col)
            continue
        if counts.iloc[0] / counts.sum() >= threshold:
            out.append(col)
    return out


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, PrepReport]:
    """
    Clean `df` into (X, y, identifiers, report).

    Returns
    -------
    X : DataFrame
        Cleaned features. Categoricals are still raw strings — encoding is done
        later by the preprocessor (see module docstring on why).
    y : Series
        The binary `Converted` target.
    identifiers : DataFrame
        Prospect ID / Lead Number, kept aside (NOT used as features) so we can
        attach them back to the scored output for joining.
    report : PrepReport
    """
    report = PrepReport(n_rows=len(df), n_cols_raw=df.shape[1])
    report.dropped = {
        "leakage / not known at intake": [],
        "identifier": [],
        ">40% null": [],
        "near-constant": [],
    }

    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' not in data.")

    y = df[TARGET].astype(int)
    report.class_counts = y.value_counts().to_dict()

    # Keep identifiers aside for output joins, then remove from the feature pool.
    identifiers = df[[c for c in ID_COLS if c in df.columns]].copy()

    work = df.drop(columns=[TARGET])

    # 1) Drop identifiers.
    id_present = [c for c in ID_COLS if c in work.columns]
    report.dropped["identifier"] = id_present
    work = work.drop(columns=id_present)

    # 2) Drop leakage columns (only those actually present).
    leak_present = [c for c in LEAKAGE_COLS if c in work.columns]
    report.dropped["leakage / not known at intake"] = leak_present
    work = work.drop(columns=leak_present)

    # 3) Drop columns with > 40% missing.
    null_frac = work.isna().mean()
    high_null = null_frac[null_frac > NULL_DROP_THRESHOLD].index.tolist()
    report.dropped[">40% null"] = high_null
    work = work.drop(columns=high_null)

    # 4) Drop near-constant columns (computed AFTER "Select"->NaN, so dropdowns
    #    that are all-"Select" are already gone via the null filter above).
    near_const = _near_constant_cols(work, NEAR_CONSTANT_THRESHOLD)
    report.dropped["near-constant"] = near_const
    work = work.drop(columns=near_const)

    # Split remaining columns by dtype. Object/category -> categorical;
    # everything numeric -> numeric. (We treat the rare numeric-looking flag as
    # numeric, which is fine for both models.)
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in work.columns if c not in numeric_cols]

    report.kept_cols = work.columns.tolist()
    report.numeric_cols = numeric_cols
    report.categorical_cols = categorical_cols

    return work, y, identifiers, report


# --------------------------------------------------------------------------- #
# Preprocessor + split
# --------------------------------------------------------------------------- #

def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Build the imputation + encoding transformer.

    - Numeric : median imputation. WHY median (not mean): lead-behaviour
      features (visits, time on site) are right-skewed with outliers, where the
      median is the more robust "typical" value.
    - Categorical : impute missing with the explicit "Unknown" category (the
      user's choice — and "missing" is often itself informative for leads),
      then one-hot encode. handle_unknown="ignore" so a category seen only in
      the test fold doesn't crash prediction.

    `scale_numeric=True` adds standardisation — needed for logistic regression
    (gradient-descent / regularisation are scale-sensitive) but pointless for
    tree models, so scoring.py turns it on only for LR.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_UNKNOWN)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])


def get_splits(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.20, random_state: int = 42):
    """
    Stratified 80/20 split. Stratify on y so the ~38% conversion base rate is
    preserved in both folds — important for a fair ROC-AUC and an honest
    calibration check. Fixed random_state makes the whole run reproducible.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# --------------------------------------------------------------------------- #
# Persisting the processed splits
# --------------------------------------------------------------------------- #

def save_processed(X_train, X_test, y_train, y_test,
                   identifiers: pd.DataFrame,
                   out_dir: str = "data/processed") -> None:
    """
    Persist the cleaned train/test splits so later stages (scoring, backtest)
    reuse the EXACT same rows instead of re-deriving them.

    We save the cleaned-but-not-yet-encoded features (categoricals still as
    strings) plus the target as one CSV per fold. Encoding is intentionally
    NOT baked in here — it's fit on the training fold by each model's pipeline
    to stay leak-free (see module docstring). The original row index is kept
    so identifiers / raw rows can be joined back later.
    """
    os.makedirs(out_dir, exist_ok=True)
    for name, Xpart, ypart in [("train", X_train, y_train),
                               ("test", X_test, y_test)]:
        frame = Xpart.copy()
        frame[TARGET] = ypart
        # attach identifiers for traceability (kept out of the feature columns)
        ids = identifiers.loc[Xpart.index]
        for col in ids.columns:
            frame.insert(0, col, ids[col])
        path = os.path.join(out_dir, f"{name}.csv")
        frame.to_csv(path, index=True, index_label="row_index")
        print(f"Saved {name:5s} split -> {path}  ({len(frame):,} rows, "
              f"{Xpart.shape[1]} features)")


# --------------------------------------------------------------------------- #
# CLI entry point — prints the data-quality report
# --------------------------------------------------------------------------- #

def run(data_dir: str = "data", processed_dir: str = "data/processed"):
    """Load + clean + split, print the report, save splits, return everything."""
    raw = load_raw(data_dir)
    X, y, identifiers, report = prepare(raw)
    report.print_report()
    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"\nTrain/test split: {len(X_train):,} train / {len(X_test):,} test "
          f"(stratified 80/20, random_state=42)")
    save_processed(X_train, X_test, y_train, y_test, identifiers, processed_dir)
    return X, y, identifiers, report, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run()
