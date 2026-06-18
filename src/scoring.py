"""
scoring.py — Train two models and produce trustworthy conversion probabilities.

We train:
  1. Logistic Regression — the interpretable baseline. Naturally well-calibrated
     and easy to explain to stakeholders ("this coefficient means...").
  2. Gradient Boosting   — the performance model. Usually higher ROC-AUC, but
     its raw probabilities are often mis-calibrated, which matters here.

WHY CALIBRATION GETS FIRST-CLASS TREATMENT:
    LeadFund's allocation engine doesn't just rank leads — it relies on the
    *probability itself* (e.g. "this lead has a 0.7 chance, that one 0.3").
    If a model says 0.7 but such leads actually convert 0.5 of the time, the
    allocation math is wrong even when the ranking (AUC) looks great. So for
    each model we report the Brier score and a reliability table, and we also
    show what isotonic calibration would do.

Output: data/scored_leads.csv — one row per TEST lead with its identifiers,
true label, and each model's predicted probability.

Compatible with Python 3.11 / scikit-learn 1.5.x.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from data_prep import build_preprocessor, load_raw, prepare, get_splits
from value import OCCUPATION_COL, expected_value_from_occupation

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.5  # for the 0/1 precision/recall summary only


# --------------------------------------------------------------------------- #
# Model builders — each model carries its OWN preprocessor so imputers/encoders
# are fit on the training fold only (no leakage; see data_prep docstring).
# --------------------------------------------------------------------------- #

def build_logistic(numeric_cols, categorical_cols) -> Pipeline:
    # scale_numeric=True: logistic regression is scale-sensitive.
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=True)
    clf = LogisticRegression(max_iter=1000, class_weight=None, random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def build_gradient_boosting(numeric_cols, categorical_cols) -> Pipeline:
    # Trees are scale-invariant, so no standardisation needed.
    pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=False)
    clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(name: str, y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Print ROC-AUC, precision/recall, and a calibration check for one model."""
    preds = (proba >= DECISION_THRESHOLD).astype(int)

    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)  # area under PR curve
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    brier = brier_score_loss(y_true, proba)  # lower = better-calibrated

    print(f"\n{'#' * 64}\n# {name}\n{'#' * 64}")
    print(f"ROC-AUC                 : {auc:.4f}")
    print(f"Average precision (PR)  : {ap:.4f}")
    print(f"Precision @ {DECISION_THRESHOLD:<3}        : {prec:.4f}")
    print(f"Recall    @ {DECISION_THRESHOLD:<3}        : {rec:.4f}")
    print(f"F1        @ {DECISION_THRESHOLD:<3}        : {f1:.4f}")
    print(f"Brier score (calib.)    : {brier:.4f}   (0 = perfect, lower better)")

    print("\nClassification report:")
    print(classification_report(y_true, preds, digits=3, zero_division=0))

    # ---- Calibration / reliability table -------------------------------- #
    # We bin the predicted probabilities and compare the average predicted
    # probability in each bin to the ACTUAL conversion fraction in that bin.
    # A trustworthy model has predicted ≈ actual on every row.
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    print("Calibration (10 quantile bins): predicted vs. actual conversion")
    print(f"  {'pred prob':>10} | {'actual':>8} | gap")
    max_gap = 0.0
    for mp, fp in zip(mean_pred, frac_pos):
        gap = fp - mp
        max_gap = max(max_gap, abs(gap))
        flag = "  <-- off" if abs(gap) > 0.10 else ""
        print(f"  {mp:>10.3f} | {fp:>8.3f} | {gap:+.3f}{flag}")
    verdict = ("looks trustworthy" if max_gap <= 0.10
               else "MIS-CALIBRATED — consider calibrating before allocation")
    print(f"  worst bin gap = {max_gap:.3f}  ->  {verdict}")

    return {"name": name, "roc_auc": auc, "avg_precision": ap,
            "precision": prec, "recall": rec, "f1": f1,
            "brier": brier, "max_calib_gap": max_gap}


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def main(data_dir: str = "data") -> pd.DataFrame:
    # 1) Load + clean + split (report printed by data_prep).
    raw = load_raw(data_dir)
    X, y, identifiers, report = prepare(raw)
    report.print_report()
    X_train, X_test, y_train, y_test = get_splits(X, y)
    print(f"\nTrain: {len(X_train):,}   Test: {len(X_test):,}   "
          f"(stratified 80/20, random_state={RANDOM_STATE})")

    num, cat = report.numeric_cols, report.categorical_cols

    # 2) Train both models.
    models = {
        "Logistic Regression (baseline)": build_logistic(num, cat),
        "Gradient Boosting (performance)": build_gradient_boosting(num, cat),
    }
    proba = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        # predict_proba[:, 1] = probability of the positive class (Converted=1).
        proba[name] = model.predict_proba(X_test)[:, 1]
        evaluate(name, y_test.to_numpy(), proba[name])

    # 3) Bonus: show whether calibrating Gradient Boosting helps. We refit it
    #    inside an isotonic CalibratedClassifierCV (cross-validated on the
    #    training data only) and report the new Brier score. This tells the
    #    allocation team whether to consume raw or calibrated probabilities.
    gb_cal = CalibratedClassifierCV(
        build_gradient_boosting(num, cat), method="isotonic", cv=5
    )
    gb_cal.fit(X_train, y_train)
    gb_cal_proba = gb_cal.predict_proba(X_test)[:, 1]
    print(f"\nCalibrated Gradient Boosting Brier: "
          f"{brier_score_loss(y_test, gb_cal_proba):.4f} "
          f"(vs raw {brier_score_loss(y_test, proba['Gradient Boosting (performance)']):.4f})")

    # 4) Save test-set leads + probabilities for the allocation engine.
    out = identifiers.loc[X_test.index].copy()
    out["Converted"] = y_test.values
    out["prob_logistic"] = proba["Logistic Regression (baseline)"]
    out["prob_gradient_boosting"] = proba["Gradient Boosting (performance)"]
    out["prob_gradient_boosting_calibrated"] = gb_cal_proba

    # Attach the per-lead expected_value proxy (occupation-derived; see value.py).
    # The allocation engine optimises probability * value, so this column is
    # what lets value-aware strategies differ from a plain probability sort.
    out["expected_value"] = expected_value_from_occupation(
        raw.loc[X_test.index, OCCUPATION_COL]
    ).to_numpy()

    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scored_leads.csv")
    out.to_csv(out_path, index=True, index_label="row_index")
    print(f"\nSaved {len(out):,} scored test leads -> {out_path}")
    return out


if __name__ == "__main__":
    main()
