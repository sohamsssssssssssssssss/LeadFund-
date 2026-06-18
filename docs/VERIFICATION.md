# LeadFund — Technical Verification Report

**Method:** every claim below was checked by reading the file or re-running the code.
Discrepancies with README/comments are flagged. Items described but not running are
flagged **DESCRIBED BUT NOT IMPLEMENTED**.

---

## 1. DATA — verified against `data/raw/Leads X Education.csv`

| Check | Actual value | Source |
|---|---|---|
| Exact data row count | **9,240** | `pd.read_csv(...).shape` |
| Column count | 37 | same |
| Target column name | **`Converted`** | column present = True |
| Binary? | **Yes**, strictly `{0, 1}`, dtype `int64` | `set(unique)=={0,1}` |

**Columns kept after `data_prep.py` runs — 11 features** (live `python -m data_prep`
output, `PrepReport.print_report` in `src/data_prep.py`):
- Numeric (3): `TotalVisits`, `Total Time Spent on Website`, `Page Views Per Visit`
- Categorical (8): `Lead Origin`, `Lead Source`, `Do Not Email`, `Country`,
  `Specialization`, `What is your current occupation`, `City`,
  `A free copy of Mastering The Interview`

**25 columns dropped**, by reason (exactly as printed): leakage (4)
`Tags, Lead Quality, Last Activity, Last Notable Activity`; identifier (2)
`Prospect ID, Lead Number`; >40% null (6) incl. the 4 Asymmetrique columns;
near-constant (13).

**Real class balance:** Class 0: **5,679 (61.5%)** · Class 1: **3,561 (38.5%)**.
Split: **7,392 train / 1,848 test** (stratified 80/20, `random_state=42`).

---

## 2. SCORING — verified by re-running `python -m scoring`

**Models trained** (`src/scoring.py`): `LogisticRegression(max_iter=1000)` (scaled) and
`GradientBoostingClassifier()`. A `CalibratedClassifierCV(isotonic, cv=5)` is fit on GB
for comparison only.

| Model | ROC-AUC | Brier | Calibration verdict (own check) |
|---|---|---|---|
| Logistic Regression | **0.8483** | 0.1525 | worst bin gap 0.117 → **"MIS-CALIBRATED"** |
| Gradient Boosting (raw) | **0.8629** | 0.1423 | worst bin gap 0.082 → **"looks trustworthy"** |
| GB isotonic-calibrated | — | 0.1415 | (computed, comparison only) |

**Which probabilities feed downstream:** the **raw Gradient Boosting** column
`prob_gradient_boosting` (`allocation.py` `PROB_COL = "prob_gradient_boosting"`).

⚠️ **Discrepancy:** README said "calibrated probabilities." In running code the engine
consumes the **raw** GB probabilities, not the calibrated column. A
`prob_gradient_boosting_calibrated` column **is written to `scored_leads.csv` but never
consumed**. The raw GB was *verified* well-calibrated (gap 0.082), so the claim is
defensible in spirit, but the precise statement is **raw-but-verified-calibrated**
probabilities. LR's own check labels it MIS-CALIBRATED; harmless because LR is not used
downstream.

---

## 3. VALUE MODEL — verified in `src/value.py`

**Mapping as written (lines 44–55):**
```python
VALUE_MAP = {
    "Working Professional": 3.0,
    "Businessman": 2.5,
    "Housewife": 1.5,
    "Other": 1.5,
    "Student": 1.0,
    "Unemployed": 1.0,
}
DEFAULT_VALUE = 1.5   # missing / "Select" / unseen
```

**Test-set distribution (1,848 leads):** 1.0 → 1,137 · 1.5 → 567 · 2.5 → 1 · 3.0 → 143.
By tier: Base (1.0) = 1,137; Mid (1.5) = 567; High (≥2.5) = 144.

⚠️ The **2.5 (Businessman) tier has exactly 1 lead** in the test set — the high-value
signal is effectively "Working Professional (3.0) vs everyone."

---

## 4. ALLOCATION — verified in `src/allocation.py`

**Strategy functions that exist** (and in the `STRATEGIES` registry):
`allocate_random`, `allocate_sort`, `allocate_thompson`, `allocate_sort_value`,
`allocate_thompson_value`.

**`allocate_thompson_value`** — `(leads, budget, n_samples=50, random_state=42, k=20.0)`,
lines 222–231:
```python
alpha = p * k + 1.0
beta = (1.0 - p) * k + 1.0
rng = np.random.default_rng(random_state)
selection_counts = np.zeros(n, dtype=np.int64)
for _ in range(n_samples):
    sampled_value = rng.beta(alpha, beta) * v   # expected value this draw
    top = np.argpartition(-sampled_value, n_select - 1)[:n_select]
    selection_counts[top] += 1
```
Beta prior `alpha = p·k + 1`, `beta = (1−p)·k + 1`; **k = 20.0**, **n_samples = 50**;
ties broken by point expected value `p·v`.

---

## 5. BACKTEST — full real output from `python -m backtest`

Test set: **1,848 leads | 712 conversions (38.5%) | total realisable value 1,006.0 units.**

**(a) % of REAL conversions captured**

| budget | random | sort | thompson | sort_value | thompson_value |
|---|---|---|---|---|---|
| 5% | 4.9% | 12.1% | 12.1% | 12.2% | 12.4% |
| 10% | 10.4% | 23.9% | 23.6% | 22.9% | 22.8% |
| 20% | 18.8% | 43.3% | 44.1% | 43.1% | 43.1% |
| 30% | 29.4% | 62.8% | 62.9% | 60.8% | 60.8% |
| 50% | 51.5% | 85.7% | 85.5% | 85.5% | 85.3% |

**(b) % of total VALUE captured**

| budget | random | sort | thompson | sort_value | thompson_value |
|---|---|---|---|---|---|
| 5% | 5.4% | **22.0%** | 21.4% | 25.9% | **26.2%** |
| 10% | 10.2% | 38.5% | 38.1% | 41.1% | 41.3% |
| 20% | 18.2% | 54.8% | 55.4% | 55.9% | 56.2% |
| 30% | 26.8% | 69.1% | 69.3% | 69.1% | 69.1% |
| 50% | 52.8% | 88.2% | 88.1% | 88.2% | 87.9% |

**5% value-capture confirmed:** naive `sort` = 22.0%, `thompson_value` = 26.2%
(gap **+4.27pp**). At 30% it is **TIED (+0.00pp)**; at 50% `thompson_value`
**LOST (−0.30pp)**. Reported, not hidden.

---

## 6. DASHBOARD — data flow verified

Numbers come from a **generated JSON, traceable to CSV — not hardcoded**:
- `dashboard/src/App.jsx` line 2: `import data from "./data/backtest.json";`
- `src/export_dashboard.py` reads `backtest_results.csv` + `scored_leads.csv` → writes
  `dashboard/src/data/backtest.json`, reusing the real `allocate_thompson_value`.
- Components read from `props.data` via `lookup()`.

Hardcoded-number scan of components: the only match for `26.2 / 22.0` is a **code
comment** in `Race.jsx` explaining the redesign — not a rendered value.

---

## 7. WHAT IS NOT BUILT

| Component | Status |
|---|---|
| Chat / conversation layer | **DESCRIBED BUT NOT IMPLEMENTED.** Only text in `About.jsx`. |
| Live API / backend server | **Does not exist.** No flask/fastapi/uvicorn/express. Offline batch + static frontend. |
| Real-time ingestion / streaming | **Does not exist.** No kafka/websocket/stream/cron. |
| Feedback / online-learning loop | **DESCRIBED BUT NOT IMPLEMENTED as a learning loop.** `alpha`/`beta` derived fresh from the static predicted `p` each call (`allocation.py` 222–223); no posterior update from observed conversions. Thompson Sampling here is **one-shot batch with a static prior**, not sequential/online learning. |
| Automated tests | **None.** No `tests/`, no `test_*.py`/`*.spec` (only inside `node_modules`). |
| Monitoring / logging | **None.** `print()` to stdout only. |

---

## Summary of discrepancies

1. **"Calibrated probabilities":** engine consumes **raw** `prob_gradient_boosting`; the
   calibrated column is computed but **unused**. Raw GB verified well-calibrated (gap
   0.082), so defensible, but wording is imprecise.
2. **Thompson "learning loop":** Beta prior is static (from `p`); **no online update from
   outcomes**. Batch sampling with a fixed prior, not sequential learning.
3. **High-value tier is 1 lead** (Businessman) in the test set; effective value signal is
   binary (Working Professional vs rest).
4. Everything else (row count, target, AUC ~0.86, 5% value 26.2% vs 22.0%,
   dashboard-from-JSON) **matches the code's real output.**
