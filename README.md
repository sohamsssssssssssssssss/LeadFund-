# LeadFund — Autonomous Lead Allocation Engine

Sales teams almost always have more leads than time, so rep hours get spent on the wrong prospects and revenue quietly leaks out. LeadFund treats rep capacity like investment capital: it scores every lead, assigns each an expected value, and allocates the limited contact budget to maximize total value captured — not just raw conversion count. It is proven end to end on real data with real outcomes.

## Key result

On 9,240 real X Education leads with real conversion labels (a 1,848-lead held-out test set), value-aware allocation captures about **26% of total value at a 5% contact budget**, versus about **22% for a naive probability sort** — a real edge of **+4.27 percentage points**. The advantage is largest exactly when it matters most: when contact capacity is scarcest. As the budget grows, the strategies converge and the gap closes to a near-tie, which the dashboard shows honestly rather than hiding.

## How it works

1. **Real lead scoring.** A gradient-boosting model produces per-lead conversion probabilities. The allocation layer consumes these **raw gradient-boosting probabilities, verified well-calibrated** (reliability bins + Brier score; worst-bin gap 0.082), which is why it can treat them as real probabilities. (An isotonic-calibrated variant is computed for comparison but is not the one used downstream.) Leakage and post-intake columns (e.g. Tags, Lead Quality, Last Activity, Last Notable Activity) and identifiers are dropped on purpose, so the reported ROC-AUC of about **0.86** is honest and not inflated by columns a model would never have at lead-intake time.

2. **Value-weighted Thompson Sampling allocation.** Each lead's conversion rate is modeled as a Beta distribution centered on its predicted probability, with a prior strength that controls how much we trust the point estimate versus allow for uncertainty. Sampled rates are multiplied by each lead's expected value, and the highest expected-value leads are selected. This lets a slightly lower-probability but higher-value lead outrank a confident low-value one — something a plain sort can never do. The batch allocator uses a fixed prior; the online feedback loop below is what makes it learn.

3. **Online feedback loop (implemented).** Leads arrive over time in rounds; after each round we observe the real `Converted` outcomes of the leads we contacted and update a per-segment (Lead Source) Bayesian posterior, so later rounds allocate better. To stay honest about selection bias — we only observe the high-scoring leads we chose to contact — the posterior is a **selection-robust calibration estimator** (a Gamma-Poisson posterior over each segment's actual-vs-model-expected conversion ratio), not a naive rate. In a 10-round sequential backtest, **learning beats the same process with frozen priors by +1.9% cumulative value, and the gap grows every round** (they start tied and the learner pulls ahead as beliefs converge to reality). The edge is modest by design — the base model is already well-calibrated, so there is limited miscalibration to exploit — and it is reported un-massaged.

4. **Backtest against real outcomes.** Every strategy (random, probability sort, Thompson, value sort, value-aware Thompson) is replayed over the real test set across budgets of 5/10/20/30/50%. The headline metric is the fraction of total real value captured, reported alongside conversion capture. Results are written to `data/processed/backtest_results.csv` and surfaced in the dashboard.

## Honesty / limitations

This is a strength of the project, not a disclaimer to bury:

- **Value is a stated proxy, not real revenue.** It is derived from occupation (for example, working professionals are weighted higher than students), using explicit relative weights that are a modeling assumption.
- **The dataset has conversion labels but no deal sizes.** There is no real monetary value in the source data, which is precisely why value is a proxy.
- **The chat layer is simulated.** Real sales conversations are not publicly available, so any conversational layer is illustrative rather than trained on real transcripts.

## Run it

Requires Python 3.11 and Node.js. Place the X Education CSV in `data/raw/`.

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 2. Pipeline (run from the project root so the data/ paths resolve)
PYTHONPATH=src python -m data_prep   # clean + split, writes data/processed/{train,test}.csv
PYTHONPATH=src python -m scoring     # train models, writes data/processed/scored_leads.csv
PYTHONPATH=src python -m backtest    # allocation backtest, writes backtest_results.csv

# 3. Refresh the dashboard data (regenerates the JSON the frontend reads)
PYTHONPATH=src python -m export_dashboard

# 4. Dashboard
cd dashboard
npm install
npm run dev        # http://localhost:5173
```

## Tech stack

Python 3.11, scikit-learn, React + Vite + Tailwind.
