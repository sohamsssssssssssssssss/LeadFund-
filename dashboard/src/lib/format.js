// Small formatting + data-lookup helpers shared across components.
// NOTE: none of these invent numbers — they only format values that come from
// data/processed/backtest_results.csv via the generated backtest.json.

export const pct = (x, dp = 1) => `${(x * 100).toFixed(dp)}%`;
export const pp = (x, dp = 2) => `${x >= 0 ? "+" : ""}${x.toFixed(dp)}pp`;
export const budgetLabel = (b) => `${Math.round(b * 100)}%`;

// Display metadata for each strategy (name + accent colour for the UI).
export const STRATEGY_META = {
  random: { label: "Random", short: "Random", accent: "#64748b" },
  sort: { label: "Naive Sort", short: "Naive Sort", accent: "#94a3b8" },
  thompson: { label: "Thompson (prob)", short: "Thompson", accent: "#a78bfa" },
  sort_value: { label: "Value Sort", short: "Value Sort", accent: "#38bdf8" },
  thompson_value: { label: "LeadFund", short: "LeadFund", accent: "#34d399" },
};

// Pull one result row (a strategy at a budget) out of the flat results list.
export function lookup(results, strategy, budget) {
  return results.find(
    (r) => r.strategy === strategy && Math.abs(r.budget - budget) < 1e-9
  );
}

// Map a numeric value-proxy to a human tier label for the lead queue.
export function valueTier(value) {
  if (value >= 2.5) return "High value";
  if (value >= 1.5) return "Mid value";
  return "Base value";
}
