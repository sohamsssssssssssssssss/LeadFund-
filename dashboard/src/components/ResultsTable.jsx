import React, { useState } from "react";
import { STRATEGY_META, lookup, pct } from "../lib/format";

/**
 * ResultsTable — full transparency. Pivots the entire backtest (every strategy
 * at every budget) so a judge can inspect all the numbers. Toggle between the
 * value-capture and conversion-capture metrics. All values come straight from
 * data.results (backtest.json -> backtest_results.csv).
 */

const STRATS = ["random", "sort", "thompson", "sort_value", "thompson_value"];

const METRICS = {
  value: { label: "Value captured", field: "valueCaptureRate" },
  conversions: { label: "Conversions captured", field: "captureRate" },
};

export default function ResultsTable({ data }) {
  const [metric, setMetric] = useState("value");
  const field = METRICS[metric].field;

  // Best strategy per budget (to highlight the leader in each row).
  const bestPerBudget = {};
  data.budgets.forEach((b) => {
    let best = -1;
    let bestStrat = null;
    STRATS.forEach((s) => {
      const v = lookup(data.results, s, b)[field];
      if (v > best) {
        best = v;
        bestStrat = s;
      }
    });
    bestPerBudget[b] = bestStrat;
  });

  return (
    <section className="rounded-3xl border border-edge bg-panel/40 p-6 sm:p-8">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-slate-400">
            Full Results
          </p>
          <h2 className="mt-1 text-2xl font-bold text-slate-50">
            Every strategy, every budget
          </h2>
        </div>
        <div className="flex gap-1 rounded-xl border border-edge bg-ink/50 p-1">
          {Object.entries(METRICS).map(([k, m]) => (
            <button
              key={k}
              onClick={() => setMetric(k)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                metric === k
                  ? "bg-leadfund/15 text-leadfund"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto rounded-2xl border border-edge">
        <table className="w-full text-sm">
          <thead className="bg-ink/70 text-xs uppercase tracking-wider text-slate-500">
            <tr>
              <th className="px-4 py-3 text-left font-medium">Budget</th>
              {STRATS.map((s) => (
                <th key={s} className="px-4 py-3 text-right font-medium">
                  <span style={{ color: STRATEGY_META[s].accent }}>
                    {STRATEGY_META[s].short}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-edge/60">
            {data.budgets.map((b) => (
              <tr key={b} className="hover:bg-edge/20">
                <td className="nums px-4 py-3 text-left font-semibold text-slate-300">
                  {pct(b, 0)}
                </td>
                {STRATS.map((s) => {
                  const v = lookup(data.results, s, b)[field];
                  const isBest = bestPerBudget[b] === s;
                  return (
                    <td
                      key={s}
                      className={`nums px-4 py-3 text-right ${
                        isBest
                          ? "font-bold text-leadfund"
                          : "text-slate-300"
                      }`}
                    >
                      {pct(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-xs text-slate-600">
        Bold = best strategy at that budget for the selected metric. Note the
        uncertainty-aware (Thompson) variants land close to their greedy
        counterparts — expected with well-calibrated probabilities — while
        value-awareness drives the gains at small budgets.
      </p>
    </section>
  );
}
