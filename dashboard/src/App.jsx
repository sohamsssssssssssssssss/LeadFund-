import React, { useState } from "react";
import data from "./data/backtest.json"; // generated from backtest_results.csv
import Race from "./components/Race";
import LeadQueue from "./components/LeadQueue";
import ResultsTable from "./components/ResultsTable";
import About from "./components/About";
import { pct } from "./lib/format";

/**
 * App — single-page LeadFund dashboard. Owns the one piece of shared state
 * (the selected budget) and passes it to the race + lead queue. All data is in
 * React state / props only — no localStorage or sessionStorage.
 */
export default function App() {
  // Default budget = 5%: the budget where LeadFund's value edge is largest.
  const [budget, setBudget] = useState(0.05);
  const { meta } = data;

  return (
    <div className="mx-auto max-w-6xl px-4 py-10 sm:px-6 sm:py-14">
      {/* Header */}
      <header className="mb-10">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-leadfund/15 text-lg font-black text-leadfund ring-1 ring-leadfund/30">
            ◆
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-slate-50">
              LeadFund
            </h1>
            <p className="text-xs text-slate-500">
              Autonomous Lead Allocation Engine — leads as a portfolio
            </p>
          </div>
        </div>

        {/* Real headline stats from the data. */}
        <div className="mt-6 flex flex-wrap gap-3">
          <Stat label="Real leads (test set)" value={meta.nLeads.toLocaleString()} />
          <Stat label="Real conversions" value={meta.totalConversions.toLocaleString()} />
          <Stat label="Base conversion rate" value={pct(meta.baseRate, 1)} />
          <Stat
            label="Total value (proxy units)"
            value={meta.totalValue.toLocaleString()}
          />
        </div>
      </header>

      <main className="space-y-8">
        {/* HERO — build/show first */}
        <Race
          data={data}
          budget={budget}
          budgets={data.budgets}
          onBudgetChange={setBudget}
        />

        {/* SECONDARY */}
        <LeadQueue data={data} budget={budget} />
        <ResultsTable data={data} />

        {/* HONESTY */}
        <About meta={meta} />
      </main>

      <footer className="mt-12 text-center text-xs text-slate-600">
        LeadFund · value-aware lead allocation on real X Education data
      </footer>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="rounded-xl border border-white/10 bg-panel/50 px-5 py-3">
      <div className="nums text-2xl font-bold text-slate-100">{value}</div>
      <div className="text-xs uppercase tracking-wider text-slate-500">
        {label}
      </div>
    </div>
  );
}
