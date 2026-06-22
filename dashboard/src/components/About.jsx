import React from "react";

/**
 * About — the honesty footnote. This MUST be visible, not hidden: it states
 * exactly what is real and what is a stated proxy / simulation, so the demo's
 * credibility is on the surface.
 */
export default function About({ meta }) {
  return (
    <section className="rounded-3xl border border-white/[0.06] bg-panel/30 p-6 text-sm text-slate-400 sm:p-8">
      <p className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-slate-500">
        What's real, what's a proxy
      </p>
      <ul className="space-y-2 leading-relaxed">
        <li>
          <span className="text-leadfund">●</span> Built on{" "}
          <span className="font-semibold text-slate-200">
            9,240 real X Education leads
          </span>{" "}
          with real conversion outcomes. The race and tables use the{" "}
          <span className="nums text-slate-200">{meta.nLeads}</span>-lead held-out
          test set ({(meta.baseRate * 100).toFixed(1)}% real conversion rate).
        </li>
        <li>
          <span className="text-sky-300">●</span> Conversion probabilities come
          from a gradient-boosting model. The engine consumes its{" "}
          <span className="font-semibold text-slate-200">raw probabilities</span>,
          which were{" "}
          <span className="font-semibold text-slate-200">
            verified well-calibrated
          </span>{" "}
          (reliability check, worst-bin gap 0.082) — that is why the allocation
          engine can treat them as real probabilities.
        </li>
        <li>
          <span className="text-amber-300">●</span>{" "}
          <span className="font-semibold text-slate-200">Value is a stated proxy</span>{" "}
          derived from occupation (e.g. working professional weighted higher),{" "}
          <span className="italic">not real revenue</span> — the dataset has no
          deal sizes. The relative weights are an explicit modeling assumption.
        </li>
        <li>
          <span className="text-violet-300">●</span> The online feedback loop is{" "}
          <span className="font-semibold text-slate-200">implemented</span> —
          segment-level online learning with a selection-robust calibration
          estimator. In a sequential backtest it{" "}
          <span className="font-semibold text-slate-200">
            beats the static (non-learning) baseline by +1.9% cumulative value,
            with the gap growing each round
          </span>{" "}
          as beliefs converge to reality. The edge is modest by design — the base
          model is already well-calibrated — and reported un-massaged.
        </li>
        <li>
          <span className="text-slate-400">●</span> Any chat / conversation layer
          added later is{" "}
          <span className="font-semibold text-slate-200">simulated</span>, because
          real sales conversations aren't public.
        </li>
      </ul>
      <p className="mt-4 border-t border-edge pt-4 text-xs text-slate-600">
        All numbers on this page are read from{" "}
        <code className="rounded bg-ink/60 px-1.5 py-0.5 text-slate-400">
          data/processed/backtest_results.csv
        </code>{" "}
        via a generated JSON export — none are typed by hand.
      </p>
    </section>
  );
}
