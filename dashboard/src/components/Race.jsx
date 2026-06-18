import React, { useEffect, useRef, useState } from "react";
import { lookup, pct, pp, budgetLabel } from "../lib/format";

/**
 * Race — the hero. A side-by-side animated race between the two strategies on
 * the REAL value-capture numbers for the selected budget:
 *   LEFT  "Naive Sort"  = strategy "sort"            (prob-only)
 *   RIGHT "LeadFund"    = strategy "thompson_value"  (value-aware Thompson)
 *
 * On Run, both counters climb (eased) toward their true value-capture rate and
 * a 100-dot grid fills to that percentage. Every target comes from props.data;
 * nothing here is hardcoded.
 */

const DOTS = 100; // each dot = 1% of total realisable value captured
const DURATION = 2200; // ms

// Animate two counters from 0 to their targets; re-arm whenever targets change.
function useRace(targetNaive, targetLF) {
  const [vals, setVals] = useState({ naive: 0, lf: 0 });
  const [running, setRunning] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const raf = useRef(0);

  // Reset to zero when the budget (and therefore the targets) changes.
  useEffect(() => {
    cancelAnimationFrame(raf.current);
    setVals({ naive: 0, lf: 0 });
    setRunning(false);
    setHasRun(false);
  }, [targetNaive, targetLF]);

  const run = () => {
    cancelAnimationFrame(raf.current);
    const start = performance.now();
    setRunning(true);
    setHasRun(true);
    const tick = (now) => {
      const t = Math.min(1, (now - start) / DURATION);
      const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
      setVals({ naive: targetNaive * eased, lf: targetLF * eased });
      if (t < 1) {
        raf.current = requestAnimationFrame(tick);
      } else {
        setVals({ naive: targetNaive, lf: targetLF }); // snap to exact real value
        setRunning(false);
      }
    };
    raf.current = requestAnimationFrame(tick);
  };

  useEffect(() => () => cancelAnimationFrame(raf.current), []);
  return { vals, running, hasRun, run };
}

function DotGrid({ filled, accent }) {
  // filled = fraction in [0,1]; render DOTS cells, light up the first N.
  const n = Math.round(filled * DOTS);
  return (
    <div className="grid grid-cols-10 gap-1">
      {Array.from({ length: DOTS }).map((_, i) => (
        <div
          key={i}
          className="aspect-square rounded-[3px] transition-colors duration-300"
          style={{
            backgroundColor: i < n ? accent : "rgba(148,163,184,0.10)",
            boxShadow: i < n ? `0 0 6px ${accent}55` : "none",
          }}
        />
      ))}
    </div>
  );
}

function Runner({ title, subtitle, accent, value, isWinner }) {
  return (
    <div
      className="flex-1 rounded-2xl border bg-panel/70 p-5 transition-shadow"
      style={{
        borderColor: isWinner ? `${accent}66` : "#1c2433",
        boxShadow: isWinner ? `0 0 30px ${accent}22` : "none",
      }}
    >
      <div className="mb-3 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold tracking-wide text-slate-100">
            {title}
          </h3>
          <p className="text-xs text-slate-500">{subtitle}</p>
        </div>
        <span
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: accent, boxShadow: `0 0 10px ${accent}` }}
        />
      </div>

      <div className="mb-4 flex items-baseline gap-2">
        <span
          className="nums text-5xl font-bold tracking-tight"
          style={{ color: accent }}
        >
          {pct(value, 1)}
        </span>
        <span className="text-xs text-slate-500">of total value</span>
      </div>

      {/* progress bar */}
      <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-edge/60">
        <div
          className="h-full rounded-full transition-[width] duration-150 ease-out"
          style={{ width: `${value * 100}%`, backgroundColor: accent }}
        />
      </div>

      <DotGrid filled={value} accent={accent} />
    </div>
  );
}

export default function Race({ data, budget, budgets, onBudgetChange }) {
  const naiveRow = lookup(data.results, "sort", budget);
  const lfRow = lookup(data.results, "thompson_value", budget);
  const targetNaive = naiveRow.valueCaptureRate;
  const targetLF = lfRow.valueCaptureRate;

  const { vals, running, hasRun, run } = useRace(targetNaive, targetLF);

  // Live gap follows the animation; final value is the real number.
  const gapPp = (vals.lf - vals.naive) * 100;
  const finalGapPp = (targetLF - targetNaive) * 100;
  const lfAhead = targetLF >= targetNaive;
  const budgetIdx = budgets.findIndex((b) => Math.abs(b - budget) < 1e-9);

  return (
    <section className="rounded-3xl border border-edge bg-panel/40 p-6 sm:p-8">
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-leadfund">
            The Race
          </p>
          <h2 className="mt-1 text-2xl font-bold text-slate-50">
            Value captured: Naive Sort vs. LeadFund
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Same contact budget, same real leads — who captures more of the
            total deal value?
          </p>
        </div>

        <button
          onClick={run}
          disabled={running}
          className="rounded-xl bg-leadfund px-6 py-3 text-sm font-bold text-ink shadow-lg shadow-leadfund/20 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {running ? "Running…" : hasRun ? "Run again" : "Run the race"}
        </button>
      </div>

      {/* Budget slider — 5 real budgets (5/10/20/30/50%). */}
      <div className="mb-6 rounded-2xl border border-edge bg-ink/40 p-4">
        <div className="mb-2 flex items-center justify-between">
          <label className="text-xs font-medium uppercase tracking-wider text-slate-400">
            Contact budget
          </label>
          <span className="nums rounded-md bg-leadfund/10 px-2 py-0.5 text-sm font-bold text-leadfund">
            {budgetLabel(budget)} of leads
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={budgets.length - 1}
          step={1}
          value={budgetIdx}
          onChange={(e) => onBudgetChange(budgets[Number(e.target.value)])}
          className="w-full accent-leadfund"
        />
        <div className="mt-1 flex justify-between text-xs text-slate-500">
          {budgets.map((b) => (
            <span
              key={b}
              className={
                Math.abs(b - budget) < 1e-9 ? "font-bold text-leadfund" : ""
              }
            >
              {budgetLabel(b)}
            </span>
          ))}
        </div>
      </div>

      {/* The two runners + the gap badge between them. */}
      <div className="flex flex-col items-stretch gap-4 lg:flex-row">
        <Runner
          title="Naive Sort"
          subtitle="prob-only ranking — what everyone builds"
          accent="#94a3b8"
          value={vals.naive}
          isWinner={false}
        />

        <div className="flex flex-col items-center justify-center gap-2 px-2">
          <span className="text-[10px] uppercase tracking-widest text-slate-500">
            value gap
          </span>
          <div
            className={`nums rounded-xl border px-4 py-3 text-center ${
              hasRun && !running ? "animate-pulseGap" : ""
            }`}
            style={{
              borderColor: lfAhead ? "#34d39966" : "#f8717166",
              backgroundColor: lfAhead ? "#34d39911" : "#f8717111",
            }}
          >
            <div
              className="text-2xl font-extrabold"
              style={{ color: lfAhead ? "#34d399" : "#f87171" }}
            >
              {pp(gapPp)}
            </div>
            <div className="text-[10px] text-slate-500">LeadFund vs Naive</div>
          </div>
          {hasRun && !running && Math.abs(finalGapPp) < 0.1 && (
            <span className="max-w-[8rem] text-center text-[10px] leading-tight text-slate-500">
              near-tie at this budget — and we own that
            </span>
          )}
        </div>

        <Runner
          title="LeadFund"
          subtitle="value-aware Thompson allocation"
          accent="#34d399"
          value={vals.lf}
          isWinner={lfAhead}
        />
      </div>

      {/* Plain-language readout of the real end state. */}
      <p className="mt-5 text-center text-sm text-slate-400">
        At a <span className="font-semibold text-slate-200">{budgetLabel(budget)}</span>{" "}
        budget, LeadFund captures{" "}
        <span className="nums font-semibold text-leadfund">{pct(targetLF)}</span>{" "}
        of total value vs Naive Sort's{" "}
        <span className="nums font-semibold text-naive">{pct(targetNaive)}</span>{" "}
        — a real edge of{" "}
        <span className="nums font-semibold text-slate-100">{pp(finalGapPp)}</span>.
      </p>
    </section>
  );
}
