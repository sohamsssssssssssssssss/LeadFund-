import React, { useEffect, useRef, useState } from "react";
import { lookup, pct, pp, budgetLabel } from "../lib/format";

/**
 * Race — the hero. A side-by-side race between the two strategies on the REAL
 * value numbers for the selected budget:
 *   LEFT  "Naive Sort"  = strategy "sort"            (prob-only)
 *   RIGHT "LeadFund"    = strategy "thompson_value"  (value-aware Thompson)
 *
 * Design intent: make the GAP the hero. The old version mapped each strategy
 * onto a 0-100%-of-total bar, so a real edge (26.2% vs 22.0%) read as a near-
 * tie. Here the headline is the RELATIVE value lift (+19.5% more value), the
 * cards show value UNITS captured on a leader-relative bar (264 vs 221 visibly
 * differ), and the honest share-of-total ("+Xpp") stays as secondary context.
 *
 * Every number is derived from props.data (backtest.json) — nothing hardcoded.
 */

const DURATION = 2200; // ms

// Format value-unit figures: integer when whole, else one decimal. Real values
// land on .0/.5; during the animation they count up smoothly.
const fmtUnits = (n) => (Number.isInteger(n) ? `${n}` : n.toFixed(1));
// Signed relative-lift percentage, e.g. "+19.5%" / "-0.3%".
const fmtLift = (frac) => `${frac >= 0 ? "+" : ""}${(frac * 100).toFixed(1)}%`;

// Animate two rates from 0 to their targets; re-arm whenever targets (budget)
// change. easeOutCubic so the counters decelerate into the real end state.
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

function Runner({ title, subtitle, accent, barColor, units, rate, barFrac, isWinner, raceDone }) {
  return (
    <div
      className="flex-1 rounded-2xl border bg-panel/70 p-5 transition-all duration-300"
      style={{
        borderColor: isWinner ? `${accent}66` : "rgba(255,255,255,0.06)",
        boxShadow: isWinner ? `0 0 30px ${accent}22` : "none",
      }}
    >
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-base font-semibold tracking-wide text-slate-100">
            {title}
          </h3>
          <p className="text-xs text-slate-500">{subtitle}</p>
        </div>
        <span
          className="h-3 w-3 rounded-full"
          style={{ backgroundColor: accent, boxShadow: `0 0 10px ${accent}` }}
        />
      </div>

      {/* Value UNITS captured — larger, bolder numbers. */}
      <div className="flex items-baseline gap-2">
        <span
          className="nums text-6xl font-bold tracking-tight"
          style={{ color: accent }}
        >
          {fmtUnits(units)}
        </span>
        <span className="text-xs text-slate-500">value units captured</span>
      </div>
      {/* Share-of-total as honest secondary context. */}
      <p className="nums mt-1 text-xs text-slate-500">
        {pct(rate)} of total realisable value
      </p>

      {/* Leader-relative bar. LeadFund gets a bright green fill with glow,
          Naive Sort gets a muted grey — making the difference visceral. */}
      <div className="mt-4 h-3 w-full overflow-hidden rounded-full bg-edge/60">
        <div
          className={`h-full rounded-full transition-all duration-200 ease-out ${
            isWinner && raceDone ? "bar-win-glow" : ""
          }`}
          style={{
            width: `${barFrac * 100}%`,
            backgroundColor: barColor,
            color: barColor,
            boxShadow: isWinner ? `0 0 6px ${barColor}` : "none",
          }}
        />
      </div>
    </div>
  );
}

export default function Race({ data, budget, budgets, onBudgetChange }) {
  const totalValue = data.meta.totalValue;
  const naiveRow = lookup(data.results, "sort", budget);
  const lfRow = lookup(data.results, "thompson_value", budget);
  const targetNaive = naiveRow.valueCaptureRate;
  const targetLF = lfRow.valueCaptureRate;

  const { vals, running, hasRun, run } = useRace(targetNaive, targetLF);

  // Everything below is derived from the two animated rates, so the hero,
  // unit counters, and bars move together and snap to the real end state.
  const naiveUnits = vals.naive * totalValue;
  const lfUnits = vals.lf * totalValue;
  const deltaUnits = lfUnits - naiveUnits;
  const relLift = vals.naive > 0 ? vals.lf / vals.naive - 1 : 0;
  const gapPp = (vals.lf - vals.naive) * 100;
  const leaderMax = Math.max(naiveUnits, lfUnits, 1e-9);
  // Widen the bar scale by 1.5x so neither bar hits 100% of the container
  // — makes the gap between 221 and 264 immediately visible.
  const barScale = leaderMax * 1.5;

  // Real (target) end state — used for winner styling and the bottom readout,
  // so they don't depend on mid-animation values.
  const finalNaiveUnits = targetNaive * totalValue;
  const finalLfUnits = targetLF * totalValue;
  const finalGapPp = (targetLF - targetNaive) * 100;
  const finalRelLift = targetNaive > 0 ? targetLF / targetNaive - 1 : 0;
  const isTie = Math.abs(finalGapPp) < 0.1;
  const lfAhead = targetLF >= targetNaive;

  const heroColor = lfAhead ? "#34d399" : "#f87171";
  const budgetIdx = budgets.findIndex((b) => Math.abs(b - budget) < 1e-9);

  return (
    <section className="rounded-3xl border border-white/[0.06] bg-panel/40 p-6 sm:p-8">
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

      {/* HERO band — the single number to remember: relative value lift. */}
      <div
        className="relative mb-6 overflow-hidden rounded-2xl border border-white/5 px-6 py-8 text-center"
        style={{
          borderColor: `${heroColor}55`,
          backgroundColor: `${heroColor}08`,
        }}
      >
        {/* Animated green radial glow behind the number — feels like the
            number is radiating, not just sitting there. */}
        <div
          className={`absolute inset-0 ${hasRun && !running ? "animate-glowPulse" : "opacity-40"}`}
          style={{
            background: `radial-gradient(45% 45% at 50% 45%, ${heroColor}55, transparent 70%)`,
            boxShadow: `inset 0 0 120px ${heroColor}18`,
            pointerEvents: "none",
          }}
        />
        <div className="relative z-10">
          <div
            className="nums text-7xl font-extrabold leading-none tracking-tight sm:text-8xl"
            style={{ color: heroColor }}
          >
            {fmtLift(relLift)}
          </div>
          <p className="mt-3 text-xs text-slate-400">
            {lfAhead ? "more" : "less"} total value captured for the same{" "}
            <span className="font-semibold text-slate-200">
              {budgetLabel(budget)}
            </span>{" "}
            contact budget
          </p>
          <p className="nums mt-1 text-xs text-slate-500">
            {deltaUnits >= 0 ? "+" : ""}
            {fmtUnits(deltaUnits)} value units · {pp(gapPp)} of total
          </p>
        </div>
      </div>

      {/* Two runner cards — value units on a leader-relative bar. */}
      <div className="flex flex-col items-stretch gap-4 lg:flex-row">
        <Runner
          title="Naive Sort"
          subtitle="prob-only ranking — what everyone builds"
          accent="#94a3b8"
          barColor="#475569"
          units={naiveUnits}
          rate={vals.naive}
          barFrac={naiveUnits / barScale}
          isWinner={!isTie && !lfAhead}
          raceDone={hasRun && !running}
        />
        <Runner
          title="LeadFund"
          subtitle="value-aware Thompson allocation"
          accent="#34d399"
          barColor="#34d399"
          units={lfUnits}
          rate={vals.lf}
          barFrac={lfUnits / barScale}
          isWinner={!isTie && lfAhead}
          raceDone={hasRun && !running}
        />
      </div>

      {/* Plain-language readout of the REAL end state, branching on real gap. */}
      <p className="mt-5 text-center text-sm text-slate-400">
        {isTie ? (
          <>
            A dead heat, and we own that: when you can contact a third of your
            leads, ranking is enough.
          </>
        ) : finalGapPp > 0 ? (
          <>
            LeadFund captures{" "}
            <span className="nums font-semibold text-leadfund">
              {fmtUnits(finalLfUnits)}
            </span>{" "}
            value units vs Naive Sort's{" "}
            <span className="nums font-semibold text-naive">
              {fmtUnits(finalNaiveUnits)}
            </span>{" "}
            —{" "}
            <span className="nums font-semibold text-slate-100">
              {fmtLift(finalRelLift)}
            </span>{" "}
            more value ({pp(finalGapPp)} of total). The edge is largest when
            contact capacity is scarcest.
          </>
        ) : (
          <>
            Naive edges ahead by a hair ({pp(finalGapPp)}); at this budget you
            contact nearly everyone, so allocation stops mattering.
          </>
        )}
      </p>
    </section>
  );
}
