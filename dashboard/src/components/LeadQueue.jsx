import React from "react";
import { pct, valueTier } from "../lib/format";

/**
 * LeadQueue — the explainability story. Shows the top leads LeadFund actually
 * selected (value-aware Thompson) at the current budget, ranked by expected
 * value = conversion probability x value proxy. Each row has a plain-language
 * "why". Data comes from data.leadQueue[budgetKey] in backtest.json.
 */

const tierColor = (tier) =>
  tier === "High value"
    ? "text-leadfund"
    : tier === "Mid value"
    ? "text-sky-300"
    : "text-slate-400";

function whyText(lead) {
  const tier = valueTier(lead.value);
  return `${tier}: ${lead.occupation}, ${pct(lead.prob, 0)} conversion prob`;
}

export default function LeadQueue({ data, budget }) {
  // Keys in backtest.json are stringified floats: "0.05","0.1","0.2",...
  const key = String(budget);
  const leads = data.leadQueue[key] ?? [];

  return (
    <section className="rounded-3xl border border-white/[0.06] bg-panel/40 p-6 sm:p-8">
      <div className="mb-1 flex items-center justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-[0.2em] text-sky-300">
            The Queue
          </p>
          <h2 className="mt-1 text-2xl font-bold text-slate-50">
            Who LeadFund would contact first
          </h2>
        </div>
        <span className="nums rounded-md bg-edge/60 px-2 py-1 text-xs text-slate-400">
          top {leads.length} picks
        </span>
      </div>
      <p className="mb-4 text-sm text-slate-400">
        Ranked by expected value (conversion probability × value proxy) — not by
        probability alone.
      </p>

      <div className="scroll-thin max-h-[28rem] overflow-y-auto rounded-2xl border border-edge">
        <table className="w-full text-left text-sm">
          <thead className="sticky top-0 bg-ink/90 text-xs uppercase tracking-wider text-slate-500 backdrop-blur">
            <tr>
              <th className="px-4 py-3 font-medium">#</th>
              <th className="px-4 py-3 font-medium">Lead</th>
              <th className="px-4 py-3 font-medium">Contact</th>
              <th className="px-4 py-3 font-medium">Value tier</th>
              <th className="px-4 py-3 text-right font-medium">Conv. prob</th>
              <th className="px-4 py-3 text-right font-medium">Value</th>
              <th className="px-4 py-3 text-right font-medium">Exp. value</th>
              <th className="px-4 py-3 font-medium">Why</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-edge/60">
            {leads.map((lead, i) => {
              const tier = valueTier(lead.value);
              return (
                <tr key={lead.leadNumber} className="hover:bg-edge/30">
                  <td className="nums px-4 py-2.5 text-slate-600">{i + 1}</td>
                  <td className="nums px-4 py-2.5 text-slate-300">
                    #{lead.leadNumber}
                  </td>
                  <td className="px-4 py-2.5 font-mono text-xs text-slate-400">
                    {lead.contact}
                  </td>
                  <td className={`px-4 py-2.5 font-medium ${tierColor(tier)}`}>
                    {tier}
                  </td>
                  <td className="nums px-4 py-2.5 text-right text-slate-200">
                    {pct(lead.prob, 0)}
                  </td>
                  <td className="nums px-4 py-2.5 text-right text-slate-400">
                    {lead.value.toFixed(1)}×
                  </td>
                  <td className="nums px-4 py-2.5 text-right font-semibold text-leadfund">
                    {lead.expectedScore.toFixed(2)}
                  </td>
                  <td className="px-4 py-2.5 text-xs text-slate-500">
                    {whyText(lead)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-3 space-y-1.5 text-xs text-slate-600">
        <p>
          Contact info is masked (shown as a lead ref number) — the X Education
          dataset does not include phone or email fields. In a live deployment this
          column would display the lead's actual phone number or email pulled from
          the CRM.
        </p>
        <p>
          "Value" is a stated proxy (occupation-derived relative weight), not real
          revenue — see the note below.
        </p>
      </div>
    </section>
  );
}
