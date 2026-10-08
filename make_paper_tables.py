"""Regenerates the LaTeX rows of Tables III and IV and every number quoted in
Section VII from the results files, so each figure in the paper traces to
saved output.

Usage: python make_paper_tables.py [results_dir]      (default: results)
Writes <results_dir>/paper_tables.tex and prints the quoted numbers.

Table rows show mean ± 95% CI. Quoted numbers are the means rounded as in
the paper; a few quoted values are derived from these (e.g. "8 epochs
faster" = variance-alone latency minus trend-gated latency, 31.0 - 23.2).
Keys refer to the results files (see README.md, section 4).
"""
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "results"
aahr = json.load(open(os.path.join(d, "aahr_results.json")))
ns = json.load(open(os.path.join(d, "netsim_results.json")))


def pm(s, digits=1, scale=1.0):
    """LaTeX '$mean\\pm ci$' for a ci95 summary; scale=100 turns ratios into %."""
    return f"${s['mean'] * scale:.{digits}f}\\pm{s['ci'] * scale:.{digits}f}$"


def m(s, digits=1, scale=1.0):
    """Rounded mean of a ci95 summary, as quoted in the text."""
    return round(s["mean"] * scale, digits)


# Table III: synthetic latency from experiment C; Edge-IIoTset columns from B2
# (EWMA-aggregated risk, tau off).
lines =["% Table III: policy & synthetic latency & Edge-IIoTset nuisance & Edge-IIoTset latency"]
labels3 = {"fixed mu_min": r"Fixed $\mu_{\min}=0.1$", "fixed mu_max": r"Fixed $\mu_{\max}=0.4$",
           "adaptive Eq.7": r"Adaptive, Eq.~\eqref{eq:volatility} only",
           "adaptive trend-gated": r"Adaptive, trend-gated"}
for v, lab in labels3.items():
    c, b = aahr["C_synthetic"][v], aahr["B2_ewma"][v]
    lines.append(f"{lab} & {pm(c['latency'])} & {pm(b['nuisance'], 2)} & {pm(b['latency'])} \\\\")

# Table IV: network simulation with the tau override on (netsim "with_tau").
lines.append("% Table IV: policy & S0 sw/100 & S0 PDR% & S1 latency & S1/S2/S3/S4 AER%")
labels4 = {"B1 delay-only": "B1 Delay-only", "AAHR fixed mu_min": r"AAHR, fixed $\mu_{\min}$",
           "AAHR fixed mu_max": r"AAHR, fixed $\mu_{\max}$",
           "AAHR Eq.7": r"AAHR, Eq.~\eqref{eq:volatility} only", "AAHR trend-gated": "AAHR, trend-gated"}
w = ns["with_tau"]
for n, lab in labels4.items():
    r = [pm(w["S0"][n]["switches_per_100"]), pm(w["S0"][n]["pdr"], 1, 100), pm(w["S1"][n]["latency"]),
         pm(w["S1"][n]["aer"], 0, 100), pm(w["S2"][n]["aer"], 0, 100), pm(w["S3"][n]["aer"], 0, 100),
         pm(w["S4"][n]["aer"], 0, 100)]
    lines.append(f"{lab} & " + " & ".join(r) + " \\\\")

out = os.path.join(d, "paper_tables.tex")
open(out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[written] {out}")

cls = aahr["classifier"]
b1, b2, b3, b3b = aahr["B1_raw"], aahr["B2_ewma"], aahr["B3_ewma_tau"], aahr["B3b_ewma_tau_strict"]
g, e7, fast = "adaptive trend-gated", "adaptive Eq.7", "fixed mu_min"
print("\n=== Numbers quoted in Section VII-A (Edge-IIoTset / synthetic) ===")
print(f"Classifier: rows={cls['rows']}, ROC-AUC={cls['auc']:.3f}, top feature={cls['top_feature']} "
      f"({cls['top_importance']:.2f})")
print("Raw per-packet risk, post-onset switches per policy:",
      {p: m(s["post_onset"]) for p, s in b1.items()})
print(f"Latency: variance alone {m(b2[e7]['latency'])} vs fixed mu_max {m(b2['fixed mu_max']['latency'])}; "
      f"trend-gated {m(b2[g]['latency'])} vs fixed mu_min {m(b2[fast]['latency'])}; "
      f"nuisance trend-gated {m(b2[g]['nuisance'], 2)} vs fixed mu_min {m(b2[fast]['nuisance'], 2)}")
print(f"Synthetic latency: trend-gated {m(aahr['C_synthetic'][g]['latency'])}, "
      f"fixed mu_min {m(aahr['C_synthetic'][fast]['latency'])}, variance alone {m(aahr['C_synthetic'][e7]['latency'])}")
print("kappa sensitivity (latency):",
      {k: {p: m(s["latency"]) for p, s in v.items()} for k, v in aahr["B4_kappa"].items()},
      "| kappa=1:", {p: m(b2[p]["latency"]) for p in (e7, g)})
print("t_crit sensitivity (nuisance, latency):",
      {k: (m(v[g]["nuisance"], 2), m(v[g]["latency"])) for k, v in aahr["B4_tcrit"].items()})
print(f"tau={cls['tau']:.3f}: latency per policy", {p: m(s["latency"]) for p, s in b3.items()},
      "extra nuisance vs tau off", {p: round(b3[p]["nuisance"]["mean"] - b2[p]["nuisance"]["mean"], 2) for p in b3})
print(f"stricter tau={cls['tau_per_packet']:.3f}: latency per policy", {p: m(s["latency"]) for p, s in b3b.items()},
      "nuisance", {p: m(s["nuisance"], 2) for p, s in b3b.items()})
print("Static telemetry (Theorem 2):", aahr["A"])
print("Zero-clip exercised (aggregated-risk runs, of", aahr["seeds"], "):", b2[g]["clip_runs"])

print("\n=== Numbers quoted in Section VII-B (network simulation) ===")
c = ns["classifier"]
print(f"Classifier held-out AUC={c['test_auc']:.2f}, tau={c['tau']:.3f}")
print("AER % with tau (S1..S4):", {p: [m(w[s][p]["aer"], 0, 100) for s in ("S1", "S2", "S3", "S4")] for p in w["S1"]})
print("S2 runs never leaving attacked paths:", {p: w["S2"][p]["never_reacted"] for p in w["S2"]})
print("S0 PDR %:", {p: m(w["S0"][p]["pdr"], 1, 100) for p in w["S0"]})
print("S0 switches/100 epochs:", {p: m(w["S0"][p]["switches_per_100"]) for p in w["S0"]})
wo = ns["without_tau"]
print("tau off: S2 latency", {p: m(wo["S2"][p]["latency"]) for p in wo["S2"]})
print("tau off: S3/S4 AER %", {p: (m(wo["S3"][p]["aer"], 0, 100), m(wo["S4"][p]["aer"], 0, 100)) for p in wo["S3"]})
