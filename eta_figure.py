"""Figure: SBX crossover spread (eta_c) sweep -- operator order-preservation tau and
real-coded NSGA-II mean hypervolume, showing that NO eta_c lifts the GA above the
random-key ceiling and that tau and HV are perfectly anti-correlated.
Output: ../figures/eta_sweep.pdf"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("app/data/results")
FIG = Path("../figures")
e = json.load(open(R / "eta_sweep.json"))
sw = e["sweep"]
eta = [r["eta_c"] for r in sw]
tau = [r["sbx_tau_mean"] for r in sw]
hv = [r["hv_mean"] for r in sw]
rr = e["random_restart_hv"]
pt = e["perm_tier_min_hv"]

fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.set_xscale("log")
ax.axhspan(rr, pt, color="0.85", zorder=0)
ax.axhline(rr, color="0.35", ls="--", lw=1.0, zorder=1)
ax.axhline(pt, color="0.35", ls=":", lw=1.0, zorder=1)
ax.plot(eta, hv, "o-", color="black", lw=1.6, ms=5, zorder=3, label="NSGA-II mean HV")
ax.set_xlabel(r"SBX distribution index $\eta_c$ (log scale; smaller $=$ more disruptive)")
ax.set_ylabel("mean hypervolume")
ax.set_xticks(eta); ax.set_xticklabels([str(int(x)) for x in eta])
ax.text(eta[-1], rr, " random restart ", va="bottom", ha="right", fontsize=8, color="0.25")
ax.text(eta[-1], pt, " permutation tier ", va="bottom", ha="right", fontsize=8, color="0.25")

ax2 = ax.twinx()
ax2.plot(eta, tau, "s--", color="0.45", lw=1.2, ms=4, zorder=2, label=r"SBX order-preservation $\tau$")
ax2.set_ylabel(r"Kendall $\tau$ (SPV order)")
ax2.set_ylim(0.88, 1.005)

l1, lab1 = ax.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax.legend(l1 + l2, lab1 + lab2, loc="center right", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "eta_sweep.pdf")
fig.savefig(FIG / "eta_sweep.png", dpi=200)
print("saved", FIG / "eta_sweep.pdf",
      f"| across sweep: HV always below random restart ({rr:,}); tau-HV Kendall {e['tau_vs_hv_kendall']['tau']:.2f}")
