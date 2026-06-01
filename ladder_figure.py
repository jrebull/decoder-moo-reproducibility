"""The 'ladder' figure: per-run hypervolume for all six methods on the base
instance (same 25k-eval budget, same feasibility-preserving decoder), ordered to
show that the representation-operator match -- not the metaheuristic family --
governs performance. Three permutation-native paradigms (swarm, decomposition,
GA) cluster at the top. -> ../MICAI/figures/ladder.pdf"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox": "tight"})
R = Path("app/data/results")
st = json.load(open(R / "stats_test.json"))
ctl = json.load(open(R / "controls.json"))
dm = json.load(open(R / "discrete_mohho.json"))
pn = json.load(open(R / "perm_nsga.json"))
md = json.load(open(R / "perm_moead.json"))

data = [np.array(st["nsga2_hv"]) / 1e6,
        np.array(ctl["random_restart"]["per_seed_hv"]) / 1e6,
        np.array(st["mohho_hv"]) / 1e6,
        np.array(dm["per_run_hv"]) / 1e6,
        np.array(md["per_run_hv"]) / 1e6,
        np.array(pn["per_run_hv"]) / 1e6]
labels = ["NSGA-II\n(real-coded)", "Random\nrestart", "MOHHO\n(real-coded)",
          "Discrete-\nMOHHO", "perm-\nMOEA/D", "perm-\nNSGA-II"]
cols = ["#E67E22", "#9AA3AF", "#E8B04B", "#2E86DE", "#16A085", "#27AE60"]

fig, ax = plt.subplots(figsize=(7.0, 3.5))
parts = ax.violinplot(data, positions=range(1, 7), showmeans=False,
                      showextrema=False, widths=0.85)
for b, c in zip(parts["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.30); b.set_edgecolor(c)
bp = ax.boxplot(data, positions=range(1, 7), widths=0.34, patch_artist=True,
                showfliers=True, medianprops=dict(color="k", lw=1.1))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.65)
ax.set_xticks(range(1, 7)); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel(r"Hypervolume ($\times 10^{6}$)")
ax.axvspan(0.5, 3.5, color="#E67E22", alpha=0.05)
ax.axvspan(3.5, 6.5, color="#2E86DE", alpha=0.05)
lo = ax.get_ylim()[0]
ax.text(2.0, lo, "random-key encoding", ha="center", va="bottom", fontsize=7.5, color="#9C5410")
ax.text(5.0, lo, "permutation-native (3 paradigms)", ha="center", va="bottom", fontsize=7.5, color="#1B5E9C")
ax.grid(axis="y", alpha=0.25)
fig.savefig("../MICAI/figures/ladder.pdf"); fig.savefig("../MICAI/figures/ladder.png", dpi=200)
print("saved 6-method ladder | means:", [f"{float(np.mean(x))*1e6:,.0f}" for x in data])
