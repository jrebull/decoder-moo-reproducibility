"""
Publication figures for the MICAI/LNCS paper. English labels, vectorial PDF
(plus PNG backup), serif fonts to match the Springer LNCS body text.

Reads only released artifacts (CSV/JSON in app/data/results) so every figure is
reproducible. Outputs to ../MICAI/figures/.

Figures:
  convergence.pdf     HV mean +/- std vs iteration (30 runs), early-saturation markers
  pareto3d.pdf        combined Pareto front in (f1,f2,f3), colored by f3, FIFO star
  pareto_f1f2.pdf     f1-f2 projection colored by f3 (waste), FIFO dominated
  hv_box.pdf          HV distribution MOHHO vs NSGA-II (30 runs each) + test annotation
  nsga2_overlay.pdf   MOHHO vs NSGA-II combined fronts in f1-f2 (if nsga2_front.json)
"""
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

R = Path("app/data/results")
FIG = Path("../MICAI/figures")
FIG.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, RED, GREY = "#2E86DE", "#E67E22", "#E74C3C", "#9AA3AF"
millions = FuncFormatter(lambda x, _: f"{x/1e6:.2f}")


def load_front():
    pareto, fifo = [], None
    for row in csv.DictReader(open(R / "pareto_front.csv")):
        pt = (float(row["f1"]), float(row["f2"]), float(row["f3"]))
        if row["type"] == "pareto":
            pareto.append(pt)
        else:
            fifo = pt
    return np.array(pareto), fifo


def fig_convergence():
    it, mean, std = [], [], []
    for row in csv.DictReader(open(R / "convergence.csv")):
        it.append(int(row["iteration"]))
        mean.append(float(row["hv_mean"]))
        std.append(float(row["hv_std"]))
    it, mean, std = np.array(it), np.array(mean), np.array(std)
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.fill_between(it, mean - std, mean + std, color=BLUE, alpha=0.18,
                    label=r"$\pm 1$ s.d. (30 runs)")
    ax.plot(it, mean, color=BLUE, lw=1.6, label="mean hypervolume")
    final = mean[-1]
    for frac, lab in [(0.95, "95%"), (0.99, "99%")]:
        idx = int(np.argmax(mean >= frac * final))
        ax.axvline(idx, color=GREY, ls=":", lw=1.0)
        ax.text(idx + 6, mean.min() + 0.06 * (final - mean.min()),
                f"{lab} @ it.\\,{idx}", fontsize=7.5, color="#555", rotation=90,
                va="bottom")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hypervolume")
    ax.yaxis.set_major_formatter(millions)
    ax.text(0.015, 1.02, r"$\times 10^{6}$", transform=ax.transAxes, fontsize=7.5)
    ax.set_xlim(0, it.max())
    ax.legend(loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.25)
    save(fig, "convergence")


def fig_pareto3d():
    P, fifo = load_front()
    fig = plt.figure(figsize=(5.4, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="viridis_r",
                    s=14, alpha=0.85, edgecolors="none")
    ax.scatter([fifo[0]], [fifo[1]], [fifo[2]], marker="*", s=170, c=RED,
               edgecolors="k", linewidths=0.5, label="FIFO baseline")
    # extreme solutions
    for m, lab in [(0, "min $f_1$"), (1, "min $f_2$"), (2, "min $f_3$")]:
        e = P[np.argmin(P[:, m])]
        ax.scatter([e[0]], [e[1]], [e[2]], s=55, facecolors="none",
                   edgecolors="k", linewidths=1.0)
    ax.set_xlabel(r"$f_1$  waiting load", labelpad=2)
    ax.set_ylabel(r"$f_2$  disparity (yr)", labelpad=2)
    ax.set_zlabel(r"$f_3$  waste (visas)", labelpad=2)
    ax.view_init(elev=22, azim=-58)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
    cb.set_label(r"$f_3$ (wasted visas)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, "pareto3d")


def fig_pareto_f1f2():
    P, fifo = load_front()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    sc = ax.scatter(P[:, 0], P[:, 1], c=P[:, 2], cmap="viridis_r", s=18,
                    alpha=0.85, edgecolors="none")
    ax.scatter([fifo[0]], [fifo[1]], marker="*", s=180, c=RED, edgecolors="k",
               linewidths=0.5, zorder=5, label="FIFO baseline (dominated)")
    ax.set_xlabel(r"$f_1$ — unserved waiting load")
    ax.set_ylabel(r"$f_2$ — inter-country disparity (years)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$f_3$ — wasted visas")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25)
    save(fig, "pareto_f1f2")


def fig_hv_box():
    st = json.load(open(R / "stats_test.json"))
    mh = np.array(st["mohho_hv"]) / 1e6
    ng = np.array(st["nsga2_hv"]) / 1e6
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    parts = ax.violinplot([ng, mh], positions=[1, 2], showmeans=False,
                          showextrema=False, widths=0.8)
    for b, c in zip(parts["bodies"], [ORANGE, BLUE]):
        b.set_facecolor(c)
        b.set_alpha(0.30)
        b.set_edgecolor(c)
    bp = ax.boxplot([ng, mh], positions=[1, 2], widths=0.32, patch_artist=True,
                    showfliers=True, medianprops=dict(color="k", lw=1.2))
    for patch, c in zip(bp["boxes"], [ORANGE, BLUE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["NSGA-II", "MOHHO"])
    ax.set_ylabel(r"Hypervolume ($\times 10^{6}$)")
    p = st["p_one_sided"]
    a12 = st["A12"]
    ax.set_title(f"Mann--Whitney $p={p:.1e}$,  $A_{{12}}={a12:.2f}$", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "hv_box")


def fig_nsga_overlay():
    f = R / "nsga2_front.json"
    if not f.exists():
        print("skip nsga2_overlay (nsga2_front.json not ready)")
        return
    ng = np.array(json.load(open(f))["front"])
    P, fifo = load_front()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.scatter(ng[:, 0], ng[:, 1], s=16, c=ORANGE, alpha=0.6,
               label=f"NSGA-II ({len(ng)} sol.)", edgecolors="none")
    ax.scatter(P[:, 0], P[:, 1], s=16, c=BLUE, alpha=0.7,
               label=f"MOHHO ({len(P)} sol.)", edgecolors="none")
    ax.scatter([fifo[0]], [fifo[1]], marker="*", s=180, c=RED, edgecolors="k",
               linewidths=0.5, zorder=5, label="FIFO baseline")
    ax.set_xlabel(r"$f_1$ — unserved waiting load")
    ax.set_ylabel(r"$f_2$ — inter-country disparity (years)")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(alpha=0.25)
    save(fig, "nsga2_overlay")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print("saved", name)


if __name__ == "__main__":
    fig_convergence()
    fig_pareto3d()
    fig_pareto_f1f2()
    fig_hv_box()
    fig_nsga_overlay()
