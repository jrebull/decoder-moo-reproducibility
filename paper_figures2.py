"""More impactful figures for the paper:
   pareto3d_v2.pdf      -> enhanced 3D front with a drop-line proving FIFO is dominated
   country_impact.pdf   -> policy output: visas gained/lost per country vs FIFO (diverging)
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.ticker import FuncFormatter

from app.core.problem import VisaProblem
from app.core.fifo import run_baseline
from app.core.mohho import run_mohho, evaluate_hawk

plt.rcParams.update({"font.family": "serif", "font.size": 9, "savefig.bbox": "tight"})
FIG = Path("../MICAI/figures")
R = Path("app/data/results")
BLUE, RED, GREEN, GREY = "#2E86DE", "#E74C3C", "#27AE60", "#9AA3AF"


def load_front():
    P, fifo = [], None
    for r in csv.DictReader(open(R / "pareto_front.csv")):
        pt = (float(r["f1"]), float(r["f2"]), float(r["f3"]))
        (P.append(pt) if r["type"] == "pareto" else None)
        if r["type"] != "pareto":
            fifo = pt
    return np.array(P), fifo


def fig_pareto3d_v2():
    P, fifo = load_front()
    fig = plt.figure(figsize=(6.0, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    # faint projection on the floor (z=0 plane) to add depth
    zfloor = P[:, 2].min()
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), zfloor), s=6, c=GREY,
               alpha=0.12, edgecolors="none")
    sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=P[:, 2], cmap="viridis_r",
                    s=22, alpha=0.92, edgecolors="none", depthshade=True)
    # FIFO star + drop-line down to the front's f3 level -> visualizes domination
    ax.scatter([fifo[0]], [fifo[1]], [fifo[2]], marker="*", s=320, c=RED,
               edgecolors="k", linewidths=0.6, label="FIFO baseline (dominated)")
    ax.plot([fifo[0], fifo[0]], [fifo[1], fifo[1]], [fifo[2], zfloor],
            color=RED, ls=":", lw=1.3)
    for m in range(3):
        e = P[np.argmin(P[:, m])]
        ax.scatter([e[0]], [e[1]], [e[2]], s=70, facecolors="none",
                   edgecolors="k", linewidths=1.1)
    ax.set_xlabel(r"$f_1$  waiting load", labelpad=3)
    ax.set_ylabel(r"$f_2$  disparity (yr)", labelpad=3)
    ax.set_zlabel(r"$f_3$  waste (visas)", labelpad=3)
    ax.view_init(elev=18, azim=-66)
    ax.xaxis.pane.set_alpha(0.04); ax.yaxis.pane.set_alpha(0.04); ax.zaxis.pane.set_alpha(0.04)
    cb = fig.colorbar(sc, ax=ax, shrink=0.58, pad=0.09)
    cb.set_label(r"$f_3$ (wasted visas)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "pareto3d_v2.pdf"); fig.savefig(FIG / "pareto3d_v2.png", dpi=200)
    plt.close(fig); print("saved pareto3d_v2")


def fig_country_impact():
    problem = VisaProblem()
    fifo_alloc, _ = run_baseline(problem)
    fifo_c = {}
    for g in problem.groups:
        fifo_c[g["country"]] = fifo_c.get(g["country"], 0) + fifo_alloc[g["index"]]
    # one MOHHO run; pick a balanced full-utilization (f3=0) knee solution
    pos, fit, _ = run_mohho(problem, seed=42)
    f = np.array(fit)
    zero_waste = [i for i in range(len(f)) if f[i, 2] == 0]
    pool = zero_waste if zero_waste else list(range(len(f)))
    sub = f[pool]
    n1 = (sub[:, 0] - sub[:, 0].min()) / (np.ptp(sub[:, 0]) + 1e-9)
    n2 = (sub[:, 1] - sub[:, 1].min()) / (np.ptp(sub[:, 1]) + 1e-9)
    knee = pool[int(np.argmin(n1 + n2))]            # balanced f1/f2 at zero waste
    alloc, fsel = evaluate_hawk(pos[knee], problem)
    moh_c = {}
    for g in problem.groups:
        moh_c[g["country"]] = moh_c.get(g["country"], 0) + alloc[g["index"]]

    EN = {"India": "India", "China": "China", "Filipinas": "Philippines",
          "Mexico": "Mexico", "Afganistan": "Afghanistan", "Irak": "Iraq",
          "Corea del Sur": "South Korea", "Pakistan": "Pakistan", "Iran": "Iran",
          "Taiwan": "Taiwan", "Brasil": "Brazil", "Canada": "Canada",
          "Reino Unido": "United Kingdom", "Nigeria": "Nigeria", "Japon": "Japan",
          "Bangladesh": "Bangladesh", "Colombia": "Colombia", "Alemania": "Germany",
          "Vietnam": "Vietnam", "Etiopia": "Ethiopia", "Resto del Mundo": "Rest of World"}
    countries = sorted(moh_c, key=lambda c: moh_c[c] - fifo_c.get(c, 0))
    delta = [moh_c[c] - fifo_c.get(c, 0) for c in countries]
    y = np.arange(len(countries))
    colors = [BLUE if d >= 0 else RED for d in delta]
    fig, ax = plt.subplots(figsize=(5.4, 6.0))
    ax.barh(y, delta, color=colors, alpha=0.85, edgecolor="k", linewidth=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([EN[c] for c in countries], fontsize=8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):+,}"))
    ax.set_xlim(min(delta) * 1.35, max(delta) * 1.18)
    ax.set_xlabel("Visas reallocated vs. FIFO  (gained $+$ / lost $-$)")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for yi, d in zip(y, delta):
        ax.text(d + (max(delta) * 0.012 if d >= 0 else min(delta) * 0.012), yi,
                f"{d:+,}", va="center", ha="left" if d >= 0 else "right",
                fontsize=6.6, color="#333")
    fig.tight_layout()
    fig.savefig(FIG / "country_impact.pdf"); fig.savefig(FIG / "country_impact.png", dpi=200)
    plt.close(fig)
    print(f"saved country_impact | selected policy f=({fsel[0]:.3f},{fsel[1]:.3f},{fsel[2]:.0f}) "
          f"visas={sum(moh_c.values()):,} vs FIFO {sum(fifo_c.values()):,}")


if __name__ == "__main__":
    fig_pareto3d_v2()
    fig_country_impact()
