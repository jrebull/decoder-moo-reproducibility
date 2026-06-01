"""Impressive 3D Pareto front -> reporte_final/figures/fig_pareto3d.png"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.ticker import FuncFormatter
from pathlib import Path

FIG = Path("../reporte_final/figures")
pts, fifo = [], None
for r in csv.DictReader(open("app/data/results/pareto_front.csv")):
    p = (float(r["f1"]), float(r["f2"]), float(r["f3"]))
    if r["type"] == "pareto": pts.append(p)
    else: fifo = p
P = np.array(pts)
f1, f2, f3 = P[:,0], P[:,1], P[:,2]

BG = "#0e1420"; INK = "#e6edf6"; SUB = "#9fb0c8"
plt.rcParams.update({"text.color":INK,"axes.labelcolor":INK,
                     "xtick.color":SUB,"ytick.color":SUB})
fig = plt.figure(figsize=(9.0, 6.6)); fig.patch.set_facecolor(BG)
ax = fig.add_subplot(111, projection="3d"); ax.set_facecolor(BG)

# scatter coloreado por f3 (desperdicio)
sc = ax.scatter(f1, f2, f3, c=f3, cmap="plasma", s=26, alpha=0.92,
                edgecolors="white", linewidths=0.15, depthshade=True)

# punto FIFO (dominado)
ax.scatter([fifo[0]],[fifo[1]],[fifo[2]], marker="*", s=420, c="#ff3b4e",
           edgecolors="white", linewidths=1.0, label="FIFO (sistema actual)", zorder=10)

# soluciones extremas anotadas
def mark(idx, txt, dz=900):
    p = P[idx]; ax.scatter(*p, marker="o", s=90, facecolors="none",
                           edgecolors="#FCEFC7", linewidths=1.6, zorder=9)
    ax.text(p[0], p[1], p[2]+dz, txt, color="#FCEFC7", fontsize=8, weight="bold")
mark(int(np.argmin(f1)), "min $f_1$")
mark(int(np.argmin(f2)), "min $f_2$")
mark(int(np.argmin(f3)), "min $f_3$")

ax.set_xlabel("\n$f_1$  carga de espera", fontsize=10)
ax.set_ylabel("\n$f_2$  disparidad (años)", fontsize=10)
ax.set_zlabel("\n$f_3$  desperdicio (visas)", fontsize=10)
ax.zaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v):,}"))
ax.tick_params(labelsize=7.5)
# panes oscuros + grid sutil
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.set_pane_color((0.09,0.12,0.18,1.0))
    axis.pane.set_edgecolor((1,1,1,0.08))
    axis._axinfo["grid"]["color"] = (1,1,1,0.07)
ax.view_init(elev=20, azim=-58)
ax.set_title("Frente de Pareto combinado en 3D — 406 soluciones no dominadas\n"
             "(color = desperdicio $f_3$; el FIFO queda por encima del frente, dominado)",
             color=INK, fontsize=12.5, weight="bold", pad=4)
cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
cb.set_label("$f_3$ — visas desperdiciadas", color=SUB, fontsize=8.5)
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{int(v):,}"))
cb.ax.tick_params(colors=SUB, labelsize=7); cb.outline.set_edgecolor((1,1,1,0.15))
ax.legend(loc="upper left", fontsize=9, facecolor="#172033", edgecolor="#2a3650", labelcolor=INK)
plt.tight_layout()
plt.savefig(FIG/"fig_pareto3d.png", dpi=300, facecolor=BG, bbox_inches="tight")
print("saved fig_pareto3d.png  | pts:", len(pts), "| FIFO:", fifo)
