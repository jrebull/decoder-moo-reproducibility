"""High-impact RESULTS figures -> reporte_final/figures/
   fig_scenarios.png  : radar de los 5 escenarios de política (salida del modelo)
   fig_entregable.png : asignación recomendada por país (lo que recibe el usuario)
"""
import json, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
from app.core.problem import VisaProblem
from app.core.fifo import run_baseline
from app.core.mohho import run_mohho, evaluate_hawk

FIG = Path("../reporte_final/figures")
R = Path("app/data/results")
thousands = FuncFormatter(lambda x, _: f"{int(x):,}")

summary = json.load(open(R/"summary.json"))
pareto = []
for row in csv.DictReader(open(R/"pareto_front.csv")):
    if row["type"] == "pareto":
        pareto.append((float(row["f1"]), float(row["f2"]), float(row["f3"])))

def knee(front):
    pts = sorted(front, key=lambda p: p[0])
    f1 = np.array([p[0] for p in pts]); f2 = np.array([p[1] for p in pts])
    n1 = (f1-f1.min())/(f1.max()-f1.min()); n2 = (f2-f2.min())/(f2.max()-f2.min())
    p1, p2 = np.array([n1[0],n2[0]]), np.array([n1[-1],n2[-1]])
    v = p2-p1; v = v/np.linalg.norm(v)
    d = [np.linalg.norm((np.array([n1[i],n2[i]])-p1) - np.dot(np.array([n1[i],n2[i]])-p1, v)*v) for i in range(len(pts))]
    return pts[int(np.argmax(d))]

kn = knee(pareto)
SC = {  # nombre: (f1, f2, f3)
    "Humanitario":     tuple(summary["best_f1"]),
    "Equilibrio":      kn,
    "Equidad":         tuple(summary["best_f2"]),
    "Máx. utilización":tuple(summary["best_f3"]),
    "FIFO (actual)":   (summary["baseline"]["f1"], summary["baseline"]["f2"], summary["baseline"]["f3"]),
}
print("Escenarios:")
for k,v in SC.items(): print(f"  {k:18s} f1={v[0]:.4f} f2={v[1]:.4f} f3={v[2]:.0f}")

COL = {"Humanitario":"#2E86DE","Equilibrio":"#8E44AD","Equidad":"#16A085",
       "Máx. utilización":"#E67E22","FIFO (actual)":"#C0392B"}

# ---------- 1) RADAR de escenarios ----------
labels = [r"$f_1$  espera"+"\n(menor = mejor)", r"$f_2$  disparidad"+"\n(menor = mejor)", r"$f_3$  desperdicio"+"\n(menor = mejor)"]
arr = {k: np.array(v, float) for k,v in SC.items()}
allv = np.array(list(arr.values()))
lo, hi = allv.min(0), allv.max(0)
# quality in [0,1], 1 = best (menor objetivo)
qual = {k: (hi-arr[k])/(hi-lo+1e-9) for k in arr}
ang = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist(); ang += ang[:1]
fig, ax = plt.subplots(figsize=(7.4,5.2), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks([0.25,0.5,0.75,1.0]); ax.set_yticklabels(["25%","50%","75%","mejor"], fontsize=7, color="#777")
ax.set_ylim(0,1.0)
for k in SC:
    vals = qual[k].tolist(); vals += vals[:1]
    lw = 3.2 if k=="Equilibrio" else (2.6 if k=="FIFO (actual)" else 2.0)
    ls = "--" if k=="FIFO (actual)" else "-"
    ax.plot(ang, vals, lw=lw, ls=ls, color=COL[k], label=k, zorder=4 if k=="Equilibrio" else 3)
    ax.fill(ang, vals, color=COL[k], alpha=0.10)
ax.set_title("Los cinco escenarios de política: perfil de calidad por objetivo",
             fontsize=12.5, weight="bold", pad=22)
ax.legend(loc="upper right", bbox_to_anchor=(1.34,1.12), fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig(FIG/"fig_scenarios.png", dpi=300, bbox_inches="tight")
plt.close(); print("saved fig_scenarios.png")

# ---------- 2) ENTREGABLE: asignación recomendada por país (Equilibrio) ----------
problem = VisaProblem()
fifo_alloc, fifo_fit = run_baseline(problem)
fifo_country = {}
for g in problem.groups:
    fifo_country[g["country"]] = fifo_country.get(g["country"],0)+fifo_alloc[g["index"]]
# run one MOHHO, pick solution closest to knee
pos, fit, _ = run_mohho(problem, seed=42)
fn = np.array([(hi-np.array(f))/(hi-lo+1e-9) for f in fit])  # not used; pick by distance to knee in raw norm
# pick by f1-f2 distance to the knee (así se define el "equilibrio")
tgt = (np.array(kn)-lo)/(hi-lo+1e-9)
dist = [np.hypot(((np.array(f)-lo)/(hi-lo+1e-9))[0]-tgt[0],
                 ((np.array(f)-lo)/(hi-lo+1e-9))[1]-tgt[1]) for f in fit]
best = int(np.argmin(dist))
alloc,_ = evaluate_hawk(pos[best], problem)
moh_country = {}
for g in problem.groups:
    moh_country[g["country"]] = moh_country.get(g["country"],0)+alloc[g["index"]]
fsel = fit[best]
print(f"Entregable (Equilibrio elegido): f1={fsel[0]:.3f} f2={fsel[1]:.3f} f3={fsel[2]:.0f}  visas={int(sum(moh_country.values())):,}")

# top countries by MOHHO allocation
order = sorted(moh_country, key=lambda c: moh_country[c], reverse=True)[:14][::-1]
y = np.arange(len(order))
moh = [moh_country[c] for c in order]; fif = [fifo_country.get(c,0) for c in order]
fig, ax = plt.subplots(figsize=(8.6,5.6))
ax.barh(y+0.2, moh, height=0.4, color="#2E86DE", label="Política recomendada (Equilibrio)")
ax.barh(y-0.2, fif, height=0.4, color="#B0B7C3", label="Sistema actual (FIFO)")
ax.set_yticks(y); ax.set_yticklabels(order, fontsize=9)
ax.xaxis.set_major_formatter(thousands)
ax.set_xlabel("Visas asignadas")
ax.set_title("Entregable al tomador de decisiones: asignación de visas por país\n"
             f"(escenario Equilibrio · $f_1$={fsel[0]:.2f}, $f_2$={fsel[1]:.2f} años, "
             f"{int(sum(moh_country.values())):,}/140,000 visas usadas)",
             fontsize=11.5, weight="bold")
ax.legend(loc="lower right", fontsize=9, frameon=True)
ax.grid(axis="x", alpha=0.25); ax.spines[["top","right"]].set_visible(False)
for yi, mv in zip(y, moh):
    ax.text(mv+max(moh)*0.01, yi+0.2, f"{int(mv):,}", va="center", fontsize=7.5, color="#1B3B6F")
plt.tight_layout()
plt.savefig(FIG/"fig_entregable.png", dpi=300, bbox_inches="tight")
plt.close(); print("saved fig_entregable.png")
