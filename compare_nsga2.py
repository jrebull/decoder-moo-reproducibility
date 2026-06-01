"""
Fair benchmark: MOHHO vs NSGA-II on the SAME problem, encoding (SPV, R^105),
greedy decoder, objectives (f1,f2,f3) and hypervolume reference point.
Only the search operators differ. Same protocol: 50 individuals, 500 generations,
30 independent runs (seeds 1-30).

Outputs:
  - app/data/results/nsga2_comparison.json   (metrics)
  - ../reporte_final/figures/nsga2_comparison.png   (overlaid fronts)
"""
import json, time, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.core.config import LB, UB, NUM_GROUPS
from app.core.problem import VisaProblem
from app.core.mohho import evaluate_hawk, compute_hypervolume, dominates, crowding_distance, HV_REF_POINT

RESULTS = Path("app/data/results")
FIGDIR = Path("../reporte_final/figures")
SEEDS = list(range(1, 31))
POP, GEN = 50, 500
ETA_C, ETA_M, PC = 20.0, 20.0, 0.9
PM = 1.0 / NUM_GROUPS

# ----------------------- NSGA-II core -----------------------
def fast_nondominated_sort(fits):
    n = len(fits)
    S = [[] for _ in range(n)]; ndom = [0]*n; rank = [0]*n; fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if dominates(fits[p], fits[q]): S[p].append(q)
            elif dominates(fits[q], fits[p]): ndom[p] += 1
        if ndom[p] == 0:
            rank[p] = 0; fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    rank[q] = i+1; nxt.append(q)
        i += 1; fronts.append(nxt)
    fronts.pop()
    return fronts, rank

def sbx(p1, p2, rng):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() <= 0.5 and abs(p1[i]-p2[i]) > 1e-12:
            u = rng.random()
            beta = (2*u)**(1/(ETA_C+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(ETA_C+1))
            c1[i] = 0.5*((1+beta)*p1[i] + (1-beta)*p2[i])
            c2[i] = 0.5*((1-beta)*p1[i] + (1+beta)*p2[i])
    return np.clip(c1, LB, UB), np.clip(c2, LB, UB)

def poly_mutate(x, rng):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < PM:
            u = rng.random()
            delta = (2*u)**(1/(ETA_M+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(ETA_M+1))
            y[i] = x[i] + delta*(UB-LB)
    return np.clip(y, LB, UB)

def tournament(rank, cd, rng):
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]: return a
    if rank[b] < rank[a]: return b
    return a if cd[a] >= cd[b] else b

def run_nsga2(problem, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LB, UB, size=(POP, NUM_GROUPS))
    fits = [evaluate_hawk(pop[i], problem)[1] for i in range(POP)]
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0]*POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr): cd[idx] = d[k]
        # offspring
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]; p2 = pop[tournament(rank, cd, rng)]
            if rng.random() <= PC: c1, c2 = sbx(p1, p2, rng)
            else: c1, c2 = p1.copy(), p2.copy()
            off.append(poly_mutate(c1, rng));
            if len(off) < POP: off.append(poly_mutate(c2, rng))
        off = np.array(off)
        off_fits = [evaluate_hawk(off[i], problem)[1] for i in range(POP)]
        # merge (mu+lambda) and select N by rank + crowding
        comb = np.vstack([pop, off]); comb_fits = fits + off_fits
        fronts, _ = fast_nondominated_sort(comb_fits)
        new_idx = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= POP:
                new_idx += fr
            else:
                d = crowding_distance([comb_fits[i] for i in fr])
                order = sorted(range(len(fr)), key=lambda k: d[k], reverse=True)
                new_idx += [fr[k] for k in order[:POP-len(new_idx)]]
                break
        pop = comb[new_idx]; fits = [comb_fits[i] for i in new_idx]
    # final non-dominated set
    fronts, _ = fast_nondominated_sort(fits)
    front_fits = [fits[i] for i in fronts[0]]
    return front_fits

# ----------------------- metrics -----------------------
def nondominated(points):
    pts = list({(round(p[0],6), round(p[1],6), round(p[2],6)) for p in points})
    keep = []
    for p in pts:
        if not any(dominates(q, p) for q in pts if q != p):
            keep.append(p)
    return keep

def normalize(points, lo, span):
    return np.array([[(p[m]-lo[m])/span[m] for m in range(3)] for p in points])

def igd(front, ref, lo, span):
    F = normalize(front, lo, span); Z = normalize(ref, lo, span)
    return float(np.mean([np.min(np.linalg.norm(F - z, axis=1)) for z in Z]))

def spacing(front, lo, span):
    F = normalize(front, lo, span)
    if len(F) < 2: return 0.0
    d = []
    for i in range(len(F)):
        dist = np.linalg.norm(F - F[i], axis=1); dist[i] = np.inf
        d.append(np.min(dist))
    d = np.array(d)
    return float(np.sqrt(np.sum((d.mean()-d)**2)/(len(d)-1)))

# ----------------------- run -----------------------
def main():
    problem = VisaProblem()
    summary = json.load(open(RESULTS/"summary.json"))
    # MOHHO combined front from csv
    mohho_front = []
    fifo = None
    for r in csv.DictReader(open(RESULTS/"pareto_front.csv")):
        pt = (float(r["f1"]), float(r["f2"]), float(r["f3"]))
        if r["type"] == "pareto": mohho_front.append(pt)
        else: fifo = pt
    print(f"MOHHO combined front: {len(mohho_front)} | FIFO {fifo}")

    # NSGA-II 30 runs
    nsga_runs, nsga_hv, all_nsga = [], [], []
    t0 = time.time()
    for s in SEEDS:
        ts = time.time()
        front = run_nsga2(problem, s)
        hv = compute_hypervolume(front)
        nsga_hv.append(hv); all_nsga += front
        print(f"  NSGA-II seed {s:2d}: {len(front):3d} sols, HV={hv:,.0f}  ({time.time()-ts:.1f}s)")
    nsga_front = nondominated(all_nsga)
    print(f"NSGA-II combined front: {len(nsga_front)} | total {time.time()-t0:.0f}s")

    def extremes(front):
        return {"best_f1": list(min(front, key=lambda p: p[0])),
                "best_f2": list(min(front, key=lambda p: p[1])),
                "best_f3": list(min(front, key=lambda p: p[2]))}

    # reference front Z = nondominated union of both
    Z = nondominated(mohho_front + nsga_front)
    allp = mohho_front + nsga_front + Z
    lo = [min(p[m] for p in allp) for m in range(3)]
    hi = [max(p[m] for p in allp) for m in range(3)]
    span = [max(hi[m]-lo[m], 1e-9) for m in range(3)]

    out = {
        "protocol": {"pop": POP, "gen": GEN, "runs": len(SEEDS),
                     "encoding": "SPV (R^105) + greedy decoder (identico a MOHHO)",
                     "operators": "SBX (eta=20) + mutacion polinomial (eta=20), pc=0.9, pm=1/d",
                     "hv_ref_point": list(HV_REF_POINT)},
        "mohho": {
            "hv_mean": summary["hv_stats"]["mean"], "hv_std": summary["hv_stats"]["std"],
            "combined_front_size": len(mohho_front),
            "combined_front_hv": compute_hypervolume(mohho_front),
            "igd": igd(mohho_front, Z, lo, span), "spacing": spacing(mohho_front, lo, span),
        },
        "nsga2": {
            "hv_mean": float(np.mean(nsga_hv)), "hv_std": float(np.std(nsga_hv)),
            "hv_min": float(np.min(nsga_hv)), "hv_max": float(np.max(nsga_hv)),
            "per_run_hv": nsga_hv,
            "combined_front_size": len(nsga_front),
            "combined_front_hv": compute_hypervolume(nsga_front),
            "igd": igd(nsga_front, Z, lo, span), "spacing": spacing(nsga_front, lo, span),
            "extremes": extremes(nsga_front),
        },
        "mohho_extremes": extremes(mohho_front),
        "reference_front_size": len(Z),
        "fifo": list(fifo),
    }
    json.dump(out, open(RESULTS/"nsga2_comparison.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("mohho","nsga2","reference_front_size")}, indent=2))

    # ---------- figure: f1-f2 overlay ----------
    FIGDIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.6))
    mh = np.array(mohho_front); ng = np.array(nsga_front)
    plt.scatter(ng[:,0], ng[:,1], s=14, c="#E67E22", alpha=0.55, label=f"NSGA-II ({len(nsga_front)} sol.)", edgecolors="none")
    plt.scatter(mh[:,0], mh[:,1], s=14, c="#2E86DE", alpha=0.7, label=f"MOHHO ({len(mohho_front)} sol.)", edgecolors="none")
    plt.scatter([fifo[0]],[fifo[1]], s=120, marker="*", c="#E74C3C", label="FIFO (baseline)", zorder=5, edgecolors="k", linewidths=0.4)
    plt.xlabel(r"$f_1$ — carga de espera (años)"); plt.ylabel(r"$f_2$ — disparidad entre países (años)")
    plt.title("Frente de Pareto combinado: MOHHO vs. NSGA-II")
    plt.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(FIGDIR/"nsga2_comparison.png", dpi=300)
    print("figure saved:", FIGDIR/"nsga2_comparison.png")

if __name__ == "__main__":
    main()
