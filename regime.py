"""
Structural-regime sweep (path to 9/10): does the result hold, and how does the
*value of the search* change, as the demand/supply ratio tightens?

The base instance has total demand ~6.2x supply, which the paper concedes makes
f1 nearly degenerate and the greedy decoder near-optimal. We build structurally
different instances by globally scaling demand n_g by alpha (recomputing the
spillover-adjusted category caps, per-country caps, and total demand), spanning
from the easy base regime down to a tight scarcity regime (~1.2x). For each we
run the full ablation and measure:
  - the f1 RANGE across the MOHHO front (degeneracy: does f1 become non-trivial?),
  - the SEARCH gap MOHHO-vs-random (does the swarm earn more when the problem is hard?),
  - MOHHO vs NSGA-II, and whether FIFO is dominated.

Output: app/data/results/regime.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

from app.core.config import V, LB, UB, NUM_GROUPS
from app.core.data import build_groups, compute_spillover, compute_country_caps
from app.core.problem import VisaProblem
from app.core.fifo import run_baseline
from app.core.mohho import run_mohho, compute_hypervolume, dominates
from compare_nsga2 import run_nsga2, nondominated
from controls import run_random

RESULTS = Path("app/data/results")
S = 10
SEEDS = list(range(42, 42 + S))
ALPHAS = [1.0, 0.5, 0.3, 0.2]      # demand scale -> ratios ~6.2, 3.1, 1.9, 1.2


def scaled_problem(alpha):
    p = VisaProblem.__new__(VisaProblem)
    groups = build_groups()
    for g in groups:
        g["n"] = max(1, int(round(g["n"] * alpha)))
    p.groups = groups
    p.category_caps = compute_spillover(groups)
    p.country_caps = compute_country_caps(groups)
    p.total_visas = V
    p.total_demand = sum(g["n"] for g in groups)
    p._groups_by_country = {}
    for g in groups:
        p._groups_by_country.setdefault(g["country"], []).append(g)
    p._country_w_max = {c: max(g["w"] for g in gs)
                        for c, gs in p._groups_by_country.items()}
    return p


def main():
    t0 = time.time()
    out = []
    for alpha in ALPHAS:
        prob = scaled_problem(alpha)
        ratio = prob.total_demand / V
        _, fifo_fit = run_baseline(prob)

        mohho_hv, mohho_all = [], []
        for s in SEEDS:
            _, fits, _ = run_mohho(prob, seed=s)
            mohho_hv.append(compute_hypervolume(fits)); mohho_all += fits
        rand_hv = [compute_hypervolume(run_random(prob, s)) for s in SEEDS]
        nsga_hv = [compute_hypervolume(run_nsga2(prob, s)) for s in range(1, S + 1)]

        front = np.array(nondominated(mohho_all))
        f1_range = float(front[:, 0].max() - front[:, 0].min())
        f2_range = float(front[:, 1].max() - front[:, 1].min())
        mohho_m, rand_m, nsga_m = (float(np.mean(mohho_hv)),
                                   float(np.mean(rand_hv)), float(np.mean(nsga_hv)))
        # at very tight ratios the fixed HV reference point can be non-dominating
        # (HV -> 0); flag and guard rather than crash.
        degenerate = min(mohho_m, rand_m, nsga_m) <= 0
        _, p_rand = mannwhitneyu(mohho_hv, rand_hv, alternative="greater")
        _, p_nsga = mannwhitneyu(mohho_hv, nsga_hv, alternative="greater")
        rec = {
            "alpha": alpha, "demand": prob.total_demand, "ratio": ratio,
            "f1_range": f1_range, "f2_range": f2_range,
            "mohho_hv": mohho_m, "random_hv": rand_m, "nsga2_hv": nsga_m,
            "degenerate_hv": degenerate,
            "swarm_gap_pct": (100 * (mohho_m - rand_m) / rand_m) if rand_m > 0 else None,
            "mohho_vs_nsga2_pct": (100 * (mohho_m - nsga_m) / nsga_m) if nsga_m > 0 else None,
            "p_mohho_vs_random": float(p_rand),
            "p_mohho_vs_nsga2": float(p_nsga),
            "fifo_dominated": bool(any(dominates(tuple(p), tuple(fifo_fit)) for p in front)),
            "front_size": len(front),
        }
        out.append(rec)
        sg = rec['swarm_gap_pct']; mn = rec['mohho_vs_nsga2_pct']
        print(f"alpha={alpha} ratio={ratio:.2f}x f1_range={f1_range:.4f} "
              f"swarm_gap={'n/a' if sg is None else f'{sg:.2f}%'} (p={p_rand:.1e}) "
              f"MOHHO>NSGA={'n/a' if mn is None else f'{mn:.1f}%'} "
              f"FIFOdom={rec['fifo_dominated']} degen={degenerate}")

    json.dump({"sweep": out, "seeds": S, "elapsed_s": time.time() - t0},
              open(RESULTS / "regime.json", "w"), indent=2)
    print(f"\ntotal {time.time()-t0:.0f}s -> regime.json")
    print("HARDNESS CHECK (does swarm gap grow as ratio shrinks?):")
    for r in out:
        sg = r['swarm_gap_pct']
        print(f"  ratio {r['ratio']:.2f}x -> f1_range {r['f1_range']:.4f}, "
              f"swarm gap {'n/a' if sg is None else f'{sg:+.2f}%'}")


if __name__ == "__main__":
    main()
