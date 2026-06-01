"""
Generalization study (reviewer's highest-leverage request): does the qualitative
result -- feasibility-preserving decoder dominates search; MOHHO > random restart
> NSGA-II; FIFO strictly dominated -- hold beyond the single base instance?

We build K perturbed-demand instances (each group's backlog n_g jitterred by a
fixed-seed uniform +/-20%, with spillover-adjusted category caps and per-country
caps recomputed accordingly) and run the full ablation on each.

Output: app/data/results/second_instance.json
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
K = 5                       # perturbed instances
S = 10                      # seeds per method per instance
SEEDS = list(range(42, 42 + S))
JITTER = (0.8, 1.2)


def perturbed_problem(pseed):
    p = VisaProblem.__new__(VisaProblem)
    groups = build_groups()
    rng = np.random.default_rng(pseed)
    for g in groups:
        g["n"] = max(1, int(round(g["n"] * rng.uniform(*JITTER))))
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


def fifo_dominated_by(front, fifo_fit):
    return any(dominates(tuple(p), tuple(fifo_fit)) for p in front)


def main():
    t0 = time.time()
    instances = []
    for k in range(1, K + 1):
        pseed = 1000 + k
        prob = perturbed_problem(pseed)
        _, fifo_fit = run_baseline(prob)

        mohho_hv, mohho_all = [], []
        for s in SEEDS:
            _, fits, _ = run_mohho(prob, seed=s)
            mohho_hv.append(compute_hypervolume(fits)); mohho_all += fits
        rand_hv, rand_all = [], []
        for s in SEEDS:
            af = run_random(prob, s)
            rand_hv.append(compute_hypervolume(af)); rand_all += af
        nsga_hv = []
        for s in range(1, S + 1):
            front = run_nsga2(prob, s)
            nsga_hv.append(compute_hypervolume(front))

        mohho_front = nondominated(mohho_all)
        mw_u, mw_p = mannwhitneyu(np.array(mohho_hv), np.array(nsga_hv),
                                  alternative="greater")
        rec = {
            "instance": k, "perturb_seed": pseed,
            "total_demand": prob.total_demand,
            "fifo": list(fifo_fit),
            "mohho_hv_mean": float(np.mean(mohho_hv)),
            "random_hv_mean": float(np.mean(rand_hv)),
            "nsga2_hv_mean": float(np.mean(nsga_hv)),
            "mohho_front_size": len(mohho_front),
            "fifo_dominated": bool(fifo_dominated_by(mohho_front, fifo_fit)),
            "mohho_gt_random": float(np.mean(mohho_hv)) > float(np.mean(rand_hv)),
            "random_gt_nsga2": float(np.mean(rand_hv)) > float(np.mean(nsga_hv)),
            "mohho_gt_nsga2": float(np.mean(mohho_hv)) > float(np.mean(nsga_hv)),
            "mohho_vs_nsga2_p": float(mw_p),
            "mohho_vs_nsga2_A12": float(mw_u / (len(mohho_hv) * len(nsga_hv))),
        }
        instances.append(rec)
        print(f"inst {k} (demand {prob.total_demand:,}): "
              f"MOHHO {rec['mohho_hv_mean']:,.0f} | random {rec['random_hv_mean']:,.0f} "
              f"| NSGA {rec['nsga2_hv_mean']:,.0f} | FIFO dominated {rec['fifo_dominated']} "
              f"| M>R {rec['mohho_gt_random']} R>N {rec['random_gt_nsga2']} "
              f"| p={rec['mohho_vs_nsga2_p']:.1e}")

    summary = {
        "n_instances": K, "seeds_per_method": S, "jitter": JITTER,
        "fifo_dominated_all": all(r["fifo_dominated"] for r in instances),
        "mohho_gt_random_count": sum(r["mohho_gt_random"] for r in instances),
        "random_gt_nsga2_count": sum(r["random_gt_nsga2"] for r in instances),
        "mohho_gt_nsga2_count": sum(r["mohho_gt_nsga2"] for r in instances),
        "max_p_mohho_vs_nsga2": max(r["mohho_vs_nsga2_p"] for r in instances),
    }
    out = {"instances": instances, "summary": summary, "elapsed_s": time.time() - t0}
    json.dump(out, open(RESULTS / "second_instance.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"total {out['elapsed_s']:.0f}s -> second_instance.json")


if __name__ == "__main__":
    main()
