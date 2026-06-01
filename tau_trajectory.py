"""
REVIEWER REBUTTAL EXPERIMENT (MIT critique: tau was measured on uniform-random
parents, not on the parents an actual search produces). We re-measure the SBX
SPV-order-preservation tau ALONG A REAL NSGA-II TRAJECTORY: at several iterations
we snapshot the live population, draw parent pairs by the algorithm's own binary
tournament (rank+crowding), apply SBX, and record tau(parent SPV order -> child
SPV order). Converged populations are clustered, so |p1-p2| shrinks and SBX is, if
anything, even closer to identity -- this checks whether the near-identity finding
holds on-trajectory, not just on random draws. We also report the HHO contraction
operator's tau on the same on-trajectory parents for contrast.

Output: app/data/results/tau_trajectory.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

from app.core.config import NUM_GROUPS, LB, UB
from app.core.problem import VisaProblem
from app.core.mohho import evaluate_hawk
from app.core import hho
from compare_nsga2 import (sbx, fast_nondominated_sort, crowding_distance,
                           tournament, POP, GEN)

RESULTS = Path("app/data/results")
SNAPSHOTS = [1, 10, 50, 100, 250, 499]
PAIRS_PER_SNAPSHOT = 400
SEED = 42


def spv_tau(parent, child):
    rp = np.argsort(np.argsort(parent)); rc = np.argsort(np.argsort(child))
    t, _ = kendalltau(rp, rc)
    return 0.0 if np.isnan(t) else t


def main():
    p = VisaProblem()
    rng = np.random.default_rng(SEED)
    pop = rng.uniform(LB, UB, size=(POP, NUM_GROUPS))
    fits = [evaluate_hawk(pop[i], p)[1] for i in range(POP)]
    snap = {}

    def measure(it):
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr):
                cd[idx] = d[k]
        sbx_t, hho_t, gaps = [], [], []
        for _ in range(PAIRS_PER_SNAPSHOT):
            a = pop[tournament(rank, cd, rng)]
            b = pop[tournament(rank, cd, rng)]
            gaps.append(float(np.mean(np.abs(a - b))))
            c1, _ = sbx(a, b, rng)
            sbx_t.append(spv_tau(a, c1))
            # HHO contraction toward 'b' as a leader, mid-energy
            hc = hho.op3_soft_siege(a, b, 0.7, rng)
            hho_t.append(spv_tau(a, hc))
        snap[it] = {
            "sbx_tau_mean": float(np.mean(sbx_t)), "sbx_tau_std": float(np.std(sbx_t)),
            "hho_tau_mean": float(np.mean(hho_t)), "hho_tau_std": float(np.std(hho_t)),
            "mean_parent_gap": float(np.mean(gaps)),
        }
        print(f"  iter {it:4d}: SBX tau={np.mean(sbx_t):.3f}  HHO tau={np.mean(hho_t):+.3f}  "
              f"parent|gap|={np.mean(gaps):.3f}")

    print(f"On-trajectory SPV-order preservation (NSGA-II, seed {SEED}, {PAIRS_PER_SNAPSHOT} pairs/snapshot):")
    for it in range(GEN):
        if it in SNAPSHOTS:
            measure(it)
        # advance one NSGA-II generation
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr):
                cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]; p2 = pop[tournament(rank, cd, rng)]
            if rng.random() <= 0.9:
                c1, c2 = sbx(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            from compare_nsga2 import poly_mutate
            off.append(poly_mutate(c1, rng))
            if len(off) < POP:
                off.append(poly_mutate(c2, rng))
        off = np.array(off)
        off_fits = [evaluate_hawk(off[i], p)[1] for i in range(POP)]
        comb = np.vstack([pop, off]); comb_fits = fits + off_fits
        fronts, _ = fast_nondominated_sort(comb_fits)
        new_idx = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= POP:
                new_idx += fr
            else:
                d = crowding_distance([comb_fits[i] for i in fr])
                order = sorted(range(len(fr)), key=lambda k: d[k], reverse=True)
                new_idx += [fr[k] for k in order[:POP - len(new_idx)]]
                break
        pop = comb[new_idx]; fits = [comb_fits[i] for i in new_idx]

    sbx_all = [v["sbx_tau_mean"] for v in snap.values()]
    out = {"seed": SEED, "snapshots": SNAPSHOTS, "pairs_per_snapshot": PAIRS_PER_SNAPSHOT,
           "per_iter": snap,
           "sbx_tau_min": float(min(sbx_all)), "sbx_tau_max": float(max(sbx_all)),
           "sbx_tau_overall": float(np.mean(sbx_all))}
    json.dump(out, open(RESULTS / "tau_trajectory.json", "w"), indent=2)
    print(f"\nSBX on-trajectory tau stays in [{min(sbx_all):.3f}, {max(sbx_all):.3f}] "
          f"across the whole run (mean {np.mean(sbx_all):.3f}).")
    print("saved tau_trajectory.json")


if __name__ == "__main__":
    main()
