"""
Permutation-aware NSGA-II (path to 9/10): closes the "NSGA-II only loses because
its real-coded operators are mismatched to random keys" objection. Here NSGA-II
operates DIRECTLY on permutations of the 105 groups with order crossover (OX) and
swap mutation, decoded by the SAME greedy decoder. If MOHHO still wins, the gap
is not merely an operator-representation mismatch.

Output: app/data/results/perm_nsga.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

from app.core.config import NUM_GROUPS
from app.core.decoder import decode
from app.core.problem import VisaProblem
from app.core.mohho import compute_hypervolume, crowding_distance, dominates
from compare_nsga2 import fast_nondominated_sort, tournament, nondominated

RESULTS = Path("app/data/results")
POP, GEN, RUNS = 50, 500, 30
PM = 0.3                       # per-individual swap-mutation probability


def eval_perm(perm, problem):
    alloc = decode(list(perm), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    return problem.evaluate(alloc)


def ox(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.choice(n, 2, replace=False))
    seg = set(p1[a:b + 1].tolist())
    child = -np.ones(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in seg]
    k = 0
    for i in list(range(b + 1, n)) + list(range(0, a)):
        child[i] = fill[k]; k += 1
    return child


def swap_mut(perm, rng):
    y = perm.copy()
    if rng.random() < PM:
        for _ in range(rng.integers(1, 4)):
            i, j = rng.integers(0, len(y), size=2)
            y[i], y[j] = y[j], y[i]
    return y


def run_perm_nsga(problem, seed):
    rng = np.random.default_rng(seed)
    pop = np.array([rng.permutation(NUM_GROUPS) for _ in range(POP)])
    fits = [eval_perm(pop[i], problem) for i in range(POP)]
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr):
                cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]
            p2 = pop[tournament(rank, cd, rng)]
            off.append(swap_mut(ox(p1, p2, rng), rng))
            if len(off) < POP:
                off.append(swap_mut(ox(p2, p1, rng), rng))
        off = np.array(off)
        off_fits = [eval_perm(off[i], problem) for i in range(POP)]
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
    fronts, _ = fast_nondominated_sort(fits)
    return [fits[i] for i in fronts[0]]


def main():
    problem = VisaProblem()
    t0 = time.time()
    hv, allp = [], []
    for s in range(1, RUNS + 1):
        front = run_perm_nsga(problem, s)
        hv.append(compute_hypervolume(front)); allp += front
        print(f"perm-NSGA seed {s:2d}: {len(front)} sols HV={hv[-1]:,.0f} ({time.time()-t0:.0f}s)")
    comb = nondominated(allp)
    mohho_hv = json.load(open(RESULTS / "stats_test.json"))["mohho_hv"]
    u, p = mannwhitneyu(mohho_hv, hv, alternative="greater")
    out = {
        "pop": POP, "gen": GEN, "runs": RUNS,
        "operators": "order crossover (OX) + swap mutation on permutations",
        "per_run_hv": hv,
        "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv)),
        "combined_front_size": len(comb),
        "combined_front_hv": compute_hypervolume(comb),
        "mohho_vs_permnsga_U": float(u),
        "mohho_vs_permnsga_p": float(p),
        "mohho_vs_permnsga_A12": float(u / (len(mohho_hv) * len(hv))),
        "elapsed_s": time.time() - t0,
    }
    json.dump(out, open(RESULTS / "perm_nsga.json", "w"), indent=2)
    print(f"\nperm-NSGA HV mean {out['hv_mean']:,.0f} +/- {out['hv_std']:,.0f} "
          f"| combined {out['combined_front_hv']:,.0f} ({out['combined_front_size']} sols)")
    print(f"MOHHO vs perm-NSGA: p={p:.2e}, A12={out['mohho_vs_permnsga_A12']:.3f}")


if __name__ == "__main__":
    main()
