"""
Representation-matched MOHHO (the 'supreme' move): the ablation/perm-NSGA result
shows the representation-operator match dominates. Guided by that diagnosis, we
equip Harris Hawks Optimization with PERMUTATION-NATIVE operators and let the
escape energy E schedule the perturbation strength and the pull toward the
archive leader ('rabbit'):

  |E| >= 1  (exploration): recombine with a random hawk, or heavily shuffle.
  0.5<=|E|<1 (soft besiege): order-crossover toward the rabbit + several swaps.
  |E| < 0.5 (hard besiege): inherit a larger segment from the rabbit + few swaps.
  Levy dive: occasional segment reversal (2-opt-like) for diversification.

Same feasibility-preserving greedy decoder, same external crowding-pruned archive,
same 50x500 budget and 30 seeds as MOHHO. If this tops permutation-NSGA-II, the
swarm's advantage is restored once its operators match the representation.

Output: app/data/results/discrete_mohho.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

from app.core.config import NUM_GROUPS
from app.core.problem import VisaProblem
from app.core.decoder import decode
from app.core.mohho import (dominates, update_archive, compute_hypervolume,
                            select_leader)
from compare_nsga2 import nondominated

RESULTS = Path("app/data/results")
POP, GEN, ARCH = 50, 500, 100
SEEDS = list(range(1, 31))


def eval_perm(perm, problem):
    alloc = decode(list(perm), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    return problem.evaluate(alloc)


def ox(p1, p2, rng):
    """Order crossover: keep a segment of p1, fill the rest in p2's order."""
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


def nswaps(perm, k, rng):
    y = perm.copy()
    for _ in range(max(1, k)):
        i, j = rng.integers(0, len(y), size=2)
        y[i], y[j] = y[j], y[i]
    return y


def reverse_seg(perm, rng):
    y = perm.copy()
    a, b = sorted(rng.choice(len(y), 2, replace=False))
    y[a:b + 1] = y[a:b + 1][::-1]
    return y


def step(pop, i, fit_i, ap, af, t, rng, problem):
    e = 2 * (2 * rng.random() - 1) * (1 - t / GEN)
    absE = abs(e)
    rabbit = select_leader(ap, af, rng)
    if absE >= 1:                                   # exploration
        if rng.random() < 0.5:
            j = rng.integers(POP)
            child = ox(pop[j], rabbit, rng)
        else:
            child = nswaps(pop[i], int(rng.integers(NUM_GROUPS // 4,
                                                    NUM_GROUPS // 2)), rng)
    else:                                           # exploitation toward rabbit
        # hard besiege (small |E|) inherits a larger segment & perturbs less
        child = ox(rabbit, pop[i], rng)
        k = max(1, int(round(absE * (NUM_GROUPS // 6))))
        child = nswaps(child, k, rng)
        if rng.random() < 0.5:                      # Levy-like dive
            child = reverse_seg(child, rng)
    fit = eval_perm(child, problem)
    update_archive(ap, af, child, fit, ARCH, rng)
    if dominates(fit, fit_i):
        return child, fit
    return pop[i], fit_i


def run_discrete_mohho(problem, seed):
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(NUM_GROUPS) for _ in range(POP)]
    fits = [eval_perm(pop[i], problem) for i in range(POP)]
    ap, af = [], []
    for i in range(POP):
        update_archive(ap, af, pop[i], fits[i], ARCH, rng)
    for t in range(GEN):
        for i in range(POP):
            pop[i], fits[i] = step(pop, i, fits[i], ap, af, t, rng, problem)
    return af


def main():
    problem = VisaProblem()
    t0 = time.time()
    hv, allp = [], []
    for s in SEEDS:
        af = run_discrete_mohho(problem, s)
        hv.append(compute_hypervolume(af)); allp += af
        print(f"discrete-MOHHO seed {s}: {len(af)} sols HV={hv[-1]:,.0f} ({time.time()-t0:.0f}s)")
    comb = nondominated(allp)

    st = json.load(open(RESULTS / "stats_test.json"))
    mohho_hv = st["mohho_hv"]                       # paired (same seeds 1-30)
    perm = json.load(open(RESULTS / "perm_nsga.json"))["per_run_hv"]  # independent

    w_s, w_p = wilcoxon(np.array(hv), np.array(mohho_hv), alternative="greater")
    u_v, u_p = mannwhitneyu(np.array(hv), np.array(perm), alternative="greater")
    out = {
        "pop": POP, "gen": GEN, "archive": ARCH, "seeds": SEEDS,
        "per_run_hv": hv,
        "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv)),
        "hv_median": float(np.median(hv)),
        "combined_front_size": len(comb),
        "combined_front_hv": compute_hypervolume(comb),
        "vs_mohho_paired_wilcoxon_p": float(w_p),
        "vs_mohho_better_count": int(np.sum(np.array(hv) > np.array(mohho_hv))),
        "vs_permnsga_mannwhitney_p": float(u_p),
        "vs_permnsga_A12": float(u_v / (len(hv) * len(perm))),
        "elapsed_s": time.time() - t0,
    }
    json.dump(out, open(RESULTS / "discrete_mohho.json", "w"), indent=2)
    print("\n=== DISCRETE-MOHHO ===")
    print(f"HV mean {out['hv_mean']:,.0f} +/- {out['hv_std']:,.0f} | "
          f"combined {out['combined_front_hv']:,.0f} ({out['combined_front_size']} sols)")
    print(f"vs MOHHO (paired): p={w_p:.2e}, better {out['vs_mohho_better_count']}/30 "
          f"(MOHHO mean {np.mean(mohho_hv):,.0f})")
    print(f"vs perm-NSGA-II: p={u_p:.2e}, A12={out['vs_permnsga_A12']:.3f} "
          f"(perm-NSGA mean {np.mean(perm):,.0f})")
    print(f"total {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
