"""
Permutation MOEA/D (decomposition paradigm) -- the third metaheuristic FAMILY on
the ladder, requested to test whether 'the family is second-order' holds across
paradigms (dominance/archive: NSGA-II & HHO; decomposition: MOEA/D). Same greedy
decoder, permutation-native operators (OX + swap), matched 25k-eval budget, 30
seeds. Tchebycheff aggregation on objectives normalized by the HV reference point.

Output: app/data/results/perm_moead.json
"""
import json, time, itertools
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

from app.core.config import NUM_GROUPS
from app.core.problem import VisaProblem
from app.core.decoder import decode
from app.core.mohho import compute_hypervolume, dominates
from compare_nsga2 import nondominated

RESULTS = Path("app/data/results")
H = 8                      # simplex-lattice divisions -> C(H+2,2)=45 weight vectors
T = 10                     # neighborhood size
SCALE = np.array([10.0, 16.0, 50000.0])   # HV reference point -> objective scaling
BUDGET = 25_000


def eval_perm(perm, problem):
    alloc = decode(list(perm), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    return np.array(problem.evaluate(alloc), dtype=float)


def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    seg = set(p1[a:b + 1].tolist()); child = -np.ones(n, dtype=int)
    child[a:b + 1] = p1[a:b + 1]; fill = [g for g in p2 if g not in seg]; k = 0
    for i in list(range(b + 1, n)) + list(range(0, a)):
        child[i] = fill[k]; k += 1
    return child


def swap_mut(perm, rng, pm=0.3):
    y = perm.copy()
    if rng.random() < pm:
        for _ in range(rng.integers(1, 4)):
            i, j = rng.integers(0, len(y), size=2); y[i], y[j] = y[j], y[i]
    return y


def weights_lattice(h):
    w = [np.array(c, float) / h
         for c in itertools.product(range(h + 1), repeat=3) if sum(c) == h]
    return np.array(w) + 1e-6


def tcheby(fn, lam, z):
    return np.max(lam * (fn - z))


def run_perm_moead(problem, seed):
    rng = np.random.default_rng(seed)
    W = weights_lattice(H)
    N = len(W)
    gens = max(1, BUDGET // N)
    # neighborhoods
    B = [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T] for i in range(N)]
    pop = [rng.permutation(NUM_GROUPS) for _ in range(N)]
    fit = [eval_perm(pop[i], problem) for i in range(N)]
    fn = [f / SCALE for f in fit]
    z = np.min(np.array(fn), axis=0)
    ap, af = [], []
    for i in range(N):
        _update_archive(ap, af, pop[i], tuple(fit[i]))
    for _ in range(gens):
        for i in range(N):
            k, l = rng.choice(B[i], 2, replace=False)
            child = swap_mut(ox(pop[k], pop[l], rng), rng)
            cf = eval_perm(child, problem); cfn = cf / SCALE
            z = np.minimum(z, cfn)
            for j in B[i]:
                if tcheby(cfn, W[j], z) <= tcheby(fn[j], W[j], z):
                    pop[j] = child; fit[j] = cf; fn[j] = cfn
            _update_archive(ap, af, child, tuple(cf))
    return af


def _update_archive(ap, af, pos, fit, cap=100):
    for f in af:
        if dominates(f, fit) or f == fit:
            return
    keep = [i for i, f in enumerate(af) if not dominates(fit, f)]
    af[:] = [af[i] for i in keep]
    ap[:] = [ap[i] for i in keep]
    af.append(fit); ap.append(pos)
    if len(af) > cap:
        af.pop(0); ap.pop(0)


def main():
    problem = VisaProblem()
    t0 = time.time()
    hv, allp = [], []
    for s in range(1, 31):
        af = run_perm_moead(problem, s)
        hv.append(compute_hypervolume(af)); allp += af
        print(f"perm-MOEA/D seed {s:2d}: {len(af)} sols HV={hv[-1]:,.0f} ({time.time()-t0:.0f}s)")
    comb = nondominated(allp)
    dm = json.load(open(RESULTS / "discrete_mohho.json"))["per_run_hv"]
    pn = json.load(open(RESULTS / "perm_nsga.json"))["per_run_hv"]
    rnd = json.load(open(RESULTS / "controls.json"))["random_restart"]["per_seed_hv"]
    _, p_rnd = mannwhitneyu(hv, rnd, alternative="greater")
    out = {
        "n_weights": len(weights_lattice(H)), "neighborhood": T, "runs": 30,
        "operators": "Tchebycheff decomposition + OX + swap (permutation)",
        "per_run_hv": hv, "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv)),
        "combined_front_size": len(comb), "combined_front_hv": compute_hypervolume(comb),
        "vs_random_p": float(p_rnd),
        "vs_discrete_mohho_mean_gap_pct": 100 * (np.mean(hv) - np.mean(dm)) / np.mean(dm),
        "vs_permnsga_mean_gap_pct": 100 * (np.mean(hv) - np.mean(pn)) / np.mean(pn),
        "elapsed_s": time.time() - t0,
    }
    json.dump(out, open(RESULTS / "perm_moead.json", "w"), indent=2)
    print(f"\nperm-MOEA/D HV mean {out['hv_mean']:,.0f} +/- {out['hv_std']:,.0f} "
          f"(CV {100*out['hv_std']/out['hv_mean']:.3f}%) | combined {out['combined_front_hv']:,.0f}")
    print(f"vs random p={p_rnd:.2e} | vs discrete-MOHHO {out['vs_discrete_mohho_mean_gap_pct']:+.2f}% "
          f"| vs perm-NSGA {out['vs_permnsga_mean_gap_pct']:+.2f}%")
    print(f"total {out['elapsed_s']:.0f}s")


if __name__ == "__main__":
    main()
