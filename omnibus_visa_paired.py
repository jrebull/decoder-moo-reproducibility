"""
STATISTICS-REFEREE FIX: the original visa omnibus used Kruskal-Wallis +
unpaired Mann-Whitney, asserting the six methods are independent. They are not:
several methods shared seed sets, so the design is partially paired. We resolve
this the same way the MOMKP omnibus already does it -- run ALL SIX methods on a
SINGLE COMMON seed set (1-30) and apply the paired Friedman test with a Nemenyi
post-hoc (the Demsar standard). This makes the visa and MOMKP analyses
methodologically identical and removes the paired/unpaired confound.

Output: app/data/results/omnibus_visa_paired.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare, rankdata

from app.core.config import POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE
from app.core.problem import VisaProblem
from app.core.mohho import run_mohho, compute_hypervolume
from compare_nsga2 import run_nsga2
from controls import run_random
from discrete_mohho import run_discrete_mohho
from perm_nsga import run_perm_nsga
from perm_moead import run_perm_moead

R = Path("app/data/results")
SEEDS = list(range(1, 31))                 # ONE common seed set for all six methods
POP, IT, ARC = POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def hv_mohho(p, s):
    _, fits, _ = run_mohho(p, seed=s, pop_size=POP, max_iter=IT, archive_size=ARC)
    return compute_hypervolume(fits)


METHODS = {
    "NSGA-II":        lambda p, s: compute_hypervolume(run_nsga2(p, s)),
    "Random restart": lambda p, s: compute_hypervolume(run_random(p, s)),
    "MOHHO":          hv_mohho,
    "Discrete-MOHHO": lambda p, s: compute_hypervolume(run_discrete_mohho(p, s)),
    "perm-MOEA/D":    lambda p, s: compute_hypervolume(run_perm_moead(p, s)),
    "perm-NSGA-II":   lambda p, s: compute_hypervolume(run_perm_nsga(p, s)),
}


def main():
    p = VisaProblem()
    names = list(METHODS)
    X = np.zeros((len(names), len(SEEDS)))      # k x N matrix of per-seed HV
    for j, s in enumerate(SEEDS):
        for i, n in enumerate(names):
            X[i, j] = METHODS[n](p, s)
        print(f"seed {s:2d}: " + " ".join(f"{n}={X[i,j]:,.0f}" for i, n in enumerate(names)))
    # per-block ranks, rank 1 = best (largest HV)
    ranks = np.zeros_like(X)
    for col in range(X.shape[1]):
        ranks[:, col] = rankdata(-X[:, col])
    avg = {names[i]: float(ranks[i].mean()) for i in range(len(names))}
    chi, pval = friedmanchisquare(*[X[i] for i in range(len(names))])
    k, Nn = len(names), X.shape[1]
    CD = Q05[k] * np.sqrt(k * (k + 1) / (6 * Nn))
    # pairwise: rank-difference vs CD (Nemenyi)
    pairwise = []
    for i in range(len(names)):
        for jx in range(i + 1, len(names)):
            diff = abs(avg[names[i]] - avg[names[jx]])
            pairwise.append({"a": names[i], "b": names[jx],
                             "rank_diff": float(diff), "significant": bool(diff > CD)})
    out = {"test": "Friedman + Nemenyi (common seeds 1-30, paired)",
           "seeds": SEEDS, "pop": POP, "iter": IT, "archive": ARC,
           "chi2": float(chi), "p": float(pval), "k": k, "N": Nn,
           "avg_rank": avg, "nemenyi_CD": float(CD),
           "hv_mean": {names[i]: float(X[i].mean()) for i in range(len(names))},
           "hv_std": {names[i]: float(X[i].std()) for i in range(len(names))},
           "pairwise_nemenyi": pairwise}
    json.dump(out, open(R / "omnibus_visa_paired.json", "w"), indent=2)
    print(f"\n=== VISA Friedman (paired, common seeds) chi2={chi:.1f} p={pval:.2e} CD={CD:.3f} ===")
    for n, r in sorted(avg.items(), key=lambda x: x[1]):
        print(f"  {n:16s} avg rank {r:.2f}  (HV {X[names.index(n)].mean():,.0f})")
    print("significant pairwise gaps (rank diff > CD):")
    for pr in pairwise:
        if pr["significant"]:
            print(f"  {pr['a']} vs {pr['b']}: {pr['rank_diff']:.2f}")
    print("saved omnibus_visa_paired.json")


if __name__ == "__main__":
    main()
