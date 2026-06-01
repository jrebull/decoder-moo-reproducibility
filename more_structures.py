"""
THIRD AND FOURTH STRUCTURALLY-DISTINCT PROBLEMS (top-tier generalization): to move
the cross-structure evidence beyond n=1, we replicate the SIX-METHOD LADDER on two
further classic multi-objective combinatorial structures that share NOTHING with the
visa allocation or the knapsack except the SPV+permutation methodology:

  (A) mo-TSP  -- tri-objective travelling salesman: a permutation is a CYCLIC tour;
      three independent cost matrices give three tour-length objectives. No selection,
      no capacity, no scheduling -- pure sequencing with wrap-around cost.
  (B) mo-PFSP -- tri-objective permutation flow-shop scheduling: a permutation is a job
      order on M machines; objectives are makespan, total flowtime, and total tardiness.
      Machine-precedence structure, no capacity, no selection.

Same six methods, budget, and 30 seeds as MOMKP (imported verbatim from
second_problem.py; we only swap the problem object and its size N). If the
random-key < permutation-native ordering and the GA-freeze replicate here too, the
'representation governs, family is second-order' regularity holds across FOUR
structurally distinct problems (visa, knapsack, TSP, flow-shop).

Output: app/data/results/more_structures.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare, rankdata

import second_problem as sp
from second_problem import (run_random, run_nsga_realcoded, run_hho_realcoded,
                            run_permnsga, run_discrete_mohho, run_permmoead)
from app.core.mohho import compute_hypervolume
from compare_nsga2 import nondominated

RESULTS = Path("app/data/results")
SEEDS = list(range(1, 31))
REF = (1.0, 1.0, 1.0)
Q05 = {6: 2.850}

METHODS = [
    ("NSGA-II (real-coded)", "random key", "GA", run_nsga_realcoded),
    ("Random restart", "random key", "---", run_random),
    ("MOHHO (real-coded)", "random key", "swarm", run_hho_realcoded),
    ("Discrete-MOHHO", "permutation", "swarm", run_discrete_mohho),
    ("perm-MOEA/D", "permutation", "decomp.", run_permmoead),
    ("perm-NSGA-II", "permutation", "GA", run_permnsga),
]


class MOTSP:
    """Tri-objective TSP: permutation = cyclic tour; 3 cost matrices -> 3 tour lengths."""
    def __init__(self, n=100, seed=7):
        rng = np.random.default_rng(seed)
        self.n = n
        self.D = []
        for _ in range(3):
            pts = rng.uniform(0, 1, size=(n, 2))
            M = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
            self.D.append(M)
        # fixed upper-bound scale per objective so f in (0,1), REF=(1,1,1) dominates
        self.scale = [float(n * M.max()) for M in self.D]

    def eval_perm(self, perm):
        perm = np.asarray(perm)
        nxt = np.roll(perm, -1)
        return tuple(float(self.D[k][perm, nxt].sum() / self.scale[k]) for k in range(3))

    def eval_keys(self, keys):
        return self.eval_perm(np.argsort(keys))


class MOPFSP:
    """Tri-objective permutation flow-shop: makespan, total flowtime, total tardiness."""
    def __init__(self, n=50, m=10, seed=7):
        rng = np.random.default_rng(seed)
        self.n, self.m = n, m
        self.p = rng.integers(1, 100, size=(n, m)).astype(float)   # processing times
        # loose, fixed due dates (tight enough that tardiness discriminates)
        self.due = 0.5 * self.p.sum(axis=1).mean() * rng.uniform(0.8, 1.6, size=n)
        tot = self.p.sum()
        self.scale = (float(tot), float(n * tot), float(n * tot))  # loose upper bounds

    def eval_perm(self, perm):
        perm = np.asarray(perm)
        m = self.m
        C = np.zeros(m)                      # completion time on each machine
        comp = np.zeros(self.n)              # completion (last machine) per job position
        for pos, job in enumerate(perm):
            C[0] += self.p[job, 0]
            for k in range(1, m):
                C[k] = max(C[k], C[k - 1]) + self.p[job, k]
            comp[pos] = C[m - 1]
        cmax = C[m - 1]
        flow = comp.sum()
        # tardiness: due date is per JOB; map back via perm order
        due_in_order = self.due[perm]
        tard = np.maximum(0.0, comp - due_in_order).sum()
        return (float(cmax / self.scale[0]), float(flow / self.scale[1]),
                float(tard / self.scale[2]))

    def eval_keys(self, keys):
        return self.eval_perm(np.argsort(keys))


def run_ladder(prob, label):
    """Run the six methods on `prob` (sets the second_problem globals N, PM)."""
    sp.N = prob.n
    sp.PM = 1.0 / prob.n
    out = {}; t0 = time.time()
    print(f"\n=== {label} (N={prob.n}) ===")
    for name, enc, par, fn in METHODS:
        hv, allf = [], []
        for s in SEEDS:
            af = fn(prob, s); hv.append(compute_hypervolume(af, REF)); allf += af
        comb = nondominated(allf)
        out[name] = {"encoding": enc, "paradigm": par, "per_run_hv": hv,
                     "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv)),
                     "cv": float(np.std(hv) / np.mean(hv)),
                     "combined_hv": compute_hypervolume(comb, REF), "combined_sols": len(comb)}
        print(f"  {name:22s} HV={np.mean(hv):.4f}+/-{np.std(hv):.4f} "
              f"CV={100*np.std(hv)/np.mean(hv):.2f}%  ({time.time()-t0:.0f}s)")
    # Friedman + Nemenyi across the six (common seeds -> paired)
    names = list(out)
    X = np.array([out[n]["per_run_hv"] for n in names])
    ranks = np.zeros_like(X)
    for col in range(X.shape[1]):
        ranks[:, col] = rankdata(-X[:, col])
    avg = {names[i]: float(ranks[i].mean()) for i in range(len(names))}
    chi, p = friedmanchisquare(*[X[i] for i in range(len(names))])
    CD = Q05[6] * np.sqrt(6 * 7 / (6 * X.shape[1]))
    rk = [m for m in out if out[m]["encoding"] == "random key"]
    pn = [m for m in out if out[m]["encoding"] == "permutation"]
    replicates = min(out[m]["hv_mean"] for m in pn) > max(out[m]["hv_mean"] for m in rk)
    nsga_worst = avg["NSGA-II (real-coded)"] == max(avg.values())
    print(f"  Friedman chi2={chi:.1f} p={p:.1e} CD={CD:.3f} | "
          f"perm-tier>random-key-tier? {replicates} | NSGA worst? {nsga_worst}")
    return {"methods": out, "avg_rank": avg, "friedman_chi2": float(chi),
            "friedman_p": float(p), "nemenyi_CD": float(CD),
            "perm_tier_beats_random_key_tier": bool(replicates),
            "nsga_realcoded_worst": bool(nsga_worst)}


def main():
    t0 = time.time()
    res = {
        "mo-TSP": {"desc": "tri-objective TSP, 100 cities, 3 cost matrices",
                   **run_ladder(MOTSP(n=100, seed=7), "mo-TSP")},
        "mo-PFSP": {"desc": "tri-objective permutation flow-shop, 50 jobs x 10 machines "
                            "(makespan, flowtime, tardiness)",
                    **run_ladder(MOPFSP(n=50, m=10, seed=7), "mo-PFSP")},
    }
    res["seeds"] = SEEDS; res["budget"] = sp.POP * sp.GEN; res["elapsed_s"] = time.time() - t0
    json.dump(res, open(RESULTS / "more_structures.json", "w"), indent=2)
    print("\n=== CROSS-STRUCTURE SUMMARY ===")
    for prob in ("mo-TSP", "mo-PFSP"):
        r = res[prob]
        print(f"{prob}: replicates={r['perm_tier_beats_random_key_tier']} "
              f"NSGA-worst={r['nsga_realcoded_worst']} (chi2={r['friedman_chi2']:.1f})")
    print(f"-> more_structures.json ({res['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
