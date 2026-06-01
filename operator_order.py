"""
MECHANISM experiment (reviewer's 9.4 lever): WHY does a real-coded Harris Hawks
optimizer tolerate the SPV random-key encoding better than a real-coded GA? The
hypothesis is that HHO's contraction-toward-leader updates PRESERVE the SPV-induced
order (the decoded permutation), whereas SBX interpolation SCRAMBLES it. We measure
this directly: apply each operator family to random key vectors and report the
Kendall-tau rank correlation between the parent's SPV order and the child's SPV
order (tau=1 -> order fully preserved; tau~0 -> order destroyed). This turns the
'HHO respects the representation' assertion into a measured fact.

Output: app/data/results/operator_order.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

from app.core.config import NUM_GROUPS, LB, UB
from app.core import hho
from compare_nsga2 import sbx, poly_mutate  # real-coded GA operators (eta=20, pm=1/d)

RESULTS = Path("app/data/results")
K = 3000
D = NUM_GROUPS


def tau(parent, child):
    # compare SPV orders via ranks (double argsort) so the metric is always defined,
    # even if an operator collapses the child to (near-)constant keys.
    rp = np.argsort(np.argsort(parent))
    rc = np.argsort(np.argsort(child))
    t, _ = kendalltau(rp, rc)
    return 0.0 if np.isnan(t) else t


def main():
    rng = np.random.default_rng(123)
    res = {}

    def run(name, fn):
        ts = []
        for _ in range(K):
            xi = rng.uniform(LB, UB, D)
            child = fn(xi, rng)
            ts.append(tau(xi, child))
        res[name] = {"mean_tau": float(np.mean(ts)), "std_tau": float(np.std(ts))}
        print(f"  {name:34s} Kendall tau = {np.mean(ts):.3f} +/- {np.std(ts):.3f}")

    print("SPV-order preservation by operator (tau=1 preserves order, 0 destroys it):")
    # --- real-coded GA operators ---
    run("SBX crossover (GA)", lambda xi, r: sbx(xi, r.uniform(LB, UB, D), r)[0])
    run("Polynomial mutation (GA)", lambda xi, r: poly_mutate(xi, r))
    # --- HHO operators (toward a random 'rabbit'/leader) ---
    run("HHO exploration (|E|>=1, OP2)",
        lambda xi, r: hho.op2_exploration_mean(xi, r.uniform(LB, UB, D),
                                               r.uniform(LB, UB, D), r))
    run("HHO soft besiege (OP3, |E|=0.7)",
        lambda xi, r: hho.op3_soft_siege(xi, r.uniform(LB, UB, D), 0.7, r))
    run("HHO hard besiege (OP4, |E|=0.3)",
        lambda xi, r: hho.op4_hard_siege(xi, r.uniform(LB, UB, D), 0.3, r))

    json.dump({"trials": K, "dim": D, "operators": res},
              open(RESULTS / "operator_order.json", "w"), indent=2)
    ga = np.mean([res["SBX crossover (GA)"]["mean_tau"],
                  res["Polynomial mutation (GA)"]["mean_tau"]])
    hhomean = np.mean([res["HHO soft besiege (OP3, |E|=0.7)"]["mean_tau"],
                       res["HHO hard besiege (OP4, |E|=0.3)"]["mean_tau"]])
    print(f"\nGA operators mean tau ~ {ga:.3f} | HHO besiege mean tau ~ {hhomean:.3f}")
    print("-> HHO preserves SPV order far better than SBX." if hhomean > ga + 0.1
          else "-> inconclusive")
    print("saved operator_order.json")


if __name__ == "__main__":
    main()
