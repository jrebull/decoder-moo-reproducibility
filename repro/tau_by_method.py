"""
tau_by_method.py (FASE 1.3) — mide order-preservation tau (Kendall, sobre el orden
SPV decodificado) del operador de variacion de CADA metodo del ladder v5, para la
figura HV vs tau:
  - nsga2_realcoded   : SBX entre padres vivos (on-trajectory, ya en tau_trajectory)
  - naive_mohho       : siege/exploracion HHO sobre padres vivos
  - competent_mohho   : su paso de variacion (HHO offspring + poly-mut [+SBX])
  - perm_nsga2/perm_moead/discrete_mohho : OX + swap sobre permutaciones (tau del
    operador permutacional sobre el orden, no del random-key)
  - random_restart    : sin operador (muestreo independiente) -> tau indefinido (n/a)

Para los continuos medimos sobre poblaciones VIVAS de una corrida real (no padres
uniformes). Escribe app/data/results/tau_by_method.json.
Presupuesto via env: PAIRS (def 400), SEED (def 1).
"""
import os, json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
import _bootstrap; _bootstrap.bootstrap_engine()

from app.core.config import NUM_GROUPS, LB, UB
from app.core.problem import VisaProblem
from app.core.mohho import evaluate_hawk
from app.core import hho as H
import competent_mohho as C
from compare_nsga2 import sbx, fast_nondominated_sort, crowding_distance, tournament, poly_mutate
from perm_nsga import ox as perm_ox

RESULTS = Path(_bootstrap.results_dir())
PAIRS = int(os.environ.get("PAIRS", 400))
SEED = int(os.environ.get("SEED", 1))
POP, GEN = 50, 500
SNAPS = [1, 50, 150, 300, 499]
PM_COMP = 0.15


def spv(v): return np.argsort(np.argsort(v))
def tau(a_order, c_order):
    t, _ = kendalltau(a_order, c_order)
    return 0.0 if np.isnan(t) else float(t)
def spv_tau(parent, child): return tau(spv(parent), spv(child))


def evolve_population_and_measure(p, op_name):
    """Run a real NSGA-II trajectory (live population) and at snapshots apply
    `op_name`'s variation operator to live parents, recording SPV-order tau."""
    rng = np.random.default_rng(SEED)
    pop = rng.uniform(LB, UB, size=(POP, NUM_GROUPS))
    fits = [evaluate_hawk(pop[i], p)[1] for i in range(POP)]
    taus = []
    for it in range(GEN):
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr): cd[idx] = d[k]
        if it in SNAPS:
            xmean = pop.mean(axis=0)
            for _ in range(PAIRS):
                a = pop[tournament(rank, cd, rng)]
                b = pop[tournament(rank, cd, rng)]
                if op_name == "nsga2_realcoded":
                    c, _2 = sbx(a, b, rng); c = poly_mutate(c, rng)
                elif op_name == "naive_mohho":
                    c = H.op3_soft_siege(a, b, 0.7, rng)   # siege toward leader b
                elif op_name == "competent_mohho":
                    c = C.offspring_move(a, rng, NUM_GROUPS, pm=PM_COMP, use_sbx=True)
                taus.append(spv_tau(a, c))
        # advance one NSGA-II generation (shared live trajectory)
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]; p2 = pop[tournament(rank, cd, rng)]
            c1, c2 = sbx(p1, p2, rng) if rng.random() <= 0.9 else (p1.copy(), p2.copy())
            off.append(poly_mutate(c1, rng))
            if len(off) < POP: off.append(poly_mutate(c2, rng))
        off = np.array(off); ofits = [evaluate_hawk(off[i], p)[1] for i in range(POP)]
        comb = np.vstack([pop, off]); cfits = fits + ofits
        fronts, _ = fast_nondominated_sort(cfits)
        ni = []
        for fr in fronts:
            if len(ni) + len(fr) <= POP: ni += fr
            else:
                d = crowding_distance([cfits[i] for i in fr])
                ni += [fr[k] for k in sorted(range(len(fr)), key=lambda k: d[k], reverse=True)[:POP-len(ni)]]
                break
        pop = comb[ni]; fits = [cfits[i] for i in ni]
    return float(np.mean(taus)), float(np.std(taus))


def perm_tau(p, n_pairs=PAIRS):
    """tau of OX+swap on permutations (order operator of the perm tier)."""
    rng = np.random.default_rng(SEED)
    taus = []
    for _ in range(n_pairs):
        a = rng.permutation(NUM_GROUPS); b = rng.permutation(NUM_GROUPS)
        child = perm_ox(a, b, rng)
        if rng.random() < 0.3:
            for _2 in range(rng.integers(1, 4)):
                i, j = rng.integers(0, NUM_GROUPS, size=2); child[i], child[j] = child[j], child[i]
        # tau of the decoded ORDER (the permutation IS the order)
        taus.append(tau(np.argsort(np.argsort(a)), np.argsort(np.argsort(child))))
    return float(np.mean(taus)), float(np.std(taus))


def main():
    p = VisaProblem()
    out = {"seed": SEED, "pairs": PAIRS, "snapshots": SNAPS, "methods": {}}
    for op in ["nsga2_realcoded", "naive_mohho", "competent_mohho"]:
        m, s = evolve_population_and_measure(p, op)
        out["methods"][op] = {"tau_mean": round(m, 4), "tau_std": round(s, 4), "operator": "continuous"}
        print(f"  {op:18s} tau = {m:+.3f} +/- {s:.3f}")
    pm, ps = perm_tau(p)
    for op in ["perm_nsga2", "perm_moead", "discrete_mohho"]:
        out["methods"][op] = {"tau_mean": round(pm, 4), "tau_std": round(ps, 4),
                              "operator": "permutation (OX+swap)"}
    print(f"  perm tier (OX+swap) tau = {pm:+.3f} +/- {ps:.3f}")
    out["methods"]["random_restart"] = {"tau_mean": None, "operator": "none (independent sampling)"}
    (RESULTS / "tau_by_method.json").write_text(json.dumps(out, indent=2))
    print("-> tau_by_method.json")


if __name__ == "__main__":
    main()
