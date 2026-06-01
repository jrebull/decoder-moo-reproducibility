"""
OPERATOR x ARCHITECTURE FACTORIAL (reviewer's #1 lever): the "metaheuristic family
is second-order" claim was tested with three selection architectures that all share
ONE operator set (OX+swap), so only the selection/replacement architecture was
varied. Here we vary BOTH axes: three permutation operator sets x three selection
architectures = 9 cells, 30 seeds each, on the visa problem under the same greedy
decoder and matched budget. A two-way variance decomposition then attributes
hypervolume variation to the operator axis vs. the architecture axis, converting
the near-circular claim into a measured one.

  Operator sets:   OX+swap | PMX+insertion | CX+inversion
  Architectures:   perm-NSGA-II (Pareto) | perm-MOEA/D (Tchebycheff) | Discrete-MOHHO (energy)

Output: app/data/results/factorial_ops.json
"""
import json, time, itertools
from pathlib import Path
import numpy as np

from app.core.config import NUM_GROUPS
from app.core.decoder import decode
from app.core.problem import VisaProblem
from app.core.mohho import (compute_hypervolume, crowding_distance, dominates,
                            update_archive, select_leader, HV_REF_POINT)
from compare_nsga2 import fast_nondominated_sort, tournament, nondominated

RESULTS = Path("app/data/results")
POP, GEN, RUNS = 50, 500, 30
ARC = 100
PM = 0.3
N = NUM_GROUPS
SCALE = np.array(HV_REF_POINT, dtype=float)


def eval_perm(perm, problem):
    alloc = decode(list(perm), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    return problem.evaluate(alloc)


# ---------------- crossovers (one child, valid permutation) ----------------
def ox(p1, p2, rng):
    a, b = sorted(rng.choice(N, 2, replace=False))
    seg = set(p1[a:b + 1].tolist()); c = -np.ones(N, dtype=int); c[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in seg]; k = 0
    for i in list(range(b + 1, N)) + list(range(0, a)):
        c[i] = fill[k]; k += 1
    return c


def pmx(p1, p2, rng):
    a, b = sorted(rng.choice(N, 2, replace=False))
    c = -np.ones(N, dtype=int); c[a:b + 1] = p1[a:b + 1]
    seg = set(p1[a:b + 1].tolist())
    pos_p2 = {v: i for i, v in enumerate(p2)}
    for i in range(a, b + 1):
        v = int(p2[i])
        if v in seg:
            continue
        j = i                                  # follow the mapping out of the segment
        while a <= j <= b:
            j = pos_p2[int(p1[j])]
        c[j] = v
    for i in range(N):
        if c[i] == -1:
            c[i] = p2[i]
    return c


def cx(p1, p2, rng):
    c = -np.ones(N, dtype=int); pos_p1 = {v: i for i, v in enumerate(p1)}
    visited = np.zeros(N, bool); cyc = 0
    while not visited.all():
        start = int(np.where(~visited)[0][0]); j = start; from_p1 = (cyc % 2 == 0)
        while True:
            visited[j] = True
            c[j] = p1[j] if from_p1 else p2[j]
            j = pos_p1[p2[j]]
            if j == start:
                break
        cyc += 1
    return c


# ---------------- mutations (exactly k operations) ----------------
def m_swap(perm, rng, k):
    y = perm.copy()
    for _ in range(k):
        i, j = rng.integers(0, N, size=2); y[i], y[j] = y[j], y[i]
    return y


def m_insert(perm, rng, k):
    y = list(perm)
    for _ in range(k):
        i = int(rng.integers(0, len(y))); v = y.pop(i)
        j = int(rng.integers(0, len(y) + 1)); y.insert(j, v)
    return np.array(y)


def m_invert(perm, rng, k):
    y = perm.copy()
    for _ in range(k):
        a, b = sorted(rng.integers(0, N, size=2)); y[a:b + 1] = y[a:b + 1][::-1]
    return y


OPSETS = {
    "OX+swap":       (ox, m_swap),
    "PMX+insertion": (pmx, m_insert),
    "CX+inversion":  (cx, m_invert),
}


# ---------------- architectures, parameterized by (xover, mut) ----------------
def run_nsga(problem, seed, xover, mut):
    rng = np.random.default_rng(seed)
    pop = np.array([rng.permutation(N) for _ in range(POP)])
    fits = [eval_perm(pop[i], problem) for i in range(POP)]
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits); cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr):
                cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]; p2 = pop[tournament(rank, cd, rng)]
            ch = xover(p1, p2, rng)
            if rng.random() < PM:
                ch = mut(ch, rng, int(rng.integers(1, 4)))
            off.append(ch)
        off = np.array(off); off_f = [eval_perm(off[i], problem) for i in range(POP)]
        comb = np.vstack([pop, off]); cf = fits + off_f
        fronts, _ = fast_nondominated_sort(cf); ni = []
        for fr in fronts:
            if len(ni) + len(fr) <= POP:
                ni += fr
            else:
                d = crowding_distance([cf[i] for i in fr])
                ni += [fr[k] for k in sorted(range(len(fr)), key=lambda k: d[k], reverse=True)[:POP - len(ni)]]
                break
        pop = comb[ni]; fits = [cf[i] for i in ni]
    fronts, _ = fast_nondominated_sort(fits)
    return [fits[i] for i in fronts[0]]


def run_moead(problem, seed, xover, mut):
    rng = np.random.default_rng(seed)
    W = np.array([np.array(c, float) / 8 for c in itertools.product(range(9), repeat=3)
                  if sum(c) == 8]) + 1e-6
    NW = len(W); gens = max(1, (POP * GEN) // NW); T = 10
    B = [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T] for i in range(NW)]
    pop = [rng.permutation(N) for _ in range(NW)]
    fit = [np.array(eval_perm(pop[i], problem), float) for i in range(NW)]
    fn = [f / SCALE for f in fit]; z = np.min(np.array(fn), axis=0)
    ap, af = [], []
    for i in range(NW):
        update_archive(ap, af, pop[i], tuple(fit[i]), ARC, rng)
    for _ in range(gens):
        for i in range(NW):
            k, l = rng.choice(B[i], 2, replace=False)
            ch = xover(pop[k], pop[l], rng)
            if rng.random() < PM:
                ch = mut(ch, rng, int(rng.integers(1, 4)))
            cf = np.array(eval_perm(ch, problem), float); cfn = cf / SCALE
            z = np.minimum(z, cfn)
            for j in B[i]:
                if np.max(W[j] * (cfn - z)) <= np.max(W[j] * (fn[j] - z)):
                    pop[j] = ch; fit[j] = cf; fn[j] = cfn
            update_archive(ap, af, ch, tuple(cf), ARC, rng)
    return af


def run_discrete(problem, seed, xover, mut):
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(N) for _ in range(POP)]
    fits = [eval_perm(pop[i], problem) for i in range(POP)]
    ap, af = [], []
    for i in range(POP):
        update_archive(ap, af, pop[i], fits[i], ARC, rng)
    for t in range(GEN):
        for i in range(POP):
            e = 2 * (2 * rng.random() - 1) * (1 - t / GEN); ae = abs(e)
            rab = select_leader(ap, af, rng)
            if ae >= 1:
                ch = xover(pop[int(rng.integers(POP))], rab, rng) if rng.random() < 0.5 \
                    else mut(pop[i], rng, max(1, N // 4))
            else:
                ch = xover(rab, pop[i], rng)
                ch = mut(ch, rng, max(1, int(round(ae * (N // 6)))))
            cf = eval_perm(ch, problem)
            update_archive(ap, af, ch, cf, ARC, rng)
            if dominates(cf, fits[i]):
                pop[i], fits[i] = ch, cf
    return af


ARCHS = {"perm-NSGA-II": run_nsga, "perm-MOEA/D": run_moead, "Discrete-MOHHO": run_discrete}


def main():
    p = VisaProblem()
    t0 = time.time()
    cells = {}                                   # (op, arch) -> per-seed HV list
    for opname, (xo, mu) in OPSETS.items():
        for aname, fn in ARCHS.items():
            hv = []
            for s in range(1, RUNS + 1):
                af = fn(p, s, xo, mu); hv.append(compute_hypervolume(af))
            cells[(opname, aname)] = hv
            print(f"  {opname:14s} x {aname:14s}  HV={np.mean(hv):,.0f}+-{np.std(hv):,.0f} "
                  f"({time.time()-t0:.0f}s)")

    ops = list(OPSETS); archs = list(ARCHS)
    M = np.array([[np.mean(cells[(o, a)]) for a in archs] for o in ops])   # 3x3 means
    allvals = np.array([cells[(o, a)] for o in ops for a in archs])         # 9 x 30
    flat = allvals.flatten(); grand = flat.mean()
    # two-way variance decomposition (balanced, n=30 per cell)
    n = RUNS
    op_means = allvals.reshape(len(ops), len(archs), n).mean(axis=(1, 2))
    ar_means = allvals.reshape(len(ops), len(archs), n).mean(axis=(0, 2))
    cell_means = allvals.reshape(len(ops), len(archs), n).mean(axis=2)
    SS_tot = float(((flat - grand) ** 2).sum())
    SS_op = float(n * len(archs) * ((op_means - grand) ** 2).sum())
    SS_ar = float(n * len(ops) * ((ar_means - grand) ** 2).sum())
    SS_cell = float(n * ((cell_means - grand) ** 2).sum())
    SS_int = SS_cell - SS_op - SS_ar
    SS_res = SS_tot - SS_cell
    eta = {"operator": SS_op / SS_tot, "architecture": SS_ar / SS_tot,
           "interaction": SS_int / SS_tot, "residual_seed": SS_res / SS_tot}
    out = {
        "design": "3 operator sets x 3 architectures, 30 seeds, visa problem, matched budget",
        "operators": ops, "architectures": archs,
        "cell_hv_mean": {f"{o} | {a}": float(np.mean(cells[(o, a)])) for o in ops for a in archs},
        "cell_hv_std": {f"{o} | {a}": float(np.std(cells[(o, a)])) for o in ops for a in archs},
        "operator_marginal_hv": {o: float(op_means[i]) for i, o in enumerate(ops)},
        "architecture_marginal_hv": {a: float(ar_means[j]) for j, a in enumerate(archs)},
        "operator_range_hv": float(op_means.max() - op_means.min()),
        "architecture_range_hv": float(ar_means.max() - ar_means.min()),
        "eta_squared": eta, "grand_mean_hv": float(grand),
        "elapsed_s": time.time() - t0,
    }
    json.dump(out, open(RESULTS / "factorial_ops.json", "w"), indent=2)
    print("\n=== VARIANCE DECOMPOSITION (eta^2, fraction of HV variance) ===")
    for k, v in eta.items():
        print(f"  {k:14s} {100*v:5.1f}%")
    print(f"operator marginal range {out['operator_range_hv']:,.0f} HV | "
          f"architecture marginal range {out['architecture_range_hv']:,.0f} HV")
    print(f"cell means span {M.min():,.0f}--{M.max():,.0f} (grand {grand:,.0f})")
    print(f"-> factorial_ops.json ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
