"""
SECOND, STRUCTURALLY-DISTINCT PROBLEM (journal-grade generalization): a 3-objective
multidimensional 0/1 knapsack (MOMKP) -- a classic combinatorial MO structure with
NO per-country caps, NO spillover cascade, NO waiting-time objective. Same
methodology: a feasibility-preserving SPV + greedy decoder (add items in SPV order
while every capacity holds), and the SAME six-method ladder. If the ordering
random-key methods < permutation-native cluster replicates here, the 'representation
governs, family is second-order' regularity generalizes across problem structures.

Output: app/data/results/second_problem.json
"""
import json, time
from pathlib import Path
import numpy as np

from app.core.mohho import (compute_hypervolume, dominates, crowding_distance,
                            select_leader)
from compare_nsga2 import fast_nondominated_sort, nondominated
from app.core import hho

RESULTS = Path("app/data/results")
N, MCAP, POP, GEN = 120, 4, 50, 500
SEEDS = list(range(1, 31))
REF = (1.0, 1.0, 1.0)
ETA, PC, PM = 20.0, 0.9, 1.0 / N


class MOMKP:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.profit = rng.integers(10, 100, size=(N, 3)).astype(float)
        self.weight = rng.integers(10, 100, size=(N, MCAP)).astype(float)
        self.cap = 0.5 * self.weight.sum(axis=0)
        self.pmax = self.profit.sum(axis=0)
        self.n = N

    def decode_perm(self, perm):
        load = np.zeros(MCAP); sel = []
        for i in perm:
            if np.all(load + self.weight[i] <= self.cap):
                sel.append(int(i)); load += self.weight[i]
        return sel

    def evaluate(self, sel):
        p = self.profit[sel].sum(axis=0) if sel else np.zeros(3)
        f = (self.pmax - p) / self.pmax
        return (float(f[0]), float(f[1]), float(f[2]))

    def eval_perm(self, perm):
        return self.evaluate(self.decode_perm(perm))

    def eval_keys(self, keys):
        return self.eval_perm(np.argsort(keys))


# ---------- generic archive (keeps positions for leader-based methods) ----------
def arch_add(ap, af, pos, fit, cap=100):
    for f in af:
        if dominates(f, fit) or f == fit:
            return
    keep = [i for i, f in enumerate(af) if not dominates(fit, f)]
    af[:] = [af[i] for i in keep]; ap[:] = [ap[i] for i in keep]
    af.append(fit); ap.append(pos)
    if len(af) > cap:
        af.pop(0); ap.pop(0)


# ---------- permutation operators ----------
def ox(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    seg = set(p1[a:b + 1].tolist()); c = -np.ones(n, dtype=int); c[a:b + 1] = p1[a:b + 1]
    fill = [g for g in p2 if g not in seg]; k = 0
    for i in list(range(b + 1, n)) + list(range(0, a)):
        c[i] = fill[k]; k += 1
    return c


def swap_mut(perm, rng, pm=0.3):
    y = perm.copy()
    if rng.random() < pm:
        for _ in range(rng.integers(1, 4)):
            i, j = rng.integers(0, len(y), size=2); y[i], y[j] = y[j], y[i]
    return y


def rev_seg(perm, rng):
    y = perm.copy(); a, b = sorted(rng.choice(len(y), 2, replace=False))
    y[a:b + 1] = y[a:b + 1][::-1]; return y


# ---------- real-coded operators (keys in [0,1]^N) ----------
def sbx(p1, p2, rng):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-12:
            u = rng.random()
            beta = (2 * u) ** (1 / (ETA + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (ETA + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mut(x, rng):
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < PM:
            u = rng.random()
            d = (2 * u) ** (1 / (ETA + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (ETA + 1))
            y[i] = x[i] + d
    return np.clip(y, 0, 1)


def tourney(rank, cd, rng):
    a, b = rng.integers(0, len(rank), size=2)
    if rank[a] < rank[b]: return a
    if rank[b] < rank[a]: return b
    return a if cd[a] >= cd[b] else b


# ============================ the six methods ============================
def run_random(prob, seed):
    rng = np.random.default_rng(seed); ap, af = [], []
    for _ in range(POP * GEN):
        p = rng.permutation(N); arch_add(ap, af, p, prob.eval_perm(p))
    return af


def run_nsga_realcoded(prob, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0, 1, size=(POP, N)); fits = [prob.eval_keys(pop[i]) for i in range(POP)]
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits); cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr): cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tourney(rank, cd, rng)]; p2 = pop[tourney(rank, cd, rng)]
            c1, c2 = sbx(p1, p2, rng) if rng.random() <= PC else (p1.copy(), p2.copy())
            off.append(poly_mut(c1, rng))
            if len(off) < POP: off.append(poly_mut(c2, rng))
        off = np.array(off); offf = [prob.eval_keys(off[i]) for i in range(POP)]
        comb = np.vstack([pop, off]); cf = fits + offf
        fronts, _ = fast_nondominated_sort(cf); ni = []
        for fr in fronts:
            if len(ni) + len(fr) <= POP: ni += fr
            else:
                d = crowding_distance([cf[i] for i in fr])
                ni += [fr[k] for k in sorted(range(len(fr)), key=lambda k: d[k], reverse=True)[:POP - len(ni)]]; break
        pop = comb[ni]; fits = [cf[i] for i in ni]
    fronts, _ = fast_nondominated_sort(fits)
    return [fits[i] for i in fronts[0]]


def run_hho_realcoded(prob, seed):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0, 1, size=(POP, N)); fits = [prob.eval_keys(pop[i]) for i in range(POP)]
    ap, af = [], []
    for i in range(POP): arch_add(ap, af, pop[i], fits[i])
    for t in range(GEN):
        xm = pop.mean(0)
        for i in range(POP):
            e = hho.escape_energy(t, GEN, rng); ae = abs(e)
            rab = select_leader(ap, af, rng)
            if ae >= 1:
                if rng.random() >= 0.5:
                    new = hho.op1_exploration_random(pop[i], pop[rng.integers(POP)], rng)
                else:
                    new = hho.op2_exploration_mean(pop[i], rab, xm, rng)
            elif rng.random() >= 0.5:
                new = hho.op3_soft_siege(pop[i], rab, e, rng) if ae >= 0.5 else hho.op4_hard_siege(pop[i], rab, e, rng)
            else:
                y, z = (hho.op5_soft_siege_levy(pop[i], rab, e, rng) if ae >= 0.5
                        else hho.op6_hard_siege_levy(pop[i], rab, e, xm, rng))
                # FE-budget parity: single trial point per hawk per iteration (eval y
                # only), matching the one-eval-per-individual budget of every other method.
                fy = prob.eval_keys(y); arch_add(ap, af, y, fy)
                if dominates(fy, fits[i]): pop[i], fits[i] = y, fy
                continue
            nf = prob.eval_keys(new); arch_add(ap, af, new, nf)
            if dominates(nf, fits[i]): pop[i], fits[i] = new, nf
    return af


def run_permnsga(prob, seed):
    rng = np.random.default_rng(seed)
    pop = np.array([rng.permutation(N) for _ in range(POP)]); fits = [prob.eval_perm(pop[i]) for i in range(POP)]
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits); cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr): cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tourney(rank, cd, rng)]; p2 = pop[tourney(rank, cd, rng)]
            off.append(swap_mut(ox(p1, p2, rng), rng))
            if len(off) < POP: off.append(swap_mut(ox(p2, p1, rng), rng))
        off = np.array(off); offf = [prob.eval_perm(off[i]) for i in range(POP)]
        comb = np.vstack([pop, off]); cf = fits + offf
        fronts, _ = fast_nondominated_sort(cf); ni = []
        for fr in fronts:
            if len(ni) + len(fr) <= POP: ni += fr
            else:
                d = crowding_distance([cf[i] for i in fr])
                ni += [fr[k] for k in sorted(range(len(fr)), key=lambda k: d[k], reverse=True)[:POP - len(ni)]]; break
        pop = comb[ni]; fits = [cf[i] for i in ni]
    fronts, _ = fast_nondominated_sort(fits)
    return [fits[i] for i in fronts[0]]


def run_discrete_mohho(prob, seed):
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(N) for _ in range(POP)]; fits = [prob.eval_perm(pop[i]) for i in range(POP)]
    ap, af = [], []
    for i in range(POP): arch_add(ap, af, pop[i], fits[i])
    for t in range(GEN):
        for i in range(POP):
            e = 2 * (2 * rng.random() - 1) * (1 - t / GEN); ae = abs(e)
            rab = select_leader(ap, af, rng)
            if ae >= 1:
                child = ox(pop[rng.integers(POP)], rab, rng) if rng.random() < 0.5 \
                    else swap_mut(pop[i], rng, pm=1.0)
            else:
                child = ox(rab, pop[i], rng)
                k = max(1, int(round(ae * (N // 6))))
                for _ in range(k):
                    a, b = rng.integers(0, N, size=2); child[a], child[b] = child[b], child[a]
                if rng.random() < 0.5: child = rev_seg(child, rng)
            cf = prob.eval_perm(child); arch_add(ap, af, child, cf)
            if dominates(cf, fits[i]): pop[i], fits[i] = child, cf
    return af


def run_permmoead(prob, seed):
    import itertools
    rng = np.random.default_rng(seed)
    W = np.array([np.array(c, float) / 8 for c in itertools.product(range(9), repeat=3) if sum(c) == 8]) + 1e-6
    NW = len(W); gens = max(1, POP * GEN // NW); T = 10
    B = [np.argsort(np.linalg.norm(W - W[i], axis=1))[:T] for i in range(NW)]
    pop = [rng.permutation(N) for _ in range(NW)]; fit = [np.array(prob.eval_perm(pop[i])) for i in range(NW)]
    z = np.min(np.array(fit), axis=0); ap, af = [], []
    for i in range(NW): arch_add(ap, af, pop[i], tuple(fit[i]))
    for _ in range(gens):
        for i in range(NW):
            k, l = rng.choice(B[i], 2, replace=False)
            child = swap_mut(ox(pop[k], pop[l], rng), rng); cf = np.array(prob.eval_perm(child))
            z = np.minimum(z, cf)
            for j in B[i]:
                if np.max(W[j] * (cf - z)) <= np.max(W[j] * (fit[j] - z)):
                    pop[j] = child; fit[j] = cf
            arch_add(ap, af, child, tuple(cf))
    return af


METHODS = [
    ("NSGA-II (real-coded)", "random key", "GA", run_nsga_realcoded),
    ("Random restart", "random key", "---", run_random),
    ("MOHHO (real-coded)", "random key", "swarm", run_hho_realcoded),
    ("Discrete-MOHHO", "permutation", "swarm", run_discrete_mohho),
    ("perm-MOEA/D", "permutation", "decomp.", run_permmoead),
    ("perm-NSGA-II", "permutation", "GA", run_permnsga),
]


def main():
    prob = MOMKP(seed=7)
    print(f"MOMKP: N={N} items, {MCAP} capacities, caps={prob.cap.astype(int)}")
    t0 = time.time(); out = {}
    for name, enc, par, fn in METHODS:
        hv, allf = [], []
        for s in SEEDS:
            af = fn(prob, s); hv.append(compute_hypervolume(af, REF)); allf += af
        comb = nondominated(allf)
        out[name] = {"encoding": enc, "paradigm": par,
                     "per_run_hv": hv, "hv_mean": float(np.mean(hv)), "hv_std": float(np.std(hv)),
                     "cv": float(np.std(hv) / np.mean(hv)),
                     "combined_hv": compute_hypervolume(comb, REF), "combined_sols": len(comb)}
        print(f"  {name:22s} HV={np.mean(hv):.4f}+/-{np.std(hv):.4f} CV={100*np.std(hv)/np.mean(hv):.2f}% "
              f"comb={compute_hypervolume(comb,REF):.4f} ({len(comb)})  ({time.time()-t0:.0f}s)")
    json.dump({"problem": "MOMKP (N=120, 4 capacities, 3 objectives)", "ref_point": REF,
               "budget": POP * GEN, "seeds": SEEDS, "methods": out, "elapsed_s": time.time() - t0},
              open(RESULTS / "second_problem.json", "w"), indent=2)
    print(f"\n-> second_problem.json  ({time.time()-t0:.0f}s)")
    rk = [m for m in out if out[m]["encoding"] == "random key"]
    pn = [m for m in out if out[m]["encoding"] == "permutation"]
    print(f"random-key tier: {max(out[m]['hv_mean'] for m in rk):.4f} (best) | "
          f"permutation tier: {min(out[m]['hv_mean'] for m in pn):.4f} (worst)")
    print("REGULARITY REPLICATES?" , min(out[m]['hv_mean'] for m in pn) > max(out[m]['hv_mean'] for m in rk))


if __name__ == "__main__":
    main()
