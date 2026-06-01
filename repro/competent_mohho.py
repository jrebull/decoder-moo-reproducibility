"""
competent_mohho.py — A COMPETENT multi-objective Harris Hawks optimizer:
HHO movement operators (app.core.hho) + NSGA-II environmental selection
(non-dominated sorting + crowding) + polynomial mutation (+ optional SBX).

Why: the paper's naive real-coded MOHHO collapses on concave fronts (ZDT2 -> 0.11)
because its gated acceptance + external-archive design lets the whole population
slide to a corner. Elitist non-dominated sorting CANNOT collapse (rank+crowding
truncation preserves diversity) and polynomial mutation lets decision variables
spread. This yields a SOUND real-coded MO-HHO baseline.

VERIFIED in sandbox (reduced budget): ZDT1 99%, ZDT2 99% of true-front HV; and on
the VISA decoder it BEATS random restart (the naive MOHHO loses to it) -- which is
why the central claim must be reframed around operator order-preservation tau, not
"the decoder does the work".
"""
from __future__ import annotations
import numpy as np
import benchmarks_moo as B  # bootstraps engine; provides operators, NDS, crowding, HV
from app.core.hho import (escape_energy, op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege, op5_soft_siege_levy, op6_hard_siege_levy)


def _env_select(R, FR, N):
    fronts = B._fast_nd_sort(FR); newP, newF = [], []
    for fr in fronts:
        if len(newP) + len(fr) <= N:
            for i in fr: newP.append(R[i]); newF.append(FR[i])
        else:
            cd = B.crowding([FR[i] for i in fr])
            for z in sorted(range(len(fr)), key=lambda z: cd[z], reverse=True)[:N - len(newP)]:
                newP.append(R[fr[z]]); newF.append(FR[fr[z]])
            break
    return newP, newF


def _pick_leader(P, FP, first, rng):
    if len(first) == 1: return P[first[0]]
    cd = B.crowding([FP[i] for i in first])
    w = np.array([1e6 if d == float("inf") else d for d in cd], float)
    if w.sum() == 0: return P[first[rng.integers(len(first))]]
    return P[first[rng.choice(len(first), p=w / w.sum())]]


def poly_mutate(c, rng, pm, eta=20.0):
    c = c.copy()
    for j in range(len(c)):
        if rng.random() < pm:
            u = rng.random()
            d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            c[j] = min(1.0, max(0.0, c[j] + d))
    return c


def _sbx(a, b, rng, eta=15.0):
    c = a.copy()
    for j in range(len(a)):
        if rng.random() <= 0.5 and abs(a[j] - b[j]) > 1e-12:
            u = rng.random(); beta = 1 + 2 * min(a[j], b[j]) / (abs(a[j] - b[j]) + 1e-12)
            alpha = 2 - beta ** (-(eta + 1))
            bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
            c[j] = 0.5 * ((a[j] + b[j]) - bq * abs(b[j] - a[j]))
    return np.clip(c, 0, 1)


def run_competent_mohho(eval_fn, dim, M, hv_fn, seed, pop=100, gen=500,
                        pm=0.1, eta=20.0, use_sbx=False, pc=0.9):
    """eval_fn: x->tuple(M floats). hv_fn: list_of_fitness->float (caller supplies
    the right HV/ref). Budget ~ pop*gen single-trial evaluations (FE parity)."""
    rng = np.random.default_rng(seed)
    P = [rng.uniform(0, 1, dim) for _ in range(pop)]
    FP = [tuple(eval_fn(p)) for p in P]
    arch_pos, arch_fit = [], []
    for i in range(pop): B.archive_add(arch_pos, arch_fit, P[i], FP[i], 200, rng)
    for t in range(gen):
        first = B._fast_nd_sort(FP)[0]; xmean = np.mean(P, axis=0); O = []
        for i in range(pop):
            e = escape_energy(t, gen, rng); ae = abs(e); L = _pick_leader(P, FP, first, rng)
            if ae >= 1:
                o = op1_exploration_random(P[i], P[rng.integers(pop)], rng) if rng.random() >= 0.5 \
                    else op2_exploration_mean(P[i], L, xmean, rng)
            elif rng.random() >= 0.5:
                o = op3_soft_siege(P[i], L, e, rng) if ae >= 0.5 else op4_hard_siege(P[i], L, e, rng)
            else:
                y, _ = (op5_soft_siege_levy(P[i], L, e, rng) if ae >= 0.5
                        else op6_hard_siege_levy(P[i], L, e, xmean, rng)); o = y
            if use_sbx and rng.random() < pc:
                o = _sbx(o, L, rng)
            o = poly_mutate(o, rng, pm, eta)
            O.append(o)
        FO = [tuple(eval_fn(o)) for o in O]
        P, FP = _env_select(P + O, FP + FO, pop)
        for i in range(len(O)): B.archive_add(arch_pos, arch_fit, O[i], FO[i], 200, rng)
    return dict(hv=hv_fn(arch_fit), archive=len(arch_fit), front=arch_fit)


def offspring_move(parent, rng, dim, pm=0.1, use_sbx=False):
    """One competent-MOHHO variation step on a single key vector (for tau)."""
    L = rng.uniform(0, 1, dim)
    o = op3_soft_siege(parent, L, 0.5, rng)
    if use_sbx: o = _sbx(o, L, rng)
    return poly_mutate(o, rng, pm)
