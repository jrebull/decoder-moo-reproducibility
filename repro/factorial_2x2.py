"""
factorial_2x2.py (v6 FASE 1) — EXPERIMENTO CAUSAL: 2x2 controlado que aisla las dos
condiciones de la regla, sobre UN MISMO esqueleto real-coded (random-key + greedy
decoder), mismo presupuesto (25,000 evals), 30 seeds, seed=1, HV ref identico.

Dos factores binarios, variando SOLO esto:
  A. Operador : order-changing (HHO move + poly-mut, tau~0) vs near-identity (SBX eta=20, tau~0.99)
  B. Seleccion: diversity-preserving (NSGA-II non-dominated sorting + crowding) vs gated (dominance-gated)

4 celdas + random restart baseline (muestreo inline, identico protocolo). Prediccion
falsable: SOLO (order-change ^ NDS) supera a random; se reporta el ANOVA 2-factor.

Salida: app/data/results/factorial_2x2_conditions.json
Env: POP(50) GEN(500) SEEDS(30) PM(0.15)
"""
import os, json, time, statistics as stx
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, f as fdist
import _bootstrap; _bootstrap.bootstrap_engine()

from app.core.problem import VisaProblem
from app.core import mohho as M
from app.core.hho import (escape_energy, op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege, op5_soft_siege_levy, op6_hard_siege_levy)
import benchmarks_moo as B
from competent_mohho import _env_select, _pick_leader, poly_mutate, _sbx

POP = int(os.environ.get("POP", 50)); GEN = int(os.environ.get("GEN", 500))
SEEDS = int(os.environ.get("SEEDS", 30)); PM = float(os.environ.get("PM", 0.15))
RESULTS = Path(_bootstrap.results_dir())
p = VisaProblem(); DIM = M.NUM_GROUPS


def ev(h): return M.evaluate_hawk(h, p)[1]
def HV(F): return M.compute_hypervolume([tuple(x) for x in F])


def order_change_offspring(P, i, FP, first, xmean, t, rng):
    """HHO movement (changes decoded order, tau~0) + polynomial mutation."""
    e = escape_energy(t, GEN, rng); ae = abs(e); L = _pick_leader(P, FP, first, rng)
    if ae >= 1:
        o = (op1_exploration_random(P[i], P[rng.integers(len(P))], rng)
             if rng.random() >= 0.5 else op2_exploration_mean(P[i], L, xmean, rng))
    elif rng.random() >= 0.5:
        o = op3_soft_siege(P[i], L, e, rng) if ae >= 0.5 else op4_hard_siege(P[i], L, e, rng)
    else:
        y, _ = (op5_soft_siege_levy(P[i], L, e, rng) if ae >= 0.5
                else op6_hard_siege_levy(P[i], L, e, xmean, rng))
        o = y
    return poly_mutate(o, rng, PM)


def near_identity_offspring(P, i, FP, first, xmean, t, rng):
    """SBX with a mate (near-identity on decoded order, tau~0.99) + light poly-mut."""
    mate = P[rng.integers(len(P))]
    o = _sbx(P[i], mate, rng, eta=20.0)
    return poly_mutate(o, rng, 1.0 / DIM)


def run_cell(operator, selection, seed):
    """One 2x2 cell. operator in {order,near}; selection in {nds,gated}.
    Budget = POP*GEN single-trial evaluations (FE parity)."""
    rng = np.random.default_rng(seed)
    P = [rng.uniform(0, 1, DIM) for _ in range(POP)]
    FP = [tuple(ev(P[i])) for i in range(POP)]
    arch_pos, arch_fit = [], []
    for i in range(POP):
        B.archive_add(arch_pos, arch_fit, P[i], FP[i], 100, rng)
    moved = []
    for t in range(GEN):
        first = B._fast_nd_sort(FP)[0]; xmean = np.mean(P, axis=0); O = []
        for i in range(POP):
            o = (order_change_offspring(P, i, FP, first, xmean, t, rng) if operator == "order"
                 else near_identity_offspring(P, i, FP, first, xmean, t, rng))
            O.append(o)
        FO = [tuple(ev(o)) for o in O]
        if selection == "nds":
            P, FP = _env_select(P + O, FP + FO, POP)
            moved.append(1.0)
        else:  # gated: offspring replaces parent only if it dominates
            mv = 0
            for i in range(POP):
                if B.dominates(FO[i], FP[i]):
                    P[i] = O[i]; FP[i] = FO[i]; mv += 1
            moved.append(mv / POP)
        for i in range(len(O)):
            B.archive_add(arch_pos, arch_fit, O[i], FO[i], 100, rng)
    return {"hv": HV(arch_fit), "archive": len(arch_fit),
            "moved_fraction_mean": float(np.mean(moved))}


def run_random_cell(seed):
    """Blind random restart, identical decoder+budget, inline (no external import)."""
    rng = np.random.default_rng(seed); ap, af = [], []
    for _ in range(POP * GEN):
        h = rng.uniform(0, 1, DIM)
        M.update_archive(ap, af, h, ev(h), 100, rng)
    return HV(af)


CELLS = {"order_nds": ("order", "nds"), "order_gated": ("order", "gated"),
         "near_nds": ("near", "nds"), "near_gated": ("near", "gated")}


def main():
    t0 = time.time(); seeds = list(range(1, 1 + SEEDS))
    rnd = [run_random_cell(s) for s in seeds]
    rnd_mean = stx.mean(rnd)
    print(f"  random_restart HV={rnd_mean:,.0f}")
    cells = {}; hv_by_cell = {}
    for name, (op, sel) in CELLS.items():
        hvs = []; mvs = []
        for s in seeds:
            r = run_cell(op, sel, s); hvs.append(r["hv"]); mvs.append(r["moved_fraction_mean"])
        hv_by_cell[name] = hvs
        u, pg = mannwhitneyu(hvs, rnd, alternative="greater")
        a12 = float(u / (len(hvs) * len(rnd)))
        cells[name] = {"operator": op, "selection": sel,
            "hv_mean": round(stx.mean(hvs), 1), "hv_std": round(stx.pstdev(hvs), 1),
            "moved_fraction_mean": round(float(np.mean(mvs)), 4),
            "vs_random_pct": round(100 * (stx.mean(hvs) - rnd_mean) / rnd_mean, 2),
            "beats_random": stx.mean(hvs) > rnd_mean,
            "mwu_p_greater_random": float(pg), "A12_vs_random": round(a12, 3)}
        c = cells[name]
        print(f"  {name:12s} op={op:5s} sel={sel:5s} HV={c['hv_mean']:>11,.0f} "
              f"vs_rand={c['vs_random_pct']:+5.2f}% beats={c['beats_random']} "
              f"p={pg:.1e} moved={c['moved_fraction_mean']:.3f}")

    # ---- balanced 2x2 ANOVA (variance decomposition), nrep = SEEDS ----
    A = {n: CELLS[n][0] for n in CELLS}; Bf = {n: CELLS[n][1] for n in CELLS}
    allv = np.array([hv_by_cell[n] for n in CELLS])  # 4 cells x nrep
    grand = allv.mean(); sst = ((allv - grand) ** 2).sum(); nrep = SEEDS

    def mean_where(fac, val):
        idx = [i for i, n in enumerate(CELLS) if (A[n] if fac == "A" else Bf[n]) == val]
        return allv[idx].mean()
    mA = {v: mean_where("A", v) for v in ("order", "near")}
    mB = {v: mean_where("B", v) for v in ("nds", "gated")}
    ssA = 2 * nrep * sum((mA[v] - grand) ** 2 for v in mA)
    ssB = 2 * nrep * sum((mB[v] - grand) ** 2 for v in mB)
    cellmeans = {n: allv[i].mean() for i, n in enumerate(CELLS)}
    sscells = nrep * sum((cellmeans[n] - grand) ** 2 for n in CELLS)
    ssAB = sscells - ssA - ssB; sse = sst - sscells
    eta = lambda ss: round(float(ss / sst), 4)
    dfAB, dfE = 1, 4 * (nrep - 1)
    F_AB = (ssAB / dfAB) / (sse / dfE) if sse > 0 else float("inf")
    p_AB = float(1 - fdist.cdf(F_AB, dfAB, dfE))

    winners = [n for n in CELLS if cells[n]["beats_random"]]
    out = {"budget": {"pop": POP, "gen": GEN, "evals": POP * GEN, "seeds": seeds}, "pm": PM,
        "random_restart": {"hv_mean": round(rnd_mean, 1), "hv_std": round(stx.pstdev(rnd), 1)},
        "cells": cells,
        "anova": {"eta2_operator_A": eta(ssA), "eta2_selection_B": eta(ssB),
                  "eta2_interaction_AxB": eta(ssAB), "eta2_residual": eta(sse),
                  "F_interaction": round(float(F_AB), 2), "p_interaction": p_AB,
                  "interaction_significant": bool(p_AB < 0.05)},
        "winners_vs_random": winners,
        "only_order_nds_wins": bool(winners == ["order_nds"]),
        "elapsed_s": round(time.time() - t0, 1)}
    (RESULTS / "factorial_2x2_conditions.json").write_text(json.dumps(out, indent=2))
    a = out["anova"]
    print(f"\nANOVA eta2: A(op)={a['eta2_operator_A']} B(sel)={a['eta2_selection_B']} "
          f"AxB={a['eta2_interaction_AxB']} (F={a['F_interaction']}, p={a['p_interaction']:.3f}, "
          f"sig={a['interaction_significant']})")
    print(f"winners vs random: {winners} | only_order_nds={out['only_order_nds_wins']}")
    print(f"-> factorial_2x2_conditions.json ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
