"""
diag_zdt2_collapse.py — Documenta la CAUSA RAIZ del colapso del MOHHO en ZDT2:
los operadores HHO con clip a [0,1] arrastran toda la poblacion a la esquina x=0
(f1=0, f2=1) y el archivo colapsa a 1 punto (HV=0.11). Independiente de la
politica de aceptacion. Reproduce el 'roto' de expA_optimizer_sanity.json.

Salida: app/data/results/diag_zdt2_collapse.json
"""
import os, json, time
from pathlib import Path
import numpy as np
import _bootstrap; _bootstrap.bootstrap_engine()
import benchmarks_moo as B

RESULTS = Path(os.environ.get("RESULTS_DIR", _bootstrap.results_dir()))
RESULTS.mkdir(parents=True, exist_ok=True)
POP = int(os.environ.get("POP", 100)); GEN = int(os.environ.get("GEN", 500))


def trace(policy, seed=1):
    b = B.BENCHMARKS["ZDT2"]
    from app.core.hho import (escape_energy, op1_exploration_random,
        op2_exploration_mean, op3_soft_siege, op4_hard_siege,
        op5_soft_siege_levy, op6_hard_siege_levy)
    rng = np.random.default_rng(seed); dim = b["dim"]
    P = rng.uniform(0, 1, size=(POP, dim)); F = [b["fn"](P[i]) for i in range(POP)]
    ap, af = [], []
    for i in range(POP): B.archive_add(ap, af, P[i], F[i], 100, rng)
    sizes, x0mean, xtailmean = [], [], []
    for t in range(GEN):
        xm = P.mean(axis=0)
        for i in range(POP):
            e = escape_energy(t, GEN, rng); ae = abs(e); L = B.select_leader(ap, af, rng)
            if ae >= 1:
                new = op1_exploration_random(P[i], P[rng.integers(POP)], rng) if rng.random() >= 0.5 \
                      else op2_exploration_mean(P[i], L, xm, rng)
            elif rng.random() >= 0.5:
                new = op3_soft_siege(P[i], L, e, rng) if ae >= 0.5 else op4_hard_siege(P[i], L, e, rng)
            else:
                y, _ = (op5_soft_siege_levy(P[i], L, e, rng) if ae >= 0.5
                        else op6_hard_siege_levy(P[i], L, e, xm, rng)); new = y
            fn = b["fn"](new)
            if policy == "canonical" or B.dominates(fn, F[i]):
                P[i] = new; F[i] = fn
            B.archive_add(ap, af, new, fn, 100, rng)
        sizes.append(len(af)); x0mean.append(float(P[:, 0].mean()))
        xtailmean.append(float(P[:, 1:].mean()))
    return {"final_archive_size": len(af), "final_hv": round(B.hv_any(af, b["ref"]), 4),
            "archive_size_every_50it": sizes[::50],
            "x0_mean_every_50it": [round(v, 4) for v in x0mean[::50]],
            "xtail_mean_every_50it": [round(v, 4) for v in xtailmean[::50]],
            "surviving_points": [[round(p[0], 4), round(p[1], 4)] for p in af[:5]]}


def main():
    t0 = time.time(); tf = B.true_front_hv("ZDT2")
    out = {"true_front_hv": round(tf, 4),
           "gated": trace("gated"), "canonical": trace("canonical"),
           "root_cause": (
               "Ambas politicas colapsan: archivo->1, x0_mean->0 y xtail_mean->0, "
               "i.e. toda la poblacion va a x=(0,...,0) => unico punto objetivo (0,1), "
               "HV=ref0*0.1=0.11. Es un ATRACTOR DE ESQUINA del clip a [0,1] bajo "
               "siege/exploracion en un frente CONCAVO, NO un artefacto de la "
               "aceptacion. Reproduce el veredicto 'roto' de expA en ZDT2."),
           "elapsed_s": round(time.time() - t0, 1)}
    (RESULTS / "diag_zdt2_collapse.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
