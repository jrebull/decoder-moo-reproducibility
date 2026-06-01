"""
diag_mohho_freeze.py — Mide si la poblacion del MOHHO real-coded se CONGELA bajo la
regla de aceptacion dominance-gated del codigo actual, en el problema visa.

Reproduce el step-loop del motor (app.core.mohho) instrumentado para registrar, por
iteracion: fraccion de hawks que cambian de posicion y desplazamiento L2 medio.
Compara contra una variante CANONICA (mover siempre) y contra random restart, al
MISMO presupuesto.

Salida: app/data/results/diag_mohho_freeze.json
Presupuesto via env: POP (def 50), ITER (def 500), SEEDS (def 30). Para prueba
rapida: POP=20 ITER=80 SEEDS=3 python diag_mohho_freeze.py
"""
import os, json, time
from pathlib import Path
import numpy as np
import _bootstrap; _bootstrap.bootstrap_engine()

from app.core.problem import VisaProblem
from app.core import mohho as M
from app.core import hho as H

POP   = int(os.environ.get("POP", 50))
ITER  = int(os.environ.get("ITER", 500))
SEEDS = int(os.environ.get("SEEDS", 30))
SEED_BASE = int(os.environ.get("SEED_BASE", 1))
ARCH = 100
RESULTS = Path(os.environ.get("RESULTS_DIR", _bootstrap.results_dir()))
RESULTS.mkdir(parents=True, exist_ok=True)


def run_instrumented(problem, seed, pop, it, canonical):
    rng = np.random.default_rng(seed)
    dim = M.NUM_GROUPS
    P = rng.uniform(0, 1, size=(pop, dim))
    F = [M.evaluate_hawk(P[i], problem)[1] for i in range(pop)]
    ap, af = [], []
    for i in range(pop):
        M.update_archive(ap, af, P[i], F[i], ARCH, rng)
    moved, disp = [], []
    for t in range(it):
        xm = P.mean(axis=0); prev = P.copy()
        for i in range(pop):
            e = H.escape_energy(t, it, rng); ae = abs(e)
            L = M.select_leader(ap, af, rng)
            if ae >= 1:
                new = M._exploration_step(P, i, L, xm, pop, rng)
            elif rng.random() >= 0.5:
                new = M._siege_step(P[i], L, e, ae, rng)
            else:
                if ae >= 0.5: y, _ = H.op5_soft_siege_levy(P[i], L, e, rng)
                else:         y, _ = H.op6_hard_siege_levy(P[i], L, e, xm, rng)
                new = y
            fnew = M.evaluate_hawk(new, problem)[1]
            if canonical or M.dominates(fnew, F[i]):
                P[i] = new; F[i] = fnew
            M.update_archive(ap, af, new, fnew, ARCH, rng)
        d = np.linalg.norm(P - prev, axis=1)
        moved.append(float(np.mean(d > 1e-9))); disp.append(float(np.mean(d)))
    return dict(hv=M.compute_hypervolume(af), archive=len(af),
                moved_fraction_per_iter_mean=float(np.mean(moved)),
                mean_displacement_per_iter=float(np.mean(disp)),
                moved_first10=moved[:10])


def run_random(problem, seed, budget):
    rng = np.random.default_rng(seed); ap, af = [], []
    for _ in range(budget):
        h = rng.uniform(0, 1, size=M.NUM_GROUPS)
        f = M.evaluate_hawk(h, problem)[1]
        M.update_archive(ap, af, h, f, ARCH, rng)
    return M.compute_hypervolume(af), len(af)


def main():
    t0 = time.time(); p = VisaProblem()
    gated, canon, rand = [], [], []
    for s in range(SEED_BASE, SEED_BASE + SEEDS):
        gated.append(run_instrumented(p, s, POP, ITER, canonical=False))
        canon.append(run_instrumented(p, s, POP, ITER, canonical=True))
        hvr, _ = run_random(p, s, POP * ITER); rand.append(hvr)
    def agg(rs, k): return float(np.mean([r[k] for r in rs]))
    out = {
        "budget": {"pop": POP, "iter": ITER, "evals": POP * ITER, "seeds": SEEDS, "seed_base": SEED_BASE},
        "gated_current_code": {
            "hv_mean": agg(gated, "hv"),
            "moved_fraction_per_iter_mean": agg(gated, "moved_fraction_per_iter_mean"),
            "mean_displacement_per_iter": agg(gated, "mean_displacement_per_iter"),
            "archive_mean": agg(gated, "archive")},
        "canonical_always_move": {
            "hv_mean": agg(canon, "hv"),
            "moved_fraction_per_iter_mean": agg(canon, "moved_fraction_per_iter_mean"),
            "mean_displacement_per_iter": agg(canon, "mean_displacement_per_iter"),
            "archive_mean": agg(canon, "archive")},
        "random_restart": {"hv_mean": float(np.mean(rand))},
        "interpretation": (
            "Si gated.moved_fraction ~ 0.01 => la poblacion esta CONGELADA por la "
            "regla dominance-gated: la propuesta HHO salta pero el gate la rechaza. "
            "El texto del paper (pag.14, 'position updates induce large permutation "
            "jumps so the swarm traverses the space') describe la PROPUESTA, no la "
            "trayectoria realizada. Reconciliar."),
        "elapsed_s": round(time.time() - t0, 1)}
    (RESULTS / "diag_mohho_freeze.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
