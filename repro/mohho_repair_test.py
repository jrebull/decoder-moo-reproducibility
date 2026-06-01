"""
mohho_repair_test.py (FASE 1.1) -- prueba reparaciones del atractor de esquina del
MOHHO real-coded en frentes concavos (ZDT2). Reusa los operadores REALES del motor
via benchmarks_moo. Tres reparaciones combinables:

  R1 reflexion-en-bordes : reemplaza clip a [0,1] por reflexion triangular (sin
                           atractor de esquina). Se aplica monkeypatch a
                           app.core.hho.clip_bounds, de modo que los SEIS operadores
                           reales reboten en vez de pegarse a 0/1.
  R2 archivo eps-dominancia: rejilla de epsilon-cajas para preservar diversidad en
                           frentes concavos (evita archivo=1).
  R3 damping del siege   : limita el paso (new = cur + gamma*(new-cur)) para evitar
                           convergencia prematura a la esquina.

Adopta la MINIMA combinacion con HV/true >= 0.90 en ZDT1, ZDT2 y DTLZ2.
Salida: app/data/results/mohho_repair_selection.json
Presupuesto via env: POP (def 60), GEN (def 300), SEEDS (def 3).
"""
import os, json, time, itertools
from pathlib import Path
import numpy as np

import _bootstrap; _bootstrap.bootstrap_engine()
import app.core.hho as H
import benchmarks_moo as B

POP = int(os.environ.get("POP", 60))
GEN = int(os.environ.get("GEN", 300))
SEEDS = int(os.environ.get("SEEDS", 3))
RESULTS = Path(_bootstrap.results_dir())
THR = 0.90


# ---------------- R1: reflexion triangular en [0,1] ----------------
_clip_orig = H.clip_bounds
def _reflect(x):
    x = np.abs(np.asarray(x, float)) % 2.0      # fold into [0,2)
    return np.where(x > 1.0, 2.0 - x, x)         # fold [1,2)->[0,1]


# ---------------- R2: archivo con epsilon-dominancia ----------------
def _eps_box(fit, eps):
    return tuple(int(np.floor(v / eps)) for v in fit)

def _eps_dom(a_box, b_box):
    return (all(a_box[m] <= b_box[m] for m in range(len(a_box)))
            and any(a_box[m] < b_box[m] for m in range(len(a_box))))

def archive_add_eps(pos_list, fit_list, new_pos, new_fit, max_size, rng, eps):
    """Epsilon-dominance archive (Laumanns et al.): one point per eps-box, no
    index tracking (robust). Preserves spread on concave fronts."""
    nb = _eps_box(new_fit, eps)
    for f in fit_list:
        fb = _eps_box(f, eps)
        if B.dominates(f, new_fit) or _eps_dom(fb, nb):
            return
        if fb == nb and sum(f) <= sum(new_fit):
            return
    keep = []
    for f, p in zip(fit_list, pos_list):
        fb = _eps_box(f, eps)
        removed = (B.dominates(new_fit, f) or _eps_dom(nb, fb)
                   or (fb == nb and sum(new_fit) < sum(f)))
        if not removed:
            keep.append((f, p))
    pos_list[:] = [p for _, p in keep]
    fit_list[:] = [f for f, _ in keep]
    pos_list.append(np.asarray(new_pos).copy()); fit_list.append(tuple(new_fit))
    if len(pos_list) > max_size:
        cd = B.crowding(fit_list)
        fin = [i for i in range(len(cd)) if cd[i] != float("inf")]
        j = min(fin, key=lambda i: cd[i]) if fin else 0
        pos_list.pop(j); fit_list.pop(j)


def run_repaired(eval_fn, dim, M, ref, seed, repairs, pop=POP, gen=GEN,
                 archive_size=100, gamma=0.5, eps=0.02):
    """MOHHO (canonical acceptance) con reparaciones seleccionables."""
    use_R1, use_R2, use_R3 = ("R1" in repairs), ("R2" in repairs), ("R3" in repairs)
    if use_R1:
        H.clip_bounds = _reflect
    try:
        rng = np.random.default_rng(seed)
        P = rng.uniform(0, 1, size=(pop, dim))
        F = [tuple(eval_fn(P[i])) for i in range(pop)]
        ap, af = [], []
        def add(p, f):
            if use_R2: archive_add_eps(ap, af, p, f, archive_size, rng, eps)
            else: B.archive_add(ap, af, p, f, archive_size, rng)
        for i in range(pop): add(P[i], F[i])
        for t in range(gen):
            xm = P.mean(axis=0)
            for i in range(pop):
                e = H.escape_energy(t, gen, rng); ae = abs(e)
                L = B.select_leader(ap, af, rng)
                if ae >= 1:
                    new = (H.op1_exploration_random(P[i], P[rng.integers(pop)], rng)
                           if rng.random() >= 0.5 else
                           H.op2_exploration_mean(P[i], L, xm, rng))
                elif rng.random() >= 0.5:
                    new = (H.op3_soft_siege(P[i], L, e, rng) if ae >= 0.5
                           else H.op4_hard_siege(P[i], L, e, rng))
                else:
                    if ae >= 0.5: new, _ = H.op5_soft_siege_levy(P[i], L, e, rng)
                    else:         new, _ = H.op6_hard_siege_levy(P[i], L, e, xm, rng)
                if use_R3:
                    new = P[i] + gamma * (np.asarray(new) - P[i])
                    new = _reflect(new) if use_R1 else np.clip(new, 0, 1)
                fnew = tuple(eval_fn(new))
                P[i] = new; F[i] = fnew          # canonical acceptance
                add(new, fnew)
        return B.hv_any(af, ref), len(af)
    finally:
        H.clip_bounds = _clip_orig


def main():
    t0 = time.time()
    seeds = list(range(1, 1 + SEEDS))
    combos = [(), ("R1",), ("R2",), ("R3",), ("R1", "R2"), ("R1", "R3"),
              ("R1", "R2", "R3")]
    true_hv = {n: B.true_front_hv(n) for n in B.BENCHMARKS}
    out = {"budget": {"pop": POP, "gen": GEN, "seeds": seeds},
           "threshold_hv_over_true": THR, "true_front_hv": {n: round(true_hv[n], 4) for n in true_hv},
           "variants": {}}
    for combo in combos:
        key = "+".join(combo) if combo else "baseline(clip)"
        per = {}
        for n, b in B.BENCHMARKS.items():
            hv = [run_repaired(b["fn"], b["dim"], b["M"], b["ref"], s, combo)[0]
                  for s in seeds]
            per[n] = round(float(np.mean(hv)) / true_hv[n], 4)
        per["min_over_three"] = round(min(per.values()), 4)
        per["sane_all"] = bool(per["min_over_three"] >= THR)
        out["variants"][key] = per
        print(f"  {key:18s} ZDT1={per['ZDT1']*100:5.1f}% ZDT2={per['ZDT2']*100:5.1f}% "
              f"DTLZ2={per['DTLZ2']*100:5.1f}% min={per['min_over_three']*100:5.1f}% sane={per['sane_all']}")
    # minimal sane combo (fewest repairs; ties -> R1<R2<R3 order)
    sane = [(c, out["variants"]["+".join(c) if c else "baseline(clip)"])
            for c in combos if out["variants"]["+".join(c) if c else "baseline(clip)"]["sane_all"]]
    sane.sort(key=lambda x: (len(x[0]), x[0]))
    out["adopted"] = ("+".join(sane[0][0]) if sane and sane[0][0] else
                      (sane[0][0] if sane else None))
    out["any_sane"] = bool(sane)
    out["recommendation"] = (
        f"Adoptar la minima reparacion sana: {out['adopted']}." if sane else
        "NINGUNA reparacion razonable sanea ZDT2 -> SCOPING honesto (declarar el "
        "MOHHO real-coded como baseline que colapsa en frentes concavos).")
    out["elapsed_s"] = round(time.time() - t0, 1)
    (RESULTS / "mohho_repair_selection.json").write_text(json.dumps(out, indent=2))
    print(f"\nADOPTED: {out['adopted']} | any_sane={out['any_sane']} | {out['elapsed_s']:.0f}s")
    print("->", RESULTS / "mohho_repair_selection.json")


if __name__ == "__main__":
    main()
