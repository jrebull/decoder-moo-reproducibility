"""
REVIEWER CONTROL ("did you cripple HHO?"): the main study equalizes the
function-evaluation (FE) budget by letting each hawk issue ONE trial point per
iteration, whereas canonical HHO's rapid dive (OP5/OP6) evaluates TWO and keeps
the better. A reviewer could argue the single-trial choice weakened HHO and that
its loss to random restart is an artifact. We test this directly two ways:

  (1) Canonical two-point HHO at its NATIVE 500-iteration schedule -- which costs
      ~42% MORE evaluations than every competitor (already reflected in the old
      35,552-FE run); and
  (2) Canonical two-point HHO at the COMMON 25,050-FE budget (iterations reduced
      so total evaluations match the other methods), with full energy decay.

If neither beats blind random restart (309,821), the "MOHHO does not beat the
decoder" finding is robust to the dive convention -- the single-trial choice is
not what sinks the swarm.

Output: app/data/results/control_canonical_hho.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

import app.core.mohho as mh
from app.core.mohho import (update_archive, compute_hypervolume, dominates,
                            _step_hawk, LB, UB)
from app.core.config import NUM_GROUPS, POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE
from app.core.problem import VisaProblem

RESULTS = Path("app/data/results")
SEEDS = list(range(1, 31))
POP, ARC = POPULATION_SIZE, ARCHIVE_SIZE
BUDGET = POP * MAX_ITERATIONS + POP        # 25,050 -- the common FE budget
# random_restart_hv y fe_fair_mohho_hv se DERIVAN en main() de controls.json /
# summary.json (canonicos, seeds 1-30) -> cero hardcode, sin drift de bloque.

_orig_eval = mh.evaluate_hawk


def canonical_levy(xi, fit_i, y, z, problem):
    """Restore the canonical two-point rapid dive: eval y, then z, keep the better."""
    cands = []
    _, fy = mh.evaluate_hawk(y, problem); cands.append((y, fy))
    if dominates(fy, fit_i):
        return (y, fy), cands
    _, fz = mh.evaluate_hawk(z, problem); cands.append((z, fz))
    if dominates(fz, fit_i):
        return (z, fz), cands
    return None, cands


def run_canonical(problem, seed, max_iter, fe_cap=None):
    """Canonical two-point HHO; optional hard FE cap (stop mid-run when reached)."""
    cnt = {"n": 0}
    def counting(h, p):
        cnt["n"] += 1
        return _orig_eval(h, p)
    mh.evaluate_hawk = counting
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LB, UB, size=(POP, NUM_GROUPS))
    fits = [counting(pop[i], problem)[1] for i in range(POP)]
    ap, af = [], []
    for i in range(POP):
        update_archive(ap, af, pop[i], fits[i], ARC, rng)
    for t in range(max_iter):
        xm = np.mean(pop, axis=0)
        for i in range(POP):
            _step_hawk(i, pop, fits, ap, af, xm, t, max_iter, POP, ARC, problem, rng)
            if fe_cap and cnt["n"] >= fe_cap:
                break
        if fe_cap and cnt["n"] >= fe_cap:
            break
    mh.evaluate_hawk = _orig_eval
    return af, cnt["n"]


def main():
    p = VisaProblem()
    mh._greedy_select_levy = canonical_levy        # canonical 2-point dive
    # constantes derivadas de los JSON canonicos (seeds 1-30), sin hardcode
    random_restart_hv = json.load(open(RESULTS / "controls.json"))["random_restart"]["stats"]["mean"]
    fe_fair_mohho_hv = json.load(open(RESULTS / "summary.json"))["hv_stats"]["mean"]
    out = {"budget": BUDGET, "random_restart_hv": round(random_restart_hv, 1),
           "fe_fair_mohho_hv": round(fe_fair_mohho_hv, 1), "seeds": SEEDS}

    # (1) native 500-iter schedule (over-budget) and (2) FE-capped to 25,050
    for tag, max_iter, cap in [("native500", MAX_ITERATIONS, None),
                               ("fe_matched", MAX_ITERATIONS, BUDGET)]:
        hv, fes = [], []
        for s in SEEDS:
            af, n = run_canonical(p, s, max_iter, cap)
            hv.append(compute_hypervolume(af)); fes.append(n)
        hv = np.array(hv)
        # paired Wilcoxon vs random restart per-seed values (controls.json)
        ctl = json.load(open(RESULTS / "controls.json"))
        rr = np.array(ctl["random_restart"]["per_seed_hv"])
        w, pgr = wilcoxon(hv, rr, alternative="greater")   # canonical HHO > random?
        out[tag] = {
            "mean_fe": float(np.mean(fes)), "hv_mean": float(hv.mean()),
            "hv_std": float(hv.std()),
            "vs_random_pct": float(100 * (hv.mean() - rr.mean()) / rr.mean()),
            "beats_random": bool(hv.mean() > rr.mean()),
            "wilcoxon_p_canonical_gt_random": float(pgr),
            "canonical_better_than_random_count": int((hv > rr).sum()),
        }
        print(f"[{tag}] mean FE {np.mean(fes):,.0f} | HV {hv.mean():,.0f}±{hv.std():,.0f} "
              f"| vs random {out[tag]['vs_random_pct']:+.1f}% beats={out[tag]['beats_random']} "
              f"(Wilcoxon p[canon>rand]={pgr:.3f}, better {out[tag]['canonical_better_than_random_count']}/30)")

    json.dump(out, open(RESULTS / "control_canonical_hho.json", "w"), indent=2)
    print(f"\nrandom restart HV = {random_restart_hv:,.0f} ; FE-fair single-trial MOHHO = {fe_fair_mohho_hv:,.0f}")
    print("saved control_canonical_hho.json")


if __name__ == "__main__":
    main()
