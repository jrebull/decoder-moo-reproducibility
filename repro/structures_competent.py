"""
structures_competent.py (v6 FASE 2) — anade el MO-HHO COMPETENTE (random-key:
HHO + non-dominated sorting + mutacion) a la Tabla 8 de 4 estructuras, recomputando
los rangos de Friedman a 7 metodos. Reusa los HV per-seed YA guardados de los 6
metodos (visa: ladder_v5.json; knapsack: second_problem.json; TSP/flow-shop:
more_structures.json) y solo corre el competente nuevo (3 estructuras x 30 seeds;
visa ya esta en ladder_v5). Mismo presupuesto 25,000 evals, seeds 1..30.

Salida: app/data/results/structures_v6.json
Env: POP(50) GEN(500) SEEDS(30) PM(0.15) FAST(0 -> usa POP/GEN reducidos si =1)
"""
import os, json, time
from pathlib import Path
import numpy as np
from scipy.stats import friedmanchisquare, rankdata
import _bootstrap; _bootstrap.bootstrap_engine()

from app.core.mohho import compute_hypervolume
import competent_mohho as C
from second_problem import MOMKP
from more_structures import MOTSP, MOPFSP

POP = int(os.environ.get("POP", 50)); GEN = int(os.environ.get("GEN", 500))
SEEDS = int(os.environ.get("SEEDS", 30)); PM = float(os.environ.get("PM", 0.15))
REF = (1.0, 1.0, 1.0)
RESULTS = Path(_bootstrap.results_dir())
seeds = list(range(1, 1 + SEEDS))

# canonical 6-method labels as stored, mapped to a short name
SIX = ["NSGA-II (real-coded)", "Random restart", "MOHHO (real-coded)",
       "Discrete-MOHHO", "perm-MOEA/D", "perm-NSGA-II"]
COMP = "competent MO-HHO (rk)"


def run_competent_on(prob):
    hv = []
    for s in seeds:
        front = C.run_competent_mohho(prob.eval_keys, prob.n, 3,
                                      lambda F: compute_hypervolume(F, REF),
                                      s, POP, GEN, pm=PM, use_sbx=True)["front"]
        hv.append(compute_hypervolume(front, REF))
    return hv


def avg_ranks(per_seed):
    """per_seed: {method: [hv per seed]}; rank 1 = best (highest HV) each seed."""
    methods = list(per_seed.keys())
    H = np.array([per_seed[m] for m in methods])      # methods x seeds
    ranks = np.zeros_like(H)
    for j in range(H.shape[1]):
        ranks[:, j] = rankdata(-H[:, j])               # higher HV -> rank 1
    return {m: float(ranks[i].mean()) for i, m in enumerate(methods)}


def main():
    t0 = time.time()
    out = {"budget": {"pop": POP, "gen": GEN, "evals": POP * GEN, "seeds": seeds},
           "competent_cfg": {"pm": PM, "use_sbx": True}, "structures": {}}

    # ---- VISA: 7 methods already in ladder_v5 (per-seed HV) ----
    lv5 = json.load(open(RESULTS / "ladder_v5.json"))["methods"]
    name_map_v5 = {"nsga2_realcoded": "NSGA-II (real-coded)", "naive_mohho": "MOHHO (real-coded)",
                   "competent_mohho": COMP, "random_restart": "Random restart",
                   "perm_nsga2": "perm-NSGA-II", "perm_moead": "perm-MOEA/D",
                   "discrete_mohho": "Discrete-MOHHO"}
    visa_ps = {name_map_v5[k]: lv5[k]["hv_per_seed"] for k in lv5}
    out["structures"]["visa"] = {"avg_rank": avg_ranks(visa_ps),
                                 "competent_rank": None, "note": "from ladder_v5 (7 methods)"}

    # ---- knapsack (MOMKP): 6 stored + competent fresh ----
    spm = json.load(open(RESULTS / "second_problem.json"))["methods"]
    print("running competent on MOMKP..."); comp_mk = run_competent_on(MOMKP())
    mk_ps = {m: spm[m]["per_run_hv"] for m in SIX}; mk_ps[COMP] = comp_mk
    out["structures"]["knapsack"] = {"avg_rank": avg_ranks(mk_ps), "competent_hv_mean": float(np.mean(comp_mk))}

    # ---- TSP (MOTSP) ----
    ms = json.load(open(RESULTS / "more_structures.json"))
    print("running competent on MOTSP..."); comp_tsp = run_competent_on(MOTSP())
    tsp_ps = {m: ms["mo-TSP"]["methods"][m]["per_run_hv"] for m in SIX}; tsp_ps[COMP] = comp_tsp
    out["structures"]["TSP"] = {"avg_rank": avg_ranks(tsp_ps), "competent_hv_mean": float(np.mean(comp_tsp))}

    # ---- flow-shop (MOPFSP) ----
    print("running competent on MOPFSP..."); comp_pf = run_competent_on(MOPFSP())
    pf_ps = {m: ms["mo-PFSP"]["methods"][m]["per_run_hv"] for m in SIX}; pf_ps[COMP] = comp_pf
    out["structures"]["flow-shop"] = {"avg_rank": avg_ranks(pf_ps), "competent_hv_mean": float(np.mean(comp_pf))}

    # ---- sanity: recomputed 6-method ranks vs stored avg_rank ----
    sanity = {}
    for struct, stored_key, ps6 in [
        ("knapsack", None, {m: spm[m]["per_run_hv"] for m in SIX}),
        ("TSP", "mo-TSP", {m: ms["mo-TSP"]["methods"][m]["per_run_hv"] for m in SIX}),
        ("flow-shop", "mo-PFSP", {m: ms["mo-PFSP"]["methods"][m]["per_run_hv"] for m in SIX})]:
        recomputed = avg_ranks(ps6)
        sanity[struct] = recomputed
    out["sanity_recomputed_6method_ranks"] = sanity

    # ---- where does the competent place / is the old claim still true? ----
    placement = {}
    for st, d in out["structures"].items():
        ar = d["avg_rank"]
        order = sorted(ar, key=lambda m: ar[m])
        comp_pos = order.index(COMP) + 1
        best = order[0]
        # is a permutation-native method still strictly best? (old claim)
        perm_best = best in ("perm-NSGA-II", "perm-MOEA/D", "Discrete-MOHHO")
        placement[st] = {"competent_position_of_7": comp_pos, "best_method": best,
                         "perm_native_still_best": perm_best,
                         "competent_avg_rank": round(ar[COMP], 3)}
    out["placement"] = placement
    out["elapsed_s"] = round(time.time() - t0, 1)
    (RESULTS / "structures_v6.json").write_text(json.dumps(out, indent=2))
    for st, p in placement.items():
        print(f"  {st:10s} competent rank {p['competent_avg_rank']:.2f} (pos {p['competent_position_of_7']}/7) "
              f"| best={p['best_method']} perm_best={p['perm_native_still_best']}")
    print(f"-> structures_v6.json ({out['elapsed_s']:.0f}s)")


if __name__ == "__main__":
    main()
