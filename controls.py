"""
Ablation controls requested by the reviewer, to attribute MOHHO's edge to the
SEARCH rather than to the decoder or the archive:

  (A) Random-key / random-restart baseline -- pure random search under an
      IDENTICAL function-evaluation budget and the SAME crowding-pruned archive.
      Answers: does the swarm earn its keep over decoding random permutations?

  (B) NSGA-II + external archive -- NSGA-II augmented with the SAME size-100
      crowding-pruned external archive as MOHHO. Answers: is MOHHO's gap a
      search advantage or an archive artifact?

  (C) Hypervolume reference-point sensitivity -- recompute the combined-front HV
      ranking under several reference points (especially varying the f1 bound),
      since f1 has a narrow range. Answers: is the ranking robust to the HV ref?

Output: app/data/results/controls.json
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

from app.core.config import LB, UB, NUM_GROUPS, POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE
from app.core.problem import VisaProblem
from app.core.mohho import (evaluate_hawk, compute_hypervolume, dominates,
                            update_archive, HV_REF_POINT)
from compare_nsga2 import (run_nsga2, fast_nondominated_sort, sbx, poly_mutate,
                           tournament, nondominated, SEEDS as NSGA_SEEDS)
from app.core.mohho import crowding_distance

RESULTS = Path("app/data/results")
MOHHO_SEEDS = list(range(1, 31))            # same seeds as the main MOHHO study
POP, GEN = POPULATION_SIZE, MAX_ITERATIONS  # 50, 500
BUDGET = POP * GEN                           # matched function-evaluation budget


# ---------------- (A) Random-restart baseline ----------------
def run_random(problem, seed, budget=BUDGET, archive_size=ARCHIVE_SIZE):
    """Pure random search: draw `budget` random hawks, decode, feed the same
    crowding-pruned archive. No operators, no population dynamics."""
    rng = np.random.default_rng(seed)
    ap, af = [], []
    for _ in range(budget):
        h = rng.uniform(LB, UB, size=NUM_GROUPS)
        _, fit = evaluate_hawk(h, problem)
        update_archive(ap, af, h, fit, archive_size, rng)
    return af


# ---------------- (B) NSGA-II + external archive ----------------
def run_nsga2_archive(problem, seed, archive_size=ARCHIVE_SIZE):
    """NSGA-II with the SAME size-100 crowding-pruned external archive as MOHHO."""
    rng = np.random.default_rng(seed)
    pop = rng.uniform(LB, UB, size=(POP, NUM_GROUPS))
    fits = [evaluate_hawk(pop[i], problem)[1] for i in range(POP)]
    ap, af = [], []
    for i in range(POP):
        update_archive(ap, af, pop[i], fits[i], archive_size, rng)
    for _ in range(GEN):
        fronts, rank = fast_nondominated_sort(fits)
        cd = [0.0] * POP
        for fr in fronts:
            d = crowding_distance([fits[i] for i in fr])
            for k, idx in enumerate(fr):
                cd[idx] = d[k]
        off = []
        while len(off) < POP:
            p1 = pop[tournament(rank, cd, rng)]
            p2 = pop[tournament(rank, cd, rng)]
            if rng.random() <= 0.9:
                c1, c2 = sbx(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            off.append(poly_mutate(c1, rng))
            if len(off) < POP:
                off.append(poly_mutate(c2, rng))
        off = np.array(off)
        off_fits = [evaluate_hawk(off[i], problem)[1] for i in range(POP)]
        for i in range(POP):
            update_archive(ap, af, off[i], off_fits[i], archive_size, rng)
        comb = np.vstack([pop, off]); comb_fits = fits + off_fits
        fronts, _ = fast_nondominated_sort(comb_fits)
        new_idx = []
        for fr in fronts:
            if len(new_idx) + len(fr) <= POP:
                new_idx += fr
            else:
                d = crowding_distance([comb_fits[i] for i in fr])
                order = sorted(range(len(fr)), key=lambda k: d[k], reverse=True)
                new_idx += [fr[k] for k in order[:POP - len(new_idx)]]
                break
        pop = comb[new_idx]; fits = [comb_fits[i] for i in new_idx]
    return af  # external archive front


def stats(hvs):
    a = np.asarray(hvs, float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def main():
    problem = VisaProblem()
    t0 = time.time()
    st = json.load(open(RESULTS / "stats_test.json"))
    mohho_hv = st["mohho_hv"]                       # seeds 42-71 (existing)

    # ---- (A) random-restart, paired on the SAME seeds as MOHHO ----
    rand_hv, rand_all = [], []
    for s in MOHHO_SEEDS:
        af = run_random(problem, s)
        rand_hv.append(compute_hypervolume(af)); rand_all += af
        print(f"random seed {s}: {len(af)} sols HV={rand_hv[-1]:,.0f}")
    rand_front = nondominated(rand_all)

    # paired test MOHHO vs random (same seeds) -> Wilcoxon signed-rank, one-sided
    w_stat, w_p = wilcoxon(np.array(mohho_hv), np.array(rand_hv), alternative="greater")

    # ---- (B) NSGA-II + archive ----
    nsga_arch_hv, nsga_arch_all = [], []
    for s in NSGA_SEEDS:
        af = run_nsga2_archive(problem, s)
        nsga_arch_hv.append(compute_hypervolume(af)); nsga_arch_all += af
        print(f"nsga+archive seed {s}: {len(af)} sols HV={nsga_arch_hv[-1]:,.0f}")
    nsga_arch_front = nondominated(nsga_arch_all)

    # MOHHO vs NSGA+archive (independent seeds) -> Mann-Whitney one-sided
    mw_u, mw_p = mannwhitneyu(np.array(mohho_hv), np.array(nsga_arch_hv),
                              alternative="greater")
    A12 = float(mw_u / (len(mohho_hv) * len(nsga_arch_hv)))

    # ---- (C) HV reference-point sensitivity on the combined fronts ----
    import csv
    mohho_front = []
    for r in csv.DictReader(open(RESULTS / "pareto_front.csv")):
        if r["type"] == "pareto":
            mohho_front.append((float(r["f1"]), float(r["f2"]), float(r["f3"])))
    nsga_front = [tuple(p) for p in json.load(open(RESULTS / "nsga2_front.json"))["front"]]

    refs = {
        "baseline_(10,16,20000)": (10.0, 16.0, 20000.0),
        "tight_f1_(9.5,16,20000)": (9.5, 16.0, 20000.0),
        "loose_f1_(11,16,20000)": (11.0, 16.0, 20000.0),
        "tight_all_(9.5,14,10000)": (9.5, 14.0, 10000.0),
        "loose_all_(12,18,30000)": (12.0, 18.0, 30000.0),
    }
    ref_sweep = {}
    for name, rp in refs.items():
        hv_m = compute_hypervolume(mohho_front, rp)
        hv_n = compute_hypervolume(nsga_front, rp)
        hv_r = compute_hypervolume(rand_front, rp)
        ref_sweep[name] = {
            "mohho": hv_m, "nsga2": hv_n, "random": hv_r,
            "mohho_gt_nsga2": hv_m > hv_n, "mohho_gt_random": hv_m > hv_r,
            "mohho_vs_nsga2_pct": 100 * (hv_m - hv_n) / hv_n if hv_n else None,
        }

    out = {
        "budget_evals": BUDGET,
        "random_restart": {
            "per_seed_hv": rand_hv, "stats": stats(rand_hv),
            "combined_front_size": len(rand_front),
            "combined_front_hv": compute_hypervolume(rand_front),
            "seeds": MOHHO_SEEDS,
        },
        "mohho_vs_random_paired_wilcoxon": {
            "statistic": float(w_stat), "p_one_sided": float(w_p),
            "mohho_median": float(np.median(mohho_hv)),
            "random_median": float(np.median(rand_hv)),
            "mohho_better_count": int(np.sum(np.array(mohho_hv) > np.array(rand_hv))),
        },
        "nsga2_archive": {
            "per_seed_hv": nsga_arch_hv, "stats": stats(nsga_arch_hv),
            "combined_front_size": len(nsga_arch_front),
            "combined_front_hv": compute_hypervolume(nsga_arch_front),
            "seeds": NSGA_SEEDS,
        },
        "mohho_vs_nsga2archive_mannwhitney": {
            "U": float(mw_u), "p_one_sided": float(mw_p), "A12": A12,
        },
        "hv_reference_sensitivity": ref_sweep,
        "elapsed_s": time.time() - t0,
    }
    json.dump(out, open(RESULTS / "controls.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"random-restart HV mean {out['random_restart']['stats']['mean']:,.0f} "
          f"(combined {out['random_restart']['combined_front_hv']:,.0f}, "
          f"{out['random_restart']['combined_front_size']} sols)")
    print(f"MOHHO vs random: Wilcoxon p={w_p:.2e}, "
          f"MOHHO better in {out['mohho_vs_random_paired_wilcoxon']['mohho_better_count']}/30")
    print(f"NSGA+archive HV mean {out['nsga2_archive']['stats']['mean']:,.0f} "
          f"(combined {out['nsga2_archive']['combined_front_hv']:,.0f})")
    print(f"MOHHO vs NSGA+archive: MWU p={mw_p:.2e}, A12={A12:.3f}")
    print("ref-point sweep (MOHHO>NSGA, MOHHO>random):")
    for k, v in ref_sweep.items():
        print(f"  {k}: {v['mohho_gt_nsga2']}, {v['mohho_gt_random']} "
              f"(MOHHO +{v['mohho_vs_nsga2_pct']:.1f}% vs NSGA)")
    print(f"total {out['elapsed_s']:.0f}s -> controls.json")


if __name__ == "__main__":
    main()
