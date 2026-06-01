"""Regenerate the base visa-problem results from the (recalibrated) data:
summary.json, pareto_front.csv, convergence.csv, run_XX.json, stats_test.json,
sensitivity_analysis.json, nsga2_front.json. Run FIRST after a data change.
30 MOHHO runs (seeds 1-30) + 30 NSGA-II runs (seeds 1-30)."""
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, ranksums

from app.core.config import POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE
from app.core.problem import VisaProblem
from app.core.fifo import run_baseline
from app.core.mohho import run_mohho, compute_hypervolume, HV_REF_POINT
from compare_nsga2 import run_nsga2, nondominated

R = Path("app/data/results")
MO_SEEDS = list(range(1, 31))
NS_SEEDS = list(range(1, 31))
POP, IT, ARC = POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE


def main():
    p = VisaProblem()
    _, fifo = run_baseline(p)
    fifo = [float(x) for x in fifo]
    print("FIFO", fifo, "ref", HV_REF_POINT, "total_demand", p.total_demand)

    mo_hv, mo_all, hv_hist = [], [], []
    for i, s in enumerate(MO_SEEDS):
        _, fits, hist = run_mohho(p, seed=s, pop_size=POP, max_iter=IT, archive_size=ARC)
        hv = compute_hypervolume(fits)
        mo_hv.append(hv); mo_all += fits; hv_hist.append(hist)
        json.dump({"run": i, "seed": s, "num_pareto": len(fits), "hv_final": hv,
                   "pareto_front": [list(map(float, f)) for f in fits]},
                  open(R / f"run_{i:02d}.json", "w"))
        print(f"MOHHO {s}: {len(fits)} sols HV={hv:,.0f}")
    combined = nondominated(mo_all)
    combined = [tuple(map(float, c)) for c in combined]

    # combined-front extremes
    bf1 = min(combined, key=lambda c: c[0])
    bf2 = min(combined, key=lambda c: c[1])
    bf3 = min(combined, key=lambda c: c[2])
    summary = {
        "num_runs": len(MO_SEEDS), "pop_size": POP, "max_iter": IT, "archive_size": ARC,
        "baseline": {"f1": fifo[0], "f2": fifo[1], "f3": fifo[2]},
        "combined_pareto_size": len(combined),
        "hv_stats": {"mean": float(np.mean(mo_hv)), "std": float(np.std(mo_hv)),
                     "min": float(np.min(mo_hv)), "max": float(np.max(mo_hv))},
        "best_f1": list(bf1), "best_f2": list(bf2), "best_f3": list(bf3),
        "hv_ref_point": list(HV_REF_POINT),
    }
    json.dump(summary, open(R / "summary.json", "w"), indent=2)

    with open(R / "pareto_front.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["f1", "f2", "f3", "type"])
        for c in sorted(combined, key=lambda x: x[0]):
            w.writerow([f"{c[0]:.6f}", f"{c[1]:.6f}", int(round(c[2])), "pareto"])
        w.writerow([f"{fifo[0]:.6f}", f"{fifo[1]:.6f}", int(round(fifo[2])), "fifo"])

    # convergence (mean/std HV per iteration across the 30 runs)
    H = np.array(hv_hist)  # 30 x IT
    with open(R / "convergence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["iteration", "hv_mean", "hv_std"])
        for t in range(IT):
            w.writerow([t, f"{H[:, t].mean():.6f}", f"{H[:, t].std():.6f}"])

    # NSGA-II 30 runs
    ns_hv, ns_all = [], []
    for s in NS_SEEDS:
        front = run_nsga2(p, s); ns_hv.append(compute_hypervolume(front)); ns_all += front
        print(f"NSGA {s}: {len(front)} sols HV={ns_hv[-1]:,.0f}")
    ns_front = nondominated(ns_all)
    json.dump({"front": [list(map(float, c)) for c in ns_front], "size": len(ns_front)},
              open(R / "nsga2_front.json", "w"))

    U, p1 = mannwhitneyu(mo_hv, ns_hv, alternative="greater")
    _, p2 = ranksums(mo_hv, ns_hv)
    A12 = float(U / (len(mo_hv) * len(ns_hv)))
    json.dump({"mohho_hv": mo_hv, "mohho_seeds": MO_SEEDS, "nsga2_hv": ns_hv,
               "mannwhitney_U": float(U), "p_one_sided": float(p1),
               "A12": A12, "ranksum_p_two_sided": float(p2)},
              open(R / "stats_test.json", "w"), indent=2)

    # sensitivity: archive size (seed 1), convergence iters, f1 range
    arch = {}
    for a in (100, 200, 500):
        _, fits, _ = run_mohho(p, seed=1, pop_size=POP, max_iter=IT, archive_size=a)
        arch[f"archive_{a}"] = {"solutions": len(fits), "hv": round(compute_hypervolume(fits), 2),
                                "saturated": len(fits) >= a}
    hv_mean_curve = H.mean(axis=0)
    final = hv_mean_curve[-1]
    i95 = int(np.argmax(hv_mean_curve >= 0.95 * final))
    i99 = int(np.argmax(hv_mean_curve >= 0.99 * final))
    cf = np.array(combined)
    f1r, f2r, f3r = (float(cf[:, k].max() - cf[:, k].min()) for k in range(3))
    sens = {
        "archive_sensitivity": {"seed": 1, **arch,
            "hv_delta_100_vs_200": f"{100*abs(arch['archive_100']['hv']-arch['archive_200']['hv'])/arch['archive_100']['hv']:.2f}%",
            "hv_delta_100_vs_500": f"{100*abs(arch['archive_100']['hv']-arch['archive_500']['hv'])/arch['archive_100']['hv']:.2f}%"},
        "convergence_analysis": {"final_hv": round(float(final), 2), "iter_95pct": i95, "iter_99pct": i99,
            "hv_iter_300": round(float(hv_mean_curve[min(300, IT-1)]), 2),
            "hv_gain_300_to_500": f"{100*(final-hv_mean_curve[min(300,IT-1)])/hv_mean_curve[min(300,IT-1)]:.2f}%"},
        "f1_range_analysis": {"f1_range": round(f1r, 4), "f2_range": round(f2r, 4), "f3_range": round(f3r, 1),
            "f1_normalized_width": round(f1r / HV_REF_POINT[0], 4),
            "f2_normalized_width": round(f2r / HV_REF_POINT[1], 4),
            "f1_vs_f2_ratio": round((f2r / HV_REF_POINT[1]) / (f1r / HV_REF_POINT[0]), 1)},
    }
    json.dump(sens, open(R / "sensitivity_analysis.json", "w"), indent=2)

    f3zero = sum(1 for c in combined if c[2] == 0)
    print("\n=== BASE SUMMARY (new data) ===")
    print(f"FIFO {fifo} | combined front {len(combined)} ({f3zero} zero-waste)")
    print(f"MOHHO HV {summary['hv_stats']['mean']:,.0f}+-{summary['hv_stats']['std']:,.0f} "
          f"CV {100*summary['hv_stats']['std']/summary['hv_stats']['mean']:.2f}%")
    print(f"NSGA HV {np.mean(ns_hv):,.0f}+-{np.std(ns_hv):,.0f} CV {100*np.std(ns_hv)/np.mean(ns_hv):.2f}%")
    print(f"MOHHO vs NSGA: +{100*(summary['hv_stats']['mean']-np.mean(ns_hv))/np.mean(ns_hv):.1f}% p={p1:.2e} A12={A12:.3f}")
    print(f"extremes: bestf1 {bf1} bestf2 {bf2} bestf3 {bf3}")
    print(f"conv: 95%@{i95} 99%@{i99} | f1_range {f1r:.3f} f2_range {f2r:.3f} ratio {sens['f1_range_analysis']['f1_vs_f2_ratio']}x")
    print(f"archive: {arch}")


if __name__ == "__main__":
    main()
