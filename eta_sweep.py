"""
REVIEWER REBUTTAL EXPERIMENT (MIT's most dangerous question): is "real-coded
NSGA-II falls below random restart" merely an artifact of leaving the SBX
distribution index at eta_c=20 (a near-identity-on-the-decoded-order setting)?
We sweep eta_c over a wide range and, at each value, report BOTH (i) the SBX
operator's SPV-order preservation tau and (ii) the resulting NSGA-II mean
hypervolume over 30 seeds. If a wider-spread GA (small eta_c) crosses the
random-key ceiling (random restart ~ 309,821) or the permutation tier (~316k),
the "family is second-order, representation is first-order" thesis weakens. If
the GA stays pinned below random restart at every eta_c, the thesis is robust:
no SBX spread escapes the random-key landscape.

Output: app/data/results/eta_sweep.json
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

import compare_nsga2 as cn
from compare_nsga2 import run_nsga2, nondominated, SEEDS
from app.core.config import NUM_GROUPS, LB, UB
from app.core.problem import VisaProblem
from app.core.mohho import compute_hypervolume

RESULTS = Path("app/data/results")
ETAS = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
TAU_TRIALS = 3000
RANDOM_RESTART_HV = 309_821      # from controls.json (random-key blind-sampling ceiling)
PERM_TIER_MIN_HV = 314_846       # weakest permutation-native method (perm-MOEA/D)


def sbx_tau(eta_c, n=TAU_TRIALS, seed=123):
    """Kendall tau between a parent's SPV order and its SBX child's, at this eta_c."""
    rng = np.random.default_rng(seed)
    old = cn.ETA_C
    cn.ETA_C = eta_c
    ts = []
    for _ in range(n):
        p1 = rng.uniform(LB, UB, NUM_GROUPS)
        p2 = rng.uniform(LB, UB, NUM_GROUPS)
        c1, _ = cn.sbx(p1, p2, rng)
        rp = np.argsort(np.argsort(p1)); rc = np.argsort(np.argsort(c1))
        t, _ = kendalltau(rp, rc)
        ts.append(0.0 if np.isnan(t) else t)
    cn.ETA_C = old
    return float(np.mean(ts)), float(np.std(ts))


def main():
    p = VisaProblem()
    out = {"etas": ETAS, "seeds": list(SEEDS), "tau_trials": TAU_TRIALS,
           "random_restart_hv": RANDOM_RESTART_HV, "perm_tier_min_hv": PERM_TIER_MIN_HV,
           "sweep": []}
    old_eta = cn.ETA_C
    for eta in ETAS:
        tau_m, tau_s = sbx_tau(eta)
        cn.ETA_C = eta
        hvs, allpts = [], []
        for s in SEEDS:
            front = run_nsga2(p, s)
            hvs.append(compute_hypervolume(front)); allpts += front
        cn.ETA_C = old_eta
        hvs = np.array(hvs)
        comb = compute_hypervolume(nondominated(allpts))
        row = {
            "eta_c": eta, "sbx_tau_mean": tau_m, "sbx_tau_std": tau_s,
            "hv_mean": float(hvs.mean()), "hv_std": float(hvs.std()),
            "hv_max": float(hvs.max()), "combined_front_hv": comb,
            "beats_random_restart": bool(hvs.mean() > RANDOM_RESTART_HV),
            "reaches_perm_tier": bool(hvs.mean() >= PERM_TIER_MIN_HV),
        }
        out["sweep"].append(row)
        print(f"eta_c={eta:6.1f} | tau={tau_m:.3f} | HV {hvs.mean():,.0f}+-{hvs.std():,.0f} "
              f"(max {hvs.max():,.0f}, comb {comb:,.0f}) | >random={row['beats_random_restart']} "
              f">=permtier={row['reaches_perm_tier']}")

    best = max(out["sweep"], key=lambda r: r["hv_mean"])
    out["best_eta_c"] = best["eta_c"]
    out["best_hv_mean"] = best["hv_mean"]
    out["any_beats_random_restart"] = any(r["beats_random_restart"] for r in out["sweep"])
    out["any_reaches_perm_tier"] = any(r["reaches_perm_tier"] for r in out["sweep"])
    # correlation between tau and HV across the sweep
    taus = [r["sbx_tau_mean"] for r in out["sweep"]]
    hvm = [r["hv_mean"] for r in out["sweep"]]
    rho, pval = kendalltau(taus, hvm)
    out["tau_vs_hv_kendall"] = {"tau": float(rho), "p": float(pval)}
    json.dump(out, open(RESULTS / "eta_sweep.json", "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"best eta_c {best['eta_c']} -> HV {best['hv_mean']:,.0f} "
          f"(random restart {RANDOM_RESTART_HV:,}, perm tier >= {PERM_TIER_MIN_HV:,})")
    print(f"any eta_c beats random restart? {out['any_beats_random_restart']}")
    print(f"any eta_c reaches permutation tier? {out['any_reaches_perm_tier']}")
    print(f"tau vs HV across sweep: Kendall {rho:.3f} (p={pval:.3f})")
    print("saved eta_sweep.json")


if __name__ == "__main__":
    main()
