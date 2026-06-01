"""
Taguchi orthogonal-array design of experiments (DOE) for MOHHO parameter tuning.

Answers the reviewer's request for a *formal, systematic* justification of the
population size and iteration budget (beyond empiricism), using an L9(3^4)
orthogonal array and a larger-the-better signal-to-noise (S/N) analysis on the
hypervolume (HV) response. Includes a confirmation run that contrasts the
additive-model prediction at the optimal level combination with its observed S/N.

Factors (3 levels each):
  A  Population size N        {30, 50, 70}
  B  Iteration budget  T      {100, 300, 500}
  C  External archive size    {50, 100, 150}
  D  Levy exponent  beta      {1.3, 1.5, 1.7}

Response: hypervolume of the final external archive (reference point (10,16,50000)).
Replicates: REPS independent seeds per configuration (distinct from the 30-run
benchmark seeds 1-71 to avoid contamination).

Outputs:
  app/data/results/taguchi.json          (full DOE: array, per-config HV, S/N,
                                           response tables, optimum, confirmation)
  ../figures/taguchi_main_effects.pdf   (S/N main-effects plot, vectorial)
"""
import json
import time
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import app.core.hho as hho
from app.core.problem import VisaProblem
from app.core.mohho import run_mohho, compute_hypervolume

RESULTS = Path("app/data/results")
FIGDIR = Path("../figures")
REPS = 12                      # replicate seeds per configuration
SEED0 = 200                    # replicate seeds: 200 .. 200+REPS-1

# Factor levels
LEVELS = {
    "A_pop":     [30, 50, 70],
    "B_iter":    [100, 300, 500],
    "C_archive": [50, 100, 150],
    "D_beta":    [1.3, 1.5, 1.7],
}
FACTORS = ["A_pop", "B_iter", "C_archive", "D_beta"]

# Standard L9(3^4) orthogonal array (1-indexed levels)
L9 = [
    [1, 1, 1, 1],
    [1, 2, 2, 2],
    [1, 3, 3, 3],
    [2, 1, 2, 3],
    [2, 2, 3, 1],
    [2, 3, 1, 2],
    [3, 1, 3, 2],
    [3, 2, 1, 3],
    [3, 3, 2, 1],
]


def config_from_row(row):
    """Map a 1-indexed L9 row to concrete parameter values."""
    return {
        "N":       LEVELS["A_pop"][row[0] - 1],
        "T":       LEVELS["B_iter"][row[1] - 1],
        "archive": LEVELS["C_archive"][row[2] - 1],
        "beta":    LEVELS["D_beta"][row[3] - 1],
    }


def sn_larger_is_better(ys):
    """Taguchi larger-the-better S/N ratio (dB)."""
    ys = np.asarray(ys, dtype=float)
    return float(-10.0 * np.log10(np.mean(1.0 / ys ** 2)))


def run_config(problem, cfg, seeds):
    """Run MOHHO for every seed at a fixed configuration; return HV list."""
    hho.BETA_LEVY = cfg["beta"]          # patch the Levy exponent for this config
    hvs = []
    for s in seeds:
        _, fits, _ = run_mohho(
            problem, seed=s,
            pop_size=cfg["N"], max_iter=cfg["T"], archive_size=cfg["archive"],
        )
        hvs.append(compute_hypervolume(fits))
    return hvs


def main():
    problem = VisaProblem()
    seeds = list(range(SEED0, SEED0 + REPS))
    t0 = time.time()

    # ---- 1. Run the L9 array ----
    rows = []
    for i, row in enumerate(L9, 1):
        cfg = config_from_row(row)
        ts = time.time()
        hvs = run_config(problem, cfg, seeds)
        sn = sn_larger_is_better(hvs)
        rows.append({
            "run": i, "levels": row, "config": cfg,
            "hv_mean": float(np.mean(hvs)), "hv_std": float(np.std(hvs)),
            "hv_min": float(np.min(hvs)), "hv_max": float(np.max(hvs)),
            "sn": sn, "hv": [float(h) for h in hvs],
        })
        print(f"L9 run {i}: N={cfg['N']} T={cfg['T']} arch={cfg['archive']} "
              f"beta={cfg['beta']}  HV={np.mean(hvs):,.0f}  S/N={sn:.3f}  "
              f"({time.time()-ts:.1f}s)")

    grand_mean_sn = float(np.mean([r["sn"] for r in rows]))
    grand_mean_hv = float(np.mean([r["hv_mean"] for r in rows]))

    # ---- 2. Response tables (mean S/N and mean HV per factor level) ----
    response_sn = {}
    response_hv = {}
    deltas = {}
    optimum_levels = {}
    for fi, fac in enumerate(FACTORS):
        sn_by_level = []
        hv_by_level = []
        for lvl in (1, 2, 3):
            sel = [r for r in rows if r["levels"][fi] == lvl]
            sn_by_level.append(float(np.mean([r["sn"] for r in sel])))
            hv_by_level.append(float(np.mean([r["hv_mean"] for r in sel])))
        response_sn[fac] = sn_by_level
        response_hv[fac] = hv_by_level
        deltas[fac] = float(max(sn_by_level) - min(sn_by_level))
        optimum_levels[fac] = int(np.argmax(sn_by_level) + 1)   # 1-indexed

    # factor importance ranking by S/N delta (range)
    ranking = sorted(FACTORS, key=lambda f: deltas[f], reverse=True)

    # ---- 3. Optimal configuration + additive-model prediction ----
    opt_cfg = {
        "N":       LEVELS["A_pop"][optimum_levels["A_pop"] - 1],
        "T":       LEVELS["B_iter"][optimum_levels["B_iter"] - 1],
        "archive": LEVELS["C_archive"][optimum_levels["C_archive"] - 1],
        "beta":    LEVELS["D_beta"][optimum_levels["D_beta"] - 1],
    }
    # additive model: predicted S/N = grand_mean + sum(best_level_effect - grand_mean)
    predicted_sn = grand_mean_sn + sum(
        response_sn[f][optimum_levels[f] - 1] - grand_mean_sn for f in FACTORS
    )

    # ---- 4. Confirmation run at the optimum (fresh seeds) ----
    conf_seeds = list(range(SEED0 + 100, SEED0 + 100 + REPS))
    conf_hvs = run_config(problem, opt_cfg, conf_seeds)
    confirmation = {
        "config": opt_cfg, "seeds": conf_seeds,
        "hv_mean": float(np.mean(conf_hvs)), "hv_std": float(np.std(conf_hvs)),
        "sn_observed": sn_larger_is_better(conf_hvs),
        "sn_predicted": predicted_sn,
        "hv": [float(h) for h in conf_hvs],
    }

    # ---- 5. Adopted-configuration confirmation (the one used in the main study) ----
    adopted_cfg = {"N": 50, "T": 500, "archive": 100, "beta": 1.5}
    adopted_hvs = run_config(problem, adopted_cfg, conf_seeds)
    adopted = {
        "config": adopted_cfg,
        "hv_mean": float(np.mean(adopted_hvs)), "hv_std": float(np.std(adopted_hvs)),
        "sn_observed": sn_larger_is_better(adopted_hvs),
        "hv": [float(h) for h in adopted_hvs],
    }

    out = {
        "design": "L9(3^4) Taguchi orthogonal array",
        "response": "hypervolume (larger-the-better S/N)",
        "reps_per_config": REPS,
        "replicate_seeds": seeds,
        "levels": LEVELS,
        "l9_array": L9,
        "runs": rows,
        "grand_mean_sn": grand_mean_sn,
        "grand_mean_hv": grand_mean_hv,
        "response_sn": response_sn,
        "response_hv": response_hv,
        "sn_delta": deltas,
        "factor_ranking": ranking,
        "optimum_levels": optimum_levels,
        "optimum_config": opt_cfg,
        "predicted_sn_at_optimum": predicted_sn,
        "confirmation": confirmation,
        "adopted_config_confirmation": adopted,
        "elapsed_s": time.time() - t0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RESULTS / "taguchi.json", "w"), indent=2)
    print("\nfactor S/N deltas:", {k: round(v, 3) for k, v in deltas.items()})
    print("ranking (most->least influential):", ranking)
    print("optimum config:", opt_cfg)
    print(f"predicted S/N {predicted_sn:.3f}  |  confirmation S/N "
          f"{confirmation['sn_observed']:.3f}  (HV {confirmation['hv_mean']:,.0f})")
    print(f"adopted 50x500 S/N {adopted['sn_observed']:.3f}  "
          f"(HV {adopted['hv_mean']:,.0f})")
    print("saved:", RESULTS / "taguchi.json", f"| total {out['elapsed_s']:.0f}s")

    # ---- 6. Main-effects figure (S/N per level, one panel per factor) ----
    FIGDIR.mkdir(parents=True, exist_ok=True)
    labels = {
        "A_pop": "A: population $N$",
        "B_iter": "B: iteration budget $T$",
        "C_archive": "C: archive size",
        "D_beta": r"D: Lévy exponent $\beta$",
    }
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, fac in zip(axes, FACTORS):
        xs = LEVELS[fac]
        ys = response_sn[fac]
        ax.plot(range(3), ys, "o-", color="#2E86DE", lw=1.8, ms=6)
        ax.axhline(grand_mean_sn, color="#888", ls="--", lw=0.9)
        best = optimum_levels[fac] - 1
        ax.plot(best, ys[best], "o", color="#E74C3C", ms=9, zorder=5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_title(labels[fac], fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("S/N ratio (dB)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "taguchi_main_effects.pdf")
    fig.savefig(FIGDIR / "taguchi_main_effects.png", dpi=200)
    print("figure saved:", FIGDIR / "taguchi_main_effects.pdf")


if __name__ == "__main__":
    main()
