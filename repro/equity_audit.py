"""Alternative fairness audit for the MICAI visa case study.

This script recomputes the 30-seed MOHHO combined front and evaluates external
country-level fairness metrics that are not optimized directly:
  - standard deviation of country mean waits,
  - Gini coefficient of country mean waits,
  - Jain index on inverse waits (higher is fairer),
  - served-country count.

The goal is not to introduce a fourth objective. It is a robustness check for
the paper's f2=max-min disparity objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import _bootstrap

_bootstrap.bootstrap_engine()

from app.core.decoder import decode
from app.core.fifo import run_baseline
from app.core.mohho import run_mohho, dominates
from app.core.problem import VisaProblem

RESULTS = Path(_bootstrap.results_dir())
SEEDS = list(range(1, 31))
EPS = 1e-9


def country_waits(problem: VisaProblem, alloc: dict[int, int]) -> dict[str, float]:
    waits: dict[str, float] = {}
    for country, groups in problem._groups_by_country.items():
        total = sum(alloc[g["index"]] for g in groups)
        if total > 0:
            waits[country] = sum(alloc[g["index"]] * g["w"] for g in groups) / total
        else:
            waits[country] = float(problem._country_w_max[country])
    return waits


def gini(values: list[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    n = arr.size
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1) / n)


def jain(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    denom = arr.size * np.sum(arr * arr)
    if denom <= EPS:
        return 0.0
    return float((np.sum(arr) ** 2) / denom)


def fairness_metrics(problem: VisaProblem, alloc: dict[int, int]) -> dict[str, float]:
    waits = country_waits(problem, alloc)
    vals = list(waits.values())
    inv = [1.0 / (1.0 + v) for v in vals]
    return {
        "f2_gap": float(max(vals) - min(vals)),
        "wait_std": float(np.std(vals)),
        "wait_gini": gini(vals),
        "jain_inverse_wait": jain(inv),
        "served_countries": int(sum(
            1 for country, groups in problem._groups_by_country.items()
            if sum(alloc[g["index"]] for g in groups) > 0
        )),
    }


def spv_alloc(position: np.ndarray, problem: VisaProblem) -> dict[int, int]:
    perm = list(np.argsort(position, kind="stable"))
    return decode(perm, problem.groups, problem.total_visas,
                  problem.country_caps, problem.category_caps)


def nondominated_records(records: list[dict]) -> list[dict]:
    out: list[dict] = []
    for i, rec in enumerate(records):
        fit = tuple(rec["fitness"])
        if any(i != j and dominates(tuple(other["fitness"]), fit)
               for j, other in enumerate(records)):
            continue
        out.append(rec)
    return out


def best(records: list[dict], key: str, reverse: bool = False) -> dict:
    return sorted(records, key=lambda r: r["fairness"][key], reverse=reverse)[0]


def main() -> None:
    problem = VisaProblem()
    positions: list[np.ndarray] = []
    records: list[dict] = []

    for seed in SEEDS:
        run_positions, run_fits, _ = run_mohho(problem, seed=seed)
        positions.extend(run_positions)
        for pos, fit in zip(run_positions, run_fits):
            alloc = spv_alloc(pos, problem)
            records.append({
                "seed": seed,
                "fitness": [float(x) for x in fit],
                "fairness": fairness_metrics(problem, alloc),
            })
        print(f"seed {seed:02d}: archive {len(run_fits)}")

    front = nondominated_records(records)
    _, fifo_fit = run_baseline(problem)
    # run_baseline returns an allocation as first value.
    fifo_alloc, _ = run_baseline(problem)
    fifo_metrics = fairness_metrics(problem, fifo_alloc)

    keys_lower = ["f2_gap", "wait_std", "wait_gini"]
    keys_higher = ["jain_inverse_wait", "served_countries"]
    summary = {
        "seeds": SEEDS,
        "n_records": len(records),
        "n_combined_nondominated": len(front),
        "fifo": {
            "fitness": [float(x) for x in fifo_fit],
            "fairness": fifo_metrics,
        },
        "front_ranges": {},
        "best_by_metric": {},
    }
    for key in keys_lower:
        vals = [r["fairness"][key] for r in front]
        summary["front_ranges"][key] = {
            "min": float(min(vals)),
            "median": float(np.median(vals)),
            "max": float(max(vals)),
            "fifo": float(fifo_metrics[key]),
        }
        rec = best(front, key)
        summary["best_by_metric"][key] = {
            "fitness": rec["fitness"],
            "fairness": rec["fairness"],
            "beats_fifo": bool(rec["fairness"][key] <= fifo_metrics[key] + EPS),
        }
    for key in keys_higher:
        vals = [r["fairness"][key] for r in front]
        summary["front_ranges"][key] = {
            "min": float(min(vals)),
            "median": float(np.median(vals)),
            "max": float(max(vals)),
            "fifo": float(fifo_metrics[key]),
        }
        rec = best(front, key, reverse=True)
        summary["best_by_metric"][key] = {
            "fitness": rec["fitness"],
            "fairness": rec["fairness"],
            "beats_fifo": bool(rec["fairness"][key] >= fifo_metrics[key] - EPS),
        }

    out_path = RESULTS / "equity_audit.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "n_combined_nondominated": summary["n_combined_nondominated"],
        "fifo": summary["fifo"],
        "front_ranges": summary["front_ranges"],
    }, indent=2))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
