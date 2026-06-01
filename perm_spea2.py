"""Permutation SPEA2 baseline for the the paper visa ladder.

SPEA2 adds an indicator-free, strength-based evolutionary baseline to the
existing dominance (NSGA-II), decomposition (MOEA/D), and swarm (Discrete-MOHHO)
families. It uses the same permutation representation, OX crossover, swap
mutation, greedy decoder, population size, archive size, generation count, and
seed block as the other permutation-native methods.

Output: app/data/results/perm_spea2.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from app.core.config import NUM_GROUPS
from app.core.decoder import decode
from app.core.mohho import compute_hypervolume, dominates
from app.core.problem import VisaProblem
from compare_nsga2 import nondominated
from perm_nsga import ox, swap_mut

RESULTS = Path("app/data/results")
POP, GEN, ARCH, RUNS = 50, 500, 100, 30


def eval_perm(perm: np.ndarray, problem: VisaProblem) -> tuple[float, float, float]:
    alloc = decode(list(perm), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    return problem.evaluate(alloc)


def density(fits: list[tuple[float, float, float]]) -> np.ndarray:
    x = np.asarray(fits, dtype=float)
    scale = np.maximum(np.ptp(x, axis=0), 1e-9)
    z = x / scale
    d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    k = max(1, int(np.sqrt(len(fits))))
    sigma = np.sort(d, axis=1)[:, k - 1]
    return 1.0 / (sigma + 2.0)


def spea2_fitness(fits: list[tuple[float, float, float]]) -> np.ndarray:
    n = len(fits)
    strength = np.zeros(n)
    raw = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j and dominates(fits[i], fits[j]):
                strength[i] += 1
    for i in range(n):
        for j in range(n):
            if i != j and dominates(fits[j], fits[i]):
                raw[i] += strength[j]
    return raw + density(fits)


def environmental_select(pop: list[np.ndarray],
                         fits: list[tuple[float, float, float]],
                         size: int) -> tuple[list[np.ndarray], list[tuple[float, float, float]], np.ndarray]:
    fitvals = spea2_fitness(fits)
    selected = [i for i, value in enumerate(fitvals) if value < 1.0]
    if len(selected) < size:
        selected += [i for i in np.argsort(fitvals) if i not in selected][:size - len(selected)]
    elif len(selected) > size:
        selected = truncate(selected, fits, size)
    return [pop[i].copy() for i in selected], [fits[i] for i in selected], fitvals[selected]


def truncate(indices: list[int], fits: list[tuple[float, float, float]], size: int) -> list[int]:
    keep = indices[:]
    while len(keep) > size:
        x = np.asarray([fits[i] for i in keep], dtype=float)
        scale = np.maximum(np.ptp(x, axis=0), 1e-9)
        z = x / scale
        d = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        order = np.sort(d, axis=1)
        remove_local = min(range(len(keep)), key=lambda i: tuple(order[i]))
        keep.pop(remove_local)
    return keep


def binary_tournament(pool: list[np.ndarray], fitvals: np.ndarray,
                      rng: np.random.Generator) -> np.ndarray:
    i, j = rng.integers(0, len(pool), size=2)
    if fitvals[i] < fitvals[j]:
        return pool[i]
    if fitvals[j] < fitvals[i]:
        return pool[j]
    return pool[i] if rng.random() < 0.5 else pool[j]


def run_perm_spea2(problem: VisaProblem, seed: int) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(seed)
    pop = [rng.permutation(NUM_GROUPS) for _ in range(POP)]
    fits = [eval_perm(ind, problem) for ind in pop]
    archive: list[np.ndarray] = []
    archive_fits: list[tuple[float, float, float]] = []
    archive_fitvals = np.zeros(0)

    for _ in range(GEN):
        union = pop + archive
        union_fits = fits + archive_fits
        archive, archive_fits, archive_fitvals = environmental_select(union, union_fits, ARCH)
        mating = archive if archive else pop
        mating_fitvals = archive_fitvals if archive else spea2_fitness(fits)
        offspring: list[np.ndarray] = []
        while len(offspring) < POP:
            p1 = binary_tournament(mating, mating_fitvals, rng)
            p2 = binary_tournament(mating, mating_fitvals, rng)
            offspring.append(swap_mut(ox(p1, p2, rng), rng))
            if len(offspring) < POP:
                offspring.append(swap_mut(ox(p2, p1, rng), rng))
        pop = offspring
        fits = [eval_perm(ind, problem) for ind in pop]

    final_front = nondominated(archive_fits + fits)
    return [tuple(map(float, f)) for f in final_front]


def main() -> None:
    problem = VisaProblem()
    t0 = time.time()
    hv: list[float] = []
    all_fronts: list[tuple[float, float, float]] = []
    for seed in range(1, RUNS + 1):
        front = run_perm_spea2(problem, seed)
        hv.append(compute_hypervolume(front))
        all_fronts.extend(front)
        print(f"perm-SPEA2 seed {seed:02d}: {len(front)} sols HV={hv[-1]:,.0f} ({time.time()-t0:.0f}s)")

    combined = nondominated(all_fronts)
    random_hv = json.load(open(RESULTS / "controls.json"))["random_restart"]["per_seed_hv"]
    perm_nsga_hv = json.load(open(RESULTS / "perm_nsga.json"))["per_run_hv"]
    _, p_random = mannwhitneyu(hv, random_hv, alternative="greater")
    u_nsga, p_nsga = mannwhitneyu(hv, perm_nsga_hv, alternative="two-sided")
    out = {
        "pop": POP,
        "gen": GEN,
        "archive": ARCH,
        "runs": RUNS,
        "operators": "SPEA2 strength fitness + OX + swap mutation on permutations",
        "per_run_hv": hv,
        "hv_mean": float(np.mean(hv)),
        "hv_std": float(np.std(hv)),
        "cv_pct": float(100 * np.std(hv) / np.mean(hv)),
        "combined_front_size": len(combined),
        "combined_front_hv": compute_hypervolume(combined),
        "vs_random_p": float(p_random),
        "vs_perm_nsga_p_two_sided": float(p_nsga),
        "vs_perm_nsga_A12": float(u_nsga / (len(hv) * len(perm_nsga_hv))),
        "elapsed_s": time.time() - t0,
    }
    (RESULTS / "perm_spea2.json").write_text(json.dumps(out, indent=2))
    print("\n=== PERM-SPEA2 ===")
    print(f"HV mean {out['hv_mean']:,.0f} +/- {out['hv_std']:,.0f} "
          f"(CV {out['cv_pct']:.2f}%) | combined {out['combined_front_hv']:,.0f} "
          f"({out['combined_front_size']} sols)")
    print(f"vs random p={p_random:.2e} | vs perm-NSGA-II p={p_nsga:.2e}, "
          f"A12={out['vs_perm_nsga_A12']:.3f}")


if __name__ == "__main__":
    main()
