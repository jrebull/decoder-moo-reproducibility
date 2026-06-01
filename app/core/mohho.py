"""Multi-Objective Harris Hawks Optimization (MOHHO)."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from app.core.config import (
    POPULATION_SIZE, MAX_ITERATIONS, ARCHIVE_SIZE,
    LB, UB, NUM_GROUPS,
)
from app.core.decoder import spv, decode
from app.core.hho import (
    escape_energy,
    op1_exploration_random, op2_exploration_mean,
    op3_soft_siege, op4_hard_siege,
    op5_soft_siege_levy, op6_hard_siege_levy,
)
from app.core.problem import VisaProblem

HV_REF_POINT: tuple[float, float, float] = (10.0, 16.0, 20_000.0)
CD_INF_REPLACEMENT: float = 1e6
Fitness3 = tuple[float, float, float]


def dominates(a: Fitness3, b: Fitness3) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] <= b[2]) and \
           (a[0] < b[0] or a[1] < b[1] or a[2] < b[2])


def crowding_distance(fitnesses: list[Fitness3]) -> list[float]:
    n = len(fitnesses)
    if n <= 2:
        return [float("inf")] * n

    distances = [0.0] * n
    for m in range(3):
        indices = sorted(range(n), key=lambda i: fitnesses[i][m])
        distances[indices[0]] = float("inf")
        distances[indices[-1]] = float("inf")
        f_min = fitnesses[indices[0]][m]
        f_max = fitnesses[indices[-1]][m]
        span = f_max - f_min
        if span == 0:
            continue
        for k in range(1, n - 1):
            diff = fitnesses[indices[k + 1]][m] - fitnesses[indices[k - 1]][m]
            distances[indices[k]] += diff / span

    return distances


def select_leader(
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    cd = crowding_distance(archive_fitnesses)
    cd_finite = [d if d != float("inf") else CD_INF_REPLACEMENT for d in cd]
    total = sum(cd_finite)
    if total == 0:
        return archive_positions[rng.integers(len(archive_positions))]
    probs = np.array(cd_finite) / total
    idx = rng.choice(len(archive_positions), p=probs)
    return archive_positions[idx]


def _is_duplicate(new_fit: Fitness3, archive_fitnesses: list[Fitness3]) -> bool:
    for af in archive_fitnesses:
        if (abs(af[0] - new_fit[0]) < 1e-12 and
            abs(af[1] - new_fit[1]) < 1e-12 and
            abs(af[2] - new_fit[2]) < 1e-12):
            return True
    return False


def update_archive(
    archive_positions: list[NDArray[np.float64]],
    archive_fitnesses: list[Fitness3],
    new_pos: NDArray[np.float64],
    new_fit: Fitness3,
    max_size: int,
    rng: np.random.Generator,
) -> None:
    if _is_duplicate(new_fit, archive_fitnesses):
        return

    dominated_by_new = []
    for i, af in enumerate(archive_fitnesses):
        if dominates(af, new_fit):
            return
        if dominates(new_fit, af):
            dominated_by_new.append(i)

    for i in sorted(dominated_by_new, reverse=True):
        archive_positions.pop(i)
        archive_fitnesses.pop(i)

    archive_positions.append(new_pos.copy())
    archive_fitnesses.append(new_fit)

    if len(archive_positions) > max_size:
        cd = crowding_distance(archive_fitnesses)
        finite_indices = [i for i in range(len(cd)) if cd[i] != float("inf")]
        if finite_indices:
            min_cd_idx = min(finite_indices, key=lambda i: cd[i])
        else:
            n = len(archive_positions)
            min_cd_idx = int(rng.integers(1, n - 1)) if n > 2 else 0
        archive_positions.pop(min_cd_idx)
        archive_fitnesses.pop(min_cd_idx)


def evaluate_hawk(
    hawk: NDArray[np.float64], problem: VisaProblem
) -> tuple[dict[int, int], Fitness3]:
    perm = spv(hawk)
    alloc = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    fitness = problem.evaluate(alloc)
    return alloc, fitness


def _greedy_select_levy(xi, fit_i, y, z, problem):
    # FE-budget parity: each hawk issues a SINGLE trial point per iteration (the
    # Levy dive y), so MOHHO uses exactly pop x iter function evaluations---matching
    # NSGA-II, random restart, and the permutation-native methods. (The canonical
    # rapid dive evaluates both y and z; that gave MOHHO ~40% more evaluations than
    # its competitors, so we equalize to one evaluation per individual per iteration.)
    candidates = []
    _, fit_y = evaluate_hawk(y, problem)
    candidates.append((y, fit_y))
    if dominates(fit_y, fit_i):
        return (y, fit_y), candidates
    return None, candidates


def _step_hawk(i, population, fitnesses, archive_positions, archive_fitnesses,
               x_mean, t, max_iter, pop_size, archive_size, problem, rng):
    e = escape_energy(t, max_iter, rng)
    abs_e = abs(e)
    x_rabbit = select_leader(archive_positions, archive_fitnesses, rng)

    if abs_e >= 1:
        new_pos = _exploration_step(population, i, x_rabbit, x_mean, pop_size, rng)
        _accept_and_archive(i, new_pos, population, fitnesses,
                            archive_positions, archive_fitnesses, archive_size, problem, rng)
    elif rng.random() >= 0.5:
        new_pos = _siege_step(population[i], x_rabbit, e, abs_e, rng)
        _accept_and_archive(i, new_pos, population, fitnesses,
                            archive_positions, archive_fitnesses, archive_size, problem, rng)
    else:
        _levy_step(i, population, fitnesses, x_rabbit, x_mean, e, abs_e,
                   archive_positions, archive_fitnesses, archive_size, problem, rng)


def _exploration_step(population, i, x_rabbit, x_mean, pop_size, rng):
    if rng.random() >= 0.5:
        rand_idx = rng.integers(pop_size)
        return op1_exploration_random(population[i], population[rand_idx], rng)
    return op2_exploration_mean(population[i], x_rabbit, x_mean, rng)


def _siege_step(xi, x_rabbit, e, abs_e, rng):
    if abs_e >= 0.5:
        return op3_soft_siege(xi, x_rabbit, e, rng)
    return op4_hard_siege(xi, x_rabbit, e, rng)


def _levy_step(i, population, fitnesses, x_rabbit, x_mean, e, abs_e,
               archive_positions, archive_fitnesses, archive_size, problem, rng):
    if abs_e >= 0.5:
        y, z = op5_soft_siege_levy(population[i], x_rabbit, e, rng)
    else:
        y, z = op6_hard_siege_levy(population[i], x_rabbit, e, x_mean, rng)

    winner, candidates = _greedy_select_levy(population[i], fitnesses[i], y, z, problem)
    if winner is not None:
        new_pos, fit_new = winner
        population[i] = new_pos
        fitnesses[i] = fit_new
    for cand_pos, cand_fit in candidates:
        update_archive(archive_positions, archive_fitnesses,
                       cand_pos, cand_fit, archive_size, rng)


def _accept_and_archive(i, new_pos, population, fitnesses,
                        archive_positions, archive_fitnesses, archive_size, problem, rng):
    _, fit_new = evaluate_hawk(new_pos, problem)
    if dominates(fit_new, fitnesses[i]):
        population[i] = new_pos
        fitnesses[i] = fit_new
    update_archive(archive_positions, archive_fitnesses,
                   new_pos, fit_new, archive_size, rng)


def run_mohho(
    problem: VisaProblem,
    seed: int,
    pop_size: int = POPULATION_SIZE,
    max_iter: int = MAX_ITERATIONS,
    archive_size: int = ARCHIVE_SIZE,
    callback: Callable[[int, list[Fitness3], list[NDArray[np.float64]]], None] | None = None,
) -> tuple[list[NDArray[np.float64]], list[Fitness3], list[float]]:
    rng = np.random.default_rng(seed)
    dim = NUM_GROUPS

    population = rng.uniform(LB, UB, size=(pop_size, dim))
    fitnesses: list[Fitness3] = []
    for i in range(pop_size):
        _, fit = evaluate_hawk(population[i], problem)
        fitnesses.append(fit)

    archive_positions: list[NDArray[np.float64]] = []
    archive_fitnesses: list[Fitness3] = []
    for i in range(pop_size):
        update_archive(archive_positions, archive_fitnesses,
                       population[i], fitnesses[i], archive_size, rng)

    hv_history: list[float] = []

    for t in range(max_iter):
        x_mean = np.mean(population, axis=0)
        for i in range(pop_size):
            _step_hawk(i, population, fitnesses,
                       archive_positions, archive_fitnesses,
                       x_mean, t, max_iter, pop_size, archive_size,
                       problem, rng)

        hv = compute_hypervolume(archive_fitnesses)
        hv_history.append(hv)

        if callback:
            callback(t, archive_fitnesses, archive_positions)

    return archive_positions, archive_fitnesses, hv_history


def _hv_2d(pts: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    sorted_pts = sorted(pts, key=lambda p: p[0])
    hv = 0.0
    prev = ref[1]
    for fa, fb in sorted_pts:
        if fa < ref[0] and fb < ref[1]:
            h = prev - fb
            if h > 0:
                hv += (ref[0] - fa) * h
                prev = fb
    return hv


def compute_hypervolume(
    fitnesses: list[Fitness3],
    ref_point: tuple[float, float, float] | None = None,
) -> float:
    if not fitnesses:
        return 0.0
    if ref_point is None:
        ref_point = HV_REF_POINT

    pts = [(f1, f2, f3) for f1, f2, f3 in fitnesses
           if f1 < ref_point[0] and f2 < ref_point[1] and f3 < ref_point[2]]
    if not pts:
        return 0.0

    pts.sort(key=lambda p: p[0])
    hv = 0.0
    for i, (f1, f2, f3) in enumerate(pts):
        f1_next = pts[i + 1][0] if i + 1 < len(pts) else ref_point[0]
        width_f1 = f1_next - f1
        slice_pts = [(p[1], p[2]) for p in pts[:i + 1]]
        hv_slice = _hv_2d(slice_pts, (ref_point[1], ref_point[2]))
        hv += width_f1 * hv_slice

    return hv
