"""Baseline FIFO simulation of the current visa allocation system."""

from app.core.decoder import decode
from app.core.problem import VisaProblem


def fifo_permutation(groups: list[dict]) -> list[int]:
    sorted_groups = sorted(groups, key=lambda g: (g["d"], g["index"]))
    return [g["index"] for g in sorted_groups]


def run_baseline(problem: VisaProblem) -> tuple[dict[int, int], tuple[float, float, float]]:
    perm = fifo_permutation(problem.groups)
    alloc = decode(perm, problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)
    fitness = problem.evaluate(alloc)
    return alloc, fitness
