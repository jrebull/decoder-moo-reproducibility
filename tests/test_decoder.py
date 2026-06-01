"""Tests for the SPV decoding and the greedy decoder constraints."""

import numpy as np

from app.core.decoder import spv, decode


def test_spv_is_a_valid_permutation(problem):
    rng = np.random.default_rng(0)
    hawk = rng.uniform(0, 1, size=len(problem.groups))
    perm = spv(hawk)
    assert sorted(perm) == list(range(len(problem.groups)))


def test_decode_respects_all_caps(problem):
    rng = np.random.default_rng(1)
    hawk = rng.uniform(0, 1, size=len(problem.groups))
    alloc = decode(spv(hawk), problem.groups, problem.total_visas,
                   problem.country_caps, problem.category_caps)

    # Each group: non-negative and never above its own demand.
    for g in problem.groups:
        assert 0 <= alloc[g["index"]] <= g["n"]

    # Global visa budget.
    assert sum(alloc.values()) <= problem.total_visas

    # Per-country and per-category caps.
    by_country: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for g in problem.groups:
        by_country[g["country"]] = by_country.get(g["country"], 0) + alloc[g["index"]]
        by_category[g["category"]] = by_category.get(g["category"], 0) + alloc[g["index"]]

    for country, used in by_country.items():
        assert used <= problem.country_caps[country]
    for category, used in by_category.items():
        assert used <= problem.category_caps[category]


def test_decode_is_deterministic_for_same_permutation(problem):
    rng = np.random.default_rng(2)
    hawk = rng.uniform(0, 1, size=len(problem.groups))
    perm = spv(hawk)
    a = decode(perm, problem.groups, problem.total_visas,
               problem.country_caps, problem.category_caps)
    b = decode(perm, problem.groups, problem.total_visas,
               problem.country_caps, problem.category_caps)
    assert a == b
