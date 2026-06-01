"""Tests for the MOHHO core: dominance, crowding distance, archive invariants
and — critically — seed reproducibility (guards the rng fix in update_archive)."""

from app.core.mohho import (
    run_mohho, dominates, crowding_distance,
)


def test_dominates_logic():
    assert dominates((1, 1, 1), (2, 2, 2))          # strictly better in all
    assert dominates((1, 2, 2), (2, 2, 2))          # better in one, equal in rest
    assert not dominates((2, 2, 2), (1, 1, 1))      # strictly worse
    assert not dominates((1, 1, 1), (1, 1, 1))      # equal is not domination


def test_crowding_distance_marks_boundaries_infinite():
    fits = [(1.0, 5.0, 9.0), (2.0, 4.0, 8.0), (3.0, 3.0, 7.0), (4.0, 2.0, 6.0)]
    cd = crowding_distance(fits)
    assert len(cd) == len(fits)
    assert cd.count(float("inf")) >= 2  # at least the two extremes


def test_run_mohho_is_reproducible(problem):
    """Same seed → identical archive and HV history. This is the invariant the
    rng fix protects: no unseeded global random anywhere in the run."""
    kw = dict(seed=42, pop_size=12, max_iter=20, archive_size=15)
    _, fits_a, hv_a = run_mohho(problem, **kw)
    _, fits_b, hv_b = run_mohho(problem, **kw)
    assert fits_a == fits_b
    assert hv_a == hv_b


def test_different_seeds_can_differ(problem):
    kw = dict(pop_size=12, max_iter=20, archive_size=15)
    _, fits_a, _ = run_mohho(problem, seed=1, **kw)
    _, fits_b, _ = run_mohho(problem, seed=999, **kw)
    # Not a hard guarantee, but with different seeds the fronts should not be
    # byte-identical; if they are, the seeding is broken.
    assert fits_a != fits_b


def test_archive_is_nondominated(problem):
    _, fits, _ = run_mohho(problem, seed=7, pop_size=12, max_iter=20, archive_size=15)
    for i, a in enumerate(fits):
        for j, b in enumerate(fits):
            if i != j:
                assert not dominates(a, b)


def test_archive_size_is_capped(problem):
    cap = 10
    _, fits, _ = run_mohho(problem, seed=3, pop_size=20, max_iter=25, archive_size=cap)
    assert len(fits) <= cap


def test_hypervolume_is_positive(problem):
    _, _, hv = run_mohho(problem, seed=5, pop_size=12, max_iter=20, archive_size=15)
    assert hv[-1] > 0
