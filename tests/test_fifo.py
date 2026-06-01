"""Tests for the FIFO baseline -- must match the verified reference values."""

import pytest

from app.core.fifo import run_baseline, fifo_permutation


def test_fifo_permutation_is_a_permutation(problem):
    perm = fifo_permutation(problem.groups)
    assert sorted(perm) == sorted(g["index"] for g in problem.groups)


def test_fifo_baseline_matches_verified_values(problem):
    """Locks in the MICAI baseline: f1=8.7891, f2=13.0, f3=1940."""
    _, (f1, f2, f3) = run_baseline(problem)
    assert f1 == pytest.approx(8.7891, rel=1e-6)
    assert f2 == pytest.approx(13.0, rel=1e-9)
    assert f3 == pytest.approx(1940.0, rel=1e-9)


def test_fifo_is_deterministic(problem):
    _, fit_a = run_baseline(problem)
    _, fit_b = run_baseline(problem)
    assert fit_a == fit_b
