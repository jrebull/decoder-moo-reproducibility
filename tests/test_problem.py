"""Tests for the VisaProblem objective functions (f1, f2, f3)."""

from app.core.config import NUM_GROUPS, V


def test_group_count(problem):
    assert len(problem.groups) == NUM_GROUPS == 105


def test_total_visas(problem):
    assert problem.total_visas == V == 140_000


def test_f3_empty_allocation_equals_all_visas(problem):
    """With nothing assigned, every visa is wasted."""
    x = {g["index"]: 0 for g in problem.groups}
    assert problem.f3(x) == float(V)


def test_f1_positive_when_demand_unmet(problem):
    """All demand unmet → positive weighted waiting load."""
    x = {g["index"]: 0 for g in problem.groups}
    assert problem.f1(x) > 0


def test_evaluate_returns_three_floats(problem):
    x = {g["index"]: 0 for g in problem.groups}
    fit = problem.evaluate(x)
    assert len(fit) == 3
    assert all(isinstance(v, float) for v in fit)


def test_f1_zero_when_all_demand_met(problem):
    """Assigning every group its full demand drives the waiting load to 0."""
    empty = {g["index"]: 0 for g in problem.groups}
    full = {g["index"]: g["n"] for g in problem.groups}
    assert problem.f1(full) == 0.0
    assert problem.f1(full) < problem.f1(empty)


def test_f2_non_negative(problem):
    """Disparity is a max-min spread and can never be negative."""
    x = {g["index"]: 0 for g in problem.groups}
    assert problem.f2(x) >= 0
