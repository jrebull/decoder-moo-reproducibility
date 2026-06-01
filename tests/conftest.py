"""Shared pytest fixtures for the MOHHO backend test suite."""

import pytest

from app.core.problem import VisaProblem


@pytest.fixture(scope="session")
def problem() -> VisaProblem:
    """A single VisaProblem instance reused across the suite (read-only)."""
    return VisaProblem()
