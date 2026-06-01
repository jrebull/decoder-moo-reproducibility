"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field


class Fitness(BaseModel):
    f1: float
    f2: float
    f3: float


class ParetoPoint(BaseModel):
    f1: float
    f2: float
    f3: float
    visas_used: int = 0


class GroupInfo(BaseModel):
    index: int
    country: str
    category: str
    n: int
    d: int
    w: int
    source: str = ""


class ScenarioRequest(BaseModel):
    scenario: str = Field(
        ..., description="One of: humanitario, equilibrio, equidad, max_utilizacion, fifo"
    )


class OptimizeRequest(BaseModel):
    pop_size: int = Field(default=30, ge=10, le=80)
    max_iter: int = Field(default=100, ge=20, le=500)
    seed: int = Field(default=42, ge=0)


class AllocationRow(BaseModel):
    country: str
    flag: str
    categories: dict[str, int]
    total: int


class AllocationResponse(BaseModel):
    scenario: str
    fitness: Fitness
    visas_used: int
    utilization: float
    rows: list[AllocationRow]
    matrix: list[list[int]]


class ImpactRow(BaseModel):
    country: str
    flag: str
    fifo_visas: int
    scenario_visas: int
    delta: int
    max_wait: int


class ImpactResponse(BaseModel):
    scenario: str
    rows: list[ImpactRow]


class ConvergenceResponse(BaseModel):
    iterations: list[int]
    hv_mean: list[float]
    hv_std: list[float]


class RunSummary(BaseModel):
    run: int
    seed: int
    num_pareto: int
    hv_final: float
    pareto_front: list[Fitness]


class SummaryResponse(BaseModel):
    num_runs: int
    pop_size: int
    max_iter: int
    archive_size: int
    combined_pareto_size: int
    baseline: Fitness
    hv_stats: dict[str, float]
    best_f1: list[float]
    best_f2: list[float]
    best_f3: list[float]


class SimulationMessage(BaseModel):
    type: str  # "iteration", "complete", "error"
    iteration: int = 0
    max_iter: int = 0
    archive_size: int = 0
    hv: float = 0.0
    pareto_front: list[Fitness] = []
    hawks: list[list[float]] = []
    fitnesses: list[Fitness] = []
