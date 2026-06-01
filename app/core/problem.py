"""Problem definition: objective functions f1, f2, f3."""

from app.core.config import V
from app.core.data import build_groups, compute_spillover, compute_country_caps


class VisaProblem:
    def __init__(self) -> None:
        self.groups: list[dict] = build_groups()
        self.category_caps: dict[str, int] = compute_spillover(self.groups)
        self.country_caps: dict[str, int] = compute_country_caps(self.groups)
        self.total_visas: int = V
        self.total_demand: int = sum(g["n"] for g in self.groups)

        self._groups_by_country: dict[str, list[dict]] = {}
        for g in self.groups:
            self._groups_by_country.setdefault(g["country"], []).append(g)

        self._country_w_max: dict[str, int] = {}
        for country, gs in self._groups_by_country.items():
            self._country_w_max[country] = max(g["w"] for g in gs)

    def evaluate(self, x: dict[int, int]) -> tuple[float, float, float]:
        return self.f1(x), self.f2(x), self.f3(x)

    def f1(self, x: dict[int, int]) -> float:
        numerator = sum((g["n"] - x[g["index"]]) * g["w"] for g in self.groups)
        return numerator / self.total_demand

    def f2(self, x: dict[int, int]) -> float:
        w_country: dict[str, float] = {}
        for country, gs in self._groups_by_country.items():
            total_assigned = sum(x[g["index"]] for g in gs)
            if total_assigned > 0:
                weighted_wait = sum(x[g["index"]] * g["w"] for g in gs)
                w_country[country] = weighted_wait / total_assigned
            else:
                w_country[country] = float(self._country_w_max[country])
        w_values = list(w_country.values())
        return max(w_values) - min(w_values)

    def f3(self, x: dict[int, int]) -> float:
        return float(self.total_visas - sum(x[g["index"]] for g in self.groups))
