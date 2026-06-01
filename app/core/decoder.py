"""SPV decoding and greedy decoder."""

import numpy as np
from numpy.typing import NDArray


def spv(hawk: NDArray[np.float64]) -> list[int]:
    return list(np.argsort(hawk, kind="stable"))


def decode(
    permutation: list[int],
    groups: list[dict],
    total_visas: int,
    country_caps: dict[str, int],
    category_caps: dict[str, int],
) -> dict[int, int]:
    x: dict[int, int] = {g["index"]: 0 for g in groups}
    v_remaining = total_visas
    country_usage: dict[str, int] = {}
    category_usage: dict[str, int] = {}
    group_lookup = {g["index"]: g for g in groups}

    for g_idx in permutation:
        g = group_lookup[g_idx]
        country = g["country"]
        category = g["category"]
        cap_country = country_caps[country] - country_usage.get(country, 0)
        cap_category = category_caps[category] - category_usage.get(category, 0)
        x_g = min(g["n"], v_remaining, cap_country, cap_category)
        x[g_idx] = x_g
        v_remaining -= x_g
        country_usage[country] = country_usage.get(country, 0) + x_g
        category_usage[category] = category_usage.get(category, 0) + x_g

    return x
