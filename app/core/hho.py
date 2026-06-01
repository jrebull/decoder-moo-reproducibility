"""Harris Hawks Optimization - 6 operators."""

import math
import numpy as np
from numpy.typing import NDArray
from app.core.config import BETA_LEVY, LB, UB


def levy_flight(dim: int, rng: np.random.Generator) -> NDArray[np.float64]:
    beta = BETA_LEVY
    sigma = (
        math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return 0.01 * u / (np.abs(v) ** (1 / beta))


def clip_bounds(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.clip(x, LB, UB)


def escape_energy(t: int, max_t: int, rng: np.random.Generator) -> float:
    e0 = 2 * rng.random() - 1
    return 2 * e0 * (1 - t / max_t)


def op1_exploration_random(xi, x_rand, rng):
    r1 = rng.random()
    r2 = rng.random()
    return clip_bounds(x_rand - r1 * np.abs(x_rand - 2 * r2 * xi))


def op2_exploration_mean(xi, x_rabbit, x_mean, rng):
    dim = xi.shape[0]
    r3 = rng.random()
    r4 = rng.random(size=dim)
    return clip_bounds((x_rabbit - x_mean) - r3 * (LB + r4 * (UB - LB)))


def op3_soft_siege(xi, x_rabbit, energy, rng):
    j = 2 * (1 - rng.random())
    delta_x = x_rabbit - xi
    return clip_bounds(delta_x - energy * np.abs(j * x_rabbit - xi))


def op4_hard_siege(xi, x_rabbit, energy, rng):
    delta_x = x_rabbit - xi
    return clip_bounds(x_rabbit - energy * np.abs(delta_x))


def op5_soft_siege_levy(xi, x_rabbit, energy, rng):
    dim = xi.shape[0]
    j = 2 * (1 - rng.random())
    y = clip_bounds(x_rabbit - energy * np.abs(j * x_rabbit - xi))
    s = rng.random(size=dim)
    z = clip_bounds(y + s * levy_flight(dim, rng))
    return y, z


def op6_hard_siege_levy(xi, x_rabbit, energy, x_mean, rng):
    dim = xi.shape[0]
    j = 2 * (1 - rng.random())
    y = clip_bounds(x_rabbit - energy * np.abs(j * x_rabbit - x_mean))
    s = rng.random(size=dim)
    z = clip_bounds(y + s * levy_flight(dim, rng))
    return y, z
