from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass(frozen=True)
class Optimum:
    x_star: np.ndarray
    cost_star: float
    basin_notes: str = ""


@dataclass(frozen=True)
class Problem:
    id: str
    residual: Callable
    jacobian: Callable | None
    hessian: Callable | None
    starts: dict[str, np.ndarray]
    optima: list[Optimum]
    kind: Literal["residuals", "scalar"]
    tags: frozenset[str]
    probe_points: list[np.ndarray]
    source: str
