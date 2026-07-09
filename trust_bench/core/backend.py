from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from trust_bench.core.config import RunConfig
from trust_bench.core.problem import Problem
from trust_bench.core.provenance import EnvProvenance
from trust_bench.core.result import RunResult


@dataclass(frozen=True)
class MethodCapabilities:
    kind: Literal["residuals", "scalar"]
    losses: frozenset[str]
    bounds: bool
    analytic_hessian: bool
    derivative_modes: frozenset[str]


@dataclass(frozen=True)
class Capabilities:
    methods: dict[str, MethodCapabilities]


class Backend(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abstractmethod
    def environment(self) -> EnvProvenance: ...

    @abstractmethod
    def solve(self, problem: Problem, method: str, start: str, config: RunConfig) -> RunResult: ...
