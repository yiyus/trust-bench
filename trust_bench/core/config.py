import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    max_iter: int | None = None
    tolerance: float | None = None
    bounds: Any = None
    derivative_mode: str | None = None
    loss: str = "linear"
    x_scale: Any = None
    f_scale: float | None = None
    # Opt-in: a plain solve is a single call (timing=None), unchanged
    # from before RunResult.timing existed. Repeated warm-up+measured
    # solves cost real wall-clock time - multiplying every call by
    # WARMUP+N_REPS regardless of problem cost is what real report
    # generation wants, not what an ordinary correctness check needs.
    measure_timing: bool = False

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**data)
