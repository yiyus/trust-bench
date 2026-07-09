import dataclasses
from dataclasses import dataclass
from enum import Enum

from trust_bench.core.config import RunConfig
from trust_bench.core.provenance import EnvProvenance


class RunStatus(Enum):
    CONVERGED = "CONVERGED"
    MAX_ITER = "MAX_ITER"
    FAILED = "FAILED"
    DIVERGED = "DIVERGED"
    ERROR = "ERROR"


@dataclass
class TimingStats:
    median: float
    mad: float
    n_reps: int
    warmup: int
    thread_count: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TimingStats":
        return cls(**data)


@dataclass
class RunResult:
    problem_id: str
    backend: str
    method: str
    start: str
    x_final: list[float]
    cost_final: float
    dist_to_opt: float | None
    cost_gap: float | None
    grad_norm_final: float | None
    status: RunStatus
    n_iter: int | None
    n_feval: int | None
    n_jeval: int | None
    n_heval: int | None
    trace: list[list[float]] | None
    timing: TimingStats | None
    config: RunConfig
    provenance: EnvProvenance
    harness_git_sha: str
    timestamp: str

    def __post_init__(self):
        if self.provenance is None:
            raise TypeError("RunResult requires provenance")

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        data = dict(data)
        data["status"] = RunStatus(data["status"])
        data["provenance"] = EnvProvenance.from_dict(data["provenance"])
        data["config"] = RunConfig.from_dict(data["config"])
        if data["timing"] is not None:
            data["timing"] = TimingStats.from_dict(data["timing"])
        return cls(**data)
