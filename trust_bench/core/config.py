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

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        return cls(**data)
