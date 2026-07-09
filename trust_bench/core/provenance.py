import dataclasses
import hashlib
import os
import platform
from dataclasses import dataclass


@dataclass
class EnvProvenance:
    backend_name: str
    backend_version: str
    language_runtime: str
    blas_lapack: str
    os: str
    cpu_model: str
    cpu_count: int
    machine_fingerprint: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "EnvProvenance":
        return cls(**data)


def _detect_cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def capture() -> EnvProvenance:
    python_version = platform.python_version()
    os_name = f"{platform.system()} {platform.release()}"
    cpu_model = _detect_cpu_model()
    cpu_count = os.cpu_count() or 1
    machine_fingerprint = hashlib.sha256(
        f"{platform.node()}|{cpu_model}|{os_name}".encode()
    ).hexdigest()

    return EnvProvenance(
        backend_name="python",
        backend_version=python_version,
        language_runtime=f"{platform.python_implementation()} {python_version}",
        blas_lapack="unknown",
        os=os_name,
        cpu_model=cpu_model,
        cpu_count=cpu_count,
        machine_fingerprint=machine_fingerprint,
    )
