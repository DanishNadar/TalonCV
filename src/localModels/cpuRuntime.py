import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class CpuRuntimeConfiguration:
    availableCpus: int
    workerThreads: int
    interopThreads: int
    cpuOnly: bool

    def asDict(self) -> dict[str, Any]:
        return asdict(self)


def _positiveInteger(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


@lru_cache(maxsize=1)
def getCpuRuntimeConfiguration() -> CpuRuntimeConfiguration:
    available = max(1, os.cpu_count() or 2)
    defaultThreads = max(1, available - 1)
    requested = _positiveInteger(os.environ.get("TALONCV_CPU_THREADS"), defaultThreads)
    workerThreads = max(1, min(requested, available))
    interopThreads = max(1, min(2, workerThreads))
    return CpuRuntimeConfiguration(
        availableCpus=available,
        workerThreads=workerThreads,
        interopThreads=interopThreads,
        cpuOnly=os.environ.get("TALONCV_ENV", "local").lower() == "production",
    )


def configureCpuRuntime() -> CpuRuntimeConfiguration:
    """Bound native runtimes before model imports and configure PyTorch when present."""
    configuration = getCpuRuntimeConfiguration()
    value = str(configuration.workerThreads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        import torch

        torch.set_num_threads(configuration.workerThreads)
        try:
            torch.set_num_interop_threads(configuration.interopThreads)
        except RuntimeError:
            # PyTorch only permits this before its first parallel operation.
            pass
    except ImportError:
        pass
    return configuration
