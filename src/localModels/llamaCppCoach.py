from functools import lru_cache
from pathlib import Path
from typing import Any

from src.localModels.config import validateModelFiles
from src.localModels.cpuRuntime import configureCpuRuntime


class LlamaCppCoachError(RuntimeError):
    """Raised when the local GGUF coach cannot be loaded or used."""


@lru_cache(maxsize=1)
def loadLlamaCppCoach(modelPath: str, contextSize: int, threads: int):
    path = Path(modelPath)
    readiness = validateModelFiles(
        "localCoach",
        {"resolvedPath": str(path), "backend": "llama.cpp"},
    )
    if not readiness["requiredFilesPresent"]:
        raise LlamaCppCoachError(f"The configured local GGUF coaching model is missing: {path}")
    try:
        from llama_cpp import Llama

        return Llama(
            model_path=str(path),
            n_ctx=max(512, min(int(contextSize), 8192)),
            n_threads=max(1, int(threads)),
            n_threads_batch=max(1, int(threads)),
            n_gpu_layers=0,
            use_mmap=True,
            verbose=False,
        )
    except Exception as error:
        raise LlamaCppCoachError(
            f"The local GGUF coaching model at {path} could not be loaded: {error}"
        ) from error


def generateLlamaCppText(
    messages: list[dict[str, str]],
    section: dict[str, Any],
    maxNewTokens: int,
) -> tuple[str, dict[str, Any]]:
    cpu = configureCpuRuntime()
    configuredThreads = section.get("threads", "auto")
    threads = cpu.workerThreads if str(configuredThreads).lower() == "auto" else int(configuredThreads)
    threads = max(1, min(threads, cpu.availableCpus))
    contextSize = int(section.get("contextSize", 2048))
    model = loadLlamaCppCoach(section["resolvedPath"], contextSize, threads)
    try:
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=max(32, min(int(maxNewTokens), 700)),
            temperature=max(0.0, min(float(section.get("temperature", 0.1)), 1.0)),
            top_p=0.9,
            seed=0,
        )
        text = str(response["choices"][0]["message"]["content"] or "").strip()
        usage = response.get("usage", {})
    except Exception as error:
        raise LlamaCppCoachError(f"Local GGUF coaching generation failed: {error}") from error
    return text, {
        "backend": "llama.cpp",
        "contextSize": contextSize,
        "maxNewTokens": max(32, min(int(maxNewTokens), 700)),
        "temperature": max(0.0, min(float(section.get("temperature", 0.1)), 1.0)),
        "threads": threads,
        "nGpuLayers": 0,
        "memoryMapped": True,
        "promptTokens": usage.get("prompt_tokens"),
        "completionTokens": usage.get("completion_tokens"),
    }


def unloadLlamaCppCoach() -> None:
    loadLlamaCppCoach.cache_clear()
