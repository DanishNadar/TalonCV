import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


projectRoot = Path(__file__).resolve().parents[2]
defaultConfigPath = Path(
    os.environ.get("TALONCV_MODEL_CONFIG", projectRoot / "config" / "models.json")
).expanduser()
if not defaultConfigPath.is_absolute():
    defaultConfigPath = projectRoot / defaultConfigPath
remotePathPattern = re.compile(r"^(?:https?|s3|gs|ftp)://", re.IGNORECASE)

modelSetupCommands = {
    "transcription": "hf download Systran/faster-whisper-small.en --local-dir models/faster-whisper-small.en",
    "faceDetection": 'hf download AdamCodd/YOLOv11n-face-detection --include "*.pt" --local-dir models/yolo11n-face',
    "semanticAnalysis": "hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir models/all-MiniLM-L6-v2",
    "localCoach": "hf download Qwen/Qwen2.5-1.5B-Instruct --local-dir models/qwen2.5-1.5b-instruct",
}


class LocalModelConfigurationError(RuntimeError):
    """Raised when local model configuration could permit a remote lookup."""


def enforceOfflineRuntime() -> None:
    """Enable offline safeguards understood by the installed model libraries."""
    values = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "DO_NOT_TRACK": "true",
        "YOLO_OFFLINE": "true",
    }
    for key, value in values.items():
        os.environ[key] = value


@lru_cache(maxsize=4)
def loadModelConfig(configPath: str | Path = defaultConfigPath) -> dict[str, Any]:
    enforceOfflineRuntime()
    path = Path(configPath).resolve()
    if not path.exists():
        raise LocalModelConfigurationError(
            f"Local model configuration is missing: {path}. Restore config/models.json before starting TalonCV."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LocalModelConfigurationError(f"Could not read local model configuration {path}: {error}") from error
    for name in ("transcription", "faceDetection", "semanticAnalysis", "localCoach"):
        if name not in payload or "path" not in payload[name]:
            raise LocalModelConfigurationError(f"config/models.json is missing {name}.path.")
        resolved = resolveLocalPath(payload[name]["path"])
        payload[name] = {**payload[name], "resolvedPath": str(resolved)}
    runtime = payload.get("runtime", {})
    networkingDisabled = runtime.get("networkingDisabled", runtime.get("networkingDisabledForInference"))
    if networkingDisabled is not True:
        raise LocalModelConfigurationError(
            "runtime.networkingDisabled or runtime.networkingDisabledForInference must be true in the model config."
        )
    runtime["networkingDisabled"] = True
    runtime["networkingDisabledForInference"] = True
    payload["runtime"] = runtime
    payload["configPath"] = str(path)
    return payload


def resolveLocalPath(value: str | Path) -> Path:
    raw = str(value).strip()
    if not raw or remotePathPattern.match(raw) or raw.startswith(("hf://", "hub://")):
        raise LocalModelConfigurationError(f"Model paths must be explicit local filesystem paths, received: {raw!r}")
    path = Path(raw)
    if not path.is_absolute():
        path = projectRoot / path
    return path.resolve()


def getModelSection(name: str, configPath: str | Path = defaultConfigPath) -> dict[str, Any]:
    config = loadModelConfig(str(Path(configPath).resolve()))
    if name not in config:
        raise LocalModelConfigurationError(f"Unknown local model section: {name}")
    section = dict(config[name])
    overrides = {
        "transcription": {"beamSize": "TALONCV_WHISPER_BEAM"},
        "faceDetection": {
            "analysisFps": "TALONCV_VISUAL_ANALYSIS_FPS",
            "imageSize": "TALONCV_YOLO_IMAGE_SIZE",
        },
        "localCoach": {"maxNewTokens": "TALONCV_COACH_MAX_NEW_TOKENS"},
    }.get(name, {})
    for key, environmentName in overrides.items():
        if os.environ.get(environmentName):
            try:
                section[key] = float(os.environ[environmentName]) if key == "analysisFps" else int(os.environ[environmentName])
            except ValueError as error:
                raise LocalModelConfigurationError(f"{environmentName} must be numeric.") from error
    return section


def selectDevice(configured: str = "auto") -> str:
    configured = str(configured or "auto").lower()
    if configured not in {"auto", "cpu", "cuda"}:
        raise LocalModelConfigurationError(f"Unsupported local inference device: {configured}")
    try:
        import torch

        cudaAvailable = bool(torch.cuda.is_available())
    except ImportError:
        cudaAvailable = False
    if configured == "auto":
        return "cuda" if cudaAvailable else "cpu"
    if configured == "cuda" and not cudaAvailable:
        raise LocalModelConfigurationError(
            "CUDA was selected in config/models.json, but CUDA is unavailable. Set the device to auto or cpu."
        )
    return configured


def directorySizeBytes(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validateModelFiles(name: str, section: dict[str, Any] | None = None) -> dict[str, Any]:
    section = section or getModelSection(name)
    path = Path(section["resolvedPath"])
    expected = {
        "transcription": ["model.bin", "config.json", "tokenizer.json"],
        "faceDetection": [path.name],
        "semanticAnalysis": ["config.json", "tokenizer.json"],
        "localCoach": [path.name] if str(section.get("backend", "transformers")).lower() == "llama.cpp" else ["config.json", "tokenizer.json"],
    }[name]
    if name == "faceDetection":
        missing = [] if path.is_file() and path.suffix.lower() == ".pt" else [str(path)]
    elif name == "localCoach" and str(section.get("backend", "transformers")).lower() == "llama.cpp":
        missing = [] if path.is_file() and path.suffix.lower() == ".gguf" else [str(path)]
    else:
        missing = [filename for filename in expected if not (path / filename).exists()]
        if name in {"semanticAnalysis", "localCoach"}:
            hasWeights = any(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists()
            if not hasWeights:
                missing.append("*.safetensors or pytorch_model.bin")
    setupCommand = modelSetupCommands[name]
    if name == "localCoach" and str(section.get("backend", "transformers")).lower() == "llama.cpp":
        setupCommand = (
            "hf download Qwen/Qwen2.5-1.5B-Instruct-GGUF "
            "qwen2.5-1.5b-instruct-q4_k_m.gguf "
            "--local-dir /models/qwen2.5-1.5b-instruct-gguf"
        )
    return {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "requiredFilesPresent": not missing,
        "missingFiles": missing,
        "sizeBytes": directorySizeBytes(path),
        "sizeGb": round(directorySizeBytes(path) / (1024**3), 3),
        "setupCommand": setupCommand,
    }


def getModelRuntimeSignature() -> dict[str, Any]:
    signature = {}
    for name in ("transcription", "faceDetection", "semanticAnalysis", "localCoach"):
        section = getModelSection(name)
        readiness = validateModelFiles(name, section)
        path = Path(section["resolvedPath"])
        latestModified = 0
        if path.is_file():
            latestModified = path.stat().st_mtime_ns
        elif path.is_dir():
            markers = [
                item
                for pattern in ("config.json", "model.bin", "*.pt", "*.safetensors")
                for item in path.glob(pattern)
                if item.is_file()
            ]
            latestModified = max((item.stat().st_mtime_ns for item in markers), default=0)
        signature[name] = {
            "path": str(path),
            "ready": readiness["requiredFilesPresent"],
            "sizeBytes": readiness["sizeBytes"],
            "latestModifiedTimeNs": latestModified,
            "settings": {
                key: section.get(key)
                for key in (
                    "enabled",
                    "device",
                    "computeTypeCpu",
                    "computeTypeCuda",
                    "maxSequenceLength",
                    "maxInputTokens",
                    "maxNewTokens",
                    "repetitionPenalty",
                    "backend",
                    "contextSize",
                    "temperature",
                    "threads",
                    "beamSize",
                    "analysisFps",
                    "imageSize",
                )
                if key in section
            },
        }
    return signature


enforceOfflineRuntime()
