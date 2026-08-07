import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.localModels.config import (
    LocalModelConfigurationError,
    getModelSection,
    modelSetupCommands,
    selectDevice,
    validateModelFiles,
)
from src.localModels.cpuRuntime import configureCpuRuntime
from src.multimodalPipeline.artifacts import readJson, writeJson, writeText


transcriptionVersion = "transcription-v1"


class TranscriptionError(RuntimeError):
    """Actionable local transcription error suitable for the web interface."""


def getTranscriptionConfig() -> dict[str, Any]:
    try:
        section = getModelSection("transcription")
        device = selectDevice(section.get("device", "auto"))
    except LocalModelConfigurationError as error:
        raise TranscriptionError(str(error)) from error
    computeType = section.get("computeTypeCuda", "float16") if device == "cuda" else section.get("computeTypeCpu", "int8")
    cpu = configureCpuRuntime()
    return {
        "path": section["resolvedPath"],
        "device": device,
        "computeType": str(computeType),
        "beamSize": max(1, min(int(section.get("beamSize", 5)), 8)),
        "cpuThreads": cpu.workerThreads if device == "cpu" else None,
    }


@lru_cache(maxsize=3)
def _loadModel(modelPath: str, device: str, computeType: str, cpuThreads: int | None = None):
    path = Path(modelPath)
    readiness = validateModelFiles("transcription", {"resolvedPath": str(path)})
    if not readiness["requiredFilesPresent"]:
        raise TranscriptionError(
            f"The local faster-whisper model is missing or incomplete at {path}. "
            f"Run: {modelSetupCommands['transcription']}"
        )
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise TranscriptionError(
            "Local transcription is unavailable because faster-whisper is not installed. "
            "Install requirements.txt and restart the app."
        ) from error
    try:
        options: dict[str, Any] = {
            "device": device,
            "compute_type": computeType,
            "local_files_only": True,
        }
        if device == "cpu" and cpuThreads is not None:
            options.update(cpu_threads=max(1, int(cpuThreads)), num_workers=1)
        return WhisperModel(str(path), **options)
    except Exception as error:
        raise TranscriptionError(
            f"The local transcription model at '{path}' could not be loaded. "
            "Verify the local files and configured device/compute type. TalonCV will not download a model at runtime. "
            f"Technical detail: {error}"
        ) from error


def warmTranscriptionModel(progressCallback: Callable[[str], None] | None = None) -> dict[str, str]:
    config = getTranscriptionConfig()
    if progressCallback:
        progressCallback(f"Loading local transcription model from {config['path']}...")
    _loadModel(config["path"], config["device"], config["computeType"], config.get("cpuThreads"))
    return config


def transcribeAudio(
    wavPath: str | Path,
    recordingStem: str,
    transcriptJsonPath: str | Path,
    transcriptTextPath: str | Path,
    sourceFingerprint: dict[str, Any],
    force: bool = False,
    progressCallback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    config = getTranscriptionConfig()
    cached = readJson(transcriptJsonPath, {})
    if (
        not force
        and cached.get("analysisVersion") == transcriptionVersion
        and cached.get("sourceFingerprint") == sourceFingerprint
        and cached.get("modelConfiguration") == config
        and Path(transcriptTextPath).exists()
    ):
        cached["cached"] = True
        return cached

    model = _loadModel(config["path"], config["device"], config["computeType"], config.get("cpuThreads"))
    if progressCallback:
        progressCallback("Transcribing speech locally with word timestamps...")
    try:
        segmentIterator, info = model.transcribe(
            str(wavPath),
            beam_size=config["beamSize"],
            word_timestamps=True,
            vad_filter=True,
            condition_on_previous_text=True,
        )
        segments = []
        confidenceValues = []
        for segment in segmentIterator:
            words = []
            for word in segment.words or []:
                probability = float(word.probability) if word.probability is not None else None
                if probability is not None:
                    confidenceValues.append(probability)
                words.append(
                    {
                        "start": _round(word.start),
                        "end": _round(word.end),
                        "word": str(word.word),
                        "probability": _round(probability, 5),
                    }
                )
            averageLogProbability = (
                float(segment.avg_logprob) if getattr(segment, "avg_logprob", None) is not None else None
            )
            segmentConfidence = (
                max(0.0, min(1.0, math.exp(averageLogProbability)))
                if averageLogProbability is not None
                else None
            )
            if segmentConfidence is not None:
                confidenceValues.append(segmentConfidence)
            segments.append(
                {
                    "id": int(segment.id),
                    "start": _round(segment.start),
                    "end": _round(segment.end),
                    "text": str(segment.text).strip(),
                    "confidence": _round(segmentConfidence, 5),
                    "avgLogProbability": _round(averageLogProbability, 5),
                    "noSpeechProbability": _round(getattr(segment, "no_speech_prob", None), 5),
                    "words": words,
                }
            )
    except Exception as error:
        raise TranscriptionError(
            "Local speech transcription failed. Confirm that the decoded WAV is playable and the configured "
            f"model is available. Technical detail: {error}"
        ) from error

    fullText = " ".join(segment["text"] for segment in segments if segment["text"]).strip()
    warnings = []
    if not fullText:
        warnings.append("No speech was detected in the decoded audio.")
    averageConfidence = sum(confidenceValues) / len(confidenceValues) if confidenceValues else None
    if averageConfidence is not None and averageConfidence < 0.55:
        warnings.append(
            "Transcription confidence was limited; content-related conclusions should be treated cautiously."
        )
    transcript = {
        "analysisVersion": transcriptionVersion,
        "recordingIdentifier": recordingStem,
        "sourceFingerprint": sourceFingerprint,
        "language": getattr(info, "language", None),
        "languageProbability": _round(getattr(info, "language_probability", None), 5),
        "durationSeconds": _round(getattr(info, "duration", None)),
        "durationAfterVadSeconds": _round(getattr(info, "duration_after_vad", None)),
        "text": fullText,
        "segments": segments,
        "averageConfidence": _round(averageConfidence, 5),
        "modelConfiguration": config,
        "warnings": warnings,
        "cached": False,
    }
    writeText(transcriptTextPath, fullText + ("\n" if fullText else ""))
    writeJson(transcriptJsonPath, transcript)
    return transcript


def transcriptionDiagnostics(loadModel: bool = False) -> dict[str, Any]:
    config = getTranscriptionConfig()
    try:
        import faster_whisper

        installed = True
        version = getattr(faster_whisper, "__version__", "unknown")
    except ImportError:
        installed = False
        version = None
    loadError = None
    loaded = _loadModel.cache_info().currsize > 0
    readiness = validateModelFiles("transcription")
    if loadModel and readiness["requiredFilesPresent"]:
        try:
            _loadModel(config["path"], config["device"], config["computeType"], config.get("cpuThreads"))
            loaded = True
        except TranscriptionError as error:
            loadError = str(error)
    return {
        "installed": installed,
        "version": version,
        "configuration": config,
        "modelLoaded": loaded,
        "loaded": loaded,
        "loadError": loadError,
        **readiness,
        "localPathReady": readiness["requiredFilesPresent"],
        "runtimeDownloadsDisabled": True,
        "setupCommand": modelSetupCommands["transcription"],
        "note": "Only an explicit local model directory is accepted; runtime downloads are disabled.",
    }


def _round(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return round(numeric, digits) if math.isfinite(numeric) else None
