import shutil
import wave
from pathlib import Path
from typing import Any

import av
import numpy as np

from src.multimodalPipeline.artifacts import mediaFingerprint, readJson, writeJson


supportedMediaExtensions = {
    ".mp4",
    ".mov",
    ".m4v",
    ".webm",
    ".mkv",
    ".avi",
    ".wav",
    ".m4a",
    ".mp3",
    ".aac",
    ".ogg",
    ".flac",
}


class MediaAnalysisError(RuntimeError):
    """Actionable media validation or decoding error for the UI."""


def inspectMedia(mediaPath: str | Path) -> dict[str, Any]:
    mediaPath = Path(mediaPath)
    result: dict[str, Any] = {
        "path": str(mediaPath.resolve()),
        "valid": False,
        "hasVideo": False,
        "hasAudio": False,
        "durationSeconds": None,
        "video": None,
        "audio": None,
        "warnings": [],
    }
    if not mediaPath.exists() or mediaPath.stat().st_size == 0:
        result["warnings"].append("The selected media file is missing or empty.")
        return result

    try:
        with av.open(str(mediaPath)) as container:
            videoStream = next((stream for stream in container.streams if stream.type == "video"), None)
            audioStream = next((stream for stream in container.streams if stream.type == "audio"), None)
            result["hasVideo"] = videoStream is not None
            result["hasAudio"] = audioStream is not None
            if container.duration is not None:
                result["durationSeconds"] = round(float(container.duration / av.time_base), 5)
            if videoStream is not None:
                videoDuration = _streamDuration(videoStream)
                result["video"] = {
                    "codec": videoStream.codec_context.name,
                    "width": videoStream.codec_context.width,
                    "height": videoStream.codec_context.height,
                    "averageRate": float(videoStream.average_rate) if videoStream.average_rate else None,
                    "durationSeconds": videoDuration,
                }
                result["durationSeconds"] = result["durationSeconds"] or videoDuration
            if audioStream is not None:
                audioDuration = _streamDuration(audioStream)
                result["audio"] = {
                    "codec": audioStream.codec_context.name,
                    "sampleRate": audioStream.codec_context.sample_rate,
                    "channels": audioStream.codec_context.channels,
                    "durationSeconds": audioDuration,
                }
                result["durationSeconds"] = result["durationSeconds"] or audioDuration
            result["valid"] = result["hasVideo"] or result["hasAudio"]
    except (av.error.FFmpegError, OSError, ValueError) as error:
        result["warnings"].append(f"The media container could not be decoded: {error}")
    return result


def _streamDuration(stream: Any) -> float | None:
    if stream.duration is None or stream.time_base is None:
        return None
    return round(float(stream.duration * stream.time_base), 5)


def extractAudioToWav(
    mediaPath: str | Path,
    wavPath: str | Path,
    metadataPath: str | Path | None = None,
    targetSampleRate: int = 16000,
    force: bool = False,
) -> dict[str, Any]:
    """Decode any supported audio stream into deterministic mono PCM16 WAV."""
    mediaPath = Path(mediaPath)
    wavPath = Path(wavPath)
    metadataPath = Path(metadataPath) if metadataPath else wavPath.with_suffix(".meta.json")
    fingerprint = mediaFingerprint(mediaPath)
    cached = readJson(metadataPath, {})
    if (
        not force
        and wavPath.exists()
        and wavPath.stat().st_size > 44
        and cached.get("sourceFingerprint") == fingerprint
        and cached.get("sampleRate") == targetSampleRate
    ):
        return {**cached, "cached": True}

    mediaInfo = inspectMedia(mediaPath)
    if not mediaInfo["valid"]:
        raise MediaAnalysisError(mediaInfo["warnings"][0] if mediaInfo["warnings"] else "Invalid media file.")
    if not mediaInfo["hasAudio"]:
        return {
            "available": False,
            "cached": False,
            "sourceFingerprint": fingerprint,
            "warnings": ["No audio stream was found in the selected media."],
        }

    wavPath.parent.mkdir(parents=True, exist_ok=True)
    sampleBlocks: list[np.ndarray] = []
    try:
        with av.open(str(mediaPath)) as container:
            audioStream = next(stream for stream in container.streams if stream.type == "audio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=targetSampleRate)
            for frame in container.decode(audioStream):
                converted = resampler.resample(frame)
                convertedFrames = converted if isinstance(converted, list) else [converted]
                for convertedFrame in convertedFrames:
                    if convertedFrame is not None:
                        sampleBlocks.append(convertedFrame.to_ndarray().reshape(-1).astype(np.int16))
            flushed = resampler.resample(None)
            flushedFrames = flushed if isinstance(flushed, list) else [flushed]
            for convertedFrame in flushedFrames:
                if convertedFrame is not None:
                    sampleBlocks.append(convertedFrame.to_ndarray().reshape(-1).astype(np.int16))
    except (av.error.FFmpegError, OSError, ValueError) as error:
        raise MediaAnalysisError(f"The audio stream could not be decoded: {error}") from error

    samples = np.concatenate(sampleBlocks) if sampleBlocks else np.array([], dtype=np.int16)
    if samples.size == 0:
        return {
            "available": False,
            "cached": False,
            "sourceFingerprint": fingerprint,
            "warnings": ["An audio stream exists, but it contained no decodable samples."],
        }

    temporary = wavPath.with_suffix(wavPath.suffix + ".tmp")
    with wave.open(str(temporary), "wb") as wavFile:
        wavFile.setnchannels(1)
        wavFile.setsampwidth(2)
        wavFile.setframerate(targetSampleRate)
        wavFile.writeframes(samples.tobytes())
    temporary.replace(wavPath)

    peak = float(np.max(np.abs(samples.astype(np.float32))) / 32768.0) if samples.size else 0.0
    metadata = {
        "available": True,
        "cached": False,
        "sourcePath": str(mediaPath.resolve()),
        "wavPath": str(wavPath.resolve()),
        "sourceFingerprint": fingerprint,
        "sampleRate": targetSampleRate,
        "channels": 1,
        "sampleCount": int(samples.size),
        "durationSeconds": round(samples.size / targetSampleRate, 5),
        "peakAmplitude": round(peak, 6),
        "warnings": ["The decoded audio is effectively silent."] if peak < 0.001 else [],
    }
    writeJson(metadataPath, metadata)
    return metadata


def copyUploadedMedia(uploadedFile: Any, outputDirectory: str | Path) -> Path:
    outputDirectory = Path(outputDirectory)
    outputDirectory.mkdir(parents=True, exist_ok=True)
    extension = Path(uploadedFile.name).suffix.lower()
    if extension not in supportedMediaExtensions:
        raise MediaAnalysisError(f"Unsupported media extension: {extension or '(none)'}.")
    outputPath = outputDirectory / uploadedFile.name
    with outputPath.open("wb") as destination:
        shutil.copyfileobj(uploadedFile, destination)
    return outputPath


def codecDiagnostics() -> dict[str, Any]:
    checks: dict[str, Any] = {"pyavVersion": av.__version__, "h264Encoder": False, "aacEncoder": False}
    try:
        av.codec.Codec("libx264", "w")
        checks["h264Encoder"] = True
    except (av.error.FFmpegError, ValueError):
        pass
    try:
        av.codec.Codec("aac", "w")
        checks["aacEncoder"] = True
    except (av.error.FFmpegError, ValueError):
        pass
    checks["ready"] = checks["h264Encoder"] and checks["aacEncoder"]
    return checks

