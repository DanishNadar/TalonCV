import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from src.audioPipeline.mediaUtils import inspectMedia, supportedMediaExtensions


class MediaValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def canonicalUuid(value: Any, field: str = "identifier") -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(f"Invalid {field}.") from error


def validateStoragePath(path: str, expectedUserId: str | None = None) -> str:
    normalized = str(PurePosixPath(str(path)))
    if not path or path.startswith(("/", "\\")) or ".." in PurePosixPath(path).parts or "\\" in path:
        raise ValueError("Invalid storage object path.")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", normalized):
        raise ValueError("Storage object path contains unsupported characters.")
    if expectedUserId:
        expected = f"users/{canonicalUuid(expectedUserId, 'user id')}/"
        if not normalized.startswith(expected):
            raise ValueError("Storage object path does not belong to the expected user.")
    return normalized


def recordingStoragePath(userId: str, sessionId: str, recordingId: str, extension: str = ".webm") -> str:
    suffix = extension.lower() if extension.lower() in supportedMediaExtensions else ".webm"
    return (
        f"users/{canonicalUuid(userId, 'user id')}/sessions/{canonicalUuid(sessionId, 'session id')}"
        f"/recordings/{canonicalUuid(recordingId, 'recording id')}/original{suffix}"
    )


def artifactStoragePath(userId: str, sessionId: str, jobId: str, filename: str) -> str:
    safeName = Path(filename).name
    if safeName != filename or not re.fullmatch(r"[A-Za-z0-9._-]+", safeName):
        raise ValueError("Invalid artifact filename.")
    return (
        f"users/{canonicalUuid(userId, 'user id')}/sessions/{canonicalUuid(sessionId, 'session id')}"
        f"/analysis/{canonicalUuid(jobId, 'job id')}/{safeName}"
    )


def validateMediaFile(path: str | Path, maxBytes: int, maxDurationSeconds: int, maxDimension: int) -> dict[str, Any]:
    mediaPath = Path(path)
    if not mediaPath.is_file() or mediaPath.stat().st_size <= 0:
        raise MediaValidationError("missing_media", "The uploaded recording is missing or empty.")
    if mediaPath.stat().st_size > maxBytes:
        raise MediaValidationError("oversize_media", "The recording exceeds the 250 MB public limit.")
    if mediaPath.suffix.lower() not in supportedMediaExtensions:
        raise MediaValidationError("unsupported_media", "The recording format is not supported.")
    info = inspectMedia(mediaPath)
    if not info.get("valid"):
        raise MediaValidationError("corrupt_media", "The recording could not be decoded.")
    duration = float(info.get("durationSeconds") or 0)
    if duration <= 0:
        raise MediaValidationError("corrupt_media", "The recording has no measurable duration.")
    if duration > maxDurationSeconds + 0.5:
        raise MediaValidationError("excessive_duration", "The recording exceeds the five-minute public limit.")
    video = info.get("video") or {}
    if max(int(video.get("width") or 0), int(video.get("height") or 0)) > maxDimension:
        raise MediaValidationError("pathological_dimensions", "The video dimensions exceed the supported limit.")
    return info
