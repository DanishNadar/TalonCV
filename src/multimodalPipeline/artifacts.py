import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


projectRoot = Path(__file__).resolve().parents[2]
demoRoot = projectRoot / "data" / "demo"
analysisVersion = "multimodal-v4"


@dataclass(frozen=True)
class ArtifactPaths:
    media: Path
    audio: Path
    audioMeta: Path
    session: Path
    transcriptText: Path
    transcriptJson: Path
    audioFeatures: Path
    audioEvents: Path
    responseAnalysis: Path
    semanticAnalysis: Path
    visualFeatures: Path
    visualEvents: Path
    multimodalMoments: Path
    multimodal: Path
    scores: Path
    deterministicReport: Path
    report: Path
    localCoaching: Path
    localCoachMeta: Path
    diagnostics: Path
    llmReadyJson: Path
    llmReadyPrompt: Path

    def asDict(self) -> dict[str, Path]:
        return asdict(self)


def getArtifactPaths(mediaPath: str | Path, outputRoot: str | Path | None = None) -> ArtifactPaths:
    mediaPath = Path(mediaPath)
    stem = mediaPath.stem
    root = Path(outputRoot) if outputRoot is not None else demoRoot
    reportRoot = root / "reports" if outputRoot is not None else projectRoot / "reports"
    return ArtifactPaths(
        media=mediaPath,
        audio=root / "audio" / f"{stem}.wav",
        audioMeta=root / "audio" / f"{stem}_audio_meta.json",
        session=root / "sessions" / f"{stem}_session.json",
        transcriptText=root / "transcripts" / f"{stem}_transcript.txt",
        transcriptJson=root / "transcripts" / f"{stem}_transcript.json",
        audioFeatures=root / "audioFeatures" / f"{stem}_audio_features.json",
        audioEvents=root / "audioEvents" / f"{stem}_audio_events.json",
        responseAnalysis=root / "responseAnalysis" / f"{stem}_response_analysis.json",
        semanticAnalysis=root / "semanticAnalysis" / f"{stem}_semantic_analysis.json",
        visualFeatures=root / "features" / f"{stem}_features.csv",
        visualEvents=root / "events" / f"{stem}_events.json",
        multimodalMoments=root / "multimodal" / f"{stem}_moments.json",
        multimodal=root / "multimodal" / f"{stem}_multimodal.json",
        scores=root / "scores" / f"{stem}_scores.json",
        deterministicReport=reportRoot / f"{stem}_deterministic_review.md",
        report=reportRoot / f"{stem}_review.md",
        localCoaching=reportRoot / f"{stem}_local_coaching.md",
        localCoachMeta=reportRoot / f"{stem}_local_coaching_meta.json",
        diagnostics=root / "diagnostics" / f"{stem}_diagnostics.json",
        llmReadyJson=root / "llmReady" / f"{stem}_llmReady.json",
        llmReadyPrompt=root / "llmReady" / f"{stem}_llmReadyPrompt.txt",
    )


def ensureArtifactFolders(paths: ArtifactPaths) -> None:
    for path in paths.asDict().values():
        if path == paths.media:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)


def mediaFingerprint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return {
        "sizeBytes": stat.st_size,
        "modifiedTimeNs": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def stableHash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def readJson(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return default


def writeJson(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def writeText(path: str | Path, value: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)
    return path


def saveSessionContext(
    mediaPath: str | Path,
    context: dict[str, Any],
    paths: ArtifactPaths | None = None,
) -> Path:
    paths = paths or getArtifactPaths(mediaPath)
    fingerprint = mediaFingerprint(mediaPath)
    normalized = {
        "recordingStem": Path(mediaPath).stem,
        "interviewQuestion": str(context.get("interviewQuestion") or "").strip(),
        "targetRole": str(context.get("targetRole") or "").strip(),
        "jobDescription": str(context.get("jobDescription") or "").strip(),
        "desiredCompetencies": str(context.get("desiredCompetencies") or "").strip(),
        "sourceFingerprint": fingerprint,
        "analysisVersion": analysisVersion,
    }
    return writeJson(paths.session, normalized)


def loadSessionContext(mediaPath: str | Path) -> dict[str, Any]:
    paths = getArtifactPaths(mediaPath)
    return readJson(paths.session, {}) or {}
