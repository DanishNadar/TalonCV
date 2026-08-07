import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.audioPipeline.audioAnalyzer import analyzeAudio, audioAnalysisVersion
from src.audioPipeline.mediaUtils import MediaAnalysisError, extractAudioToWav, inspectMedia
from src.audioPipeline.semanticAnalyzer import SemanticAnalysisError, analyzeSemanticResponse
from src.audioPipeline.transcriptAnalyzer import analyzeTranscript
from src.audioPipeline.transcription import (
    TranscriptionError,
    getTranscriptionConfig,
    transcribeAudio,
    transcriptionVersion,
)
from src.multimodalPipeline.alignment import alignMultimodalEvents, alignmentVersion
from src.multimodalPipeline.artifacts import (
    analysisVersion,
    ensureArtifactFolders,
    getArtifactPaths,
    mediaFingerprint,
    readJson,
    saveSessionContext,
    stableHash,
    writeJson,
    writeText,
)
from src.multimodalPipeline.reportBuilder import buildMultimodalReport
from src.multimodalPipeline.scoring import buildCoachingScores
from src.localModels.config import getModelRuntimeSignature
from src.localModels.cpuRuntime import configureCpuRuntime
from src.localModels.localCoach import LocalCoachError, generateLocalCoaching


ProgressCallback = Callable[[str, float, str], None]


class MultimodalAnalysisError(RuntimeError):
    """Top-level actionable analysis error for Streamlit and CLI callers."""


def runMultimodalAnalysis(
    mediaPath: str | Path,
    sessionContext: dict[str, Any] | None = None,
    force: bool = False,
    progressCallback: ProgressCallback | None = None,
    yoloFaceDetector: Any = None,
    transcriber: Callable[..., dict[str, Any]] = transcribeAudio,
    semanticAnalyzer: Callable[..., dict[str, Any]] = analyzeSemanticResponse,
    localCoach: Callable[..., dict[str, Any]] = generateLocalCoaching,
    visualAnalyzer: Callable[[Path, Any], tuple[list[dict[str, Any]], list[dict[str, Any]], float]] | None = None,
    artifactRoot: str | Path | None = None,
) -> dict[str, Any]:
    mediaPath = Path(mediaPath)
    cpu = configureCpuRuntime()
    modelSignature = getModelRuntimeSignature()
    paths = getArtifactPaths(mediaPath, artifactRoot)
    ensureArtifactFolders(paths)
    _progress(progressCallback, "validateMedia", 0.02, "Validating the selected local media file...")
    _progress(progressCallback, "inspectStreams", 0.05, "Inspecting local audio and video streams...")
    mediaInfo = inspectMedia(mediaPath)
    if not mediaInfo.get("valid"):
        warning = (mediaInfo.get("warnings") or ["The selected media is invalid or unsupported."])[0]
        raise MultimodalAnalysisError(warning)
    fingerprint = mediaFingerprint(mediaPath)
    requestedContext = sessionContext or {}
    existingContext = readJson(paths.session, {}) or {}
    contextFields = ("interviewQuestion", "targetRole", "jobDescription", "desiredCompetencies")
    context = requestedContext if any(str(requestedContext.get(key, "")).strip() for key in contextFields) else existingContext
    saveSessionContext(mediaPath, context, paths=paths)
    context = readJson(paths.session, {}) or context
    cacheKey = stableHash(
        {
            "analysisVersion": analysisVersion,
            "sourceFingerprint": fingerprint,
            "sessionContext": {
                key: context.get(key)
                for key in ("interviewQuestion", "targetRole", "jobDescription", "desiredCompetencies")
            },
            "localModels": modelSignature,
        }
    )
    cached = readJson(paths.multimodal, {})
    if (
        not force
        and cached.get("complete")
        and cached.get("cacheKey") == cacheKey
        and paths.report.exists()
    ):
        cached["cached"] = True
        _progress(progressCallback, "complete", 1.0, "Reused valid cached multimodal artifacts.")
        return cached

    warnings = list(mediaInfo.get("warnings", []))
    transcript: dict[str, Any] = {}
    audioFeatures: dict[str, Any] = {}
    audioEvents: list[dict[str, Any]] = []
    responseAnalysis: dict[str, Any] = {}
    semanticAnalysis: dict[str, Any] = {}
    visualRows: list[dict[str, Any]] = []
    visualEvents: list[dict[str, Any]] = []

    if mediaInfo.get("hasAudio"):
        _progress(progressCallback, "extractAudio", 0.11, "Extracting and normalizing audio locally...")
        try:
            audioMeta = extractAudioToWav(mediaPath, paths.audio, paths.audioMeta, force=force)
        except MediaAnalysisError as error:
            audioMeta = {"available": False, "warnings": [str(error)]}
            warnings.append(str(error))
        if audioMeta.get("available"):
            _progress(progressCallback, "transcribeLocally", 0.2, "Loading faster-whisper from its local directory...")
            try:
                transcript = transcriber(
                    paths.audio,
                    mediaPath.stem,
                    paths.transcriptJson,
                    paths.transcriptText,
                    fingerprint,
                    force=force,
                    progressCallback=(
                        (lambda message: _progress(progressCallback, "transcribeLocally", 0.28, message))
                        if progressCallback
                        else None
                    ),
                )
            except TranscriptionError as error:
                warnings.append(str(error))
                transcript = {
                    "recordingIdentifier": mediaPath.stem,
                    "text": "",
                    "segments": [],
                    "averageConfidence": None,
                    "warnings": [str(error)],
                    "modelConfiguration": getTranscriptionConfig(),
                }
                writeText(paths.transcriptText, "")
                writeJson(paths.transcriptJson, transcript)

            _progress(progressCallback, "analyzeAudioQuality", 0.34, "Analyzing recording quality, noise, clipping, and dropouts...")
            try:
                audioFeatures, audioEvents = analyzeAudio(paths.audio, transcript, fingerprint)
                writeJson(paths.audioFeatures, audioFeatures)
                writeJson(paths.audioEvents, audioEvents)
                _progress(progressCallback, "analyzeVocalDelivery", 0.42, "Analyzing pace, pauses, volume, pitch, energy, and emphasis...")
            except (OSError, ValueError) as error:
                warning = f"Audio analysis could not be completed: {error}"
                warnings.append(warning)
                audioFeatures = {"available": False, "warnings": [warning]}
                writeJson(paths.audioFeatures, audioFeatures)
                writeJson(paths.audioEvents, [])
        else:
            unavailableWarnings = audioMeta.get("warnings", []) or ["No decodable audio samples were available."]
            warnings.extend(unavailableWarnings)
            transcript, audioFeatures = _saveUnavailableAudioArtifacts(
                paths, fingerprint, mediaPath.stem, unavailableWarnings
            )
    else:
        warning = "The selected media has no audio stream; transcript and vocal analysis were skipped."
        warnings.append(warning)
        transcript, audioFeatures = _saveUnavailableAudioArtifacts(paths, fingerprint, mediaPath.stem, [warning])

    _progress(progressCallback, "analyzeAnswerContent", 0.49, "Analyzing answer clarity, structure, specificity, and evidence...")
    responseAnalysis = analyzeTranscript(transcript, context)
    _progress(progressCallback, "analyzeSemanticRelevance", 0.56, "Running local MiniLM relevance and redundancy analysis...")
    try:
        semanticAnalysis = semanticAnalyzer(transcript, context)
    except SemanticAnalysisError as error:
        semanticAnalysis = {"available": False, "warnings": [str(error)]}
    responseAnalysis = analyzeTranscript(transcript, context, semanticAnalysis=semanticAnalysis)
    writeJson(paths.semanticAnalysis, semanticAnalysis)
    writeJson(paths.responseAnalysis, responseAnalysis)

    if mediaInfo.get("hasVideo"):
        _progress(progressCallback, "analyzeVisualDelivery", 0.64, "Running local YOLOv11 and MediaPipe visual analysis...")
        try:
            analyzer = visualAnalyzer or _defaultVisualAnalyzer
            visualRows, visualEvents, _ = analyzer(mediaPath, yoloFaceDetector)
            _saveVisualArtifacts(paths, visualRows, visualEvents)
        except Exception as error:
            warning = f"Visual analysis was unavailable: {error}"
            warnings.append(warning)
            _saveVisualArtifacts(paths, [], [])
    else:
        warnings.append("The selected media has no video stream; visual analysis was skipped.")
        _saveVisualArtifacts(paths, [], [])

    _progress(progressCallback, "alignEvidence", 0.75, "Aligning transcript, audio, and visual evidence by timestamp...")
    multimodalMoments = alignMultimodalEvents(transcript, responseAnalysis, audioEvents, visualEvents)
    writeJson(paths.multimodalMoments, multimodalMoments)
    _progress(progressCallback, "calculateScores", 0.81, "Calculating explainable scores from available modalities...")
    scoreBundle = buildCoachingScores(
        mediaInfo, audioFeatures, audioEvents, responseAnalysis, visualEvents, multimodalMoments
    )
    writeJson(paths.scores, scoreBundle)

    _progress(progressCallback, "generateDeterministicReport", 0.87, "Building the deterministic evidence-based report...")
    report = buildMultimodalReport(
        mediaPath,
        context,
        mediaInfo,
        transcript,
        responseAnalysis,
        audioFeatures,
        audioEvents,
        visualEvents,
        multimodalMoments,
        scoreBundle,
    )
    writeText(paths.deterministicReport, report)
    writeText(paths.report, report)

    result = {
        "analysisVersion": analysisVersion,
        "alignmentVersion": alignmentVersion,
        "complete": True,
        "cached": False,
        "cacheKey": cacheKey,
        "sourceFingerprint": fingerprint,
        "recordingStem": mediaPath.stem,
        "mediaInfo": mediaInfo,
        "sessionContext": context,
        "transcript": transcript,
        "responseAnalysis": responseAnalysis,
        "semanticAnalysis": semanticAnalysis,
        "audioFeatures": audioFeatures,
        "audioEvents": audioEvents,
        "visualEvents": visualEvents,
        "moments": multimodalMoments,
        "scores": scoreBundle,
        "localCoaching": {},
        "enhancedCoachingStatus": "pending",
        "provenance": {
            "taloncvVersion": analysisVersion,
            "analysisVersion": analysisVersion,
            "gitCommit": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "unknown")[:12],
            "createdAt": datetime.now(UTC).isoformat(),
            "models": modelSignature,
            "cpuConfiguration": cpu.asDict(),
            "analysisConfiguration": {
                "sequentialInference": True,
                "visualAnalysisFps": modelSignature["faceDetection"]["settings"].get("analysisFps"),
                "yoloImageSize": modelSignature["faceDetection"]["settings"].get("imageSize"),
                "whisperBeamSize": modelSignature["transcription"]["settings"].get("beamSize"),
            },
        },
        "performanceMetrics": {
            "recordingDurationSeconds": mediaInfo.get("durationSeconds"),
            "analyzedFrameCount": len(visualRows),
            "transcriptWordCount": len(str(transcript.get("text") or "").split()),
        },
        "warnings": list(
            dict.fromkeys(warnings + _collectWarnings(transcript, responseAnalysis, semanticAnalysis, audioFeatures))
        ),
        "artifactPaths": {name: str(path) for name, path in paths.asDict().items()},
    }
    # Save the complete deterministic result before potentially slow optional
    # local generation. A crash or interruption cannot erase core analysis.
    writeJson(paths.multimodal, result)
    _progress(progressCallback, "generateLocalCoaching", 0.92, "Generating enhanced coaching with the local Qwen model...")
    paths.localCoaching.unlink(missing_ok=True)
    paths.localCoachMeta.unlink(missing_ok=True)
    try:
        enhancedCoaching = localCoach(
            result,
            progressCallback=(
                (lambda message: _progress(progressCallback, "generateLocalCoaching", 0.95, message))
                if progressCallback
                else None
            ),
        )
        result["localCoaching"] = enhancedCoaching
        result["enhancedCoachingStatus"] = "complete" if enhancedCoaching.get("available") else "unavailable"
        if enhancedCoaching.get("available"):
            localReport = _buildLocalCoachingReport(enhancedCoaching)
            writeText(paths.localCoaching, localReport)
            writeJson(
                paths.localCoachMeta,
                {
                    "analysisVersion": enhancedCoaching.get("analysisVersion"),
                    "modelPath": enhancedCoaching.get("modelPath"),
                    "device": enhancedCoaching.get("device"),
                    "generation": enhancedCoaching.get("generation", {}),
                    "localFilesOnly": True,
                    "sourceFingerprint": fingerprint,
                },
            )
            report = report + "\n\n## Local enhanced coaching\n\n" + enhancedCoaching["text"] + "\n"
            writeText(paths.report, report)
    except LocalCoachError as error:
        warning = str(error)
        result["warnings"].append(warning)
        result["localCoaching"] = {"available": False, "warnings": [warning]}
        result["enhancedCoachingStatus"] = "failed"

    _progress(progressCallback, "validateArtifacts", 0.98, "Validating and atomically saving local artifacts...")
    writeJson(paths.multimodal, result)
    _progress(progressCallback, "complete", 1.0, "Offline multimodal analysis complete.")
    return result


def loadMultimodalAnalysis(mediaPath: str | Path) -> dict[str, Any]:
    return readJson(getArtifactPaths(mediaPath).multimodal, {}) or {}


def _defaultVisualAnalyzer(
    mediaPath: Path, yoloFaceDetector: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    from scripts.analyzeInterviewDemo import analyzeVideo, createEvents, loadMediaPipe
    from src.localModels.config import getModelSection

    analysisFps = float(getModelSection("faceDetection").get("analysisFps", 3))
    try:
        rows, duration = analyzeVideo(mediaPath, loadMediaPipe(), yoloFaceDetector, analysisFps=analysisFps)
    except SystemExit as error:
        raise RuntimeError(str(error)) from error
    return rows, createEvents(rows, duration), duration


def _saveVisualArtifacts(paths: Any, rows: list[dict[str, Any]], events: list[dict[str, Any]]) -> None:
    paths.visualFeatures.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty and not list(frame.columns):
        frame = pd.DataFrame(columns=["timestampSeconds", "frameLabels"])
    frame.to_csv(paths.visualFeatures, index=False)
    writeJson(paths.visualEvents, events)


def _saveUnavailableAudioArtifacts(
    paths: Any,
    fingerprint: dict[str, Any],
    recordingStem: str,
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if paths.audio.resolve() != paths.media.resolve():
        paths.audio.unlink(missing_ok=True)
    writeJson(
        paths.audioMeta,
        {
            "available": False,
            "cached": False,
            "sourceFingerprint": fingerprint,
            "warnings": warnings,
        },
    )
    transcript = {
        "analysisVersion": transcriptionVersion,
        "recordingIdentifier": recordingStem,
        "sourceFingerprint": fingerprint,
        "language": None,
        "durationSeconds": None,
        "text": "",
        "segments": [],
        "averageConfidence": None,
        "modelConfiguration": getTranscriptionConfig(),
        "warnings": warnings,
        "cached": False,
    }
    audioFeatures = {
        "analysisVersion": audioAnalysisVersion,
        "available": False,
        "sourceFingerprint": fingerprint,
        "warnings": warnings,
    }
    writeText(paths.transcriptText, "")
    writeJson(paths.transcriptJson, transcript)
    writeJson(paths.audioFeatures, audioFeatures)
    writeJson(paths.audioEvents, [])
    return transcript, audioFeatures


def _progress(callback: ProgressCallback | None, stage: str, fraction: float, message: str) -> None:
    if callback:
        callback(stage, fraction, message)


def _collectWarnings(*sources: dict[str, Any]) -> list[str]:
    output = []
    for source in sources:
        for warning in source.get("warnings", []) if isinstance(source, dict) else []:
            if warning not in output:
                output.append(warning)
    return output


def _buildLocalCoachingReport(coaching: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TalonCV Local Enhanced Coaching",
            "",
            coaching.get("text", ""),
            "",
            "## Safety note",
            "",
            coaching.get("safetyNote", "Generated locally from supplied structured evidence."),
            "",
        ]
    )
