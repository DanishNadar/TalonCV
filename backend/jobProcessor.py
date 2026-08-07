import copy
import mimetypes
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from backend.database import JobDatabase
from backend.security import (
    MediaValidationError,
    artifactStoragePath,
    canonicalUuid,
    recordingStoragePath,
    validateMediaFile,
    validateStoragePath,
)
from backend.settings import Settings
from backend.storage import ObjectStorage


class JobCancelled(RuntimeError):
    pass


class WorkerShutdown(RuntimeError):
    pass


artifactTypes = {
    "session": "session_metadata.json",
    "audio": "extracted_audio.wav",
    "audioMeta": "audio_metadata.json",
    "transcriptText": "transcript.txt",
    "transcriptJson": "transcript.json",
    "audioFeatures": "audio_features.json",
    "audioEvents": "audio_events.json",
    "responseAnalysis": "response_analysis.json",
    "semanticAnalysis": "semantic_analysis.json",
    "visualEvents": "visual_events.json",
    "visualFeatures": "visual_features.csv",
    "multimodalMoments": "multimodal_moments.json",
    "scores": "scores.json",
    "multimodal": "analysis.json",
    "deterministicReport": "report.md",
    "report": "enhanced_report.md",
    "localCoaching": "local_coaching.md",
    "localCoachMeta": "local_coaching_metadata.json",
}


class JobProcessor:
    def __init__(
        self,
        settings: Settings,
        database: JobDatabase,
        storage: ObjectStorage,
        workerId: str,
        shouldStop: Callable[[], bool] | None = None,
    ):
        self.settings = settings
        self.database = database
        self.storage = storage
        self.workerId = workerId
        self.shouldStop = shouldStop or (lambda: False)

    def _ensureActive(self, jobId: str) -> None:
        if self.shouldStop():
            raise WorkerShutdown("The worker is shutting down.")
        if self.database.cancellationRequested(jobId):
            raise JobCancelled("Analysis was cancelled by the user.")

    def process(self, claimedJob: dict[str, Any]) -> None:
        jobId = canonicalUuid(claimedJob["id"], "job id")
        workspace = (self.settings.tempRoot / jobId).resolve()
        if workspace.parent != self.settings.tempRoot.resolve():
            raise ValueError("Unsafe temporary workspace.")
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        stageStarted = started
        currentStage = "preparing_media"
        stageDurations: dict[str, float] = {}
        try:
            if shutil.disk_usage(workspace).free < self.settings.minimumFreeDiskGb * 1024**3:
                raise MediaValidationError("insufficient_disk", "The analysis worker needs more temporary disk space.")
            context = self.database.loadJobContext(claimedJob)
            recording = context["recording"]
            session = context["session"]
            userId = canonicalUuid(claimedJob["user_id"], "user id")
            if recording.get("user_id") != userId or session.get("user_id") != userId:
                raise ValueError("Job ownership metadata is inconsistent.")
            objectPath = validateStoragePath(recording["storage_path"], userId)
            suffix = Path(objectPath).suffix.lower()
            expectedPath = recordingStoragePath(userId, session["id"], recording["id"], suffix)
            if objectPath != expectedPath:
                raise ValueError("Recording storage metadata does not match its owner and identifiers.")
            mediaPath = workspace / f"recording{suffix}"
            self._ensureActive(jobId)
            self.database.updateProgress(jobId, currentStage, 3, self.workerId)
            self.storage.downloadRecording(objectPath, mediaPath)
            self._ensureActive(jobId)
            self.database.updateProgress(jobId, "validating_recording", 6, self.workerId)
            mediaInfo = validateMediaFile(
                mediaPath,
                self.settings.maxUploadBytes,
                self.settings.maxDurationSeconds,
                self.settings.maxVideoDimension,
            )

            def progress(stage: str, fraction: float, _message: str) -> None:
                nonlocal currentStage, stageStarted
                now = time.perf_counter()
                if stage != currentStage:
                    stageDurations[currentStage] = round(stageDurations.get(currentStage, 0) + now - stageStarted, 4)
                    currentStage = stage
                    stageStarted = now
                self._ensureActive(jobId)
                self.database.updateProgress(jobId, stage, max(7, min(98, round(fraction * 100))), self.workerId)

            from src.multimodalPipeline.pipeline import runMultimodalAnalysis

            result = runMultimodalAnalysis(
                mediaPath,
                sessionContext={
                    "interviewQuestion": session.get("interview_question", ""),
                    "targetRole": session.get("target_role", ""),
                    "jobDescription": session.get("job_description", ""),
                    "desiredCompetencies": session.get("desired_competencies", ""),
                },
                force=True,
                progressCallback=progress,
                artifactRoot=workspace / "artifacts",
            )
            stageDurations[currentStage] = round(
                stageDurations.get(currentStage, 0) + time.perf_counter() - stageStarted, 4
            )
            stageDurations["totalAnalysisSeconds"] = round(time.perf_counter() - started, 4)
            resultForStorage = _redactLocalPaths(result)
            resultForStorage.setdefault("provenance", {})["stageDurations"] = stageDurations
            resultForStorage["artifactPaths"] = {
                artifactType: artifactStoragePath(userId, session["id"], jobId, filename)
                for artifactType, filename in artifactTypes.items()
                if result.get("artifactPaths", {}).get(artifactType)
                and Path(result["artifactPaths"][artifactType]).is_file()
            }
            from src.multimodalPipeline.artifacts import writeJson

            writeJson(result["artifactPaths"]["multimodal"], resultForStorage)
            # Cancellation is observed up to this safe persistence boundary. Once
            # uploads begin, the worker completes them to avoid orphaned objects.
            self._ensureActive(jobId)
            uploaded = self._uploadArtifacts(result, userId, session["id"], jobId)
            resultForStorage["artifactPaths"] = {item["artifact_type"]: item["storage_path"] for item in uploaded}
            self.database.completeJob(
                jobId,
                _databaseResult(resultForStorage, mediaInfo, stageDurations),
                uploaded,
            )
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _uploadArtifacts(
        self,
        result: dict[str, Any],
        userId: str,
        sessionId: str,
        jobId: str,
    ) -> list[dict[str, Any]]:
        uploaded = []
        localPaths = result.get("artifactPaths", {})
        for artifactType, filename in artifactTypes.items():
            sourceValue = localPaths.get(artifactType)
            if not sourceValue:
                continue
            source = Path(sourceValue)
            if not source.is_file() or source.stat().st_size == 0:
                continue
            objectPath = artifactStoragePath(userId, sessionId, jobId, filename)
            contentType = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            metadata = self.storage.uploadArtifact(objectPath, source, contentType)
            uploaded.append({"artifact_type": artifactType, **metadata})
        return uploaded


def _databaseResult(result: dict[str, Any], mediaInfo: dict[str, Any], durations: dict[str, float]) -> dict[str, Any]:
    transcript = result.get("transcript", {})
    audio = result.get("audioFeatures", {})
    scores = result.get("scores", {})
    words = str(transcript.get("text") or "").split()
    return {
        "scores": scores,
        "coverage": scores.get("coverage", result.get("mediaInfo", {})),
        "summary": {
            "recordingDurationSeconds": mediaInfo.get("durationSeconds"),
            "speakingTimeSeconds": audio.get("speakingDurationSeconds"),
            "wordCount": len(words),
            "wordsPerMinute": audio.get("speechRateWpm"),
            "fillerCount": result.get("responseAnalysis", {}).get("fillerCount"),
            "longPauseCount": audio.get("longPauseCount"),
            "transcriptConfidence": transcript.get("averageConfidence"),
            "enhancedCoachingStatus": result.get("enhancedCoachingStatus"),
            "momentCount": len(result.get("moments", [])),
            "analyzedFrameCount": result.get("performanceMetrics", {}).get("analyzedFrameCount"),
        },
        "model_versions": result.get("provenance", {}).get("models", {}),
        "stage_durations": durations,
        "warnings": result.get("warnings", []),
        "analysis_version": result.get("analysisVersion", "multimodal-v4"),
    }


def _redactLocalPaths(value: Any) -> Any:
    if isinstance(value, list):
        return [_redactLocalPaths(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {}
    for key, item in value.items():
        if key in {"path", "sourcePath", "wavPath", "modelPath", "resolvedPath", "configPath"} and isinstance(item, str):
            output[key] = Path(item).name
        else:
            output[key] = _redactLocalPaths(item)
    return output
