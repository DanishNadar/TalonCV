import importlib.util
import shutil
from pathlib import Path
from typing import Any

from src.audioPipeline.mediaUtils import codecDiagnostics, inspectMedia
from src.audioPipeline.semanticAnalyzer import semanticDiagnostics
from src.audioPipeline.transcription import transcriptionDiagnostics
from src.cvPipeline.yoloFaceDetector import yoloDiagnostics
from src.localModels.config import loadModelConfig
from src.localModels.localCoach import localCoachDiagnostics
from src.multimodalPipeline.artifacts import demoRoot, getArtifactPaths, projectRoot


offlineStatuses = {
    "ready": "Ready for offline demo",
    "models": "Local models missing",
    "codecs": "Media codecs unavailable",
    "permissions": "Browser media permissions not yet tested",
    "folders": "Artifact folders unavailable",
    "disk": "Insufficient disk space",
}


def buildDemoDiagnostics(
    mediaPath: str | Path | None = None,
    loadModels: bool = False,
    browserPermissionsTested: bool = False,
) -> dict[str, Any]:
    requiredPackages = [
        "accelerate",
        "av",
        "cv2",
        "faster_whisper",
        "mediapipe",
        "numpy",
        "pandas",
        "streamlit",
        "streamlit_webrtc",
        "torch",
        "transformers",
        "ultralytics",
    ]
    packages = {name: importlib.util.find_spec(name) is not None for name in requiredPackages}
    config = loadModelConfig()
    models = {
        "transcription": transcriptionDiagnostics(loadModel=loadModels),
        "semanticAnalysis": semanticDiagnostics(loadModel=loadModels),
        "faceDetection": yoloDiagnostics(loadModel=loadModels),
        "localCoach": localCoachDiagnostics(loadModel=loadModels),
    }
    codecs = codecDiagnostics()
    folders = _folderDiagnostics()
    disk = shutil.disk_usage(projectRoot)
    minimumDiskGb = float(config.get("runtime", {}).get("minimumFreeDiskGb", 6))
    diskReady = disk.free >= minimumDiskGb * 1024**3
    media = inspectMedia(mediaPath) if mediaPath else None
    artifacts = {}
    if mediaPath:
        artifacts = {name: path.exists() for name, path in getArtifactPaths(mediaPath).asDict().items()}
    requiredModelsReady = all(
        model.get("requiredFilesPresent", False)
        for name, model in models.items()
        if name != "localCoach" or model.get("enabled", True)
    )
    requestedLoadsReady = not loadModels or all(
        model.get("loaded", False) and not model.get("loadError")
        for name, model in models.items()
        if name != "localCoach" or model.get("enabled", True)
    )
    statusKey = "ready"
    if not folders["ready"]:
        statusKey = "folders"
    elif not diskReady:
        statusKey = "disk"
    elif not codecs["ready"]:
        statusKey = "codecs"
    elif not requiredModelsReady or not requestedLoadsReady or not all(packages.values()):
        statusKey = "models"
    elif not browserPermissionsTested:
        statusKey = "permissions"
    return {
        "offlineStatus": offlineStatuses[statusKey],
        "offlineReady": statusKey == "ready",
        "requiredPackages": packages,
        "requiredPackagesReady": all(packages.values()),
        "codecs": codecs,
        "artifactFolders": folders,
        "disk": {
            "freeGb": round(disk.free / (1024**3), 2),
            "minimumFreeGb": minimumDiskGb,
            "ready": diskReady,
        },
        "models": models,
        "modelLoadChecksRequested": loadModels,
        "modelConfiguration": {
            "path": config["configPath"],
            "allPathsLocal": True,
            "runtimeNetworkingDisabled": config["runtime"]["networkingDisabled"],
        },
        "runtimePolicy": {
            "externalInferenceApis": "disabled",
            "runtimeModelDownloads": "disabled",
            "localModelLoading": "required",
            "offlineOperationAfterSetup": "supported",
            "externalCredentialConfigurationPresent": False,
            "publicIceInfrastructureConfigured": False,
        },
        "browserPermissions": {
            "testedThisSession": browserPermissionsTested,
            "note": "Record a short camera/microphone take to complete this readiness check.",
        },
        "selectedMedia": media,
        "artifacts": artifacts,
    }


def _folderDiagnostics() -> dict[str, Any]:
    folders = [
        demoRoot / name
        for name in (
            "recordings",
            "audio",
            "sessions",
            "transcripts",
            "audioFeatures",
            "audioEvents",
            "responseAnalysis",
            "semanticAnalysis",
            "features",
            "events",
            "multimodal",
            "scores",
            "diagnostics",
            "llmReady",
        )
    ] + [projectRoot / "reports"]
    result = {}
    for folder in folders:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            probe = folder / ".taloncv_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            result[str(folder.relative_to(projectRoot))] = True
        except OSError:
            result[str(folder)] = False
    return {"paths": result, "ready": all(result.values())}
