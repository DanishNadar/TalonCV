import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from backend import __version__
from backend.database import SupabaseJobDatabase
from backend.runtimeState import readWorkerState
from backend.settings import getSettings
from backend.storage import SupabaseObjectStorage


app = FastAPI(title="TalonCV CPU Backend", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _check(callback) -> bool:
    try:
        return bool(callback())
    except Exception:
        return False


def readiness() -> dict[str, Any]:
    settings = getSettings()
    os.environ.setdefault("TALONCV_MODEL_CONFIG", str(settings.modelConfig))
    databaseReady = False
    storageReady = False
    if settings.supabaseConfigured:
        databaseReady = _check(lambda: SupabaseJobDatabase(settings).ping())
        storageReady = _check(lambda: SupabaseObjectStorage(settings).ping())
    try:
        from src.localModels.config import loadModelConfig, validateModelFiles

        config = loadModelConfig(str(settings.modelConfig))
        modelPathsReady = all(
            validateModelFiles(name, config[name])["requiredFilesPresent"]
            for name in ("transcription", "faceDetection", "semanticAnalysis", "localCoach")
        )
        modelConfigValid = True
    except Exception:
        modelPathsReady = False
        modelConfigValid = False
    try:
        settings.tempRoot.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.tempRoot, delete=True):
            pass
        temporaryWritable = shutil.disk_usage(settings.tempRoot).free >= settings.minimumFreeDiskGb * 1024**3
    except OSError:
        temporaryWritable = False
    worker = readWorkerState()
    workerUpdated = worker.get("updatedAt")
    workerReady = worker.get("status") in {"idle", "processing"}
    if workerUpdated:
        try:
            age = (datetime.now(UTC) - datetime.fromisoformat(workerUpdated)).total_seconds()
            workerReady = workerReady and age < 90
        except ValueError:
            workerReady = False
    return {
        "database": databaseReady,
        "storage": storageReady,
        "modelPaths": modelPathsReady,
        "modelConfiguration": modelConfigValid,
        "temporaryDisk": temporaryWritable,
        "worker": workerReady,
        "workerStatus": worker.get("status", "unknown"),
        "cpuOnly": True,
    }


@app.get("/ready")
def ready() -> dict[str, Any]:
    return readiness()


@app.get("/ready/deep")
def deepReady(x_taloncv_admin_token: str | None = Header(default=None)) -> dict[str, Any]:
    expected = os.environ.get("TALONCV_ADMIN_HEALTH_TOKEN")
    if not expected or x_taloncv_admin_token != expected:
        raise HTTPException(status_code=404, detail="Not found")
    result = readiness()
    result["deepModelLoad"] = "Run scripts/verifyProductionModels.py inside the worker container."
    return result


@app.get("/version")
def version() -> dict[str, str]:
    return {
        "version": __version__,
        "analysisVersion": "multimodal-v4",
        "gitCommit": os.environ.get("RAILWAY_GIT_COMMIT_SHA", "unknown")[:12],
    }
