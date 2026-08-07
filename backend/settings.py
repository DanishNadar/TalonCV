import os
from dataclasses import dataclass
from pathlib import Path


def _integer(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    environment: str
    supabaseUrl: str
    supabaseServiceRoleKey: str
    modelConfig: Path
    tempRoot: Path
    recordingsBucket: str
    artifactsBucket: str
    maxUploadBytes: int
    maxDurationSeconds: int
    maxVideoDimension: int
    minimumFreeDiskGb: int
    workerPollSeconds: int
    staleJobMinutes: int
    maxAttempts: int
    port: int

    @property
    def supabaseConfigured(self) -> bool:
        return bool(self.supabaseUrl and self.supabaseServiceRoleKey)


def getSettings() -> Settings:
    projectRoot = Path(__file__).resolve().parents[1]
    modelConfig = Path(
        os.environ.get("TALONCV_MODEL_CONFIG", projectRoot / "config" / "models.production.json")
    ).expanduser()
    if not modelConfig.is_absolute():
        modelConfig = projectRoot / modelConfig
    return Settings(
        environment=os.environ.get("TALONCV_ENV", "production"),
        supabaseUrl=os.environ.get("SUPABASE_URL", "").rstrip("/"),
        supabaseServiceRoleKey=os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
        modelConfig=modelConfig.resolve(),
        tempRoot=Path(os.environ.get("TALONCV_TEMP_ROOT", "/tmp/taloncv")).resolve(),
        recordingsBucket=os.environ.get("TALONCV_RECORDINGS_BUCKET", "taloncv-recordings"),
        artifactsBucket=os.environ.get("TALONCV_ARTIFACTS_BUCKET", "taloncv-artifacts"),
        maxUploadBytes=_integer("TALONCV_MAX_UPLOAD_BYTES", 250 * 1024 * 1024),
        maxDurationSeconds=_integer("TALONCV_MAX_DURATION_SECONDS", 300),
        maxVideoDimension=_integer("TALONCV_MAX_VIDEO_DIMENSION", 4096),
        minimumFreeDiskGb=_integer("TALONCV_MIN_FREE_DISK_GB", 2),
        workerPollSeconds=_integer("TALONCV_WORKER_POLL_SECONDS", 2),
        staleJobMinutes=_integer("TALONCV_STALE_JOB_MINUTES", 20),
        maxAttempts=_integer("TALONCV_MAX_ATTEMPTS", 3),
        port=_integer("PORT", 8000),
    )
