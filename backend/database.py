from datetime import UTC, datetime
from typing import Any, Protocol

from backend.settings import Settings


class JobDatabase(Protocol):
    def ping(self) -> bool: ...
    def recoverStaleJobs(self, staleMinutes: int) -> int: ...
    def claimJob(self, workerId: str) -> dict[str, Any] | None: ...
    def loadJobContext(self, job: dict[str, Any]) -> dict[str, Any]: ...
    def updateProgress(self, jobId: str, stage: str, progress: int, workerId: str) -> None: ...
    def cancellationRequested(self, jobId: str) -> bool: ...
    def acknowledgeCancellation(self, jobId: str, workerId: str) -> None: ...
    def completeJob(self, jobId: str, result: dict[str, Any], artifacts: list[dict[str, Any]]) -> None: ...
    def failJob(self, jobId: str, errorCode: str, errorMessage: str, recoverable: bool) -> None: ...
    def releaseJob(self, jobId: str, workerId: str) -> None: ...


class SupabaseJobDatabase:
    def __init__(self, settings: Settings):
        if not settings.supabaseConfigured:
            raise RuntimeError("Supabase service configuration is missing.")
        from supabase import Client, create_client

        self.client: Client = create_client(settings.supabaseUrl, settings.supabaseServiceRoleKey)

    @staticmethod
    def _data(response: Any) -> Any:
        return getattr(response, "data", response)

    def ping(self) -> bool:
        self.client.table("analysis_jobs").select("id", count="exact").limit(1).execute()
        return True

    def recoverStaleJobs(self, staleMinutes: int) -> int:
        response = self.client.rpc("recover_stale_analysis_jobs", {"p_stale_minutes": staleMinutes}).execute()
        data = self._data(response)
        return int(data or 0)

    def claimJob(self, workerId: str) -> dict[str, Any] | None:
        response = self.client.rpc("claim_analysis_job", {"p_worker_id": workerId}).execute()
        data = self._data(response)
        if isinstance(data, list):
            return dict(data[0]) if data else None
        return dict(data) if data else None

    def loadJobContext(self, job: dict[str, Any]) -> dict[str, Any]:
        recording = self._data(
            self.client.table("recordings").select("*").eq("id", job["recording_id"]).single().execute()
        )
        session = self._data(
            self.client.table("interview_sessions").select("*").eq("id", job["session_id"]).single().execute()
        )
        return {"job": job, "recording": dict(recording), "session": dict(session)}

    def updateProgress(self, jobId: str, stage: str, progress: int, workerId: str) -> None:
        self.client.table("analysis_jobs").update(
            {
                "status": "processing",
                "stage": stage,
                "progress": max(0, min(int(progress), 99)),
                "worker_id": workerId,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        ).eq("id", jobId).eq("worker_id", workerId).in_("status", ["claimed", "processing"]).is_(
            "cancellation_requested_at", "null"
        ).execute()

    def cancellationRequested(self, jobId: str) -> bool:
        data = self._data(
            self.client.table("analysis_jobs").select("status,cancellation_requested_at").eq("id", jobId).single().execute()
        )
        return bool(data and (data.get("status") == "cancelled" or data.get("cancellation_requested_at")))

    def acknowledgeCancellation(self, jobId: str, workerId: str) -> None:
        self.client.rpc(
            "acknowledge_analysis_job_cancellation",
            {"p_job_id": jobId, "p_worker_id": workerId},
        ).execute()

    def completeJob(self, jobId: str, result: dict[str, Any], artifacts: list[dict[str, Any]]) -> None:
        self.client.rpc(
            "complete_analysis_job",
            {"p_job_id": jobId, "p_result": result, "p_artifacts": artifacts},
        ).execute()

    def failJob(self, jobId: str, errorCode: str, errorMessage: str, recoverable: bool) -> None:
        self.client.rpc(
            "fail_analysis_job",
            {
                "p_job_id": jobId,
                "p_error_code": errorCode,
                "p_error_message": errorMessage[:500],
                "p_recoverable": recoverable,
            },
        ).execute()

    def releaseJob(self, jobId: str, workerId: str) -> None:
        self.client.rpc("release_analysis_job", {"p_job_id": jobId, "p_worker_id": workerId}).execute()
