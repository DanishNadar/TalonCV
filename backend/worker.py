import hashlib
import logging
import os
import signal
import socket
import threading
import time
import uuid

from backend.settings import getSettings


# Configure native thread pools before importing media/model modules that load
# NumPy, OpenBLAS, PyTorch, CTranslate2, or tokenizers.
startupSettings = getSettings()
os.environ.setdefault("TALONCV_ENV", startupSettings.environment)
os.environ.setdefault("TALONCV_MODEL_CONFIG", str(startupSettings.modelConfig))
from src.localModels.config import enforceOfflineRuntime
from src.localModels.cpuRuntime import configureCpuRuntime

enforceOfflineRuntime()
startupCpu = configureCpuRuntime()

from backend.database import SupabaseJobDatabase
from backend.jobProcessor import JobCancelled, JobProcessor, WorkerShutdown
from backend.runtimeState import writeWorkerState
from backend.security import MediaValidationError
from backend.storage import SupabaseObjectStorage


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("taloncv.worker")
shutdownRequested = False


def _shutdown(_signum, _frame) -> None:
    global shutdownRequested
    shutdownRequested = True


def workerId() -> str:
    stableHost = hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()[:10]
    return f"{stableHost}-{uuid.uuid4().hex[:8]}"


def run() -> None:
    settings = getSettings()
    cpu = startupCpu
    database = SupabaseJobDatabase(settings)
    storage = SupabaseObjectStorage(settings)
    identifier = workerId()
    processor = JobProcessor(settings, database, storage, identifier, shouldStop=lambda: shutdownRequested)
    logger.info("worker_started worker_id=%s cpu_threads=%s", identifier, cpu.workerThreads)
    writeWorkerState("idle", workerId=identifier, cpuThreads=cpu.workerThreads)
    idle = max(1, settings.workerPollSeconds)
    lastRecovery = 0.0
    activeJob = None
    while not shutdownRequested:
        try:
            if time.monotonic() - lastRecovery > 60:
                recovered = database.recoverStaleJobs(settings.staleJobMinutes)
                if recovered:
                    logger.warning("stale_jobs_recovered count=%s", recovered)
                lastRecovery = time.monotonic()
            activeJob = database.claimJob(identifier)
            if not activeJob:
                writeWorkerState("idle", workerId=identifier)
                time.sleep(idle)
                idle = min(idle * 2, 15)
                continue
            idle = max(1, settings.workerPollSeconds)
            jobId = activeJob["id"]
            writeWorkerState("processing", workerId=identifier, jobId=jobId)
            logger.info("job_started job_id=%s", jobId)
            heartbeatStop = threading.Event()

            def heartbeat() -> None:
                while not heartbeatStop.wait(20):
                    writeWorkerState("processing", workerId=identifier, jobId=jobId)

            heartbeatThread = threading.Thread(target=heartbeat, name="taloncv-worker-heartbeat", daemon=True)
            heartbeatThread.start()
            try:
                processor.process(activeJob)
                logger.info("job_completed job_id=%s", jobId)
            except JobCancelled:
                database.acknowledgeCancellation(jobId, identifier)
                logger.info("job_cancelled job_id=%s", jobId)
            except WorkerShutdown:
                database.releaseJob(jobId, identifier)
                logger.info("job_released_for_shutdown job_id=%s", jobId)
            except MediaValidationError as error:
                database.failJob(jobId, error.code, str(error), recoverable=False)
                logger.warning("job_rejected job_id=%s error_code=%s", jobId, error.code)
            except Exception as error:
                database.failJob(
                    jobId,
                    "analysis_failed",
                    "Analysis encountered a recoverable backend error. TalonCV will retry automatically.",
                    recoverable=True,
                )
                logger.exception("job_failed job_id=%s error_code=analysis_failed", jobId)
            finally:
                heartbeatStop.set()
                heartbeatThread.join(timeout=2)
            activeJob = None
        except Exception:
            logger.exception("worker_loop_error")
            writeWorkerState("degraded", workerId=identifier)
            time.sleep(min(idle, 15))
    if activeJob:
        database.releaseJob(activeJob["id"], identifier)
    writeWorkerState("stopped", workerId=identifier)
    logger.info("worker_stopped worker_id=%s", identifier)


def main() -> None:
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    run()


if __name__ == "__main__":
    main()
