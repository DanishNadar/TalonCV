import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.database import SupabaseJobDatabase
from backend.jobProcessor import JobCancelled, JobProcessor
from backend.security import (
    MediaValidationError,
    artifactStoragePath,
    recordingStoragePath,
    validateMediaFile,
    validateStoragePath,
)
from backend.settings import getSettings
from src.localModels.cpuRuntime import getCpuRuntimeConfiguration
from src.localModels.llamaCppCoach import generateLlamaCppText, loadLlamaCppCoach


projectRoot = Path(__file__).resolve().parents[1]
USER = "11111111-1111-4111-8111-111111111111"
SESSION = "22222222-2222-4222-8222-222222222222"
RECORDING = "33333333-3333-4333-8333-333333333333"
JOB = "44444444-4444-4444-8444-444444444444"


class FakeDatabase:
    def __init__(self, cancel=False):
        self.cancel = cancel
        self.progress = []
        self.completed = None

    def loadJobContext(self, job):
        return {
            "job": job,
            "recording": {"id": RECORDING, "user_id": USER, "storage_path": recordingStoragePath(USER, SESSION, RECORDING)},
            "session": {"id": SESSION, "user_id": USER, "interview_question": "Tell me about a project."},
        }

    def updateProgress(self, jobId, stage, progress, workerId):
        self.progress.append((jobId, stage, progress, workerId))

    def cancellationRequested(self, _jobId):
        return self.cancel

    def completeJob(self, jobId, result, artifacts):
        self.completed = (jobId, result, artifacts)


class FakeStorage:
    def __init__(self):
        self.uploaded = []

    def downloadRecording(self, _objectPath, destination):
        destination.write_bytes(b"fixture")

    def uploadArtifact(self, objectPath, source, contentType):
        self.uploaded.append((objectPath, source.name, contentType))
        return {"storage_path": objectPath, "content_type": contentType, "size_bytes": source.stat().st_size, "sha256": "a" * 64}


class Response:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, data=None):
        self.data = data
        self.values = None

    def select(self, *_args, **_kwargs): return self
    def update(self, values): self.values = values; return self
    def eq(self, *_args): return self
    def in_(self, *_args): return self
    def is_(self, *_args): return self
    def limit(self, *_args): return self
    def single(self): return self
    def execute(self): return Response(self.data)


class BackendProductionTests(unittest.TestCase):
    def test_fastapi_health_readiness_and_version_are_safe(self):
        from fastapi.testclient import TestClient
        from backend.api import app

        client = TestClient(app)
        self.assertEqual({"status": "ok"}, client.get("/health").json())
        readiness = client.get("/ready").json()
        for key in ("database", "storage", "modelPaths", "temporaryDisk", "worker", "cpuOnly"):
            self.assertIn(key, readiness)
        serialized = json.dumps(readiness).lower()
        self.assertNotIn("service_role", serialized)
        self.assertNotIn("transcript", serialized)
        version = client.get("/version").json()
        self.assertEqual("multimodal-v4", version["analysisVersion"])

    def test_storage_paths_are_user_scoped_and_reject_traversal(self):
        recording = recordingStoragePath(USER, SESSION, RECORDING)
        artifact = artifactStoragePath(USER, SESSION, JOB, "transcript.json")
        self.assertTrue(recording.startswith(f"users/{USER}/sessions/{SESSION}/"))
        self.assertTrue(artifact.endswith("/transcript.json"))
        self.assertEqual(recording, validateStoragePath(recording, USER))
        for value in ("../secret", "/absolute/file", f"users/{USER}/../other"):
            with self.assertRaises(ValueError):
                validateStoragePath(value, USER)

    def test_media_validation_rejects_missing_corrupt_oversize_and_excessive_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording.webm"
            with self.assertRaisesRegex(MediaValidationError, "missing or empty"):
                validateMediaFile(path, 100, 300, 4096)
            path.write_bytes(b"x" * 101)
            with self.assertRaisesRegex(MediaValidationError, "250 MB"):
                validateMediaFile(path, 100, 300, 4096)
            path.write_bytes(b"fixture")
            with patch("backend.security.inspectMedia", return_value={"valid": False}):
                with self.assertRaisesRegex(MediaValidationError, "could not be decoded"):
                    validateMediaFile(path, 100, 300, 4096)
            with patch("backend.security.inspectMedia", return_value={"valid": True, "durationSeconds": 301, "video": {}}):
                with self.assertRaisesRegex(MediaValidationError, "five-minute"):
                    validateMediaFile(path, 100, 300, 4096)

    def test_job_processor_uploads_results_and_always_cleans_temporary_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = replace(getSettings(), tempRoot=Path(folder))
            database = FakeDatabase()
            storage = FakeStorage()

            def pipeline(mediaPath, **kwargs):
                kwargs["progressCallback"]("transcribeLocally", 0.3, "ignored")
                root = Path(kwargs["artifactRoot"])
                transcript = root / "transcripts" / "recording_transcript.json"
                analysis = root / "multimodal" / "recording_multimodal.json"
                transcript.parent.mkdir(parents=True, exist_ok=True)
                analysis.parent.mkdir(parents=True, exist_ok=True)
                transcript.write_text("{}", encoding="utf-8")
                analysis.write_text("{}", encoding="utf-8")
                return {
                    "analysisVersion": "multimodal-v4",
                    "mediaInfo": {"path": str(mediaPath), "durationSeconds": 1},
                    "transcript": {"text": "A concise answer."},
                    "audioFeatures": {}, "responseAnalysis": {}, "scores": {}, "warnings": [],
                    "provenance": {"models": {}},
                    "artifactPaths": {"transcriptJson": str(transcript), "multimodal": str(analysis)},
                }

            processor = JobProcessor(settings, database, storage, "worker-test")
            with patch("backend.jobProcessor.validateMediaFile", return_value={"valid": True, "durationSeconds": 1}), patch(
                "src.multimodalPipeline.pipeline.runMultimodalAnalysis", side_effect=pipeline
            ):
                processor.process({"id": JOB, "recording_id": RECORDING, "session_id": SESSION, "user_id": USER})
            self.assertIsNotNone(database.completed)
            self.assertEqual(2, len(storage.uploaded))
            self.assertFalse((Path(folder) / JOB).exists())
            result = database.completed[1]
            self.assertIn("totalAnalysisSeconds", result["stage_durations"])

    def test_job_processor_observes_cancellation_between_pipeline_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            database = FakeDatabase(cancel=True)
            processor = JobProcessor(replace(getSettings(), tempRoot=Path(folder)), database, FakeStorage(), "worker-test")

            def pipeline(_mediaPath, **kwargs):
                kwargs["progressCallback"]("extractAudio", 0.1, "ignored")

            with patch("backend.jobProcessor.validateMediaFile", return_value={"valid": True, "durationSeconds": 1}), patch(
                "src.multimodalPipeline.pipeline.runMultimodalAnalysis", side_effect=pipeline
            ):
                with self.assertRaises(JobCancelled):
                    processor.process({"id": JOB, "recording_id": RECORDING, "session_id": SESSION, "user_id": USER})
            self.assertFalse((Path(folder) / JOB).exists())

    def test_database_claim_recovery_retry_and_progress_use_atomic_rpcs(self):
        client = MagicMock()
        client.rpc.side_effect = lambda name, args: Query([{"id": JOB}] if name == "claim_analysis_job" else 2)
        database = SupabaseJobDatabase.__new__(SupabaseJobDatabase)
        database.client = client
        self.assertEqual(JOB, database.claimJob("worker")["id"])
        self.assertEqual(2, database.recoverStaleJobs(20))
        database.failJob(JOB, "analysis_failed", "safe message", True)
        database.acknowledgeCancellation(JOB, "worker")
        names = [call.args[0] for call in client.rpc.call_args_list]
        self.assertEqual(
            [
                "claim_analysis_job",
                "recover_stale_analysis_jobs",
                "fail_analysis_job",
                "acknowledge_analysis_job_cancellation",
            ],
            names,
        )

    def test_progress_updates_cannot_revive_a_cancelled_job(self):
        query = Query()
        client = MagicMock()
        client.table.return_value = query
        database = SupabaseJobDatabase.__new__(SupabaseJobDatabase)
        database.client = client
        database.updateProgress(JOB, "transcribeLocally", 30, "worker")
        self.assertEqual("processing", query.values["status"])
        source = Path(SupabaseJobDatabase.updateProgress.__code__.co_filename).read_text(encoding="utf-8")
        self.assertIn('.in_("status", ["claimed", "processing"])', source)
        self.assertIn('"cancellation_requested_at", "null"', source)

    def test_production_migration_contains_claim_lock_rls_limits_and_recovery(self):
        source = (projectRoot / "supabase" / "migrations" / "202608070001_production_schema.sql").read_text(encoding="utf-8").lower()
        for required in (
            "for update skip locked", "one_active_job_per_user", "recover_stale_analysis_jobs",
            "enable row level security", "auth.uid()", "cancel_analysis_job", "taloncv-recordings",
            "taloncv-artifacts", "guest_daily_limit",
            "cancellation_requested_at", "acknowledge_analysis_job_cancellation",
            "exception when unique_violation", "grant select, insert on table public.recordings",
        ):
            self.assertIn(required, source)

    def test_public_web_has_no_service_role_or_supabase_dependency(self):
        package = json.loads((projectRoot / "web" / "package.json").read_text(encoding="utf-8"))
        dependencies = {**package.get("dependencies", {}), **package.get("devDependencies", {})}
        self.assertFalse(any("supabase" in name.lower() for name in dependencies))
        paths = [*(projectRoot / "web").rglob("*.ts"), *(projectRoot / "web").rglob("*.tsx")]
        for path in paths:
            if any(part in {"node_modules", ".next"} for part in path.parts):
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("SUPABASE_SERVICE_ROLE_KEY", source, str(path))
            self.assertNotIn("@supabase/", source, str(path))

    def test_public_deployment_has_no_railway_configuration(self):
        self.assertFalse((projectRoot / "railway.toml").exists())
        self.assertFalse((projectRoot / "Dockerfile").exists())

    def test_worker_keeps_readiness_heartbeat_fresh_during_long_inference(self):
        source = (projectRoot / "backend" / "worker.py").read_text(encoding="utf-8")
        self.assertIn('heartbeatStop.wait(20)', source)
        self.assertIn('writeWorkerState("processing", workerId=identifier, jobId=jobId)', source)

    def test_generated_artifact_uploads_are_idempotent_across_retries(self):
        storageSource = (projectRoot / "backend" / "storage.py").read_text(encoding="utf-8")
        self.assertIn('"upsert": "true"', storageSource)
        migration = (projectRoot / "supabase" / "migrations" / "202608070001_production_schema.sql").read_text(encoding="utf-8")
        self.assertGreaterEqual(migration.count("where id=p_job_id and status in ('claimed','processing');"), 2)

    def test_public_next_app_is_static_and_secret_free(self):
        next_config = (projectRoot / "web" / "next.config.ts").read_text(encoding="utf-8")
        environment = (projectRoot / "web" / ".env.example").read_text(encoding="utf-8")
        self.assertIn('output: "export"', next_config)
        self.assertNotIn("SUPABASE", environment)
        self.assertNotIn("SERVICE_ROLE", environment)

    def test_cpu_runtime_overrides_native_thread_pool_defaults(self):
        source = (projectRoot / "src" / "localModels" / "cpuRuntime.py").read_text(encoding="utf-8")
        self.assertIn('os.environ[name] = value', source)
        entrypoint = (projectRoot / "backend" / "entrypoint.sh").read_text(encoding="utf-8")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertIn(f'export {name}="$TALONCV_CPU_THREADS"', entrypoint)

    def test_llama_cpp_loader_is_local_cpu_bounded_cached_and_memory_mapped(self):
        constructor = MagicMock()
        fakeModule = types.SimpleNamespace(Llama=constructor)
        loadLlamaCppCoach.cache_clear()
        with patch.dict(sys.modules, {"llama_cpp": fakeModule}), patch(
            "src.localModels.llamaCppCoach.validateModelFiles", return_value={"requiredFilesPresent": True}
        ):
            first = loadLlamaCppCoach("C:/models/coach.gguf", 2048, 3)
            second = loadLlamaCppCoach("C:/models/coach.gguf", 2048, 3)
        self.assertIs(first, second)
        self.assertEqual(1, constructor.call_count)
        options = constructor.call_args.kwargs
        self.assertEqual(0, options["n_gpu_layers"])
        self.assertEqual(3, options["n_threads"])
        self.assertTrue(options["use_mmap"])

    def test_llama_cpp_generation_remains_evidence_constrained(self):
        model = MagicMock()
        model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Use only the supplied evidence."}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }
        with patch("src.localModels.llamaCppCoach.configureCpuRuntime", return_value=getCpuRuntimeConfiguration()), patch(
            "src.localModels.llamaCppCoach.loadLlamaCppCoach", return_value=model
        ):
            text, metadata = generateLlamaCppText(
                [{"role": "system", "content": "Use only evidence."}],
                {"resolvedPath": "C:/models/coach.gguf", "threads": 2, "contextSize": 2048, "temperature": 0.1},
                64,
            )
        self.assertIn("evidence", text)
        self.assertEqual(0, metadata["nGpuLayers"])
        self.assertLessEqual(metadata["threads"], getCpuRuntimeConfiguration().availableCpus)


if __name__ == "__main__":
    unittest.main()
