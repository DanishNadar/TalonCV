import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.audioPipeline.semanticAnalyzer import SemanticAnalysisError, _loadSemanticModel
from src.audioPipeline.transcription import TranscriptionError, _loadModel
from src.cvPipeline.webcamRecorder import RecordingState, WebcamRecorder
from src.cvPipeline.yoloFaceDetector import resolve_yolo_face_model
from src.localModels.config import loadModelConfig, resolveLocalPath
from src.localModels.localCoach import (
    LocalCoachError,
    _loadLocalCoach,
    sanitizeUnsupportedTimestamps,
)


projectRoot = Path(__file__).resolve().parents[1]


class LocalOnlyArchitectureTests(unittest.TestCase):
    def test_repository_has_no_prohibited_runtime_paths(self):
        prohibited = [
            "open" + "ai",
            "HF" + "_TOKEN",
            "HF" + "_MODEL",
            "router." + "huggingface",
            "stun.l." + "google",
            "hf_hub" + "_download",
            "snapshot" + "_download",
            "list_repo" + "_files",
        ]
        candidates = [projectRoot / "app.py", projectRoot / ".env.example", projectRoot / "requirements.txt"]
        candidates.extend((projectRoot / "src").rglob("*.py"))
        candidates.extend((projectRoot / "backend").rglob("*.py"))
        candidates.extend((projectRoot / "scripts").rglob("*.py"))
        candidates.extend((projectRoot / "web" / "app").rglob("*.ts"))
        candidates.extend((projectRoot / "web" / "app").rglob("*.tsx"))
        candidates.extend((projectRoot / "web" / "components").rglob("*.ts"))
        candidates.extend((projectRoot / "web" / "components").rglob("*.tsx"))
        candidates.extend((projectRoot / "web" / "lib").rglob("*.ts"))
        violations = []
        for path in candidates:
            text = path.read_text(encoding="utf-8").lower()
            for needle in prohibited:
                if needle.lower() in text:
                    violations.append(f"{path.relative_to(projectRoot)}: {needle}")
        self.assertEqual([], violations)

    def test_model_configuration_resolves_explicit_local_paths(self):
        config = loadModelConfig()
        self.assertTrue(config["runtime"]["networkingDisabled"])
        for name in ("transcription", "faceDetection", "semanticAnalysis", "localCoach"):
            path = Path(config[name]["resolvedPath"])
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_relative_to(projectRoot))
        for value in ("https://example.invalid/model", "hf://owner/model", "hub://owner/model"):
            with self.assertRaises(Exception):
                resolveLocalPath(value)

    def test_missing_models_fail_before_any_loader_is_called(self):
        missing = str(projectRoot / "models" / "definitely-missing")
        _loadModel.cache_clear()
        with patch("faster_whisper.WhisperModel") as constructor:
            with self.assertRaisesRegex(TranscriptionError, "hf download Systran"):
                _loadModel(missing, "cpu", "int8")
            constructor.assert_not_called()

        _loadSemanticModel.cache_clear()
        with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
            with self.assertRaisesRegex(SemanticAnalysisError, "hf download sentence-transformers"):
                _loadSemanticModel(missing, "cpu")
            tokenizer.assert_not_called()

        _loadLocalCoach.cache_clear()
        with patch("transformers.AutoTokenizer.from_pretrained") as tokenizer:
            with self.assertRaisesRegex(LocalCoachError, "hf download Qwen"):
                _loadLocalCoach(missing, "cpu")
            tokenizer.assert_not_called()

    def test_transformer_loaders_are_local_files_only(self):
        local_path = str(projectRoot / "models" / "test-local-model")
        model = MagicMock()
        with patch("src.audioPipeline.semanticAnalyzer.validateModelFiles", return_value={"requiredFilesPresent": True}), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ) as tokenizer, patch("transformers.AutoModel.from_pretrained", return_value=model) as loader:
            _loadSemanticModel.cache_clear()
            _loadSemanticModel(local_path, "cpu")
            tokenizer.assert_called_once_with(local_path, local_files_only=True, trust_remote_code=False)
            loader.assert_called_once_with(local_path, local_files_only=True, trust_remote_code=False)

        coach_model = MagicMock()
        with patch("src.localModels.localCoach.validateModelFiles", return_value={"requiredFilesPresent": True}), patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ) as tokenizer, patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=coach_model) as loader:
            _loadLocalCoach.cache_clear()
            _loadLocalCoach(local_path, "cpu")
            tokenizer.assert_called_once_with(local_path, local_files_only=True, trust_remote_code=False)
            self.assertEqual(local_path, loader.call_args.args[0])
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])

    def test_faster_whisper_receives_a_local_directory(self):
        local_path = str(projectRoot / "models" / "test-local-whisper")
        with patch("src.audioPipeline.transcription.validateModelFiles", return_value={"requiredFilesPresent": True}), patch(
            "faster_whisper.WhisperModel", return_value=MagicMock()
        ) as constructor:
            _loadModel.cache_clear()
            _loadModel(local_path, "cpu", "int8")
            constructor.assert_called_once_with(local_path, device="cpu", compute_type="int8", local_files_only=True)

    def test_yolo_resolution_never_calls_a_remote_fallback(self):
        with patch("src.cvPipeline.yoloFaceDetector.validateModelFiles", return_value={"requiredFilesPresent": False}):
            with self.assertRaisesRegex(RuntimeError, "hf download AdamCodd"):
                resolve_yolo_face_model()

    def test_unsupported_generated_timestamps_are_removed(self):
        text = "Preserve the phrase at 1.25s.\nInvented replay at 99.0s.\nPractice the opening."
        cleaned, removed = sanitizeUnsupportedTimestamps(text, [0.0, 1.25, 2.5])
        self.assertEqual(1, removed)
        self.assertIn("1.25s", cleaned)
        self.assertNotIn("99.0s", cleaned)

    def test_recorder_state_disconnect_duplicate_stop_and_bounded_queue(self):
        with tempfile.TemporaryDirectory() as folder:
            recorder = WebcamRecorder(folder, queue_capacity=8)
            self.assertEqual(RecordingState.disconnected, recorder.state)
            recorder.set_connected(True)
            frame = np.full((48, 64, 3), 90, dtype=np.uint8)
            recorder.add_frame(frame)
            recorder.add_audio_samples(np.ones(1024, dtype=np.float32) * 0.02, 48000)
            self.assertEqual(RecordingState.ready, recorder.state)
            self.assertTrue(recorder.start())
            for index in range(80):
                recorder.add_frame(frame, media_timestamp=index / 30)
                recorder.add_audio_samples(
                    np.sin(np.arange(1600) * 2 * np.pi * 220 / 48000).astype(np.float32) * 0.1,
                    48000,
                    index / 30,
                )
            recorder.set_connected(False)
            output, stats = recorder.stop()
            self.assertIsNotNone(output, stats)
            self.assertLessEqual(stats["maxQueuedMemoryBytes"], 8 * (frame.nbytes + 1600 * 4))
            duplicate, duplicate_stats = recorder.stop()
            self.assertEqual(output, duplicate)
            self.assertTrue(duplicate_stats["duplicateStop"])
            self.assertFalse(output.with_suffix(".partial.mp4").exists())

    def test_all_eight_tabs_and_complete_download_labels_are_present(self):
        source = (projectRoot / "app.py").read_text(encoding="utf-8")
        for label in (
            "Overview",
            "Transcript",
            "Answer Quality",
            "Vocal Delivery",
            "Visual Cues",
            "Multimodal Moments",
            "Full Report",
            "Downloads",
        ):
            self.assertIn(f'"{label}"', source)
        for label in (
            "Original recording",
            "Extracted WAV",
            "Transcript JSON",
            "Semantic analysis",
            "Multimodal moments",
            "Deterministic report",
            "Local enhanced coaching",
            "Offline diagnostics",
        ):
            self.assertIn(f'("{label}"', source)


if __name__ == "__main__":
    unittest.main()
