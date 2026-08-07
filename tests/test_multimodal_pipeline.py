import json
import tempfile
import time
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import av
import numpy as np

from src.audioPipeline.audioAnalyzer import analyzeAudio
from src.audioPipeline.mediaUtils import codecDiagnostics, extractAudioToWav, inspectMedia
from src.audioPipeline.transcriptAnalyzer import analyzeTranscript, detectFillers
from src.audioPipeline.transcription import transcribeAudio
from src.cvPipeline.webcamRecorder import WebcamRecorder
from src.multimodalPipeline.alignment import alignMultimodalEvents, overlaps
from src.multimodalPipeline.artifacts import getArtifactPaths, readJson, writeJson, writeText
from src.multimodalPipeline.pipeline import runMultimodalAnalysis
from src.multimodalPipeline.reportBuilder import buildMultimodalReport
from src.multimodalPipeline.scoring import buildCoachingScores


def unavailable_semantic(*_args, **_kwargs):
    return {
        "available": False,
        "questionRelevance": {"available": False, "score": None},
        "roleAlignment": {"available": False, "score": None},
        "segmentAssessments": [],
        "mostRelevantSegments": [],
        "topicDriftSegments": [],
        "vagueSegments": [],
        "semanticRedundancy": [],
        "warnings": ["Semantic model intentionally stubbed in controller unit test."],
    }


def unavailable_coach(*_args, **_kwargs):
    return {"available": False, "warnings": ["Local coach intentionally stubbed in controller unit test."]}


def write_wav(path, samples, sample_rate=16000):
    samples = np.clip(np.asarray(samples), -1.0, 1.0)
    pcm = (samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def write_test_media(path, with_video=True, with_audio=True, duration=1.2):
    with av.open(str(path), mode="w") as container:
        video_stream = None
        audio_stream = None
        if with_video:
            video_stream = container.add_stream("libx264", rate=10)
            video_stream.width = 64
            video_stream.height = 64
            video_stream.pix_fmt = "yuv420p"
        if with_audio:
            audio_stream = container.add_stream("aac", rate=16000)
            audio_stream.layout = "mono"

        if video_stream is not None:
            for index in range(max(2, int(duration * 10))):
                pixels = np.full((64, 64, 3), 40 + index * 3, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
                frame.pts = index
                frame.time_base = Fraction(1, 10)
                for packet in video_stream.encode(frame):
                    container.mux(packet)
            for packet in video_stream.encode():
                container.mux(packet)

        if audio_stream is not None:
            sample_rate = 16000
            timeline = np.arange(int(duration * sample_rate)) / sample_rate
            samples = (0.18 * np.sin(2 * np.pi * 220 * timeline)).astype(np.float32)
            for start in range(0, samples.size, 1024):
                block = samples[start : start + 1024].reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(block, format="fltp", layout="mono")
                frame.sample_rate = sample_rate
                frame.pts = start
                frame.time_base = Fraction(1, sample_rate)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)


def transcript_fixture():
    text = (
        "Um, on one project our team needed to reduce response time. "
        "I analyzed the queue, I built an automated routing rule, and as a result we reduced delays by 30 percent. "
        "Overall, the change helped the support team respond faster."
    )
    words = text.replace(",", "").replace(".", "").split()
    timestamps = [
        {"word": f" {word}", "start": index * 0.12, "end": index * 0.12 + 0.1, "probability": 0.93}
        for index, word in enumerate(words)
    ]
    return {
        "text": text,
        "averageConfidence": 0.91,
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": max(item["end"] for item in timestamps),
                "text": text,
                "confidence": 0.9,
                "words": timestamps,
            }
        ],
        "warnings": [],
    }


class MultimodalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_directory.name)
        self.created_stems = []

    def tearDown(self):
        for stem in self.created_stems:
            placeholder = self.temp_path / f"{stem}.mp4"
            for name, path in getArtifactPaths(placeholder).asDict().items():
                if name != "media":
                    path.unlink(missing_ok=True)
        self.temp_directory.cleanup()

    def unique_path(self, extension):
        stem = f"taloncv_test_{uuid4().hex}"
        self.created_stems.append(stem)
        return self.temp_path / f"{stem}{extension}"

    @unittest.skipUnless(codecDiagnostics()["ready"], "Packaged H.264/AAC encoders are unavailable")
    def test_recorder_creates_decodable_synchronized_av_mp4(self):
        recorder = WebcamRecorder(self.temp_path)
        recorder.set_connected(True)
        recorder.add_frame(np.full((64, 64, 3), 70, dtype=np.uint8))
        recorder.add_audio_samples(np.ones(1024, dtype=np.float32) * 0.01, 48000)
        self.assertTrue(recorder.ready_for_av_recording)
        self.assertTrue(recorder.start())
        recorder.add_frame(np.full((64, 64, 3), 80, dtype=np.uint8))
        timeline = np.arange(48000) / 48000
        recorder.add_audio_samples(0.15 * np.sin(2 * np.pi * 220 * timeline), 48000, 0.0)
        time.sleep(0.02)
        recorder.add_frame(np.full((64, 64, 3), 100, dtype=np.uint8))
        output, stats = recorder.stop()
        self.assertIsNotNone(output, stats)
        info = inspectMedia(output)
        self.assertTrue(info["hasVideo"])
        self.assertTrue(info["hasAudio"])
        self.assertEqual(info["video"]["codec"], "h264")
        self.assertEqual(info["audio"]["codec"], "aac")
        self.assertTrue(stats["audioAudible"])

    @unittest.skipUnless(codecDiagnostics()["ready"], "Packaged H.264/AAC encoders are unavailable")
    def test_audio_extraction_supports_av_audio_only_and_no_audio(self):
        av_path = self.unique_path(".mp4")
        write_test_media(av_path, with_video=True, with_audio=True)
        wav_path = self.temp_path / "decoded.wav"
        metadata_path = self.temp_path / "decoded.json"
        metadata = extractAudioToWav(av_path, wav_path, metadata_path)
        self.assertTrue(metadata["available"])
        self.assertEqual(metadata["sampleRate"], 16000)
        self.assertGreater(metadata["peakAmplitude"], 0.1)
        self.assertTrue(extractAudioToWav(av_path, wav_path, metadata_path)["cached"])

        audio_only = self.unique_path(".wav")
        write_wav(audio_only, np.sin(np.arange(8000) * 2 * np.pi * 220 / 16000) * 0.1)
        self.assertTrue(inspectMedia(audio_only)["hasAudio"])
        self.assertFalse(inspectMedia(audio_only)["hasVideo"])

        video_only = self.unique_path(".mp4")
        write_test_media(video_only, with_video=True, with_audio=False)
        unavailable = extractAudioToWav(video_only, self.temp_path / "none.wav")
        self.assertFalse(unavailable["available"])
        self.assertIn("No audio stream", unavailable["warnings"][0])

    def test_audio_analysis_detects_silence_pause_clipping_and_speech_rate(self):
        sample_rate = 16000
        tone = 0.18 * np.sin(2 * np.pi * 180 * np.arange(sample_rate) / sample_rate)
        pause = np.zeros(int(1.1 * sample_rate))
        clipped = np.ones(int(0.2 * sample_rate))
        path = self.unique_path(".wav")
        write_wav(path, np.concatenate([tone, pause, clipped, tone]))
        transcript = transcript_fixture()
        features, events = analyzeAudio(path, transcript)
        event_types = {event["eventType"] for event in events}
        self.assertTrue(features["available"])
        self.assertIn("longPause", event_types)
        self.assertIn("audioClipping", event_types)
        self.assertIn("rapidSpeech", event_types)
        self.assertGreater(features["speechRateWpm"], 180)

        silent_path = self.unique_path(".wav")
        write_wav(silent_path, np.zeros(sample_rate))
        silent_features, _ = analyzeAudio(silent_path)
        self.assertFalse(silent_features["available"])
        self.assertTrue(any("silent" in warning for warning in silent_features["warnings"]))

    def test_transcript_analysis_is_contextual_and_question_aware(self):
        transcript = transcript_fixture()
        response = analyzeTranscript(
            transcript,
            {"interviewQuestion": "Tell me about a project where you improved response time."},
        )
        self.assertTrue(response["available"])
        self.assertGreaterEqual(response["starAnalysis"]["componentsPresent"], 3)
        self.assertTrue(response["relevanceAnalysis"]["available"])
        self.assertGreater(response["relevanceAnalysis"]["score"], 35)
        self.assertTrue(response["strongPhrases"])
        self.assertIn("um", [item["phrase"] for item in response["fillerOccurrences"]])

        no_question = analyzeTranscript(transcript, {})
        self.assertFalse(no_question["relevanceAnalysis"]["available"])
        self.assertIsNone(no_question["rubric"]["relevance"]["score"])
        low_confidence_transcript = {
            **transcript,
            "averageConfidence": 0.38,
            "segments": [{**segment, "confidence": 0.38} for segment in transcript["segments"]],
        }
        low_confidence = analyzeTranscript(low_confidence_transcript, {"interviewQuestion": "What changed?"})
        self.assertEqual("low", low_confidence["confidence"])
        self.assertTrue(any("confidence" in warning.lower() for warning in low_confidence["warnings"]))
        fillers = detectFillers([{"start": 0, "end": 1, "text": "I would like to lead. So, um, I did."}])
        self.assertNotIn("like", [item["phrase"] for item in fillers])
        self.assertIn("um", [item["phrase"] for item in fillers])

    def test_transcription_schema_and_cache_with_model_adapter(self):
        class FakeModel:
            def transcribe(self, *_args, **_kwargs):
                word = SimpleNamespace(start=0.0, end=0.3, word=" Hello", probability=0.95)
                segment = SimpleNamespace(
                    id=0,
                    start=0.0,
                    end=0.5,
                    text=" Hello world",
                    avg_logprob=-0.1,
                    no_speech_prob=0.02,
                    words=[word],
                )
                info = SimpleNamespace(language="en", language_probability=0.99, duration=0.5, duration_after_vad=0.5)
                return iter([segment]), info

        wav_path = self.unique_path(".wav")
        write_wav(wav_path, np.zeros(8000))
        json_path = self.temp_path / "transcript.json"
        text_path = self.temp_path / "transcript.txt"
        fingerprint = {"sha256": "test", "sizeBytes": wav_path.stat().st_size}
        with patch("src.audioPipeline.transcription._loadModel", return_value=FakeModel()) as loader:
            transcript = transcribeAudio(wav_path, wav_path.stem, json_path, text_path, fingerprint)
            cached = transcribeAudio(wav_path, wav_path.stem, json_path, text_path, fingerprint)
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(transcript["text"], "Hello world")
        self.assertEqual(transcript["segments"][0]["words"][0]["word"], " Hello")
        self.assertIn("averageConfidence", transcript)
        self.assertTrue(cached["cached"])

    def test_alignment_overlap_schema_order_and_examples(self):
        self.assertTrue(overlaps(1, 3, 2, 4))
        self.assertFalse(overlaps(1, 2, 2, 3))
        transcript = {"segments": [{"start": 0, "end": 5, "text": "I built the tool and improved speed."}]}
        response = {
            "fillerOccurrences": [{"phrase": "um", "startTime": 3.0, "endTime": 3.4, "confidence": "high"}],
            "strongPhrases": [{"text": "I built the tool", "startTime": 1.0, "endTime": 2.0, "reasons": ["action"]}],
            "metrics": {"hasConclusion": False},
        }
        audio_events = [
            {"eventType": "longPause", "startTime": 0.2, "endTime": 0.9, "reliability": "high"},
            {"eventType": "strongVocalEmphasis", "startTime": 1.0, "endTime": 1.5, "reliability": "medium"},
        ]
        visual_events = [
            {"eventType": "lookingDown", "startTime": 0.1, "endTime": 1.0},
            {"eventType": "handGestureActivity", "startTime": 1.1, "endTime": 1.8},
            {"eventType": "cameraFacing", "startTime": 1.0, "endTime": 2.2},
            {"eventType": "possibleFidgeting", "startTime": 2.9, "endTime": 3.5},
        ]
        moments = alignMultimodalEvents(transcript, response, audio_events, visual_events)
        categories = {moment["alignmentCategory"] for moment in moments}
        self.assertIn("pauseWithDownwardGaze", categories)
        self.assertIn("emphasisWithGesture", categories)
        self.assertIn("fillerWithVisualMovement", categories)
        self.assertIn("strongContentWithVisualSupport", categories)
        self.assertEqual(moments, sorted(moments, key=lambda item: (item["startTime"], item["alignmentCategory"])))
        self.assertTrue(all("coachingRecommendation" in moment for moment in moments))

    def test_scores_exclude_missing_modalities_and_explain_weights(self):
        response = analyzeTranscript(transcript_fixture(), {})
        bundle = buildCoachingScores(
            {"hasVideo": False, "hasAudio": True, "durationSeconds": 4},
            {"available": True, "clippingPercentage": 0, "dropoutRatio": 0, "overallRmsDb": -22},
            [],
            response,
            [],
            [],
        )
        scores = bundle["scores"]
        self.assertIsNone(scores["visualDelivery"]["score"])
        self.assertIsNone(scores["multimodalAlignment"]["score"])
        overall = scores["overallInterviewPracticeDelivery"]
        self.assertIsNotNone(overall["score"])
        self.assertIn("visualDelivery", overall["componentBreakdown"]["excludedComponents"])
        self.assertIn("not hiring scores", bundle["safetyNote"])

    def test_report_contains_all_sixteen_sections_and_missing_modality_language(self):
        response = analyzeTranscript(transcript_fixture(), {})
        scores = buildCoachingScores(
            {"hasVideo": False, "hasAudio": True, "durationSeconds": 4},
            {"available": True, "durationSeconds": 4, "sampleRate": 16000, "sourceChannels": 1},
            [],
            response,
            [],
            [],
        )
        report = buildMultimodalReport(
            "answer.wav",
            {},
            {"hasVideo": False, "hasAudio": True, "durationSeconds": 4, "warnings": []},
            transcript_fixture(),
            response,
            {"available": True, "durationSeconds": 4, "sampleRate": 16000, "sourceChannels": 1},
            [],
            [],
            [],
            scores,
        )
        for section_number in range(1, 17):
            self.assertIn(f"## {section_number}.", report)
        self.assertIn("Visual analysis was unavailable", report)
        self.assertIn("not a hiring model", report)

    def test_report_supports_visual_only_and_complete_multimodal_evidence(self):
        visual_event = {
            "eventType": "cameraFacing",
            "startTime": 0.5,
            "endTime": 1.5,
            "durationSeconds": 1.0,
            "description": "Camera-facing geometry was measured.",
        }
        visual_scores = buildCoachingScores(
            {"hasVideo": True, "hasAudio": False, "durationSeconds": 3}, {}, [], {}, [visual_event], []
        )
        visual_only = buildMultimodalReport(
            "visual.mp4",
            {},
            {"hasVideo": True, "hasAudio": False, "durationSeconds": 3, "warnings": []},
            {},
            {},
            {},
            [],
            [visual_event],
            [],
            visual_scores,
        )
        self.assertIn("Transcript unavailable", visual_only)
        self.assertIn("cameraFacing", visual_only)
        self.assertIn("Audio-quality analysis was unavailable", visual_only)

        transcript = transcript_fixture()
        response = analyzeTranscript(transcript, {"interviewQuestion": "What did you improve?"})
        audio_features = {
            "available": True,
            "durationSeconds": 4,
            "sampleRate": 16000,
            "sourceChannels": 1,
            "speechRateWpm": 140,
            "speechRatio": 0.7,
            "silenceRatio": 0.3,
        }
        moment = {
            "startTime": 0.5,
            "endTime": 1.5,
            "classification": "strength",
            "alignmentCategory": "strongContentWithVisualSupport",
            "explanation": "Specific content aligned with camera-facing delivery.",
            "coachingRecommendation": "Preserve this delivery pattern.",
            "transcriptExcerpt": "I built an automated routing rule.",
        }
        complete_scores = buildCoachingScores(
            {"hasVideo": True, "hasAudio": True, "durationSeconds": 4},
            audio_features,
            [],
            response,
            [visual_event],
            [moment],
        )
        complete = buildMultimodalReport(
            "complete.mp4",
            {"interviewQuestion": "What did you improve?"},
            {"hasVideo": True, "hasAudio": True, "durationSeconds": 4, "warnings": []},
            transcript,
            response,
            audio_features,
            [],
            [visual_event],
            [moment],
            complete_scores,
        )
        self.assertIn("strongContentWithVisualSupport", complete)
        self.assertIn("Specific content aligned", complete)

    def test_artifact_paths_are_deterministic_and_corrupt_json_is_safe(self):
        source = self.unique_path(".mp4")
        source.write_bytes(b"media")
        first = getArtifactPaths(source)
        second = getArtifactPaths(source)
        self.assertEqual(first, second)
        self.assertTrue(first.transcriptJson.name.endswith("_transcript.json"))
        corrupt = self.temp_path / "corrupt.json"
        corrupt.write_text("{bad json", encoding="utf-8")
        self.assertEqual(readJson(corrupt, {"safe": True}), {"safe": True})

    @unittest.skipUnless(codecDiagnostics()["ready"], "Packaged H.264/AAC encoders are unavailable")
    def test_controller_completes_with_video_only_and_skips_transcriber(self):
        video_path = self.unique_path(".mp4")
        write_test_media(video_path, with_video=True, with_audio=False)
        stale_paths = getArtifactPaths(video_path)
        stale_paths.audio.parent.mkdir(parents=True, exist_ok=True)
        stale_paths.audio.write_bytes(b"stale audio")

        def transcriber_should_not_run(*_args, **_kwargs):
            raise AssertionError("Transcriber was called for media without audio")

        def fake_visual(*_args, **_kwargs):
            rows = [{"timestampSeconds": 0.0, "frameLabels": "cameraFacing"}]
            events = [
                {
                    "eventType": "cameraFacing",
                    "startTime": 0.0,
                    "endTime": 0.8,
                    "durationSeconds": 0.8,
                    "description": "Camera-facing geometry was measured.",
                }
            ]
            return rows, events, 1.2

        result = runMultimodalAnalysis(
            video_path,
            {"interviewQuestion": "Tell me about your project."},
            force=True,
            transcriber=transcriber_should_not_run,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
            visualAnalyzer=fake_visual,
        )
        self.assertTrue(result["complete"])
        self.assertFalse(result["mediaInfo"]["hasAudio"])
        self.assertFalse(result["responseAnalysis"]["available"])
        self.assertTrue(any("no audio stream" in warning.lower() for warning in result["warnings"]))
        self.assertTrue(stale_paths.report.exists())
        self.assertFalse(stale_paths.audio.exists())
        self.assertEqual(readJson(stale_paths.transcriptJson)["text"], "")

    @unittest.skipUnless(codecDiagnostics()["ready"], "Packaged H.264/AAC encoders are unavailable")
    def test_controller_integrates_audio_transcript_visual_alignment_and_report(self):
        media_path = self.unique_path(".mp4")
        write_test_media(media_path, with_video=True, with_audio=True, duration=1.5)

        def fake_transcriber(_wav, stem, json_path, text_path, fingerprint, **_kwargs):
            transcript = {**transcript_fixture(), "recordingIdentifier": stem, "sourceFingerprint": fingerprint}
            writeJson(json_path, transcript)
            writeText(text_path, transcript["text"])
            return transcript

        def fake_visual(*_args, **_kwargs):
            rows = [{"timestampSeconds": 0.0, "frameLabels": "cameraFacing|stablePosture"}]
            events = [
                {
                    "eventType": "cameraFacing",
                    "startTime": 0.0,
                    "endTime": 1.4,
                    "durationSeconds": 1.4,
                    "description": "Camera-facing geometry was measured.",
                },
                {
                    "eventType": "stablePosture",
                    "startTime": 0.0,
                    "endTime": 1.4,
                    "durationSeconds": 1.4,
                    "description": "Posture measurements stayed stable.",
                },
            ]
            return rows, events, 1.5

        result = runMultimodalAnalysis(
            media_path,
            {"interviewQuestion": "Tell me about a project where you improved response time."},
            force=True,
            transcriber=fake_transcriber,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
            visualAnalyzer=fake_visual,
        )
        scores = result["scores"]["scores"]
        self.assertTrue(result["mediaInfo"]["hasAudio"])
        self.assertTrue(result["mediaInfo"]["hasVideo"])
        self.assertTrue(result["transcript"]["text"])
        self.assertTrue(result["visualEvents"])
        self.assertTrue(any(moment["alignmentCategory"] == "strongContentWithVisualSupport" for moment in result["moments"]))
        self.assertIsNotNone(scores["audioRecordingQuality"]["score"])
        self.assertIsNotNone(scores["verbalResponseQuality"]["score"])
        self.assertIsNotNone(scores["visualDelivery"]["score"])
        self.assertIsNotNone(scores["multimodalAlignment"]["score"])
        report = getArtifactPaths(media_path).report.read_text(encoding="utf-8")
        self.assertIn("## 12. Multimodal alignment moments", report)

    def test_controller_completes_audio_only_and_preserves_artifacts(self):
        audio_path = self.unique_path(".wav")
        sample_rate = 16000
        samples = 0.15 * np.sin(2 * np.pi * 220 * np.arange(sample_rate * 2) / sample_rate)
        write_wav(audio_path, samples)
        stale_paths = getArtifactPaths(audio_path)
        writeJson(stale_paths.visualEvents, [{"eventType": "staleVisualEvent"}])
        stale_paths.visualFeatures.parent.mkdir(parents=True, exist_ok=True)
        stale_paths.visualFeatures.write_text("timestampSeconds,frameLabels\n0,stale\n", encoding="utf-8")

        def fake_transcriber(_wav, stem, json_path, text_path, fingerprint, **_kwargs):
            transcript = {**transcript_fixture(), "recordingIdentifier": stem, "sourceFingerprint": fingerprint}
            writeJson(json_path, transcript)
            writeText(text_path, transcript["text"])
            return transcript

        def visual_should_not_run(*_args, **_kwargs):
            raise AssertionError("Visual analyzer was called for audio-only media")

        result = runMultimodalAnalysis(
            audio_path,
            {"interviewQuestion": "Tell me about a project where you improved response time."},
            force=True,
            transcriber=fake_transcriber,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
            visualAnalyzer=visual_should_not_run,
        )
        paths = getArtifactPaths(audio_path)
        self.assertTrue(result["complete"])
        self.assertFalse(result["mediaInfo"]["hasVideo"])
        self.assertTrue(result["responseAnalysis"]["available"])
        self.assertTrue(paths.transcriptJson.exists())
        self.assertTrue(paths.audioFeatures.exists())
        self.assertTrue(paths.multimodal.exists())
        self.assertIsNone(result["scores"]["scores"]["visualDelivery"]["score"])
        self.assertEqual(readJson(paths.visualEvents), [])
        self.assertEqual(paths.visualFeatures.read_text(encoding="utf-8").strip(), "timestampSeconds,frameLabels")

    def test_controller_reuses_cache_preserves_context_and_invalidates_changed_source(self):
        audio_path = self.unique_path(".wav")
        sample_rate = 16000
        timeline = np.arange(sample_rate) / sample_rate
        write_wav(audio_path, 0.12 * np.sin(2 * np.pi * 180 * timeline))
        transcription_calls = []

        def fake_transcriber(_wav, stem, json_path, text_path, fingerprint, **_kwargs):
            transcription_calls.append(fingerprint["sha256"])
            transcript = {**transcript_fixture(), "recordingIdentifier": stem, "sourceFingerprint": fingerprint}
            writeJson(json_path, transcript)
            writeText(text_path, transcript["text"])
            return transcript

        first = runMultimodalAnalysis(
            audio_path,
            {"interviewQuestion": "Describe a measurable improvement."},
            force=True,
            transcriber=fake_transcriber,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
        )
        reused = runMultimodalAnalysis(
            audio_path,
            force=False,
            transcriber=fake_transcriber,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
        )
        self.assertFalse(first["cached"])
        self.assertTrue(reused["cached"])
        self.assertEqual(reused["sessionContext"]["interviewQuestion"], "Describe a measurable improvement.")
        self.assertEqual(len(transcription_calls), 1)

        write_wav(audio_path, 0.18 * np.sin(2 * np.pi * 260 * timeline))
        invalidated = runMultimodalAnalysis(
            audio_path,
            force=False,
            transcriber=fake_transcriber,
            semanticAnalyzer=unavailable_semantic,
            localCoach=unavailable_coach,
        )
        self.assertFalse(invalidated["cached"])
        self.assertNotEqual(first["sourceFingerprint"]["sha256"], invalidated["sourceFingerprint"]["sha256"])
        self.assertEqual(len(transcription_calls), 2)


if __name__ == "__main__":
    unittest.main()
