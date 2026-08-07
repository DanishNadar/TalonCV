import unittest
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
from streamlit.testing.v1 import AppTest

from scripts.smokeOfflinePipeline import blockOutboundSockets
from src.audioPipeline.transcriptAnalyzer import analyzeTranscript
from src.multimodalPipeline.artifacts import getArtifactPaths, writeJson, writeText
from src.multimodalPipeline.scoring import buildCoachingScores


projectRoot = Path(__file__).resolve().parents[1]


class StreamlitApplicationTests(unittest.TestCase):
    def test_complete_eight_tab_dashboard_renders_without_exceptions(self):
        stem = f"taloncv_streamlit_smoke_{uuid4().hex}"
        mediaPath = projectRoot / "data" / "demo" / "recordings" / f"{stem}.wav"
        mediaPath.parent.mkdir(parents=True, exist_ok=True)
        sampleRate = 16000
        samples = (0.15 * np.sin(2 * np.pi * 220 * np.arange(sampleRate * 2) / sampleRate)).astype(np.float32)
        pcm = (samples * 32767).astype("<i2")
        with wave.open(str(mediaPath), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sampleRate)
            output.writeframes(pcm.tobytes())
        paths = getArtifactPaths(mediaPath)
        transcript = {
            "text": "I built a routing rule and reduced delays by 30 percent.",
            "averageConfidence": 0.93,
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "I built a routing rule and reduced delays by 30 percent.",
                    "confidence": 0.93,
                }
            ],
            "warnings": [],
        }
        context = {"interviewQuestion": "Tell me about a process you improved."}
        semantic = {
            "available": True,
            "questionRelevance": {"available": True, "score": 82.0, "similarity": 0.58},
            "roleAlignment": {"available": False, "score": None},
            "segmentAssessments": [],
            "mostRelevantSegments": [],
            "topicDriftSegments": [],
            "vagueSegments": [],
            "semanticRedundancy": [],
            "warnings": [],
        }
        response = analyzeTranscript(transcript, context, semanticAnalysis=semantic)
        mediaInfo = {
            "valid": True,
            "hasAudio": True,
            "hasVideo": False,
            "durationSeconds": 2.0,
            "warnings": [],
        }
        audioFeatures = {
            "available": True,
            "durationSeconds": 2.0,
            "speechRatio": 0.9,
            "speechRateWpm": 145.0,
            "longPauseCount": 0,
            "overallRmsDb": -18.0,
            "noiseFloorDb": -55.0,
            "clippingPercentage": 0.0,
            "dropoutRatio": 0.0,
            "snrProxyDb": 28.0,
            "volumeConsistencyStdDb": 3.2,
            "energyVariationDb": 8.0,
            "pitchVariationSemitones": 2.0,
            "speechRateVariationWpm": 10.0,
            "fragmentedSpeechSegmentCount": 0,
        }
        scores = buildCoachingScores(mediaInfo, audioFeatures, [], response, [], [])
        analysis = {
            "complete": True,
            "mediaInfo": mediaInfo,
            "sessionContext": context,
            "transcript": transcript,
            "responseAnalysis": response,
            "semanticAnalysis": semantic,
            "audioFeatures": audioFeatures,
            "audioEvents": [],
            "visualEvents": [],
            "moments": [],
            "scores": scores,
            "localCoaching": {"available": False},
            "warnings": [],
        }
        writeJson(paths.multimodal, analysis)
        writeText(paths.report, "# TalonCV Streamlit Smoke Report\n\nDeterministic local report fixture.\n")
        writeText(paths.deterministicReport, "# TalonCV Streamlit Smoke Report\n")
        try:
            with blockOutboundSockets() as networkAttempts:
                app = AppTest.from_file(str(projectRoot / "app.py"), default_timeout=60).run()
            self.assertEqual([], list(app.exception))
            self.assertEqual([], networkAttempts)
            self.assertEqual(
                [
                    "Overview",
                    "Transcript",
                    "Answer Quality",
                    "Vocal Delivery",
                    "Visual Cues",
                    "Multimodal Moments",
                    "Full Report",
                    "Downloads",
                ],
                [tab.label for tab in app.tabs],
            )
            self.assertTrue(any(button.label == "Run Multimodal Analysis" for button in app.button))
        finally:
            for name, path in getArtifactPaths(mediaPath).asDict().items():
                if name != "media":
                    path.unlink(missing_ok=True)
            mediaPath.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
