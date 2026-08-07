import json
import unittest
from pathlib import Path

import numpy as np

from scripts.analyzeInterviewDemo import createEvents
from src.cvPipeline.cueClassifier import applyCueClassifier
from src.cvPipeline.cueDataset import canonicalCueName, cueFromRecordingPath, cueRecordingTarget, recordableCueTypes
from src.cvPipeline.cueRules import getFrameLabels
from src.cvPipeline.yoloFaceDetector import YoloFaceDetector


class FakeBox:
    def __init__(self, coordinates, confidence):
        self.cls = np.array([0])
        self.xyxy = np.array([coordinates], dtype=float)
        self.conf = np.array([confidence], dtype=float)


class FakeResult:
    names = {0: "face"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeYoloModel:
    names = {0: "face"}

    def predict(self, *_args, **_kwargs):
        return [FakeResult([FakeBox((20, 10, 80, 90), 0.8), FakeBox((5, 5, 25, 30), 0.6)])]


class FakeCueClassifier:
    def predict(self, rows):
        return [
            {
                "candidate": "handGestureActivity",
                "label": "handGestureActivity",
                "confidence": 0.91,
                "probabilities": {"baseline": 0.09, "handGestureActivity": 0.91},
            }
            for _ in rows
        ]


class ConflictingCueClassifier:
    def predict(self, rows):
        return [
            {
                "candidate": "lookingAway",
                "label": "lookingAway",
                "confidence": 0.95,
                "probabilities": {"cameraFacing": 0.05, "lookingAway": 0.95},
            }
            for _ in rows
        ]


class CuePipelineTests(unittest.TestCase):
    def test_every_recordable_cue_has_a_training_ready_iteration_target(self):
        for cue in recordableCueTypes:
            target = cueRecordingTarget(cue)
            self.assertGreaterEqual(target["minimumTakes"], 30)
            self.assertGreaterEqual(target["targetTakes"], target["minimumTakes"])

    def test_cue_filename_parser_accepts_renamed_files_and_alias_formatting(self):
        self.assertEqual(canonicalCueName("Camera Facing"), "cameraFacing")
        self.assertEqual(cueFromRecordingPath(Path("cameraFacing__take_001.mp4")), "cameraFacing")
        self.assertEqual(cueFromRecordingPath(Path("lookingAway/recording-01.mp4")), "lookingAway")

    def test_yolo_detector_exposes_count_scale_confidence_and_edge_margin(self):
        detector = YoloFaceDetector(FakeYoloModel(), "fake")
        result = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result["faceCount"], 2)
        self.assertEqual(result["faceDetectionConfidence"], 0.8)
        self.assertEqual(result["faceAreaProxy"], 0.48)
        self.assertEqual(result["faceEdgeMarginProxy"], 0.1)

    def test_fixed_practical_cues_are_emitted(self):
        row = {
            "faceDetected": True,
            "faceCount": 2,
            "faceDetectionConfidence": 0.2,
            "faceMeshDetected": False,
            "poseDetected": True,
            "faceCenterX": 0.85,
            "faceCenterY": 0.4,
            "faceHeight": 0.7,
            "faceEdgeMarginProxy": 0.0,
            "brightnessProxy": 0.1,
            "contrastProxy": 0.08,
            "sharpnessProxy": 0.02,
            "postureProxy": 0.2,
            "bodyLeanProxy": 0.3,
            "bodyCenterOffsetProxy": 0.25,
            "handRaisedCount": 1,
            "headTiltProxy": 0.2,
            "mouthOpennessProxy": 0.08,
            "faceNoseOffsetXProxy": -0.5,
        }
        labels = getFrameLabels(row, {})
        expected = {
            "multipleFaces",
            "lowFaceConfidence",
            "facePartiallyOutOfFrame",
            "faceTooClose",
            "offCenterFraming",
            "faceMeshMissing",
            "dimLighting",
            "lowContrast",
            "blurryImage",
            "shoulderTilt",
            "bodyLean",
            "bodyOffCenter",
            "handsRaised",
            "headTilt",
            "mouthOpen",
            "headTurnedLeft",
        }
        self.assertTrue(expected.issubset(labels))

    def test_ml_cue_adds_without_replacing_rule_cues_and_keeps_provenance(self):
        rows = [{"ruleFrameLabels": "cameraFacing", "frameLabels": "cameraFacing"}]
        applyCueClassifier(rows, classifier=FakeCueClassifier())
        self.assertEqual(rows[0]["frameLabels"], "cameraFacing,handGestureActivity")
        self.assertEqual(rows[0]["mlCueLabel"], "handGestureActivity")
        sources = json.loads(rows[0]["frameLabelSources"])
        self.assertEqual(sources["cameraFacing"], ["rule"])
        self.assertEqual(sources["handGestureActivity"], ["ml"])

    def test_event_records_ml_source_and_mean_confidence(self):
        rows = [
            {
                "timestampSeconds": 0.0,
                "frameLabels": "handGestureActivity",
                "frameLabelSources": '{"handGestureActivity": ["ml"]}',
                "mlCueLabel": "handGestureActivity",
                "mlCueConfidence": 0.8,
            },
            {
                "timestampSeconds": 0.5,
                "frameLabels": "handGestureActivity",
                "frameLabelSources": '{"handGestureActivity": ["ml"]}',
                "mlCueLabel": "handGestureActivity",
                "mlCueConfidence": 0.9,
            },
        ]
        events = createEvents(rows, 1.0)
        self.assertEqual(events[0]["detectionSources"], ["ml"])
        self.assertEqual(events[0]["mlConfidenceMean"], 0.85)

    def test_ml_cue_cannot_override_a_conflicting_rule(self):
        rows = [{"ruleFrameLabels": "cameraFacing", "frameLabels": "cameraFacing"}]
        applyCueClassifier(rows, classifier=ConflictingCueClassifier())
        self.assertEqual(rows[0]["frameLabels"], "cameraFacing")
        self.assertEqual(rows[0]["mlCueLabel"], "lookingAway")
        self.assertFalse(rows[0]["mlCueApplied"])


if __name__ == "__main__":
    unittest.main()
