import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


projectRoot = Path(__file__).resolve().parents[2]
defaultCueModelPath = projectRoot / "models" / "cueClassifier.joblib"
defaultCueMetadataPath = projectRoot / "models" / "cueClassifier.metadata.json"
modelSchemaVersion = 1

# Raw and temporal numeric evidence only. Rule outputs, timestamps, filenames, and
# ground-truth columns are deliberately excluded to prevent target leakage.
modelFeatureNames = [
    "faceDetected",
    "faceCount",
    "faceDetectionConfidence",
    "faceMeshDetected",
    "poseDetected",
    "faceCenterX",
    "faceCenterY",
    "faceWidth",
    "faceHeight",
    "faceAreaProxy",
    "faceEdgeMarginProxy",
    "noseX",
    "noseY",
    "shoulderLeftX",
    "shoulderLeftY",
    "shoulderRightX",
    "shoulderRightY",
    "elbowLeftX",
    "elbowLeftY",
    "elbowRightX",
    "elbowRightY",
    "wristLeftX",
    "wristLeftY",
    "wristRightX",
    "wristRightY",
    "hipLeftX",
    "hipLeftY",
    "hipRightX",
    "hipRightY",
    "poseVisibilityProxy",
    "wristLeftVisibility",
    "wristRightVisibility",
    "headMovementProxy",
    "headHorizontalChangeProxy",
    "noseYChangeProxy",
    "postureProxy",
    "postureChangeProxy",
    "bodyCenterOffsetProxy",
    "bodyLeanProxy",
    "handRaisedCount",
    "handMovementProxy",
    "cameraFacingProxy",
    "faceNoseOffsetXProxy",
    "mouthWidthProxy",
    "mouthOpennessProxy",
    "mouthMovementProxy",
    "mouthCornerLiftProxy",
    "eyebrowRaiseProxy",
    "eyeOpennessProxy",
    "eyeBalanceProxy",
    "headTiltProxy",
    "facialMovementProxy",
    "blinkLikeChangeProxy",
    "movementSpikeCount",
    "expressionSpikeCount",
    "brightnessProxy",
    "contrastProxy",
    "sharpnessProxy",
]

exclusiveCueGroups = [
    {"cameraFacing", "lookingAway"},
    {"centeredFraming", "offCenterFraming"},
    {"faceTooClose", "faceTooFar"},
    {"dimLighting", "overexposedLighting"},
    {"headTurnedLeft", "headTurnedRight"},
]


def featureFrame(rows, featureNames=None):
    featureNames = featureNames or modelFeatureNames
    frame = pd.DataFrame(rows)
    frame = frame.reindex(columns=featureNames)
    for column in frame.columns:
        if frame[column].dtype == bool:
            frame[column] = frame[column].astype(int)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


class CueClassifier:
    def __init__(self, bundle, modelPath=None):
        if bundle.get("schemaVersion") != modelSchemaVersion:
            raise ValueError(
                f"Unsupported cue model schema {bundle.get('schemaVersion')}; expected {modelSchemaVersion}."
            )
        self.pipeline = bundle["pipeline"]
        self.featureNames = bundle["featureNames"]
        self.classes = list(bundle["classes"])
        self.confidenceThreshold = float(bundle.get("confidenceThreshold", 0.60))
        self.modelPath = Path(modelPath) if modelPath else None

    def predict(self, rows):
        if not rows:
            return []

        features = featureFrame(rows, self.featureNames)
        probabilities = self.pipeline.predict_proba(features)
        predictions = []
        for rowProbabilities in probabilities:
            bestIndex = int(rowProbabilities.argmax())
            candidate = self.classes[bestIndex]
            confidence = float(rowProbabilities[bestIndex])
            predictions.append(
                {
                    "candidate": candidate,
                    "label": candidate if confidence >= self.confidenceThreshold and candidate != "baseline" else None,
                    "confidence": confidence,
                    "probabilities": {
                        cue: round(float(probability), 5)
                        for cue, probability in zip(self.classes, rowProbabilities)
                    },
                }
            )
        return predictions


def loadCueClassifier(modelPath=defaultCueModelPath):
    modelPath = Path(modelPath)
    if not modelPath.exists():
        return None
    try:
        import joblib
    except ImportError as error:
        raise RuntimeError("scikit-learn/joblib is required to load the learned cue classifier.") from error
    return CueClassifier(joblib.load(modelPath), modelPath)


def labelsConflict(candidate, existingLabels):
    for cueGroup in exclusiveCueGroups:
        if candidate in cueGroup and any(label in cueGroup and label != candidate for label in existingLabels):
            return True
    return False


def applyCueClassifier(rows, classifier=None, modelPath=defaultCueModelPath):
    classifier = classifier or loadCueClassifier(modelPath)
    if classifier is None:
        for row in rows:
            row.setdefault("mlCueCandidate", None)
            row.setdefault("mlCueLabel", None)
            row.setdefault("mlCueApplied", False)
            row.setdefault("mlCueConfidence", None)
            row.setdefault("mlCueProbabilities", None)
        return rows

    predictions = classifier.predict(rows)
    for row, prediction in zip(rows, predictions):
        ruleLabels = [label for label in str(row.get("ruleFrameLabels") or "").split(",") if label]
        combinedLabels = list(ruleLabels)
        acceptedLabel = prediction["label"]
        canApply = bool(acceptedLabel and not labelsConflict(acceptedLabel, ruleLabels))
        if canApply and acceptedLabel not in combinedLabels:
            combinedLabels.append(acceptedLabel)

        sources = {label: ["rule"] for label in ruleLabels}
        if canApply:
            sources.setdefault(acceptedLabel, []).append("ml")

        row["mlCueCandidate"] = prediction["candidate"]
        row["mlCueLabel"] = acceptedLabel
        row["mlCueApplied"] = canApply
        row["mlCueConfidence"] = round(prediction["confidence"], 5)
        row["mlCueProbabilities"] = json.dumps(prediction["probabilities"], sort_keys=True)
        row["frameLabelSources"] = json.dumps(sources, sort_keys=True)
        row["frameLabels"] = ",".join(combinedLabels)
    return rows


def buildModelMetadata(bundle, metrics=None):
    return {
        "schemaVersion": bundle["schemaVersion"],
        "trainedAtUtc": datetime.now(timezone.utc).isoformat(),
        "classes": list(bundle["classes"]),
        "featureNames": list(bundle["featureNames"]),
        "confidenceThreshold": bundle["confidenceThreshold"],
        "recordingCount": bundle.get("recordingCount"),
        "frameCount": bundle.get("frameCount"),
        "scikitLearnVersion": bundle.get("scikitLearnVersion"),
        "pandasVersion": pd.__version__,
        "evaluation": metrics,
        "integration": "Accepted ML cues are added only when they do not conflict with an existing rule cue.",
    }
