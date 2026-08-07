import json
import sys
from pathlib import Path

import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.cvPipeline.cueDefinitions import eventCueMap  # noqa: E402


featureDir = projectRoot / "data" / "demo" / "features"
groundTruthDir = projectRoot / "data" / "demo" / "groundTruth"
outputPath = groundTruthDir / "trainingDataset.csv"
availableIndicators = sorted(eventCueMap.keys())


def getRecordingStem(groundTruthPath):
    suffix = "_groundTruth.json"
    name = groundTruthPath.name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def loadSegments(groundTruthPath):
    try:
        return json.loads(groundTruthPath.read_text())
    except json.JSONDecodeError:
        print(f"Could not read ground truth JSON, skipping: {groundTruthPath}")
        return []


def indicatorsAtTimestamp(segments, timestampSeconds):
    matched = set()
    for segment in segments:
        if segment["startTime"] <= timestampSeconds <= segment["endTime"]:
            matched.add(segment["indicator"])
    return matched


def buildRecordingFrame(recordingStem, features, segments):
    features = features.copy()
    features["recordingStem"] = recordingStem

    matchedByRow = [indicatorsAtTimestamp(segments, timestamp) for timestamp in features["timestampSeconds"]]

    for indicator in availableIndicators:
        features[f"gt_{indicator}"] = [1 if indicator in matched else 0 for matched in matchedByRow]

    features["groundTruthIndicators"] = [",".join(sorted(matched)) for matched in matchedByRow]
    return features


def predictedIndicators(frameLabels):
    text = "" if pd.isna(frameLabels) else str(frameLabels).strip()
    if not text or text.lower() == "nan":
        return set()
    return {label.strip() for label in text.split(",") if label.strip()}


def summarizeAgreement(dataset):
    if "frameLabels" not in dataset.columns:
        print("No frameLabels column found in feature CSVs; skipping rule-based agreement summary.")
        return

    predictedByRow = dataset["frameLabels"].apply(predictedIndicators)
    rows = []

    for indicator in availableIndicators:
        actual = dataset[f"gt_{indicator}"].astype(bool)
        predicted = predictedByRow.apply(lambda matched, ind=indicator: ind in matched)

        truePositive = int((actual & predicted).sum())
        falsePositive = int((~actual & predicted).sum())
        falseNegative = int((actual & ~predicted).sum())

        precision = truePositive / (truePositive + falsePositive) if (truePositive + falsePositive) else None
        recall = truePositive / (truePositive + falseNegative) if (truePositive + falseNegative) else None

        rows.append(
            {
                "indicator": indicator,
                "labeledFrames": int(actual.sum()),
                "precision": round(precision, 3) if precision is not None else None,
                "recall": round(recall, 3) if recall is not None else None,
            }
        )

    summary = pd.DataFrame(rows)
    print()
    print("Rule-based detector agreement with your ground truth (frames with no ground truth segment count as negative):")
    print(summary.to_string(index=False))


def buildGroundTruthDataset():
    groundTruthDir.mkdir(parents=True, exist_ok=True)
    groundTruthFiles = sorted(groundTruthDir.glob("*_groundTruth.json"))

    if not groundTruthFiles:
        print(f"No ground truth files found in {groundTruthDir}.")
        print("Label at least one recording with: python -m streamlit run scripts/labelGroundTruth.py")
        raise SystemExit(1)

    recordingFrames = []

    for groundTruthPath in groundTruthFiles:
        recordingStem = getRecordingStem(groundTruthPath)
        featurePath = featureDir / f"{recordingStem}_features.csv"

        if not featurePath.exists():
            print(f"Skipping {recordingStem}: no matching feature CSV at {featurePath}.")
            print("Run scripts/analyzeInterviewDemo.py on that recording first.")
            continue

        segments = loadSegments(groundTruthPath)
        if not segments:
            print(f"Skipping {recordingStem}: ground truth file has no segments.")
            continue

        features = pd.read_csv(featurePath)
        recordingFrames.append(buildRecordingFrame(recordingStem, features, segments))
        print(f"Included {recordingStem}: {len(features)} frames, {len(segments)} labeled segments.")

    if not recordingFrames:
        print("No recordings had both a feature CSV and labeled ground truth. Nothing to build.")
        raise SystemExit(1)

    dataset = pd.concat(recordingFrames, ignore_index=True)
    dataset.to_csv(outputPath, index=False)

    print()
    print("Ground truth dataset built.")
    print(f"Recordings included: {len(recordingFrames)}")
    print(f"Total frames: {len(dataset)}")
    print(f"Saved to: {outputPath}")

    summarizeAgreement(dataset)


if __name__ == "__main__":
    buildGroundTruthDataset()
