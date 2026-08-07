import argparse
import json
import sys
from pathlib import Path

import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.cvPipeline.cueClassifier import (  # noqa: E402
    buildModelMetadata,
    defaultCueMetadataPath,
    defaultCueModelPath,
    featureFrame,
    modelFeatureNames,
    modelSchemaVersion,
)
from src.cvPipeline.cueDataset import canonicalCueName  # noqa: E402


defaultFeatureDir = projectRoot / "data" / "cueTraining" / "features"
defaultBrowserModelPath = projectRoot / "web" / "public" / "models" / "cue-classifier.json"


def parseOptions():
    parser = argparse.ArgumentParser(
        description="Train the additive TalonCV visual-cue classifier from per-recording feature CSVs."
    )
    parser.add_argument("--feature-dir", type=Path, default=defaultFeatureDir)
    parser.add_argument("--model-path", type=Path, default=defaultCueModelPath)
    parser.add_argument("--metadata-path", type=Path, default=defaultCueMetadataPath)
    parser.add_argument("--browser-model-path", type=Path, default=defaultBrowserModelPath)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.60,
        help="Minimum top-class probability before ML can add a cue to the rule decision.",
    )
    parser.add_argument("--trees", type=int, default=400, help="Number of random-forest trees.")
    args = parser.parse_args()
    if not 0 < args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in (0, 1].")
    if args.trees < 10:
        parser.error("--trees must be at least 10.")
    return args


def absolutePath(path):
    return path if path.is_absolute() else projectRoot / path


def loadTrainingRows(featureDir):
    csvPaths = sorted(path for path in featureDir.rglob("*_features.csv") if path.is_file())
    if not csvPaths:
        raise SystemExit(f"No per-recording feature CSVs found in {featureDir}.")

    frames = []
    for csvPath in csvPaths:
        frame = pd.read_csv(csvPath)
        if "groundTruthCue" not in frame.columns:
            print(f"Skipping {csvPath.name}: groundTruthCue column is missing.")
            continue
        frame["groundTruthCue"] = frame["groundTruthCue"].apply(canonicalCueName)
        frame = frame[frame["groundTruthCue"].notna()].copy()
        if frame.empty:
            print(f"Skipping {csvPath.name}: it has no recognized cue labels.")
            continue
        if "recordingStem" not in frame.columns:
            frame["recordingStem"] = csvPath.name[: -len("_features.csv")]
        frame["recordingId"] = frame.get("sourceVideo", frame["recordingStem"]).fillna(frame["recordingStem"])
        frames.append(frame)

    if not frames:
        raise SystemExit("No valid labeled training rows were found.")
    return pd.concat(frames, ignore_index=True)


def buildPipeline(treeCount):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=treeCount,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def recordingWeights(data):
    frameCounts = data["recordingId"].value_counts()
    weights = data["recordingId"].map(lambda recordingId: 1.0 / frameCounts[recordingId]).astype(float)
    return weights * (len(weights) / weights.sum())


def fitPipeline(data, treeCount):
    pipeline = buildPipeline(treeCount)
    pipeline.fit(
        featureFrame(data.to_dict("records")),
        data["groundTruthCue"],
        classifier__sample_weight=recordingWeights(data),
    )
    return pipeline


def browserTree(tree):
    values = []
    for nodeValues in tree.value:
        counts = nodeValues[0].tolist()
        total = sum(counts) or 1
        values.append([float(value / total) for value in counts])
    return {
        "childrenLeft": tree.children_left.tolist(),
        "childrenRight": tree.children_right.tolist(),
        "feature": tree.feature.tolist(),
        "threshold": [float(value) for value in tree.threshold.tolist()],
        "value": values,
    }


def browserModel(pipeline, confidenceThreshold):
    imputer = pipeline.named_steps["imputer"]
    classifier = pipeline.named_steps["classifier"]
    return {
        "schemaVersion": 1,
        "modelType": "sklearn-random-forest-classifier",
        "featureNames": list(modelFeatureNames),
        "classes": [str(value) for value in classifier.classes_],
        "confidenceThreshold": float(confidenceThreshold),
        "imputer": {
            "strategy": "median",
            "statistics": [float(value) for value in imputer.statistics_],
            "indicatorFeatures": [int(value) for value in getattr(imputer.indicator_, "features_", [])],
        },
        "trees": [browserTree(estimator.tree_) for estimator in classifier.estimators_],
    }


def evaluateByHeldOutRecordings(data, treeCount):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

    recordings = data[["recordingId", "groundTruthCue"]].drop_duplicates()
    counts = recordings.groupby("groundTruthCue")["recordingId"].nunique()
    if (counts < 2).any():
        missing = ", ".join(f"{cue}={count}" for cue, count in counts[counts < 2].items())
        return {
            "status": "not-run",
            "reason": f"Need at least two separate recordings per cue for a leakage-safe holdout ({missing}).",
        }

    heldOutIds = set()
    for _, cueRecordings in recordings.sort_values("recordingId").groupby("groundTruthCue"):
        heldOutIds.add(cueRecordings.iloc[-1]["recordingId"])

    train = data[~data["recordingId"].isin(heldOutIds)]
    test = data[data["recordingId"].isin(heldOutIds)]
    evaluationPipeline = fitPipeline(train, treeCount)
    actual = test["groundTruthCue"]
    predicted = evaluationPipeline.predict(featureFrame(test.to_dict("records")))
    labels = sorted(data["groundTruthCue"].unique())
    return {
        "status": "complete",
        "strategy": "one entire recording per cue held out; no frames from a held-out recording enter training",
        "trainRecordings": int(train["recordingId"].nunique()),
        "testRecordings": int(test["recordingId"].nunique()),
        "testFrames": int(len(test)),
        "accuracy": round(float(accuracy_score(actual, predicted)), 5),
        "balancedAccuracy": round(float(balanced_accuracy_score(actual, predicted)), 5),
        "classificationReport": classification_report(
            actual, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "confusionMatrixLabels": labels,
        "confusionMatrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def trainCueClassifier():
    import sklearn

    args = parseOptions()
    featureDir = absolutePath(args.feature_dir)
    modelPath = absolutePath(args.model_path)
    metadataPath = absolutePath(args.metadata_path)
    browserModelPath = absolutePath(args.browser_model_path)
    data = loadTrainingRows(featureDir)

    classCounts = data.groupby("groundTruthCue").agg(
        frames=("groundTruthCue", "size"), recordings=("recordingId", "nunique")
    )
    if len(classCounts) < 2:
        raise SystemExit("Training needs at least two different cues (include baseline as one useful class).")

    print("Training coverage:")
    print(classCounts.to_string())
    print()
    metrics = evaluateByHeldOutRecordings(data, args.trees)
    if metrics["status"] == "complete":
        print(
            f"Held-recording evaluation: accuracy={metrics['accuracy']:.3f}, "
            f"balanced accuracy={metrics['balancedAccuracy']:.3f}"
        )
    else:
        print(f"Evaluation skipped: {metrics['reason']}")

    pipeline = fitPipeline(data, args.trees)
    classes = list(pipeline.named_steps["classifier"].classes_)
    bundle = {
        "schemaVersion": modelSchemaVersion,
        "pipeline": pipeline,
        "featureNames": modelFeatureNames,
        "classes": classes,
        "confidenceThreshold": args.confidence_threshold,
        "recordingCount": int(data["recordingId"].nunique()),
        "frameCount": int(len(data)),
        "scikitLearnVersion": sklearn.__version__,
    }

    try:
        import joblib
    except ImportError as error:
        raise SystemExit("Install requirements.txt so scikit-learn/joblib are available.") from error

    modelPath.parent.mkdir(parents=True, exist_ok=True)
    metadataPath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, modelPath)
    metadataPath.write_text(json.dumps(buildModelMetadata(bundle, metrics), indent=2))
    browserModelPath.parent.mkdir(parents=True, exist_ok=True)
    browserModelPath.write_text(json.dumps(browserModel(pipeline, args.confidence_threshold), separators=(",", ":")))
    print(f"\nSaved classifier: {modelPath}")
    print(f"Saved metadata/evaluation: {metadataPath}")
    print(f"Saved browser cue classifier: {browserModelPath}")
    print("Future analyses will load this classifier automatically and add only non-conflicting high-confidence cues.")


if __name__ == "__main__":
    trainCueClassifier()
