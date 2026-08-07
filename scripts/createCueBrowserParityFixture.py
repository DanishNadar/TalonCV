"""Create a deterministic sklearn-to-browser cue-classifier parity fixture for tests."""

import json
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from scripts.trainCueClassifier import browserModel, buildPipeline  # noqa: E402
from src.cvPipeline.cueClassifier import featureFrame, modelFeatureNames  # noqa: E402


output_path = project_root / "web" / "tests" / "fixtures" / "cue-classifier-parity.json"


def row(face_center, hand_movement, missing=False):
    values = {name: 0.0 for name in modelFeatureNames}
    values.update({"faceDetected": 1, "faceCenterX": face_center, "handMovementProxy": hand_movement})
    if missing:
        values.pop("handMovementProxy")
    return values


def main():
    rows = [row(0.45, 0.01), row(0.48, 0.02), row(0.51, 0.03), row(0.47, 0.02), row(0.50, 0.8), row(0.53, 0.9), row(0.49, 0.85), row(0.52, 0.95), row(0.50, 0.92, missing=True)]
    labels = ["baseline", "baseline", "baseline", "baseline", "handGestureActivity", "handGestureActivity", "handGestureActivity", "handGestureActivity", "handGestureActivity"]
    pipeline = buildPipeline(25)
    features = featureFrame(rows)
    pipeline.fit(features, labels)
    probabilities = pipeline.predict_proba(features)
    classes = [str(value) for value in pipeline.named_steps["classifier"].classes_]
    expected = []
    for values in probabilities:
        best = int(values.argmax())
        expected.append({"candidate": classes[best], "confidence": float(values[best]), "probabilities": {label: float(value) for label, value in zip(classes, values)}})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"rows": rows, "expected": expected, "classifier": browserModel(pipeline, 0.6)}, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
