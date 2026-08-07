import argparse
import sys
from pathlib import Path

import pandas as pd


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from scripts.analyzeInterviewDemo import analyzeVideo, loadMediaPipe  # noqa: E402
from src.cvPipeline.cueDataset import cueFromRecordingPath  # noqa: E402
from src.cvPipeline.yoloFaceDetector import loadYoloFaceDetector  # noqa: E402


defaultRecordingDir = projectRoot / "data" / "cueTraining" / "recordings"
defaultFeatureDir = projectRoot / "data" / "cueTraining" / "features"


def parseOptions():
    parser = argparse.ArgumentParser(
        description="Run YOLO/MediaPipe over cue-named MP4 files and create one training CSV per recording."
    )
    parser.add_argument("--input-dir", type=Path, default=defaultRecordingDir)
    parser.add_argument("--output-dir", type=Path, default=defaultFeatureDir)
    parser.add_argument("--force", action="store_true", help="Re-extract CSVs that already exist.")
    return parser.parse_args()


def absolutePath(path):
    return path if path.is_absolute() else projectRoot / path


def extractCueTrainingData():
    args = parseOptions()
    inputDir = absolutePath(args.input_dir)
    outputDir = absolutePath(args.output_dir)
    videos = sorted(inputDir.rglob("*.mp4")) if inputDir.exists() else []
    if not videos:
        raise SystemExit(
            f"No MP4 files found under {inputDir}. Record some with scripts/recordCueSamples.py first."
        )

    labeledVideos = []
    for videoPath in videos:
        cue = cueFromRecordingPath(videoPath)
        if cue is None:
            print(f"Skipping {videoPath.name}: rename it to <cue>__anything.mp4 or put it in a cue-named folder.")
            continue
        labeledVideos.append((videoPath, cue))

    if not labeledVideos:
        raise SystemExit("None of the videos had a recognized cue name.")

    outputDir.mkdir(parents=True, exist_ok=True)
    mp = loadMediaPipe()
    yoloFaceDetector = loadYoloFaceDetector()
    manifestRows = []

    for videoPath, cue in labeledVideos:
        relativeVideoPath = videoPath.relative_to(inputDir)
        featurePath = outputDir / relativeVideoPath.parent / f"{videoPath.stem}_features.csv"
        featurePath.parent.mkdir(parents=True, exist_ok=True)
        if featurePath.exists() and not args.force:
            frameCount = len(pd.read_csv(featurePath, usecols=["timestampSeconds"]))
            print(f"Reusing {featurePath.relative_to(outputDir)} ({frameCount} frames).")
        else:
            rows, _ = analyzeVideo(
                videoPath,
                mp,
                yoloFaceDetector=yoloFaceDetector,
                includeLearnedCues=False,
            )
            features = pd.DataFrame(rows)
            features.insert(0, "sourceVideo", str(videoPath.resolve()))
            features.insert(1, "recordingStem", videoPath.stem)
            features.insert(2, "groundTruthCue", cue)
            features.to_csv(featurePath, index=False)
            frameCount = len(features)
            print(f"Saved {featurePath.relative_to(outputDir)}: {cue}, {frameCount} frames.")

        manifestRows.append(
            {
                "recordingStem": videoPath.stem,
                "groundTruthCue": cue,
                "sourceVideo": str(videoPath.resolve()),
                "featureCsv": str(featurePath.resolve()),
                "frameCount": frameCount,
            }
        )

    manifestPath = outputDir / "cueTrainingManifest.csv"
    pd.DataFrame(manifestRows).to_csv(manifestPath, index=False)
    print(f"\nExtracted/reused {len(manifestRows)} recording CSV(s).")
    print(f"Manifest: {manifestPath}")
    print("Next: python scripts/trainCueClassifier.py")


if __name__ == "__main__":
    extractCueTrainingData()
