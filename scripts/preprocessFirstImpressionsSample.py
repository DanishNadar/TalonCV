from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


projectRoot = Path(__file__).resolve().parents[1]
sampleManifestPath = projectRoot / "data" / "processed" / "manifests" / "firstImpressionsSampleManifest.csv"
preprocessManifestPath = projectRoot / "data" / "processed" / "manifests" / "firstImpressionsPreprocessManifest.csv"
frameOutputPath = projectRoot / "data" / "processed" / "firstImpressionsSample" / "frames"
frameIntervalSeconds = 5


def readSampleManifest():
    if not sampleManifestPath.exists():
        print(f"Sample manifest does not exist yet: {sampleManifestPath}")
        print("Run scripts/createFirstImpressionsSample.py first.")
        raise SystemExit(1)

    return pd.read_csv(sampleManifestPath)


def savePreviewFrames(videoCapture, sampleId, fps, frameCount):
    if fps <= 0 or frameCount <= 0:
        return 0

    sampleFramePath = frameOutputPath / sampleId
    sampleFramePath.mkdir(parents=True, exist_ok=True)

    frameStep = max(int(round(fps * frameIntervalSeconds)), 1)
    extractedFrameCount = 0

    for frameIndex in range(0, int(frameCount), frameStep):
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        success, frame = videoCapture.read()

        if not success:
            continue

        framePath = sampleFramePath / f"{sampleId}_frame{frameIndex:06d}.jpg"
        cv2.imwrite(str(framePath), frame)
        extractedFrameCount += 1

    return extractedFrameCount


def inspectVideo(row):
    samplePath = projectRoot / row["samplePath"]
    videoCapture = cv2.VideoCapture(str(samplePath))

    if not videoCapture.isOpened():
        print(f"Could not open video: {samplePath}")
        return {
            "sampleId": row["sampleId"],
            "fileName": row["fileName"],
            "samplePath": row["samplePath"],
            "fps": 0,
            "frameCount": 0,
            "durationSeconds": 0,
            "width": 0,
            "height": 0,
            "extractedFrameCount": 0,
        }

    fps = float(videoCapture.get(cv2.CAP_PROP_FPS))
    frameCount = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    durationSeconds = frameCount / fps if fps > 0 else 0

    extractedFrameCount = savePreviewFrames(
        videoCapture=videoCapture,
        sampleId=row["sampleId"],
        fps=fps,
        frameCount=frameCount,
    )

    videoCapture.release()

    return {
        "sampleId": row["sampleId"],
        "fileName": row["fileName"],
        "samplePath": row["samplePath"],
        "fps": round(fps, 3),
        "frameCount": frameCount,
        "durationSeconds": round(durationSeconds, 3),
        "width": width,
        "height": height,
        "extractedFrameCount": extractedFrameCount,
    }


def preprocessSample():
    frameOutputPath.mkdir(parents=True, exist_ok=True)
    preprocessManifestPath.parent.mkdir(parents=True, exist_ok=True)

    sampleManifest = readSampleManifest()
    rows = []

    for _, row in tqdm(sampleManifest.iterrows(), total=len(sampleManifest), desc="Preprocessing videos"):
        rows.append(inspectVideo(row))

    preprocessManifest = pd.DataFrame(
        rows,
        columns=[
            "sampleId",
            "fileName",
            "samplePath",
            "fps",
            "frameCount",
            "durationSeconds",
            "width",
            "height",
            "extractedFrameCount",
        ],
    )
    preprocessManifest.to_csv(preprocessManifestPath, index=False)

    print("Preprocessing complete.")
    print(f"Preview frames saved to: {frameOutputPath}")
    print(f"Preprocess manifest saved to: {preprocessManifestPath}")
    print(f"Manifest rows written: {len(preprocessManifest)}")


if __name__ == "__main__":
    preprocessSample()
