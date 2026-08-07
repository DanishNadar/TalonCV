import math
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


projectRoot = Path(__file__).resolve().parents[1]
sampleManifestPath = projectRoot / "data" / "processed" / "manifests" / "firstImpressionsSampleManifest.csv"
featuresManifestPath = projectRoot / "data" / "processed" / "manifests" / "baselineCvFeatures.csv"
framesPerSecondToProcess = 2


def loadMediaPipe():
    try:
        import mediapipe as mp
    except Exception as error:
        print("MediaPipe could not be imported.")
        print("This often happens on newer Python versions such as Python 3.14.")
        print("Use Python 3.12 for this step:")
        print("python3.12 -m venv .venv")
        print("source .venv/bin/activate")
        print("pip install -r requirements.txt")
        raise SystemExit(1) from error

    if not hasattr(mp, "solutions"):
        print("MediaPipe imported, but the expected MediaPipe Solutions API is not available.")
        print("This is happening in the current Python environment and is common with Python 3.14.")
        print("Use Python 3.12 for this baseline script:")
        print("python3.12 -m venv .venv")
        print("source .venv/bin/activate")
        print("pip install -r requirements.txt")
        print("python scripts/extractBaselineCvFeatures.py")
        raise SystemExit(1)

    return mp


def readSampleManifest():
    if not sampleManifestPath.exists():
        print(f"Sample manifest does not exist yet: {sampleManifestPath}")
        print("Run scripts/createFirstImpressionsSample.py first.")
        raise SystemExit(1)

    return pd.read_csv(sampleManifestPath)


def emptyFeatureRow(sampleId, fileName, timestampSeconds):
    return {
        "sampleId": sampleId,
        "fileName": fileName,
        "timestampSeconds": round(timestampSeconds, 3),
        "faceDetected": False,
        "poseDetected": False,
        "faceCenterX": None,
        "faceCenterY": None,
        "faceWidth": None,
        "faceHeight": None,
        "shoulderLeftX": None,
        "shoulderLeftY": None,
        "shoulderRightX": None,
        "shoulderRightY": None,
        "noseX": None,
        "noseY": None,
        "headMovementProxy": None,
        "postureProxy": None,
        "cameraFacingProxy": None,
    }


def getFaceBox(faceDetection):
    boundingBox = faceDetection.location_data.relative_bounding_box
    faceCenterX = boundingBox.xmin + boundingBox.width / 2
    faceCenterY = boundingBox.ymin + boundingBox.height / 2

    return {
        "faceCenterX": round(faceCenterX, 5),
        "faceCenterY": round(faceCenterY, 5),
        "faceWidth": round(boundingBox.width, 5),
        "faceHeight": round(boundingBox.height, 5),
    }


def getPosePoints(poseResults, poseLandmark):
    if not poseResults.pose_landmarks:
        return {}

    landmarks = poseResults.pose_landmarks.landmark
    leftShoulder = landmarks[poseLandmark.LEFT_SHOULDER.value]
    rightShoulder = landmarks[poseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[poseLandmark.NOSE.value]

    return {
        "shoulderLeftX": round(leftShoulder.x, 5),
        "shoulderLeftY": round(leftShoulder.y, 5),
        "shoulderRightX": round(rightShoulder.x, 5),
        "shoulderRightY": round(rightShoulder.y, 5),
        "noseX": round(nose.x, 5),
        "noseY": round(nose.y, 5),
    }


def distanceBetweenPoints(pointA, pointB):
    if None in pointA or None in pointB:
        return None

    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)


def calculatePostureProxy(featureRow):
    leftX = featureRow["shoulderLeftX"]
    leftY = featureRow["shoulderLeftY"]
    rightX = featureRow["shoulderRightX"]
    rightY = featureRow["shoulderRightY"]

    if None in (leftX, leftY, rightX, rightY):
        return None

    shoulderWidth = abs(rightX - leftX)
    shoulderTilt = abs(rightY - leftY)

    if shoulderWidth == 0:
        return None

    return round(shoulderTilt / shoulderWidth, 5)


def calculateCameraFacingProxy(featureRow):
    faceCenterX = featureRow["faceCenterX"]
    noseX = featureRow["noseX"]
    faceWidth = featureRow["faceWidth"]

    if None in (faceCenterX, noseX, faceWidth) or faceWidth == 0:
        return None

    noseOffset = abs(noseX - faceCenterX)
    return round(max(0, 1 - noseOffset / (faceWidth / 2)), 5)


def processFrame(frame, sampleId, fileName, timestampSeconds, faceDetector, poseDetector, poseLandmark):
    featureRow = emptyFeatureRow(sampleId, fileName, timestampSeconds)
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faceResults = faceDetector.process(rgbFrame)
    if faceResults.detections:
        featureRow["faceDetected"] = True
        featureRow.update(getFaceBox(faceResults.detections[0]))

    poseResults = poseDetector.process(rgbFrame)
    if poseResults.pose_landmarks:
        featureRow["poseDetected"] = True
        featureRow.update(getPosePoints(poseResults, poseLandmark))

    featureRow["postureProxy"] = calculatePostureProxy(featureRow)
    featureRow["cameraFacingProxy"] = calculateCameraFacingProxy(featureRow)

    return featureRow


def addHeadMovement(rows):
    previousNosePointBySample = {}

    for row in rows:
        sampleId = row["sampleId"]
        nosePoint = (row["noseX"], row["noseY"])
        previousNosePoint = previousNosePointBySample.get(sampleId)

        row["headMovementProxy"] = distanceBetweenPoints(previousNosePoint, nosePoint) if previousNosePoint else None

        if None not in nosePoint:
            previousNosePointBySample[sampleId] = nosePoint

    return rows


def processVideo(manifestRow, faceDetector, poseDetector, poseLandmark):
    sampleId = manifestRow["sampleId"]
    fileName = manifestRow["fileName"]
    samplePath = projectRoot / manifestRow["samplePath"]
    videoCapture = cv2.VideoCapture(str(samplePath))

    if not videoCapture.isOpened():
        print(f"Could not open video: {samplePath}")
        return []

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frameCount = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    frameStep = max(int(round(fps / framesPerSecondToProcess)), 1) if fps > 0 else 1
    rows = []

    for frameIndex in range(0, frameCount, frameStep):
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
        success, frame = videoCapture.read()

        if not success:
            continue

        timestampSeconds = frameIndex / fps if fps > 0 else 0
        rows.append(
            processFrame(
                frame=frame,
                sampleId=sampleId,
                fileName=fileName,
                timestampSeconds=timestampSeconds,
                faceDetector=faceDetector,
                poseDetector=poseDetector,
                poseLandmark=poseLandmark,
            )
        )

    videoCapture.release()
    return rows


def extractBaselineFeatures():
    mp = loadMediaPipe()
    sampleManifest = readSampleManifest()
    featuresManifestPath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading sample manifest: {sampleManifestPath}")
    print(f"Processing {len(sampleManifest)} sampled videos at {framesPerSecondToProcess} fps.")

    rows = []
    poseLandmark = mp.solutions.pose.PoseLandmark

    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as faceDetector:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as poseDetector:
            for _, manifestRow in tqdm(sampleManifest.iterrows(), total=len(sampleManifest), desc="Extracting CV features"):
                rows.extend(processVideo(manifestRow, faceDetector, poseDetector, poseLandmark))

    rows = addHeadMovement(rows)
    features = pd.DataFrame(
        rows,
        columns=[
            "sampleId",
            "fileName",
            "timestampSeconds",
            "faceDetected",
            "poseDetected",
            "faceCenterX",
            "faceCenterY",
            "faceWidth",
            "faceHeight",
            "shoulderLeftX",
            "shoulderLeftY",
            "shoulderRightX",
            "shoulderRightY",
            "noseX",
            "noseY",
            "headMovementProxy",
            "postureProxy",
            "cameraFacingProxy",
        ],
    )
    features.to_csv(featuresManifestPath, index=False)

    print("Baseline CV feature extraction complete.")
    print(f"Rows written: {len(features)}")
    print(f"Feature CSV saved to: {featuresManifestPath}")


if __name__ == "__main__":
    print(f"Python version: {sys.version.split()[0]}")
    extractBaselineFeatures()
