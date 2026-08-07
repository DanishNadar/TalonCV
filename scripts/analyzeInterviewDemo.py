import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.cvPipeline.cueDefinitions import getCueInfo, getEventDescription
from src.cvPipeline.cueClassifier import applyCueClassifier
from src.cvPipeline.cueRules import buildCueCalibration, getFrameLabels
from src.cvPipeline.reportUtils import buildReviewReport
from src.cvPipeline.yoloFaceDetector import loadYoloFaceDetector


featureOutputPath = projectRoot / "data" / "demo" / "features"
eventOutputPath = projectRoot / "data" / "demo" / "events"
reportOutputPath = projectRoot / "reports"
framesPerSecondToAnalyze = 3


def loadMediaPipe():
    try:
        import mediapipe as mp
    except Exception as error:
        print("MediaPipe could not be imported.")
        print("Use the Python 3.12 environment from the README, then run this script again.")
        raise SystemExit(1) from error

    if not hasattr(mp, "solutions"):
        print("MediaPipe imported, but the expected MediaPipe Solutions API is not available.")
        print("Use Python 3.12 with mediapipe==0.10.21 from requirements.txt.")
        raise SystemExit(1)

    return mp


def getVideoPath():
    if len(sys.argv) < 2:
        print("Please provide a video path.")
        print("Example:")
        print("python scripts/analyzeInterviewDemo.py data/demo/recordings/YOUR_VIDEO_FILE.mp4")
        raise SystemExit(1)

    videoPath = Path(sys.argv[1])

    if not videoPath.is_absolute():
        videoPath = projectRoot / videoPath

    if not videoPath.exists():
        print(f"Video file does not exist: {videoPath}")
        raise SystemExit(1)

    return videoPath


def getOutputStem(videoPath):
    return videoPath.stem


def emptyFeatureRow(timestampSeconds):
    return {
        "timestampSeconds": round(timestampSeconds, 3),
        "faceDetected": False,
        "faceCount": 0,
        "faceDetectionSource": None,
        "faceDetectionModel": None,
        "faceDetectionConfidence": None,
        "faceMeshDetected": False,
        "poseDetected": False,
        "faceCenterX": None,
        "faceCenterY": None,
        "faceWidth": None,
        "faceHeight": None,
        "faceAreaProxy": None,
        "faceEdgeMarginProxy": None,
        "noseX": None,
        "noseY": None,
        "shoulderLeftX": None,
        "shoulderLeftY": None,
        "shoulderRightX": None,
        "shoulderRightY": None,
        "elbowLeftX": None,
        "elbowLeftY": None,
        "elbowRightX": None,
        "elbowRightY": None,
        "wristLeftX": None,
        "wristLeftY": None,
        "wristRightX": None,
        "wristRightY": None,
        "hipLeftX": None,
        "hipLeftY": None,
        "hipRightX": None,
        "hipRightY": None,
        "poseVisibilityProxy": None,
        "wristLeftVisibility": None,
        "wristRightVisibility": None,
        "headMovementProxy": None,
        "headHorizontalChangeProxy": None,
        "postureProxy": None,
        "bodyCenterOffsetProxy": None,
        "bodyLeanProxy": None,
        "handRaisedCount": None,
        "handMovementProxy": None,
        "cameraFacingProxy": None,
        "faceNoseOffsetXProxy": None,
        "mouthWidthProxy": None,
        "mouthOpennessProxy": None,
        "mouthMovementProxy": None,
        "mouthCornerLiftProxy": None,
        "eyebrowRaiseProxy": None,
        "eyeOpennessProxy": None,
        "eyeBalanceProxy": None,
        "headTiltProxy": None,
        "facialMovementProxy": None,
        "blinkLikeChangeProxy": None,
        "brightnessProxy": None,
        "contrastProxy": None,
        "sharpnessProxy": None,
        "expressionSpikeCount": 0,
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
    leftElbow = landmarks[poseLandmark.LEFT_ELBOW.value]
    rightElbow = landmarks[poseLandmark.RIGHT_ELBOW.value]
    leftWrist = landmarks[poseLandmark.LEFT_WRIST.value]
    rightWrist = landmarks[poseLandmark.RIGHT_WRIST.value]
    leftHip = landmarks[poseLandmark.LEFT_HIP.value]
    rightHip = landmarks[poseLandmark.RIGHT_HIP.value]
    nose = landmarks[poseLandmark.NOSE.value]

    return {
        "noseX": round(nose.x, 5),
        "noseY": round(nose.y, 5),
        "shoulderLeftX": round(leftShoulder.x, 5),
        "shoulderLeftY": round(leftShoulder.y, 5),
        "shoulderRightX": round(rightShoulder.x, 5),
        "shoulderRightY": round(rightShoulder.y, 5),
        "elbowLeftX": round(leftElbow.x, 5),
        "elbowLeftY": round(leftElbow.y, 5),
        "elbowRightX": round(rightElbow.x, 5),
        "elbowRightY": round(rightElbow.y, 5),
        "wristLeftX": round(leftWrist.x, 5),
        "wristLeftY": round(leftWrist.y, 5),
        "wristRightX": round(rightWrist.x, 5),
        "wristRightY": round(rightWrist.y, 5),
        "hipLeftX": round(leftHip.x, 5),
        "hipLeftY": round(leftHip.y, 5),
        "hipRightX": round(rightHip.x, 5),
        "hipRightY": round(rightHip.y, 5),
        "poseVisibilityProxy": round(
            min(leftShoulder.visibility, rightShoulder.visibility, leftHip.visibility, rightHip.visibility), 5
        ),
        "wristLeftVisibility": round(leftWrist.visibility, 5),
        "wristRightVisibility": round(rightWrist.visibility, 5),
    }


def getLandmarkPoint(landmarks, index):
    landmark = landmarks[index]
    return landmark.x, landmark.y


def verticalDistance(landmarks, topIndex, bottomIndex):
    top = landmarks[topIndex]
    bottom = landmarks[bottomIndex]
    return abs(bottom.y - top.y)


def getFaceMeshFeatures(faceMeshResults, featureRow):
    if not faceMeshResults.multi_face_landmarks:
        return {}

    faceHeight = featureRow["faceHeight"]
    faceWidth = featureRow["faceWidth"]

    if faceHeight in (None, 0) or faceWidth in (None, 0):
        return {}

    landmarks = faceMeshResults.multi_face_landmarks[0].landmark
    mouthLeft = getLandmarkPoint(landmarks, 61)
    mouthRight = getLandmarkPoint(landmarks, 291)
    upperLip = getLandmarkPoint(landmarks, 13)
    lowerLip = getLandmarkPoint(landmarks, 14)
    mouthCenterY = (upperLip[1] + lowerLip[1]) / 2
    mouthCornerY = (mouthLeft[1] + mouthRight[1]) / 2
    mouthWidth = abs(mouthRight[0] - mouthLeft[0])
    mouthOpenness = abs(lowerLip[1] - upperLip[1])

    leftEyeOpenness = verticalDistance(landmarks, 159, 145)
    rightEyeOpenness = verticalDistance(landmarks, 386, 374)
    eyeOpenness = (leftEyeOpenness + rightEyeOpenness) / 2
    eyeBalance = 1 - min(
        abs(leftEyeOpenness - rightEyeOpenness) / max(leftEyeOpenness, rightEyeOpenness, 0.00001),
        1,
    )

    leftBrowRaise = landmarks[159].y - landmarks[105].y
    rightBrowRaise = landmarks[386].y - landmarks[334].y
    eyebrowRaise = (leftBrowRaise + rightBrowRaise) / 2
    leftEyeOuter = getLandmarkPoint(landmarks, 33)
    rightEyeOuter = getLandmarkPoint(landmarks, 263)
    eyeLineWidth = abs(rightEyeOuter[0] - leftEyeOuter[0])
    headTilt = abs(rightEyeOuter[1] - leftEyeOuter[1]) / max(eyeLineWidth, 0.00001)

    return {
        "faceMeshDetected": True,
        "mouthWidthProxy": round(mouthWidth / faceWidth, 5),
        "mouthOpennessProxy": round(mouthOpenness / faceHeight, 5),
        "mouthCornerLiftProxy": round((mouthCenterY - mouthCornerY) / faceHeight, 5),
        "eyebrowRaiseProxy": round(eyebrowRaise / faceHeight, 5),
        "eyeOpennessProxy": round(eyeOpenness / faceHeight, 5),
        "eyeBalanceProxy": round(max(0, eyeBalance), 5),
        "headTiltProxy": round(headTilt, 5),
    }


def distanceBetweenPoints(pointA, pointB):
    if pointA is None or pointB is None:
        return None
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


def calculateSceneFeatures(frame):
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(grayFrame.mean()) / 255.0
    contrast = float(grayFrame.std()) / 128.0
    sharpness = min(float(cv2.Laplacian(grayFrame, cv2.CV_64F).var()) / 1000.0, 1.0)
    return {
        "brightnessProxy": round(brightness, 5),
        "contrastProxy": round(contrast, 5),
        "sharpnessProxy": round(sharpness, 5),
    }


def calculateBodyFeatures(featureRow):
    leftShoulderX = featureRow["shoulderLeftX"]
    leftShoulderY = featureRow["shoulderLeftY"]
    rightShoulderX = featureRow["shoulderRightX"]
    rightShoulderY = featureRow["shoulderRightY"]
    leftHipX = featureRow["hipLeftX"]
    rightHipX = featureRow["hipRightX"]

    if None in (leftShoulderX, leftShoulderY, rightShoulderX, rightShoulderY):
        return {}

    shoulderCenterX = (leftShoulderX + rightShoulderX) / 2
    shoulderWidth = abs(rightShoulderX - leftShoulderX)
    features = {"bodyCenterOffsetProxy": round(abs(shoulderCenterX - 0.5), 5)}

    if shoulderWidth > 0 and None not in (leftHipX, rightHipX) and (featureRow["poseVisibilityProxy"] or 0) >= 0.45:
        hipCenterX = (leftHipX + rightHipX) / 2
        features["bodyLeanProxy"] = round(abs(shoulderCenterX - hipCenterX) / shoulderWidth, 5)

    raisedCount = 0
    if (
        featureRow["wristLeftY"] is not None
        and (featureRow["wristLeftVisibility"] or 0) >= 0.45
        and featureRow["wristLeftY"] < leftShoulderY
    ):
        raisedCount += 1
    if (
        featureRow["wristRightY"] is not None
        and (featureRow["wristRightVisibility"] or 0) >= 0.45
        and featureRow["wristRightY"] < rightShoulderY
    ):
        raisedCount += 1
    features["handRaisedCount"] = raisedCount
    return features


def processFrame(frame, timestampSeconds, yoloFaceDetector, faceMeshDetector, poseDetector, poseLandmark):
    featureRow = emptyFeatureRow(timestampSeconds)
    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    featureRow.update(yoloFaceDetector.detect(frame))
    featureRow.update(calculateSceneFeatures(frame))

    faceMeshResults = faceMeshDetector.process(rgbFrame)
    featureRow.update(getFaceMeshFeatures(faceMeshResults, featureRow))

    poseResults = poseDetector.process(rgbFrame)
    if poseResults.pose_landmarks:
        featureRow["poseDetected"] = True
        featureRow.update(getPosePoints(poseResults, poseLandmark))

    featureRow["postureProxy"] = calculatePostureProxy(featureRow)
    featureRow["cameraFacingProxy"] = calculateCameraFacingProxy(featureRow)
    featureRow.update(calculateBodyFeatures(featureRow))

    if None not in (featureRow["noseX"], featureRow["faceCenterX"], featureRow["faceWidth"]) and featureRow["faceWidth"]:
        featureRow["faceNoseOffsetXProxy"] = round(
            (featureRow["noseX"] - featureRow["faceCenterX"]) / (featureRow["faceWidth"] / 2), 5
        )

    return featureRow


def addMovementFeatures(rows):
    previousNosePoint = None
    previousNoseY = None
    previousNoseX = None
    previousPostureProxy = None
    previousMouthCornerLift = None
    previousEyebrowRaise = None
    previousEyeOpenness = None
    previousMouthOpenness = None
    previousLeftWristPoint = None
    previousRightWristPoint = None
    for row in rows:
        nosePoint = (row["noseX"], row["noseY"])
        row["headMovementProxy"] = distanceBetweenPoints(previousNosePoint, nosePoint)

        if None not in nosePoint:
            previousNosePoint = nosePoint

        if row["noseY"] is not None and previousNoseY is not None:
            row["noseYChangeProxy"] = abs(row["noseY"] - previousNoseY)
        else:
            row["noseYChangeProxy"] = None

        if row["noseY"] is not None:
            previousNoseY = row["noseY"]

        if row["noseX"] is not None and previousNoseX is not None:
            row["headHorizontalChangeProxy"] = abs(row["noseX"] - previousNoseX)
        else:
            row["headHorizontalChangeProxy"] = None
        if row["noseX"] is not None:
            previousNoseX = row["noseX"]

        if row["postureProxy"] is not None and previousPostureProxy is not None:
            row["postureChangeProxy"] = abs(row["postureProxy"] - previousPostureProxy)
        else:
            row["postureChangeProxy"] = None

        if row["postureProxy"] is not None:
            previousPostureProxy = row["postureProxy"]

        expressionChanges = []

        mouthCornerLift = row.get("mouthCornerLiftProxy")
        if mouthCornerLift is not None and previousMouthCornerLift is not None:
            expressionChanges.append(abs(mouthCornerLift - previousMouthCornerLift))
        if mouthCornerLift is not None:
            previousMouthCornerLift = mouthCornerLift

        eyebrowRaise = row.get("eyebrowRaiseProxy")
        if eyebrowRaise is not None and previousEyebrowRaise is not None:
            expressionChanges.append(abs(eyebrowRaise - previousEyebrowRaise))
        if eyebrowRaise is not None:
            previousEyebrowRaise = eyebrowRaise

        eyeOpenness = row.get("eyeOpennessProxy")
        if eyeOpenness is not None and previousEyeOpenness is not None:
            row["blinkLikeChangeProxy"] = abs(eyeOpenness - previousEyeOpenness)
            expressionChanges.append(row["blinkLikeChangeProxy"])
        else:
            row["blinkLikeChangeProxy"] = None
        if eyeOpenness is not None:
            previousEyeOpenness = eyeOpenness

        mouthOpenness = row.get("mouthOpennessProxy")
        if mouthOpenness is not None and previousMouthOpenness is not None:
            row["mouthMovementProxy"] = abs(mouthOpenness - previousMouthOpenness)
            expressionChanges.append(row["mouthMovementProxy"])
        else:
            row["mouthMovementProxy"] = None
        if mouthOpenness is not None:
            previousMouthOpenness = mouthOpenness

        leftWristPoint = (
            (row.get("wristLeftX"), row.get("wristLeftY"))
            if (row.get("wristLeftVisibility") or 0) >= 0.45
            else (None, None)
        )
        rightWristPoint = (
            (row.get("wristRightX"), row.get("wristRightY"))
            if (row.get("wristRightVisibility") or 0) >= 0.45
            else (None, None)
        )
        wristChanges = [
            value
            for value in (
                distanceBetweenPoints(previousLeftWristPoint, leftWristPoint),
                distanceBetweenPoints(previousRightWristPoint, rightWristPoint),
            )
            if value is not None
        ]
        row["handMovementProxy"] = round(sum(wristChanges) / len(wristChanges), 5) if wristChanges else None
        if None not in leftWristPoint:
            previousLeftWristPoint = leftWristPoint
        if None not in rightWristPoint:
            previousRightWristPoint = rightWristPoint

        row["facialMovementProxy"] = round(sum(expressionChanges), 5) if expressionChanges else None

    return rows


def addAdaptiveSpikeCounts(rows):
    calibration = buildCueCalibration(rows)
    recentMovementSpikes = []
    recentExpressionSpikes = []

    for row in rows:
        hasMovementSpike = valueInAdaptiveBand(row.get("headMovementProxy"), calibration, "headMovementProxy", "high")
        hasMovementSpike = hasMovementSpike or valueInAdaptiveBand(
            row.get("postureChangeProxy"), calibration, "postureChangeProxy", "high"
        )
        hasMovementSpike = hasMovementSpike or valueInAdaptiveBand(
            row.get("handMovementProxy"), calibration, "handMovementProxy", "high"
        )

        hasExpressionSpike = valueInAdaptiveBand(
            row.get("facialMovementProxy"), calibration, "facialMovementProxy", "high"
        )
        hasExpressionSpike = hasExpressionSpike or valueInAdaptiveBand(
            row.get("blinkLikeChangeProxy"), calibration, "blinkLikeChangeProxy", "high"
        )

        recentMovementSpikes.append(1 if hasMovementSpike else 0)
        recentMovementSpikes = recentMovementSpikes[-5:]
        row["movementSpikeCount"] = sum(recentMovementSpikes)

        recentExpressionSpikes.append(1 if hasExpressionSpike else 0)
        recentExpressionSpikes = recentExpressionSpikes[-5:]
        row["expressionSpikeCount"] = sum(recentExpressionSpikes)

    return rows


def valueInAdaptiveBand(value, calibration, key, band):
    threshold = calibration.get(key, {}).get(band)
    return value is not None and threshold is not None and value >= threshold


def addFrameLabels(rows, includeLearnedCues=True, cueClassifier=None):
    calibration = buildCueCalibration(rows)
    for row in rows:
        ruleLabels = getFrameLabels(row, calibration)
        row["ruleFrameLabels"] = ",".join(ruleLabels)
        row["frameLabels"] = row["ruleFrameLabels"]
        row["frameLabelSources"] = json.dumps({label: ["rule"] for label in ruleLabels}, sort_keys=True)

    if includeLearnedCues:
        rows = applyCueClassifier(rows, classifier=cueClassifier)
    else:
        for row in rows:
            row["mlCueCandidate"] = None
            row["mlCueLabel"] = None
            row["mlCueApplied"] = False
            row["mlCueConfidence"] = None
            row["mlCueProbabilities"] = None

    return rows


def analyzeVideo(
    videoPath,
    mp,
    yoloFaceDetector=None,
    includeLearnedCues=True,
    cueClassifier=None,
    analysisFps=None,
):
    videoCapture = cv2.VideoCapture(str(videoPath))

    if not videoCapture.isOpened():
        raise SystemExit(f"Could not open video: {videoPath}")

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frameCount = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frameCount <= 0:
        videoCapture.release()
        raise SystemExit(
            f"No frames could be read from this video: {videoPath}. "
            "It may be empty, still finishing a webcam recording, or use an unsupported codec."
        )

    configuredFps = float(analysisFps or framesPerSecondToAnalyze)
    configuredFps = max(0.5, min(configuredFps, 10.0))
    frameStep = max(int(round(fps / configuredFps)), 1) if fps > 0 else 1
    rows = []
    poseLandmark = mp.solutions.pose.PoseLandmark
    yoloFaceDetector = yoloFaceDetector or loadYoloFaceDetector()

    print(f"Analyzing video: {videoPath}")
    print(f"Processing at {configuredFps:g} fps.")
    print(f"Face detection: YOLOv11 ({yoloFaceDetector.model_source})")

    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as faceMeshDetector:
        with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as poseDetector:
            for frameIndex in tqdm(range(0, frameCount, frameStep), desc="Analyzing frames"):
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                success, frame = videoCapture.read()

                if not success:
                    continue

                timestampSeconds = frameIndex / fps if fps > 0 else 0
                rows.append(processFrame(frame, timestampSeconds, yoloFaceDetector, faceMeshDetector, poseDetector, poseLandmark))

    videoCapture.release()

    if not rows:
        raise SystemExit(
            f"The video container reported {frameCount} frames but none could be decoded: {videoPath}. "
            "The file is likely corrupt or was still being written."
        )

    rows = addMovementFeatures(rows)
    rows = addAdaptiveSpikeCounts(rows)
    rows = addFrameLabels(rows, includeLearnedCues=includeLearnedCues, cueClassifier=cueClassifier)
    durationSeconds = frameCount / fps if fps > 0 else 0

    return rows, durationSeconds


def closeEvent(events, eventType, startTime, endTime, detectionSources=None, mlConfidences=None):
    durationSeconds = max(0, endTime - startTime)

    if durationSeconds < 0.3:
        return

    cueInfo = getCueInfo(eventType)

    event = {
        "eventType": eventType,
        "cue": cueInfo["cue"],
        "startTime": round(startTime, 3),
        "endTime": round(endTime, 3),
        "durationSeconds": round(durationSeconds, 3),
        "description": getEventDescription(eventType),
        "semanticPurpose": cueInfo["semanticPurpose"],
        "detectionSources": sorted(detectionSources or {"rule"}),
    }
    if mlConfidences:
        event["mlConfidenceMean"] = round(sum(mlConfidences) / len(mlConfidences), 5)
    events.append(event)


def getLabelSources(row, eventType):
    sourceText = row.get("frameLabelSources")
    if sourceText:
        try:
            parsed = json.loads(sourceText) if isinstance(sourceText, str) else sourceText
            sources = parsed.get(eventType, [])
            if sources:
                return set(sources)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    return {"rule"}


def createEvents(rows, durationSeconds):
    activeEvents = {}
    events = []
    previousTimestamp = None
    fallbackCalibration = None

    for row in rows:
        timestampSeconds = row["timestampSeconds"]
        frameLabels = row.get("frameLabels")
        frameLabelsText = "" if frameLabels is None else str(frameLabels).strip()
        if frameLabelsText and frameLabelsText.lower() != "nan":
            currentEventTypes = {
                eventType.strip()
                for eventType in frameLabelsText.split(",")
                if eventType.strip()
            }
        else:
            if fallbackCalibration is None:
                fallbackCalibration = buildCueCalibration(rows)
            currentEventTypes = set(getFrameLabels(row, fallbackCalibration))

        for eventType in currentEventTypes:
            if eventType not in activeEvents:
                activeEvents[eventType] = {
                    "startTime": timestampSeconds,
                    "sources": set(),
                    "mlConfidences": [],
                }
            activeEvents[eventType]["sources"].update(getLabelSources(row, eventType))
            if row.get("mlCueLabel") == eventType and row.get("mlCueConfidence") is not None:
                activeEvents[eventType]["mlConfidences"].append(float(row["mlCueConfidence"]))

        for eventType in list(activeEvents.keys()):
            if eventType not in currentEventTypes:
                activeEvent = activeEvents.pop(eventType)
                closeEvent(
                    events,
                    eventType,
                    activeEvent["startTime"],
                    previousTimestamp or timestampSeconds,
                    activeEvent["sources"],
                    activeEvent["mlConfidences"],
                )

        previousTimestamp = timestampSeconds

    if previousTimestamp is not None:
        for eventType, activeEvent in activeEvents.items():
            closeEvent(
                events,
                eventType,
                activeEvent["startTime"],
                durationSeconds or previousTimestamp,
                activeEvent["sources"],
                activeEvent["mlConfidences"],
            )

    return sorted(events, key=lambda event: (event["startTime"], event["eventType"]))


def saveFeatures(rows, outputStem):
    featureOutputPath.mkdir(parents=True, exist_ok=True)
    featurePath = featureOutputPath / f"{outputStem}_features.csv"
    features = pd.DataFrame(
        rows,
        columns=[
            "timestampSeconds",
            "faceDetected",
            "faceCount",
            "faceDetectionSource",
            "faceDetectionModel",
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
            "postureProxy",
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
            "postureChangeProxy",
            "noseYChangeProxy",
            "movementSpikeCount",
            "expressionSpikeCount",
            "brightnessProxy",
            "contrastProxy",
            "sharpnessProxy",
            "ruleFrameLabels",
            "mlCueCandidate",
            "mlCueLabel",
            "mlCueApplied",
            "mlCueConfidence",
            "mlCueProbabilities",
            "frameLabelSources",
            "frameLabels",
        ],
    )
    features.to_csv(featurePath, index=False)
    return featurePath


def saveEvents(events, outputStem):
    eventOutputPath.mkdir(parents=True, exist_ok=True)
    eventPath = eventOutputPath / f"{outputStem}_events.json"

    with eventPath.open("w") as eventFile:
        json.dump(events, eventFile, indent=2)

    return eventPath


def saveReport(videoPath, rows, events, durationSeconds, outputStem):
    reportOutputPath.mkdir(parents=True, exist_ok=True)
    reportPath = reportOutputPath / f"{outputStem}_review.md"
    reportPath.write_text(buildReviewReport(videoPath, rows, events, durationSeconds))
    return reportPath


def analyzeInterviewDemo():
    videoPath = getVideoPath()
    outputStem = getOutputStem(videoPath)
    mp = loadMediaPipe()

    rows, durationSeconds = analyzeVideo(videoPath, mp)
    events = createEvents(rows, durationSeconds)

    featurePath = saveFeatures(rows, outputStem)
    eventPath = saveEvents(events, outputStem)
    reportPath = saveReport(videoPath, rows, events, durationSeconds, outputStem)

    print("Interview demo analysis complete.")
    print(f"Feature CSV saved to: {featurePath}")
    print(f"Event JSON saved to: {eventPath}")
    print(f"Review report saved to: {reportPath}")


if __name__ == "__main__":
    analyzeInterviewDemo()
