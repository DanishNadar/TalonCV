import math


def hasValue(value):
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def numericValues(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def percentile(values, percentile_value):
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * percentile_value
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def buildCueCalibration(rows):
    calibration = {}
    numeric_keys = {
        "cameraFacingProxy",
        "faceDetectionConfidence",
        "faceAreaProxy",
        "faceEdgeMarginProxy",
        "eyeBalanceProxy",
        "eyeOpennessProxy",
        "headMovementProxy",
        "headHorizontalChangeProxy",
        "headTiltProxy",
        "postureProxy",
        "postureChangeProxy",
        "bodyCenterOffsetProxy",
        "bodyLeanProxy",
        "handMovementProxy",
        "noseYChangeProxy",
        "mouthWidthProxy",
        "mouthOpennessProxy",
        "mouthMovementProxy",
        "mouthCornerLiftProxy",
        "eyebrowRaiseProxy",
        "facialMovementProxy",
        "blinkLikeChangeProxy",
        "movementSpikeCount",
        "expressionSpikeCount",
        "brightnessProxy",
        "contrastProxy",
        "sharpnessProxy",
    }

    for key in numeric_keys:
        values = numericValues(rows, key)
        calibration[key] = {
            "min": min(values) if values else None,
            "low": percentile(values, 0.25),
            "mid": percentile(values, 0.50),
            "high": percentile(values, 0.75),
            "veryHigh": percentile(values, 0.90),
            "max": max(values) if values else None,
        }

    downward_offsets = []
    expression_scores = []
    for row in rows:
        nose_y = row.get("noseY")
        face_center_y = row.get("faceCenterY")
        face_height = row.get("faceHeight")
        if all(hasValue(value) for value in (nose_y, face_center_y, face_height)) and face_height:
            downward_offsets.append((nose_y - face_center_y) / face_height)

        expression_parts = [
            row.get("mouthWidthProxy"),
            row.get("mouthCornerLiftProxy"),
            row.get("eyebrowRaiseProxy"),
        ]
        if any(hasValue(value) for value in expression_parts):
            expression_scores.append(sum(float(value or 0.0) for value in expression_parts))

    calibration["downwardFaceOffset"] = {
        "min": min(downward_offsets) if downward_offsets else None,
        "low": percentile(downward_offsets, 0.25),
        "mid": percentile(downward_offsets, 0.50),
        "high": percentile(downward_offsets, 0.75),
        "veryHigh": percentile(downward_offsets, 0.90),
        "max": max(downward_offsets) if downward_offsets else None,
    }
    calibration["expressionScore"] = {
        "min": min(expression_scores) if expression_scores else None,
        "low": percentile(expression_scores, 0.25),
        "mid": percentile(expression_scores, 0.50),
        "high": percentile(expression_scores, 0.75),
        "veryHigh": percentile(expression_scores, 0.90),
        "max": max(expression_scores) if expression_scores else None,
    }

    return calibration


def adaptiveValue(calibration, key, band):
    return calibration.get(key, {}).get(band)


def atLeast(value, calibration, key, band):
    threshold = adaptiveValue(calibration, key, band)
    value_range = calibration.get(key, {})
    if band in {"high", "veryHigh"} and value_range.get("min") == value_range.get("max"):
        return False
    if band in {"high", "veryHigh"}:
        return hasValue(value) and threshold is not None and value > threshold
    return hasValue(value) and threshold is not None and value >= threshold


def atMost(value, calibration, key, band):
    threshold = adaptiveValue(calibration, key, band)
    return hasValue(value) and threshold is not None and value <= threshold


def getFrameLabels(row, calibration=None):
    calibration = calibration or {}
    labels = []
    cameraFacingProxy = row.get("cameraFacingProxy")
    headMovementProxy = row.get("headMovementProxy")
    postureProxy = row.get("postureProxy")
    postureChangeProxy = row.get("postureChangeProxy")
    noseY = row.get("noseY")
    faceCenterY = row.get("faceCenterY")
    faceHeight = row.get("faceHeight")
    noseYChangeProxy = row.get("noseYChangeProxy")
    movementSpikeCount = row.get("movementSpikeCount", 0)
    expressionSpikeCount = row.get("expressionSpikeCount", 0)
    facialMovementProxy = row.get("facialMovementProxy")
    blinkLikeChangeProxy = row.get("blinkLikeChangeProxy")
    eyeBalanceProxy = row.get("eyeBalanceProxy")
    faceCenterX = row.get("faceCenterX")
    faceCenterY = row.get("faceCenterY")
    faceHeight = row.get("faceHeight")

    brightnessProxy = row.get("brightnessProxy")
    contrastProxy = row.get("contrastProxy")
    sharpnessProxy = row.get("sharpnessProxy")
    if hasValue(brightnessProxy) and float(brightnessProxy) < 0.18:
        labels.append("dimLighting")
    if hasValue(brightnessProxy) and float(brightnessProxy) > 0.90:
        labels.append("overexposedLighting")
    if hasValue(contrastProxy) and float(contrastProxy) < 0.12:
        labels.append("lowContrast")
    if hasValue(sharpnessProxy) and float(sharpnessProxy) < 0.035:
        labels.append("blurryImage")

    if not row.get("faceDetected", False):
        labels.append("faceMissing")
    else:
        faceCount = row.get("faceCount", 1)
        faceConfidence = row.get("faceDetectionConfidence")
        faceEdgeMargin = row.get("faceEdgeMarginProxy")
        if hasValue(faceCount) and float(faceCount) > 1:
            labels.append("multipleFaces")
        if hasValue(faceConfidence) and float(faceConfidence) < 0.35:
            labels.append("lowFaceConfidence")
        if hasValue(faceEdgeMargin) and float(faceEdgeMargin) < 0.012:
            labels.append("facePartiallyOutOfFrame")
        if hasValue(faceHeight) and float(faceHeight) > 0.62:
            labels.append("faceTooClose")
        elif hasValue(faceHeight) and float(faceHeight) < 0.18:
            labels.append("faceTooFar")

        if all(hasValue(value) for value in (faceCenterX, faceCenterY)):
            horizontalOffset = abs(float(faceCenterX) - 0.5)
            verticalPosition = float(faceCenterY)
            if horizontalOffset <= 0.12 and 0.20 <= verticalPosition <= 0.58:
                labels.append("centeredFraming")
            elif horizontalOffset > 0.20 or verticalPosition < 0.12 or verticalPosition > 0.68:
                labels.append("offCenterFraming")

        if not row.get("faceMeshDetected", False):
            labels.append("faceMeshMissing")

        if hasValue(cameraFacingProxy):
            if isCameraFacingAttention(cameraFacingProxy, eyeBalanceProxy, calibration):
                labels.append("cameraFacing")
            elif atMost(cameraFacingProxy, calibration, "cameraFacingProxy", "low"):
                labels.append("lookingAway")

    if not row.get("poseDetected", False):
        labels.append("poseMissing")
    elif hasValue(postureProxy) and atMost(postureProxy, calibration, "postureProxy", "low"):
        labels.append("stablePosture")

    if atLeast(headMovementProxy, calibration, "headMovementProxy", "veryHigh"):
        labels.append("highHeadMovement")

    if atLeast(postureChangeProxy, calibration, "postureChangeProxy", "veryHigh"):
        labels.append("postureShift")

    if isLookingDown(noseY, faceCenterY, faceHeight, calibration):
        labels.append("lookingDown")

    if isNodding(noseYChangeProxy, headMovementProxy, calibration):
        labels.append("nodding")

    if isPositiveExpressionProxy(row, calibration):
        labels.append("positiveExpression")

    if isNeutralExpressionProxy(row, calibration):
        labels.append("neutralExpression")

    if isTensionLikeInstability(
        headMovementProxy,
        postureChangeProxy,
        facialMovementProxy,
        blinkLikeChangeProxy,
        movementSpikeCount,
        expressionSpikeCount,
        calibration,
    ):
        labels.append("tensionLikeInstability")

    if isPossibleFidgeting(headMovementProxy, postureChangeProxy, blinkLikeChangeProxy, movementSpikeCount, calibration):
        labels.append("possibleFidgeting")

    eyeOpennessProxy = row.get("eyeOpennessProxy")
    if hasValue(eyeOpennessProxy) and float(eyeOpennessProxy) < 0.018:
        labels.append("eyesClosedLike")
    if atLeast(blinkLikeChangeProxy, calibration, "blinkLikeChangeProxy", "veryHigh") and expressionSpikeCount >= 2:
        labels.append("rapidBlinkLikeActivity")

    if atLeast(row.get("eyebrowRaiseProxy"), calibration, "eyebrowRaiseProxy", "veryHigh"):
        labels.append("eyebrowRaise")
    if hasValue(row.get("mouthOpennessProxy")) and float(row["mouthOpennessProxy"]) > 0.055:
        labels.append("mouthOpen")
    if atLeast(row.get("mouthMovementProxy"), calibration, "mouthMovementProxy", "high"):
        labels.append("speechLikeMouthActivity")

    faceNoseOffset = row.get("faceNoseOffsetXProxy")
    if hasValue(faceNoseOffset) and float(faceNoseOffset) < -0.35:
        labels.append("headTurnedLeft")
    elif hasValue(faceNoseOffset) and float(faceNoseOffset) > 0.35:
        labels.append("headTurnedRight")
    if hasValue(row.get("headTiltProxy")) and float(row["headTiltProxy"]) > 0.12:
        labels.append("headTilt")
    if atLeast(row.get("headHorizontalChangeProxy"), calibration, "headHorizontalChangeProxy", "veryHigh"):
        labels.append("lateralHeadMovement")

    if hasValue(postureProxy) and float(postureProxy) > 0.15:
        labels.append("shoulderTilt")
    if hasValue(row.get("bodyLeanProxy")) and float(row["bodyLeanProxy"]) > 0.22:
        labels.append("bodyLean")
    if hasValue(row.get("bodyCenterOffsetProxy")) and float(row["bodyCenterOffsetProxy"]) > 0.18:
        labels.append("bodyOffCenter")
    if atLeast(row.get("handMovementProxy"), calibration, "handMovementProxy", "high"):
        labels.append("handGestureActivity")
    if hasValue(row.get("handRaisedCount")) and float(row["handRaisedCount"]) >= 1:
        labels.append("handsRaised")

    return list(dict.fromkeys(labels))


def isLookingDown(noseY, faceCenterY, faceHeight, calibration):
    if not all(hasValue(value) for value in (noseY, faceCenterY, faceHeight)) or not faceHeight:
        return False

    downward_offset = (noseY - faceCenterY) / faceHeight
    return atLeast(downward_offset, calibration, "downwardFaceOffset", "veryHigh")


def isCameraFacingAttention(cameraFacingProxy, eyeBalanceProxy, calibration):
    if not atLeast(cameraFacingProxy, calibration, "cameraFacingProxy", "high"):
        return False

    return not hasValue(eyeBalanceProxy) or atLeast(eyeBalanceProxy, calibration, "eyeBalanceProxy", "mid")


def isNodding(noseYChangeProxy, headMovementProxy, calibration):
    return atLeast(noseYChangeProxy, calibration, "noseYChangeProxy", "high") and atLeast(
        headMovementProxy, calibration, "headMovementProxy", "high"
    )


def isPositiveExpressionProxy(row, calibration):
    if not row.get("faceDetected", False):
        return False

    cameraFacingProxy = row.get("cameraFacingProxy")
    headMovementProxy = row.get("headMovementProxy")
    expression_score = sum(
        float(row.get(key) or 0.0)
        for key in ("mouthWidthProxy", "mouthCornerLiftProxy", "eyebrowRaiseProxy")
    )

    camera_looks_centered = atLeast(cameraFacingProxy, calibration, "cameraFacingProxy", "high")
    movement_looks_settled = not hasValue(headMovementProxy) or atMost(
        headMovementProxy, calibration, "headMovementProxy", "mid"
    )
    expression_stands_out = atLeast(expression_score, calibration, "expressionScore", "high")

    return camera_looks_centered and movement_looks_settled and expression_stands_out


def isNeutralExpressionProxy(row, calibration):
    if not row.get("faceDetected", False):
        return False

    cameraFacingProxy = row.get("cameraFacingProxy")
    headMovementProxy = row.get("headMovementProxy")
    postureChangeProxy = row.get("postureChangeProxy")
    facialMovementProxy = row.get("facialMovementProxy")
    expressionSpikeCount = row.get("expressionSpikeCount", 0)

    if not atLeast(cameraFacingProxy, calibration, "cameraFacingProxy", "high"):
        return False

    headLooksStable = not hasValue(headMovementProxy) or atMost(headMovementProxy, calibration, "headMovementProxy", "low")
    postureLooksStable = not hasValue(postureChangeProxy) or atMost(
        postureChangeProxy, calibration, "postureChangeProxy", "mid"
    )
    faceLooksStable = not hasValue(facialMovementProxy) or atMost(
        facialMovementProxy, calibration, "facialMovementProxy", "mid"
    )
    expressionLooksQuiet = not hasValue(expressionSpikeCount) or atMost(
        expressionSpikeCount, calibration, "expressionSpikeCount", "mid"
    )

    return headLooksStable and postureLooksStable and faceLooksStable and expressionLooksQuiet


def isTensionLikeInstability(
    headMovementProxy,
    postureChangeProxy,
    facialMovementProxy,
    blinkLikeChangeProxy,
    movementSpikeCount,
    expressionSpikeCount,
    calibration,
):
    headSpike = atLeast(headMovementProxy, calibration, "headMovementProxy", "veryHigh")
    postureSpike = atLeast(postureChangeProxy, calibration, "postureChangeProxy", "veryHigh")
    faceSpike = atLeast(facialMovementProxy, calibration, "facialMovementProxy", "veryHigh")
    blinkSpike = atLeast(blinkLikeChangeProxy, calibration, "blinkLikeChangeProxy", "veryHigh")
    movementCluster = atLeast(movementSpikeCount, calibration, "movementSpikeCount", "high")
    expressionCluster = atLeast(expressionSpikeCount, calibration, "expressionSpikeCount", "high")

    return movementCluster or expressionCluster or (headSpike and postureSpike) or (faceSpike and (headSpike or blinkSpike))


def isPossibleFidgeting(headMovementProxy, postureChangeProxy, blinkLikeChangeProxy, movementSpikeCount, calibration):
    headMovement = atLeast(headMovementProxy, calibration, "headMovementProxy", "high")
    postureMovement = atLeast(postureChangeProxy, calibration, "postureChangeProxy", "high")
    rapidBlinkLikeChange = atLeast(blinkLikeChangeProxy, calibration, "blinkLikeChangeProxy", "high")
    repeatedMovement = atLeast(movementSpikeCount, calibration, "movementSpikeCount", "veryHigh")

    return repeatedMovement or (headMovement and postureMovement) or (rapidBlinkLikeChange and headMovement)
