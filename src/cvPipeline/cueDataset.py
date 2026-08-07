import re
from pathlib import Path

from src.cvPipeline.cueDefinitions import eventCueMap


baselineCue = "baseline"
recordableCueTypes = tuple(sorted({*eventCueMap.keys(), baselineCue}))


def cueRecordingTarget(cue):
    """Return minimum/target independent takes for a training-ready cue class."""
    canonical = canonicalCueName(cue)
    if canonical is None:
        raise ValueError(f"Unknown cue '{cue}'.")
    subtleCues = {
        "eyebrowRaise",
        "eyesClosedLike",
        "mouthOpen",
        "neutralExpression",
        "positiveExpression",
        "rapidBlinkLikeActivity",
        "speechLikeMouthActivity",
        "tensionLikeInstability",
    }
    dynamicCues = {
        "handGestureActivity",
        "handsRaised",
        "highHeadMovement",
        "lateralHeadMovement",
        "nodding",
        "possibleFidgeting",
        "postureShift",
    }
    orientationCues = {
        "bodyLean",
        "bodyOffCenter",
        "cameraFacing",
        "headTilt",
        "headTurnedLeft",
        "headTurnedRight",
        "lookingAway",
        "lookingDown",
        "shoulderTilt",
        "stablePosture",
    }
    qualityAndFramingCues = {
        "blurryImage",
        "centeredFraming",
        "dimLighting",
        "faceMeshMissing",
        "faceMissing",
        "facePartiallyOutOfFrame",
        "faceTooClose",
        "faceTooFar",
        "lowContrast",
        "lowFaceConfidence",
        "multipleFaces",
        "offCenterFraming",
        "overexposedLighting",
        "poseMissing",
    }
    if canonical == baselineCue:
        minimum, target, tier = 60, 80, "baseline diversity"
    elif canonical in subtleCues:
        minimum, target, tier = 45, 60, "subtle or high-variation cue"
    elif canonical in dynamicCues:
        minimum, target, tier = 40, 50, "dynamic cue"
    elif canonical in orientationCues:
        minimum, target, tier = 36, 48, "orientation/posture cue"
    elif canonical in qualityAndFramingCues:
        minimum, target, tier = 30, 40, "YOLO/visibility/quality cue"
    else:
        minimum, target, tier = 36, 48, "general cue"
    return {"minimumTakes": minimum, "targetTakes": target, "tier": tier}


def cueSlug(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


cueBySlug = {cueSlug(cue): cue for cue in recordableCueTypes}


def canonicalCueName(value):
    return cueBySlug.get(cueSlug(value))


def safeCueFilePart(cue):
    canonical = canonicalCueName(cue)
    if canonical is None:
        raise ValueError(f"Unknown cue '{cue}'.")
    return canonical


def cueFromRecordingPath(videoPath):
    """Infer a cue from `<cue>__anything.mp4` or from a cue-named parent folder."""
    path = Path(videoPath)
    candidates = [path.stem.split("__", 1)[0]]

    for separator in ("_take", "-take", "_recording", "-recording"):
        if separator in path.stem.lower():
            candidates.append(path.stem[: path.stem.lower().index(separator)])

    candidates.extend(parent.name for parent in path.parents[:2])
    for candidate in candidates:
        canonical = canonicalCueName(candidate)
        if canonical:
            return canonical
    return None


def cueFilename(cue, takeNumber, timestamp):
    return f"{safeCueFilePart(cue)}__take_{takeNumber:03d}__{timestamp}.mp4"
