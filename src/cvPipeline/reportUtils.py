from collections import Counter, defaultdict

from src.cvPipeline.cueDefinitions import cueDefinitions, getCueInfo


strengthEventTypes = {
    "cameraFacing",
    "stablePosture",
    "positiveExpression",
    "neutralExpression",
    "centeredFraming",
}
reviewEventTypes = {
    "lookingAway",
    "tensionLikeInstability",
    "highHeadMovement",
    "nodding",
    "lookingDown",
    "postureShift",
    "possibleFidgeting",
    "faceMissing",
    "poseMissing",
    "offCenterFraming",
    "faceTooClose",
    "faceTooFar",
    "facePartiallyOutOfFrame",
    "multipleFaces",
    "lowFaceConfidence",
    "faceMeshMissing",
    "dimLighting",
    "overexposedLighting",
    "lowContrast",
    "blurryImage",
    "eyesClosedLike",
    "rapidBlinkLikeActivity",
    "eyebrowRaise",
    "mouthOpen",
    "speechLikeMouthActivity",
    "headTurnedLeft",
    "headTurnedRight",
    "headTilt",
    "lateralHeadMovement",
    "shoulderTilt",
    "bodyLean",
    "bodyOffCenter",
    "handGestureActivity",
    "handsRaised",
}


def buildReviewReport(videoPath, rows, events, durationSeconds):
    eventCounts = Counter(event["eventType"] for event in events)
    cueCounts = Counter(event["cue"] for event in events)
    eventsByCue = defaultdict(list)

    for event in events:
        eventsByCue[event["cue"]].append(event)

    lines = [
        f"# Interview Demo Review: {videoPath.name}",
        "",
        "## Safety Note",
        "",
        "This is an adaptive visual coaching review. It is not a judgment of personality, honesty, intelligence, professionalism, or hiring suitability.",
        "",
        "## Summary",
        "",
        f"- Video duration: {durationSeconds:.2f} seconds",
        f"- Total analyzed frames: {len(rows)}",
        "",
        "## Event Counts",
        "",
    ]

    if eventCounts:
        for eventType, count in sorted(eventCounts.items()):
            lines.append(f"- {eventType}: {count}")
    else:
        lines.append("- No rule-based events were detected.")

    lines.extend(["", "## Cue Coverage Summary", ""])

    for cueInfo in cueDefinitions.values():
        lines.append(f"- {cueInfo['cue']}: {cueCounts.get(cueInfo['cue'], 0)} event(s)")

    lines.extend(["", "## Timestamped Moments by Cue", ""])

    if eventsByCue:
        for cueName in sorted(eventsByCue.keys()):
            lines.append(f"### {cueName}")
            lines.append("")
            for event in eventsByCue[cueName]:
                sourceText = "+".join(event.get("detectionSources", ["rule"]))
                confidenceText = (
                    f", mean ML confidence {event['mlConfidenceMean']:.2f}"
                    if event.get("mlConfidenceMean") is not None
                    else ""
                )
                lines.append(
                    f"- {event['startTime']:.2f}s to {event['endTime']:.2f}s: "
                    f"{event['eventType']} [{sourceText}{confidenceText}] - {event['description']}"
                )
            lines.append("")
    else:
        lines.append("- No timestamped moments to review.")
        lines.append("")

    lines.extend(["## Strengths Noticed", ""])
    addEventList(lines, events, strengthEventTypes, "No clear strength-style cue segments were detected.")

    lines.extend(["", "## Review Moments", ""])
    addEventList(lines, events, reviewEventTypes, "No review-style cue segments were detected.")

    lines.extend(
        [
            "",
            "## Cue Definitions",
            "",
        ]
    )

    for cueInfo in cueDefinitions.values():
        lines.append(f"### {cueInfo['cue']}")
        if cueInfo.get("detectionParameters"):
            lines.append(f"- Detection parameters: {', '.join(cueInfo['detectionParameters'])}")
        lines.append(f"- Detection meaning: {cueInfo['detectionMeaning']}")
        lines.append(f"- Semantic purpose: {cueInfo['semanticPurpose']}")
        lines.append(f"- Safe coaching interpretation: {cueInfo['safeInterpretation']}")
        lines.append("")

    lines.extend(
        [
            "## Next Review Step",
            "",
            "This event JSON can later be passed into an LLM for coaching feedback.",
            "",
        ]
    )

    return "\n".join(lines)


def addEventList(lines, events, selectedEventTypes, fallbackText):
    selectedEvents = [event for event in events if event["eventType"] in selectedEventTypes]

    if not selectedEvents:
        lines.append(f"- {fallbackText}")
        return

    for event in selectedEvents[:10]:
        cueInfo = getCueInfo(event["eventType"])
        lines.append(
            f"- {event['startTime']:.2f}s to {event['endTime']:.2f}s: "
            f"{event['eventType']} ({cueInfo['cue']}) - {event['description']}"
        )
