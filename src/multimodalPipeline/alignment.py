from typing import Any

from src.cvPipeline.reportUtils import strengthEventTypes


alignmentVersion = "alignment-v3"


def overlaps(startA: float, endA: float, startB: float, endB: float) -> bool:
    return min(float(endA), float(endB)) > max(float(startA), float(startB))


def alignMultimodalEvents(
    transcript: dict[str, Any] | None,
    responseAnalysis: dict[str, Any] | None,
    audioEvents: list[dict[str, Any]] | None,
    visualEvents: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    transcript = transcript or {}
    responseAnalysis = responseAnalysis or {}
    audioEvents = audioEvents or []
    visualEvents = visualEvents or []
    segments = transcript.get("segments", [])
    moments: list[dict[str, Any]] = []

    for audioEvent in audioEvents:
        momentCountBeforeEvent = len(moments)
        matchedVisual = _overlappingEvents(audioEvent, visualEvents)
        audioType = audioEvent["eventType"]
        visualTypes = {event["eventType"] for event in matchedVisual}
        if audioType == "longPause" and "lookingDown" in visualTypes:
            moments.append(
                _moment(
                    audioEvent,
                    matchedVisual,
                    segments,
                    "pauseWithDownwardGaze",
                    "review",
                    "A measurable pause overlapped with a downward head-position cue.",
                    "Replay the moment and decide whether the pause looked intentional; return toward the camera before the next key point.",
                )
            )
        if audioType == "rapidSpeech" and visualTypes & {"highHeadMovement", "lateralHeadMovement"}:
            moments.append(
                _moment(
                    audioEvent,
                    matchedVisual,
                    segments,
                    "rapidSpeechWithMovement",
                    "review",
                    "A high local word rate overlapped with noticeable head movement.",
                    "Practice the same phrase slightly slower with smaller intentional movement.",
                )
            )
        if audioType == "strongVocalEmphasis" and visualTypes & {"handGestureActivity", "handsRaised"}:
            moments.append(
                _moment(
                    audioEvent,
                    matchedVisual,
                    segments,
                    "emphasisWithGesture",
                    "strength",
                    "A rise in vocal energy aligned with a visible hand-gesture cue.",
                    "Review whether the gesture and emphasis supported the same important phrase; preserve it if they did.",
                )
            )
        if audioType == "longPause" and matchedVisual and len(moments) == momentCountBeforeEvent:
            moments.append(
                _moment(
                    audioEvent,
                    matchedVisual,
                    segments,
                    "visualCueDuringPause",
                    "context",
                    "A visual cue occurred during a measured pause rather than continuous speech.",
                    "Interpret the visual cue in the context of the pause before treating it as a delivery priority.",
                )
            )
        if audioType == "lowVolume" and matchedVisual:
            moments.append(
                _moment(
                    audioEvent,
                    matchedVisual,
                    segments,
                    "quietSpeechWithVisibleDelivery",
                    "review",
                    "A measurably quiet passage overlapped with visible delivery evidence, making the spoken point harder to evaluate.",
                    "Replay the phrase for microphone distance and audibility before changing the visual delivery.",
                )
            )

    for filler in responseAnalysis.get("fillerOccurrences", []):
        matchedVisual = _overlappingInterval(
            float(filler.get("startTime", 0)), float(filler.get("endTime", 0)), visualEvents
        )
        matchedReview = [
            event for event in matchedVisual if event["eventType"] in {"possibleFidgeting", "postureShift", "highHeadMovement"}
        ]
        if matchedReview:
            base = {
                "startTime": float(filler.get("startTime", 0)),
                "endTime": float(filler.get("endTime", 0)),
                "eventType": "fillerOccurrence",
                "reliability": filler.get("confidence", "medium"),
                "measurements": {"fillerPhrase": filler.get("phrase")},
            }
            moments.append(
                _moment(
                    base,
                    matchedReview,
                    segments,
                    "fillerWithVisualMovement",
                    "review",
                    "A contextually detected filler overlapped with a visible movement cue.",
                    "Try replacing the filler with a short pause while keeping posture comfortable and settled.",
                )
            )

    for phrase in responseAnalysis.get("strongPhrases", []):
        matchedVisual = _overlappingInterval(
            float(phrase.get("startTime", 0)), float(phrase.get("endTime", 0)), visualEvents
        )
        strengthVisual = [event for event in matchedVisual if event["eventType"] in strengthEventTypes]
        reviewVisual = [event for event in matchedVisual if event["eventType"] in {"lookingAway", "lookingDown"}]
        base = {
            "startTime": float(phrase.get("startTime", 0)),
            "endTime": float(phrase.get("endTime", 0)),
            "eventType": "strongResponseContent",
            "reliability": "medium",
            "measurements": {"strongPhraseReasons": phrase.get("reasons", [])},
        }
        if strengthVisual:
            moments.append(
                _moment(
                    base,
                    strengthVisual,
                    segments,
                    "strongContentWithVisualSupport",
                    "strength",
                    "A specific action or result phrase aligned with a visual strength cue.",
                    "Use this timestamp as a reference for delivering another important example.",
                    transcriptOverride=phrase.get("text"),
                )
            )
        elif reviewVisual:
            moments.append(
                _moment(
                    base,
                    reviewVisual,
                    segments,
                    "strongContentWithAttentionShift",
                    "review",
                    "A strong, specific phrase occurred while camera-facing attention was lower.",
                    "Practice returning toward the camera before stating the result or key takeaway.",
                    transcriptOverride=phrase.get("text"),
                )
            )

    if responseAnalysis.get("metrics", {}).get("hasConclusion") and segments:
        finalSegment = segments[-1]
        matchedVisual = _overlappingInterval(
            float(finalSegment.get("start", 0)), float(finalSegment.get("end", 0)), visualEvents
        )
        conclusionStrength = [
            event for event in matchedVisual if event["eventType"] in {"cameraFacing", "positiveExpression", "stablePosture"}
        ]
        if conclusionStrength:
            base = {
                "startTime": float(finalSegment.get("start", 0)),
                "endTime": float(finalSegment.get("end", 0)),
                "eventType": "answerConclusion",
                "reliability": "medium",
                "measurements": {"conclusionMarker": True},
            }
            moments.append(
                _moment(
                    base,
                    conclusionStrength,
                    segments,
                    "conclusionWithVisualSupport",
                    "strength",
                    "The answer's conclusion marker aligned with a camera-facing, positive-expression, or stable-posture cue.",
                    "Consider preserving this conclusion delivery pattern in the next take.",
                )
            )

    return _deduplicate(moments)


def _moment(
    baseEvent: dict[str, Any],
    visualEvents: list[dict[str, Any]],
    transcriptSegments: list[dict[str, Any]],
    category: str,
    classification: str,
    explanation: str,
    recommendation: str,
    transcriptOverride: str | None = None,
) -> dict[str, Any]:
    start = float(baseEvent.get("startTime", baseEvent.get("start", 0)))
    end = float(baseEvent.get("endTime", baseEvent.get("end", start)))
    excerpt = transcriptOverride or _transcriptExcerpt(transcriptSegments, start, end)
    audioTypes = [] if baseEvent.get("eventType") in {"strongResponseContent", "answerConclusion", "fillerOccurrence"} else [baseEvent.get("eventType")]
    evidenceSources = ["transcript"] if excerpt else []
    if audioTypes:
        evidenceSources.append("audio")
    if visualEvents:
        evidenceSources.append("visual")
    reliability = str(baseEvent.get("reliability") or "medium")
    return {
        "startTime": round(start, 3),
        "endTime": round(max(start, end), 3),
        "durationSeconds": round(max(0.0, end - start), 3),
        "transcriptExcerpt": excerpt,
        "audioEvents": [eventType for eventType in audioTypes if eventType],
        "audioMeasurements": baseEvent.get("measurements", {}),
        "visualEvents": sorted({event["eventType"] for event in visualEvents}),
        "visualEvidence": [
            {
                "eventType": event.get("eventType"),
                "startTime": event.get("startTime"),
                "endTime": event.get("endTime"),
                "reliability": event.get("reliability") or event.get("confidence"),
                "measurements": event.get("measurements", {}),
            }
            for event in visualEvents
        ],
        "alignmentCategory": category,
        "classification": classification,
        "explanation": explanation,
        "coachingRecommendation": recommendation,
        "evidenceSources": evidenceSources,
        "confidence": reliability,
    }


def _overlappingEvents(event: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _overlappingInterval(float(event.get("startTime", 0)), float(event.get("endTime", 0)), candidates)


def _overlappingInterval(start: float, end: float, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in candidates
        if overlaps(start, end, float(event.get("startTime", 0)), float(event.get("endTime", 0)))
    ]


def _transcriptExcerpt(segments: list[dict[str, Any]], start: float, end: float) -> str:
    return " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if overlaps(start, end, float(segment.get("start", 0)), float(segment.get("end", 0)))
    ).strip()


def _deduplicate(moments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for moment in sorted(moments, key=lambda item: (item["startTime"], item["alignmentCategory"])):
        key = (moment["alignmentCategory"], moment["startTime"], moment["endTime"])
        if key not in seen:
            seen.add(key)
            output.append(moment)
    return output
