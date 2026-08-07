from typing import Any

from src.cvPipeline.reportUtils import reviewEventTypes, strengthEventTypes


scoreVersion = "scores-v1"


def buildCoachingScores(
    mediaInfo: dict[str, Any],
    audioFeatures: dict[str, Any] | None,
    audioEvents: list[dict[str, Any]] | None,
    responseAnalysis: dict[str, Any] | None,
    visualEvents: list[dict[str, Any]] | None,
    multimodalMoments: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    audioFeatures = audioFeatures or {}
    audioEvents = audioEvents or []
    responseAnalysis = responseAnalysis or {}
    visualEvents = visualEvents or []
    multimodalMoments = multimodalMoments or []

    scores = {
        "audioRecordingQuality": _audioQualityScore(audioFeatures),
        "vocalDelivery": _vocalDeliveryScore(audioFeatures, audioEvents),
        "verbalResponseQuality": _verbalScore(responseAnalysis),
        "visualDelivery": _visualScore(mediaInfo, visualEvents),
        "multimodalAlignment": _alignmentScore(mediaInfo, audioFeatures, multimodalMoments),
    }
    availableWeights = {
        "audioRecordingQuality": 0.15,
        "vocalDelivery": 0.20,
        "verbalResponseQuality": 0.30,
        "visualDelivery": 0.20,
        "multimodalAlignment": 0.15,
    }
    included = {
        name: weight
        for name, weight in availableWeights.items()
        if scores[name].get("score") is not None
    }
    excluded = [name for name in availableWeights if name not in included]
    if included:
        totalWeight = sum(included.values())
        components = {
            name: {
                "score": scores[name]["score"],
                "normalizedWeight": round(weight / totalWeight, 4),
            }
            for name, weight in included.items()
        }
        overall = round(
            sum(scores[name]["score"] * weight / totalWeight for name, weight in included.items()), 1
        )
        confidence = "high" if len(included) == 5 else "medium" if len(included) >= 3 else "limited"
        scores["overallInterviewPracticeDelivery"] = _score(
            overall,
            confidence,
            [f"{name}: {item['score']} at {item['normalizedWeight']:.0%}" for name, item in components.items()],
            _collect(scores, "positiveObservations"),
            _collect(scores, "practiceAreas"),
            "Weighted mean of available modality scores; unavailable components are excluded and remaining weights are renormalized.",
            {"components": components, "excludedComponents": excluded},
        )
    else:
        scores["overallInterviewPracticeDelivery"] = _unavailable(
            "No usable audio, transcript, or visual evidence was available."
        )
    return {"analysisVersion": scoreVersion, "scores": scores, "safetyNote": _safetyNote()}


def _audioQualityScore(features: dict[str, Any]) -> dict[str, Any]:
    if not features.get("available"):
        return _unavailable("No audible decoded waveform was available.")
    score = 100.0
    clipping = float(features.get("clippingPercentage") or 0)
    dropout = float(features.get("dropoutRatio") or 0)
    overall = features.get("overallRmsDb")
    snr = features.get("snrProxyDb")
    score -= min(clipping * 8, 35)
    score -= min(dropout * 100, 30)
    if overall is not None and overall < -40:
        score -= min((-40 - overall) * 1.5, 25)
    if snr is not None and snr < 15:
        score -= min((15 - snr) * 1.5, 20)
    positives = []
    practice = []
    if clipping < 0.1:
        positives.append("Very little digital clipping was measured.")
    else:
        practice.append("Reduce microphone gain or increase microphone distance to avoid clipping.")
    if dropout < 0.01:
        positives.append("Few near-silent audio dropouts were measured.")
    else:
        practice.append("Check microphone stability because near-silent dropouts were detected.")
    if overall is not None and overall < -40:
        practice.append("Increase input level or move closer to the microphone.")
    evidence = [
        f"Overall RMS level: {overall} dBFS",
        f"Clipping: {clipping:.4f}%",
        f"Dropout ratio: {dropout:.2%}",
        f"SNR proxy: {snr if snr is not None else 'unavailable'} dB",
    ]
    return _score(score, "high", evidence, positives, practice, "100 minus clipping, dropout, low-level, and low-SNR proxy deductions.")


def _vocalDeliveryScore(features: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    if not features.get("available"):
        return _unavailable("No audible decoded waveform was available for vocal-delivery analysis.")
    score = 75.0
    pace = features.get("speechRateWpm")
    volumeStd = features.get("volumeConsistencyStdDb")
    energyVariation = features.get("energyVariationDb")
    pitchVariation = features.get("pitchVariationSemitones")
    if pace is not None:
        if 105 <= pace <= 175:
            score += 12
        else:
            score -= min(abs(pace - (105 if pace < 105 else 175)) * 0.25, 25)
    if volumeStd is not None:
        score += 7 if 2 <= volumeStd <= 9 else -8
    if energyVariation is not None:
        score += 6 if energyVariation >= 6 else -6
    if pitchVariation is not None:
        score += 5 if pitchVariation >= 1.5 else -5
    longPauses = sum(event["eventType"] == "longPause" for event in events)
    fragmented = sum(event["eventType"] == "speechFragmentation" for event in events)
    score -= min(longPauses * 2, 10)
    score -= min(fragmented * 2, 10)
    positives = []
    practice = []
    if pace is not None and 105 <= pace <= 175:
        positives.append("The measured overall word rate fell in a broadly conversational practice range.")
    elif pace is not None:
        practice.append("Rehearse the answer near a conversational 105–175 words per minute without forcing a speaking style.")
    if energyVariation is not None and energyVariation >= 6:
        positives.append("The audio showed measurable energy variation that may support emphasis.")
    elif energyVariation is not None:
        practice.append("Add natural emphasis to the main action and result rather than increasing volume everywhere.")
    if longPauses:
        practice.append("Review sustained pauses and keep the ones that clarify structure.")
    return _score(
        score,
        "medium" if pitchVariation is None else "high",
        [
            f"Speech rate: {pace if pace is not None else 'unavailable'} WPM",
            f"Volume consistency standard deviation: {volumeStd if volumeStd is not None else 'unavailable'} dB",
            f"Energy range: {energyVariation if energyVariation is not None else 'unavailable'} dB",
            f"Pitch variation: {pitchVariation if pitchVariation is not None else 'unavailable'} semitones",
            f"Long pauses: {longPauses}; short fragments: {fragmented}",
        ],
        positives,
        practice,
        "Base 75 with observable pace, volume consistency, energy/pitch variation, pause, and fragmentation adjustments.",
    )


def _verbalScore(response: dict[str, Any]) -> dict[str, Any]:
    if not response.get("available"):
        return _unavailable("No transcript was available for deterministic response analysis.")
    item = response.get("rubric", {}).get("overallVerbalResponse", {})
    score = item.get("score")
    metrics = response.get("metrics", {})
    positives = []
    if metrics.get("exampleMarkerCount", 0):
        positives.append("The response included an observable example marker.")
    if metrics.get("resultMarkerCount", 0):
        positives.append("The response included result-oriented wording.")
    if metrics.get("hasConclusion"):
        positives.append("The response included a recognizable conclusion marker.")
    return _score(
        score,
        response.get("confidence", "limited"),
        [
            f"Words: {metrics.get('wordCount', 0)}",
            f"Fillers per 100 words: {metrics.get('fillerRatePer100Words', 0)}",
            f"STAR-style components: {response.get('starAnalysis', {}).get('componentsPresent', 0)}/4",
            f"Relevance: {response.get('relevanceAnalysis', {}).get('score', 'unavailable')}",
        ],
        positives,
        response.get("practiceAreas", []),
        item.get("formula", "Mean of available deterministic verbal-response rubric components."),
    )


def _visualScore(mediaInfo: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    if not mediaInfo.get("hasVideo"):
        return _unavailable("The selected media has no video stream.")
    duration = float(mediaInfo.get("durationSeconds") or 0)
    strengths = [event for event in events if event["eventType"] in strengthEventTypes]
    reviews = [event for event in events if event["eventType"] in reviewEventTypes]
    strengthSeconds = sum(float(event.get("durationSeconds", 0)) for event in strengths)
    reviewSeconds = sum(float(event.get("durationSeconds", 0)) for event in reviews)
    score = 70 + min(strengthSeconds / max(duration, 1) * 20, 15) - min(reviewSeconds / max(duration, 1) * 15, 25)
    positives = [f"{event['eventType']} at {event['startTime']:.2f}s" for event in strengths[:3]]
    practice = [f"Review {event['eventType']} at {event['startTime']:.2f}s in context." for event in reviews[:3]]
    return _score(
        score,
        "medium" if events else "limited",
        [f"Visual strength events: {len(strengths)}", f"Visual review/system events: {len(reviews)}"],
        positives,
        practice,
        "Base 70 with capped strength-duration additions and review-duration deductions; this is a coaching proxy, not a trait score.",
    )


def _alignmentScore(
    mediaInfo: dict[str, Any], audioFeatures: dict[str, Any], moments: list[dict[str, Any]]
) -> dict[str, Any]:
    if not mediaInfo.get("hasVideo") or not audioFeatures.get("available"):
        return _unavailable("Both usable audio and video are required for multimodal alignment scoring.")
    strengths = [moment for moment in moments if moment.get("classification") == "strength"]
    reviews = [moment for moment in moments if moment.get("classification") == "review"]
    score = 70 + min(len(strengths) * 6, 24) - min(len(reviews) * 5, 25)
    return _score(
        score,
        "medium" if moments else "limited",
        [f"Aligned strength moments: {len(strengths)}", f"Aligned review moments: {len(reviews)}"],
        [moment["explanation"] for moment in strengths[:3]],
        [moment["coachingRecommendation"] for moment in reviews[:3]],
        "Base 70 with capped additions for aligned strengths and deductions for aligned review moments.",
    )


def _score(
    value: float | None,
    confidence: str,
    evidence: list[str],
    positives: list[str],
    practice: list[str],
    formula: str,
    breakdown: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = round(max(0.0, min(100.0, float(value))), 1) if value is not None else None
    return {
        "score": value,
        "rating": _rating(value),
        "confidence": confidence,
        "dataCoverage": "available" if value is not None else "unavailable",
        "evidence": evidence,
        "positiveObservations": positives or ["No specific positive observation was available for this component."],
        "practiceAreas": practice or ["No specific practice priority was identified for this component."],
        "formula": formula,
        "componentBreakdown": breakdown or {},
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "rating": "Unavailable",
        "confidence": "unavailable",
        "dataCoverage": "unavailable",
        "evidence": [reason],
        "positiveObservations": [],
        "practiceAreas": [],
        "formula": "Excluded because the required modality was unavailable.",
        "componentBreakdown": {},
    }


def _rating(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    if value >= 85:
        return "Strong evidence"
    if value >= 70:
        return "Generally effective"
    if value >= 55:
        return "Mixed evidence"
    return "Practice priority"


def _collect(scores: dict[str, dict[str, Any]], key: str) -> list[str]:
    output = []
    for score in scores.values():
        for item in score.get(key, []):
            if item not in output and not item.startswith("No specific"):
                output.append(item)
    return output[:5]


def _safetyNote() -> str:
    return (
        "These are explainable interview-practice coaching scores based on available recording evidence. "
        "They are not hiring scores and do not assess personality, honesty, intelligence, emotion, mental state, "
        "professional worth, protected characteristics, or suitability for employment."
    )

