from pathlib import Path
from typing import Any


def buildMultimodalReport(
    mediaPath: str | Path,
    sessionContext: dict[str, Any],
    mediaInfo: dict[str, Any],
    transcript: dict[str, Any] | None,
    responseAnalysis: dict[str, Any] | None,
    audioFeatures: dict[str, Any] | None,
    audioEvents: list[dict[str, Any]] | None,
    visualEvents: list[dict[str, Any]] | None,
    multimodalMoments: list[dict[str, Any]] | None,
    scoreBundle: dict[str, Any],
) -> str:
    transcript = transcript or {}
    responseAnalysis = responseAnalysis or {}
    audioFeatures = audioFeatures or {}
    audioEvents = audioEvents or []
    visualEvents = visualEvents or []
    multimodalMoments = multimodalMoments or []
    scores = scoreBundle.get("scores", {})
    overall = scores.get("overallInterviewPracticeDelivery", {})
    coverage = _coverage(mediaInfo, transcript, audioFeatures, visualEvents)
    strengths = _strongestMoments(responseAnalysis, audioEvents, visualEvents, multimodalMoments)
    reviews = _reviewMoments(responseAnalysis, audioEvents, visualEvents, multimodalMoments)
    practicePlan = _practicePlan(responseAnalysis, scores, multimodalMoments)

    lines = [f"# TalonCV Multimodal Interview Review: {Path(mediaPath).name}", ""]
    _section(lines, "1. Session overview")
    lines.extend(
        [
            f"- Media duration: {_format(mediaInfo.get('durationSeconds'), ' seconds')}",
            f"- Video available: {'yes' if mediaInfo.get('hasVideo') else 'no'}",
            f"- Audio available: {'yes' if mediaInfo.get('hasAudio') else 'no'}",
            f"- Transcript words: {responseAnalysis.get('metrics', {}).get('wordCount', 0)}",
            f"- Analysis coverage: {', '.join(coverage)}",
            "",
        ]
    )

    _section(lines, "2. Interview question and target role")
    lines.extend(
        [
            f"- Interview question: {sessionContext.get('interviewQuestion') or 'Not supplied'}",
            f"- Target role: {sessionContext.get('targetRole') or 'Not supplied'}",
            f"- Job context: {sessionContext.get('jobDescription') or 'Not supplied'}",
            f"- Desired competencies: {sessionContext.get('desiredCompetencies') or 'Not supplied'}",
            "",
        ]
    )

    _section(lines, "3. Data coverage and analysis confidence")
    for name, score in scores.items():
        lines.append(
            f"- {name}: {score.get('rating', 'Unavailable')}"
            f" ({score.get('score') if score.get('score') is not None else 'no score'}; confidence {score.get('confidence', 'unavailable')})"
        )
        for evidence in score.get("evidence", [])[:2]:
            lines.append(f"  - Evidence: {evidence}")
    for warning in _warnings(transcript, responseAnalysis, audioFeatures, mediaInfo):
        lines.append(f"- Warning: {warning}")
    lines.append("")

    _section(lines, "4. Overall coaching summary")
    if overall.get("score") is not None:
        lines.append(
            f"The available evidence produced an overall interview-practice delivery score of "
            f"{overall['score']}/100 ({overall['rating']}). This is a coaching summary, not a hiring assessment."
        )
    else:
        lines.append("An overall score was not calculated because sufficient modality evidence was unavailable.")
    excluded = overall.get("componentBreakdown", {}).get("excludedComponents", [])
    if excluded:
        lines.append(f"Excluded unavailable components: {', '.join(excluded)}.")
    lines.append("")

    _section(lines, "5. Strongest moments")
    _items(lines, strengths, "No evidence-backed strength moment was identified.")

    _section(lines, "6. Most important moments to review")
    _items(lines, reviews, "No sustained evidence-backed review moment was identified.")

    _section(lines, "7. Transcript")
    if transcript.get("text"):
        lines.append(transcript["text"])
        lines.append("")
        for segment in transcript.get("segments", []):
            lines.append(
                f"- {float(segment.get('start', 0)):.2f}s–{float(segment.get('end', 0)):.2f}s: "
                f"{segment.get('text', '').strip()} (confidence {_format(segment.get('confidence'))})"
            )
    else:
        lines.append("Transcript unavailable: no usable speech transcript was generated.")
    lines.append("")

    _section(lines, "8. Answer-quality analysis")
    if responseAnalysis.get("available"):
        metrics = responseAnalysis.get("metrics", {})
        relevance = responseAnalysis.get("relevanceAnalysis", {})
        lines.extend(
            [
                f"- Response length: {metrics.get('wordCount', 0)} words",
                f"- Fillers: {metrics.get('fillerCount', 0)} ({metrics.get('fillerRatePer100Words', 0)} per 100 words)",
                f"- Examples/actions/results: {metrics.get('exampleMarkerCount', 0)}/{metrics.get('actionMarkerCount', 0)}/{metrics.get('resultMarkerCount', 0)}",
                f"- STAR-style coverage: {responseAnalysis.get('starAnalysis', {}).get('componentsPresent', 0)}/4 observable components",
                f"- Relevance: {relevance.get('score') if relevance.get('available') else 'Unavailable because no question was supplied'}",
                f"- Clear conclusion marker: {'yes' if metrics.get('hasConclusion') else 'no'}",
            ]
        )
        for name, item in responseAnalysis.get("rubric", {}).items():
            lines.append(f"- {name}: {item.get('score')} — {item.get('rating')} ({item.get('formula')})")
        development = responseAnalysis.get("answerDevelopment", {})
        lines.extend(
            [
                f"- Opening: {development.get('openingNote', 'unavailable')}",
                f"- Length: {development.get('lengthAssessment', 'unavailable')}",
                f"- Example/result quality: {development.get('exampleQuality', 'unavailable')} / {development.get('resultQuality', 'unavailable')}",
                f"- Responsibility and action separated: {'yes' if development.get('responsibilityAndActionSeparated') else 'not clearly'}",
                f"- Missing elements: {', '.join(development.get('missingElements', [])) or 'none detected'}",
                f"- Semantic repetitions: {development.get('semanticRepetitionCount', 0)}",
            ]
        )
        for passage in development.get("passagesToRevise", []):
            lines.append(
                f"- Revise {float(passage.get('startTime') or 0):.2f}s: {passage.get('text', '')} — "
                f"{passage.get('recommendation', '')}"
            )
    else:
        lines.append("Answer-quality analysis was unavailable because no transcript was produced.")
    lines.append("")

    _section(lines, "9. Vocal-delivery analysis")
    if audioFeatures.get("available"):
        lines.extend(
            [
                f"- Speech rate: {_format(audioFeatures.get('speechRateWpm'), ' WPM')}",
                f"- Speech-rate variation: {_format(audioFeatures.get('speechRateVariationWpm'), ' WPM')}",
                f"- Volume consistency: {_format(audioFeatures.get('volumeConsistencyStdDb'), ' dB standard deviation')}",
                f"- Energy variation: {_format(audioFeatures.get('energyVariationDb'), ' dB')}",
                f"- Pitch median/variation: {_format(audioFeatures.get('pitchMedianHz'), ' Hz')} / {_format(audioFeatures.get('pitchVariationSemitones'), ' semitones')}",
                f"- Long pauses: {audioFeatures.get('longPauseCount', 0)}",
            ]
        )
        _eventLines(lines, [event for event in audioEvents if event["eventType"] not in {"audioClipping", "lowAudioQuality", "audioDropout"}])
    else:
        lines.append("Vocal-delivery analysis was unavailable because audible audio was not present.")
    lines.append("")

    _section(lines, "10. Audio-quality analysis")
    if audioFeatures:
        lines.extend(
            [
                f"- Duration/sample rate/channels: {_format(audioFeatures.get('durationSeconds'), 's')} / {audioFeatures.get('sampleRate', 'unavailable')} / {audioFeatures.get('sourceChannels', 'unavailable')}",
                f"- Overall level: {_format(audioFeatures.get('overallRmsDb'), ' dBFS')}",
                f"- Speech/silence ratio: {_formatRatio(audioFeatures.get('speechRatio'))} / {_formatRatio(audioFeatures.get('silenceRatio'))}",
                f"- Clipping: {_format(audioFeatures.get('clippingPercentage'), '%')}",
                f"- Dropout ratio: {_formatRatio(audioFeatures.get('dropoutRatio'))}",
                f"- SNR proxy: {_format(audioFeatures.get('snrProxyDb'), ' dB')}",
            ]
        )
        _eventLines(lines, [event for event in audioEvents if event["eventType"] in {"audioClipping", "lowAudioQuality", "audioDropout", "lowVolume", "highVolume"}])
    else:
        lines.append("Audio-quality analysis was unavailable because the media contained no audio stream.")
    lines.append("")

    _section(lines, "11. Visual-cue analysis")
    if mediaInfo.get("hasVideo"):
        _eventLines(lines, visualEvents)
    else:
        lines.append("Visual analysis was unavailable because the selected media had no video stream.")
    lines.append("")

    _section(lines, "12. Multimodal alignment moments")
    if multimodalMoments:
        for moment in multimodalMoments:
            lines.append(
                f"- {moment['startTime']:.2f}s–{moment['endTime']:.2f}s [{moment['classification']} / {moment['alignmentCategory']}]: "
                f"{moment['explanation']} Recommendation: {moment['coachingRecommendation']}"
            )
            if moment.get("transcriptExcerpt"):
                lines.append(f"  - Transcript: {moment['transcriptExcerpt']}")
    else:
        lines.append("No combined timestamp moment was available; at least two usable modalities are needed.")
    lines.append("")

    _section(lines, "13. Timestamped evidence")
    timestamped = _timestampedEvidence(audioEvents, visualEvents, multimodalMoments)
    _items(lines, timestamped, "No timestamped event evidence was available.")

    _section(lines, "14. Suggested answer improvements")
    _items(lines, responseAnalysis.get("practiceAreas", []), "No transcript-based answer improvement was available.")
    for step in responseAnalysis.get("suggestedAnswerStructure", []):
        lines.append(f"- Structure: {step}")
    lines.append("")

    _section(lines, "15. Prioritized practice plan")
    for index, item in enumerate(practicePlan[:3], start=1):
        lines.append(f"{index}. {item}")
    lines.append("")

    _section(lines, "16. Safety and interpretation limitations")
    lines.extend(
        [
            "TalonCV is an interview-practice coaching application, not a hiring model.",
            "",
            "Transcription may contain mistakes. Audio, verbal, and visual measurements are approximate coaching proxies. "
            "Verify important wording and timestamps against playback.",
            "",
            "This report does not infer personality, honesty, intelligence, mental state, anxiety, internal confidence, emotion, "
            "employability, professional worth, protected characteristics, or hiring suitability. It does not penalize accent, "
            "dialect, non-native speech, or speech differences.",
            "",
            "All recording, transcription, semantic analysis, coaching, visual analysis, scoring, alignment, and reporting run locally from explicit filesystem model paths.",
            "",
        ]
    )
    return "\n".join(lines)


def _coverage(mediaInfo: dict[str, Any], transcript: dict[str, Any], audioFeatures: dict[str, Any], visualEvents: list[dict[str, Any]]) -> list[str]:
    output = []
    if audioFeatures.get("available"):
        output.append("audio")
    if transcript.get("text"):
        output.append("transcript")
    if mediaInfo.get("hasVideo"):
        output.append("video")
    if visualEvents:
        output.append("visual cues")
    return output or ["no usable modality"]


def _strongestMoments(response: dict[str, Any], audioEvents: list[dict[str, Any]], visualEvents: list[dict[str, Any]], moments: list[dict[str, Any]]) -> list[str]:
    output = [
        f"{moment['startTime']:.2f}s: {moment['explanation']}"
        for moment in moments
        if moment.get("classification") == "strength"
    ]
    output.extend(
        f"{phrase['startTime']:.2f}s: {phrase['text']} ({', '.join(phrase['reasons'])})"
        for phrase in response.get("strongPhrases", [])
    )
    output.extend(
        f"{event['startTime']:.2f}s: visual {event['eventType']} — {event.get('description', '')}"
        for event in visualEvents
        if event.get("eventType") in {"cameraFacing", "positiveExpression", "stablePosture", "centeredFraming"}
    )
    return output[:8]


def _reviewMoments(response: dict[str, Any], audioEvents: list[dict[str, Any]], visualEvents: list[dict[str, Any]], moments: list[dict[str, Any]]) -> list[str]:
    output = [
        f"{moment['startTime']:.2f}s: {moment['explanation']} {moment['coachingRecommendation']}"
        for moment in moments
        if moment.get("classification") == "review"
    ]
    output.extend(
        f"{event['startTime']:.2f}s: audio {event['eventType']} — {event.get('coachingInterpretation', '')}"
        for event in audioEvents
        if event.get("eventType") in {"rapidSpeech", "longPause", "audioClipping", "lowAudioQuality", "speechFragmentation"}
    )
    output.extend(
        f"{event['startTime']:.2f}s: visual {event['eventType']} — {event.get('description', '')}"
        for event in visualEvents
        if event.get("eventType") in {"lookingAway", "possibleFidgeting", "postureShift", "highHeadMovement"}
    )
    return output[:8]


def _practicePlan(response: dict[str, Any], scores: dict[str, Any], moments: list[dict[str, Any]]) -> list[str]:
    items = list(response.get("practiceAreas", []))
    for score in scores.values():
        items.extend(score.get("practiceAreas", []))
    items.extend(
        moment["coachingRecommendation"]
        for moment in moments
        if moment.get("classification") == "review"
    )
    unique = []
    for item in items:
        if item and item not in unique and not item.startswith("No specific"):
            unique.append(item)
    return unique or ["Record another take with one clear example, an explicit action, and an observable result."]


def _timestampedEvidence(audioEvents: list[dict[str, Any]], visualEvents: list[dict[str, Any]], moments: list[dict[str, Any]]) -> list[str]:
    output = []
    output.extend(
        f"{event['startTime']:.2f}s–{event['endTime']:.2f}s audio/{event['eventType']}: {event.get('explanation', '')}"
        for event in audioEvents
    )
    output.extend(
        f"{event['startTime']:.2f}s–{event['endTime']:.2f}s visual/{event['eventType']}: {event.get('description', '')}"
        for event in visualEvents
    )
    output.extend(
        f"{moment['startTime']:.2f}s–{moment['endTime']:.2f}s combined/{moment['alignmentCategory']}: {moment['explanation']}"
        for moment in moments
    )
    return sorted(output, key=lambda item: float(item.split("s", 1)[0].split("–", 1)[0]))[:30]


def _warnings(*sources: dict[str, Any]) -> list[str]:
    output = []
    for source in sources:
        for warning in source.get("warnings", []) if isinstance(source, dict) else []:
            if warning not in output:
                output.append(warning)
    return output


def _eventLines(lines: list[str], events: list[dict[str, Any]]) -> None:
    if not events:
        lines.append("- No events were available in this section.")
        return
    for event in events[:20]:
        lines.append(
            f"- {float(event.get('startTime', 0)):.2f}s–{float(event.get('endTime', 0)):.2f}s "
            f"{event.get('eventType')}: {event.get('explanation') or event.get('description') or event.get('coachingInterpretation', '')}"
        )


def _section(lines: list[str], title: str) -> None:
    lines.extend([f"## {title}", ""])


def _items(lines: list[str], items: list[str], fallback: str) -> None:
    if not items:
        lines.append(f"- {fallback}")
    else:
        lines.extend(f"- {item}" for item in items)
    lines.append("")


def _format(value: Any, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def _formatRatio(value: Any) -> str:
    return f"{float(value):.1%}" if value is not None else "unavailable"
