import re
from collections import Counter
from typing import Any


transcriptAnalysisVersion = "response-v1"

defaultFillerPhrases = [
    "you know",
    "kind of",
    "sort of",
    "i guess",
    "basically",
    "actually",
    "um",
    "uh",
    "like",
    "so",
    "right",
]
hedgingPhrases = [
    "maybe",
    "perhaps",
    "probably",
    "i think",
    "i feel like",
    "i guess",
    "somewhat",
    "might have",
    "could be",
]
genericClaims = [
    "hard worker",
    "team player",
    "people person",
    "detail oriented",
    "good communicator",
    "work well under pressure",
    "always give 110 percent",
]
stopWords = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    "your",
}


def analyzeTranscript(
    transcript: dict[str, Any] | None,
    sessionContext: dict[str, Any] | None = None,
    fillerPhrases: list[str] | None = None,
    semanticAnalysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sessionContext = sessionContext or {}
    semanticAnalysis = semanticAnalysis or {}
    text = str((transcript or {}).get("text") or "").strip()
    segments = (transcript or {}).get("segments", [])
    question = str(sessionContext.get("interviewQuestion") or "").strip()
    transcriptConfidence = (transcript or {}).get("averageConfidence")
    warnings = list((transcript or {}).get("warnings", []))
    if not text:
        return {
            "analysisVersion": transcriptAnalysisVersion,
            "available": False,
            "question": question,
            "metrics": {"wordCount": 0},
            "rubric": _unavailableRubric(question),
            "fillerOccurrences": [],
            "strongPhrases": [],
            "semanticAnalysis": semanticAnalysis,
            "practiceAreas": ["Provide an audible response before evaluating answer structure."],
            "warnings": warnings + ["No transcript text was available for response analysis."],
            "confidence": "unavailable",
        }

    words = _words(text)
    sentences = _sentences(text)
    fillers = detectFillers(segments or [{"start": 0.0, "end": 0.0, "text": text}], fillerPhrases)
    hedges = _phraseOccurrences(text, hedgingPhrases)
    repeatedPhrases = _repeatedNgrams(words)
    unfinishedThoughts = sum(
        bool(re.search(r"\b(and|but|so|because|then|which)\s*[.?!]*$", sentence, flags=re.IGNORECASE))
        for sentence in sentences
    )
    longSentences = sum(len(_words(sentence)) > 35 for sentence in sentences)
    genericMatches = [phrase for phrase in genericClaims if phrase in text.lower()]
    numericMatches = re.findall(r"\b(?:\d+(?:\.\d+)?%?|one|two|three|four|five|six|seven|eight|nine|ten)\b", text.lower())
    exampleMarkers = _phraseOccurrences(
        text,
        ["for example", "for instance", "a time when", "in my role", "on one project", "the situation"],
    )
    actionMarkers = _phraseOccurrences(
        text,
        ["i created", "i implemented", "i led", "i organized", "i analyzed", "i resolved", "i decided", "i built", "i changed", "i worked with"],
    )
    resultMarkers = _phraseOccurrences(
        text,
        ["as a result", "the result", "which led to", "we achieved", "improved", "increased", "decreased", "reduced", "saved"],
    )
    conclusionMarkers = _phraseOccurrences(
        text,
        ["overall", "ultimately", "in conclusion", "that experience", "what i learned", "since then", "the key takeaway"],
    )
    star = _starAnalysis(text, words, exampleMarkers, actionMarkers, resultMarkers)
    relevance = _relevance(text, question, semanticAnalysis)
    repetitionRate = min(1.0, len(repeatedPhrases) * 3 / max(len(words), 1))
    fillerRate = len(fillers) * 100 / max(len(words), 1)
    hedgeRate = len(hedges) * 100 / max(len(words), 1)
    averageSentenceLength = len(words) / max(len(sentences), 1)

    clarityScore = _clamp(
        100
        - max(averageSentenceLength - 25, 0) * 1.5
        - longSentences * 8
        - unfinishedThoughts * 10
        - min(fillerRate, 10) * 2
    )
    concisenessScore = _concisenessScore(len(words), repetitionRate)
    specificityScore = _clamp(
        35
        + min(len(exampleMarkers), 2) * 15
        + min(len(actionMarkers), 3) * 8
        + min(len(resultMarkers), 3) * 10
        + min(len(numericMatches), 3) * 7
        - len(genericMatches) * 8
    )
    organizationScore = _clamp(
        30
        + star["componentsPresent"] * 12
        + (12 if conclusionMarkers else 0)
        - unfinishedThoughts * 8
        - min(repetitionRate * 100, 20)
    )
    completenessScore = _clamp(
        25
        + min(len(words), 120) / 120 * 25
        + star["componentsPresent"] * 10
        + (10 if conclusionMarkers else 0)
    )
    relevanceScore = relevance["score"]
    componentScores = [clarityScore, concisenessScore, specificityScore, organizationScore, completenessScore]
    if relevanceScore is not None:
        componentScores.append(relevanceScore)
    overallScore = round(sum(componentScores) / len(componentScores), 1)

    strongPhrases = _strongPhrases(segments, numericMatches)
    answerDevelopment = _answerDevelopment(
        segments,
        words,
        sentences,
        star,
        strongPhrases,
        bool(conclusionMarkers),
        semanticAnalysis,
    )
    practiceAreas = _practiceAreas(
        fillerRate,
        hedgeRate,
        specificityScore,
        organizationScore,
        concisenessScore,
        relevance,
        bool(conclusionMarkers),
    )
    confidenceLabel = _confidenceLabel(transcriptConfidence)
    if confidenceLabel == "low":
        warnings.append(
            "Content findings have reduced reliability because the transcript confidence was low; verify wording against playback."
        )

    rubric = {
        "overallVerbalResponse": _rubricItem(overallScore, "Mean of available deterministic response components."),
        "clarity": _rubricItem(
            clarityScore,
            "Starts at 100; deductions use sentence length, unfinished thoughts, long sentences, and contextual filler rate.",
        ),
        "conciseness": _rubricItem(
            concisenessScore,
            "Uses response word-count range and repeated-phrase rate; it does not reward an accent or speaking style.",
        ),
        "specificity": _rubricItem(
            specificityScore,
            "Uses example, action, result, and measurable-detail markers with a small deduction for unsupported generic claims.",
        ),
        "organization": _rubricItem(
            organizationScore,
            "Uses observable context/action/result markers, a conclusion marker, repetition, and unfinished thoughts.",
        ),
        "completeness": _rubricItem(
            completenessScore,
            "Uses response length, context/action/result coverage, and whether the answer contains a conclusion marker.",
        ),
        "relevance": (
            _rubricItem(
                relevanceScore,
                "Blend of deterministic keyword coverage and local MiniLM similarity when the local semantic model is available.",
            )
            if relevanceScore is not None
            else {
                "score": None,
                "rating": "Unavailable",
                "formula": "No interview question was supplied, so relevance is not claimed.",
            }
        ),
    }
    return {
        "analysisVersion": transcriptAnalysisVersion,
        "available": True,
        "question": question,
        "targetRole": str(sessionContext.get("targetRole") or ""),
        "transcriptConfidence": transcriptConfidence,
        "confidence": confidenceLabel,
        "metrics": {
            "wordCount": len(words),
            "sentenceCount": len(sentences),
            "averageSentenceWords": round(averageSentenceLength, 2),
            "fillerCount": len(fillers),
            "fillerRatePer100Words": round(fillerRate, 2),
            "hedgeCount": len(hedges),
            "hedgeRatePer100Words": round(hedgeRate, 2),
            "unfinishedThoughtCount": unfinishedThoughts,
            "longSentenceCount": longSentences,
            "repeatedPhraseCount": len(repeatedPhrases) + len(semanticAnalysis.get("semanticRedundancy", [])),
            "genericClaimCount": len(genericMatches),
            "measurableDetailCount": len(numericMatches),
            "exampleMarkerCount": len(exampleMarkers),
            "actionMarkerCount": len(actionMarkers),
            "resultMarkerCount": len(resultMarkers),
            "hasConclusion": bool(conclusionMarkers),
        },
        "starAnalysis": star,
        "relevanceAnalysis": relevance,
        "roleAlignmentAnalysis": semanticAnalysis.get(
            "roleAlignment",
            {
                "available": False,
                "score": None,
                "explanation": "Local semantic role alignment was unavailable.",
            },
        ),
        "semanticAnalysis": semanticAnalysis,
        "answerDevelopment": answerDevelopment,
        "rubric": rubric,
        "fillerOccurrences": fillers,
        "hedgingPhrases": hedges,
        "repeatedPhrases": repeatedPhrases,
        "genericClaims": genericMatches,
        "strongPhrases": strongPhrases,
        "practiceAreas": practiceAreas,
        "suggestedAnswerStructure": _suggestedStructure(question, star),
        "warnings": warnings,
        "fairnessNote": (
            "The rubric uses transcript structure and observable wording only. It does not assess accent, dialect, "
            "native-language status, personality, intelligence, or hiring suitability."
        ),
    }


def detectFillers(segments: list[dict[str, Any]], fillerPhrases: list[str] | None = None) -> list[dict[str, Any]]:
    phrases = sorted(fillerPhrases or defaultFillerPhrases, key=len, reverse=True)
    occurrences = []
    for segment in segments:
        text = str(segment.get("text") or "")
        lowered = text.lower()
        occupied: list[tuple[int, int]] = []
        for phrase in phrases:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", lowered):
                if any(match.start() < end and match.end() > start for start, end in occupied):
                    continue
                if phrase == "like" and _legitimateLikeUse(lowered, match.start(), match.end()):
                    continue
                if phrase == "so" and not _discourseBoundary(lowered, match.start()):
                    continue
                if phrase == "right" and not (
                    _discourseBoundary(lowered, match.start()) or match.end() >= len(lowered.rstrip(" .,!?:;"))
                ):
                    continue
                occupied.append((match.start(), match.end()))
                occurrences.append(
                    {
                        "phrase": phrase,
                        "startTime": float(segment.get("start", 0)),
                        "endTime": float(segment.get("end", segment.get("start", 0))),
                        "context": text.strip(),
                        "confidence": "medium" if phrase in {"like", "so", "right", "actually"} else "high",
                    }
                )
    return sorted(occurrences, key=lambda item: (item["startTime"], item["phrase"]))


def _legitimateLikeUse(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 15) : start]
    after = text[end : min(len(text), end + 12)]
    return bool(
        re.search(r"\b(would|feel|felt|look|looked|looks|sound|sounds|seem|seems)\s+$", before)
        or re.match(r"\s+to\b", after)
    )


def _discourseBoundary(text: str, start: int) -> bool:
    return not text[:start].strip() or bool(re.search(r"[.!?]\s*$", text[:start]))


def _relevance(text: str, question: str, semanticAnalysis: dict[str, Any] | None = None) -> dict[str, Any]:
    if not question:
        return {
            "available": False,
            "score": None,
            "explanation": "No interview question was supplied, so relevance is unavailable.",
            "matchedKeywords": [],
            "questionKeywords": [],
        }
    responseWords = set(_contentWords(text))
    questionWords = set(_contentWords(question))
    matched = sorted(responseWords & questionWords)
    coverage = len(matched) / max(len(questionWords), 1)
    keywordScore = _clamp(35 + coverage * 65)
    semantic = (semanticAnalysis or {}).get("questionRelevance", {})
    semanticScore = semantic.get("score") if semantic.get("available") else None
    score = _clamp(keywordScore * 0.35 + float(semanticScore) * 0.65) if semanticScore is not None else keywordScore
    return {
        "available": True,
        "score": score,
        "explanation": (
            "Combined deterministic keyword coverage with local MiniLM semantic similarity."
            if semanticScore is not None
            else "Deterministic keyword coverage; the local MiniLM component was unavailable."
        ),
        "matchedKeywords": matched,
        "questionKeywords": sorted(questionWords),
        "keywordCoverage": round(coverage, 3),
        "keywordScore": keywordScore,
        "semanticScore": semanticScore,
        "semanticSimilarity": semantic.get("similarity"),
    }


def _answerDevelopment(
    segments: list[dict[str, Any]],
    words: list[str],
    sentences: list[str],
    star: dict[str, Any],
    strongPhrases: list[dict[str, Any]],
    hasConclusion: bool,
    semanticAnalysis: dict[str, Any],
) -> dict[str, Any]:
    firstSentenceWords = len(_words(sentences[0])) if sentences else 0
    openingClear = 5 <= firstSentenceWords <= 30
    lengthAssessment = "balanced"
    if len(words) < 45:
        lengthAssessment = "short"
    elif len(words) > 280:
        lengthAssessment = "long"
    components = star.get("components", {})
    missing = [name for name, present in components.items() if not present]
    if not hasConclusion:
        missing.append("conclusion")
    if not strongPhrases:
        missing.append("supported measurable or action-oriented claim")
    strongest = strongPhrases[0] if strongPhrases else None
    duration = max((float(segment.get("end", 0)) for segment in segments), default=0.0)
    strongestBuried = bool(strongest and duration and float(strongest["startTime"]) > duration * 0.55)
    vagueSegments = semanticAnalysis.get("vagueSegments", [])
    topicDrift = semanticAnalysis.get("topicDriftSegments", [])
    revised = []
    for item in [*vagueSegments, *topicDrift]:
        revised.append(
            {
                "startTime": item.get("startTime"),
                "endTime": item.get("endTime"),
                "text": item.get("text", ""),
                "reason": (
                    "This passage may be vague or weakly connected to the supplied question."
                    if item in vagueSegments
                    else "This passage may drift from the supplied question."
                ),
                "recommendation": "Replace general wording with one specific responsibility, action, or observable result.",
            }
        )
    return {
        "openingClear": openingClear,
        "openingNote": (
            "The opening established a reasonably concise starting point."
            if openingClear
            else "Open with a direct 5–30 word answer before adding context."
        ),
        "conclusionComplete": hasConclusion,
        "lengthAssessment": lengthAssessment,
        "exampleQuality": "complete" if components.get("situation") and components.get("action") else "developing",
        "resultQuality": "supported" if components.get("result") and strongPhrases else "needs clearer evidence",
        "responsibilityAndActionSeparated": bool(components.get("task") and components.get("action")),
        "missingElements": list(dict.fromkeys(missing)),
        "strongestSentenceBuried": strongestBuried,
        "strongestSentence": strongest,
        "semanticRepetitionCount": len(semanticAnalysis.get("semanticRedundancy", [])),
        "passagesToRevise": revised[:5],
    }


def _starAnalysis(text: str, words: list[str], examples: list[str], actions: list[str], results: list[str]) -> dict[str, Any]:
    lowered = text.lower()
    situation = bool(examples or re.search(r"\b(situation|challenge|project|role|team|client)\b", lowered))
    task = bool(re.search(r"\b(goal|needed to|responsible for|objective|my task|we had to)\b", lowered))
    action = bool(actions or re.search(r"\bi\s+(led|built|created|organized|analyzed|resolved|implemented|decided|changed)\b", lowered))
    result = bool(results or re.search(r"\b(result|outcome|improved|increased|decreased|reduced|saved|achieved|learned)\b", lowered))
    components = {"situation": situation, "task": task, "action": action, "result": result}
    return {
        "appropriate": len(words) >= 50,
        "components": components,
        "componentsPresent": sum(components.values()),
        "balanceNote": (
            "The response includes all observable STAR-style components."
            if all(components.values())
            else "Consider making the missing context, responsibility, action, or result more explicit where appropriate."
        ),
    }


def _strongPhrases(segments: list[dict[str, Any]], numericMatches: list[str]) -> list[dict[str, Any]]:
    output = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        lowered = text.lower()
        reasons = []
        if re.search(r"\d", text) or any(number in lowered for number in numericMatches):
            reasons.append("measurable detail")
        if re.search(r"\bi\s+(led|built|created|implemented|resolved|organized|analyzed|improved)\b", lowered):
            reasons.append("clear personal action")
        if re.search(r"\b(result|outcome|improved|increased|decreased|reduced|saved|achieved)\b", lowered):
            reasons.append("result language")
        if reasons and len(text.split()) >= 5:
            output.append(
                {
                    "startTime": float(segment.get("start", 0)),
                    "endTime": float(segment.get("end", 0)),
                    "text": text,
                    "reasons": reasons,
                }
            )
    return output[:5]


def _practiceAreas(
    fillerRate: float,
    hedgeRate: float,
    specificity: float,
    organization: float,
    conciseness: float,
    relevance: dict[str, Any],
    hasConclusion: bool,
) -> list[str]:
    areas = []
    if fillerRate > 3:
        areas.append("Replace one repeated filler pattern with a short silent pause.")
    if hedgeRate > 3:
        areas.append("State completed actions directly where the evidence supports them instead of repeatedly hedging.")
    if specificity < 65:
        areas.append("Add one concrete example, your specific action, and a measurable or observable result.")
    if organization < 65:
        areas.append("Practice a context → responsibility → action → result sequence.")
    if conciseness < 65:
        areas.append("Remove repeated setup and move the main action earlier in the response.")
    if relevance.get("available") and relevance.get("score", 100) < 60:
        areas.append("Echo the interview question's main topic in the opening sentence and answer it directly.")
    if not hasConclusion:
        areas.append("Finish with a one-sentence result or takeaway rather than letting the answer trail off.")
    return areas[:5] or ["Preserve the current structure and rehearse it once more with a different example."]


def _suggestedStructure(question: str, star: dict[str, Any]) -> list[str]:
    opening = (
        f"Open with a direct one-sentence answer to: {question}"
        if question
        else "Open with a one-sentence statement of the main point you want the interviewer to remember."
    )
    return [
        opening,
        "Give only the context needed to understand the challenge or responsibility.",
        "Describe your specific actions using clear first-person verbs.",
        "State the measurable or observable result and what you learned.",
        "Close by connecting the example to the target role when relevant.",
    ]


def _phraseOccurrences(text: str, phrases: list[str]) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in phrases for _ in re.finditer(rf"\b{re.escape(phrase)}\b", lowered)]


def _repeatedNgrams(words: list[str]) -> list[dict[str, Any]]:
    normalized = [word for word in words if word not in stopWords]
    counts = Counter(" ".join(normalized[index : index + 3]) for index in range(max(0, len(normalized) - 2)))
    return [
        {"phrase": phrase, "count": count}
        for phrase, count in counts.most_common(10)
        if count >= 2 and len(phrase) >= 8
    ]


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?", text.lower())


def _contentWords(text: str) -> list[str]:
    return [word for word in _words(text) if word not in stopWords and len(word) > 2]


def _sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]
    return sentences or [text]


def _concisenessScore(wordCount: int, repetitionRate: float) -> float:
    if wordCount < 25:
        lengthScore = 55 + wordCount
    elif wordCount <= 250:
        lengthScore = 100
    else:
        lengthScore = max(45, 100 - (wordCount - 250) * 0.25)
    return _clamp(lengthScore - repetitionRate * 80)


def _rubricItem(score: float | None, formula: str) -> dict[str, Any]:
    return {"score": round(score, 1) if score is not None else None, "rating": _rating(score), "formula": formula}


def _rating(score: float | None) -> str:
    if score is None:
        return "Unavailable"
    if score >= 85:
        return "Strong evidence"
    if score >= 70:
        return "Generally effective"
    if score >= 55:
        return "Mixed evidence"
    return "Practice priority"


def _confidenceLabel(value: Any) -> str:
    if value is None:
        return "limited"
    if float(value) >= 0.8:
        return "high"
    if float(value) >= 0.6:
        return "medium"
    return "low"


def _clamp(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 1)


def _unavailableRubric(question: str) -> dict[str, Any]:
    names = ["overallVerbalResponse", "clarity", "conciseness", "specificity", "organization", "completeness"]
    rubric = {name: {"score": None, "rating": "Unavailable", "formula": "No transcript was available."} for name in names}
    rubric["relevance"] = {
        "score": None,
        "rating": "Unavailable",
        "formula": "No transcript was available." if question else "No interview question was supplied.",
    }
    return rubric
