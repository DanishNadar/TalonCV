import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.localModels.config import getModelSection, modelSetupCommands, selectDevice, validateModelFiles
from src.localModels.cpuRuntime import configureCpuRuntime


semanticAnalysisVersion = "semantic-v1"


class SemanticAnalysisError(RuntimeError):
    """Actionable local semantic-model error."""


@lru_cache(maxsize=2)
def _loadSemanticModel(modelPath: str, device: str):
    configureCpuRuntime()
    path = Path(modelPath)
    readiness = validateModelFiles("semanticAnalysis", {"resolvedPath": str(path)})
    if not readiness["requiredFilesPresent"]:
        raise SemanticAnalysisError(
            f"The local MiniLM semantic model is missing or incomplete at {path}. "
            f"Run: {modelSetupCommands['semanticAnalysis']}"
        )
    try:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        model = AutoModel.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as error:
        raise SemanticAnalysisError(
            f"The local MiniLM model at {path} could not be loaded. TalonCV will not download it at runtime. "
            f"Technical detail: {error}"
        ) from error


def semanticDiagnostics(loadModel: bool = False) -> dict[str, Any]:
    section = getModelSection("semanticAnalysis")
    readiness = validateModelFiles("semanticAnalysis", section)
    device = selectDevice(section.get("device", "auto"))
    loadError = None
    loaded = _loadSemanticModel.cache_info().currsize > 0
    if loadModel and readiness["requiredFilesPresent"]:
        try:
            _loadSemanticModel(section["resolvedPath"], device)
            loaded = True
        except SemanticAnalysisError as error:
            loadError = str(error)
    return {
        **readiness,
        "device": device,
        "loaded": loaded,
        "loadError": loadError,
        "localFilesOnly": True,
        "runtimeDownloadsDisabled": True,
    }


def analyzeSemanticResponse(
    transcript: dict[str, Any] | None,
    sessionContext: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transcript = transcript or {}
    context = sessionContext or {}
    segments = [segment for segment in transcript.get("segments", []) if str(segment.get("text") or "").strip()]
    fullText = str(transcript.get("text") or "").strip()
    question = str(context.get("interviewQuestion") or "").strip()
    roleContext = " ".join(
        str(context.get(key) or "").strip()
        for key in ("targetRole", "jobDescription", "desiredCompetencies")
        if str(context.get(key) or "").strip()
    )
    if not fullText:
        return _unavailable("No transcript text was available for semantic analysis.", question, roleContext)

    section = getModelSection("semanticAnalysis")
    device = selectDevice(section.get("device", "auto"))
    try:
        tokenizer, model = _loadSemanticModel(section["resolvedPath"], device)
    except SemanticAnalysisError as error:
        result = _unavailable(str(error), question, roleContext)
        result["setupCommand"] = modelSetupCommands["semanticAnalysis"]
        return result

    segmentTexts = [str(segment["text"]).strip() for segment in segments]
    texts = [fullText, *segmentTexts]
    questionIndex = None
    roleIndex = None
    if question:
        questionIndex = len(texts)
        texts.append(question)
    if roleContext:
        roleIndex = len(texts)
        texts.append(roleContext)
    embeddings = _encode(texts, tokenizer, model, device, int(section.get("maxSequenceLength", 256)))
    answerEmbedding = embeddings[0]
    segmentEmbeddings = embeddings[1 : 1 + len(segmentTexts)]

    questionSimilarity = _cosine(answerEmbedding, embeddings[questionIndex]) if questionIndex is not None else None
    questionScore = _similarityScore(questionSimilarity)
    roleSimilarity = _cosine(answerEmbedding, embeddings[roleIndex]) if roleIndex is not None else None
    segmentAssessments = []
    for index, (segment, embedding) in enumerate(zip(segments, segmentEmbeddings)):
        relevanceSimilarity = _cosine(embedding, embeddings[questionIndex]) if questionIndex is not None else None
        roleSegmentSimilarity = _cosine(embedding, embeddings[roleIndex]) if roleIndex is not None else None
        text = segmentTexts[index]
        vague = _looksVague(text) or (relevanceSimilarity is not None and relevanceSimilarity < 0.18)
        segmentAssessments.append(
            {
                "startTime": float(segment.get("start", 0)),
                "endTime": float(segment.get("end", 0)),
                "text": text,
                "questionSimilarity": _round(relevanceSimilarity),
                "questionRelevanceScore": _similarityScore(relevanceSimilarity),
                "roleSimilarity": _round(roleSegmentSimilarity),
                "vagueOrOffTopic": vague,
                "marker": (
                    "mostRelevant"
                    if relevanceSimilarity is not None and relevanceSimilarity >= 0.45
                    else "possibleTopicDrift"
                    if relevanceSimilarity is not None and relevanceSimilarity < 0.18
                    else "vagueContent"
                    if vague
                    else "supportingContent"
                ),
            }
        )

    redundantPairs = []
    for left in range(len(segmentEmbeddings)):
        for right in range(left + 1, len(segmentEmbeddings)):
            similarity = _cosine(segmentEmbeddings[left], segmentEmbeddings[right])
            if similarity >= 0.86:
                redundantPairs.append(
                    {
                        "firstStartTime": float(segments[left].get("start", 0)),
                        "secondStartTime": float(segments[right].get("start", 0)),
                        "similarity": _round(similarity),
                        "explanation": "These transcript segments were semantically similar and may repeat the same point.",
                    }
                )

    mostRelevant = sorted(
        [item for item in segmentAssessments if item["questionSimilarity"] is not None],
        key=lambda item: item["questionSimilarity"],
        reverse=True,
    )[:3]
    topicDrift = [item for item in segmentAssessments if item["marker"] == "possibleTopicDrift"]
    vagueSegments = [item for item in segmentAssessments if item["vagueOrOffTopic"]]
    transcriptConfidence = transcript.get("averageConfidence")
    confidence = "high" if transcriptConfidence is not None and float(transcriptConfidence) >= 0.8 else "medium"
    if transcriptConfidence is None or float(transcriptConfidence) < 0.6:
        confidence = "low"
    return {
        "analysisVersion": semanticAnalysisVersion,
        "available": True,
        "modelPath": section["resolvedPath"],
        "device": device,
        "localFilesOnly": True,
        "questionRelevance": {
            "available": bool(question),
            "similarity": _round(questionSimilarity),
            "score": questionScore,
            "explanation": (
                "Local MiniLM similarity between the supplied question and the complete response."
                if question
                else "No question was supplied, so semantic question relevance is unavailable."
            ),
        },
        "roleAlignment": {
            "available": bool(roleContext),
            "similarity": _round(roleSimilarity),
            "score": _similarityScore(roleSimilarity),
            "explanation": (
                "Local MiniLM similarity to the supplied role, job description, and competency context."
                if roleContext
                else "No role or competency context was supplied, so role alignment is unavailable."
            ),
        },
        "segmentAssessments": segmentAssessments,
        "mostRelevantSegments": mostRelevant,
        "topicDriftSegments": topicDrift,
        "vagueSegments": vagueSegments,
        "semanticRedundancy": redundantPairs,
        "confidence": confidence,
        "warnings": (
            ["Semantic content findings have reduced confidence because transcript confidence was limited."]
            if confidence == "low"
            else []
        ),
    }


def _encode(texts, tokenizer, model, device, maximumLength):
    import torch

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=maximumLength,
        return_tensors="pt",
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model(**encoded)
    mask = encoded["attention_mask"].unsqueeze(-1).expand(output.last_hidden_state.size()).float()
    summed = torch.sum(output.last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    embeddings = summed / counts
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().cpu()


def _cosine(left, right) -> float:
    return float((left * right).sum().item())


def _similarityScore(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(100.0, (value - 0.05) / 0.65 * 100)), 1)


def _looksVague(text: str) -> bool:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    concrete = bool(
        re.search(r"\d|\b(i built|i led|i created|i implemented|i resolved|result|reduced|increased|improved|saved)\b", text.lower())
    )
    generic = bool(re.search(r"\b(things|stuff|somehow|really good|worked hard|team player|a lot)\b", text.lower()))
    return generic or (len(words) < 8 and not concrete)


def _round(value: float | None) -> float | None:
    return round(float(value), 5) if value is not None and math.isfinite(float(value)) else None


def _unavailable(reason: str, question: str, roleContext: str) -> dict[str, Any]:
    return {
        "analysisVersion": semanticAnalysisVersion,
        "available": False,
        "localFilesOnly": True,
        "questionRelevance": {"available": False, "score": None},
        "roleAlignment": {"available": False, "score": None},
        "segmentAssessments": [],
        "mostRelevantSegments": [],
        "topicDriftSegments": [],
        "vagueSegments": [],
        "semanticRedundancy": [],
        "warnings": [reason],
        "questionSupplied": bool(question),
        "roleContextSupplied": bool(roleContext),
        "confidence": "unavailable",
    }
