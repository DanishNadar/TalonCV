import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from src.localModels.config import getModelSection, modelSetupCommands, selectDevice, validateModelFiles
from src.localModels.llamaCppCoach import (
    LlamaCppCoachError,
    generateLlamaCppText,
    loadLlamaCppCoach,
    unloadLlamaCppCoach,
)


localCoachVersion = "local-coach-v2"


class LocalCoachError(RuntimeError):
    """Actionable local coaching-model error."""


@lru_cache(maxsize=2)
def _loadLocalCoach(modelPath: str, device: str):
    path = Path(modelPath)
    readiness = validateModelFiles("localCoach", {"resolvedPath": str(path)})
    if not readiness["requiredFilesPresent"]:
        raise LocalCoachError(
            f"The local Qwen coaching model is missing or incomplete at {path}. "
            f"Run: {modelSetupCommands['localCoach']}"
        )
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False
        )
        dtype = torch.float16 if device == "cuda" else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        return tokenizer, model
    except Exception as error:
        raise LocalCoachError(
            f"The local coaching model at {path} could not be loaded. TalonCV will not download it at runtime. "
            f"Technical detail: {error}"
        ) from error


def localCoachDiagnostics(loadModel: bool = False) -> dict[str, Any]:
    section = getModelSection("localCoach")
    readiness = validateModelFiles("localCoach", section)
    backend = str(section.get("backend", "transformers")).lower()
    device = "cpu" if backend == "llama.cpp" else selectDevice(section.get("device", "auto"))
    loadError = None
    loaded = (
        loadLlamaCppCoach.cache_info().currsize > 0
        if backend == "llama.cpp"
        else _loadLocalCoach.cache_info().currsize > 0
    )
    if loadModel and section.get("enabled", True) and readiness["requiredFilesPresent"]:
        try:
            if backend == "llama.cpp":
                from src.localModels.cpuRuntime import getCpuRuntimeConfiguration

                cpu = getCpuRuntimeConfiguration()
                configured = section.get("threads", "auto")
                threads = cpu.workerThreads if str(configured).lower() == "auto" else int(configured)
                loadLlamaCppCoach(section["resolvedPath"], int(section.get("contextSize", 2048)), threads)
            else:
                _loadLocalCoach(section["resolvedPath"], device)
            loaded = True
        except (LocalCoachError, LlamaCppCoachError) as error:
            loadError = str(error)
    return {
        **readiness,
        "enabled": bool(section.get("enabled", True)),
        "device": device,
        "backend": backend,
        "loaded": loaded,
        "loadError": loadError,
        "localFilesOnly": True,
        "runtimeDownloadsDisabled": True,
        "maxInputTokens": int(section.get("maxInputTokens", section.get("contextSize", 2048))),
        "maxNewTokens": int(section.get("maxNewTokens", 320)),
    }


def generateLocalCoaching(
    analysis: dict[str, Any],
    progressCallback: Callable[[str], None] | None = None,
    maxNewTokens: int | None = None,
) -> dict[str, Any]:
    section = getModelSection("localCoach")
    if not section.get("enabled", True):
        return {
            "analysisVersion": localCoachVersion,
            "available": False,
            "warnings": ["Local enhanced coaching is disabled in config/models.json."],
        }
    backend = str(section.get("backend", "transformers")).lower()
    device = "cpu" if backend == "llama.cpp" else selectDevice(section.get("device", "auto"))
    if progressCallback:
        progressCallback("Loading the local Qwen coaching model from disk...")
    prompt = buildLocalCoachPrompt(analysis)
    if progressCallback:
        progressCallback("Generating evidence-constrained coaching locally...")
    messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful interview-practice coach. Use only supplied evidence. "
                    "Never reveal chain-of-thought; provide concise conclusions and recommendations only."
                ),
            },
            {"role": "user", "content": prompt},
        ]
    generationLimit = int(maxNewTokens or section.get("maxNewTokens", 320))
    try:
        if backend == "llama.cpp":
            rawText, generation = generateLlamaCppText(messages, section, generationLimit)
        else:
            tokenizer, model = _loadLocalCoach(section["resolvedPath"], device)
            import torch

            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=int(section.get("maxInputTokens", 768)),
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max(32, min(generationLimit, 700)),
                    do_sample=False,
                    repetition_penalty=float(section.get("repetitionPenalty", 1.05)),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generatedTokens = output[0][inputs["input_ids"].shape[1] :]
            rawText = tokenizer.decode(generatedTokens, skip_special_tokens=True).strip()
            generation = {
                "backend": "transformers",
                "inputTokens": int(inputs["input_ids"].shape[1]),
                "maxNewTokens": max(32, min(generationLimit, 700)),
                "doSample": False,
                "repetitionPenalty": float(section.get("repetitionPenalty", 1.05)),
            }
    except Exception as error:
        raise LocalCoachError(f"Local coaching generation failed: {error}") from error
    cleaned = _stripInternalReasoning(rawText)
    allowedTimestamps = collectEvidenceTimestamps(analysis)
    sanitized, removedLines = sanitizeUnsupportedTimestamps(cleaned, allowedTimestamps)
    if not sanitized:
        raise LocalCoachError(
            "The local coach did not produce usable evidence-constrained text. Deterministic analysis remains available."
        )
    warnings = []
    if removedLines:
        warnings.append(
            f"Removed {removedLines} generated line(s) containing timestamps absent from the supplied evidence."
        )
    return {
        "analysisVersion": localCoachVersion,
        "available": True,
        "text": sanitized,
        "modelPath": section["resolvedPath"],
        "device": device,
        "backend": backend,
        "generation": generation,
        "localFilesOnly": True,
        "evidenceTimestampCount": len(allowedTimestamps),
        "warnings": warnings,
        "safetyNote": (
            "This locally generated wording is grounded in deterministic evidence and is interview-practice coaching, "
            "not a hiring recommendation or psychological assessment."
        ),
    }


def generateLocalCoachingFromPrompt(
    prompt: str,
    progressCallback: Callable[[str], None] | None = None,
    maxNewTokens: int | None = None,
) -> dict[str, Any]:
    section = getModelSection("localCoach")
    if not section.get("enabled", True):
        raise LocalCoachError("Local enhanced coaching is disabled in config/models.json.")
    backend = str(section.get("backend", "transformers")).lower()
    device = "cpu" if backend == "llama.cpp" else selectDevice(section.get("device", "auto"))
    if progressCallback:
        progressCallback("Loading the local Qwen coaching model from disk...")
    messages = [
        {
            "role": "system",
            "content": (
                "Use only the supplied structured interview evidence. Never invent facts, timestamps, "
                "psychological claims, or hiring recommendations. Do not output hidden reasoning."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        generationLimit = max(32, min(int(maxNewTokens or section.get("maxNewTokens", 320)), 700))
        if backend == "llama.cpp":
            text, generation = generateLlamaCppText(messages, section, generationLimit)
        else:
            tokenizer, model = _loadLocalCoach(section["resolvedPath"], device)
            import torch

            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(
                rendered,
                return_tensors="pt",
                truncation=True,
                max_length=int(section.get("maxInputTokens", 768)),
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=generationLimit,
                    do_sample=False,
                    repetition_penalty=float(section.get("repetitionPenalty", 1.05)),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            generation = {
                "backend": "transformers",
                "inputTokens": int(inputs["input_ids"].shape[1]),
                "maxNewTokens": generationLimit,
                "doSample": False,
                "repetitionPenalty": float(section.get("repetitionPenalty", 1.05)),
            }
    except Exception as error:
        raise LocalCoachError(f"Local coaching generation failed: {error}") from error
    allowed = sorted(
        {
            float(value)
            for value in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*s\b", prompt, flags=re.IGNORECASE)
        }
    )
    sanitized, removed = sanitizeUnsupportedTimestamps(_stripInternalReasoning(text), allowed)
    if not sanitized:
        raise LocalCoachError("The local coach did not produce usable evidence-constrained text.")
    return {
        "analysisVersion": localCoachVersion,
        "available": True,
        "text": sanitized,
        "modelPath": section["resolvedPath"],
        "device": device,
        "backend": backend,
        "generation": generation,
        "localFilesOnly": True,
        "warnings": [f"Removed {removed} unsupported timestamp line(s)."] if removed else [],
    }


def buildLocalCoachPrompt(analysis: dict[str, Any]) -> str:
    response = analysis.get("responseAnalysis", {})
    semantic = analysis.get("semanticAnalysis", {})
    transcript = analysis.get("transcript", {})
    session = analysis.get("sessionContext", {})
    audioFeatures = analysis.get("audioFeatures", {})
    development = response.get("answerDevelopment", {})
    moments = sorted(
        analysis.get("moments", []),
        key=lambda item: 0 if item.get("classification") in {"strength", "review"} else 1,
    )
    payload = {
        "context": {
            "question": str(session.get("interviewQuestion") or "")[:180],
            "role": str(session.get("targetRole") or "")[:100],
            "job": str(session.get("jobDescription") or "")[:160],
            "competencies": str(session.get("desiredCompetencies") or "")[:120],
        },
        "transcript": {
            "confidence": transcript.get("averageConfidence"),
            "segments": [
                {
                    "time": f"{float(item.get('start', 0)):.2f}-{float(item.get('end', 0)):.2f}s",
                    "text": str(item.get("text") or "")[:120],
                    "confidence": item.get("confidence"),
                }
                for item in transcript.get("segments", [])[:4]
            ],
        },
        "answer": {
            "overallScore": response.get("rubric", {}).get("overallVerbalResponse", {}).get("score"),
            "relevanceScore": response.get("relevanceAnalysis", {}).get("score"),
            "roleScore": response.get("roleAlignmentAnalysis", {}).get("score"),
            "star": response.get("starAnalysis", {}).get("components", {}),
            "missing": development.get("missingElements", []),
            "strong": [
                {
                    "time": f"{float(item.get('startTime', 0)):.2f}s",
                    "text": str(item.get("text") or "")[:120],
                }
                for item in response.get("strongPhrases", [])[:2]
            ],
            "improve": [str(item)[:140] for item in response.get("practiceAreas", [])[:2]],
            "vagueTimes": [item.get("startTime") for item in semantic.get("vagueSegments", [])[:2]],
        },
        "audio": {
            "metrics": {
                "rmsDb": audioFeatures.get("overallRmsDb"),
                "snrDb": audioFeatures.get("snrProxyDb"),
                "clippingPct": audioFeatures.get("clippingPercentage"),
                "wpm": audioFeatures.get("speechRateWpm"),
                "longPauses": audioFeatures.get("longPauseCount"),
            },
            "events": [
                [item.get("eventType"), item.get("startTime"), item.get("endTime"), item.get("reliability")]
                for item in analysis.get("audioEvents", [])[:2]
            ],
        },
        "visualEvents": [
            [item.get("eventType"), item.get("startTime"), item.get("endTime")]
            for item in analysis.get("visualEvents", [])[:2]
        ],
        "moments": [
            {
                "time": f"{float(item.get('startTime', 0)):.2f}s",
                "class": item.get("classification"),
                "audio": item.get("audioEvents", []),
                "visual": item.get("visualEvents", []),
                "recommendation": str(item.get("coachingRecommendation") or "")[:120],
            }
            for item in moments[:2]
        ],
        "scores": {
            name: item.get("score")
            for name, item in analysis.get("scores", {}).get("scores", {}).items()
        },
        "warning": str((analysis.get("warnings") or [""])[0])[:120],
    }
    return f"""Use only this evidence for concise interview coaching. Give strengths, content/organization, recording versus vocal delivery, visual/multimodal delivery, three priorities, an outline, a clearly labeled example using only speaker facts, and supplied replay times. Never invent facts/times or infer traits, emotion, honesty, intelligence, or hiring suitability. Respect low transcript confidence and speech differences. No hidden reasoning.
{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}
"""


def collectEvidenceTimestamps(analysis: dict[str, Any]) -> list[float]:
    values = {0.0}
    collections = [
        analysis.get("transcript", {}).get("segments", []),
        analysis.get("audioEvents", []),
        analysis.get("visualEvents", []),
        analysis.get("moments", []),
        analysis.get("responseAnalysis", {}).get("strongPhrases", []),
        analysis.get("responseAnalysis", {}).get("fillerOccurrences", []),
    ]
    for collection in collections:
        for item in collection:
            for key in ("start", "end", "startTime", "endTime"):
                if item.get(key) is not None:
                    values.add(round(float(item[key]), 2))
    return sorted(values)


def sanitizeUnsupportedTimestamps(text: str, allowedTimestamps: list[float]) -> tuple[str, int]:
    allowed = [float(value) for value in allowedTimestamps]
    output = []
    removed = 0
    for line in text.splitlines():
        mentioned = [float(value) for value in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*s\b", line, flags=re.IGNORECASE)]
        if mentioned and any(not any(abs(value - candidate) <= 0.06 for candidate in allowed) for value in mentioned):
            removed += 1
            continue
        output.append(line.rstrip())
    return "\n".join(output).strip(), removed


def _stripInternalReasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:analysis|reasoning).*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def unloadLocalCoach() -> None:
    """Release a cached coach when a constrained worker is under memory pressure."""
    _loadLocalCoach.cache_clear()
    unloadLlamaCppCoach()
