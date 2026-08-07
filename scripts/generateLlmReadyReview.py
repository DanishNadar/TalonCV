import json
import sys
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.cvPipeline.cueDefinitions import cueDefinitions
from src.multimodalPipeline.artifacts import readJson


llmReadyPath = projectRoot / "data" / "demo" / "llmReady"


def getInputPath():
    if len(sys.argv) < 2:
        print("Please provide an events JSON path or matching review Markdown path.")
        print("Example:")
        print("python scripts/generateLlmReadyReview.py data/demo/events/YOUR_EVENTS_FILE.json")
        raise SystemExit(1)

    inputPath = Path(sys.argv[1])

    if not inputPath.is_absolute():
        inputPath = projectRoot / inputPath

    if not inputPath.exists():
        print(f"Input file does not exist: {inputPath}")
        raise SystemExit(1)

    return inputPath


def getEventsPath(inputPath):
    if inputPath.suffix.lower() == ".json":
        return inputPath

    outputStem = inputPath.stem

    if outputStem.endswith("_review"):
        outputStem = outputStem[: -len("_review")]

    eventsPath = projectRoot / "data" / "demo" / "events" / f"{outputStem}_events.json"

    if eventsPath.exists():
        print(f"Using matching events JSON: {eventsPath}")
        return eventsPath

    print("The LLM-ready generator needs an events JSON file.")
    print(f"You passed: {inputPath}")
    print(f"I also looked for: {eventsPath}")
    print("Run analyzeInterviewDemo.py first, then pass the matching file from data/demo/events.")
    raise SystemExit(1)


def readEvents(eventsPath):
    try:
        with eventsPath.open() as eventsFile:
            return json.load(eventsFile)
    except json.JSONDecodeError as error:
        print(f"Could not read events JSON: {eventsPath}")
        print("Make sure you pass a .json file from data/demo/events, not a Markdown report.")
        raise SystemExit(1) from error


def getOutputStem(eventsPath):
    outputStem = eventsPath.stem

    if outputStem.endswith("_events"):
        outputStem = outputStem[: -len("_events")]
    elif outputStem.endswith("_multimodal"):
        outputStem = outputStem[: -len("_multimodal")]

    return outputStem


def buildLlmReadyJson(events, multimodalAnalysis=None):
    multimodalAnalysis = multimodalAnalysis or {}
    return {
        "reviewType": "multimodalInterviewPracticeAnalysis",
        "safetyInstructions": [
            "Use cue definitions for interpretation.",
            "Give interview coaching feedback.",
            "Be specific with timestamps.",
            "Be constructive.",
            "Do not make claims about personality, honesty, intelligence, professionalism, hiring suitability, or emotion.",
            "Use careful phrases such as visually appeared, may be worth reviewing, and could practice.",
            "Separate verbal, vocal, visual, and combined observations.",
            "Do not invent events or treat low-confidence transcription as definite wording.",
            "Do not penalize accents, dialects, non-native speech, or speech differences.",
            "Do not make hiring recommendations.",
        ],
        "cueDefinitions": cueDefinitions,
        "sessionContext": multimodalAnalysis.get("sessionContext", {}),
        "transcript": multimodalAnalysis.get("transcript", {}),
        "responseAnalysis": multimodalAnalysis.get("responseAnalysis", {}),
        "audioFeatures": multimodalAnalysis.get("audioFeatures", {}),
        "audioEvents": multimodalAnalysis.get("audioEvents", []),
        "events": events,
        "multimodalMoments": multimodalAnalysis.get("moments", []),
        "scores": multimodalAnalysis.get("scores", {}),
        "dataQualityWarnings": multimodalAnalysis.get("warnings", []),
    }


def buildPrompt(events, multimodalAnalysis=None):
    payload = buildLlmReadyJson(events, multimodalAnalysis)
    payloadJson = json.dumps(payload, indent=2)

    return f"""You are helping review a multimodal interview-practice recording.

Use only the structured evidence below. Give interview coaching feedback that is specific, constructive, and grounded in supplied timestamps.

Do not make claims about personality, honesty, intelligence, professionalism, hiring suitability, or true emotional state.
Do not diagnose emotions.
Do not penalize accents, dialects, non-native speech, informal but understandable phrasing, or speech differences.
Do not interpret low-confidence transcription as definite wording, and do not invent missing events.
Do not make hiring recommendations.
Use careful language such as "the recording showed", "may be worth reviewing", and "could practice".

Structured local analysis:

{payloadJson}

Write concise coaching feedback with separate sections for:
- Verbal response strengths and review areas
- Vocal delivery and recording-quality observations
- Visual strengths and review moments
- Multimodal alignment moments
- A stronger answer structure
- Three prioritized practice steps

Ground each substantive claim in the supplied evidence and timestamps. Raw audio and video were not provided to you.
"""


def saveLlmReadyFiles(eventsPath, events, multimodalAnalysis=None):
    llmReadyPath.mkdir(parents=True, exist_ok=True)
    outputStem = getOutputStem(eventsPath)
    jsonPath = llmReadyPath / f"{outputStem}_llmReady.json"
    promptPath = llmReadyPath / f"{outputStem}_llmReadyPrompt.txt"

    llmReadyJson = buildLlmReadyJson(events, multimodalAnalysis)
    promptText = buildPrompt(events, multimodalAnalysis)

    with jsonPath.open("w") as jsonFile:
        json.dump(llmReadyJson, jsonFile, indent=2)

    promptPath.write_text(promptText)

    return jsonPath, promptPath


def generateLlmReadyReview():
    inputPath = getInputPath()
    directPayload = readJson(inputPath, None) if inputPath.suffix.lower() == ".json" else None
    if isinstance(directPayload, dict) and (
        "moments" in directPayload or "responseAnalysis" in directPayload or "mediaInfo" in directPayload
    ):
        eventsPath = inputPath
        events = directPayload.get("visualEvents", [])
        multimodalAnalysis = directPayload
    else:
        eventsPath = getEventsPath(inputPath)
        events = readEvents(eventsPath)
        outputStem = getOutputStem(eventsPath)
        multimodalPath = projectRoot / "data" / "demo" / "multimodal" / f"{outputStem}_multimodal.json"
        multimodalAnalysis = readJson(multimodalPath, {}) or {}
    jsonPath, promptPath = saveLlmReadyFiles(eventsPath, events, multimodalAnalysis)

    print("LLM-ready review files created.")
    print(f"LLM-ready JSON saved to: {jsonPath}")
    print(f"LLM-ready prompt saved to: {promptPath}")


if __name__ == "__main__":
    generateLlmReadyReview()
