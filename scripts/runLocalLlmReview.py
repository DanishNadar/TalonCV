import sys
from datetime import datetime, timezone
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.localModels.localCoach import LocalCoachError, generateLocalCoachingFromPrompt
from src.multimodalPipeline.artifacts import writeJson, writeText


def getPromptPath():
    if len(sys.argv) < 2:
        print("Provide a local LLM-ready prompt file.")
        print("Example: python scripts/runLocalLlmReview.py data/demo/llmReady/<stem>_llmReadyPrompt.txt")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    if not path.is_absolute():
        path = projectRoot / path
    if not path.exists():
        print(f"Prompt file does not exist: {path}")
        raise SystemExit(1)
    return path


def outputStem(promptPath):
    stem = promptPath.stem
    return stem[: -len("_llmReadyPrompt")] if stem.endswith("_llmReadyPrompt") else stem


def runLocalReview(promptPath):
    try:
        coaching = generateLocalCoachingFromPrompt(promptPath.read_text(encoding="utf-8"), print)
    except LocalCoachError as error:
        print(str(error))
        raise SystemExit(1) from error
    stem = outputStem(promptPath)
    reportPath = projectRoot / "reports" / f"{stem}_local_coaching.md"
    metaPath = projectRoot / "reports" / f"{stem}_local_coaching_meta.json"
    report = "\n".join(
        [
            "# TalonCV Local Enhanced Coaching",
            "",
            coaching["text"],
            "",
            "Generated entirely on this machine from structured evidence; no raw media was provided to the model.",
            "",
        ]
    )
    writeText(reportPath, report)
    writeJson(
        metaPath,
        {
            "analysisVersion": coaching["analysisVersion"],
            "modelPath": coaching["modelPath"],
            "device": coaching["device"],
            "generation": coaching["generation"],
            "localFilesOnly": True,
            "promptPath": str(promptPath.resolve()),
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
    )
    return reportPath, metaPath


def main():
    promptPath = getPromptPath()
    reportPath, metaPath = runLocalReview(promptPath)
    print(f"Local coaching report: {reportPath}")
    print(f"Local coaching metadata: {metaPath}")


if __name__ == "__main__":
    main()
