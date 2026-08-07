import argparse
import json
import sys
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from src.audioPipeline.semanticAnalyzer import semanticDiagnostics
from src.audioPipeline.transcription import transcriptionDiagnostics
from src.cvPipeline.yoloFaceDetector import yoloDiagnostics
from src.localModels.config import loadModelConfig, validateModelFiles
from src.localModels.localCoach import localCoachDiagnostics


modelNames = ("transcription", "faceDetection", "semanticAnalysis", "localCoach")


def verifyLocalModels(loadModels=True, selectedModel=None):
    config = loadModelConfig()
    names = [selectedModel] if selectedModel else list(modelNames)
    checks = {}
    for name in names:
        structural = validateModelFiles(name, config[name])
        if loadModels and structural["requiredFilesPresent"]:
            if name == "transcription":
                loaded = transcriptionDiagnostics(loadModel=True)
            elif name == "faceDetection":
                loaded = yoloDiagnostics(loadModel=True)
            elif name == "semanticAnalysis":
                loaded = semanticDiagnostics(loadModel=True)
            else:
                loaded = localCoachDiagnostics(loadModel=True)
            checks[name] = {**structural, **loaded}
        else:
            checks[name] = {**structural, "loaded": False, "loadError": None}
    ready = all(
        item["requiredFilesPresent"] and (not loadModels or item.get("loaded", False)) and not item.get("loadError")
        for item in checks.values()
    )
    return {
        "ready": ready,
        "loadModels": loadModels,
        "configPath": config["configPath"],
        "runtimeNetworkingDisabled": config["runtime"]["networkingDisabled"],
        "models": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="Verify TalonCV local model files and offline loading.")
    parser.add_argument("--files-only", action="store_true", help="Verify files without loading model weights.")
    parser.add_argument("--model", choices=modelNames, help="Verify only one configured model.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress normal text output.")
    args = parser.parse_args()
    result = verifyLocalModels(loadModels=not args.files_only, selectedModel=args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    elif not args.quiet:
        print(f"Configuration: {result['configPath']}")
        print("Runtime networking: disabled")
        for name, item in result["models"].items():
            fileStatus = "ready" if item["requiredFilesPresent"] else "missing"
            loadStatus = "loaded" if item.get("loaded") else "not loaded"
            if args.files_only:
                loadStatus = "load check skipped"
            print(f"{name}: {fileStatus}; {loadStatus}; {item['sizeGb']:.3f} GB; {item['path']}")
            if item.get("missingFiles"):
                print(f"  Missing: {', '.join(item['missingFiles'])}")
                print(f"  Setup: {item['setupCommand']}")
            if item.get("loadError"):
                print(f"  Load error: {item['loadError']}")
        print("READY" if result["ready"] else "NOT READY")
    raise SystemExit(0 if result["ready"] else 1)


if __name__ == "__main__":
    main()
