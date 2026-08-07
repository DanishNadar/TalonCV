import argparse
import json
import os
import sys
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))


def verifyProductionModels(loadModels: bool = True) -> dict:
    configPath = Path(os.environ.get("TALONCV_MODEL_CONFIG", projectRoot / "config" / "models.production.json"))
    os.environ["TALONCV_MODEL_CONFIG"] = str(configPath)
    os.environ["TALONCV_ENV"] = "production"
    from src.localModels.config import loadModelConfig, validateModelFiles

    config = loadModelConfig(str(configPath))
    checks = {
        name: validateModelFiles(name, config[name])
        for name in ("transcription", "faceDetection", "semanticAnalysis", "localCoach")
    }
    result = {"configPath": str(configPath.resolve()), "checks": checks, "allFilesReady": all(item["requiredFilesPresent"] for item in checks.values())}
    if loadModels and result["allFilesReady"]:
        from src.audioPipeline.semanticAnalyzer import semanticDiagnostics
        from src.audioPipeline.transcription import transcriptionDiagnostics
        from src.cvPipeline.yoloFaceDetector import yoloDiagnostics
        from src.localModels.localCoach import localCoachDiagnostics

        result["loads"] = {
            "transcription": transcriptionDiagnostics(loadModel=True),
            "faceDetection": yoloDiagnostics(loadModel=True),
            "semanticAnalysis": semanticDiagnostics(loadModel=True),
            "localCoach": localCoachDiagnostics(loadModel=True),
        }
        result["allModelsLoaded"] = all(item.get("loaded", False) for item in result["loads"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify production model files and local-only CPU loads.")
    parser.add_argument("--metadata-only", action="store_true", help="Check files without allocating the models.")
    options = parser.parse_args()
    result = verifyProductionModels(loadModels=not options.metadata_only)
    print(json.dumps(result, indent=2))
    if not result["allFilesReady"] or (not options.metadata_only and not result.get("allModelsLoaded")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
