import argparse
import contextlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))


def values(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def runSingle(mediaPath: Path) -> dict:
    import csv
    import psutil

    from src.multimodalPipeline.pipeline import runMultimodalAnalysis

    started = time.perf_counter()
    last = started
    active = "startup"
    durations = {}

    def progress(stage, _fraction, _message):
        nonlocal active, last
        now = time.perf_counter()
        durations[active] = durations.get(active, 0) + now - last
        active = stage
        last = now

    with contextlib.redirect_stdout(sys.stderr):
        result = runMultimodalAnalysis(mediaPath, force=True, progressCallback=progress)
    now = time.perf_counter()
    durations[active] = durations.get(active, 0) + now - last
    duration = float(result.get("mediaInfo", {}).get("durationSeconds") or 0)
    transcriptionSeconds = durations.get("transcribeLocally", 0)
    visualFeaturePath = Path(result.get("artifactPaths", {}).get("visualFeatures", ""))
    analyzedFrames = 0
    if visualFeaturePath.is_file():
        with visualFeaturePath.open("r", encoding="utf-8", newline="") as stream:
            analyzedFrames = max(0, sum(1 for _row in csv.reader(stream)) - 1)
    visualSeconds = durations.get("analyzeVisualDelivery", 0)
    return {
        "cpuModel": platform.processor() or platform.machine(),
        "cpuCount": psutil.cpu_count(logical=True),
        "ramAvailableBytes": psutil.virtual_memory().available,
        "configuration": {
            "visualFps": float(os.environ.get("TALONCV_VISUAL_ANALYSIS_FPS", 3)),
            "yoloImageSize": int(os.environ.get("TALONCV_YOLO_IMAGE_SIZE", 480)),
            "whisperBeam": int(os.environ.get("TALONCV_WHISPER_BEAM", 3)),
        },
        "audioDurationSeconds": duration,
        "transcriptionSeconds": round(transcriptionSeconds, 4),
        "transcriptionRealtimeFactor": round(transcriptionSeconds / duration, 4) if duration else None,
        "visualSeconds": round(visualSeconds, 4),
        "visualFramesPerSecondProcessed": round(analyzedFrames / visualSeconds, 4) if visualSeconds else None,
        "semanticSeconds": round(durations.get("analyzeSemanticRelevance", 0), 4),
        "qwenSeconds": round(durations.get("generateLocalCoaching", 0), 4),
        "analyzedFrameCount": analyzedFrames,
        "transcriptWordCount": len(str(result.get("transcript", {}).get("text") or "").split()),
        "processRssBytes": psutil.Process().memory_info().rss,
        "stageDurations": {key: round(value, 4) for key, value in durations.items()},
        "totalAnalysisSeconds": round(time.perf_counter() - started, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TalonCV's sequential CPU pipeline.")
    parser.add_argument("media", type=Path)
    parser.add_argument("--visual-fps", default="3", help="Comma-separated values, e.g. 2,3")
    parser.add_argument("--yolo-size", default="480", help="Comma-separated values, e.g. 416,480,640")
    parser.add_argument("--whisper-beam", default="3", help="Comma-separated values, e.g. 1,3,5")
    parser.add_argument("--coach-tokens", type=int, default=96, help="Bound local Qwen output during benchmarks")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args()
    if options.single:
        print(json.dumps(runSingle(options.media.resolve()), indent=2))
        return
    combinations = itertools.product(
        values(options.visual_fps, float), values(options.yolo_size, int), values(options.whisper_beam, int)
    )
    results = []
    for visualFps, yoloSize, whisperBeam in combinations:
        environment = os.environ.copy()
        environment.update(
            TALONCV_VISUAL_ANALYSIS_FPS=str(visualFps),
            TALONCV_YOLO_IMAGE_SIZE=str(yoloSize),
            TALONCV_WHISPER_BEAM=str(whisperBeam),
            TALONCV_COACH_MAX_NEW_TOKENS=str(max(32, min(options.coach_tokens, 700))),
        )
        command = [sys.executable, __file__, str(options.media), "--single"]
        completed = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
        results.append(json.loads(completed.stdout))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
