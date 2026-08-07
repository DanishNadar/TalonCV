import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))

from scripts.recordInterviewDemo import (  # noqa: E402
    getVideoWriter,
    openWebcam,
    padRemainingFrames,
    recordingPath,
    writeTimedFrames,
)
from src.cvPipeline.reportUtils import reviewEventTypes, strengthEventTypes  # noqa: E402


defaultTakeSeconds = 12


def getSessionOptions():
    parser = argparse.ArgumentParser(
        description="Record multiple short interview takes in one sitting for ground-truth labeling."
    )
    parser.add_argument("--seconds", type=float, default=defaultTakeSeconds, help="Length of each take in seconds.")
    parser.add_argument("--fps", type=float, default=30, help="Output video FPS.")
    parser.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip running the MediaPipe/YOLOv11 analysis pipeline after each take.",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        print("--seconds must be greater than 0.")
        raise SystemExit(1)
    if args.fps <= 0:
        print("--fps must be greater than 0.")
        raise SystemExit(1)

    return args.seconds, args.fps, not args.no_analyze


def printAvailableIndicators():
    print("Available indicators (act one or more out per take, then label them afterward):")
    print("  Strength:", ", ".join(sorted(strengthEventTypes)))
    print("  Review:", ", ".join(sorted(reviewEventTypes)))
    print()
    print("Label the take with scripts/labelGroundTruth.py (Streamlit) once it is recorded.")
    print()


def recordOneTake(durationSeconds, outputFps, outputPath=None, windowTitle="Labeling session recording - press q to stop"):
    recordingPath.mkdir(parents=True, exist_ok=True)
    if outputPath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputPath = recordingPath / f"interviewDemo_{timestamp}.mp4"
    else:
        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)

    videoCapture = openWebcam()
    videoWriter = getVideoWriter(videoCapture, outputPath, outputFps)
    startTime = time.monotonic()
    framesWritten = 0
    lastFrame = None
    stoppedEarly = False

    print(f"Recording take. {durationSeconds:g}s, press q to stop early.")

    while True:
        success, frame = videoCapture.read()
        if not success:
            print("Could not read a frame from the webcam.")
            break

        lastFrame = frame
        framesWritten = writeTimedFrames(
            videoWriter=videoWriter,
            frame=frame,
            framesWritten=framesWritten,
            startTime=startTime,
            durationSeconds=durationSeconds,
            outputFps=outputFps,
        )

        cv2.imshow(windowTitle, frame)

        if time.monotonic() - startTime >= durationSeconds:
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stoppedEarly = True
            break

    if not stoppedEarly and lastFrame is not None:
        framesWritten = padRemainingFrames(
            videoWriter=videoWriter,
            frame=lastFrame,
            framesWritten=framesWritten,
            durationSeconds=durationSeconds,
            outputFps=outputFps,
        )

    videoCapture.release()
    videoWriter.release()
    cv2.destroyAllWindows()

    print(f"Saved take: {outputPath}")
    return outputPath


def analyzeTake(videoPath, mp):
    from scripts.analyzeInterviewDemo import analyzeVideo, createEvents, getOutputStem, saveEvents, saveFeatures, saveReport

    outputStem = getOutputStem(videoPath)
    rows, durationSeconds = analyzeVideo(videoPath, mp)
    events = createEvents(rows, durationSeconds)
    featurePath = saveFeatures(rows, outputStem)
    saveEvents(events, outputStem)
    saveReport(videoPath, rows, events, durationSeconds, outputStem)
    print(f"Analyzed take. Feature CSV: {featurePath}")


def recordLabelingSession():
    durationSeconds, outputFps, shouldAnalyze = getSessionOptions()
    printAvailableIndicators()

    mp = None
    if shouldAnalyze:
        from scripts.analyzeInterviewDemo import loadMediaPipe

        mp = loadMediaPipe()

    recordedPaths = []
    takeNumber = 1

    while True:
        response = input(f"Press Enter to record take {takeNumber} ({durationSeconds:g}s), or type 'done' to stop: ")
        if response.strip().lower() in {"done", "d", "q", "quit", "exit"}:
            break

        videoPath = recordOneTake(durationSeconds, outputFps)
        recordedPaths.append(videoPath)

        if shouldAnalyze:
            analyzeTake(videoPath, mp)

        takeNumber += 1

    if not recordedPaths:
        print("No takes were recorded.")
        return

    print()
    print(f"Session complete. Recorded {len(recordedPaths)} take(s):")
    for path in recordedPaths:
        print(f"  {path}")
    print()
    print("Next: label the ground-truth indicators for these takes with:")
    print("  python -m streamlit run scripts/labelGroundTruth.py")


if __name__ == "__main__":
    recordLabelingSession()
