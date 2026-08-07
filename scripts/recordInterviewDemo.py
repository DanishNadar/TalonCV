import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


projectRoot = Path(__file__).resolve().parents[1]
recordingPath = projectRoot / "data" / "demo" / "recordings"
defaultOutputFps = 30


def getRecordingOptions():
    parser = argparse.ArgumentParser(description="Record a short webcam interview demo.")
    parser.add_argument("duration", nargs="?", type=float, help="Optional recording length in seconds.")
    parser.add_argument("--seconds", type=float, help="Optional recording length in seconds.")
    parser.add_argument("--fps", type=float, default=defaultOutputFps, help="Output video FPS.")
    args = parser.parse_args()

    durationSeconds = args.seconds if args.seconds is not None else args.duration

    if durationSeconds is not None and durationSeconds <= 0:
        print("Duration must be greater than 0 seconds.")
        raise SystemExit(1)

    if args.fps <= 0:
        print("FPS must be greater than 0.")
        raise SystemExit(1)

    return durationSeconds, args.fps


def openWebcam():
    videoCapture = cv2.VideoCapture(0)

    if not videoCapture.isOpened():
        print("Could not open webcam.")
        raise SystemExit(1)

    return videoCapture


def getVideoWriter(videoCapture, outputPath, outputFps):
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(str(outputPath), fourcc, outputFps, (width, height))

    if not videoWriter.isOpened():
        print("Could not create the video writer.")
        raise SystemExit(1)

    return videoWriter


def writeTimedFrames(videoWriter, frame, framesWritten, startTime, durationSeconds, outputFps):
    targetFrameCount = int(round(durationSeconds * outputFps))
    elapsedSeconds = time.monotonic() - startTime
    expectedFrameCount = min(targetFrameCount, int(elapsedSeconds * outputFps) + 1)

    while framesWritten < expectedFrameCount:
        videoWriter.write(frame)
        framesWritten += 1

    return framesWritten


def padRemainingFrames(videoWriter, frame, framesWritten, durationSeconds, outputFps):
    targetFrameCount = int(round(durationSeconds * outputFps))

    while framesWritten < targetFrameCount:
        videoWriter.write(frame)
        framesWritten += 1

    return framesWritten


def recordInterviewDemo():
    durationSeconds, outputFps = getRecordingOptions()
    recordingPath.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputPath = recordingPath / f"interviewDemo_{timestamp}.mp4"

    videoCapture = openWebcam()
    videoWriter = getVideoWriter(videoCapture, outputPath, outputFps)
    startTime = time.monotonic()
    framesWritten = 0
    lastFrame = None
    stoppedEarly = False

    print("Recording interview demo.")

    if durationSeconds is None:
        print("Press q in the preview window to stop.")
    else:
        print(f"Recording for {durationSeconds:g} seconds. Press q to stop early.")
        print(f"Saving video at {outputFps:g} FPS.")

    while True:
        success, frame = videoCapture.read()

        if not success:
            print("Could not read a frame from the webcam.")
            break

        lastFrame = frame

        if durationSeconds is None:
            videoWriter.write(frame)
            framesWritten += 1
        else:
            framesWritten = writeTimedFrames(
                videoWriter=videoWriter,
                frame=frame,
                framesWritten=framesWritten,
                startTime=startTime,
                durationSeconds=durationSeconds,
                outputFps=outputFps,
            )

        cv2.imshow("Interview Demo Recording - press q to stop", frame)

        if durationSeconds is not None and time.monotonic() - startTime >= durationSeconds:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stoppedEarly = True
            break

    if durationSeconds is not None and not stoppedEarly and lastFrame is not None:
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

    savedDurationSeconds = framesWritten / outputFps if outputFps > 0 else 0

    print("Recording saved.")
    print(f"Video path: {outputPath}")
    print(f"Frames written: {framesWritten}")
    print(f"Saved video duration: {savedDurationSeconds:.2f} seconds")


if __name__ == "__main__":
    recordInterviewDemo()
