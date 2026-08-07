import argparse
import json
import socket
import subprocess
import sys
import tempfile
import wave
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


projectRoot = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectRoot))


@contextmanager
def blockOutboundSockets():
    originalConnect = socket.socket.connect
    originalCreateConnection = socket.create_connection
    attempts: list[str] = []

    def blockedConnect(instance, address):
        attempts.append(repr(address))
        raise OSError(f"Offline smoke test blocked outbound socket connection to {address!r}")

    def blockedCreateConnection(address, *args, **kwargs):
        attempts.append(repr(address))
        raise OSError(f"Offline smoke test blocked outbound socket connection to {address!r}")

    socket.socket.connect = blockedConnect
    socket.create_connection = blockedCreateConnection
    try:
        yield attempts
    finally:
        socket.socket.connect = originalConnect
        socket.create_connection = originalCreateConnection


def createSpeechFixture(wavPath: Path) -> None:
    escapedPath = str(wavPath).replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$voice=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$voice.SetOutputToWaveFile('{escapedPath}'); "
        "$voice.Speak('I led a support routing project. I analyzed the queue and built an automated rule. "
        "As a result, response delays fell by thirty percent. Overall, the team responded faster.'); "
        "$voice.Dispose()"
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or not wavPath.exists():
        raise RuntimeError(
            "Windows local speech synthesis could not create the offline smoke fixture. "
            f"{completed.stderr.strip()}"
        )


def createAvFixture(wavPath: Path, mediaPath: Path) -> None:
    with wave.open(str(wavPath), "rb") as source:
        channels = source.getnchannels()
        sampleWidth = source.getsampwidth()
        sampleRate = source.getframerate()
        raw = source.readframes(source.getnframes())
    if sampleWidth != 2:
        raise RuntimeError(f"Expected a PCM16 speech fixture, received {sampleWidth * 8}-bit PCM.")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    duration = samples.size / sampleRate
    width, height, fps = 320, 240, 10
    with av.open(str(mediaPath), mode="w", options={"movflags": "+faststart"}) as container:
        videoStream = container.add_stream("libx264", rate=fps)
        videoStream.width = width
        videoStream.height = height
        videoStream.pix_fmt = "yuv420p"
        audioStream = container.add_stream("aac", rate=sampleRate)
        audioStream.layout = "mono"
        for index in range(max(2, int(np.ceil(duration * fps)))):
            pixels = np.full((height, width, 3), 225, dtype=np.uint8)
            center = (width // 2, height // 2 - 10)
            cv2.circle(pixels, center, 62, (185, 205, 230), -1)
            cv2.circle(pixels, (center[0] - 22, center[1] - 12), 5, (30, 30, 30), -1)
            cv2.circle(pixels, (center[0] + 22, center[1] - 12), 5, (30, 30, 30), -1)
            cv2.ellipse(pixels, (center[0], center[1] + 18), (24, 10), 0, 0, 180, (40, 40, 40), 3)
            frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in videoStream.encode(frame):
                container.mux(packet)
        for start in range(0, samples.size, 1024):
            block = samples[start : start + 1024].reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(block.astype(np.float32), format="fltp", layout="mono")
            frame.sample_rate = sampleRate
            frame.pts = start
            frame.time_base = Fraction(1, sampleRate)
            for packet in audioStream.encode(frame):
                container.mux(packet)
        for packet in videoStream.encode():
            container.mux(packet)
        for packet in audioStream.encode():
            container.mux(packet)


def runOfflineSmoke(maxCoachTokens: int = 96) -> dict[str, Any]:
    from src.audioPipeline.semanticAnalyzer import analyzeSemanticResponse
    from src.audioPipeline.transcription import transcribeAudio
    from src.cvPipeline.yoloFaceDetector import loadYoloFaceDetector
    from src.localModels.localCoach import generateLocalCoaching
    from src.multimodalPipeline.artifacts import getArtifactPaths
    from src.multimodalPipeline.pipeline import runMultimodalAnalysis

    with tempfile.TemporaryDirectory(prefix="taloncv_offline_smoke_") as folder:
        temporary = Path(folder)
        speechWav = temporary / "speech.wav"
        mediaPath = temporary / "offline_smoke.mp4"
        transcriptJson = temporary / "transcript.json"
        transcriptText = temporary / "transcript.txt"
        createSpeechFixture(speechWav)
        createAvFixture(speechWav, mediaPath)
        context = {
            "interviewQuestion": "Tell me about a time you improved a process.",
            "targetRole": "Support operations lead",
            "jobDescription": "Improve service processes and response times.",
            "desiredCompetencies": "ownership, analysis, measurable results",
        }
        with blockOutboundSockets() as attempts:
            transcript = transcribeAudio(
                speechWav,
                "offline_smoke_direct",
                transcriptJson,
                transcriptText,
                {"fixture": "local-synthetic-speech-v1"},
                force=True,
                progressCallback=lambda message: print(f"direct-transcription: {message}", flush=True),
            )
            if not transcript.get("text"):
                raise AssertionError("Real local faster-whisper produced no text for the speech fixture.")
            semantic = analyzeSemanticResponse(transcript, context)
            if not semantic.get("available"):
                raise AssertionError(f"Real local MiniLM analysis was unavailable: {semantic.get('warnings')}")
            detector = loadYoloFaceDetector()
            sampleFrame = np.full((240, 320, 3), 180, dtype=np.uint8)
            detector.detect(sampleFrame)

            def boundedCoach(analysis, progressCallback=None):
                return generateLocalCoaching(
                    analysis,
                    progressCallback=progressCallback,
                    maxNewTokens=maxCoachTokens,
                )

            result = runMultimodalAnalysis(
                mediaPath,
                sessionContext=context,
                force=True,
                yoloFaceDetector=detector,
                localCoach=boundedCoach,
                progressCallback=lambda stage, fraction, message: print(
                    f"{fraction:.0%} {stage}: {message}", flush=True
                ),
            )
            paths = getArtifactPaths(mediaPath)
            if not result.get("complete") or not paths.deterministicReport.exists() or not paths.report.exists():
                raise AssertionError("The full multimodal pipeline did not create complete reports.")
            if not result.get("localCoaching", {}).get("available"):
                raise AssertionError(
                    f"Real local Qwen generation was unavailable: {result.get('localCoaching', {}).get('warnings')}"
                )
            summary = {
                "ready": True,
                "networkAttempts": attempts,
                "transcriptText": transcript["text"],
                "transcriptConfidence": transcript.get("averageConfidence"),
                "semanticQuestionScore": semantic.get("questionRelevance", {}).get("score"),
                "yoloModel": detector.model_source,
                "localCoachCharacters": len(result["localCoaching"].get("text", "")),
                "visualEventCount": len(result.get("visualEvents", [])),
                "multimodalMomentCount": len(result.get("moments", [])),
                "reportCharacters": len(paths.report.read_text(encoding="utf-8")),
            }
        for name, path in getArtifactPaths(mediaPath).asDict().items():
            if name != "media":
                path.unlink(missing_ok=True)
        if summary["networkAttempts"]:
            raise AssertionError(f"A component attempted network access: {summary['networkAttempts']}")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TalonCV's real-model pipeline with outbound sockets blocked.")
    parser.add_argument("--max-coach-tokens", type=int, default=96)
    args = parser.parse_args()
    print(json.dumps(runOfflineSmoke(args.max_coach_tokens), indent=2))


if __name__ == "__main__":
    main()
