import queue
import threading
import time
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np


defaultMaxDurationSeconds = 300
defaultJpegQuality = 82
defaultAudioSampleRate = 48000
defaultQueueCapacity = 256
minimumVideoFrames = 2


class RecordingState:
    disconnected = "disconnected"
    requestingPermissions = "requesting_permissions"
    ready = "ready"
    recording = "recording"
    finalizing = "finalizing"
    saved = "saved"
    error = "error"


class WebcamRecorder:
    """Bounded, single-writer WebRTC H.264/AAC recorder.

    Callbacks only normalize timestamps and enqueue compressed video/audio chunks.
    One writer thread owns the PyAV container, both encoders, and muxing. This
    bounds memory use and prevents concurrent writes to the container.
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_duration_seconds: float = defaultMaxDurationSeconds,
        jpeg_quality: int = defaultJpegQuality,
        queue_capacity: int = defaultQueueCapacity,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.max_duration_seconds = float(max_duration_seconds)
        self.jpeg_quality = int(jpeg_quality)
        self.queue_capacity = max(8, int(queue_capacity))
        self._lock = threading.RLock()
        self._state = RecordingState.disconnected
        self._connected = False
        self._videoReady = False
        self._audioReady = False
        self._audioAudible = False
        self._latestVideoShape: tuple[int, int] | None = None
        self._latestAudioRate = defaultAudioSampleRate
        self._latestAudioChannels = 1
        self._queue: queue.Queue | None = None
        self._writerThread: threading.Thread | None = None
        self._finalized = threading.Event()
        self._accepting = False
        self._visualOnly = False
        self._recordingStart: float | None = None
        self._partialPath: Path | None = None
        self._outputPath: Path | None = None
        self._lastSavedPath: Path | None = None
        self._lastStats: dict[str, Any] = self._emptyStats()
        self._writerError: str | None = None
        self._durationCapReached = False
        self._videoClock: dict[str, float | None] = {}
        self._audioClock: dict[str, float | None] = {}
        self._statistics: dict[str, Any] = {}
        self._resetRecordingStatistics()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._state == RecordingState.recording and self._accepting

    @property
    def has_pending_recording(self) -> bool:
        with self._lock:
            return self._state in {RecordingState.recording, RecordingState.finalizing}

    @property
    def reached_duration_cap(self) -> bool:
        with self._lock:
            return self._durationCapReached

    @property
    def video_frame_count(self) -> int:
        with self._lock:
            return int(self._statistics.get("videoFramesReceived", 0))

    @property
    def audio_frame_count(self) -> int:
        with self._lock:
            return int(self._statistics.get("audioFramesReceived", 0))

    @property
    def dropped_video_frames(self) -> int:
        with self._lock:
            return int(self._statistics.get("droppedVideoFrames", 0))

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            if self._recordingStart is None:
                return float(self._lastStats.get("durationSeconds", 0.0)) if self._state == RecordingState.saved else 0.0
            elapsed = time.monotonic() - self._recordingStart
            return min(max(0.0, elapsed), self.max_duration_seconds)

    @property
    def camera_connected(self) -> bool:
        with self._lock:
            return self._connected and self._videoReady

    @property
    def microphone_connected(self) -> bool:
        with self._lock:
            return self._connected and self._audioReady

    @property
    def microphone_audible(self) -> bool:
        with self._lock:
            return self._audioAudible

    @property
    def ready_for_av_recording(self) -> bool:
        with self._lock:
            return self._connected and self._videoReady and self._audioReady

    @property
    def queued_memory_bytes(self) -> int:
        with self._lock:
            return int(self._statistics.get("queuedMemoryBytes", 0))

    @property
    def max_queued_memory_bytes(self) -> int:
        with self._lock:
            return int(self._statistics.get("maxQueuedMemoryBytes", 0))

    @property
    def last_saved_path(self) -> Path | None:
        with self._lock:
            return self._lastSavedPath

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._writerError or self._lastStats.get("error")

    def mark_requesting_permissions(self) -> None:
        with self._lock:
            if self._state not in {
                RecordingState.recording,
                RecordingState.finalizing,
                RecordingState.saved,
                RecordingState.error,
            }:
                self._state = RecordingState.requestingPermissions

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self._connected = bool(connected)
            if not connected:
                if self._state not in {
                    RecordingState.recording,
                    RecordingState.finalizing,
                    RecordingState.saved,
                    RecordingState.error,
                }:
                    self._state = RecordingState.disconnected
                    self._videoReady = False
                    self._audioReady = False
                    self._audioAudible = False
            elif self._state not in {RecordingState.recording, RecordingState.finalizing, RecordingState.saved}:
                self._state = RecordingState.ready if self._videoReady and self._audioReady else RecordingState.requestingPermissions

    def add_video_frame(self, frame: av.VideoFrame) -> None:
        mediaTimestamp = _frameTimestamp(frame)
        self.add_frame(frame.to_ndarray(format="bgr24"), media_timestamp=mediaTimestamp)

    def add_frame(self, bgr_frame: np.ndarray, media_timestamp: float | None = None) -> None:
        height, width = bgr_frame.shape[:2]
        with self._lock:
            self._videoReady = True
            self._latestVideoShape = (height, width)
            self._refreshReadyState()
            if not self._accepting:
                return
            timestamp = self._normalizedTimestamp("video", media_timestamp, 0.0)
        ok, encoded = cv2.imencode(
            ".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        )
        if not ok:
            with self._lock:
                self._statistics["videoEncodeFailures"] += 1
            return
        self._enqueue("video", timestamp, encoded.tobytes(), encoded.nbytes)

    def add_audio_frame(self, audioFrame: av.AudioFrame) -> None:
        samples = self._audioFrameToFloat(audioFrame)
        if samples.size == 0:
            return
        sampleRate = int(audioFrame.sample_rate or defaultAudioSampleRate)
        mediaTimestamp = _frameTimestamp(audioFrame)
        self._handleAudio(samples, sampleRate, mediaTimestamp)

    def add_audio_samples(
        self,
        samples: np.ndarray,
        sample_rate: int,
        elapsed_seconds: float | None = None,
    ) -> None:
        normalized = np.asarray(samples, dtype=np.float32)
        if normalized.ndim == 1:
            normalized = normalized.reshape(1, -1)
        self._handleAudio(normalized, int(sample_rate), elapsed_seconds, explicitElapsed=elapsed_seconds is not None)

    def _handleAudio(
        self,
        samples: np.ndarray,
        sampleRate: int,
        mediaTimestamp: float | None,
        explicitElapsed: bool = False,
    ) -> None:
        samples = np.clip(samples.astype(np.float32), -1.0, 1.0)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        duration = samples.shape[1] / max(sampleRate, 1)
        with self._lock:
            self._audioReady = True
            self._audioAudible = self._audioAudible or peak >= 0.001
            self._latestAudioRate = sampleRate
            self._latestAudioChannels = max(1, min(samples.shape[0], 2))
            self._refreshReadyState()
            if not self._accepting:
                return
            if explicitElapsed and mediaTimestamp is not None:
                timestamp = max(0.0, float(mediaTimestamp))
                last = self._audioClock.get("last")
                if last is not None and timestamp < float(last) - 0.02:
                    self._statistics["audioDiscontinuities"] += 1
                    timestamp = float(last)
                self._audioClock["last"] = timestamp
                if self._statistics["firstAudioTimestamp"] is None:
                    self._statistics["firstAudioTimestamp"] = timestamp
                self._statistics["audioTimingSource"] = "explicit_elapsed"
            else:
                timestamp = self._normalizedTimestamp("audio", mediaTimestamp, duration)
            self._statistics["audioPeak"] = max(float(self._statistics["audioPeak"]), peak)
        self._enqueue("audio", timestamp, (samples.copy(), sampleRate), samples.nbytes)

    def start(self, allow_visual_only: bool = False) -> bool:
        with self._lock:
            if self._state in {RecordingState.recording, RecordingState.finalizing}:
                return False
            if not self._connected or not self._videoReady:
                self._setError("Camera frames have not arrived yet. Connect the camera before recording.")
                return False
            if not allow_visual_only and not self._audioReady:
                self._setError("Microphone frames have not arrived yet. Connect the microphone or explicitly choose visual-only mode.")
                return False
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._outputPath = self.output_dir / f"interviewDemo_{timestamp}.mp4"
            self._partialPath = self._outputPath.with_suffix(".partial.mp4")
            self._partialPath.unlink(missing_ok=True)
            self._queue = queue.Queue(maxsize=self.queue_capacity)
            self._finalized.clear()
            self._writerError = None
            self._durationCapReached = False
            self._visualOnly = bool(allow_visual_only)
            self._recordingStart = time.monotonic()
            self._videoClock = {}
            self._audioClock = {}
            self._resetRecordingStatistics()
            self._accepting = True
            self._state = RecordingState.recording
            videoShape = self._latestVideoShape
            audioRate = self._latestAudioRate
            audioChannels = self._latestAudioChannels
            partialPath = self._partialPath
            self._writerThread = threading.Thread(
                target=self._writerLoop,
                args=(partialPath, videoShape, audioRate, audioChannels, not allow_visual_only),
                name="TalonCVMediaWriter",
                daemon=True,
            )
            self._writerThread.start()
            return True

    def stop(self) -> tuple[Path | None, dict[str, Any]]:
        with self._lock:
            if self._state == RecordingState.finalizing:
                finalEvent = self._finalized
                duplicate = True
            elif self._state in {RecordingState.saved, RecordingState.error}:
                stats = dict(self._lastStats)
                stats["duplicateStop"] = True
                return self._lastSavedPath, stats
            elif self._state != RecordingState.recording:
                return None, {**self._emptyStats(), "error": "No active recording is available to finalize."}
            else:
                self._accepting = False
                self._state = RecordingState.finalizing
                finalEvent = self._finalized
                duplicate = False
                workQueue = self._queue
        if duplicate:
            finalEvent.wait(timeout=60)
            with self._lock:
                stats = dict(self._lastStats)
                stats["duplicateStop"] = True
                return self._lastSavedPath, stats
        self._enqueueStop(workQueue)
        writer = self._writerThread
        if writer is not None:
            writer.join(timeout=60)
        with self._lock:
            if writer is not None and writer.is_alive():
                self._writerError = "Media finalization exceeded 60 seconds."
            partialPath = self._partialPath
            outputPath = self._outputPath
            writerError = self._writerError
        if writerError or partialPath is None or outputPath is None or not partialPath.exists():
            return self._finishFailure(writerError or "The partial recording was not created.")
        with self._lock:
            receivedVideoFrames = int(self._statistics.get("videoFramesReceived", 0))
            receivedAudioFrames = int(self._statistics.get("audioFramesReceived", 0))
        if receivedVideoFrames < minimumVideoFrames:
            return self._finishFailure(
                f"At least {minimumVideoFrames} video frames are required to preserve a recording."
            )
        if not self._visualOnly and receivedAudioFrames < 1:
            return self._finishFailure("No microphone audio frames were captured, so the recording was not preserved.")
        validation = self._validatePartial(partialPath, expectAudio=not self._visualOnly)
        timing = self._timingStatistics()
        validationError = validation.get("error") or self._alignmentError(timing, expectAudio=not self._visualOnly)
        if validationError:
            return self._finishFailure(validationError, validation=validation, timing=timing)
        try:
            partialPath.replace(outputPath)
        except OSError as error:
            return self._finishFailure(f"Could not atomically save the recording: {error}", validation, timing)
        stats = self._buildStats(validation, timing, error=None)
        with self._lock:
            self._state = RecordingState.saved
            self._lastSavedPath = outputPath
            self._lastStats = stats
            self._recordingStart = None
            self._finalized.set()
        return outputPath, dict(stats)

    def discard(self) -> None:
        with self._lock:
            active = self._state in {RecordingState.recording, RecordingState.finalizing}
            self._accepting = False
            workQueue = self._queue
            writer = self._writerThread
        if active:
            self._enqueueStop(workQueue)
            if writer is not None:
                writer.join(timeout=10)
        with self._lock:
            if self._partialPath is not None:
                self._partialPath.unlink(missing_ok=True)
            self._queue = None
            self._writerThread = None
            self._recordingStart = None
            self._resetRecordingStatistics()
            self._state = RecordingState.disconnected if not self._connected else RecordingState.requestingPermissions
            self._finalized.set()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "cameraConnected": self.camera_connected,
                "microphoneConnected": self.microphone_connected,
                "microphoneAudible": self._audioAudible,
                "elapsedSeconds": round(self.elapsed_seconds, 3),
                "videoFrameCount": self.video_frame_count,
                "audioFrameCount": self.audio_frame_count,
                "droppedVideoFrames": self.dropped_video_frames,
                "droppedAudioFrames": int(self._statistics.get("droppedAudioFrames", 0)),
                "queueCapacity": self.queue_capacity,
                "queuedMemoryBytes": self.queued_memory_bytes,
                "maxQueuedMemoryBytes": self.max_queued_memory_bytes,
                "durationCapReached": self._durationCapReached,
                "savedFilename": self._lastSavedPath.name if self._lastSavedPath else None,
                "error": self.last_error,
            }

    def _enqueue(self, kind: str, timestamp: float, payload: Any, payloadBytes: int) -> None:
        with self._lock:
            if not self._accepting or self._queue is None:
                return
            elapsed = time.monotonic() - (self._recordingStart or time.monotonic())
            if elapsed > self.max_duration_seconds:
                self._accepting = False
                self._durationCapReached = True
                return
            item = (kind, max(0.0, float(timestamp)), payload, int(payloadBytes))
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                key = "droppedVideoFrames" if kind == "video" else "droppedAudioFrames"
                self._statistics[key] += 1
                return
            self._statistics["queuedMemoryBytes"] += int(payloadBytes)
            self._statistics["maxQueuedMemoryBytes"] = max(
                self._statistics["maxQueuedMemoryBytes"], self._statistics["queuedMemoryBytes"]
            )
            if kind == "video":
                self._statistics["videoFramesReceived"] += 1
                self._statistics["lastVideoTimestamp"] = timestamp
            else:
                self._statistics["audioFramesReceived"] += 1
                samples, rate = payload
                self._statistics["lastAudioEndTimestamp"] = timestamp + samples.shape[1] / rate

    def _enqueueStop(self, workQueue) -> None:
        if workQueue is None:
            return
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                workQueue.put_nowait(("stop", 0.0, None, 0))
                return
            except queue.Full:
                try:
                    kind, _timestamp, _payload, payloadBytes = workQueue.get_nowait()
                except queue.Empty:
                    continue
                with self._lock:
                    self._statistics["queuedMemoryBytes"] = max(
                        0, self._statistics["queuedMemoryBytes"] - int(payloadBytes)
                    )
                    if kind == "video":
                        self._statistics["droppedVideoFrames"] += 1
                    elif kind == "audio":
                        self._statistics["droppedAudioFrames"] += 1
        with self._lock:
            self._writerError = self._writerError or "The media writer queue could not be stopped within five seconds."
            self._finalized.set()

    def _writerLoop(self, partialPath, videoShape, sourceAudioRate, sourceAudioChannels, expectAudio):
        try:
            if videoShape is None:
                raise RuntimeError("Video dimensions were unavailable when the writer started.")
            height, width = videoShape
            audioRate = defaultAudioSampleRate
            audioChannels = max(1, min(int(sourceAudioChannels), 2))
            audioLayout = "mono" if audioChannels == 1 else "stereo"
            millisecondTimeBase = Fraction(1, 1000)
            sampleTimeBase = Fraction(1, audioRate)
            audioPending = np.zeros((audioChannels, 0), dtype=np.float32)
            audioPendingStart = 0
            audioExpectedPosition = 0

            with av.open(str(partialPath), mode="w", options={"movflags": "+faststart"}) as container:
                videoStream = container.add_stream("libx264", rate=30)
                videoStream.width = width
                videoStream.height = height
                videoStream.pix_fmt = "yuv420p"
                videoStream.options = {"preset": "veryfast", "crf": "23"}
                videoStream.time_base = millisecondTimeBase
                videoStream.codec_context.time_base = millisecondTimeBase
                audioStream = None
                if expectAudio:
                    audioStream = container.add_stream("aac", rate=audioRate)
                    audioStream.layout = audioLayout
                    audioStream.bit_rate = 128000

                lastVideoPts = -1

                def encodeAudioBlock(block, pts):
                    frame = av.AudioFrame.from_ndarray(block, format="fltp", layout=audioLayout)
                    frame.sample_rate = audioRate
                    frame.pts = int(pts)
                    frame.time_base = sampleTimeBase
                    for packet in audioStream.encode(frame):
                        container.mux(packet)

                while True:
                    kind, timestamp, payload, payloadBytes = self._queue.get()
                    with self._lock:
                        self._statistics["queuedMemoryBytes"] = max(
                            0, self._statistics["queuedMemoryBytes"] - payloadBytes
                        )
                    if kind == "stop":
                        break
                    if kind == "video":
                        encoded = np.frombuffer(payload, dtype=np.uint8)
                        bgrFrame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                        if bgrFrame is None:
                            with self._lock:
                                self._statistics["videoDecodeFailures"] += 1
                            continue
                        if bgrFrame.shape[:2] != (height, width):
                            bgrFrame = cv2.resize(bgrFrame, (width, height))
                        frame = av.VideoFrame.from_ndarray(bgrFrame, format="bgr24")
                        frame.pts = max(lastVideoPts + 1, int(round(timestamp * 1000)))
                        lastVideoPts = frame.pts
                        frame.time_base = millisecondTimeBase
                        for packet in videoStream.encode(frame):
                            container.mux(packet)
                        with self._lock:
                            self._statistics["videoFramesWritten"] += 1
                    elif kind == "audio" and audioStream is not None:
                        samples, sampleRate = payload
                        samples = self._convertAudio(samples, sampleRate, audioRate, audioChannels)
                        desiredStart = max(0, int(round(timestamp * audioRate)))
                        expected = audioPendingStart + audioPending.shape[1]
                        if audioPending.shape[1] == 0:
                            audioPendingStart = desiredStart
                            expected = desiredStart
                        if desiredStart < expected:
                            overlap = expected - desiredStart
                            if overlap >= samples.shape[1]:
                                with self._lock:
                                    self._statistics["audioDiscontinuities"] += 1
                                continue
                            samples = samples[:, overlap:]
                            desiredStart = expected
                            with self._lock:
                                self._statistics["audioDiscontinuities"] += 1
                        gap = desiredStart - expected
                        while gap >= 1024:
                            if audioPending.shape[1]:
                                padding = np.zeros((audioChannels, 1024 - audioPending.shape[1]), dtype=np.float32)
                                encodeAudioBlock(np.concatenate([audioPending, padding], axis=1), audioPendingStart)
                                audioExpectedPosition = audioPendingStart + 1024
                                audioPending = np.zeros((audioChannels, 0), dtype=np.float32)
                            else:
                                encodeAudioBlock(np.zeros((audioChannels, 1024), dtype=np.float32), expected)
                                audioExpectedPosition = expected + 1024
                            expected = audioExpectedPosition
                            audioPendingStart = expected
                            gap = desiredStart - expected
                            with self._lock:
                                self._statistics["audioDiscontinuities"] += 1
                        if gap > 0:
                            audioPending = np.concatenate(
                                [audioPending, np.zeros((audioChannels, gap), dtype=np.float32)], axis=1
                            )
                        audioPending = np.concatenate([audioPending, samples], axis=1)
                        while audioPending.shape[1] >= 1024:
                            block = audioPending[:, :1024]
                            encodeAudioBlock(block, audioPendingStart)
                            audioPending = audioPending[:, 1024:]
                            audioPendingStart += 1024
                            audioExpectedPosition = audioPendingStart
                        with self._lock:
                            self._statistics["audioFramesWritten"] += 1

                if audioStream is not None and audioPending.shape[1]:
                    padding = np.zeros((audioChannels, 1024 - audioPending.shape[1]), dtype=np.float32)
                    encodeAudioBlock(np.concatenate([audioPending, padding], axis=1), audioPendingStart)
                for packet in videoStream.encode():
                    container.mux(packet)
                if audioStream is not None:
                    for packet in audioStream.encode():
                        container.mux(packet)
        except Exception as error:
            with self._lock:
                self._writerError = str(error)
        finally:
            self._finalized.set()

    @staticmethod
    def _convertAudio(samples, sourceRate, targetRate, channels):
        samples = samples[:channels]
        if samples.shape[0] == 1 and channels == 2:
            samples = np.repeat(samples, 2, axis=0)
        if samples.shape[0] == 2 and channels == 1:
            samples = samples.mean(axis=0, keepdims=True)
        if sourceRate != targetRate and samples.shape[1]:
            sourcePositions = np.arange(samples.shape[1], dtype=np.float64)
            targetLength = max(1, int(round(samples.shape[1] * targetRate / sourceRate)))
            targetPositions = np.linspace(0, max(samples.shape[1] - 1, 0), targetLength)
            samples = np.vstack(
                [np.interp(targetPositions, sourcePositions, channel) for channel in samples]
            ).astype(np.float32)
        return np.clip(samples, -1.0, 1.0).astype(np.float32)

    def _normalizedTimestamp(self, kind, mediaTimestamp, duration):
        clock = self._videoClock if kind == "video" else self._audioClock
        arrival = max(0.0, time.monotonic() - (self._recordingStart or time.monotonic()))
        fallback = max(0.0, arrival - duration)
        if mediaTimestamp is not None:
            if clock.get("firstMedia") is None:
                clock["firstMedia"] = float(mediaTimestamp)
                clock["firstArrival"] = fallback
            timestamp = float(clock["firstArrival"]) + float(mediaTimestamp) - float(clock["firstMedia"])
            source = "media_pts"
        else:
            timestamp = fallback
            source = "monotonic_fallback"
        last = clock.get("last")
        if last is not None and timestamp < float(last) - 0.02:
            self._statistics[f"{kind}Discontinuities"] += 1
            timestamp = float(last) + (0.001 if kind == "video" else 0.0)
        clock["last"] = timestamp
        if self._statistics.get(f"first{kind.title()}Timestamp") is None:
            self._statistics[f"first{kind.title()}Timestamp"] = timestamp
        self._statistics[f"{kind}TimingSource"] = source
        return timestamp

    def _validatePartial(self, path, expectAudio):
        result = {
            "valid": False,
            "hasVideo": False,
            "hasAudio": False,
            "videoCodec": None,
            "audioCodec": None,
            "audioPeak": 0.0,
            "durationSeconds": 0.0,
            "error": None,
        }
        try:
            with av.open(str(path)) as container:
                videoStream = next((stream for stream in container.streams if stream.type == "video"), None)
                audioStream = next((stream for stream in container.streams if stream.type == "audio"), None)
                result["hasVideo"] = videoStream is not None
                result["hasAudio"] = audioStream is not None
                result["videoCodec"] = videoStream.codec_context.name if videoStream else None
                result["audioCodec"] = audioStream.codec_context.name if audioStream else None
                result["durationSeconds"] = float(container.duration / av.time_base) if container.duration else 0.0
                if audioStream is not None:
                    peak = 0.0
                    for frame in container.decode(audioStream):
                        samples = self._audioFrameToFloat(frame)
                        if samples.size:
                            peak = max(peak, float(np.max(np.abs(samples))))
                    result["audioPeak"] = peak
            if not result["hasVideo"]:
                result["error"] = "The finalized MP4 did not contain a video stream."
            elif result["videoCodec"] != "h264":
                result["error"] = f"Expected H.264 video, received {result['videoCodec']}."
            elif expectAudio and not result["hasAudio"]:
                result["error"] = "The finalized MP4 did not contain the expected microphone audio stream."
            elif expectAudio and result["audioCodec"] != "aac":
                result["error"] = f"Expected AAC audio, received {result['audioCodec']}."
            elif expectAudio and result["audioPeak"] < 0.001:
                result["error"] = "The finalized audio track was effectively silent. Check microphone input and record again."
            else:
                result["valid"] = True
        except (av.error.FFmpegError, OSError, ValueError) as error:
            result["error"] = f"The finalized MP4 could not be decoded: {error}"
        return result

    def _timingStatistics(self):
        with self._lock:
            firstVideo = self._statistics.get("firstVideoTimestamp")
            firstAudio = self._statistics.get("firstAudioTimestamp")
            lastVideo = self._statistics.get("lastVideoTimestamp")
            lastAudio = self._statistics.get("lastAudioEndTimestamp")
        startOffset = firstAudio - firstVideo if firstAudio is not None and firstVideo is not None else None
        videoDuration = lastVideo - firstVideo if lastVideo is not None and firstVideo is not None else None
        audioDuration = lastAudio - firstAudio if lastAudio is not None and firstAudio is not None else None
        durationDifference = audioDuration - videoDuration if audioDuration is not None and videoDuration is not None else None
        return {
            "firstVideoTimestamp": _round(firstVideo),
            "firstAudioTimestamp": _round(firstAudio),
            "audioVideoStartOffsetSeconds": _round(startOffset),
            "videoTimelineDurationSeconds": _round(videoDuration),
            "audioTimelineDurationSeconds": _round(audioDuration),
            "audioVideoDurationDifferenceSeconds": _round(durationDifference),
            "estimatedDriftSeconds": _round(durationDifference),
            "videoTimingSource": self._statistics.get("videoTimingSource"),
            "audioTimingSource": self._statistics.get("audioTimingSource"),
            "videoDiscontinuities": self._statistics.get("videoDiscontinuities", 0),
            "audioDiscontinuities": self._statistics.get("audioDiscontinuities", 0),
        }

    @staticmethod
    def _alignmentError(timing, expectAudio):
        if not expectAudio:
            return None
        offset = timing.get("audioVideoStartOffsetSeconds")
        difference = timing.get("audioVideoDurationDifferenceSeconds")
        if offset is None or difference is None:
            return "Audio/video synchronization could not be validated because timing evidence was incomplete."
        if abs(float(offset)) > 1.5:
            return f"Audio and video started {abs(float(offset)):.2f}s apart, so the recording was not preserved as synchronized."
        if abs(float(difference)) > 2.0:
            return f"Audio/video duration drift was {abs(float(difference)):.2f}s, so the recording was not preserved."
        return None

    def _buildStats(self, validation, timing, error):
        with self._lock:
            receivedVideo = int(self._statistics["videoFramesReceived"])
            firstVideo = self._statistics.get("firstVideoTimestamp")
            lastVideo = self._statistics.get("lastVideoTimestamp")
            fps = (receivedVideo - 1) / (lastVideo - firstVideo) if receivedVideo >= 2 and lastVideo > firstVideo else 0.0
            stats = {
                "state": RecordingState.saved if not error else RecordingState.error,
                "frameCount": receivedVideo,
                "videoFramesWritten": int(self._statistics["videoFramesWritten"]),
                "fps": round(fps, 3),
                "audioFrameCount": int(self._statistics["audioFramesReceived"]),
                "audioFramesWritten": int(self._statistics["audioFramesWritten"]),
                "audioSampleRate": defaultAudioSampleRate if not self._visualOnly else None,
                "audioChannels": self._latestAudioChannels if not self._visualOnly else 0,
                "audioCaptured": bool(self._statistics["audioFramesReceived"]),
                "audioAudible": validation.get("audioPeak", 0.0) >= 0.001,
                "audioPeak": round(float(validation.get("audioPeak", 0.0)), 6),
                "durationSeconds": round(float(validation.get("durationSeconds", 0.0)), 3),
                "truncated": self._durationCapReached,
                "droppedVideoFrames": int(self._statistics["droppedVideoFrames"]),
                "droppedAudioFrames": int(self._statistics["droppedAudioFrames"]),
                "maxQueuedMemoryBytes": int(self._statistics["maxQueuedMemoryBytes"]),
                "queueCapacity": self.queue_capacity,
                "videoCodec": validation.get("videoCodec"),
                "audioCodec": validation.get("audioCodec"),
                "containerValid": bool(validation.get("valid")),
                "error": error,
                **timing,
            }
        return stats

    def _finishFailure(self, error, validation=None, timing=None):
        validation = validation or {}
        timing = timing or self._timingStatistics()
        with self._lock:
            if self._partialPath is not None:
                self._partialPath.unlink(missing_ok=True)
            stats = self._buildStats(validation, timing, error)
            self._state = RecordingState.error
            self._writerError = error
            self._lastSavedPath = None
            self._lastStats = stats
            self._recordingStart = None
            self._finalized.set()
        return None, dict(stats)

    def _setError(self, message):
        self._state = RecordingState.error
        self._writerError = message
        self._lastStats = {**self._emptyStats(), "state": RecordingState.error, "error": message}

    def _refreshReadyState(self):
        if self._connected and self._state not in {
            RecordingState.recording,
            RecordingState.finalizing,
            RecordingState.saved,
        }:
            self._state = RecordingState.ready if self._videoReady and self._audioReady else RecordingState.requestingPermissions

    def _resetRecordingStatistics(self):
        self._statistics = {
            "videoFramesReceived": 0,
            "videoFramesWritten": 0,
            "audioFramesReceived": 0,
            "audioFramesWritten": 0,
            "droppedVideoFrames": 0,
            "droppedAudioFrames": 0,
            "videoEncodeFailures": 0,
            "videoDecodeFailures": 0,
            "queuedMemoryBytes": 0,
            "maxQueuedMemoryBytes": 0,
            "audioPeak": 0.0,
            "firstVideoTimestamp": None,
            "firstAudioTimestamp": None,
            "lastVideoTimestamp": None,
            "lastAudioEndTimestamp": None,
            "videoTimingSource": None,
            "audioTimingSource": None,
            "videoDiscontinuities": 0,
            "audioDiscontinuities": 0,
        }

    @staticmethod
    def _emptyStats():
        return {
            "state": RecordingState.disconnected,
            "frameCount": 0,
            "videoFramesWritten": 0,
            "fps": 0.0,
            "audioFrameCount": 0,
            "audioFramesWritten": 0,
            "audioSampleRate": None,
            "audioChannels": 0,
            "audioCaptured": False,
            "audioAudible": False,
            "durationSeconds": 0.0,
            "truncated": False,
            "droppedVideoFrames": 0,
            "droppedAudioFrames": 0,
            "maxQueuedMemoryBytes": 0,
            "queueCapacity": 0,
            "containerValid": False,
            "error": None,
        }

    @staticmethod
    def _audioFrameToFloat(audioFrame: av.AudioFrame) -> np.ndarray:
        array = audioFrame.to_ndarray()
        channelCount = len(audioFrame.layout.channels)
        if audioFrame.format.is_planar:
            samples = array[:channelCount]
        else:
            flat = array.reshape(-1)
            usable = flat.size - (flat.size % channelCount)
            samples = flat[:usable].reshape(-1, channelCount).T
        if np.issubdtype(samples.dtype, np.unsignedinteger):
            maximum = float(np.iinfo(samples.dtype).max)
            return ((samples.astype(np.float32) / maximum) * 2.0 - 1.0).astype(np.float32)
        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            maximum = float(max(abs(info.min), info.max))
            return (samples.astype(np.float32) / maximum).astype(np.float32)
        return np.clip(samples.astype(np.float32), -1.0, 1.0)


def _frameTimestamp(frame) -> float | None:
    if frame.pts is None or frame.time_base is None:
        return None
    try:
        return float(frame.pts * frame.time_base)
    except (TypeError, ValueError, OverflowError):
        return None


def _round(value):
    return round(float(value), 5) if value is not None else None
