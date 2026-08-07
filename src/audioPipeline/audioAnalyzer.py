import math
import wave
from pathlib import Path
from typing import Any

import numpy as np


audioAnalysisVersion = "audio-v1"


def loadWav(wavPath: str | Path) -> tuple[np.ndarray, int, int]:
    """Load PCM WAV into normalized mono float32 samples."""
    with wave.open(str(wavPath), "rb") as wavFile:
        channels = wavFile.getnchannels()
        sampleWidth = wavFile.getsampwidth()
        sampleRate = wavFile.getframerate()
        raw = wavFile.readframes(wavFile.getnframes())
    if sampleWidth != 2:
        raise ValueError(f"Expected PCM16 WAV, received {sampleWidth * 8}-bit samples.")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), sampleRate, channels


def analyzeAudio(
    wavPath: str | Path,
    transcript: dict[str, Any] | None = None,
    sourceFingerprint: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples, sampleRate, sourceChannels = loadWav(wavPath)
    duration = samples.size / sampleRate if sampleRate else 0.0
    warnings: list[str] = []
    if samples.size == 0 or duration <= 0:
        return _unavailableFeatures(sampleRate, sourceChannels, warnings + ["The decoded WAV is empty."]), []

    peakAmplitude = float(np.max(np.abs(samples)))
    overallRms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
    overallDb = _db(overallRms)
    clippingRatio = float(np.mean(np.abs(samples) >= 0.99))
    if peakAmplitude < 0.001 or overallDb < -60:
        warnings.append("The decoded audio is effectively silent, so vocal measurements are unavailable.")

    frameDuration = 0.05
    hopDuration = 0.025
    frameSize = max(1, int(round(frameDuration * sampleRate)))
    hopSize = max(1, int(round(hopDuration * sampleRate)))
    frameStarts = np.arange(0, max(samples.size - frameSize + 1, 1), hopSize, dtype=int)
    if frameStarts.size == 0:
        frameStarts = np.array([0], dtype=int)
    rmsValues = np.array(
        [
            math.sqrt(float(np.mean(np.square(samples[start : start + frameSize]), dtype=np.float64)))
            if samples[start : start + frameSize].size
            else 0.0
            for start in frameStarts
        ],
        dtype=np.float64,
    )
    dbValues = np.array([_db(value) for value in rmsValues], dtype=np.float64)
    peakValues = np.array(
        [float(np.max(np.abs(samples[start : start + frameSize]))) for start in frameStarts], dtype=np.float64
    )
    noiseFloorDb = float(np.percentile(dbValues, 20))
    speechThresholdDb = max(-48.0, min(-32.0, noiseFloorDb + 8.0))
    speechMask = dbValues > speechThresholdDb
    speechRatio = float(np.mean(speechMask))
    silenceRatio = 1.0 - speechRatio
    voicedDb = dbValues[speechMask]
    volumeStdDb = float(np.std(voicedDb)) if voicedDb.size >= 2 else None
    energyVariationDb = float(np.percentile(voicedDb, 90) - np.percentile(voicedDb, 10)) if voicedDb.size else None

    noiseSamplesMask = np.repeat(~speechMask, hopSize)
    noiseSamplesMask = np.pad(noiseSamplesMask, (0, max(0, samples.size - noiseSamplesMask.size)))[: samples.size]
    noiseRms = float(np.sqrt(np.mean(np.square(samples[noiseSamplesMask]), dtype=np.float64))) if noiseSamplesMask.any() else None
    speechSamplesMask = ~noiseSamplesMask
    speechRms = (
        float(np.sqrt(np.mean(np.square(samples[speechSamplesMask]), dtype=np.float64)))
        if speechSamplesMask.any()
        else None
    )
    snrDb = _db(speechRms) - _db(noiseRms) if speechRms and noiseRms and noiseRms > 0 else None

    pitchValues = _pitchTrack(samples, sampleRate, speechThresholdDb)
    pitchMedian = float(np.median(pitchValues)) if len(pitchValues) >= 5 else None
    pitchVariationSemitones = None
    if len(pitchValues) >= 5 and pitchMedian and pitchMedian > 0:
        semitones = 12 * np.log2(np.asarray(pitchValues) / pitchMedian)
        pitchVariationSemitones = float(np.std(semitones))

    events: list[dict[str, Any]] = []
    frameTimes = frameStarts / sampleRate
    events.extend(
        _maskEvents(
            ~speechMask,
            frameTimes,
            frameDuration,
            0.8,
            "longPause",
            "The audio contained a sustained low-energy pause.",
            "Review whether this pause helped organize the answer or interrupted its flow.",
            {"speechThresholdDb": round(speechThresholdDb, 2)},
        )
    )
    dropoutMask = dbValues < -75
    events.extend(
        _maskEvents(
            dropoutMask,
            frameTimes,
            frameDuration,
            0.25,
            "audioDropout",
            "The waveform dropped to near-digital silence.",
            "Check the microphone connection if this was not an intentional pause.",
            {"thresholdDb": -75},
        )
    )
    if voicedDb.size:
        medianVoicedDb = float(np.median(voicedDb))
        events.extend(
            _maskEvents(
                speechMask & (dbValues < medianVoicedDb - 8),
                frameTimes,
                frameDuration,
                0.3,
                "lowVolume",
                "A speech-like section was substantially quieter than the recording's usual level.",
                "Practice keeping important words audible and at a consistent microphone distance.",
                {"referenceDb": round(medianVoicedDb, 2)},
            )
        )
        events.extend(
            _maskEvents(
                speechMask & (dbValues > medianVoicedDb + 7),
                frameTimes,
                frameDuration,
                0.2,
                "highVolume",
                "A section was substantially louder than the recording's usual speech level.",
                "Review whether the louder delivery sounded intentional and remained comfortable to hear.",
                {"referenceDb": round(medianVoicedDb, 2)},
            )
        )
        emphasisThreshold = float(np.percentile(voicedDb, 85))
        events.extend(
            _maskEvents(
                speechMask & (dbValues >= emphasisThreshold),
                frameTimes,
                frameDuration,
                0.15,
                "strongVocalEmphasis",
                "The audio energy rose above the recording's normal speech range.",
                "Check whether the emphasis landed on an important phrase you want to preserve.",
                {"energyThresholdDb": round(emphasisThreshold, 2)},
                reliability="medium",
            )
        )

    clippingFrameMask = peakValues >= 0.999
    events.extend(
        _maskEvents(
            clippingFrameMask,
            frameTimes,
            frameDuration,
            0.05,
            "audioClipping",
            "Samples reached the digital amplitude ceiling in this section.",
            "Move slightly farther from the microphone or reduce input gain.",
            {"clippingThreshold": 0.999},
            reliability="high",
        )
    )

    wordCount, speechRateWpm, speechRateVariation = _addTranscriptEvents(events, transcript, duration)
    fragmentedSegments = _addFragmentationEvents(events, transcript)
    abruptChangeCount = int(np.sum(np.abs(np.diff(dbValues)) > 12)) if dbValues.size > 1 else 0
    if duration >= 8 and energyVariationDb is not None and energyVariationDb < 5:
        pitchIsLimited = pitchVariationSemitones is None or pitchVariationSemitones < 1.5
        if pitchIsLimited:
            events.append(
                _event(
                    "limitedVocalVariation",
                    0.0,
                    duration,
                    {"energyVariationDb": round(energyVariationDb, 2), "pitchVariationSemitones": _round(pitchVariationSemitones)},
                    "The measurable energy and pitch range stayed relatively narrow across the response.",
                    "Try varying emphasis around the answer's key action and result while keeping delivery natural.",
                    "medium" if pitchVariationSemitones is not None else "low",
                )
            )

    qualityReasons = []
    if clippingRatio > 0.01:
        qualityReasons.append("frequent clipping")
    if snrDb is not None and snrDb < 10:
        qualityReasons.append("limited speech-to-noise separation")
    if float(np.mean(dropoutMask)) > 0.05:
        qualityReasons.append("repeated near-silent dropouts")
    if overallDb < -40:
        qualityReasons.append("low overall level")
    if qualityReasons:
        events.append(
            _event(
                "lowAudioQuality",
                0.0,
                duration,
                {"reasons": qualityReasons, "snrProxyDb": _round(snrDb), "clippingPercent": round(clippingRatio * 100, 4)},
                f"The recording showed {', '.join(qualityReasons)}.",
                "Improve the recording setup before interpreting subtle vocal-delivery cues.",
                "high",
            )
        )

    features = {
        "analysisVersion": audioAnalysisVersion,
        "available": peakAmplitude >= 0.001,
        "sourceFingerprint": sourceFingerprint,
        "wavPath": str(Path(wavPath).resolve()),
        "durationSeconds": round(duration, 5),
        "sampleRate": sampleRate,
        "sourceChannels": sourceChannels,
        "overallRmsDb": round(overallDb, 3),
        "peakAmplitude": round(peakAmplitude, 6),
        "volumeConsistencyStdDb": _round(volumeStdDb),
        "energyVariationDb": _round(energyVariationDb),
        "silenceRatio": round(silenceRatio, 5),
        "speechRatio": round(speechRatio, 5),
        "clippingPercentage": round(clippingRatio * 100, 6),
        "dropoutRatio": round(float(np.mean(dropoutMask)), 5),
        "noiseFloorDb": round(noiseFloorDb, 3),
        "speechThresholdDb": round(speechThresholdDb, 3),
        "snrProxyDb": _round(snrDb),
        "pitchMedianHz": _round(pitchMedian),
        "pitchVariationSemitones": _round(pitchVariationSemitones),
        "abruptVolumeChangeCount": abruptChangeCount,
        "wordCount": wordCount,
        "speechRateWpm": _round(speechRateWpm),
        "speechRateVariationWpm": _round(speechRateVariation),
        "longPauseCount": sum(event["eventType"] == "longPause" for event in events),
        "fragmentedSpeechSegmentCount": fragmentedSegments,
        "warnings": warnings,
    }
    return features, sorted(events, key=lambda event: (event["startTime"], event["eventType"]))


def _pitchTrack(samples: np.ndarray, sampleRate: int, speechThresholdDb: float) -> list[float]:
    windowSize = max(1, int(0.04 * sampleRate))
    hopSize = max(1, int(0.1 * sampleRate))
    minimumLag = max(1, int(sampleRate / 350))
    maximumLag = min(windowSize - 1, int(sampleRate / 70))
    pitches: list[float] = []
    if maximumLag <= minimumLag:
        return pitches
    for start in range(0, max(samples.size - windowSize + 1, 0), hopSize):
        window = samples[start : start + windowSize].astype(np.float64)
        rms = math.sqrt(float(np.mean(np.square(window))))
        if _db(rms) <= speechThresholdDb:
            continue
        window = (window - np.mean(window)) * np.hanning(windowSize)
        fftSize = 1 << (2 * windowSize - 1).bit_length()
        spectrum = np.fft.rfft(window, fftSize)
        correlation = np.fft.irfft(spectrum * np.conjugate(spectrum), fftSize)[:windowSize]
        if correlation[0] <= 0:
            continue
        region = correlation[minimumLag : maximumLag + 1]
        lag = minimumLag + int(np.argmax(region))
        reliability = correlation[lag] / correlation[0]
        if reliability >= 0.3:
            pitches.append(sampleRate / lag)
    return pitches


def _addTranscriptEvents(
    events: list[dict[str, Any]], transcript: dict[str, Any] | None, duration: float
) -> tuple[int, float | None, float | None]:
    words = _timestampedWords(transcript)
    if not words:
        return 0, None, None
    speechIntervals = [
        (float(segment.get("start", 0)), float(segment.get("end", 0)))
        for segment in (transcript or {}).get("segments", [])
        if segment.get("text", "").strip()
    ]
    speechSeconds = sum(max(0.0, end - start) for start, end in speechIntervals)
    speechRate = len(words) / (speechSeconds / 60) if speechSeconds > 0 else None

    windowRates: list[float] = []
    windowSize = 10.0
    for windowStart in np.arange(0, max(duration, windowSize), windowSize):
        windowEnd = min(windowStart + windowSize, duration)
        matched = [word for word in words if windowStart <= word["start"] < windowEnd]
        if len(matched) < 4 or windowEnd <= windowStart:
            continue
        rate = len(matched) / ((windowEnd - windowStart) / 60)
        windowRates.append(rate)
        if rate > 180:
            events.append(
                _event(
                    "rapidSpeech",
                    windowStart,
                    windowEnd,
                    {"wordsPerMinute": round(rate, 2), "wordCount": len(matched)},
                    "The timestamped transcript contained a high concentration of words in this window.",
                    "Replay this section and check whether each key point remained easy to follow.",
                    "medium",
                )
            )
        elif rate < 90:
            events.append(
                _event(
                    "slowSpeech",
                    windowStart,
                    windowEnd,
                    {"wordsPerMinute": round(rate, 2), "wordCount": len(matched)},
                    "The timestamped transcript contained a relatively low word rate in this window.",
                    "Review whether the pacing sounded deliberate or lost momentum.",
                    "medium",
                )
            )

    for previous, current in zip(words, words[1:]):
        gap = current["start"] - previous["end"]
        if gap >= 1.0 and not _eventOverlaps(events, "longPause", previous["end"], current["start"]):
            events.append(
                _event(
                    "longPause",
                    previous["end"],
                    current["start"],
                    {"wordGapSeconds": round(gap, 3)},
                    "The word timestamps contained a sustained gap between spoken words.",
                    "Review whether this pause supported organization or interrupted the answer.",
                    "high",
                )
            )
    variation = float(np.std(windowRates)) if len(windowRates) >= 2 else None
    return len(words), speechRate, variation


def _addFragmentationEvents(events: list[dict[str, Any]], transcript: dict[str, Any] | None) -> int:
    count = 0
    for segment in (transcript or {}).get("segments", []):
        words = str(segment.get("text") or "").split()
        duration = float(segment.get("end", 0)) - float(segment.get("start", 0))
        if 0 < len(words) <= 3 and duration < 2:
            count += 1
            events.append(
                _event(
                    "speechFragmentation",
                    float(segment.get("start", 0)),
                    float(segment.get("end", 0)),
                    {"wordCount": len(words)},
                    "The transcript produced a very short spoken fragment in this section.",
                    "Replay it before treating this as a delivery issue because transcription errors can also create fragments.",
                    "low" if segment.get("confidence") is None else "medium",
                )
            )
    return count


def _timestampedWords(transcript: dict[str, Any] | None) -> list[dict[str, float]]:
    words: list[dict[str, float]] = []
    for segment in (transcript or {}).get("segments", []):
        segmentWords = segment.get("words") or []
        if segmentWords:
            for word in segmentWords:
                if word.get("start") is not None and word.get("end") is not None:
                    words.append({"start": float(word["start"]), "end": float(word["end"])})
        else:
            textWords = str(segment.get("text") or "").split()
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            for index in range(len(textWords)):
                wordStart = start + (end - start) * index / max(len(textWords), 1)
                wordEnd = start + (end - start) * (index + 1) / max(len(textWords), 1)
                words.append({"start": wordStart, "end": wordEnd})
    return sorted(words, key=lambda word: word["start"])


def _maskEvents(
    mask: np.ndarray,
    times: np.ndarray,
    frameDuration: float,
    minimumDuration: float,
    eventType: str,
    explanation: str,
    coaching: str,
    measurements: dict[str, Any],
    reliability: str = "medium",
) -> list[dict[str, Any]]:
    output = []
    startIndex = None
    for index, active in enumerate(np.append(mask, False)):
        if active and startIndex is None:
            startIndex = index
        elif not active and startIndex is not None:
            start = float(times[startIndex])
            end = float(times[index - 1] + frameDuration)
            if end - start >= minimumDuration:
                output.append(_event(eventType, start, end, measurements, explanation, coaching, reliability))
            startIndex = None
    return output


def _event(
    eventType: str,
    start: float,
    end: float,
    measurements: dict[str, Any],
    explanation: str,
    coaching: str,
    reliability: str,
) -> dict[str, Any]:
    return {
        "eventType": eventType,
        "startTime": round(max(0.0, start), 3),
        "endTime": round(max(start, end), 3),
        "durationSeconds": round(max(0.0, end - start), 3),
        "measurements": measurements,
        "explanation": explanation,
        "coachingInterpretation": coaching,
        "reliability": reliability,
    }


def _eventOverlaps(events: list[dict[str, Any]], eventType: str, start: float, end: float) -> bool:
    return any(
        event["eventType"] == eventType
        and min(float(event["endTime"]), end) > max(float(event["startTime"]), start)
        for event in events
    )


def _db(value: float | None) -> float:
    return 20 * math.log10(max(float(value or 0.0), 1e-8))


def _round(value: float | None, digits: int = 3) -> float | None:
    return round(float(value), digits) if value is not None and math.isfinite(float(value)) else None


def _unavailableFeatures(sampleRate: int, channels: int, warnings: list[str]) -> dict[str, Any]:
    return {
        "analysisVersion": audioAnalysisVersion,
        "available": False,
        "durationSeconds": 0.0,
        "sampleRate": sampleRate,
        "sourceChannels": channels,
        "warnings": warnings,
    }

