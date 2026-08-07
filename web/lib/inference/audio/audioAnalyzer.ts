import type { EvidenceEvent, TranscriptArtifact } from "@/types/local";

export interface DecodedAudio { samples: Float32Array; sampleRate: number; channels: number; durationSeconds: number; }

export async function decodeAudio(blob: Blob): Promise<DecodedAudio> {
  const context = new AudioContext();
  try {
    const buffer = await context.decodeAudioData(await blob.arrayBuffer());
    const mono = new Float32Array(buffer.length);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const samples = buffer.getChannelData(channel);
      for (let index = 0; index < samples.length; index += 1) mono[index] += samples[index] / buffer.numberOfChannels;
    }
    return { samples: mono, sampleRate: buffer.sampleRate, channels: buffer.numberOfChannels, durationSeconds: buffer.duration };
  } finally { await context.close(); }
}

const db = (value: number) => 20 * Math.log10(Math.max(value, 1e-8));
const round = (value: number | null, digits = 3) => value === null || !Number.isFinite(value) ? null : Number(value.toFixed(digits));
const percentile = (values: number[], ratio: number) => {
  if (!values.length) return 0;
  const ordered = [...values].sort((a, b) => a - b); const rank = (ordered.length - 1) * ratio; const lower = Math.floor(rank); const upper = Math.ceil(rank);
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower);
};
const event = (eventType: string, startTime: number, endTime: number, explanation: string, coachingInterpretation: string, measurements: Record<string, unknown>, reliability: "low" | "medium" | "high" = "medium"): EvidenceEvent => ({ eventType, startTime: round(Math.max(0, startTime), 3) || 0, endTime: round(Math.max(startTime, endTime), 3) || 0, durationSeconds: round(Math.max(0, endTime - startTime), 3) || 0, explanation, coachingInterpretation, measurements, reliability, provenance: "rule" });

function contiguousEvents(mask: boolean[], times: number[], frameDuration: number, minimumDuration: number, eventType: string, explanation: string, coaching: string, measurements: Record<string, unknown>, reliability?: "low" | "medium" | "high"): EvidenceEvent[] {
  const result: EvidenceEvent[] = []; let start: number | undefined;
  for (let index = 0; index <= mask.length; index += 1) {
    if (mask[index] && start === undefined) start = index;
    if ((!mask[index] || index === mask.length) && start !== undefined) {
      const endIndex = index - 1; const begin = times[start]; const end = times[endIndex] + frameDuration;
      if (end - begin >= minimumDuration) result.push(event(eventType, begin, end, explanation, coaching, measurements, reliability));
      start = undefined;
    }
  }
  return result;
}

function words(transcript: TranscriptArtifact) { return transcript.segments.flatMap((segment) => segment.words?.length ? segment.words : segment.text.split(/\s+/).filter(Boolean).map((text, index, items) => ({ text, start: segment.start + ((segment.end - segment.start) * index) / items.length, end: segment.start + ((segment.end - segment.start) * (index + 1)) / items.length }))); }

export function analyzeAudio(samples: Float32Array, sampleRate: number, channels: number, transcript: TranscriptArtifact): { features: Record<string, unknown>; events: EvidenceEvent[] } {
  const duration = samples.length / sampleRate;
  if (!samples.length || !duration) return { features: { available: false, durationSeconds: 0, sampleRate, sourceChannels: channels, warnings: ["The decoded audio is empty."] }, events: [] };
  let sumSquares = 0; let peak = 0; let clippingSamples = 0;
  for (const sample of samples) { sumSquares += sample * sample; peak = Math.max(peak, Math.abs(sample)); if (Math.abs(sample) >= 0.99) clippingSamples += 1; }
  const rms = Math.sqrt(sumSquares / samples.length); const overallDb = db(rms); const clippingRatio = clippingSamples / samples.length;
  const frameSize = Math.max(1, Math.round(sampleRate * 0.05)); const hop = Math.max(1, Math.round(sampleRate * 0.025));
  const frameDb: number[] = []; const framePeak: number[] = []; const times: number[] = [];
  for (let start = 0; start < samples.length; start += hop) {
    const end = Math.min(samples.length, start + frameSize); let localSq = 0; let localPeak = 0;
    for (let index = start; index < end; index += 1) { localSq += samples[index] ** 2; localPeak = Math.max(localPeak, Math.abs(samples[index])); }
    frameDb.push(db(Math.sqrt(localSq / Math.max(1, end - start)))); framePeak.push(localPeak); times.push(start / sampleRate);
  }
  const noiseFloorDb = percentile(frameDb, 0.2); const threshold = Math.max(-48, Math.min(-32, noiseFloorDb + 8)); const speechMask = frameDb.map((value) => value > threshold); const voiced = frameDb.filter((_, index) => speechMask[index]);
  const speechRatio = speechMask.filter(Boolean).length / speechMask.length; const silenceRatio = 1 - speechRatio; const medianVoiced = percentile(voiced, 0.5); const emphasis = percentile(voiced, 0.85);
  const events: EvidenceEvent[] = [
    ...contiguousEvents(speechMask.map((value) => !value), times, 0.05, 0.8, "longPause", "The audio contained a sustained low-energy pause.", "Review whether this pause helped organize the answer or interrupted its flow.", { speechThresholdDb: round(threshold, 2) }),
    ...contiguousEvents(frameDb.map((value) => value < -75), times, 0.05, 0.25, "audioDropout", "The waveform dropped to near-digital silence.", "Check the microphone connection if this was not an intentional pause.", { thresholdDb: -75 }, "high"),
    ...contiguousEvents(speechMask.map((value, index) => value && frameDb[index] < medianVoiced - 8), times, 0.05, 0.3, "lowVolume", "A speech-like section was substantially quieter than the usual level.", "Practice keeping important words audible and at a consistent microphone distance.", { referenceDb: round(medianVoiced, 2) }),
    ...contiguousEvents(speechMask.map((value, index) => value && frameDb[index] >= emphasis), times, 0.05, 0.15, "strongVocalEmphasis", "The audio energy rose above the normal speech range.", "Check whether the emphasis landed on an important phrase you want to preserve.", { energyThresholdDb: round(emphasis, 2) }),
    ...contiguousEvents(framePeak.map((value) => value >= 0.999), times, 0.05, 0.05, "audioClipping", "Samples reached the digital amplitude ceiling.", "Move slightly farther from the microphone or reduce input gain.", { clippingThreshold: 0.999 }, "high"),
  ];
  const allWords = words(transcript); const speechSeconds = transcript.segments.reduce((total, segment) => total + Math.max(0, segment.end - segment.start), 0); const speechRate = allWords.length && speechSeconds ? allWords.length / (speechSeconds / 60) : null;
  const windowRates: number[] = [];
  for (let start = 0; start < duration; start += 10) {
    const end = Math.min(duration, start + 10); const within = allWords.filter((word) => word.start >= start && word.start < end);
    if (within.length < 4 || end <= start) continue;
    const rate = within.length / ((end - start) / 60); windowRates.push(rate);
    if (rate > 180) events.push(event("rapidSpeech", start, end, "The timestamped transcript contained a high concentration of words in this window.", "Replay this section and check whether each key point remained easy to follow.", { wordsPerMinute: round(rate, 2), wordCount: within.length }));
    if (rate < 90) events.push(event("slowSpeech", start, end, "The timestamped transcript contained a relatively low word rate in this window.", "Review whether the pacing sounded deliberate or lost momentum.", { wordsPerMinute: round(rate, 2), wordCount: within.length }));
  }
  for (let index = 1; index < allWords.length; index += 1) { const gap = allWords[index].start - allWords[index - 1].end; if (gap >= 1 && !events.some((item) => item.eventType === "longPause" && item.startTime < allWords[index].start && item.endTime > allWords[index - 1].end)) events.push(event("longPause", allWords[index - 1].end, allWords[index].start, "The word timestamps contained a sustained gap between spoken words.", "Review whether this pause supported organization or interrupted the answer.", { wordGapSeconds: round(gap, 3) }, "high")); }
  const noiseFrames = frameDb.filter((_, index) => !speechMask[index]); const snr = voiced.length && noiseFrames.length ? percentile(voiced, 0.5) - percentile(noiseFrames, 0.5) : null;
  const qualityReasons: string[] = []; if (clippingRatio > 0.01) qualityReasons.push("frequent clipping"); if (snr !== null && snr < 10) qualityReasons.push("limited speech-to-noise separation"); if (overallDb < -40) qualityReasons.push("low overall level");
  if (qualityReasons.length) events.push(event("lowAudioQuality", 0, duration, `The recording showed ${qualityReasons.join(", ")}.`, "Improve the recording setup before interpreting subtle vocal-delivery cues.", { reasons: qualityReasons, snrProxyDb: round(snr), clippingPercent: round(clippingRatio * 100, 4) }, "high"));
  return { features: { analysisVersion: "browser-audio-v1", available: peak >= 0.001, durationSeconds: round(duration, 5), sampleRate, sourceChannels: channels, overallRmsDb: round(overallDb), peakAmplitude: round(peak, 6), silenceRatio: round(silenceRatio, 5), speechRatio: round(speechRatio, 5), clippingPercentage: round(clippingRatio * 100, 6), dropoutRatio: round(frameDb.filter((value) => value < -75).length / frameDb.length, 5), noiseFloorDb: round(noiseFloorDb), speechThresholdDb: round(threshold), snrProxyDb: round(snr), volumeConsistencyStdDb: voiced.length > 1 ? round(Math.sqrt(voiced.reduce((total, value) => total + (value - medianVoiced) ** 2, 0) / voiced.length)) : null, energyVariationDb: voiced.length ? round(percentile(voiced, 0.9) - percentile(voiced, 0.1)) : null, wordCount: allWords.length, speechRateWpm: round(speechRate), speechRateVariationWpm: windowRates.length > 1 ? round(Math.sqrt(windowRates.reduce((sum, value) => sum + (value - (windowRates.reduce((a, b) => a + b, 0) / windowRates.length)) ** 2, 0) / windowRates.length)) : null, longPauseCount: events.filter((item) => item.eventType === "longPause").length, fragmentedSpeechSegmentCount: transcript.segments.filter((segment) => segment.text.split(/\s+/).filter(Boolean).length <= 3 && segment.end - segment.start < 2).length, warnings: peak < 0.001 ? ["The decoded audio is effectively silent, so vocal measurements are unavailable."] : [] }, events: events.sort((a, b) => a.startTime - b.startTime) };
}
