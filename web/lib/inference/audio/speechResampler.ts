/**
 * Transformers.js accepts raw Float32Array audio but, unlike URL input, it
 * cannot infer or convert the array's sample rate. Whisper expects 16 kHz.
 * Browser AudioContext decoding normally yields 44.1 or 48 kHz, so passing its
 * samples through unchanged makes speech run at the wrong speed for the model.
 */
export const speechSampleRate = 16_000;

export function resampleForSpeech(
  samples: Float32Array,
  sourceSampleRate: number,
  targetSampleRate = speechSampleRate,
): Float32Array {
  if (!Number.isFinite(sourceSampleRate) || sourceSampleRate <= 0) {
    throw new Error("The decoded audio did not include a valid sample rate.");
  }
  if (!Number.isFinite(targetSampleRate) || targetSampleRate <= 0) {
    throw new Error("The speech model does not have a valid target sample rate.");
  }
  if (!samples.length || sourceSampleRate === targetSampleRate) return samples;

  const sourceSamplesPerTargetSample = sourceSampleRate / targetSampleRate;
  const outputLength = Math.max(1, Math.round(samples.length / sourceSamplesPerTargetSample));
  const output = new Float32Array(outputLength);

  // Area averaging provides the low-pass step needed when downsampling and
  // avoids the severe timing error from treating 48 kHz samples as 16 kHz.
  for (let outputIndex = 0; outputIndex < outputLength; outputIndex += 1) {
    const start = outputIndex * sourceSamplesPerTargetSample;
    const end = Math.min(samples.length, (outputIndex + 1) * sourceSamplesPerTargetSample);
    const first = Math.floor(start);
    const last = Math.min(samples.length - 1, Math.ceil(end) - 1);
    let total = 0;
    let weight = 0;

    for (let sourceIndex = first; sourceIndex <= last; sourceIndex += 1) {
      const overlap = Math.max(0, Math.min(end, sourceIndex + 1) - Math.max(start, sourceIndex));
      total += samples[sourceIndex] * overlap;
      weight += overlap;
    }
    output[outputIndex] = weight ? total / weight : samples[Math.min(samples.length - 1, first)] || 0;
  }

  return output;
}
