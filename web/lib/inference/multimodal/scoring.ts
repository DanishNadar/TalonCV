import type { EvidenceEvent } from "@/types/local";

/* ---------------------------------------------------------------- primitives */

const clamp = (value: number, low = 0, high = 100) => Math.max(low, Math.min(high, value));
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const round = (value: number) => Number(value.toFixed(1));

/** Piecewise-linear score: full marks inside [idealLow, idealHigh], decaying to
 *  zero at [hardLow, hardHigh]. Gives every measurement a real 0–100 range
 *  instead of nudging a fixed base. */
function band(value: number, hardLow: number, idealLow: number, idealHigh: number, hardHigh: number): number {
  if (value >= idealLow && value <= idealHigh) return 100;
  if (value < idealLow) return clamp(((value - hardLow) / Math.max(1e-6, idealLow - hardLow)) * 100);
  return clamp(((hardHigh - value) / Math.max(1e-6, hardHigh - idealHigh)) * 100);
}

/** Fraction of the take covered by any of the named event types, merged so
 *  overlapping events are not double counted. Coverage scales with recording
 *  length; raw event counts do not, which previously punished long answers. */
function coverage(events: EvidenceEvent[], types: string[], durationSeconds: number): number {
  if (durationSeconds <= 0) return 0;
  const spans = events
    .filter((event) => types.includes(event.eventType))
    .map((event) => [Math.max(0, event.startTime), Math.min(durationSeconds, event.endTime)] as const)
    .filter(([start, end]) => end > start)
    .sort((a, b) => a[0] - b[0]);
  let total = 0;
  let cursor = -1;
  for (const [start, end] of spans) {
    const from = Math.max(start, cursor);
    if (end > from) {
      total += end - from;
      cursor = end;
    }
  }
  return clamp(total / durationSeconds, 0, 1);
}

const rating = (score: number | null) =>
  score === null
    ? "Insufficient evidence"
    : score >= 88
      ? "Strong"
      : score >= 74
        ? "Effective"
        : score >= 60
          ? "Developing"
          : score >= 42
            ? "Needs work"
            : "Priority";

interface Dimension {
  score: number | null;
  rating: string;
  confidence: string;
  dataCoverage: string;
  evidence: string[];
  positiveObservations: string[];
  practiceAreas: string[];
  formula: string;
  componentBreakdown: Record<string, unknown>;
}

const dimension = (
  score: number | null,
  confidence: string,
  evidence: string[],
  positives: string[],
  practice: string[],
  formula: string,
  componentBreakdown: Record<string, unknown> = {},
): Dimension => ({
  score: score === null ? null : round(clamp(score)),
  rating: rating(score),
  confidence,
  dataCoverage: score === null ? "unavailable" : "available",
  evidence,
  positiveObservations: positives,
  practiceAreas: practice,
  formula,
  componentBreakdown,
});

const unavailable = (reason: string) =>
  dimension(null, "unavailable", [reason], [], [], "Excluded because the required evidence was unavailable.");

/** Weighted mean of sub-components, each already 0–100. */
const combine = (parts: Array<{ value: number; weight: number }>) => {
  const total = parts.reduce((sum, part) => sum + part.weight, 0);
  return total ? parts.reduce((sum, part) => sum + part.value * part.weight, 0) / total : 0;
};

/* ------------------------------------------------------------- thresholds */

// A take shorter than this, or with fewer words than this, cannot support a
// meaningful overall judgement. Reporting a number anyway was the reason a
// one-second clip scored in the seventies.
const minimumSeconds = 10;
const minimumWords = 20;

/* -------------------------------------------------------------- the scores */

export function buildCoachingScores(
  mediaInfo: { hasAudio: boolean; hasVideo: boolean; durationSeconds?: number },
  audioFeatures: Record<string, unknown>,
  audioEvents: EvidenceEvent[],
  response: Record<string, unknown>,
  visualEvents: EvidenceEvent[],
  moments: Array<Record<string, unknown>>,
): Record<string, unknown> {
  const duration = num(mediaInfo.durationSeconds) ?? num(audioFeatures.durationSeconds) ?? 0;
  const minutes = Math.max(duration / 60, 1 / 60);
  const audioAvailable = audioFeatures.available === true;
  const speechRatio = num(audioFeatures.speechRatio) ?? 0;
  const metrics = (response.metrics ?? {}) as Record<string, unknown>;
  const wordCount = num(metrics.wordCount) ?? 0;
  // Distinguish "said nothing" from "transcription failed": only the first is a
  // performance signal, and only it should be scored.
  const spokeAudibly = audioAvailable && speechRatio >= 0.15;
  const transcriptMissingButSpoke = response.available !== true && spokeAudibly;

  /* --- Recording quality ------------------------------------------------ */
  const recordingQuality = !audioAvailable
    ? unavailable("No audible waveform was decoded from this recording.")
    : (() => {
        const clipping = num(audioFeatures.clippingPercentage) ?? 0;
        const dropout = num(audioFeatures.dropoutRatio) ?? 0;
        const level = num(audioFeatures.overallRmsDb);
        const snr = num(audioFeatures.snrProxyDb);
        const levelScore = level === null ? 70 : band(level, -55, -30, -12, 0);
        const snrScore = snr === null ? 70 : band(snr, 0, 18, 60, 90);
        const clippingScore = clamp(100 - clipping * 25);
        const dropoutScore = clamp(100 - dropout * 220);
        const value = combine([
          { value: levelScore, weight: 0.3 },
          { value: snrScore, weight: 0.3 },
          { value: clippingScore, weight: 0.25 },
          { value: dropoutScore, weight: 0.15 },
        ]);
        return dimension(
          value,
          "high",
          [
            `Level ${level === null ? "unavailable" : `${level.toFixed(1)} dBFS`}`,
            `Speech-to-noise ${snr === null ? "unavailable" : `${snr.toFixed(1)} dB`}`,
            `Clipping ${clipping.toFixed(3)}%`,
          ],
          [
            ...(clippingScore >= 90 ? ["Almost no digital clipping was measured."] : []),
            ...(snrScore >= 80 ? ["Your voice separated clearly from background noise."] : []),
          ],
          [
            ...(levelScore < 60 ? ["Move closer to the microphone or raise input gain; the recording is quiet."] : []),
            ...(clippingScore < 70 ? ["Reduce input gain or back off the microphone to stop clipping."] : []),
            ...(snrScore < 55 ? ["Record somewhere quieter; background noise is competing with your voice."] : []),
          ],
          "Weighted mean of level, speech-to-noise, clipping, and dropout sub-scores.",
          { levelScore: round(levelScore), snrScore: round(snrScore), clippingScore: round(clippingScore), dropoutScore: round(dropoutScore) },
        );
      })();

  /* --- Vocal delivery --------------------------------------------------- */
  const vocalDelivery = !audioAvailable
    ? unavailable("No audible waveform was available for vocal-delivery analysis.")
    : !spokeAudibly
      ? dimension(
          12,
          "high",
          [`Speech occupied only ${Math.round(speechRatio * 100)}% of the recording.`],
          [],
          ["Record again and answer out loud; almost no speech was detected in this take."],
          "Scored low because the recording contained essentially no speech.",
          { speechRatio: round(speechRatio) },
        )
      : (() => {
          const pace = num(audioFeatures.speechRateWpm);
          const energy = num(audioFeatures.energyVariationDb);
          const consistency = num(audioFeatures.volumeConsistencyStdDb);
          const pausesPerMinute = audioEvents.filter((event) => event.eventType === "longPause").length / minutes;
          const paceScore = pace === null ? 60 : band(pace, 45, 110, 165, 250);
          const energyScore = energy === null ? 60 : band(energy, 0, 7, 20, 34);
          const pauseScore = band(pausesPerMinute, -6, 0, 1.5, 8);
          const consistencyScore = consistency === null ? 70 : band(consistency, 18, 0, 6, 18);
          const speechTimeScore = band(speechRatio, 0, 0.45, 0.9, 1.05);
          const value = combine([
            { value: paceScore, weight: 0.32 },
            { value: pauseScore, weight: 0.22 },
            { value: energyScore, weight: 0.2 },
            { value: speechTimeScore, weight: 0.16 },
            { value: consistencyScore, weight: 0.1 },
          ]);
          return dimension(
            value,
            wordCount >= minimumWords ? "high" : "limited",
            [
              `Pace ${pace === null ? "unavailable" : `${Math.round(pace)} wpm`}`,
              `Long pauses ${pausesPerMinute.toFixed(1)}/min`,
              `Energy range ${energy === null ? "unavailable" : `${energy.toFixed(1)} dB`}`,
            ],
            [
              ...(paceScore >= 85 ? ["Your speaking pace stayed in a comfortable, easy-to-follow range."] : []),
              ...(energyScore >= 85 ? ["Your delivery carried clear vocal variation rather than a flat tone."] : []),
              ...(pauseScore >= 90 ? ["You kept the answer moving without long stalls."] : []),
            ],
            [
              ...(pace !== null && pace > 165 ? ["Slow down — you are speaking faster than a listener can comfortably follow."] : []),
              ...(pace !== null && pace < 110 ? ["Lift the pace; the delivery is slower than a natural conversational range."] : []),
              ...(pauseScore < 60 ? ["Reduce long silent gaps; rehearse the transitions between your points."] : []),
              ...(energyScore < 55 ? ["Add vocal emphasis on key words; the delivery is close to monotone."] : []),
            ],
            "Weighted mean of pace, pause burden, energy variation, speaking time, and volume consistency.",
            { paceScore: round(paceScore), pauseScore: round(pauseScore), energyScore: round(energyScore), speechTimeScore: round(speechTimeScore), consistencyScore: round(consistencyScore), pausesPerMinute: round(pausesPerMinute) },
          );
        })();

  /* --- Answer quality --------------------------------------------------- */
  const rubric = (response.rubric ?? {}) as Record<string, { score?: number }>;
  const verbalResponseQuality = transcriptMissingButSpoke
    ? unavailable("Speech was audible but transcription did not complete, so answer analysis could not run.")
    : !mediaInfo.hasAudio
      ? unavailable("No audio track was available for answer analysis.")
      : !spokeAudibly
        ? dimension(
            8,
            "high",
            ["No spoken answer was detected in the recording."],
            [],
            ["Answer the question out loud so TalonCV can analyse structure, specificity, and relevance."],
            "Scored low because no answer was given.",
            { wordCount },
          )
        : (() => {
            const value = num(rubric.overallVerbalResponse?.score);
            const lengthPenalty = wordCount < minimumWords ? clamp(40 + (wordCount / minimumWords) * 40) : 100;
            const combined = value === null ? null : combine([
              { value, weight: 0.75 },
              { value: lengthPenalty, weight: 0.25 },
            ]);
            return dimension(
              combined,
              wordCount >= minimumWords ? String(response.confidence || "medium") : "limited",
              [
                `${wordCount} words`,
                `Fillers ${String(metrics.fillerRatePer100Words ?? 0)} per 100 words`,
                `STAR markers ${String((response.starAnalysis as Record<string, unknown>)?.componentsPresent ?? 0)}/4`,
              ],
              [
                ...(num(rubric.specificity?.score) !== null && num(rubric.specificity?.score)! >= 80 ? ["You supported the answer with concrete, specific detail."] : []),
                ...(num(rubric.clarity?.score) !== null && num(rubric.clarity?.score)! >= 85 ? ["The wording stayed clear, with few fillers or hedges."] : []),
              ],
              (response.practiceAreas as string[]) || [],
              "Deterministic answer rubric, discounted when the answer is too short to assess.",
              { rubricScore: value, lengthPenalty: round(lengthPenalty), wordCount },
            );
          })();

  /* --- Visual delivery -------------------------------------------------- */
  const visualDelivery = !mediaInfo.hasVideo
    ? unavailable("No video track was available for visual-delivery analysis.")
    : visualEvents.length === 0
      ? unavailable("No visual events were produced for this recording.")
      : (() => {
          const attention = coverage(visualEvents, ["lookingAway", "lookingDown", "headTurnedLeft", "headTurnedRight"], duration);
          const framing = coverage(visualEvents, ["faceMissing", "facePartiallyOutOfFrame", "faceTooClose", "faceTooFar", "offCenterFraming", "multipleFaces"], duration);
          const stability = coverage(visualEvents, ["postureShift", "highHeadMovement", "lateralHeadMovement", "possibleFidgeting", "bodyLean", "bodyOffCenter"], duration);
          const captureQuality = coverage(visualEvents, ["dimLighting", "overexposedLighting", "lowContrast", "blurryImage", "lowFaceConfidence"], duration);
          const engaged = coverage(visualEvents, ["cameraFacing", "centeredFraming", "stablePosture"], duration);

          const attentionScore = clamp(100 - attention * 145);
          const framingScore = clamp(100 - framing * 130);
          const stabilityScore = clamp(100 - stability * 110);
          const captureScore = clamp(100 - captureQuality * 95);
          const engagementScore = clamp(35 + engaged * 75);

          const value = combine([
            { value: attentionScore, weight: 0.3 },
            { value: framingScore, weight: 0.24 },
            { value: engagementScore, weight: 0.2 },
            { value: stabilityScore, weight: 0.16 },
            { value: captureScore, weight: 0.1 },
          ]);
          const percent = (ratio: number) => `${Math.round(ratio * 100)}%`;
          return dimension(
            value,
            duration >= minimumSeconds ? "medium" : "limited",
            [
              `Camera-facing coverage ${percent(engaged)}`,
              `Attention away ${percent(attention)}`,
              `Framing problems ${percent(framing)}`,
              `Movement ${percent(stability)}`,
            ],
            [
              ...(engagementScore >= 80 ? ["You stayed oriented toward the camera for most of the answer."] : []),
              ...(framingScore >= 90 ? ["Your framing stayed steady and well positioned."] : []),
              ...(stabilityScore >= 88 ? ["Your posture held steady through the answer."] : []),
            ],
            [
              ...(attentionScore < 65 ? [`Attention moved away from the camera for ${percent(attention)} of the take — practise returning to the lens on key points.`] : []),
              ...(framingScore < 70 ? [`Framing needs attention for ${percent(framing)} of the take — recentre and set a consistent distance.`] : []),
              ...(stabilityScore < 65 ? ["Movement was frequent enough to distract; plant your posture before answering."] : []),
              ...(captureScore < 65 ? ["Improve lighting or camera sharpness before reading much into visual cues."] : []),
            ],
            "Time-weighted coverage of attention, framing, engagement, stability, and capture-quality cues.",
            {
              attentionScore: round(attentionScore),
              framingScore: round(framingScore),
              engagementScore: round(engagementScore),
              stabilityScore: round(stabilityScore),
              captureScore: round(captureScore),
              coverageRatios: { attention: round(attention), framing: round(framing), stability: round(stability), engaged: round(engaged) },
            },
          );
        })();

  /* --- Multimodal alignment --------------------------------------------- */
  const strengthMoments = moments.filter((moment) => moment.classification === "strength").length;
  const reviewMoments = moments.filter((moment) => moment.classification === "review").length;
  const multimodalAlignment =
    !mediaInfo.hasAudio || !mediaInfo.hasVideo
      ? unavailable("Audio and video are both required for multimodal alignment.")
      : moments.length === 0
        ? unavailable("No cross-modal moment met the alignment rules, so alignment was not scored.")
        : (() => {
            const ratio = strengthMoments / Math.max(1, strengthMoments + reviewMoments);
            const density = reviewMoments / minutes;
            const value = combine([
              { value: clamp(ratio * 100), weight: 0.6 },
              { value: band(density, -4, 0, 1, 6), weight: 0.4 },
            ]);
            return dimension(
              value,
              "medium",
              [`Strength moments ${strengthMoments}`, `Review moments ${reviewMoments}`, `Review density ${density.toFixed(1)}/min`],
              moments.filter((moment) => moment.classification === "strength").map((moment) => String(moment.explanation)).slice(0, 3),
              moments.filter((moment) => moment.classification === "review").map((moment) => String(moment.coachingRecommendation)).slice(0, 3),
              "Share of aligned moments that were strengths, combined with how often review moments occurred per minute.",
              { strengthMoments, reviewMoments, reviewDensityPerMinute: round(density) },
            );
          })();

  const scores = { audioRecordingQuality: recordingQuality, vocalDelivery, verbalResponseQuality, visualDelivery, multimodalAlignment };

  /* --- Overall ----------------------------------------------------------- */
  const weights: Record<keyof typeof scores, number> = {
    audioRecordingQuality: 0.1,
    vocalDelivery: 0.22,
    verbalResponseQuality: 0.36,
    visualDelivery: 0.22,
    multimodalAlignment: 0.1,
  };
  const included = (Object.keys(scores) as Array<keyof typeof scores>).filter((key) => scores[key].score !== null);
  const totalWeight = included.reduce((sum, key) => sum + weights[key], 0);
  const weighted = included.length ? included.reduce((sum, key) => sum + Number(scores[key].score) * weights[key], 0) / totalWeight : null;

  // An overall number is only meaningful with enough recording and enough
  // speech behind it. Below that the dimensions are still reported, but the
  // headline score is withheld rather than averaged into a confident-looking
  // mid-seventies result.
  // A take too short to contain an answer is withheld outright — no amount of
  // good framing makes one second scoreable. A long-but-silent or long-but-brief
  // take *is* scored, because saying nothing is itself the result.
  const tooShort = duration < minimumSeconds;
  const transcriptionBroken = transcriptMissingButSpoke && included.length < 3;
  const insufficient = tooShort || transcriptionBroken;
  const tooFewWords = spokeAudibly && wordCount < minimumWords;

  const reasons = [
    ...(tooShort ? [`The recording is ${duration.toFixed(1)}s; at least ${minimumSeconds}s is needed for an overall score.`] : []),
    ...(transcriptionBroken ? ["Transcription did not complete, so the largest scoring component is missing."] : []),
  ];

  const overall =
    insufficient
      ? dimension(null, "insufficient", reasons, [], ["Record a longer, spoken answer so TalonCV can score the full take."], "Withheld: the recording did not meet the minimum evidence thresholds.", {
          durationSeconds: round(duration),
          wordCount,
          minimumSeconds,
          minimumWords,
        })
      : weighted === null
        ? unavailable("No usable audio, transcript, or visual evidence was available.")
        : dimension(
            weighted,
            included.length >= 4 && !tooFewWords ? "high" : included.length >= 3 ? "medium" : "limited",
            included.map((key) => `${key}: ${scores[key].score}`),
            [],
            [],
            "Weighted mean of the available dimension scores; unavailable dimensions are excluded and the remaining weights renormalized.",
            {
              components: Object.fromEntries(
                included.map((key) => [key, { score: scores[key].score, normalizedWeight: Number((weights[key] / totalWeight).toFixed(4)) }]),
              ),
              excludedComponents: (Object.keys(scores) as Array<keyof typeof scores>).filter((key) => !included.includes(key)),
            },
          );

  return {
    analysisVersion: "browser-scores-v2",
    scores: { ...scores, overallInterviewPracticeDelivery: overall },
    safetyNote:
      "These are explainable interview-practice coaching scores. They are not hiring scores and do not assess personality, honesty, intelligence, emotion, mental state, protected characteristics, or suitability for employment.",
  };
}
