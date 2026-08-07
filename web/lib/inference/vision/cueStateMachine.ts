import type { EvidenceEvent } from "@/types/local";

export interface CueObservation { timestampSeconds: number; labels: string[]; confidence?: number; provenance?: EvidenceEvent["provenance"]; measurements?: Record<string, unknown>; }
interface ActiveCue { start: number; last: number; samples: number; confidenceTotal: number; provenance: EvidenceEvent["provenance"]; measurements: Record<string, unknown>; }

const descriptions: Record<string, [string, string]> = {
  cameraFacing: ["You appeared visually oriented toward the camera during this segment.", "Use this timestamp as a reference for camera-facing practice."],
  lookingAway: ["Camera-facing attention appeared lower during this segment.", "Practice returning toward the camera before a key point."],
  lookingDown: ["Your head position visually shifted downward during this segment.", "Replay it in context and return toward the camera before the next key point."],
  stablePosture: ["Your posture appeared visually stable during this segment.", "Use it as a comfortable framing and posture reference."],
  postureShift: ["Your posture visually shifted during this segment.", "Review whether the shift supported delivery or distracted from it."],
  handGestureActivity: ["Visible hand movement occurred during this segment.", "Review whether the gesture supported your delivery."],
  highHeadMovement: ["This segment contained noticeable head movement.", "Replay the passage and keep intentional movement comfortable and measured."],
  positiveExpression: ["Visible mouth and brow activity stood out during this segment. This is not an emotion claim.", "Review whether the visible expression supported the spoken point."],
  offCenterFraming: ["Your face moved noticeably away from the center of the camera frame.", "Adjust camera position or seating so your face stays comfortably centered."],
  faceMissing: ["No face was detected during this segment.", "Check lighting, framing, and whether you left the camera view before interpreting visual cues."],
  poseMissing: ["No upper-body pose was detected during this segment.", "Check camera framing if upper-body visual evidence is important."],
  dimLighting: ["The frame appeared dim during this segment.", "Increase even front lighting before relying on visual-detail cues."],
  overexposedLighting: ["The frame appeared unusually bright during this segment.", "Reduce direct front lighting or exposure before judging visual-detail cues."],
  lowContrast: ["The frame had low contrast during this segment.", "Increase separation between the speaker, lighting, and background."],
  blurryImage: ["The image-sharpness proxy was low during this segment.", "Check focus, motion, and camera cleanliness before relying on fine visual cues."],
  centeredFraming: ["Your face appeared comfortably centered in the frame.", "Use this timestamp as a framing reference."],
  faceTooClose: ["Your face occupied a large portion of the frame.", "Move slightly farther from the camera for a balanced interview frame."],
  faceTooFar: ["Your face appeared small in the frame.", "Move a little closer to improve visual clarity."],
  facePartiallyOutOfFrame: ["The face box reached the edge of the frame.", "Recenter the camera or seating position."],
  multipleFaces: ["More than one face was detected in the frame.", "Use a private, single-person frame for reliable visual analysis."],
  lowFaceConfidence: ["Face-detection confidence was limited during this segment.", "Treat other face cues cautiously and improve lighting or framing."],
  faceMeshMissing: ["A face was visible but detailed landmarks were unavailable.", "Improve lighting and keep the face unobstructed before interpreting fine cues."],
  neutralExpression: ["Facial movement appeared visually composed during this segment.", "Use it as a reference for a comfortable, steady delivery."],
  eyebrowRaise: ["Visible brow movement occurred during this segment.", "Replay it in context to see whether it supported the spoken point."],
  mouthOpen: ["Visible mouth openness occurred during this segment.", "This can be consistent with speaking; review it only in context."],
  speechLikeMouthActivity: ["Visible mouth movement aligned with active delivery.", "Review whether the articulation looked comfortable and natural."],
  eyesClosedLike: ["Eye openness was low during this segment.", "A normal blink can trigger this proxy, so replay it in context."],
  rapidBlinkLikeActivity: ["Repeated eye-openness changes occurred during this segment.", "Use the replay before making any delivery adjustment."],
  headTurnedLeft: ["Head orientation shifted toward the left side of the image.", "Practice returning toward the camera for key points."],
  headTurnedRight: ["Head orientation shifted toward the right side of the image.", "Practice returning toward the camera for key points."],
  headTilt: ["The eye-line tilt proxy increased during this segment.", "Replay it in context and keep a comfortable, level posture if desired."],
  lateralHeadMovement: ["Lateral head movement increased during this segment.", "Review whether the movement helped or distracted from the point."],
  nodding: ["Repeated vertical head movement was observed.", "Review whether the movement gave helpful emphasis."],
  shoulderTilt: ["Shoulder alignment appeared tilted during this segment.", "Use replay to find a comfortable, stable seated position."],
  bodyLean: ["Upper-body alignment shifted laterally during this segment.", "Review your seating and camera position in context."],
  bodyOffCenter: ["The upper-body center moved away from the frame center.", "Recenter before relying on visual delivery cues."],
  handsRaised: ["A visible wrist moved above shoulder level.", "Replay the gesture in context to decide whether it supported delivery."],
  possibleFidgeting: ["Head and posture movement increased together during this segment.", "Treat this as a replay prompt, not a claim about internal state."],
  default: ["An observable visual cue occurred during this segment.", "Replay the timestamp to review it in context."],
};

export class CueStateMachine {
  private readonly candidates = new Map<string, ActiveCue>();
  private readonly active = new Map<string, ActiveCue>();
  private readonly cooldowns = new Map<string, number>();
  private readonly events: EvidenceEvent[] = [];
  constructor(private readonly persistenceSamples = 2, private readonly cooldownSeconds = 0.35) {}

  observe(observation: CueObservation): void {
    const visible = new Set(observation.labels);
    for (const label of visible) {
      if ((this.cooldowns.get(label) ?? 0) > observation.timestampSeconds) continue;
      const candidate = this.candidates.get(label);
      if (candidate) { candidate.last = observation.timestampSeconds; candidate.samples += 1; candidate.confidenceTotal += observation.confidence ?? 0.7; }
      else this.candidates.set(label, { start: observation.timestampSeconds, last: observation.timestampSeconds, samples: 1, confidenceTotal: observation.confidence ?? 0.7, provenance: observation.provenance ?? "rule", measurements: observation.measurements ?? {} });
      const updated = this.candidates.get(label)!;
      if (updated.samples >= this.persistenceSamples && !this.active.has(label)) { this.active.set(label, updated); this.candidates.delete(label); }
    }
    for (const [label, active] of [...this.active]) {
      if (!visible.has(label)) { this.close(label, active, observation.timestampSeconds); }
    }
    for (const label of [...this.candidates.keys()]) if (!visible.has(label)) this.candidates.delete(label);
  }

  finish(endTime: number): EvidenceEvent[] {
    for (const [label, active] of [...this.active]) this.close(label, active, endTime);
    this.active.clear(); this.candidates.clear();
    return [...this.events];
  }

  private close(label: string, active: ActiveCue, endTime: number): void {
    const [explanation, coachingInterpretation] = descriptions[label] ?? descriptions.default;
    const end = Math.max(active.last, endTime);
    this.events.push({ eventType: label, startTime: Number(active.start.toFixed(3)), endTime: Number(end.toFixed(3)), durationSeconds: Number(Math.max(0, end - active.start).toFixed(3)), explanation, coachingInterpretation, reliability: active.samples >= 4 ? "high" : "medium", confidence: Number((active.confidenceTotal / active.samples).toFixed(3)), measurements: active.measurements, provenance: active.provenance });
    this.active.delete(label); this.cooldowns.set(label, end + this.cooldownSeconds);
  }
}
