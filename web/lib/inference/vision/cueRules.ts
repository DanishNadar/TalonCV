export interface VisualFeatureRow extends Record<string, unknown> {
  timestampSeconds: number;
  faceDetected: boolean;
  faceMeshDetected: boolean;
  poseDetected: boolean;
  faceCenterX?: number;
  faceCenterY?: number;
  faceWidth?: number;
  faceHeight?: number;
  faceCount?: number;
  faceAreaProxy?: number;
  faceEdgeMarginProxy?: number;
  faceDetectionConfidence?: number;
  noseX?: number;
  noseY?: number;
  shoulderLeftX?: number;
  shoulderLeftY?: number;
  shoulderRightX?: number;
  shoulderRightY?: number;
  wristLeftX?: number;
  wristLeftY?: number;
  wristRightX?: number;
  wristRightY?: number;
  cameraFacingProxy?: number;
  faceNoseOffsetXProxy?: number;
  headMovementProxy?: number;
  headHorizontalChangeProxy?: number;
  noseYChangeProxy?: number;
  postureProxy?: number;
  postureChangeProxy?: number;
  bodyCenterOffsetProxy?: number;
  bodyLeanProxy?: number;
  handRaisedCount?: number;
  handMovementProxy?: number;
  mouthWidthProxy?: number;
  mouthOpennessProxy?: number;
  mouthMovementProxy?: number;
  eyebrowRaiseProxy?: number;
  eyeOpennessProxy?: number;
  blinkLikeChangeProxy?: number;
  headTiltProxy?: number;
  brightnessProxy?: number;
  contrastProxy?: number;
  sharpnessProxy?: number;
}

const number = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) ? value : undefined;
const percentile = (values: number[], ratio: number) => { if (!values.length) return undefined; const data = [...values].sort((a, b) => a - b); const index = (data.length - 1) * ratio; return data[Math.floor(index)] + (data[Math.ceil(index)] - data[Math.floor(index)]) * (index - Math.floor(index)); };
export interface CueCalibration { high: Record<string, number | undefined>; low: Record<string, number | undefined>; }
export function buildCueCalibration(rows: VisualFeatureRow[]): CueCalibration {
  const keys = ["cameraFacingProxy", "headMovementProxy", "headHorizontalChangeProxy", "noseYChangeProxy", "postureProxy", "postureChangeProxy", "handMovementProxy", "mouthOpennessProxy", "mouthMovementProxy", "eyebrowRaiseProxy", "eyeOpennessProxy", "blinkLikeChangeProxy", "headTiltProxy", "bodyCenterOffsetProxy", "bodyLeanProxy"];
  return { high: Object.fromEntries(keys.map((key) => [key, percentile(rows.map((row) => number(row[key])).filter((value): value is number => value !== undefined), 0.75)])), low: Object.fromEntries(keys.map((key) => [key, percentile(rows.map((row) => number(row[key])).filter((value): value is number => value !== undefined), 0.25)])) };
}
export function frameLabels(row: VisualFeatureRow, calibration: CueCalibration): string[] {
  const labels: string[] = [];
  if ((row.brightnessProxy ?? 1) < 0.18) labels.push("dimLighting");
  if ((row.brightnessProxy ?? 0) > 0.86) labels.push("overexposedLighting");
  if ((row.contrastProxy ?? 1) < 0.22) labels.push("lowContrast");
  if ((row.sharpnessProxy ?? Number.POSITIVE_INFINITY) < 22) labels.push("blurryImage");
  if (!row.faceDetected) labels.push("faceMissing");
  else {
    const centerX = row.faceCenterX; const centerY = row.faceCenterY; const faceHeight = row.faceHeight;
    if (centerX !== undefined && centerY !== undefined && (Math.abs(centerX - 0.5) > 0.2 || centerY < 0.12 || centerY > 0.68)) labels.push("offCenterFraming");
    else if (centerX !== undefined && centerY !== undefined) labels.push("centeredFraming");
    if (faceHeight !== undefined && faceHeight > 0.62) labels.push("faceTooClose"); if (faceHeight !== undefined && faceHeight < 0.18) labels.push("faceTooFar");
    if ((row.faceEdgeMarginProxy ?? 1) < 0.015) labels.push("facePartiallyOutOfFrame");
    if ((row.faceCount ?? 0) > 1) labels.push("multipleFaces");
    if ((row.faceDetectionConfidence ?? 1) < 0.55) labels.push("lowFaceConfidence");
    if (!row.faceMeshDetected) labels.push("faceMeshMissing");
    if ((row.cameraFacingProxy ?? 0) >= (calibration.high.cameraFacingProxy ?? 0.8)) labels.push("cameraFacing"); else if ((row.cameraFacingProxy ?? 1) <= (calibration.low.cameraFacingProxy ?? 0.45)) labels.push("lookingAway");
    if ((row.mouthOpennessProxy ?? 0) > 0.055) labels.push("mouthOpen");
    if ((row.mouthMovementProxy ?? 0) >= (calibration.high.mouthMovementProxy ?? Number.POSITIVE_INFINITY)) labels.push("speechLikeMouthActivity");
    if ((row.eyebrowRaiseProxy ?? 0) >= (calibration.high.eyebrowRaiseProxy ?? Number.POSITIVE_INFINITY)) labels.push("eyebrowRaise");
    if ((row.eyeOpennessProxy ?? 1) <= (calibration.low.eyeOpennessProxy ?? -1)) labels.push("eyesClosedLike");
    if ((row.blinkLikeChangeProxy ?? 0) >= (calibration.high.blinkLikeChangeProxy ?? Number.POSITIVE_INFINITY)) labels.push("rapidBlinkLikeActivity");
    if ((row.faceNoseOffsetXProxy ?? 0) < -0.35) labels.push("headTurnedLeft");
    if ((row.faceNoseOffsetXProxy ?? 0) > 0.35) labels.push("headTurnedRight");
    if ((row.headTiltProxy ?? 0) >= (calibration.high.headTiltProxy ?? 0.16)) labels.push("headTilt");
    if ((row.noseY ?? 0) > (row.faceCenterY ?? 1) + (row.faceHeight ?? 0) * 0.12) labels.push("lookingDown");
    if ((row.eyebrowRaiseProxy ?? 0) >= (calibration.high.eyebrowRaiseProxy ?? Number.POSITIVE_INFINITY) && (row.cameraFacingProxy ?? 0) >= (calibration.high.cameraFacingProxy ?? 0.8)) labels.push("positiveExpression");
    if ((row.cameraFacingProxy ?? 0) >= (calibration.high.cameraFacingProxy ?? 0.8) && (row.headMovementProxy ?? 1) <= (calibration.low.headMovementProxy ?? -1)) labels.push("neutralExpression");
  }
  if (!row.poseDetected) labels.push("poseMissing");
  else {
    if ((row.postureChangeProxy ?? 1) <= (calibration.low.postureChangeProxy ?? -1)) labels.push("stablePosture");
    if ((row.postureChangeProxy ?? 0) >= (calibration.high.postureChangeProxy ?? Number.POSITIVE_INFINITY)) labels.push("postureShift");
    if ((row.handMovementProxy ?? 0) >= (calibration.high.handMovementProxy ?? Number.POSITIVE_INFINITY)) labels.push("handGestureActivity");
    if ((row.handRaisedCount ?? 0) > 0) labels.push("handsRaised");
    if ((row.postureProxy ?? 0) >= (calibration.high.postureProxy ?? Number.POSITIVE_INFINITY)) labels.push("shoulderTilt");
    if ((row.bodyLeanProxy ?? 0) >= (calibration.high.bodyLeanProxy ?? Number.POSITIVE_INFINITY)) labels.push("bodyLean");
    if ((row.bodyCenterOffsetProxy ?? 0) >= (calibration.high.bodyCenterOffsetProxy ?? Number.POSITIVE_INFINITY)) labels.push("bodyOffCenter");
  }
  if ((row.headMovementProxy ?? 0) >= (calibration.high.headMovementProxy ?? Number.POSITIVE_INFINITY)) labels.push("highHeadMovement");
  if ((row.headHorizontalChangeProxy ?? 0) >= (calibration.high.headHorizontalChangeProxy ?? Number.POSITIVE_INFINITY)) labels.push("lateralHeadMovement");
  if ((row.noseYChangeProxy ?? 0) >= (calibration.high.noseYChangeProxy ?? Number.POSITIVE_INFINITY)) labels.push("nodding");
  if ((row.headMovementProxy ?? 0) >= (calibration.high.headMovementProxy ?? Number.POSITIVE_INFINITY) && (row.postureChangeProxy ?? 0) >= (calibration.high.postureChangeProxy ?? Number.POSITIVE_INFINITY)) labels.push("possibleFidgeting");
  return labels;
}
