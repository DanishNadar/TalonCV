/// <reference lib="webworker" />
import { FaceDetector, FaceLandmarker, FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import { browserModels } from "@/config/browser-models";
import { applyBrowserCueClassifier, loadBrowserCueClassifier } from "@/lib/inference/vision/cueClassifier";
import { buildCueCalibration, frameLabels, type VisualFeatureRow } from "@/lib/inference/vision/cueRules";
import { CueStateMachine } from "@/lib/inference/vision/cueStateMachine";
import { BrowserYoloFaceDetector } from "@/lib/inference/vision/yoloFaceDetector";
import type { WorkerRequest, WorkerResponse } from "@/lib/inference/workerProtocol";

declare const self: DedicatedWorkerGlobalScope;
let faceDetector: FaceDetector | undefined; let faceLandmarker: FaceLandmarker | undefined; let poseLandmarker: PoseLandmarker | undefined;
let yoloFaceDetector: BrowserYoloFaceDetector | null | undefined;
let rows: VisualFeatureRow[] = []; let cancelled = false;
const visualFiles = browserModels.vision.files || [];

function send(message: WorkerResponse) { self.postMessage(message); }
function n(value: unknown): number | undefined { return typeof value === "number" && Number.isFinite(value) ? value : undefined; }
function point(landmarks: Array<{ x: number; y: number; visibility?: number }> | undefined, index: number) { const item = landmarks?.[index]; return item ? { x: item.x, y: item.y, visibility: item.visibility ?? 0 } : undefined; }

async function loadModels() {
  if (faceDetector && faceLandmarker && poseLandmarker) return;
  send({ type: "progress", progress: { stage: "loadingVision", progress: 0, message: "Loading local visual models", modelId: "vision", totalBytes: browserModels.vision.estimatedBytes } });
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm");
  const faceDetectorUrl = visualFiles[0].url; const faceLandmarkerUrl = visualFiles[1].url; const poseLandmarkerUrl = visualFiles[2].url;
  faceDetector = await FaceDetector.createFromOptions(vision, { baseOptions: { modelAssetPath: faceDetectorUrl }, runningMode: "VIDEO", minDetectionConfidence: 0.4 });
  send({ type: "progress", progress: { stage: "loadingVision", progress: 38, message: "Face detector ready", modelId: "vision", loadedBytes: visualFiles[0].bytes, totalBytes: browserModels.vision.estimatedBytes } });
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: faceLandmarkerUrl }, runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: false });
  send({ type: "progress", progress: { stage: "loadingVision", progress: 66, message: "Face landmarker ready", modelId: "vision", loadedBytes: visualFiles[0].bytes + visualFiles[1].bytes, totalBytes: browserModels.vision.estimatedBytes } });
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: poseLandmarkerUrl }, runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.4 });
  if (yoloFaceDetector === undefined) yoloFaceDetector = await BrowserYoloFaceDetector.create();
  send({ type: "progress", progress: { stage: "loadingVision", progress: 100, message: "Pose landmarker ready", modelId: "vision", loadedBytes: browserModels.vision.estimatedBytes, totalBytes: browserModels.vision.estimatedBytes } });
}

function sceneMetrics(frame: ImageBitmap) {
  const canvas = new OffscreenCanvas(Math.min(frame.width, 160), Math.min(frame.height, 90)); const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return { brightnessProxy: undefined, contrastProxy: undefined, sharpnessProxy: undefined };
  context.drawImage(frame, 0, 0, canvas.width, canvas.height); const data = context.getImageData(0, 0, canvas.width, canvas.height).data; let sum = 0; const lumas: number[] = [];
  for (let index = 0; index < data.length; index += 16) { const value = (data[index] * 0.2126 + data[index + 1] * 0.7152 + data[index + 2] * 0.0722) / 255; lumas.push(value); sum += value; }
  const mean = sum / Math.max(1, lumas.length); const deviation = Math.sqrt(lumas.reduce((total, value) => total + (value - mean) ** 2, 0) / Math.max(1, lumas.length)); let edges = 0;
  for (let y = 1; y < canvas.height - 1; y += 2) for (let x = 1; x < canvas.width - 1; x += 2) { const here = data[(y * canvas.width + x) * 4]; const horizontal = data[(y * canvas.width + x - 1) * 4] - data[(y * canvas.width + x + 1) * 4]; const vertical = data[((y - 1) * canvas.width + x) * 4] - data[((y + 1) * canvas.width + x) * 4]; edges += Math.abs(horizontal) + Math.abs(vertical) + Math.abs(here - data[((y - 1) * canvas.width + x - 1) * 4]); }
  return { brightnessProxy: Number(mean.toFixed(5)), contrastProxy: Number((deviation / 0.5).toFixed(5)), sharpnessProxy: Number((edges / Math.max(1, (canvas.width - 2) * (canvas.height - 2) / 4)).toFixed(3)) };
}

async function featureRow(frame: ImageBitmap, timestampSeconds: number): Promise<VisualFeatureRow> {
  if (!faceDetector || !faceLandmarker || !poseLandmarker) throw new Error("Visual models are not ready.");
  const timestampMs = Math.round(timestampSeconds * 1000); const detection = faceDetector.detectForVideo(frame, timestampMs); const mediaPipeFaces = detection.detections; const yoloFaces = yoloFaceDetector ? await yoloFaceDetector.detect(frame).catch(() => []) : []; const face = mediaPipeFaces[0]; const box = yoloFaces[0] ? { originX: yoloFaces[0].x, originY: yoloFaces[0].y, width: yoloFaces[0].width, height: yoloFaces[0].height } : face?.boundingBox; const faceResult = faceLandmarker.detectForVideo(frame, timestampMs); const faceLandmarks = faceResult.faceLandmarks?.[0]; const poseResult = poseLandmarker.detectForVideo(frame, timestampMs); const pose = poseResult.landmarks?.[0];
  const width = Math.max(1, frame.width); const height = Math.max(1, frame.height); const faceCenterX = box ? (box.originX + box.width / 2) / width : undefined; const faceCenterY = box ? (box.originY + box.height / 2) / height : undefined; const faceWidth = box ? box.width / width : undefined; const faceHeight = box ? box.height / height : undefined;
  const nose = point(faceLandmarks, 1) ?? point(pose, 0); const leftEyeTop = point(faceLandmarks, 159); const leftEyeBottom = point(faceLandmarks, 145); const rightEyeTop = point(faceLandmarks, 386); const rightEyeBottom = point(faceLandmarks, 374); const leftEyeOuter = point(faceLandmarks, 33); const rightEyeOuter = point(faceLandmarks, 263); const mouthLeft = point(faceLandmarks, 61); const mouthRight = point(faceLandmarks, 291); const lipTop = point(faceLandmarks, 13); const lipBottom = point(faceLandmarks, 14); const leftShoulder = point(pose, 11); const rightShoulder = point(pose, 12); const leftWrist = point(pose, 15); const rightWrist = point(pose, 16); const leftHip = point(pose, 23); const rightHip = point(pose, 24);
  const shoulderWidth = leftShoulder && rightShoulder ? Math.abs(rightShoulder.x - leftShoulder.x) : undefined; const postureProxy = shoulderWidth ? Math.abs((rightShoulder?.y ?? 0) - (leftShoulder?.y ?? 0)) / shoulderWidth : undefined; const cameraFacingProxy = nose && faceCenterX !== undefined && faceWidth ? Math.max(0, 1 - Math.abs(nose.x - faceCenterX) / Math.max(faceWidth / 2, 0.001)) : undefined;
  const mouthWidth = mouthLeft && mouthRight && faceWidth ? Math.abs(mouthRight.x - mouthLeft.x) / faceWidth : undefined; const mouthOpenness = lipTop && lipBottom && faceHeight ? Math.abs(lipBottom.y - lipTop.y) / faceHeight : undefined; const eyebrowRaise = leftEyeTop && rightEyeTop && faceHeight && faceLandmarks ? ((leftEyeTop.y - (faceLandmarks[105]?.y ?? leftEyeTop.y)) + (rightEyeTop.y - (faceLandmarks[334]?.y ?? rightEyeTop.y))) / 2 / faceHeight : undefined; const eyeOpenness = leftEyeTop && leftEyeBottom && rightEyeTop && rightEyeBottom && faceHeight ? ((Math.abs(leftEyeTop.y - leftEyeBottom.y) + Math.abs(rightEyeTop.y - rightEyeBottom.y)) / 2) / faceHeight : undefined; const headTilt = leftEyeOuter && rightEyeOuter ? Math.abs(leftEyeOuter.y - rightEyeOuter.y) / Math.max(0.001, Math.abs(leftEyeOuter.x - rightEyeOuter.x)) : undefined; const noseOffset = nose && faceCenterX !== undefined && faceWidth ? (nose.x - faceCenterX) / Math.max(0.001, faceWidth / 2) : undefined; const bodyCenterOffset = leftShoulder && rightShoulder ? Math.abs((leftShoulder.x + rightShoulder.x) / 2 - 0.5) : undefined; const bodyLean = leftShoulder && rightShoulder && leftHip && rightHip ? Math.abs((leftShoulder.x + rightShoulder.x - leftHip.x - rightHip.x) / 2) : undefined; const handRaised = [leftWrist, rightWrist].filter((wrist) => wrist && leftShoulder && rightShoulder && wrist.y < Math.min(leftShoulder.y, rightShoulder.y)).length;
  const result: VisualFeatureRow = { timestampSeconds, faceDetected: Boolean(box), faceCount: yoloFaces.length || mediaPipeFaces.length, faceMeshDetected: Boolean(faceLandmarks), poseDetected: Boolean(pose), faceCenterX, faceCenterY, faceWidth, faceHeight, faceAreaProxy: faceWidth && faceHeight ? faceWidth * faceHeight : undefined, faceEdgeMarginProxy: faceCenterX !== undefined && faceCenterY !== undefined && faceWidth !== undefined && faceHeight !== undefined ? Math.min(faceCenterX - faceWidth / 2, 1 - (faceCenterX + faceWidth / 2), faceCenterY - faceHeight / 2, 1 - (faceCenterY + faceHeight / 2)) : undefined, faceDetectionConfidence: yoloFaces[0]?.score ?? face?.categories?.[0]?.score, noseX: nose?.x, noseY: nose?.y, shoulderLeftX: leftShoulder?.x, shoulderLeftY: leftShoulder?.y, shoulderRightX: rightShoulder?.x, shoulderRightY: rightShoulder?.y, wristLeftX: leftWrist?.x, wristLeftY: leftWrist?.y, wristRightX: rightWrist?.x, wristRightY: rightWrist?.y, cameraFacingProxy, faceNoseOffsetXProxy: noseOffset, postureProxy, bodyCenterOffsetProxy: bodyCenterOffset, bodyLeanProxy: bodyLean, handRaisedCount: handRaised, mouthWidthProxy: mouthWidth, mouthOpennessProxy: mouthOpenness, eyebrowRaiseProxy: eyebrowRaise, eyeOpennessProxy: eyeOpenness, headTiltProxy: headTilt, ...sceneMetrics(frame) };
  const previous = rows[rows.length - 1];
  if (previous) {
    const distance = (ax?: number, ay?: number, bx?: number, by?: number) => ax === undefined || ay === undefined || bx === undefined || by === undefined ? undefined : Math.hypot(ax - bx, ay - by);
    result.headMovementProxy = distance(result.noseX, result.noseY, n(previous.noseX), n(previous.noseY)); result.headHorizontalChangeProxy = result.noseX === undefined || previous.noseX === undefined ? undefined : Math.abs(result.noseX - n(previous.noseX)!); result.noseYChangeProxy = result.noseY === undefined || previous.noseY === undefined ? undefined : Math.abs(result.noseY - n(previous.noseY)!); result.postureChangeProxy = distance(result.shoulderLeftX, result.shoulderLeftY, n(previous.shoulderLeftX), n(previous.shoulderLeftY)); result.handMovementProxy = Math.max(distance(result.wristLeftX, result.wristLeftY, n(previous.wristLeftX), n(previous.wristLeftY)) ?? 0, distance(result.wristRightX, result.wristRightY, n(previous.wristRightX), n(previous.wristRightY)) ?? 0); result.mouthMovementProxy = result.mouthOpennessProxy === undefined || previous.mouthOpennessProxy === undefined ? undefined : Math.abs(result.mouthOpennessProxy - n(previous.mouthOpennessProxy)!); result.blinkLikeChangeProxy = result.eyeOpennessProxy === undefined || previous.eyeOpennessProxy === undefined ? undefined : Math.abs(result.eyeOpennessProxy - n(previous.eyeOpennessProxy)!);
  }
  return result;
}

self.onmessage = async (message: MessageEvent<WorkerRequest>) => {
  const request = message.data;
  try {
    if (request.type === "cancel") { cancelled = true; return; }
    if (request.type === "dispose") { faceDetector?.close(); faceLandmarker?.close(); poseLandmarker?.close(); faceDetector = undefined; faceLandmarker = undefined; poseLandmarker = undefined; rows = []; send({ type: "ready", requestId: request.requestId }); return; }
    if (request.type === "load") { cancelled = false; await loadModels(); send({ type: "ready", requestId: request.requestId }); return; }
    if (request.type === "analyze") {
      const payload = request.payload as { action: "start" | "frame" | "finish"; frame?: ImageBitmap; timestampSeconds?: number; durationSeconds?: number; totalFrames?: number };
      if (payload.action === "start") { rows = []; cancelled = false; await loadModels(); send({ type: "ready", requestId: request.requestId }); return; }
      if (payload.action === "frame" && payload.frame) { if (!cancelled) { rows.push(await featureRow(payload.frame, payload.timestampSeconds || 0)); send({ type: "progress", requestId: request.requestId, progress: { stage: "analyzingVisual", progress: payload.totalFrames ? Math.round((rows.length / payload.totalFrames) * 100) : 0, message: `Analyzed ${rows.length} visual frames` } }); } payload.frame.close(); return; }
      if (payload.action === "finish") { const calibration = buildCueCalibration(rows); let learned = null; try { learned = await loadBrowserCueClassifier(); } catch { /* learned model is optional */ } const enriched = applyBrowserCueClassifier(rows.map((row) => ({ ...row, ruleFrameLabels: frameLabels(row, calibration).join(",") })), learned); const state = new CueStateMachine(2); for (const row of enriched) state.observe({ timestampSeconds: row.timestampSeconds, labels: String(row.frameLabels || row.ruleFrameLabels || "").split(",").filter(Boolean), confidence: row.faceDetectionConfidence ?? 0.7, provenance: row.mlCueLabel ? "combined" : "mediapipe", measurements: { faceDetectionConfidence: row.faceDetectionConfidence, cameraFacingProxy: row.cameraFacingProxy, postureProxy: row.postureProxy } }); send({ type: "result", requestId: request.requestId, payload: { rows: enriched, events: state.finish(payload.durationSeconds || rows.at(-1)?.timestampSeconds || 0), engine: "MediaPipe FaceDetector + FaceLandmarker + PoseLandmarker (browser-local equivalent face-detection path)" } }); return; }
    }
  } catch (error) { send({ type: "error", requestId: request.requestId, error: error instanceof Error ? error.message : "Visual analysis failed." }); }
};
