import * as ort from "onnxruntime-web";

export interface BrowserFaceBox { x: number; y: number; width: number; height: number; score: number; }
const inputSize = 640;

function intersectionOverUnion(left: BrowserFaceBox, right: BrowserFaceBox): number {
  const width = Math.max(0, Math.min(left.x + left.width, right.x + right.width) - Math.max(left.x, right.x));
  const height = Math.max(0, Math.min(left.y + left.height, right.y + right.height) - Math.max(left.y, right.y));
  const overlap = width * height;
  return overlap / Math.max(1e-6, left.width * left.height + right.width * right.height - overlap);
}

function nonMaximumSuppression(boxes: BrowserFaceBox[]): BrowserFaceBox[] {
  const kept: BrowserFaceBox[] = [];
  for (const candidate of boxes.sort((left, right) => right.score - left.score)) if (kept.every((box) => intersectionOverUnion(box, candidate) < 0.45)) kept.push(candidate);
  return kept;
}

function outputBoxes(data: Float32Array, dimensions: readonly number[], width: number, height: number): BrowserFaceBox[] {
  const candidates: BrowserFaceBox[] = [];
  const channelsFirst = dimensions.length === 3 && dimensions[1] <= 8;
  const count = channelsFirst ? dimensions[2] : dimensions.at(-2) || 0;
  const values = channelsFirst ? (index: number, channel: number) => data[channel * count + index] : (index: number, channel: number) => data[index * (dimensions.at(-1) || 0) + channel];
  for (let index = 0; index < count; index += 1) {
    const score = values(index, 4);
    if (!Number.isFinite(score) || score < 0.4) continue;
    const centerX = values(index, 0) * width / inputSize;
    const centerY = values(index, 1) * height / inputSize;
    const boxWidth = values(index, 2) * width / inputSize;
    const boxHeight = values(index, 3) * height / inputSize;
    if (boxWidth <= 1 || boxHeight <= 1) continue;
    candidates.push({ x: Math.max(0, centerX - boxWidth / 2), y: Math.max(0, centerY - boxHeight / 2), width: Math.min(width, boxWidth), height: Math.min(height, boxHeight), score });
  }
  return nonMaximumSuppression(candidates);
}

export class BrowserYoloFaceDetector {
  private constructor(private readonly session: ort.InferenceSession) {}

  static async create(url = "/models/yolo11n-face.onnx"): Promise<BrowserYoloFaceDetector | null> {
    try {
      const response = await fetch(url, { method: "HEAD", cache: "force-cache" });
      if (!response.ok) return null;
      return new BrowserYoloFaceDetector(await ort.InferenceSession.create(url, { executionProviders: ["wasm"] }));
    } catch { return null; }
  }

  async detect(frame: ImageBitmap): Promise<BrowserFaceBox[]> {
    const canvas = new OffscreenCanvas(inputSize, inputSize);
    const context = canvas.getContext("2d");
    if (!context) return [];
    context.drawImage(frame, 0, 0, inputSize, inputSize);
    const pixels = context.getImageData(0, 0, inputSize, inputSize).data;
    const input = new Float32Array(3 * inputSize * inputSize);
    for (let pixel = 0; pixel < inputSize * inputSize; pixel += 1) {
      input[pixel] = pixels[pixel * 4] / 255;
      input[inputSize * inputSize + pixel] = pixels[pixel * 4 + 1] / 255;
      input[inputSize * inputSize * 2 + pixel] = pixels[pixel * 4 + 2] / 255;
    }
    const inputName = this.session.inputNames[0];
    const outputs = await this.session.run({ [inputName]: new ort.Tensor("float32", input, [1, 3, inputSize, inputSize]) });
    const output = outputs[this.session.outputNames[0]];
    return outputBoxes(output.data as Float32Array, output.dims, frame.width, frame.height);
  }
}
