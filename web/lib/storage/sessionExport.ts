import { strToU8, zipSync } from "fflate";
import { getRecording } from "./mediaStore";
import type { LocalAnalysis, LocalSession } from "@/types/local";

const json = (value: unknown) => strToU8(JSON.stringify(value, null, 2));

/** Files that make up a portable TalonCV session bundle. `importSessionBundle`
 *  reads `session.json`, `analysis.json`, and `recording.*` back out. */
export async function buildSessionBundle(session: LocalSession, analysis: LocalAnalysis): Promise<Blob> {
  const recording = await getRecording(session.id);
  const extension = recording?.type.split("/")[1]?.split(";")[0] || "webm";
  const files: Record<string, Uint8Array> = {
    "report.md": strToU8(analysis.report),
    "analysis.json": json(analysis),
    "session.json": json(session),
    "transcript.txt": strToU8(analysis.transcript.text),
    "transcript.json": json(analysis.transcript),
    "audio_features.json": json(analysis.audioFeatures),
    "audio_events.json": json(analysis.audioEvents),
    "response_analysis.json": json(analysis.responseAnalysis),
    "semantic_analysis.json": json(analysis.semanticAnalysis),
    "visual_features.json": json(analysis.visualFeatures),
    "visual_events.json": json(analysis.visualEvents),
    "multimodal.json": json(analysis.moments),
    "scores.json": json(analysis.scores),
  };
  if (analysis.localCoaching?.available && analysis.localCoaching.text) {
    files["coaching.md"] = strToU8(analysis.localCoaching.text);
  }
  if (recording) files[`recording.${extension}`] = new Uint8Array(await recording.arrayBuffer());
  return new Blob([zipSync(files)], { type: "application/zip" });
}

/** Hands a blob to the browser as a download. The object URL is released on the
 *  next task so the navigation has already been queued. */
export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export function downloadText(contents: string, filename: string, type: string): void {
  downloadBlob(new Blob([contents], { type }), filename);
}

export const bundleFilename = (session: LocalSession) => `taloncv-local-${session.id}.zip`;
