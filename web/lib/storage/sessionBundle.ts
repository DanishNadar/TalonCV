import { strFromU8, unzipSync } from "fflate";
import { saveAnalysis } from "./artifactStore";
import { saveRecording } from "./mediaStore";
import { saveSession } from "./sessionStore";
import type { LocalAnalysis, LocalSession } from "@/types/local";

const mediaTypes: Record<string, string> = { webm: "video/webm", mp4: "video/mp4", ogg: "audio/ogg", wav: "audio/wav", m4a: "audio/mp4", mp3: "audio/mpeg" };

function parseJson<T>(contents: Uint8Array | undefined, label: string): T {
  if (!contents) throw new Error(`The TalonCV bundle is missing ${label}.`);
  try { return JSON.parse(strFromU8(contents)) as T; } catch { throw new Error(`The TalonCV bundle contains invalid ${label}.`); }
}

function validSession(value: unknown): value is LocalSession {
  const session = value as Partial<LocalSession>;
  return Boolean(session && typeof session === "object" && session.context && typeof session.context.interviewQuestion === "string" && session.context.interviewQuestion.trim());
}

export async function importSessionBundle(file: File): Promise<LocalSession> {
  if (!file.name.toLowerCase().endsWith(".zip")) throw new Error("Choose a TalonCV ZIP export.");
  let files: Record<string, Uint8Array>;
  try { files = unzipSync(new Uint8Array(await file.arrayBuffer())); } catch { throw new Error("This file is not a readable TalonCV ZIP export."); }
  const original = parseJson<unknown>(files["session.json"], "session.json");
  if (!validSession(original)) throw new Error("The session export has no valid interview context.");
  const recordingName = Object.keys(files).find((name) => /^recording\./i.test(name));
  if (!recordingName) throw new Error("The TalonCV bundle has no recording file.");
  const extension = recordingName.split(".").pop()?.toLowerCase() || "webm";
  const now = new Date().toISOString();
  const session: LocalSession = { ...original, id: crypto.randomUUID(), createdAt: now, updatedAt: now, recording: original.recording ? { ...original.recording, id: crypto.randomUUID(), createdAt: now, sizeBytes: files[recordingName].byteLength, mimeType: mediaTypes[extension] || original.recording.mimeType } : undefined };
  const sourceBytes = files[recordingName];
  const recordingBytes = sourceBytes.buffer.slice(sourceBytes.byteOffset, sourceBytes.byteOffset + sourceBytes.byteLength) as ArrayBuffer;
  await saveRecording(session.id, new Blob([recordingBytes], { type: session.recording?.mimeType || mediaTypes[extension] || "application/octet-stream" }));
  const analysis = files["analysis.json"] ? parseJson<LocalAnalysis>(files["analysis.json"], "analysis.json") : undefined;
  if (analysis && typeof analysis.report === "string" && analysis.transcript && Array.isArray(analysis.audioEvents) && Array.isArray(analysis.visualEvents)) {
    await saveAnalysis(session.id, { ...analysis, createdAt: now, sessionContext: session.context });
    session.analysisState = "complete";
  } else {
    session.analysisState = "idle";
  }
  await saveSession(session);
  return session;
}
