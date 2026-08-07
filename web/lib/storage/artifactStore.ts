import { getDatabase } from "./database";
import type { LocalAnalysis } from "@/types/local";

export async function saveArtifact(sessionId: string, key: string, value: unknown): Promise<void> {
  await (await getDatabase()).put("artifacts", { sessionId, key, value, updatedAt: new Date().toISOString() });
}
export async function getArtifact<T>(sessionId: string, key: string): Promise<T | undefined> {
  return (await getDatabase()).get("artifacts", [sessionId, key]).then((entry) => entry?.value as T | undefined);
}
export async function saveAnalysis(sessionId: string, analysis: LocalAnalysis): Promise<void> { await saveArtifact(sessionId, "analysis", analysis); }
export async function getAnalysis(sessionId: string): Promise<LocalAnalysis | undefined> { return getArtifact<LocalAnalysis>(sessionId, "analysis"); }
export async function listArtifacts(sessionId: string): Promise<string[]> {
  const items = await (await getDatabase()).getAllFromIndex("artifacts", "by-session", sessionId);
  return items.map((item) => item.key);
}
