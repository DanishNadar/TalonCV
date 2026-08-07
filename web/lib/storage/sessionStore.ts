import { getDatabase } from "./database";
import { deleteRecording } from "./mediaStore";
import type { LocalSession, LocalSessionContext } from "@/types/local";

export async function createSession(context: LocalSessionContext): Promise<LocalSession> {
  const now = new Date().toISOString();
  const session: LocalSession = { id: crypto.randomUUID(), createdAt: now, updatedAt: now, context, analysisState: "idle" };
  await (await getDatabase()).put("sessions", session);
  return session;
}
export async function saveSession(session: LocalSession): Promise<void> {
  session.updatedAt = new Date().toISOString();
  await (await getDatabase()).put("sessions", session);
}
export async function getSession(id: string): Promise<LocalSession | undefined> { return (await getDatabase()).get("sessions", id); }
export async function listSessions(): Promise<LocalSession[]> {
  const values = await (await getDatabase()).getAllFromIndex("sessions", "by-updated");
  return values.sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}
export async function deleteSession(id: string): Promise<void> {
  const db = await getDatabase();
  const tx = db.transaction(["sessions", "artifacts"], "readwrite");
  await tx.objectStore("sessions").delete(id);
  const artifactKeys = await tx.objectStore("artifacts").index("by-session").getAllKeys(id);
  await Promise.all(artifactKeys.map((key) => tx.objectStore("artifacts").delete(key)));
  await tx.done;
  await deleteRecording(id);
}
