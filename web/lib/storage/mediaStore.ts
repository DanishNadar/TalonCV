import { getDatabase } from "./database";

export async function saveRecording(sessionId: string, blob: Blob): Promise<void> {
  const db = await getDatabase();
  await db.put("media", { sessionId, blob, updatedAt: new Date().toISOString() });
}

export async function getRecording(sessionId: string): Promise<Blob | undefined> {
  return (await getDatabase()).get("media", sessionId).then((entry) => entry?.blob);
}

export async function deleteRecording(sessionId: string): Promise<void> { await (await getDatabase()).delete("media", sessionId); }
