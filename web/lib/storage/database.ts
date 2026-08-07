import { openDB, type DBSchema, type IDBPDatabase } from "idb";
import type { LocalSession, StoredArtifact } from "@/types/local";

interface TalonCvDb extends DBSchema {
  sessions: { key: string; value: LocalSession; indexes: { "by-updated": string } };
  media: { key: string; value: { sessionId: string; blob: Blob; updatedAt: string }; indexes: { "by-session": string } };
  artifacts: { key: [string, string]; value: StoredArtifact; indexes: { "by-session": string } };
  settings: { key: string; value: unknown };
}

let dbPromise: Promise<IDBPDatabase<TalonCvDb>> | undefined;

export function getDatabase(): Promise<IDBPDatabase<TalonCvDb>> {
  if (typeof window === "undefined") throw new Error("Local TalonCV storage is available only in a browser.");
  dbPromise ??= openDB<TalonCvDb>("taloncv-local", 1, {
    upgrade(db) {
      const sessions = db.createObjectStore("sessions", { keyPath: "id" });
      sessions.createIndex("by-updated", "updatedAt");
      const media = db.createObjectStore("media", { keyPath: "sessionId" });
      media.createIndex("by-session", "sessionId");
      const artifacts = db.createObjectStore("artifacts", { keyPath: ["sessionId", "key"] });
      artifacts.createIndex("by-session", "sessionId");
      db.createObjectStore("settings");
    },
  });
  return dbPromise;
}

export async function clearAllLocalData(): Promise<void> {
  const db = await getDatabase();
  await Promise.all([db.clear("sessions"), db.clear("media"), db.clear("artifacts"), db.clear("settings")]);
}
