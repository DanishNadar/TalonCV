"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ChangeEvent, useCallback, useEffect, useState } from "react";
import { deleteSession, listSessions } from "@/lib/storage/sessionStore";
import { importSessionBundle } from "@/lib/storage/sessionBundle";
import type { LocalSession } from "@/types/local";

export function LocalInterviewHistory() {
  const [sessions, setSessions] = useState<LocalSession[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const load = useCallback(async () => { try { setSessions(await listSessions()); } finally { setLoading(false); } }, []);
  useEffect(() => { void load(); }, [load]);
  async function remove(session: LocalSession) { if (!window.confirm(`Delete the locally stored recording and review for “${session.context.interviewQuestion}”?`)) return; await deleteSession(session.id); await load(); }
  async function importBundle(event: ChangeEvent<HTMLInputElement>) { const file = event.target.files?.[0]; event.target.value = ""; if (!file) return; try { const session = await importSessionBundle(file); router.push(`/interview?id=${encodeURIComponent(session.id)}`); } catch (error) { window.alert(error instanceof Error ? error.message : "The TalonCV bundle could not be imported."); } }
  if (loading) return <div className="skeleton-list" aria-label="Loading local interview history"><span /><span /><span /></div>;
  if (!sessions.length) return <section className="empty-card"><h2>No local interviews yet</h2><p>Record, import an answer, or restore a TalonCV bundle in this browser.</p><div className="button-row"><Link className="button primary" href="/interview/new">Start a practice</Link><label className="button ghost">Import TalonCV Session<input hidden type="file" accept="application/zip,.zip" onChange={(event) => void importBundle(event)} /></label></div></section>;
  return <><div className="button-row"><label className="button ghost">Import TalonCV Session<input hidden type="file" accept="application/zip,.zip" onChange={(event) => void importBundle(event)} /></label></div><div className="history-list">{sessions.map((session) => <article className="history-row" key={session.id}><div className="history-main"><i className={`status-dot ${session.analysisState}`} /><div><h2>{session.context.interviewQuestion}</h2><p>{session.context.targetRole || "General interview practice"} · {new Date(session.createdAt).toLocaleDateString()}</p></div></div><dl><div><dt>Duration</dt><dd>{session.recording ? `${Math.round(session.recording.durationSeconds)} sec` : "—"}</dd></div><div><dt>Local status</dt><dd>{session.analysisState.replaceAll("_", " ")}</dd></div><div><dt>Score</dt><dd>{typeof session.overallScore === "number" ? Math.round(session.overallScore) : "—"}</dd></div></dl><div className="row-actions"><Link className="button secondary small" href={`/interview?id=${encodeURIComponent(session.id)}`}>Open review</Link><button className="text-button danger-text" onClick={() => void remove(session)}>Delete</button></div></article>)}</div></>;
}
