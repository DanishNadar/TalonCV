"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { ChangeEvent, useCallback, useEffect, useMemo, useState } from "react";
import { getAnalysis } from "@/lib/storage/artifactStore";
import { importSessionBundle } from "@/lib/storage/sessionBundle";
import { buildSessionBundle, bundleFilename, downloadBlob } from "@/lib/storage/sessionExport";
import { deleteSession, listSessions } from "@/lib/storage/sessionStore";
import { MetricCard, StatusDot, TechnicalBadge } from "@/components/ui/primitives";
import type { LocalAnalysis, LocalSession } from "@/types/local";

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};

const dimensionLabels: Record<string, string> = {
  verbalResponseQuality: "Answer Quality",
  vocalDelivery: "Vocal Delivery",
  visualDelivery: "Visual Delivery",
  multimodalAlignment: "Multimodal Alignment",
  audioRecordingQuality: "Recording Quality",
};

interface Entry {
  session: LocalSession;
  analysis?: LocalAnalysis;
}

function runLabel(createdAt: string, index: number, total: number): string {
  void createdAt;
  return `RUN ${String(total - index).padStart(4, "0")}`;
}

function strongestArea(analysis: LocalAnalysis | undefined): string {
  const scores = asRecord(asRecord(analysis?.scores).scores);
  let best: { name: string; score: number } | undefined;
  for (const [name, raw] of Object.entries(scores)) {
    if (name === "overallInterviewPracticeDelivery") continue;
    const score = asRecord(raw).score;
    if (typeof score === "number" && (!best || score > best.score)) best = { name, score };
  }
  return best ? dimensionLabels[best.name] ?? best.name : "—";
}

export function LocalInterviewHistory({ showMetrics = true }: { showMetrics?: boolean }) {
  const router = useRouter();
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loading, setLoading] = useState(true);
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"recent" | "score" | "role">("recent");
  const [busy, setBusy] = useState<string>();
  const [message, setMessage] = useState<string>();

  const load = useCallback(async () => {
    try {
      const sessions = await listSessions();
      const loaded = await Promise.all(
        sessions.map(async (session) => ({
          session,
          analysis: session.analysisState === "complete" ? await getAnalysis(session.id) : undefined,
        })),
      );
      setEntries(loaded);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const metrics = useMemo(() => {
    const completed = entries.filter((entry) => entry.session.analysisState === "complete");
    const scores = completed.map((entry) => entry.session.overallScore).filter((value): value is number => typeof value === "number");
    const paces = completed
      .map((entry) => asRecord(entry.analysis?.audioFeatures).speechRateWpm)
      .filter((value): value is number => typeof value === "number");
    const totals = new Map<string, { sum: number; count: number }>();
    for (const entry of completed) {
      for (const [name, raw] of Object.entries(asRecord(asRecord(entry.analysis?.scores).scores))) {
        if (name === "overallInterviewPracticeDelivery") continue;
        const score = asRecord(raw).score;
        if (typeof score !== "number") continue;
        const current = totals.get(name) ?? { sum: 0, count: 0 };
        totals.set(name, { sum: current.sum + score, count: current.count + 1 });
      }
    }
    let review: string | undefined;
    let lowest = Infinity;
    for (const [name, value] of totals) {
      const mean = value.sum / value.count;
      if (mean < lowest) {
        lowest = mean;
        review = dimensionLabels[name] ?? name;
      }
    }
    const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);
    return {
      completed: completed.length,
      averageScore: mean(scores),
      averagePace: mean(paces),
      reviewArea: review,
    };
  }, [entries]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? entries.filter((entry) =>
          `${entry.session.context.interviewQuestion} ${entry.session.context.targetRole ?? ""}`.toLowerCase().includes(needle),
        )
      : entries;
    const sorted = [...filtered];
    if (sort === "score") sorted.sort((a, b) => (b.session.overallScore ?? -1) - (a.session.overallScore ?? -1));
    if (sort === "role") sorted.sort((a, b) => (a.session.context.targetRole ?? "").localeCompare(b.session.context.targetRole ?? ""));
    return sorted;
  }, [entries, query, sort]);

  async function remove(session: LocalSession) {
    if (!window.confirm(`Delete the locally stored recording and review for “${session.context.interviewQuestion}”?`)) return;
    await deleteSession(session.id);
    await load();
  }

  async function exportSession(entry: Entry) {
    if (!entry.analysis) return;
    setBusy(entry.session.id);
    try {
      downloadBlob(await buildSessionBundle(entry.session, entry.analysis), bundleFilename(entry.session));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "The session bundle could not be created.");
    } finally {
      setBusy(undefined);
    }
  }

  async function importBundle(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const session = await importSessionBundle(file);
      router.push(`/interview?id=${encodeURIComponent(session.id)}`);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "The TalonCV bundle could not be imported.");
    }
  }

  const importControl = (
    <label className="file-button">
      <span aria-hidden="true">↥</span>
      Import TalonCV Session
      <input type="file" accept="application/zip,.zip" onChange={(event) => void importBundle(event)} />
    </label>
  );

  if (loading) {
    return (
      <div className="skeleton-list" aria-busy="true" aria-label="Loading local interview history">
        <span />
        <span />
        <span />
      </div>
    );
  }

  return (
    <div className="stack-6">
      {showMetrics && metrics.completed > 0 ? (
        <div className="metric-grid">
          <MetricCard label="Practices completed" value={metrics.completed} note="Sessions with a finished local analysis" />
          <MetricCard
            label="Average practice score"
            value={metrics.averageScore === null ? "—" : Math.round(metrics.averageScore)}
            note="Mean across completed local sessions"
          />
          <MetricCard
            label="Average speaking pace"
            value={metrics.averagePace === null ? "—" : Math.round(metrics.averagePace)}
            unit={metrics.averagePace === null ? undefined : "wpm"}
            note="Measured from timestamped transcripts"
          />
          <MetricCard
            label="Priority review area"
            value={metrics.reviewArea ?? "—"}
            note="Lowest average scoring dimension"
          />
        </div>
      ) : null}

      {entries.length === 0 ? (
        <section className="empty-state">
          <h2>No local interviews yet</h2>
          <p>
            Record an answer, import a media file, or restore a TalonCV bundle. Everything you create stays inside this
            browser profile.
          </p>
          <div className="button-row">
            <Link className="button primary" href="/interview/new">
              Start a practice
            </Link>
            {importControl}
          </div>
        </section>
      ) : (
        <>
          <div className="history-toolbar">
            <div className="search">
              <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.4" />
                <path d="M10.5 10.5 14 14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
              <input
                type="search"
                aria-label="Search interview history"
                placeholder="Search question or role"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
              />
            </div>
            <label className="visually-hidden" htmlFor="history-sort">
              Sort sessions
            </label>
            <select id="history-sort" value={sort} onChange={(event) => setSort(event.target.value as typeof sort)}>
              <option value="recent">Most recent</option>
              <option value="score">Highest score</option>
              <option value="role">Target role</option>
            </select>
            {importControl}
          </div>

          {message ? (
            <p className="notice error" role="status">
              {message}
            </p>
          ) : null}

          <div className="history-list">
            {visible.map((entry, index) => {
              const { session, analysis } = entry;
              const moments = analysis?.moments.filter((moment) => moment.classification === "review").length ?? 0;
              return (
                <article className="history-row" key={session.id}>
                  <div className="history-main">
                    <div className="run-line">
                      <StatusDot tone={session.analysisState === "complete" ? "success" : session.analysisState === "failed" ? "error" : "warning"} />
                      <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                        {runLabel(session.createdAt, index, entries.length)}
                      </span>
                      <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                        {new Date(session.createdAt).toLocaleDateString()} ·{" "}
                        {new Date(session.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
                      </span>
                    </div>
                    <h2 title={session.context.interviewQuestion}>{session.context.interviewQuestion}</h2>
                    <span className="sub">
                      {session.context.targetRole || "General interview practice"}
                      {moments > 0 ? ` · ${moments} review moment${moments === 1 ? "" : "s"}` : ""}
                    </span>
                  </div>

                  <dl>
                    <div>
                      <dt>Duration</dt>
                      <dd>{session.recording ? `${Math.round(session.recording.durationSeconds)}s` : "—"}</dd>
                    </div>
                    <div>
                      <dt>Practice score</dt>
                      <dd>{typeof session.overallScore === "number" ? Math.round(session.overallScore) : "—"}</dd>
                    </div>
                    <div>
                      <dt>Strongest area</dt>
                      <dd title={strongestArea(analysis)}>{strongestArea(analysis)}</dd>
                    </div>
                  </dl>

                  <div className="row-actions">
                    {session.analysisState !== "complete" ? (
                      <TechnicalBadge tone="warning">{session.analysisState.replaceAll("_", " ")}</TechnicalBadge>
                    ) : null}
                    <Link className="button secondary small" href={`/interview?id=${encodeURIComponent(session.id)}`}>
                      Open
                    </Link>
                    <Link className="button ghost small" href={`/interview?id=${encodeURIComponent(session.id)}&rerun=1`}>
                      Reanalyze
                    </Link>
                    {analysis ? (
                      <button className="button ghost small" disabled={busy === session.id} onClick={() => void exportSession(entry)}>
                        {busy === session.id ? "Packaging…" : "Export"}
                      </button>
                    ) : null}
                    <button className="text-button danger-text" onClick={() => void remove(session)}>
                      Delete
                    </button>
                  </div>
                </article>
              );
            })}
          </div>

          {visible.length === 0 ? (
            <p className="notice" role="status">
              No local session matches “{query}”.
            </p>
          ) : null}
        </>
      )}
    </div>
  );
}
