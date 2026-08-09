"use client";

import { Fragment, useState, type ReactNode } from "react";
import { Meter, TechnicalBadge, TechnicalPanel, PracticeScore } from "@/components/ui/primitives";
import type { EvidenceEvent, LocalAnalysis } from "@/types/local";

export const reviewTabs = [
  "Overview",
  "Transcript",
  "Answer Quality",
  "Vocal Delivery",
  "Visual Cues",
  "Multimodal Moments",
  "Full Report",
  "Export",
] as const;

export type Tab = (typeof reviewTabs)[number];

/* ------------------------------------------------------------------ helpers */

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {};
const num = (value: unknown): number | null => (typeof value === "number" && Number.isFinite(value) ? value : null);
const list = (value: unknown): string[] => (Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []);

export const timecode = (value: unknown): string => {
  const seconds = num(value);
  if (seconds === null) return "—";
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
};

const titleize = (value: string) =>
  value.replace(/([A-Z])/g, " $1").replace(/^./, (character) => character.toUpperCase()).trim();

const dimensions: Array<{ key: string; label: string }> = [
  { key: "verbalResponseQuality", label: "Answer Quality" },
  { key: "vocalDelivery", label: "Vocal Delivery" },
  { key: "visualDelivery", label: "Visual Delivery" },
  { key: "multimodalAlignment", label: "Multimodal Alignment" },
  { key: "audioRecordingQuality", label: "Recording Quality" },
];

const scoreTone = (score: number | null) => (score === null ? "neutral" : score >= 70 ? "success" : score >= 55 ? "warning" : "accent");

/** Cues that fire continuously during ordinary speech. They are real
 *  measurements and stay in the exported evidence, but listing them as things to
 *  "review" is noise, so the Visual Cues tab keeps them behind a toggle. */
const ambientCues = new Set([
  "mouthOpen",
  "speechLikeMouthActivity",
  "eyebrowRaise",
  "neutralExpression",
  "positiveExpression",
  "eyesClosedLike",
  "rapidBlinkLikeActivity",
  "nodding",
  "headTilt",
  "faceMeshMissing",
]);

const isActionableCue = (event: EvidenceEvent) => !ambientCues.has(event.eventType);

/** Cues that stop the analysis from seeing you at all, or that a viewer would
 *  notice immediately. These are the ones worth acting on first. */
const criticalCues = new Set([
  "faceMissing",
  "poseMissing",
  "facePartiallyOutOfFrame",
  "faceTooClose",
  "faceTooFar",
  "multipleFaces",
  "blurryImage",
  "dimLighting",
  "overexposedLighting",
  "lowFaceConfidence",
]);

/** Cues worth reviewing but not disqualifying: attention, framing drift, and
 *  movement. */
const cautionCues = new Set([
  "lookingAway",
  "lookingDown",
  "headTurnedLeft",
  "headTurnedRight",
  "lateralHeadMovement",
  "highHeadMovement",
  "postureShift",
  "possibleFidgeting",
  "bodyLean",
  "bodyOffCenter",
  "shoulderTilt",
  "offCenterFraming",
  "lowContrast",
]);

type Severity = "critical" | "caution" | "positive" | "neutral";

/** Vocal events, graded the same way: red when the recording itself is
 *  compromised, amber when the delivery is worth revisiting. */
const criticalVocalCues = new Set(["audioDropout", "lowAudioQuality", "audioClipping"]);
const cautionVocalCues = new Set(["longPause", "lowVolume", "rapidSpeech", "slowSpeech"]);

function cueSeverity(eventType: string): Severity {
  if (criticalCues.has(eventType) || criticalVocalCues.has(eventType)) return "critical";
  if (cautionCues.has(eventType) || cautionVocalCues.has(eventType)) return "caution";
  if (["cameraFacing", "stablePosture", "centeredFraming", "handGestureActivity", "strongVocalEmphasis"].includes(eventType)) return "positive";
  return "neutral";
}

const severityLabel: Record<Severity, string> = {
  critical: "High priority",
  caution: "Review",
  positive: "Strength",
  neutral: "Context",
};

const severityTone: Record<Severity, "error" | "warning" | "success" | "neutral"> = {
  critical: "error",
  caution: "warning",
  positive: "success",
  neutral: "neutral",
};

const severityRank: Record<Severity, number> = { critical: 0, caution: 1, neutral: 2, positive: 3 };

function Stat({ label, value, note }: { label: string; value: ReactNode; note?: string }) {
  return (
    <article className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {note ? <span className="metric-note">{note}</span> : null}
    </article>
  );
}

function Unavailable({ children }: { children: ReactNode }) {
  return (
    <p className="notice" role="status">
      <span aria-hidden="true">◇</span>
      <span>{children}</span>
    </p>
  );
}

/* --------------------------------------------------------------- event list */

function EventList({ events, seek, severity = false }: { events: EvidenceEvent[]; seek: (time: number) => void; severity?: boolean }) {
  if (!events.length) return <Unavailable>No reliable timestamped evidence was available for this modality.</Unavailable>;
  return (
    <div className="evidence-list">
      {events.map((event, index) => {
        const level = severity ? cueSeverity(event.eventType) : "neutral";
        return (
          <article className={severity ? `sev-${level}` : undefined} key={`${event.eventType}-${event.startTime}-${index}`}>
            <button className="timestamp-button" onClick={() => seek(event.startTime)}>
              {timecode(event.startTime)}
            </button>
            <div className="ev-body">
              <div className="ev-title">
                <strong>{titleize(event.eventType)}</strong>
                {severity ? <TechnicalBadge tone={severityTone[level]}>{severityLabel[level]}</TechnicalBadge> : null}
              </div>
              <p>{event.explanation}</p>
              {event.coachingInterpretation ? <p>{event.coachingInterpretation}</p> : null}
              <div className="ev-meta">
                <TechnicalBadge plain>{event.reliability || "medium"} reliability</TechnicalBadge>
                <TechnicalBadge plain>{event.durationSeconds.toFixed(2)}s</TechnicalBadge>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

/* ----------------------------------------------------------- timeline strip */

function TimelineStrip({ analysis }: { analysis: LocalAnalysis }) {
  const total = Math.max(analysis.mediaInfo.durationSeconds, 0.001);
  const band = (event: { startTime: number; endTime: number }) => ({
    left: `${Math.max(0, Math.min(100, (event.startTime / total) * 100))}%`,
    width: `${Math.max(0.6, Math.min(100, ((event.endTime - event.startTime) / total) * 100))}%`,
  });
  return (
    <div className="timeline-strip">
      <div className="timeline-track" aria-hidden="true">
        <span className="lane audio">
          {analysis.audioEvents.map((event, index) => (
            <i key={`a${index}`} style={band(event)} />
          ))}
        </span>
        <span className="lane visual">
          {analysis.visualEvents.map((event, index) => (
            <i key={`v${index}`} style={band(event)} />
          ))}
        </span>
        <span className="lane moment">
          {analysis.moments.map((moment, index) => (
            <i key={`m${index}`} style={band({ startTime: Number(moment.startTime), endTime: Number(moment.endTime) })} />
          ))}
        </span>
      </div>
      <div className="timeline-legend">
        <span>
          <i style={{ background: "var(--talon-info)" }} /> Vocal events · {analysis.audioEvents.length}
        </span>
        <span>
          <i style={{ background: "var(--talon-text-tertiary)" }} /> Visual events · {analysis.visualEvents.length}
        </span>
        <span>
          <i style={{ background: "var(--talon-red)" }} /> Aligned moments · {analysis.moments.length}
        </span>
        <span>Duration · {timecode(analysis.mediaInfo.durationSeconds)}</span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- moment list */

function Moments({ analysis, seek }: { analysis: LocalAnalysis; seek: (time: number) => void }) {
  if (!analysis.moments.length) {
    return (
      <Unavailable>
        No cross-modal moment met the alignment rules for this take. Individual vocal and visual events remain available
        in their own tabs.
      </Unavailable>
    );
  }
  return (
    <div className="moment-list">
      {analysis.moments.map((moment, index) => {
        const classification = String(moment.classification || "context");
        const cueTypes = [...list(moment.audioEvents), ...list(moment.visualEvents)];
        const worst = cueTypes.some((type) => cueSeverity(type) === "critical") ? "critical" : classification;
        const severityClass = classification === "review" ? worst : classification;
        const audioEvents = cueTypes.length ? list(moment.audioEvents).map(titleize).join(", ") : "";
        const visualEvents = list(moment.visualEvents).map(titleize).join(", ");
        return (
          <article className={`moment ${severityClass}`} key={`${String(moment.alignmentCategory)}-${index}`}>
            <div className="moment-rail">
              <button className="timestamp-button" onClick={() => seek(Number(moment.startTime))}>
                {timecode(moment.startTime)}
              </button>
              <TechnicalBadge tone={classification === "strength" ? "success" : severityClass === "critical" ? "error" : classification === "review" ? "warning" : "neutral"}>
                {severityClass === "critical" ? "revisit" : classification}
              </TechnicalBadge>
            </div>
            <div className="moment-body">
              <blockquote className="moment-quote">
                {String(moment.transcriptExcerpt || "No overlapping transcript excerpt was available.")}
              </blockquote>
              <div className="moment-tracks">
                <div className="track-row">
                  <span className="track-name">Vocal</span>
                  <span className="track-value">{audioEvents || "No overlapping vocal event."}</span>
                </div>
                <div className="track-row">
                  <span className="track-name">Visual</span>
                  <span className="track-value">{visualEvents || "No overlapping visual event."}</span>
                </div>
                <div className="track-row">
                  <span className="track-name">Alignment</span>
                  <span className="track-value">{titleize(String(moment.alignmentCategory || "aligned evidence"))}</span>
                </div>
              </div>
              <div className="moment-interpretation">
                <span className="mono">TalonCV interpretation</span>
                <p>{String(moment.explanation || "")}</p>
                <p>
                  <strong>Practice · </strong>
                  {String(moment.coachingRecommendation || "Replay this timestamp in context.")}
                </p>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
}

/* --------------------------------------------------------------- visual cues */

function VisualCuesPanel({ analysis, seek }: { analysis: LocalAnalysis; seek: (time: number) => void }) {
  const [showAll, setShowAll] = useState(false);
  const actionable = analysis.visualEvents.filter(isActionableCue);
  const ambient = analysis.visualEvents.length - actionable.length;
  // Most severe first, then chronological, so the top of the list is the thing
  // most worth fixing rather than whatever happened at 0:00.
  const shown = [...(showAll ? analysis.visualEvents : actionable)].sort(
    (a, b) => severityRank[cueSeverity(a.eventType)] - severityRank[cueSeverity(b.eventType)] || a.startTime - b.startTime,
  );
  const critical = actionable.filter((event) => cueSeverity(event.eventType) === "critical").length;
  const caution = actionable.filter((event) => cueSeverity(event.eventType) === "caution").length;
  const framePercent = Math.round(
    (analysis.visualFeatures.filter((row) => row.faceDetected === true).length / Math.max(1, analysis.visualFeatures.length)) * 100,
  );

  return (
    <div className="tab-stack">
      <header>
        <h2>Visual cues</h2>
        <p>
          Observable framing, attention, and movement evidence. These are measurements of what the camera recorded —
          never claims about emotion, confidence, anxiety, honesty, personality, or professionalism.
        </p>
      </header>

      <div className="metric-grid">
        <Stat label="Frames analyzed" value={analysis.visualFeatures.length} note="Sampled from the recording" />
        <Stat label="High priority" value={critical} note="Cues that block a clear view of you" />
        <Stat label="To review" value={caution} note="Attention, framing drift, and movement" />
        <Stat label="Face detected" value={`${framePercent}%`} note="Of sampled frames" />
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Extraction pipeline</h2>
          <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
            deterministic
          </span>
        </div>
        <div className="panel-body">
          <dl className="spec-list" style={{ border: 0 }}>
            <div style={{ background: "transparent" }}>
              <dt>Detection</dt>
              <dd>YOLO ONNX face localization where the local asset is present, MediaPipe face detection otherwise.</dd>
            </div>
            <div style={{ background: "transparent" }}>
              <dt>Measurement</dt>
              <dd>MediaPipe face and pose landmarks produce framing, orientation, and movement features per frame.</dd>
            </div>
            <div style={{ background: "transparent" }}>
              <dt>Cue rules</dt>
              <dd>Deterministic thresholds calibrated against this recording convert features into candidate cues.</dd>
            </div>
            <div style={{ background: "transparent", borderBottom: 0 }}>
              <dt>Temporal logic</dt>
              <dd>A persistence state machine keeps only cues that hold across consecutive samples, then emits timestamped events.</dd>
            </div>
          </dl>
        </div>
      </section>

      <section className="stack-4">
        <div className="row between wrap">
          <h3 style={{ fontSize: 14 }}>Timestamped visual events</h3>
          {ambient > 0 ? (
            <button className="text-button" aria-pressed={showAll} onClick={() => setShowAll((current) => !current)}>
              {showAll ? "Hide continuous cues" : `Show ${ambient} continuous cue${ambient === 1 ? "" : "s"}`}
            </button>
          ) : null}
        </div>
        {shown.length ? (
          <EventList events={shown} seek={seek} severity />
        ) : (
          <Unavailable>
            No framing, attention, stability, or capture-quality cue needed review in this take.
          </Unavailable>
        )}
        {ambient > 0 && !showAll ? (
          <p className="fine-print">
            {ambient} continuous cue{ambient === 1 ? "" : "s"} (mouth, eyebrow, and blink-style movement) are hidden
            because they occur throughout normal speech. They remain in the JSON export.
          </p>
        ) : null}
      </section>

      <TechnicalPanel label="Visual technical details">
        <pre>{JSON.stringify(analysis.visualFeatures.slice(0, 12), null, 2)}</pre>
      </TechnicalPanel>
    </div>
  );
}

/* ------------------------------------------------------------- report render */

/** Minimal renderer for the deterministic report's markdown subset (h1, h2,
 *  bullet lists, ordered lists, paragraphs). No parser dependency. */
function ReportDocument({ markdown }: { markdown: string }) {
  const blocks: ReactNode[] = [];
  let bullets: string[] = [];
  let ordered: string[] = [];

  const flush = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`ul${blocks.length}`}>
          {bullets.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ul>,
      );
      bullets = [];
    }
    if (ordered.length) {
      blocks.push(
        <ol key={`ol${blocks.length}`}>
          {ordered.map((item, index) => (
            <li key={index}>{item}</li>
          ))}
        </ol>,
      );
      ordered = [];
    }
  };

  for (const raw of markdown.split("\n")) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flush();
      continue;
    }
    if (line.startsWith("## ")) {
      flush();
      blocks.push(<h2 key={blocks.length}>{line.slice(3)}</h2>);
    } else if (line.startsWith("# ")) {
      flush();
      blocks.push(<h1 key={blocks.length}>{line.slice(2)}</h1>);
    } else if (line.startsWith("- ")) {
      if (ordered.length) flush();
      bullets.push(line.slice(2));
    } else if (/^\d+\.\s/.test(line)) {
      if (bullets.length) flush();
      ordered.push(line.replace(/^\d+\.\s/, ""));
    } else {
      flush();
      blocks.push(<p key={blocks.length}>{line}</p>);
    }
  }
  flush();
  return <div className="report-doc">{blocks}</div>;
}

/* ------------------------------------------------------------------ payload */

export interface TabPanelProps {
  tab: Tab;
  analysis: LocalAnalysis;
  seek: (time: number) => void;
  onDownloadBundle: () => void;
  onDownloadArtifact: (kind: "report" | "transcript" | "analysis" | "recording") => void;
  onGenerateCoaching: () => void;
  coachingBusy: boolean;
  packaging: boolean;
  onRerun: () => void;
}

export function ResultTabPanel(props: TabPanelProps) {
  const { tab, analysis, seek } = props;
  const scores = asRecord(asRecord(analysis.scores).scores);
  const overall = asRecord(scores.overallInterviewPracticeDelivery);
  const response = analysis.responseAnalysis;
  const metrics = asRecord(response.metrics);
  const audio = analysis.audioFeatures;

  switch (tab) {
    /* ------------------------------------------------------------ Overview */
    case "Overview": {
      const contentEvidenceInsufficient = num(overall.score) === null && String(overall.rating) === "Insufficient evidence";
      const strengths = [
        ...(contentEvidenceInsufficient ? [] : analysis.moments.filter((moment) => moment.classification === "strength").map((moment) => String(moment.explanation))),
        ...(contentEvidenceInsufficient ? [] : dimensions.flatMap(({ key }) => list(asRecord(scores[key]).positiveObservations))),
      ].filter(Boolean);
      const reviews = [
        ...list(overall.practiceAreas),
        ...analysis.moments.filter((moment) => moment.classification === "review").map((moment) => String(moment.coachingRecommendation)),
        ...list(response.practiceAreas),
      ].filter(Boolean);
      const unique = (items: string[]) => [...new Set(items)].slice(0, 4);

      return (
        <div className="tab-stack">
          <section className="score-hero">
            <PracticeScore score={num(overall.score)} rating={String(overall.rating || "Unavailable")} />
            <div className="score-hero-copy">
              <h2>{String(overall.rating || "Unavailable")}</h2>
              {num(overall.score) === null ? (
                <div className="stack-4">
                  <p>
                    TalonCV withheld an overall score because this take does not contain enough evidence to judge fairly.
                  </p>
                  <ul className="reason-list">
                    {list(overall.evidence).map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                </div>
              ) : (
                <p>
                  The headline prioritizes answer quality, then vocal and visual delivery. Recording quality and
                  multimodal alignment remain visible as diagnostics but cannot inflate the practice score.
                </p>
              )}
              <div className="score-hero-meta">
                <TechnicalBadge tone="info">Confidence · {String(overall.confidence || "unavailable")}</TechnicalBadge>
                <TechnicalBadge>Coverage · {String(overall.dataCoverage || "unavailable")}</TechnicalBadge>
                <TechnicalBadge>{analysis.mediaInfo.hasAudio ? "Audio analyzed" : "No audio"}</TechnicalBadge>
                <TechnicalBadge>{analysis.mediaInfo.hasVideo ? "Video analyzed" : "No video"}</TechnicalBadge>
                <TechnicalBadge plain>{String(metrics.wordCount ?? 0)} transcript words</TechnicalBadge>
              </div>
            </div>
          </section>

          <div className="score-grid">
            {dimensions.map(({ key, label }) => {
              const entry = asRecord(scores[key]);
              const value = num(entry.score);
              const strength = list(entry.positiveObservations)[0];
              const practice = list(entry.practiceAreas)[0];
              return (
                <article className="score-card" key={key}>
                  <div className="head">
                    <h3>{label}</h3>
                    {value === null ? <b className="unavailable">Unavailable</b> : <b>{Math.round(value)}</b>}
                  </div>
                  <Meter label={label} value={value} tone={scoreTone(value)} hideHead />
                  <span className="rating">{String(entry.rating || "Unavailable")}</span>
                  <ul>
                    {strength ? (
                      <li className="positive">
                        <span className="glyph" aria-hidden="true">✓</span>
                        <span>{strength}</span>
                      </li>
                    ) : null}
                    {practice ? (
                      <li className="practice">
                        <span className="glyph" aria-hidden="true">↗</span>
                        <span>{practice}</span>
                      </li>
                    ) : null}
                    {!strength && !practice ? (
                      <li>
                        <span className="glyph" aria-hidden="true">·</span>
                        <span>{list(entry.evidence)[0] ?? "No evidence was recorded for this dimension."}</span>
                      </li>
                    ) : null}
                  </ul>
                </article>
              );
            })}
          </div>

          <div className="two-column">
            <section className="stack-4">
              <h2 className="section-title" style={{ fontSize: 16 }}>Strengths</h2>
              <div className="finding-list">
                {unique(strengths).length ? (
                  unique(strengths).map((item) => (
                    <div className="finding strength" key={item}>
                      <span className="glyph" aria-hidden="true">✓</span>
                      <span>{item}</span>
                    </div>
                  ))
                ) : (
                  <Unavailable>{contentEvidenceInsufficient ? "Answer-content evidence was insufficient, so delivery observations are not presented as overall strengths." : "No timestamped strength evidence met the reporting threshold for this take."}</Unavailable>
                )}
              </div>
            </section>
            <section className="stack-4">
              <h2 className="section-title" style={{ fontSize: 16 }}>Priority review areas</h2>
              <div className="finding-list">
                {unique(reviews).length ? (
                  unique(reviews).map((item) => (
                    <div className="finding review" key={item}>
                      <span className="glyph" aria-hidden="true">↗</span>
                      <span>{item}</span>
                    </div>
                  ))
                ) : (
                  <Unavailable>No review priority was produced. Replay the individual tabs for detail.</Unavailable>
                )}
              </div>
            </section>
          </div>

          <section className="stack-4">
            <h2 className="section-title" style={{ fontSize: 16 }}>Evidence timeline</h2>
            <TimelineStrip analysis={analysis} />
          </section>
        </div>
      );
    }

    /* ---------------------------------------------------------- Transcript */
    case "Transcript": {
      const semanticSegments = Array.isArray(analysis.semanticAnalysis.segmentAssessments)
        ? (analysis.semanticAnalysis.segmentAssessments as Array<Record<string, unknown>>)
        : [];
      const fillers = Array.isArray(response.fillerOccurrences)
        ? (response.fillerOccurrences as Array<Record<string, unknown>>)
        : [];

      return (
        <div className="tab-stack">
          <header>
            <h2>Transcript</h2>
            <p>Word-level timestamps produced by browser-local Whisper. Select any timecode to replay that moment.</p>
          </header>

          {!analysis.transcript.segments.length ? (
            <Unavailable>
              {analysis.transcript.warnings?.[0] ??
                "No transcript segments were produced. Record an audible answer to enable transcript coaching."}
            </Unavailable>
          ) : (
            <div className="transcript-list">
              {analysis.transcript.segments.map((segment, index) => {
                const topicMatch = num(semanticSegments.find((item) => num(item.startTime) === segment.start)?.topicMatchScore);
                const segmentFillers = fillers.filter(
                  (filler) => num(filler.startTime) !== null && num(filler.startTime)! >= segment.start && num(filler.startTime)! < segment.end,
                );
                const vocal = analysis.audioEvents.filter(
                  (event) => Math.min(event.endTime, segment.end) > Math.max(event.startTime, segment.start),
                );
                return (
                  <article key={`${segment.start}-${index}`}>
                    <button className="timestamp-button" onClick={() => seek(segment.start)}>
                      {timecode(segment.start)}
                    </button>
                    <div className="seg-body">
                      <p>{segment.text}</p>
                      <div className="seg-marks">
                        {topicMatch !== null ? (
                          <TechnicalBadge tone={topicMatch >= 70 ? "success" : topicMatch >= 45 ? "neutral" : "warning"} plain>
                            topic match {topicMatch}/100
                          </TechnicalBadge>
                        ) : null}
                        {segmentFillers.length ? (
                          <TechnicalBadge tone="warning" plain>
                            {segmentFillers.length} filler{segmentFillers.length === 1 ? "" : "s"}
                          </TechnicalBadge>
                        ) : null}
                        {vocal.slice(0, 3).map((event, position) => (
                          <TechnicalBadge key={position} tone="info" plain>
                            {titleize(event.eventType)}
                          </TechnicalBadge>
                        ))}
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}

          <TechnicalPanel label="Transcript technical details">
            <pre>
              {JSON.stringify(
                {
                  language: analysis.transcript.language ?? null,
                  segments: analysis.transcript.segments.length,
                  averageConfidence: analysis.transcript.averageConfidence,
                  model: analysis.modelVersions.speech ?? null,
                  warnings: analysis.transcript.warnings ?? [],
                },
                null,
                2,
              )}
            </pre>
          </TechnicalPanel>
        </div>
      );
    }

    /* ------------------------------------------------------ Answer Quality */
    case "Answer Quality": {
      const rubric = asRecord(response.rubric);
      const star = num(asRecord(response.starAnalysis).componentsPresent);
      const development = asRecord(response.answerDevelopment);
      const relevance = num(analysis.semanticAnalysis.questionRelevance);
      const roleRelevance = num(analysis.semanticAnalysis.roleContextRelevance);
      const strongPhrases = Array.isArray(response.strongPhrases)
        ? (response.strongPhrases as Array<Record<string, unknown>>)
        : [];

      if (!response.available) {
        return (
          <div className="tab-stack">
            <header>
              <h2>Answer quality</h2>
              <p>Deterministic response structure with optional semantic relevance.</p>
            </header>
            <Unavailable>No transcript was available, so deterministic answer analysis could not run.</Unavailable>
          </div>
        );
      }

      return (
        <div className="tab-stack">
          <header>
            <h2>Answer quality</h2>
            <p>Deterministic answer-structure rubric, with MiniLM relevance where a transcript was available.</p>
          </header>

          <div className="metric-grid">
            <Stat label="Words" value={String(metrics.wordCount ?? 0)} note={String(development.lengthAssessment ?? "")} />
            <Stat
              label="Filler rate"
              value={`${String(metrics.fillerRatePer100Words ?? 0)}`}
              note={`per 100 words · ${String(metrics.fillerCount ?? 0)} total`}
            />
            <Stat label="STAR elements" value={`${star ?? 0}/4`} note="Situation, action, result, conclusion markers" />
            <Stat
              label="Question relevance"
              value={num(asRecord(response.relevanceAnalysis).score) === null ? "—" : Math.round(num(asRecord(response.relevanceAnalysis).score)!)}
              note={relevance === null ? "MiniLM semantic match" : `MiniLM cosine ${relevance.toFixed(2)} → 0–100`}
            />
          </div>

          <section className="panel">
            <div className="panel-header">
              <h2>Deterministic rubric</h2>
              <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                browser-response-v2
              </span>
            </div>
            <div className="panel-body stack-5">
              {["relevance", "substance", "structure", "clarity", "specificity", "conciseness"].filter((key) => rubric[key]).map((key) => {
                const entry = asRecord(rubric[key]);
                const value = num(entry.score);
                return (
                  <Meter
                    key={key}
                    label={titleize(key)}
                    value={value}
                    display={value === null ? "—" : `${Math.round(value)}/100`}
                    caption={String(entry.rating ?? "")}
                    tone={scoreTone(value)}
                  />
                );
              })}
            </div>
          </section>

          <div className="two-column">
            <article className="evidence-card">
              <h3>Answer development</h3>
              <p>{String(development.exampleQuality ?? "")}</p>
              <p>{String(development.resultQuality ?? "")}</p>
              {roleRelevance !== null ? (
                <p>Role-context relevance · {roleRelevance.toFixed(2)} cosine similarity.</p>
              ) : null}
            </article>
            <article className="evidence-card">
              <h3>Suggested structure</h3>
              <ul style={{ margin: 0, paddingLeft: 18, display: "grid", gap: 6 }}>
                {list(response.suggestedAnswerStructure).map((item) => (
                  <li key={item} style={{ fontSize: 13, lineHeight: 1.55, color: "var(--talon-text-secondary)" }}>
                    {item}
                  </li>
                ))}
              </ul>
            </article>
          </div>

          {strongPhrases.length ? (
            <section className="stack-4">
              <h3 style={{ fontSize: 14 }}>Evidence from the transcript</h3>
              <div className="evidence-list">
                {strongPhrases.map((phrase, index) => (
                  <article key={index}>
                    <button className="timestamp-button" onClick={() => seek(Number(phrase.startTime))}>
                      {timecode(phrase.startTime)}
                    </button>
                    <div className="ev-body">
                      <strong>Specific action or result</strong>
                      <p>{String(phrase.text ?? "")}</p>
                      <div className="ev-meta">
                        {list(phrase.reasons).map((reason) => (
                          <TechnicalBadge key={reason} plain>
                            {reason}
                          </TechnicalBadge>
                        ))}
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ) : null}

          <TechnicalPanel label="Answer quality technical details">
            <pre>{JSON.stringify(response, null, 2)}</pre>
          </TechnicalPanel>
        </div>
      );
    }

    /* ------------------------------------------------------ Vocal Delivery */
    case "Vocal Delivery": {
      if (audio.available !== true) {
        return (
          <div className="tab-stack">
            <header>
              <h2>Vocal delivery</h2>
              <p>Signal measurements taken from the decoded waveform.</p>
            </header>
            <Unavailable>{list(audio.warnings)[0] ?? "No audible waveform was decoded from this recording."}</Unavailable>
          </div>
        );
      }
      const pace = num(audio.speechRateWpm);
      const speechRatio = num(audio.speechRatio);
      const silenceRatio = num(audio.silenceRatio);
      const rms = num(audio.overallRmsDb);
      const energy = num(audio.energyVariationDb);
      const snr = num(audio.snrProxyDb);
      const clipping = num(audio.clippingPercentage);
      const pauses = analysis.audioEvents.filter((event) => event.eventType === "longPause");

      return (
        <div className="tab-stack">
          <header>
            <h2>Vocal delivery</h2>
            <p>Frame-level signal analysis from the Web Audio decode, paired with timestamped vocal events.</p>
          </header>

          <div className="metric-grid">
            <Stat label="Speaking pace" value={pace === null ? "—" : Math.round(pace)} note="words per minute" />
            <Stat label="Speaking time" value={speechRatio === null ? "—" : `${Math.round(speechRatio * 100)}%`} note="of total duration" />
            <Stat label="Silence ratio" value={silenceRatio === null ? "—" : `${Math.round(silenceRatio * 100)}%`} note="below the speech threshold" />
            <Stat label="Long pauses" value={pauses.length} note="sustained low-energy gaps" />
          </div>

          <div className="two-column">
            <section className="panel">
              <div className="panel-header">
                <h2>Level and variation</h2>
              </div>
              <div className="panel-body stack-5">
                <Meter
                  label="Overall level"
                  value={rms === null ? null : Math.max(0, Math.min(100, ((rms + 60) / 60) * 100))}
                  display={rms === null ? "—" : `${rms.toFixed(1)} dBFS`}
                  caption="Comfortable practice range sits near −24 to −12 dBFS"
                  tone="info"
                />
                <Meter
                  label="Vocal variation"
                  value={energy === null ? null : Math.max(0, Math.min(100, (energy / 24) * 100))}
                  display={energy === null ? "—" : `${energy.toFixed(1)} dB`}
                  caption="Range between quiet and emphatic speech frames"
                  tone="success"
                />
                <Meter
                  label="Speech-to-noise proxy"
                  value={snr === null ? null : Math.max(0, Math.min(100, (snr / 40) * 100))}
                  display={snr === null ? "—" : `${snr.toFixed(1)} dB`}
                  caption="Separation between voiced and background frames"
                  tone={snr !== null && snr < 10 ? "warning" : "info"}
                />
                <Meter
                  label="Clipping"
                  value={clipping === null ? null : Math.min(100, clipping * 20)}
                  display={clipping === null ? "—" : `${clipping.toFixed(3)}%`}
                  caption="Samples that reached the digital ceiling"
                  tone={clipping !== null && clipping > 0.1 ? "warning" : "success"}
                />
              </div>
            </section>

            <section className="panel">
              <div className="panel-header">
                <h2>Pause distribution</h2>
                <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                  {pauses.length} detected
                </span>
              </div>
              <div className="panel-body">
                {pauses.length ? (
                  <div className="stack-4">
                    {pauses.slice(0, 8).map((pause, index) => (
                      <div className="meter" key={index}>
                        <div className="meter-head">
                          <span>
                            <button className="timestamp-button" onClick={() => seek(pause.startTime)}>
                              {timecode(pause.startTime)}
                            </button>
                          </span>
                          <strong>{pause.durationSeconds.toFixed(2)}s</strong>
                        </div>
                        <div className="meter-track">
                          <span
                            className="meter-fill warning"
                            style={{ width: `${Math.min(100, (pause.durationSeconds / 4) * 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <Unavailable>No sustained pause exceeded the detection threshold.</Unavailable>
                )}
              </div>
            </section>
          </div>

          <section className="stack-4">
            <h3 style={{ fontSize: 14 }}>Timestamped vocal events</h3>
            <EventList
              events={[...analysis.audioEvents].sort(
                (a, b) => severityRank[cueSeverity(a.eventType)] - severityRank[cueSeverity(b.eventType)] || a.startTime - b.startTime,
              )}
              seek={seek}
              severity
            />
          </section>

          <TechnicalPanel label="Vocal delivery technical details">
            <pre>{JSON.stringify(audio, null, 2)}</pre>
          </TechnicalPanel>
        </div>
      );
    }

    /* --------------------------------------------------------- Visual Cues */
    case "Visual Cues":
      return <VisualCuesPanel analysis={analysis} seek={seek} />;

    /* -------------------------------------------------- Multimodal Moments */
    case "Multimodal Moments": {
      return (
        <div className="tab-stack">
          <header>
            <h2>Multimodal moments</h2>
            <p>
              Where transcript, vocal, and visual evidence overlap on the same timestamp. Each moment is a replay prompt,
              not a statement about your internal state.
            </p>
          </header>
          <TimelineStrip analysis={analysis} />
          <Moments analysis={analysis} seek={seek} />
        </div>
      );
    }

    /* --------------------------------------------------------- Full Report */
    case "Full Report": {
      const coaching = analysis.localCoaching;
      return (
        <div className="tab-stack">
          <header>
            <h2>Full deterministic report</h2>
            <p>Generated from the evidence above. It stays complete whether or not the optional local model runs.</p>
          </header>

          <ReportDocument markdown={analysis.report} />

          <section className="coaching-block">
            <div className="row between">
              <h3 style={{ fontSize: 14 }}>Optional local coaching narrative</h3>
              <TechnicalBadge tone={coaching?.available ? "success" : "neutral"} dot>
                {coaching?.available ? "Generated" : "Not generated"}
              </TechnicalBadge>
            </div>
            {coaching?.available && coaching.text ? (
              <pre>{coaching.text}</pre>
            ) : (
              <>
                <p className="fine-print">
                  {coaching?.warnings?.[0] ??
                    "A small instruction model can rephrase the deterministic evidence in narrative form. It downloads once, runs in this browser, and is never required."}
                </p>
                <div className="button-row">
                  <button className="button secondary" disabled={props.coachingBusy} onClick={props.onGenerateCoaching}>
                    {props.coachingBusy ? "Generating locally…" : "Generate locally"}
                  </button>
                </div>
              </>
            )}
          </section>

          <TechnicalPanel label="Raw report markdown">
            <pre className="report-text">{analysis.report}</pre>
          </TechnicalPanel>
        </div>
      );
    }

    /* -------------------------------------------------------------- Export */
    case "Export": {
      const artifacts: Array<{ id: "report" | "transcript" | "analysis" | "recording"; title: string; note: string; enabled: boolean }> = [
        { id: "report", title: "Deterministic report", note: "report.md · Markdown coaching document", enabled: true },
        { id: "transcript", title: "Transcript", note: "transcript.txt · plain text", enabled: Boolean(analysis.transcript.text) },
        { id: "analysis", title: "JSON evidence", note: "analysis.json · every measurement, event, and score", enabled: true },
        { id: "recording", title: "Original recording", note: "The media file stored in this browser", enabled: analysis.mediaInfo.durationSeconds > 0 },
      ];

      return (
        <div className="tab-stack">
          <header>
            <h2>Export and local data</h2>
            <p>Every artifact is produced in this browser. Nothing is uploaded to create a download.</p>
          </header>

          <div className="download-list">
            <div>
              <div className="dl-copy">
                <strong>Complete TalonCV session bundle</strong>
                <span>Recording, report, transcript, evidence JSON, scores, and session context</span>
              </div>
              <button className="button primary small" disabled={props.packaging} onClick={props.onDownloadBundle}>
                {props.packaging ? "Packaging…" : "Download ZIP"}
              </button>
            </div>

            {artifacts.map((artifact) => (
              <Fragment key={artifact.id}>
                <div>
                  <div className="dl-copy">
                    <strong>{artifact.title}</strong>
                    <span>{artifact.note}</span>
                  </div>
                  <button
                    className="button ghost small"
                    disabled={!artifact.enabled}
                    onClick={() => props.onDownloadArtifact(artifact.id)}
                  >
                    Download
                  </button>
                </div>
              </Fragment>
            ))}

            <div>
              <div className="dl-copy">
                <strong>Run again</strong>
                <span>Replaces the stored analysis with a fresh local result</span>
              </div>
              <button className="button ghost small" onClick={props.onRerun}>
                Reanalyze
              </button>
            </div>
          </div>

          <div className="notice success">
            <span aria-hidden="true">✓</span>
            <span>
              Local-only export. TalonCV has no cloud storage, so bundles exist only where you save them. Re-import a
              bundle from the History page to restore a session in any browser.
            </span>
          </div>

          <TechnicalPanel label="Provenance">
            <pre>
              {JSON.stringify(
                {
                  analysisVersion: analysis.analysisVersion,
                  createdAt: analysis.createdAt,
                  models: analysis.modelVersions,
                  mediaInfo: analysis.mediaInfo,
                  warnings: analysis.warnings,
                },
                null,
                2,
              )}
            </pre>
          </TechnicalPanel>
        </div>
      );
    }

    default:
      return null;
  }
}
