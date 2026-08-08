import Image from "next/image";
import type { ReactNode } from "react";

/* ---------------------------------------------------------------- TalonLogo */

/** The TalonCV falcon mark. Supplied as scarlet artwork on black and converted
 *  to an alpha matte, so it sits on any surface without a visible plate. */
export function TalonLogo({ size = 34 }: { size?: number }) {
  return (
    <span className="brand-mark" style={{ width: size, height: size }} aria-hidden="true">
      <Image src="/taloncv-mark.png" alt="" width={size} height={size} priority />
    </span>
  );
}

export function TalonWordmark({ subtitle = "AI Interview Practice Lab" }: { subtitle?: string | null }) {
  return (
    <span className="brand-text">
      <span className="brand-name">TalonCV</span>
      {subtitle ? <span className="brand-sub">{subtitle}</span> : null}
    </span>
  );
}

/* ------------------------------------------------------------ StatusDot etc */

export type Tone = "neutral" | "accent" | "success" | "warning" | "error" | "info";

export function StatusDot({ tone = "neutral", live = false }: { tone?: Tone; live?: boolean }) {
  return <i className={`dot ${tone === "neutral" ? "" : tone} ${live ? "live" : ""}`} aria-hidden="true" />;
}

export function TechnicalBadge({
  children,
  tone = "neutral",
  dot = false,
  plain = false,
}: {
  children: ReactNode;
  tone?: Tone;
  dot?: boolean;
  plain?: boolean;
}) {
  return (
    <span className={`tech-badge ${tone === "neutral" ? "" : tone} ${plain ? "plain" : ""}`}>
      {dot ? <StatusDot tone={tone} /> : null}
      {children}
    </span>
  );
}

/* ------------------------------------------------------------ LocalAIStatus */

/** The privacy architecture made visible. Appears on the header, the recorder,
 *  and the analysis workspace. */
export function LocalAIStatus({
  variant = "default",
  label = "Local AI",
  detail = "Analysis runs on this device",
}: {
  variant?: "default" | "compact" | "block";
  label?: string;
  detail?: string;
}) {
  return (
    <span className={`local-ai ${variant === "default" ? "" : variant}`} title={detail}>
      <StatusDot tone="success" live />
      <span className="local-ai-copy">
        <strong>{label}</strong>
        <span>{detail}</span>
      </span>
    </span>
  );
}

/* ------------------------------------------------------------- SectionHeader */

export function SectionHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="section-header">
      <div className="stack">
        {eyebrow ? <span className="eyebrow">{eyebrow}</span> : null}
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="row">{actions}</div> : null}
    </div>
  );
}

/* -------------------------------------------------------------------- Panel */

export function Panel({
  children,
  header,
  meta,
  accentTop = false,
  padded = true,
  className = "",
}: {
  children: ReactNode;
  header?: ReactNode;
  meta?: ReactNode;
  accentTop?: boolean;
  padded?: boolean;
  className?: string;
}) {
  return (
    <section className={`panel ${accentTop ? "accent-top" : ""} ${className}`}>
      {header ? (
        <div className="panel-header">
          <h2>{header}</h2>
          {meta}
        </div>
      ) : null}
      <div className={padded ? "panel-body" : ""}>{children}</div>
    </section>
  );
}

export function TechnicalPanel({ label, children }: { label: string; children: ReactNode }) {
  return (
    <details className="technical">
      <summary>{label}</summary>
      <div className="technical-body">{children}</div>
    </details>
  );
}

/* --------------------------------------------------------------- MetricCard */

export function MetricCard({
  label,
  value,
  unit,
  note,
}: {
  label: string;
  value: string | number;
  unit?: string;
  note?: string;
}) {
  return (
    <article className="metric-card">
      <span className="metric-label">{label}</span>
      <span className="metric-value">
        {value}
        {unit ? <span className="unit">{unit}</span> : null}
      </span>
      {note ? <span className="metric-note">{note}</span> : null}
    </article>
  );
}

/* -------------------------------------------------------------------- Meter */

export function Meter({
  label,
  value,
  display,
  caption,
  tone = "accent",
  hideHead = false,
}: {
  label: string;
  value: number | null;
  display?: string;
  caption?: string;
  tone?: "accent" | "success" | "warning" | "info" | "neutral";
  /** Suppresses the label row when the surrounding card already states it. */
  hideHead?: boolean;
}) {
  const pct = value === null ? 0 : Math.max(0, Math.min(100, value));
  return (
    <div className="meter">
      {hideHead ? null : (
        <div className="meter-head">
          <span>{label}</span>
          <strong>{display ?? (value === null ? "—" : `${Math.round(pct)}`)}</strong>
        </div>
      )}
      <div
        className="meter-track"
        role="meter"
        aria-label={label}
        aria-valuenow={value === null ? undefined : Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuetext={display ?? (value === null ? "Unavailable" : `${Math.round(pct)} of 100`)}
      >
        <span className={`meter-fill ${value === null ? "neutral" : tone}`} style={{ width: `${value === null ? 0 : pct}%` }} />
      </div>
      {caption ? <span className="meter-caption">{caption}</span> : null}
    </div>
  );
}

/* --------------------------------------------------------------- EmptyState */

export function EmptyState({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children?: ReactNode;
}) {
  return (
    <section className="empty-state">
      <h2>{title}</h2>
      {description ? <p>{description}</p> : null}
      {children ? <div className="button-row">{children}</div> : null}
    </section>
  );
}

/* ----------------------------------------------------------------- StepRail */

export interface RailStep { id: string; name: string }

export function StepRail({ steps, current }: { steps: RailStep[]; current: string }) {
  const index = steps.findIndex((step) => step.id === current);
  return (
    <nav className="step-rail" aria-label="Session progress">
      {steps.map((step, position) => {
        const state = position < index ? "done" : position === index ? "active" : "pending";
        return (
          <div key={step.id} className="step" data-state={state} aria-current={state === "active" ? "step" : undefined}>
            <span className="idx">{String(position + 1).padStart(2, "0")}</span>
            <span className="name">{step.name}</span>
            <span className="visually-hidden">{state === "done" ? "completed" : state === "active" ? "current step" : "not started"}</span>
          </div>
        );
      })}
    </nav>
  );
}

/* ------------------------------------------------------------- PracticeScore */

export function PracticeScore({ score, rating }: { score: number | null; rating: string }) {
  const radius = 58;
  const circumference = 2 * Math.PI * radius;
  const offset = score === null ? circumference : circumference * (1 - Math.max(0, Math.min(100, score)) / 100);
  // The arc colour tracks the band so a weak take never reads as a strong one.
  const tone = score === null ? "unscored" : score >= 74 ? "good" : score >= 60 ? "mixed" : "weak";
  return (
    <div
      className={`score-dial tone-${tone}`}
      role="img"
      aria-label={`Practice score ${score === null ? "not scored" : `${Math.round(score)} out of 100`}, ${rating}`}
    >
      <svg viewBox="0 0 132 132" aria-hidden="true">
        <circle className="track" cx="66" cy="66" r={radius} />
        <circle className="value" cx="66" cy="66" r={radius} strokeDasharray={circumference} strokeDashoffset={offset} />
      </svg>
      <span className="readout">
        <b>{score === null ? "—" : Math.round(score)}</b>
        <span>{score === null ? "Not scored" : "Practice score"}</span>
      </span>
    </div>
  );
}
