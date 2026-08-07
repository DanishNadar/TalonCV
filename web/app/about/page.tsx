import Link from "next/link";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { TechnicalBadge } from "@/components/ui/primitives";

export const metadata = { title: "About" };

const guarantees = [
  ["Railway", "Not used"],
  ["Supabase", "Not used"],
  ["Database", "None"],
  ["Authentication", "None"],
  ["Docker", "Not required"],
  ["External inference API", "None"],
  ["Required secrets", "None"],
  ["Interview upload", "Never by default"],
];

const notClaims = [
  "Hiring or employability decisions",
  "Personality or trait assessment",
  "Emotion, anxiety, or confidence inference",
  "Honesty or deception detection",
  "Psychological or clinical evaluation",
];

export default function AboutPage() {
  return (
    <div className="shell page-stack">
      <header className="section-header">
        <div className="stack">
          <span className="eyebrow">About TalonCV</span>
          <h1 className="section-title">AI-powered interview practice and multimodal coaching</h1>
          <p>
            TalonCV is a browser-local research platform for interview practice. It aligns speech, vocal delivery,
            semantic relevance, and visible delivery cues on a single timeline, then explains each score from the
            evidence that produced it.
          </p>
        </div>
      </header>

      <div className="two-column">
        <PipelineDiagram />
        <section className="panel">
          <div className="panel-header">
            <h2>Architecture guarantees</h2>
            <TechnicalBadge tone="success" dot>
              Verified in CI
            </TechnicalBadge>
          </div>
          <div className="readiness">
            {guarantees.map(([label, value]) => (
              <div className="readiness-row" key={label}>
                <span />
                <span className="label">{label}</span>
                <span className="value">{value}</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <section className="panel">
        <div className="panel-header">
          <h2>Scope and limitations</h2>
          <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
            responsible use
          </span>
        </div>
        <div className="panel-body stack-5">
          <p className="lede">
            TalonCV describes observable properties of a recording — what was said, how loud and how quickly it was
            said, and how the speaker was framed and oriented on camera. Every cue is a prompt to replay a timestamp,
            never a conclusion about a person.
          </p>
          <div className="stack-4">
            <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
              TalonCV does not perform
            </span>
            <div className="finding-list">
              {notClaims.map((claim) => (
                <div className="finding review" key={claim}>
                  <span className="glyph" aria-hidden="true">✕</span>
                  <span>{claim}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="notice info">
            <span aria-hidden="true">◆</span>
            <span>
              Visual cues come from deterministic rules over YOLO and MediaPipe measurements with temporal persistence
              logic. They describe framing, orientation, and movement only.
            </span>
          </div>
        </div>
      </section>

      <section className="panel">
        <div className="panel-header">
          <h2>Project context</h2>
        </div>
        <div className="panel-body stack-4">
          <p className="lede">
            TalonCV was developed at Illinois Institute of Technology as an applied AI engineering project. The
            production application is a static export: the host serves application code and public model files, and
            receives no interview media, transcript, evidence, score, or report.
          </p>
          <div className="button-row">
            <Link className="button primary" href="/interview/new">
              Start Interview
            </Link>
            <Link className="button ghost" href="/models">
              Inspect local models
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
