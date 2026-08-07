import Link from "next/link";
import { PipelineDiagram } from "@/components/PipelineDiagram";
import { LocalAIStatus, TechnicalBadge } from "@/components/ui/primitives";

const capabilities = [
  {
    index: "01",
    title: "Record or import locally",
    body: "Capture camera and microphone in the browser, or bring an existing take. Media is written straight to IndexedDB on this device.",
  },
  {
    index: "02",
    title: "Analyze on-device",
    body: "Whisper, MiniLM, YOLO ONNX, and MediaPipe run inside dedicated Web Workers. Deterministic cue rules and scoring run alongside them.",
  },
  {
    index: "03",
    title: "Replay the evidence",
    body: "Every score traces back to a timestamp you can jump to — transcript segments, vocal events, visual cues, and aligned moments.",
  },
];

const specs = [
  ["Speech", "Whisper Tiny (English) via ONNX Runtime Web, quantized, with word-level timestamps."],
  ["Vocal delivery", "Web Audio decoding with frame-level energy, pause, clipping, and speech-rate measurement."],
  ["Language", "MiniLM sentence embeddings for question and role relevance, plus deterministic answer-structure analysis."],
  ["Vision", "YOLO ONNX face localization where the asset is present, MediaPipe face and pose landmarks otherwise."],
  ["Cue extraction", "Deterministic feature rules with temporal persistence logic that emits timestamped visual events."],
  ["Alignment", "Timestamp overlap between transcript, vocal events, and visual events produces replayable moments."],
  ["Scoring", "Explainable weighted rubric. Unavailable modalities are excluded and remaining weights renormalized."],
  ["Persistence", "IndexedDB for sessions, media, and artifacts. Cache Storage for public model files only."],
];

export default function HomePage() {
  return (
    <>
      <section className="hero shell">
        <div className="hero-copy">
          <span className="wordmark">
            <span className="rule" aria-hidden="true" />
            TalonCV
          </span>
          <h1>
            <span className="line">Practice interviews.</span>
            <span className="line">Understand your delivery.</span>
            <span className="line">
              Improve with <span className="accent">multimodal AI.</span>
            </span>
          </h1>
          <p>
            TalonCV analyzes your response, vocal delivery, semantic relevance, and visible presentation cues directly in
            your browser. The recording, transcript, analysis, and report never leave this device.
          </p>
          <div className="button-row">
            <Link className="button primary large" href="/interview/new">
              Start Interview
            </Link>
            <Link className="button ghost large" href="/dashboard">
              View Local History
            </Link>
          </div>
          <div className="hero-meta">
            <TechnicalBadge>No account</TechnicalBadge>
            <TechnicalBadge>No upload</TechnicalBadge>
            <TechnicalBadge>No API keys</TechnicalBadge>
            <TechnicalBadge tone="accent">Browser inference</TechnicalBadge>
          </div>
        </div>
        <PipelineDiagram />
      </section>

      <section className="band">
        <div className="shell band-inner stack-6">
          <div className="section-header">
            <div className="stack">
              <span className="eyebrow">How a session runs</span>
              <h2 className="section-title">Three modalities, one aligned timeline</h2>
            </div>
            <LocalAIStatus detail="No interview data leaves this device" />
          </div>
          <div className="capability-grid">
            {capabilities.map((item) => (
              <article key={item.index}>
                <span className="index">{item.index}</span>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section id="architecture" className="shell band-inner stack-6">
        <div className="section-header">
          <div className="stack">
            <span className="eyebrow">System specification</span>
            <h2 className="section-title">What actually runs in the tab</h2>
            <p>
              TalonCV is a static application. There is no backend inference server, no database, and no external
              inference API in the analysis path.
            </p>
          </div>
        </div>
        <dl className="spec-list">
          {specs.map(([term, description]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{description}</dd>
            </div>
          ))}
        </dl>
        <div className="notice info">
          <span aria-hidden="true">◆</span>
          <span>
            TalonCV is AI-powered interview practice and multimodal coaching. It does not assess personality, emotion,
            honesty, intelligence, or employability, and it produces no hiring recommendation.
          </span>
        </div>
      </section>
    </>
  );
}
