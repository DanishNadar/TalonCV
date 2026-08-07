import { TechnicalBadge } from "@/components/ui/primitives";

const streams = [
  {
    name: "Voice",
    detail: "Whisper transcription · Web Audio delivery metrics",
    glyph: (
      <span className="stream-glyph" aria-hidden="true">
        {[0, 1, 2, 3, 4, 5, 6, 7, 8].map((bar) => (
          <i key={bar} className="bar" style={{ animationDelay: `${bar * 0.11}s` }} />
        ))}
      </span>
    ),
  },
  {
    name: "Language",
    detail: "MiniLM semantic relevance · answer structure",
    glyph: (
      <span className="stream-glyph nodes" aria-hidden="true">
        {[0, 1, 2, 3, 4, 5, 6].map((node) => (
          <i key={node} style={{ animationDelay: `${node * 0.32}s` }} />
        ))}
      </span>
    ),
  },
  {
    name: "Vision",
    detail: "YOLO ONNX + MediaPipe · deterministic cue rules",
    glyph: (
      <span className="stream-glyph frame" aria-hidden="true">
        <b />
        <b />
        <b />
        <b />
        <em />
      </span>
    ),
  },
];

/** The hero's system diagram: three modality streams converging on the
 *  multimodal engine, then on the coaching report. CSS-only animation so the
 *  inference workers keep the CPU. */
export function PipelineDiagram() {
  return (
    <div className="pipeline-viz">
      <div className="viz-head">
        <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
          Analysis pipeline
        </span>
        <TechnicalBadge tone="success" dot>
          On-device
        </TechnicalBadge>
      </div>

      <div className="viz-streams">
        {streams.map((stream) => (
          <div className="viz-stream" key={stream.name}>
            <span className="stream-name">{stream.name}</span>
            {stream.glyph}
            <span className="stream-detail">{stream.detail}</span>
          </div>
        ))}
      </div>

      <div className="viz-flow">
        <span className="viz-rails" aria-hidden="true">
          <svg viewBox="0 0 300 26" preserveAspectRatio="none">
            <path d="M50 0 V10 Q50 18 150 18" />
            <path d="M150 0 V18" />
            <path d="M250 0 V10 Q250 18 150 18" />
            <path className="flow" d="M50 0 V10 Q50 18 150 18" />
            <path className="flow" d="M150 0 V18" style={{ animationDelay: "0.5s" }} />
            <path className="flow" d="M250 0 V10 Q250 18 150 18" style={{ animationDelay: "1s" }} />
          </svg>
        </span>

        <div className="viz-node">
          <span className="pulse" aria-hidden="true" />
          Multimodal alignment engine
        </div>

        <span className="viz-rails" style={{ height: 18 }} aria-hidden="true">
          <svg viewBox="0 0 300 18" preserveAspectRatio="none">
            <path d="M150 0 V18" />
            <path className="flow" d="M150 0 V18" />
          </svg>
        </span>

        <div className="viz-node terminal">Explainable coaching report</div>
      </div>

      <div className="viz-foot">
        <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
          Runtime · WASM / WebGPU
        </span>
        <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
          Egress · 0 bytes
        </span>
      </div>
    </div>
  );
}
