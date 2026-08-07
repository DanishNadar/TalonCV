import Link from "next/link";

export default function HomePage() {
  return <>
    <section className="hero shell">
      <div className="eyebrow">Private multimodal interview practice</div>
      <h1>Review what you said, how you delivered it, and where the evidence lines up.</h1>
      <p className="hero-copy">TalonCV combines transcript, vocal, visual, and timestamp-aligned evidence into practical coaching directly in your browser. Your interview never uploads to a TalonCV service.</p>
      <div className="button-row"><Link className="button primary" href="/interview/new">Start a practice interview</Link><Link className="button ghost" href="/dashboard">Open interview history</Link></div>
    </section>
    <section className="feature-band"><div className="shell feature-grid">
      <article><span>01</span><h2>Record privately</h2><p>Capture camera and microphone in your browser, then keep the take in browser-local storage.</p></article>
      <article><span>02</span><h2>Analyze locally</h2><p>Whisper, MiniLM, YOLO/MediaPipe cues, scoring, and optional coaching run in browser workers.</p></article>
      <article><span>03</span><h2>Replay the evidence</h2><p>Jump to transcript, vocal, visual, and combined coaching moments in the original recording.</p></article>
    </div></section>
    <section id="privacy" className="shell access-section"><div><div className="eyebrow">Privacy by architecture</div><h2>Practice without an account or upload.</h2></div><div className="setup-card"><h2>What stays local</h2><p>Your camera feed, recording, transcript, analysis, and exported report stay in this browser&apos;s local storage. Initial model files are public static downloads and can be cached for later offline reuse.</p><Link className="button primary" href="/interview/new">Set up a local practice</Link></div></section>
  </>;
}
