import Link from "next/link";

export default function NotFound() {
  return (
    <section className="shell centered-state">
      <span className="eyebrow">404</span>
      <h1>That page is not part of TalonCV</h1>
      <p>The route may have changed, or the local session it pointed to was deleted from this browser.</p>
      <div className="button-row">
        <Link className="button primary" href="/dashboard">
          Open history
        </Link>
        <Link className="button ghost" href="/">
          Back to home
        </Link>
      </div>
    </section>
  );
}
