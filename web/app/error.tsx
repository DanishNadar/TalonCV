"use client";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section className="shell centered-state">
      <span className="eyebrow">Runtime error</span>
      <h1>That view could not be loaded</h1>
      <p>Your locally stored interview data was not changed. Retry the view, or return to your practice archive.</p>
      <div className="button-row">
        <button className="button primary" onClick={reset}>
          Try again
        </button>
      </div>
    </section>
  );
}
