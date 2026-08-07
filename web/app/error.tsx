"use client";

export default function ErrorPage({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <section className="shell centered-state">
      <div className="status-icon error">!</div>
      <h1>That view could not be loaded.</h1>
      <p>Your interview data was not changed. Try the request again.</p>
      <button className="button primary" onClick={reset}>Try again</button>
    </section>
  );
}
