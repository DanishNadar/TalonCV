"use client";

export default function GlobalError({ reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <html lang="en"><body><main className="shell centered-state"><h1>TalonCV needs a refresh.</h1><button onClick={reset}>Reload application</button></main></body></html>
  );
}
