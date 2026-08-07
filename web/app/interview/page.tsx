"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { LocalAnalysisWorkspace } from "@/components/LocalAnalysisWorkspace";

export default function LocalInterviewPage() {
  return (
    <Suspense
      fallback={
        <section className="shell centered-state">
          <span className="spinner" aria-hidden="true" />
          <p>Opening local review…</p>
        </section>
      }
    >
      <LocalInterviewContent />
    </Suspense>
  );
}

function LocalInterviewContent() {
  const params = useSearchParams();
  const sessionId = params.get("id");
  if (!sessionId) {
    return (
      <section className="shell centered-state">
        <h1>Choose a local interview</h1>
        <p>Open a recording from your browser-local interview history.</p>
        <Link className="button primary" href="/dashboard">
          Open history
        </Link>
      </section>
    );
  }
  return <LocalAnalysisWorkspace sessionId={sessionId} autoRun={params.get("rerun") === "1"} />;
}
