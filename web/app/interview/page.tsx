"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { LocalAnalysisWorkspace } from "@/components/LocalAnalysisWorkspace";

export default function LocalInterviewPage() {
  return <Suspense fallback={<section className="shell centered-state"><p>Opening local review…</p></section>}><LocalInterviewContent /></Suspense>;
}

function LocalInterviewContent() {
  const sessionId = useSearchParams().get("id");
  if (!sessionId) return <section className="shell centered-state"><h1>Choose a local interview</h1><p>Open a recording from your browser-local interview history.</p></section>;
  return <LocalAnalysisWorkspace sessionId={sessionId} />;
}
