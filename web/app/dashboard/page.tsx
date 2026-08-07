"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";
import { LocalInterviewHistory } from "@/components/LocalInterviewHistory";
import { LocalAIStatus } from "@/components/ui/primitives";

function greeting(hour: number): string {
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

export default function DashboardPage() {
  // The static export is prerendered at build time, so the time of day can only
  // be resolved on the client. A server snapshot keeps hydration consistent.
  const salutation = useSyncExternalStore(
    () => () => undefined,
    () => greeting(new Date().getHours()),
    () => "Welcome back",
  );

  return (
    <div className="shell page-stack">
      <header className="section-header">
        <div className="stack">
          <span className="eyebrow">Practice archive</span>
          <h1 className="section-title">{salutation}</h1>
          <p>Ready for another interview practice session?</p>
        </div>
        <div className="row">
          <LocalAIStatus detail="Stored in this browser only" />
          <Link className="button primary" href="/interview/new">
            <span aria-hidden="true">+</span> New Interview
          </Link>
        </div>
      </header>

      <LocalInterviewHistory />
    </div>
  );
}
