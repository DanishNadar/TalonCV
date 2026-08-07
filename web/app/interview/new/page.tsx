"use client";

import { useCallback, useState } from "react";
import Link from "next/link";
import { LocalRecorder } from "@/components/LocalRecorder";
import { sampleQuestions } from "@/lib/sampleQuestions";
import { StepRail, TechnicalBadge } from "@/components/ui/primitives";

const steps = [
  { id: "setup", name: "Interview Setup" },
  { id: "device", name: "Device Check" },
  { id: "recording", name: "Recording" },
  { id: "analysis", name: "Local Analysis" },
  { id: "review", name: "Review" },
];

export default function NewInterviewPage() {
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [role, setRole] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [competencies, setCompetencies] = useState("");
  const [step, setStep] = useState("setup");

  const onPhaseChange = useCallback((phase: string) => {
    setStep(phase === "idle" || phase === "error" ? "setup" : phase === "recording" ? "recording" : phase === "saving" ? "analysis" : "device");
  }, []);

  return (
    <div className="shell page-stack">
      <header className="section-header">
        <div className="stack">
          <span className="eyebrow">Session configuration</span>
          <h1 className="section-title">Configure the run, then record one focused answer</h1>
          <p>
            Context sharpens relevance coaching. TalonCV never converts these signals into a hiring recommendation.
          </p>
        </div>
        <Link className="button ghost" href="/dashboard">
          History
        </Link>
      </header>

      <StepRail steps={steps} current={step} />

      <section className="panel accent-top" aria-labelledby="setup-title">
        <div className="panel-header">
          <div className="row">
            <span className="mono" style={{ color: "var(--talon-red)" }}>
              01
            </span>
            <h2 id="setup-title">Interview setup</h2>
          </div>
          <TechnicalBadge>Run configuration</TechnicalBadge>
        </div>
        <div className="panel-body stack-5">
          <div className="stack-4">
            <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
              Suggested prompts
            </span>
            <div className="chip-row">
              {sampleQuestions.map((item) => (
                <button
                  key={item}
                  type="button"
                  className="chip"
                  aria-pressed={question === item}
                  onClick={() => setQuestion(item)}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="form-grid">
            <label className="wide">
              Interview question
              <textarea required maxLength={2000} rows={2} value={question} onChange={(event) => setQuestion(event.target.value)} />
            </label>
            <label>
              Target role
              <input maxLength={500} placeholder="e.g. Software engineering intern" value={role} onChange={(event) => setRole(event.target.value)} />
            </label>
            <label>
              Desired competencies
              <input maxLength={2000} placeholder="e.g. ownership, communication" value={competencies} onChange={(event) => setCompetencies(event.target.value)} />
            </label>
            <label className="wide">
              <span className="field-label">
                Job description / context
                <span className="optional">Optional</span>
              </span>
              <textarea
                maxLength={10000}
                rows={4}
                placeholder="Paste only the context you want TalonCV to use for semantic relevance."
                value={jobDescription}
                onChange={(event) => setJobDescription(event.target.value)}
              />
            </label>
          </div>
        </div>
      </section>

      <LocalRecorder
        context={{ interviewQuestion: question, targetRole: role, jobDescription, desiredCompetencies: competencies }}
        onPhaseChange={onPhaseChange}
      />

      <p className="fine-print">
        Model files download from public static hosts on first use and are cached on this device. Manage them in{" "}
        <Link className="text-button" href="/models">
          Models
        </Link>
        .
      </p>
    </div>
  );
}
