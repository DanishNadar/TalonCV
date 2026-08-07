"use client";

import { useState } from "react";
import { LocalRecorder } from "@/components/LocalRecorder";
import { LocalModelManager } from "@/components/LocalModelManager";
import { sampleQuestions } from "@/lib/sampleQuestions";

export default function NewInterviewPage() {
  const [question, setQuestion] = useState(sampleQuestions[0]);
  const [role, setRole] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [competencies, setCompetencies] = useState("");
  return (
    <div className="shell page-stack">
      <header className="page-heading"><div className="eyebrow">New practice interview</div><h1>Set the context, then record one focused answer.</h1><p>Context improves relevance coaching. TalonCV never turns these signals into a hiring recommendation.</p></header>
      <section className="setup-card" aria-labelledby="setup-title">
        <div className="section-heading"><span className="step-label">Step 1</span><h2 id="setup-title">Interview setup</h2></div>
        <div className="form-grid">
          <div className="wide question-fields"><label>Sample question<select value={sampleQuestions.includes(question) ? question : "custom"} onChange={(event) => setQuestion(event.target.value === "custom" ? "" : event.target.value)}>{sampleQuestions.map((item) => <option key={item}>{item}</option>)}<option value="custom">Write my own question…</option></select></label><label>Interview question<textarea required maxLength={2000} rows={2} value={question} onChange={(event) => setQuestion(event.target.value)} /></label></div>
          <label>Target role<input maxLength={500} placeholder="e.g. Software engineer" value={role} onChange={(event) => setRole(event.target.value)} /></label>
          <label>Desired competencies<input maxLength={2000} placeholder="e.g. ownership, communication" value={competencies} onChange={(event) => setCompetencies(event.target.value)} /></label>
          <label className="wide">Job description <span className="optional">Optional</span><textarea maxLength={10000} rows={4} placeholder="Paste only the context you want TalonCV to use." value={jobDescription} onChange={(event) => setJobDescription(event.target.value)} /></label>
        </div>
      </section>
      <LocalRecorder context={{ interviewQuestion: question, targetRole: role, jobDescription, desiredCompetencies: competencies }} />
      <LocalModelManager />
    </div>
  );
}
