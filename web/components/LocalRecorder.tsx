"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { createSession, saveSession } from "@/lib/storage/sessionStore";
import { saveRecording } from "@/lib/storage/mediaStore";
import { maxRecordingSeconds, maxUploadBytes, permissionErrorMessage, preferredRecorderMimeType } from "@/lib/recording";
import type { LocalSessionContext } from "@/types/local";

type RecorderPhase = "idle" | "requesting" | "ready" | "recording" | "saving" | "error";

function durationOf(blob: Blob): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const media = document.createElement("video");
    const done = () => { URL.revokeObjectURL(url); resolve(Number.isFinite(media.duration) ? media.duration : 0); };
    media.preload = "metadata";
    media.onloadedmetadata = done;
    media.onerror = done;
    media.src = url;
  });
}

export function LocalRecorder({ context }: { context: LocalSessionContext }) {
  const router = useRouter();
  const previewRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const mediaRecorderRef = useRef<MediaRecorder | undefined>(undefined);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | undefined>(undefined);
  const [phase, setPhase] = useState<RecorderPhase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [message, setMessage] = useState("Camera and microphone stay on this device. Recordings are saved in this browser only.");

  useEffect(() => () => {
    window.clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
  }, []);

  async function saveLocal(blob: Blob) {
    setPhase("saving");
    try {
      const durationSeconds = await durationOf(blob);
      const hasVideo = blob.type.startsWith("video/");
      const session = await createSession(context);
      await saveRecording(session.id, blob);
      session.recording = { id: crypto.randomUUID(), mimeType: blob.type || "application/octet-stream", durationSeconds, sizeBytes: blob.size, hasAudio: true, hasVideo, createdAt: new Date().toISOString() };
      await saveSession(session);
      router.push(`/interview?id=${encodeURIComponent(session.id)}`);
    } catch (error) {
      setPhase("error");
      setMessage(error instanceof Error ? error.message : "The recording could not be saved in this browser.");
    }
  }

  async function enableDevices() {
    setPhase("requesting");
    setMessage("Requesting camera and microphone permission…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true });
      streamRef.current = stream;
      if (previewRef.current) { previewRef.current.srcObject = stream; await previewRef.current.play().catch(() => undefined); }
      const audio = new AudioContext();
      const analyser = audio.createAnalyser();
      audio.createMediaStreamSource(stream).connect(analyser);
      const bytes = new Uint8Array(analyser.fftSize);
      const meter = window.setInterval(() => { analyser.getByteTimeDomainData(bytes); const level = bytes.reduce((sum, value) => sum + Math.abs(value - 128), 0) / bytes.length; setAudioLevel(Math.min(100, level * 3.4)); }, 100);
      stream.addEventListener("inactive", () => { window.clearInterval(meter); void audio.close(); }, { once: true });
      setPhase("ready");
      setMessage("Devices are ready. A five-minute cap protects local storage.");
    } catch (error) {
      setPhase("error");
      setMessage(permissionErrorMessage(error));
    }
  }

  function startRecording() {
    const stream = streamRef.current;
    if (!stream) return;
    const recorder = new MediaRecorder(stream, { mimeType: preferredRecorderMimeType(), videoBitsPerSecond: 2_500_000 });
    mediaRecorderRef.current = recorder;
    chunksRef.current = [];
    setElapsed(0);
    recorder.ondataavailable = (event) => { if (event.data.size) chunksRef.current.push(event.data); };
    recorder.onstop = () => { void saveLocal(new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" })); };
    recorder.start(1000);
    setPhase("recording");
    setMessage("Recording locally. Stop when you finish your answer.");
    timerRef.current = window.setInterval(() => {
      setElapsed((seconds) => {
        if (seconds + 1 >= maxRecordingSeconds) mediaRecorderRef.current?.stop();
        return seconds + 1;
      });
    }, 1000);
  }

  function stopRecording() {
    window.clearInterval(timerRef.current);
    if (mediaRecorderRef.current?.state === "recording") mediaRecorderRef.current.stop();
    setPhase("saving");
  }

  async function importFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    if (file.size > maxUploadBytes) { setPhase("error"); setMessage("Choose a file smaller than 250 MB so it can stay in local browser storage."); return; }
    if (!file.type.startsWith("video/") && !file.type.startsWith("audio/")) { setPhase("error"); setMessage("Choose an audio or video recording."); return; }
    await saveLocal(file);
  }

  const active = phase === "recording";
  return (
    <section className="recorder-card" aria-labelledby="recorder-title">
      <div className="recorder-heading"><div><div className="step-label">Step 2</div><h2 id="recorder-title">Record or import locally</h2></div><span className={`state-pill ${phase}`}>{phase}</span></div>
      <div className="recorder-layout">
        <div className="preview-shell">
          <video ref={previewRef} muted playsInline aria-label="Camera preview" />
          {phase === "idle" && <span className="preview-placeholder">Enable your camera to preview locally</span>}
          {active && <span className="recording-indicator"><span />REC {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</span>}
        </div>
        <div className="device-panel">
          <h3>Local capture</h3>
          <div className="device-row"><i className={`device-dot ${phase !== "idle" && phase !== "error" ? "ok" : ""}`} /><span>Camera</span><strong>{phase === "ready" || active ? "Ready" : "Waiting"}</strong></div>
          <div className="device-row"><i className={`device-dot ${phase !== "idle" && phase !== "error" ? "ok" : ""}`} /><span>Microphone</span><strong>{phase === "ready" || active ? "Ready" : "Waiting"}</strong></div>
          <div className="meter-label"><span>Microphone level</span><span>{Math.round(audioLevel)}%</span></div>
          <div className="audio-meter"><span style={{ width: `${audioLevel}%` }} /></div>
          <p className="fine-print">Media never uploads to TalonCV. It is stored with IndexedDB in this browser.</p>
          {phase === "idle" || phase === "error" ? <button className="button primary full" onClick={() => void enableDevices()}>Enable camera + microphone</button> : active ? <button className="button danger full" onClick={stopRecording}>Stop recording</button> : <button className="button primary full" disabled={phase !== "ready"} onClick={startRecording}>Start recording</button>}
        </div>
      </div>
      <div className={`ready-banner ${phase === "ready" ? "success" : ""}`}><div><strong>{phase === "saving" ? "Saving locally" : phase === "ready" ? "Ready to record" : "Privacy-first capture"}</strong><span>{message}</span></div>{phase !== "saving" && <label className="file-button">Import a recording<input type="file" accept="video/*,audio/*" onChange={(event) => void importFile(event)} /></label>}</div>
    </section>
  );
}
