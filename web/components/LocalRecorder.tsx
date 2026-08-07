"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { createSession, saveSession } from "@/lib/storage/sessionStore";
import { saveRecording } from "@/lib/storage/mediaStore";
import { maxRecordingSeconds, maxUploadBytes, permissionErrorMessage, preferredRecorderMimeType } from "@/lib/recording";
import { LocalAIStatus, TechnicalBadge } from "@/components/ui/primitives";
import { SystemReadiness } from "@/components/ui/SystemReadiness";
import type { LocalSessionContext } from "@/types/local";

type RecorderPhase = "idle" | "requesting" | "ready" | "recording" | "saving" | "error";

const clock = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

function durationOf(blob: Blob): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const media = document.createElement("video");
    const done = () => {
      URL.revokeObjectURL(url);
      resolve(Number.isFinite(media.duration) ? media.duration : 0);
    };
    media.preload = "metadata";
    media.onloadedmetadata = done;
    media.onerror = done;
    media.src = url;
  });
}

export function LocalRecorder({
  context,
  onPhaseChange,
}: {
  context: LocalSessionContext;
  onPhaseChange?: (phase: RecorderPhase) => void;
}) {
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

  useEffect(() => {
    onPhaseChange?.(phase);
  }, [phase, onPhaseChange]);

  useEffect(
    () => () => {
      window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  async function saveLocal(blob: Blob) {
    setPhase("saving");
    try {
      const durationSeconds = await durationOf(blob);
      const hasVideo = blob.type.startsWith("video/");
      const session = await createSession(context);
      await saveRecording(session.id, blob);
      session.recording = {
        id: crypto.randomUUID(),
        mimeType: blob.type || "application/octet-stream",
        durationSeconds,
        sizeBytes: blob.size,
        hasAudio: true,
        hasVideo,
        createdAt: new Date().toISOString(),
      };
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
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true,
      });
      streamRef.current = stream;
      if (previewRef.current) {
        previewRef.current.srcObject = stream;
        await previewRef.current.play().catch(() => undefined);
      }
      const audio = new AudioContext();
      const analyser = audio.createAnalyser();
      audio.createMediaStreamSource(stream).connect(analyser);
      const bytes = new Uint8Array(analyser.fftSize);
      const meter = window.setInterval(() => {
        analyser.getByteTimeDomainData(bytes);
        const level = bytes.reduce((sum, value) => sum + Math.abs(value - 128), 0) / bytes.length;
        setAudioLevel(Math.min(100, level * 3.4));
      }, 100);
      stream.addEventListener(
        "inactive",
        () => {
          window.clearInterval(meter);
          void audio.close();
        },
        { once: true },
      );
      setPhase("ready");
      setMessage("Devices are ready. A five-minute cap keeps the take inside browser storage.");
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
    recorder.ondataavailable = (event) => {
      if (event.data.size) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      void saveLocal(new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" }));
    };
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
    if (file.size > maxUploadBytes) {
      setPhase("error");
      setMessage("Choose a file smaller than 250 MB so it can stay in local browser storage.");
      return;
    }
    if (!file.type.startsWith("video/") && !file.type.startsWith("audio/")) {
      setPhase("error");
      setMessage("Choose an audio or video recording.");
      return;
    }
    await saveLocal(file);
  }

  const active = phase === "recording";
  const deviceState = phase === "error" ? "error" : phase === "ready" || active || phase === "saving" ? "ready" : "waiting";
  const remaining = maxRecordingSeconds - elapsed;

  return (
    <section className="panel accent-top" aria-labelledby="recorder-title">
      <div className="panel-header">
        <div className="row">
          <span className="mono" style={{ color: "var(--talon-red)" }}>
            03
          </span>
          <h2 id="recorder-title">Record or import</h2>
        </div>
        <div className="row">
          <TechnicalBadge tone={active ? "error" : phase === "ready" ? "success" : "neutral"} dot>
            {phase}
          </TechnicalBadge>
          <LocalAIStatus variant="compact" label="Local" detail="Media never uploads" />
        </div>
      </div>

      <div className="panel-body">
        <div className="recorder-layout">
          <div className="preview-shell">
            <video ref={previewRef} muted playsInline aria-label="Camera preview" />
            {phase === "idle" ? (
              <span className="preview-placeholder">
                <span className="mono">Camera offline</span>
                <span>Enable your devices to preview locally</span>
              </span>
            ) : null}
            {phase !== "idle" ? <span className="frame-guides" aria-hidden="true"><i /><i /><i /><i /></span> : null}
            {active ? (
              <span className="recording-indicator">
                <span className="rec-dot" aria-hidden="true" />
                REC {clock(elapsed)}
              </span>
            ) : null}
            {phase !== "idle" ? <span className="preview-overlay-badge">On-device capture</span> : null}
          </div>

          <div className="recorder-side">
            <div className="question-card">
              <span className="mono">Prompt</span>
              <p>{context.interviewQuestion || "No interview question set."}</p>
              {context.targetRole ? <p className="role">{context.targetRole}</p> : null}
            </div>

            <div className="panel" style={{ background: "var(--talon-surface-sunken)" }}>
              <div className="panel-header">
                <h3>System readiness</h3>
                <span className="mono" style={{ color: "var(--talon-text-tertiary)" }}>
                  02
                </span>
              </div>
              <SystemReadiness camera={deviceState} microphone={deviceState} audioLevel={audioLevel} />
            </div>

            {active || elapsed > 0 ? (
              <div className="elapsed-readout">
                <span className="time">{clock(elapsed)}</span>
                <span className="cap">{active ? `${clock(Math.max(0, remaining))} left` : `cap ${clock(maxRecordingSeconds)}`}</span>
              </div>
            ) : null}

            {phase === "idle" || phase === "error" ? (
              <button className="button primary full" onClick={() => void enableDevices()}>
                Enable camera + microphone
              </button>
            ) : active ? (
              <button className="button danger full" onClick={stopRecording}>
                Stop recording
              </button>
            ) : (
              <button className="button primary full" disabled={phase !== "ready"} onClick={startRecording}>
                {phase === "saving" ? "Saving locally…" : "Start recording"}
              </button>
            )}

            <p className="fine-print">{message}</p>

            {phase !== "saving" ? (
              <label className="file-button">
                <span aria-hidden="true">↥</span>
                Import a recording
                <input type="file" accept="video/*,audio/*" onChange={(event) => void importFile(event)} />
              </label>
            ) : null}
          </div>
        </div>
      </div>
    </section>
  );
}
