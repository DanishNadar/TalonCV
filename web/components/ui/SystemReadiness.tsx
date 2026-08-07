"use client";

import { useEffect, useState } from "react";
import { detectCapabilities, resolveInferenceMode, type DeviceCapabilities } from "@/lib/inference/capabilities";
import { formatBytes } from "@/config/browser-models";
import { StatusDot, type Tone } from "./primitives";

export interface ReadinessRow {
  label: string;
  value: string;
  tone: Tone;
}

function Row({ label, value, tone }: ReadinessRow) {
  return (
    <div className="readiness-row">
      <StatusDot tone={tone} />
      <span className="label">{label}</span>
      <span className={`value ${tone === "neutral" ? "muted-value" : ""}`}>{value}</span>
    </div>
  );
}

/** A plain-language system check. Deliberately excludes raw hardware fingerprint
 *  detail — only what a user can act on. */
export function SystemReadiness({
  camera,
  microphone,
  audioLevel,
}: {
  camera: "waiting" | "ready" | "error";
  microphone: "waiting" | "ready" | "error";
  audioLevel?: number;
}) {
  const [capabilities, setCapabilities] = useState<DeviceCapabilities>();

  useEffect(() => {
    void detectCapabilities()
      .then(setCapabilities)
      .catch(() => undefined);
  }, []);

  const deviceTone = (state: "waiting" | "ready" | "error"): Tone =>
    state === "ready" ? "success" : state === "error" ? "error" : "neutral";
  const deviceValue = (state: "waiting" | "ready" | "error") =>
    state === "ready" ? "Ready" : state === "error" ? "Unavailable" : "Waiting";

  const storageValue = !capabilities
    ? "Checking"
    : capabilities.storageQuotaBytes
      ? `${formatBytes(capabilities.storageQuotaBytes)} quota`
      : "Available";

  const mode = capabilities ? resolveInferenceMode("automatic", capabilities) : null;

  const rows: ReadinessRow[] = [
    { label: "Camera", value: deviceValue(camera), tone: deviceTone(camera) },
    { label: "Microphone", value: deviceValue(microphone), tone: deviceTone(microphone) },
    { label: "Local storage", value: storageValue, tone: capabilities ? "success" : "neutral" },
    {
      label: "Browser inference",
      value: !capabilities ? "Checking" : capabilities.wasm ? "Supported" : "Unsupported",
      tone: !capabilities ? "neutral" : capabilities.wasm ? "success" : "error",
    },
    {
      label: "WebGPU",
      value: !capabilities ? "Checking" : capabilities.webgpu ? "Available" : "Not available",
      tone: !capabilities ? "neutral" : capabilities.webgpu ? "success" : "neutral",
    },
    {
      label: "Analysis mode",
      value: mode === null ? "Checking" : mode === "webgpu" ? "WebGPU" : "CPU / WASM",
      tone: "info",
    },
  ];

  const bars = 16;
  const active = Math.round(((audioLevel ?? 0) / 100) * bars);

  return (
    <div className="stack-4">
      <div className="readiness">
        {rows.map((row) => (
          <Row key={row.label} {...row} />
        ))}
      </div>

      {microphone === "ready" ? (
        <div className="stack-4" style={{ padding: "0 var(--space-4) var(--space-4)" }}>
          <div className="meter-head">
            <span>Audio level</span>
            <strong>{Math.round(audioLevel ?? 0)}%</strong>
          </div>
          <div
            className="audio-meter"
            role="meter"
            aria-label="Microphone input level"
            aria-valuenow={Math.round(audioLevel ?? 0)}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            {Array.from({ length: bars }, (_, index) => (
              <i
                key={index}
                className={index < active ? (index > bars - 4 ? "hot" : "on") : ""}
                style={{ height: `${38 + (index / bars) * 62}%` }}
              />
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}
