# TalonCV public production architecture

```mermaid
flowchart TD
  H[Static HTTPS host] --> B[User browser]
  B --> C[Camera / microphone or local file]
  C --> I[(IndexedDB local media and session data)]
  I --> W[Browser Web Workers]
  W --> S[Whisper Tiny transcription + audio DSP]
  W --> M[MiniLM semantic analysis]
  W --> V[YOLO ONNX face localization + MediaPipe landmarks]
  V --> Q[Cue rules + optional exported random forest + state machine]
  S --> A[Timestamp alignment + explainable scores]
  M --> A
  Q --> A
  A --> R[Deterministic local report and eight-tab review]
  R --> E[Client-side ZIP export/import/delete]
```

The host serves application code and public model files only. During normal analysis it receives no interview recording, audio, frame, transcript, evidence, score, or report. There is no public backend worker, database, authentication system, queue, inference API, or required secret.

WebAssembly/CPU is the baseline. WebGPU is an optional acceleration choice. Static model downloads are cached by the model runtime and browser cache; local recording/session data is persisted separately in IndexedDB.

The Python/Streamlit pipeline is a research/reference implementation, not a public production dependency.
