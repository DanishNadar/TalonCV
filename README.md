# TalonCV

TalonCV is a browser-local, multimodal interview-practice coach. The public app records or imports media, stores it in the browser, analyzes it in browser workers, and produces an explainable coaching review without an account, database, backend worker, or AI inference API.

## Public production architecture

```text
Static Next.js host
  -> user browser
  -> MediaRecorder / Web Audio / IndexedDB
  -> browser workers: Whisper + audio DSP + MiniLM + YOLO ONNX + MediaPipe
  -> cue rules + optional learned random forest + temporal events
  -> alignment + deterministic scores + report
  -> optional local SmolLM wording
```

Internet is needed only for the static site and first-use public model downloads. Recording, transcript, frames, semantic evidence, scores, and reports do not leave the browser during normal use.

The eight review tabs are Overview, Transcript, Answer Quality, Vocal Delivery, Visual Cues, Multimodal Moments, Full Report, and Downloads. Local sessions can be replayed, reanalyzed, exported as ZIP, imported, and deleted.

## Run the public app

```powershell
cd web
npm install
npm run typecheck
npm run lint
npm test
npm run build
npm run dev
```

`next.config.ts` uses static export. Deploy `web` to Vercel with no environment variables or to any static host that serves the generated `web/out` directory with HTTPS. See [DEPLOYMENT.md](DEPLOYMENT.md).

## Browser model stack

Pinned definitions live in [browser-models.ts](web/config/browser-models.ts).

| Purpose | Browser model/runtime | Quantization | Approx. public download |
| --- | --- | --- | ---: |
| Transcription | `onnx-community/whisper-tiny.en` | q4 (fast) or q8 (balanced) | 94.7 MB / 41.4 MB |
| Semantics | `Xenova/all-MiniLM-L6-v2` | q8 | 22.9 MB |
| Face localization | local `yolo11n-face.onnx` with ONNX Runtime Web | ONNX export | prepared at deployment |
| Landmarks | MediaPipe Face Detector, Face Landmarker, Pose Landmarker | float16 task assets | 9.3 MB |
| Optional coaching wording | `onnx-community/SmolLM2-135M-Instruct-ONNX-MHA` | q4f16 | 114.4 MB |

The deterministic report and scores do not depend on the optional coach. YOLO localizes faces; it is not emotion detection. Cue evidence is limited to observable recording features and never determines hiring suitability, personality, honesty, intelligence, emotion, anxiety, mental health, or protected characteristics.

## Prepare browser visual assets

The static host should contain `web/public/models/yolo11n-face.onnx` before deployment. A maintainer who has already run the local Python model setup can create it with:

```powershell
python scripts/exportYoloFaceOnnx.py
```

If the asset is absent, the browser falls back to MediaPipe face detection and reports the visual engine accordingly. End users do not run Python.

Train the existing scikit-learn random-forest cue classifier and export its browser JSON form with:

```powershell
python scripts/trainCueClassifier.py
```

This writes `web/public/models/cue-classifier.json`. Browser inference faithfully evaluates exported tree structure, imputation, class probabilities, confidence threshold, conflict rules, and provenance.

## Research/reference application

The Python/Streamlit system remains a local research reference with larger local models:

```powershell
streamlit run app.py
```

It is not the selected public deployment architecture. Its local-only setup and training utilities remain available for research and dataset preparation.
