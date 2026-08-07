# TalonCV zero-cost public deployment

The public TalonCV app is a static Next.js application. It requires no Railway service, Supabase project, Docker image, database, account, authentication provider, model API key, or server-side inference worker.

## Build and verify

```powershell
cd web
npm install
npm run typecheck
npm run lint
npm test
npm run build
```

`npm run build` creates `web/out`. Serve that directory from any HTTPS static host. HTTPS is required for camera and microphone permissions outside localhost.

To exercise the browser-local models end to end, run the gated integration tests:

```powershell
cd web
$env:TALONCV_REAL_MODELS = "1"; npm run test:e2e
```

These download the real public model files and run a full local analysis, the eight-tab review, and the ZIP export/import/delete round-trip.

## Static-host headers

If you serve the build from a host other than Vercel, reproduce the `web/vercel.json` headers there. Two of them are load-bearing rather than cosmetic:

- `script-src` must keep `'unsafe-inline'`. A static export has no nonce, and Next.js boots from inline scripts, so dropping it leaves the page rendered but unhydrated and completely non-interactive.
- `script-src` and `connect-src` must both allow `https://cdn.jsdelivr.net`. The MediaPipe and ONNX Runtime workers load their WebAssembly glue from there.

Verify headers against the real build before announcing a deploy, not against `npm run dev`, which sends none of them.

## Deploy to Vercel

1. Import the repository in Vercel.
2. Set the Root Directory to `web`.
3. Do not add environment variables; the core production app needs none.
4. Deploy. Vercel reads the static-export configuration and `web/vercel.json` static-host security headers.
5. Verify a first run: camera/microphone permission, local recording, model download progress, local analysis, eight review tabs, ZIP export, import, and deletion.

The same build output works on Cloudflare Pages, GitHub Pages (with static-host routing configured), Netlify, or another static HTTPS host.

## Browser model delivery

First use may download public static model files from the model hosts pinned in `web/config/browser-models.ts` and MediaPipe model URLs. The model manager shows capability information, cache state, download progress for preloaded vision assets, individual removal, and clear-all controls. Interview content is never sent to those hosts.

Before deployment, prepare these optional static public assets:

```powershell
# Maintainer-only; uses the existing local YOLO checkpoint.
python scripts/exportYoloFaceOnnx.py

# Maintainer-only; trains sklearn and emits browser-compatible random-forest JSON.
python scripts/trainCueClassifier.py
```

They produce `web/public/models/yolo11n-face.onnx` and `web/public/models/cue-classifier.json`. The visual pipeline degrades cleanly to MediaPipe rules when either optional asset is absent.

## Offline check after caching

1. Open TalonCV while online and run an analysis to cache the required models.
2. In the browser developer tools, confirm the requested models appear in Cache Storage.
3. Disconnect network and reload the already-cached app.
4. Record/import a short test and rerun analysis. Browser codec support and model-runtime cache behavior vary by browser; report actual results rather than assuming all model assets are available offline.

The service worker caches only application-shell and public static/model requests. It never caches media blobs, session objects, transcripts, or reports; those belong in IndexedDB.
