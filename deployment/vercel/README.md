# Vercel

Set the Vercel Root Directory to `web` and deploy with no environment variables. `web/next.config.ts` produces a static export and `web/vercel.json` provides static-host security headers.

Run locally before deploying:

```powershell
cd web
npm install
npm run typecheck
npm run lint
npm test
npm run build
```

Vercel serves the frontend only. Browser-local inference, IndexedDB persistence, and client-side exports are the complete core product.
