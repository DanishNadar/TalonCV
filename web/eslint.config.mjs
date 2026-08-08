import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTypescript,
  // `public/ort` holds vendored ONNX Runtime bundles and `out` is build output;
  // neither is ours to lint.
  globalIgnores([".next/**", "out/**", "public/ort/**", "coverage/**", "playwright-report/**", "test-results/**"]),
]);
