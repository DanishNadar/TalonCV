import { expect, test } from "@playwright/test";

test("public app exposes browser-local setup with no account path", async ({ page }) => {
  const requests: string[] = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("Improve with multimodal AI");
  await expect(page.getByRole("link", { name: "Start Interview" })).toBeVisible();
  await expect(page.getByText(/recording, transcript, analysis, and report never leave this device/i)).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Primary" }).getByRole("link", { name: "Models" })).toBeVisible();
  expect(requests.some((url) => /supabase|railway|api\.openai\.com|api-inference|replicate|groq|together/i.test(url))).toBe(false);
});

test("synthetic Chromium camera and microphone create a browser-local session", async ({ page }) => {
  await page.goto("/interview/new");
  await page.getByRole("button", { name: "Enable camera + microphone" }).click();
  await expect(page.getByRole("button", { name: "Start recording" })).toBeVisible();
  await page.getByRole("button", { name: "Start recording" }).click();
  await expect(page.getByRole("button", { name: "Stop recording" })).toBeVisible();
  await page.waitForTimeout(1200);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await expect(page.getByText("Local interview review", { exact: true })).toBeVisible({ timeout: 15000 });
  await expect(page.getByRole("heading", { name: "Ready for a local review" })).toBeVisible({ timeout: 15000 });
  await page.goto("/dashboard");
  await expect(page.getByText(/Tell me about yourself/i)).toBeVisible();
});

test("real browser model integration downloads public model assets without interview uploads", async ({ page }) => {
  test.skip(process.env.TALONCV_REAL_MODELS !== "1", "Set TALONCV_REAL_MODELS=1 to run first-use browser-model integration against public static model assets.");
  // A cold first run downloads Whisper, MiniLM, and the MediaPipe assets, so the
  // test budget has to exceed the default per-test timeout.
  test.setTimeout(600000);
  const forbidden: string[] = [];
  page.on("request", (request) => { if (/api-inference|api\.openai\.com|supabase|railway|replicate|groq|together/i.test(request.url())) forbidden.push(request.url()); });
  await page.goto("/interview/new");
  await page.getByRole("button", { name: "Enable camera + microphone" }).click();
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.waitForTimeout(4000);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await page.getByRole("button", { name: "Run local multimodal analysis" }).click({ timeout: 30000 });
  // The completion banner is transient, so assert the settled review instead.
  await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 480000 });
  await expect(page.getByRole("tab")).toHaveCount(8);
  await page.getByRole("tab", { name: "Full Report" }).click();
  await expect(page.locator(".report-text").first()).toContainText("TalonCV Local Interview Review");
  expect(forbidden).toEqual([]);

  // The speech runtime must actually initialize. Session-creation and backend
  // failures previously surfaced only as an empty transcript, which is
  // indistinguishable from a silent recording unless the warning is inspected.
  const warnings = await page.evaluate(
    () =>
      new Promise<string[]>((resolve) => {
        const open = indexedDB.open("taloncv-local");
        open.onsuccess = () => {
          const get = open.result.transaction("artifacts").objectStore("artifacts").getAll();
          get.onsuccess = () => {
            const analysis = get.result.find((row) => row.key === "analysis")?.value;
            resolve(analysis?.transcript?.warnings ?? []);
          };
        };
        open.onerror = () => resolve(["indexeddb-unavailable"]);
      }),
  );
  expect(warnings.join(" ")).not.toMatch(/Can't create a session|no available backend|Failed to fetch dynamically imported module/i);
});

test("ZIP export, re-import, and delete round-trip stays inside the browser", async ({ page }) => {
  test.skip(process.env.TALONCV_REAL_MODELS !== "1", "Set TALONCV_REAL_MODELS=1 to run the export/import round-trip, which needs a completed local analysis.");
  test.setTimeout(600000);
  page.on("dialog", (dialog) => void dialog.accept());
  await page.goto("/interview/new");
  await page.getByRole("button", { name: "Enable camera + microphone" }).click();
  await page.getByRole("button", { name: "Start recording" }).click();
  await page.waitForTimeout(4000);
  await page.getByRole("button", { name: "Stop recording" }).click();
  await page.getByRole("button", { name: "Run local multimodal analysis" }).click({ timeout: 30000 });
  await expect(page.getByRole("tab", { name: "Export" })).toBeVisible({ timeout: 480000 });

  await page.getByRole("tab", { name: "Export" }).click();
  const downloadPromise = page.waitForEvent("download", { timeout: 60000 });
  await page.getByRole("button", { name: "Download ZIP" }).click();
  const download = await downloadPromise;
  // importSessionBundle requires a .zip name, so keep the suggested filename.
  const bundle = test.info().outputPath(download.suggestedFilename());
  await download.saveAs(bundle);

  await page.goto("/dashboard");
  await expect(page.locator(".history-row")).toHaveCount(1, { timeout: 30000 });
  await page.setInputFiles('input[type="file"][accept*="zip"]', bundle);
  await expect(page.getByRole("tab")).toHaveCount(8, { timeout: 60000 });

  // The imported session must survive a reload, which is what proves it landed in
  // IndexedDB rather than in component state.
  await page.goto("/dashboard");
  await expect(page.locator(".history-row")).toHaveCount(2, { timeout: 30000 });
  await page.locator(".history-row").first().getByRole("button", { name: "Delete" }).click();
  await expect(page.locator(".history-row")).toHaveCount(1, { timeout: 30000 });
});
