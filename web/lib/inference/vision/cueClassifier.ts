import type { VisualFeatureRow } from "./cueRules";

export interface BrowserCueTree { childrenLeft: number[]; childrenRight: number[]; feature: number[]; threshold: number[]; value: number[][]; }
export interface BrowserCueClassifier { schemaVersion: number; featureNames: string[]; classes: string[]; confidenceThreshold: number; imputer: { statistics: number[]; indicatorFeatures: number[] }; trees: BrowserCueTree[]; }
const exclusiveGroups = [["cameraFacing", "lookingAway"], ["centeredFraming", "offCenterFraming"], ["faceTooClose", "faceTooFar"], ["headTurnedLeft", "headTurnedRight"]];

export function applyBrowserCueClassifier(rows: VisualFeatureRow[], classifier: BrowserCueClassifier | null): VisualFeatureRow[] {
  if (!classifier) return rows;
  return rows.map((row) => {
    const prediction = predictBrowserCue(row, classifier); const candidate = prediction.candidate; const confidence = prediction.confidence; const existing = String(row.ruleFrameLabels || "").split(",").filter(Boolean);
    const conflict = exclusiveGroups.some((group) => group.includes(candidate) && existing.some((label) => group.includes(label) && label !== candidate));
    if (candidate !== "baseline" && confidence >= classifier.confidenceThreshold && !conflict && !existing.includes(candidate)) { row.frameLabels = [...existing, candidate].join(","); row.frameLabelSources = JSON.stringify({ ...Object.fromEntries(existing.map((label) => [label, ["rule"]])), [candidate]: ["learned"] }); row.mlCueLabel = candidate; row.mlCueConfidence = confidence; }
    return row;
  });
}

export function predictBrowserCue(row: VisualFeatureRow, classifier: BrowserCueClassifier): { candidate: string; confidence: number; probabilities: Record<string, number> } {
  const numeric = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : typeof value === "boolean" ? Number(value) : undefined;
  const vector = classifier.featureNames.map((name, index) => numeric(row[name]) ?? classifier.imputer.statistics[index] ?? 0);
  const indicators = classifier.imputer.indicatorFeatures.map((index) => numeric(row[classifier.featureNames[index]]) === undefined ? 1 : 0);
  const features = [...vector, ...indicators]; const scores = classifier.classes.map(() => 0);
  for (const tree of classifier.trees) { let node = 0; while (tree.childrenLeft[node] !== -1) node = features[tree.feature[node]] <= tree.threshold[node] ? tree.childrenLeft[node] : tree.childrenRight[node]; tree.value[node].forEach((value, index) => { scores[index] += value; }); }
  const total = scores.reduce((sum, value) => sum + value, 0) || 1; const best = scores.reduce((current, value, index) => value > scores[current] ? index : current, 0); return { candidate: classifier.classes[best], confidence: scores[best] / total, probabilities: Object.fromEntries(classifier.classes.map((label, index) => [label, scores[index] / total])) };
}

export async function loadBrowserCueClassifier(url = "/models/cue-classifier.json"): Promise<BrowserCueClassifier | null> {
  const response = await fetch(url, { cache: "force-cache" });
  if (response.status === 404) return null; if (!response.ok) throw new Error("The browser cue classifier could not be loaded.");
  const data = await response.json() as BrowserCueClassifier; return data.schemaVersion === 1 ? data : null;
}
