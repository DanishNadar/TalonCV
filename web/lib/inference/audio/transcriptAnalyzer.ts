import type { TranscriptArtifact } from "@/types/local";
import type { LocalSessionContext } from "@/types/local";

const fillers = ["you know", "kind of", "sort of", "i guess", "basically", "actually", "um", "uh", "like"];
const hedges = ["maybe", "perhaps", "probably", "i think", "i feel like", "i guess", "somewhat", "might have", "could be"];
const genericClaims = ["hard worker", "team player", "people person", "detail oriented", "good communicator", "work well under pressure"];
const tokens = (text: string) => text.toLowerCase().match(/[a-z0-9']+/g) || [];
const includesPhrase = (text: string, phrase: string) => text.toLowerCase().includes(phrase);
const rating = (score: number) => score >= 85 ? "Strong evidence" : score >= 70 ? "Generally effective" : score >= 55 ? "Mixed evidence" : "Practice priority";

export function analyzeTranscript(transcript: TranscriptArtifact, context: LocalSessionContext, semantic?: Record<string, unknown>): Record<string, unknown> {
  const text = transcript.text.trim(); const wordCount = tokens(text).length;
  if (!text) return { analysisVersion: "browser-response-v1", available: false, confidence: "unavailable", metrics: { wordCount: 0 }, rubric: {}, practiceAreas: ["Record an audible answer so TalonCV can provide transcript-based coaching."], suggestedAnswerStructure: [] };
  const lower = text.toLowerCase(); const occurrences = fillers.flatMap((phrase) => {
    const matches = [...lower.matchAll(new RegExp(`\\b${phrase.replaceAll(" ", "\\s+")}\\b`, "g"))];
    return matches.map((match) => { const ratio = (match.index || 0) / Math.max(1, text.length); const segment = transcript.segments.find((item) => ratio >= 0 && item.text.toLowerCase().includes(phrase)); return { phrase, startTime: segment?.start ?? 0, endTime: segment?.end ?? 0, confidence: "medium" }; });
  });
  const hasExample = /\b(for example|for instance|when i|in my previous|on a project)\b/i.test(text); const hasAction = /\b(i led|i built|i created|i implemented|i improved|i coordinated|i decided)\b/i.test(text); const hasResult = /\b(result|reduced|increased|improved|saved|grew|delivered|%|percent)\b/i.test(text); const hasConclusion = /\b(in summary|overall|that is why|as a result|going forward)\b/i.test(text);
  const fillerRate = wordCount ? (occurrences.length / wordCount) * 100 : 0; const genericCount = genericClaims.filter((phrase) => includesPhrase(lower, phrase)).length; const hedgeCount = hedges.filter((phrase) => includesPhrase(lower, phrase)).length;
  const strongPhrases = transcript.segments.filter((segment) => /\b(result|improved|reduced|increased|delivered|led|built|created)\b/i.test(segment.text)).slice(0, 5).map((segment) => ({ startTime: segment.start, endTime: segment.end, text: segment.text, reasons: ["specific action or result wording"] }));
  const relevanceScore = typeof semantic?.questionRelevance === "number" ? semantic.questionRelevance as number : null;
  const structureScore = Math.min(100, 35 + (hasExample ? 20 : 0) + (hasAction ? 20 : 0) + (hasResult ? 20 : 0) + (hasConclusion ? 5 : 0));
  const clarityScore = Math.max(0, Math.min(100, 85 - fillerRate * 5 - hedgeCount * 3 - genericCount * 5));
  const specificityScore = Math.min(100, 35 + (hasExample ? 25 : 0) + (hasAction ? 20 : 0) + (hasResult ? 20 : 0));
  const lengthScore = wordCount < 45 ? 50 : wordCount <= 350 ? 85 : 65;
  const overall = Math.round((structureScore + clarityScore + specificityScore + lengthScore + (relevanceScore ?? 70)) / 5);
  const practiceAreas = [!hasExample && "Add one concrete situation or project example.", !hasAction && "State the action you personally took.", !hasResult && "Close the example with an observable result or learning.", fillerRate > 3 && "Replace repeated filler phrases with a brief intentional pause.", wordCount < 45 && "Develop the answer with enough context, action, and result for a listener to follow."].filter(Boolean);
  const rubricScores: Array<[string, number]> = [["structure", structureScore], ["clarity", clarityScore], ["specificity", specificityScore], ["conciseness", lengthScore], ["overallVerbalResponse", overall]];
  const rubric = Object.fromEntries(rubricScores.map(([name, score]) => [name, { score, rating: rating(score), formula: "Deterministic transcript evidence; not a personality or hiring assessment." }]));
  return { analysisVersion: "browser-response-v1", available: true, confidence: transcript.averageConfidence !== null && transcript.averageConfidence < 0.6 ? "limited" : "medium", metrics: { wordCount, fillerCount: occurrences.length, fillerRatePer100Words: Number(fillerRate.toFixed(2)), hedgeCount, genericClaimCount: genericCount, exampleMarkerCount: hasExample ? 1 : 0, actionMarkerCount: hasAction ? 1 : 0, resultMarkerCount: hasResult ? 1 : 0, hasConclusion }, fillerOccurrences: occurrences, strongPhrases, relevanceAnalysis: { available: relevanceScore !== null, score: relevanceScore }, starAnalysis: { componentsPresent: [hasExample, hasAction, hasResult, hasConclusion].filter(Boolean).length }, rubric, answerDevelopment: { missingElements: practiceAreas, openingNote: wordCount ? "Opening detected from transcript." : "Unavailable", lengthAssessment: wordCount < 45 ? "Brief response" : wordCount > 350 ? "Long response" : "Focused response", exampleQuality: hasExample ? "Concrete example marker detected" : "No concrete example marker", resultQuality: hasResult ? "Result-oriented wording detected" : "No result marker" }, practiceAreas, suggestedAnswerStructure: ["State the situation and your responsibility.", "Describe the action you personally took.", "Name the result, learning, or next step."], semanticAnalysisUsed: Boolean(semantic?.available) };
}
