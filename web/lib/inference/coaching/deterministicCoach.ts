import type { LocalAnalysis } from "@/types/local";

export function deterministicCoach(analysis: LocalAnalysis): string[] {
  const response = analysis.responseAnalysis as Record<string, unknown>; const scoreBundle = analysis.scores.scores as Record<string, { practiceAreas?: string[] }> | undefined; const actions = [...(response.practiceAreas as string[] || []), ...(scoreBundle?.vocalDelivery?.practiceAreas || []), ...(scoreBundle?.visualDelivery?.practiceAreas || [])];
  return [...new Set(actions.filter(Boolean))].slice(0, 3).length ? [...new Set(actions.filter(Boolean))].slice(0, 3) : ["Repeat the answer with one concrete situation, the action you took, and the result.", "Use the replayable timestamps to preserve what worked and adjust one observable delivery detail at a time."];
}
