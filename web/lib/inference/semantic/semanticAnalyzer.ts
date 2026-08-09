import type { LocalSessionContext, TranscriptArtifact } from "@/types/local";

type Embed = (texts: string[]) => Promise<number[][]>;

const cosine = (left: number[], right: number[]) => {
  const dot = left.reduce((sum, value, index) => sum + value * (right[index] ?? 0), 0);
  const norm = Math.sqrt(left.reduce((sum, value) => sum + value * value, 0)) * Math.sqrt(right.reduce((sum, value) => sum + value * value, 0));
  return norm ? dot / norm : 0;
};

const words = (text: string) => text.toLowerCase().match(/[a-z0-9']+/g) || [];
const clamp = (value: number) => Math.max(0, Math.min(100, value));

/** MiniLM similarity is not a percentage. In this application, unrelated
 * answers typically sit below .12 while clearly on-topic answers land near
 * .25 or higher. This calibration makes the displayed value interpretable
 * without treating semantic similarity as a quality grade. */
const topicMatchScore = (similarity: number) => Math.round(clamp(((similarity - 0.05) / 0.3) * 100));

const isContextualContinuation = (text: string, index: number) => {
  if (index === 0 || words(text).length < 5) return false;
  return /\b(this|that|these|those|also|related|project|projects|work|worked|role|experience|study|major|career|education|background|skill|skills)\b/i.test(text);
};

export async function analyzeSemanticResponse(
  transcript: TranscriptArtifact,
  context: LocalSessionContext,
  embed: Embed,
): Promise<Record<string, unknown>> {
  const segments = transcript.segments.filter((item) => item.text.trim());
  if (!transcript.text.trim() || !segments.length) {
    return {
      analysisVersion: "browser-semantic-v2",
      available: false,
      warnings: ["No transcript segments were available for semantic analysis."],
    };
  }

  const question = context.interviewQuestion.trim();
  const roleContext = [context.targetRole, context.jobDescription, context.desiredCompetencies].filter(Boolean).join(" ").trim();
  const contextualIndices = segments.flatMap((segment, index) => (isContextualContinuation(segment.text, index) ? [index] : []));
  const inputs: string[] = [];
  const questionIndex = question ? inputs.push(question) - 1 : null;
  const roleIndex = roleContext ? inputs.push(roleContext) - 1 : null;
  const segmentOffset = inputs.length;
  inputs.push(...segments.map((segment) => segment.text));
  const contextualOffset = inputs.length;
  inputs.push(...contextualIndices.map((index) => `${segments[index - 1].text} ${segments[index].text}`));

  const embeddings = await embed(inputs);
  const questionEmbedding = questionIndex === null ? null : embeddings[questionIndex];
  const roleEmbedding = roleIndex === null ? null : embeddings[roleIndex];
  const contextualEmbeddingBySegment = new Map(
    contextualIndices.map((segmentIndex, index) => [segmentIndex, embeddings[contextualOffset + index]]),
  );

  const assessments = segments.map((segment, index) => {
    const segmentEmbedding = embeddings[segmentOffset + index];
    const directQuestionSimilarity = questionEmbedding ? cosine(questionEmbedding, segmentEmbedding) : null;
    const contextualEmbedding = contextualEmbeddingBySegment.get(index);
    const contextualQuestionSimilarity = questionEmbedding && contextualEmbedding ? cosine(questionEmbedding, contextualEmbedding) : null;
    // A contextual score is available only to a substantive continuation. It
    // lets “projects related to that” inherit the sentence it completes, but
    // does not make a short fragment like “camera, um, yep” look on-topic.
    const questionRelevance = directQuestionSimilarity === null
      ? null
      : Math.max(directQuestionSimilarity, contextualQuestionSimilarity ?? Number.NEGATIVE_INFINITY);
    const roleRelevance = roleEmbedding ? cosine(roleEmbedding, segmentEmbedding) : null;

    return {
      startTime: segment.start,
      endTime: segment.end,
      text: segment.text,
      questionRelevance: questionRelevance === null ? null : Number(questionRelevance.toFixed(3)),
      topicMatchScore: questionRelevance === null ? null : topicMatchScore(questionRelevance),
      directQuestionSimilarity: directQuestionSimilarity === null ? null : Number(directQuestionSimilarity.toFixed(3)),
      contextualQuestionSimilarity: contextualQuestionSimilarity === null ? null : Number(contextualQuestionSimilarity.toFixed(3)),
      relevanceBasis: contextualQuestionSimilarity !== null && questionRelevance === contextualQuestionSimilarity ? "local context + segment" : "segment only",
      roleRelevance: roleRelevance === null ? null : Number(roleRelevance.toFixed(3)),
    };
  });

  const relevance = assessments.flatMap((item) => (typeof item.questionRelevance === "number" ? [item.questionRelevance] : []));
  const topicScores = assessments.flatMap((item) => (typeof item.topicMatchScore === "number" ? [item.topicMatchScore] : []));
  const highestTopicMatch = topicScores.length ? Math.max(...topicScores) : 0;
  const redundantPairs: Array<Record<string, unknown>> = [];
  for (let first = 0; first < assessments.length; first += 1) {
    for (let second = first + 1; second < assessments.length; second += 1) {
      const similarity = cosine(embeddings[segmentOffset + first], embeddings[segmentOffset + second]);
      if (similarity >= 0.88) {
        redundantPairs.push({
          firstStartTime: assessments[first].startTime,
          secondStartTime: assessments[second].startTime,
          similarity: Number(similarity.toFixed(3)),
        });
      }
    }
  }

  return {
    analysisVersion: "browser-semantic-v2",
    available: true,
    // This raw similarity remains for the answer-quality rubric, which applies
    // its own documented calibration. The UI uses topicMatchScore instead.
    questionRelevance: relevance.length ? Number((relevance.reduce((sum, value) => sum + value, 0) / relevance.length).toFixed(3)) : null,
    questionTopicMatchScore: topicScores.length ? Math.round(topicScores.reduce((sum, value) => sum + value, 0) / topicScores.length) : null,
    roleContextRelevance: roleEmbedding
      ? Number((assessments.reduce((sum, item) => sum + Number(item.roleRelevance ?? 0), 0) / assessments.length).toFixed(3))
      : null,
    strongestRelevantSegments: assessments
      .filter((item) => item.topicMatchScore !== null && item.topicMatchScore >= Math.max(65, highestTopicMatch - 15))
      .map((item) => ({ ...item, marker: "mostRelevant" })),
    vagueSegments: assessments
      .filter((item) => item.topicMatchScore !== null && item.topicMatchScore <= 25)
      .map((item) => ({ ...item, marker: "possibleTopicDrift", vagueOrOffTopic: true })),
    segmentAssessments: assessments,
    semanticRedundancy: redundantPairs,
    warnings: [],
  };
}
