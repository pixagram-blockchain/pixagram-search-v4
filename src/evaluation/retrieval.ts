// Retrieval metrics (spec §33): Recall@K, Precision@K, MRR, nDCG@K and (average) precision over a
// ranked list of ids against graded relevance judgments (0 = not relevant, 1 = relevant, 2 = the
// answer). The gain of a grade g is 2^g − 1, as in eval/offline (v3's published numbers).

export type Judgments = Record<string, number>;

export interface RankMetrics {
  ndcg10: number;
  p10: number;
  r10: number;
  r50: number;
  mrr: number;
  /** average precision (MAP once averaged over queries) */
  ap: number;
}

/** nDCG@k of a ranking. */
export function ndcgAt(ranked: string[], rel: Judgments, k = 10): number {
  const gains = ranked.slice(0, k).map((r) => rel[r] ?? 0);
  const dcg = gains.reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  const ideal = Object.values(rel).sort((a, b) => b - a).slice(0, k);
  const idcg = ideal.reduce((s, g, i) => s + (2 ** g - 1) / Math.log2(i + 2), 0);
  return idcg ? dcg / idcg : 0;
}

export function recallAt(ranked: string[], rel: Judgments, k: number): number {
  const relevant = Object.entries(rel).filter(([, g]) => g > 0).map(([id]) => id);
  if (!relevant.length) return 0;
  const top = new Set(ranked.slice(0, k));
  return relevant.filter((id) => top.has(id)).length / relevant.length;
}

export function precisionAt(ranked: string[], rel: Judgments, k: number): number {
  return ranked.slice(0, k).filter((id) => (rel[id] ?? 0) > 0).length / k;
}

export function reciprocalRank(ranked: string[], rel: Judgments): number {
  const i = ranked.findIndex((id) => (rel[id] ?? 0) > 0);
  return i < 0 ? 0 : 1 / (i + 1);
}

export function averagePrecision(ranked: string[], rel: Judgments): number {
  const nrel = Object.values(rel).filter((g) => g > 0).length;
  if (!nrel) return 0;
  let hits = 0;
  let sum = 0;
  ranked.forEach((id, i) => {
    if ((rel[id] ?? 0) > 0) sum += ++hits / (i + 1);
  });
  return sum / nrel;
}

export function rankMetrics(ranked: string[], rel: Judgments): RankMetrics {
  return { ndcg10: ndcgAt(ranked, rel, 10), p10: precisionAt(ranked, rel, 10), r10: recallAt(ranked, rel, 10), r50: recallAt(ranked, rel, 50), mrr: reciprocalRank(ranked, rel), ap: averagePrecision(ranked, rel) };
}

/** The mean of each metric over queries (MAP is the mean of ap). */
export function meanMetrics(rows: RankMetrics[]): RankMetrics {
  const keys: Array<keyof RankMetrics> = ["ndcg10", "p10", "r10", "r50", "mrr", "ap"];
  return Object.fromEntries(keys.map((k) => [k, rows.length ? rows.reduce((s, r) => s + r[k], 0) / rows.length : 0])) as unknown as RankMetrics;
}
