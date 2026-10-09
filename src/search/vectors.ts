// Vectorize queries: metadata filters mirroring the SQL filters each index can express, plain kNN,
// and time-sliced kNN for wider recall than Vectorize's top-100.

import type { Env } from "../env";
import type { SearchRequest } from "./params";

export type VectorKind = "image" | "text";

export interface Hit {
  id: number;
  score: number;
}

/**
 * Metadata filter for one index. Indexed properties: VEC — author, primary_color, size_class,
 * created, color_count, nsfw, listed, ai_training, orientation, transparent; VEC_TEXT — the same
 * with type instead of color_count. Everything else is re-applied when candidates are hydrated.
 */
export function vectorFilter(r: SearchRequest, kind: VectorKind = "image", range?: { from?: number; to?: number }): VectorizeVectorMetadataFilter | undefined {
  const f: Record<string, any> = {};
  if (r.authors.length) f.author = r.authors.length === 1 ? r.authors[0] : { $in: r.authors };
  if (r.colors.length) f.primary_color = r.colors.length === 1 ? r.colors[0] : { $in: r.colors };
  if (r.sizes.length) f.size_class = r.sizes.length === 1 ? r.sizes[0] : { $in: r.sizes };
  const from = Math.max(r.from ?? -Infinity, range?.from ?? -Infinity);
  const to = Math.min(r.to ?? Infinity, range?.to ?? Infinity);
  if (Number.isFinite(from) || Number.isFinite(to)) {
    const c: Record<string, number> = {};
    if (Number.isFinite(from)) c.$gte = from;
    if (Number.isFinite(to)) c.$lt = to;
    f.created = c;
  }
  if (kind === "image" && (r.minColors !== null || r.maxColors !== null)) {
    const c: Record<string, number> = {};
    if (r.minColors !== null) c.$gte = r.minColors;
    if (r.maxColors !== null) c.$lte = r.maxColors;
    f.color_count = c;
  }
  if (kind === "text" && r.type) f.type = r.type;
  if (r.nsfw === "exclude") f.nsfw = false;
  else if (r.nsfw === "only") f.nsfw = true;
  if (r.listed !== null) f.listed = r.listed;
  if (r.aiTraining !== null) f.ai_training = r.aiTraining;
  if (r.orientation.length) f.orientation = r.orientation.length === 1 ? r.orientation[0] : { $in: r.orientation };
  if (r.transparent !== null) f.transparent = r.transparent;
  return Object.keys(f).length ? (f as VectorizeVectorMetadataFilter) : undefined;
}

export function indexFor(env: Env, kind: VectorKind): VectorizeIndex | undefined {
  return kind === "image" ? env.VEC : env.VEC_TEXT;
}

export async function knn(env: Env, kind: VectorKind, vector: number[], r: SearchRequest | null, topK: number, range?: { from?: number; to?: number }): Promise<Hit[]> {
  const index = indexFor(env, kind);
  if (!index) return [];
  const res = await index.query(vector, {
    topK: Math.min(100, Math.max(1, topK)),
    filter: r ? vectorFilter(r, kind, range) : range ? vectorFilter(emptyLike(), kind, range) : undefined,
    returnValues: false,
    returnMetadata: "none",
  });
  return res.matches.map((m) => ({ id: Number(m.id), score: m.score })).filter((m) => Number.isFinite(m.id));
}

function emptyLike(): SearchRequest {
  return {
    q: "", type: null, authors: [], tags: [], colors: [], hasColors: [], minColorWeight: 0.08, sizes: [], minColors: null, maxColors: null,
    minWidth: null, maxWidth: null, minHeight: null, maxHeight: null, from: null, to: null, transparent: null, nsfw: "include", listed: null,
    aiTraining: null, orientation: [], monochrome: null, background: [], concepts: [], sort: "relevance", limit: 24, cursor: null, facets: false,
    semantic: true, rank: "v3", expand: true, explain: false, rerank: false,
  };
}

/** Split [from, to) into n equal slices. */
export function timeSlices(from: number, to: number, n: number): Array<{ from: number; to: number }> {
  const out: Array<{ from: number; to: number }> = [];
  const step = Math.max(1, Math.ceil((to - from) / n));
  for (let s = from; s < to; s += step) out.push({ from: s, to: Math.min(to, s + step) });
  return out;
}

/**
 * kNN per time slice, merged: each slice returns its own top-K, so n slices reach up to n×100
 * candidates spread over the whole period. Used to widen recall under restrictive filters and by
 * /ask for "first/last X" questions, where the earliest match must be found wherever it is.
 */
export async function slicedKnn(env: Env, kind: VectorKind, vector: number[], r: SearchRequest | null, slices: Array<{ from: number; to: number }>, topK = 100): Promise<Hit[]> {
  const parts = await Promise.all(slices.map((s) => knn(env, kind, vector, r, topK, s).catch(() => [] as Hit[])));
  const best = new Map<number, number>();
  for (const p of parts) for (const h of p) best.set(h.id, Math.max(best.get(h.id) ?? -Infinity, h.score));
  return [...best.entries()].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score);
}

/**
 * Exhaustive-as-possible kNN over a period, for /ask ("how many", "the first"): start with
 * `slices` time slices; a slice that comes back full (topK hits: it may hold more matches) is split
 * in four and searched again, until no slice is full or `budget` queries are spent. `truncated`
 * says whether some period still came back full, i.e. whether a count over the hits is a lower
 * bound.
 */
export async function adaptiveKnn(
  env: Env,
  kind: VectorKind,
  vector: number[],
  r: SearchRequest | null,
  from: number,
  to: number,
  opts: { slices: number; budget: number; topK?: number },
): Promise<{ hits: Hit[]; truncated: boolean; queries: number }> {
  const topK = opts.topK ?? 100;
  const best = new Map<number, number>();
  let pending = timeSlices(from, to, Math.max(1, opts.slices));
  let queries = 0;
  let truncated = false;
  while (pending.length) {
    const batch = pending;
    pending = [];
    const parts = await Promise.all(batch.map((s) => knn(env, kind, vector, r, topK, s).then((h) => ({ s, h }))));
    queries += batch.length;
    for (const { s, h } of parts) {
      for (const x of h) best.set(x.id, Math.max(best.get(x.id) ?? -Infinity, x.score));
      if (h.length < topK) continue;
      // Full: this period may hold more matches. Split it if the budget allows and it can be split.
      if (s.to - s.from >= 4 && queries + pending.length + 4 <= opts.budget) pending.push(...timeSlices(s.from, s.to, 4));
      else truncated = true;
    }
  }
  return { hits: [...best.entries()].map(([id, score]) => ({ id, score })).sort((a, b) => b.score - a.score), truncated, queries };
}
