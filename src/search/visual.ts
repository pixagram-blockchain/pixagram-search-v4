// Visual similarity and near-duplicates.
//
// /duplicates: exact pHash search. Band index (4 × 16 bits) for radius ≤ 15 — guaranteed to find
// every hash within the radius — and an in-SQL popcount scan beyond that. Each hit also reports
// its dHash distance as a second opinion.
//
// /similar: hybrid score over the union of the SigLIP neighbours and the pHash neighbourhood:
//   0.65 · semantic + 0.15 · pHash + 0.08 · dHash + 0.07 · colour + 0.05 · geometry
// v4.8.2: an artwork without a stored vector (its embed stage has not succeeded yet) is embedded on
// demand (src/enrich/on-demand.ts). When even that is impossible (the Space is down), the
// neighbours come from shared concepts, colours, shape and pHash, the concepts standing in for
// the semantic part ("features"), instead of an empty list.

import type { Env } from "../env";
import { bandProbes, hamming, MAX_INDEXED_RADIUS, sqlHamming } from "../enrich/phash";
import { imageVectorNow } from "../enrich/on-demand";
import { vectorId } from "../db/posts";
import type { SearchRequest } from "./params";
import { hydrateRows, rowToItem, runClause, type SearchItem } from "./service";
import { knn } from "./vectors";

export interface DuplicateHit {
  id: number;
  distance: number;
  dhash_distance: number | null;
}

/** Ids of artworks within `maxDistance` of `hash` (exact), nearest first. */
export async function phashNeighbours(env: Env, hash: string, maxDistance: number, limit: number, excludeId?: number): Promise<{ hits: DuplicateHit[]; method: "bands" | "scan" }> {
  let rows: Array<{ id: number; phash: string; dhash: string | null }>;
  let method: "bands" | "scan";
  // The exact distance is computed in SQL as well, so LIMIT only ever cuts true matches (on a
  // large corpus the band probes also return thousands of hashes that merely share one band).
  const d = sqlHamming("a.phash_hi", "a.phash_lo", hash);
  const cap = Math.max(limit * 4, 200);
  if (maxDistance <= MAX_INDEXED_RADIUS) {
    method = "bands";
    const probes = bandProbes(hash, maxDistance);
    // Values are integers we computed: inlined (a probe can hold hundreds of them, above D1's
    // 100 bound-parameter limit).
    const where = probes.map((p) => `(b.band = ${p.band} AND b.val IN (${p.values.join(",")}))`).join(" OR ");
    rows = await runClause(env.DB, {
      sql: `SELECT DISTINCT b.post_id AS id, a.phash, a.dhash, ${d} AS distance FROM phash_bands b
            JOIN artworks a ON a.post_id = b.post_id JOIN posts p ON p.id = b.post_id
            WHERE p.deleted = 0 AND (${where}) AND a.phash_hi IS NOT NULL AND ${d} <= ?
            ORDER BY distance LIMIT ?`,
      params: [maxDistance, cap],
    });
  } else {
    method = "scan";
    rows = await runClause(env.DB, {
      sql: `SELECT a.post_id AS id, a.phash, a.dhash FROM artworks a JOIN posts p ON p.id = a.post_id
            WHERE p.deleted = 0 AND a.phash_hi IS NOT NULL AND ${d} <= ? ORDER BY ${d} LIMIT ?`,
      params: [maxDistance, cap],
    });
  }
  return { hits: rankByHamming(rows, hash, maxDistance, limit, excludeId), method };
}

function rankByHamming(rows: Array<{ id: number; phash: string; dhash?: string | null }>, hash: string, maxDistance: number, limit: number, excludeId?: number, dh?: string | null): DuplicateHit[] {
  return rows
    .filter((x) => x.id !== excludeId && x.phash)
    .map((x) => ({ id: x.id, distance: hamming(hash, x.phash), dhash_distance: dh && x.dhash ? hamming(dh, x.dhash) : null }))
    .filter((x) => x.distance <= maxDistance)
    .sort((a, b) => a.distance - b.distance || (a.dhash_distance ?? 64) - (b.dhash_distance ?? 64) || a.id - b.id)
    .slice(0, limit);
}

/**
 * Artworks within `maxDistance` of a pHash. `r` filters them like a search (nsfw, author, type…);
 * the API always passes one, so NSFW copies stay out unless the caller asks with nsfw=include.
 */
export async function duplicatesOfHash(
  env: Env,
  hash: string,
  maxDistance: number,
  limit: number,
  excludeId?: number,
  dhashOf?: string | null,
  r: SearchRequest | null = null,
): Promise<{ items: Array<SearchItem & { distance: number; dhash_distance: number | null }>; phash: string; method: "bands" | "scan" }> {
  const { hits, method } = await phashNeighbours(env, hash, maxDistance, r ? Math.max(limit * 4, 100) : limit, excludeId);
  const rows = await hydrateRows(env.DB, hits.map((h) => h.id), r);
  const items = hits
    .filter((h) => rows.has(h.id))
    .slice(0, limit)
    .map((h) => {
      const row = rows.get(h.id)!;
      return { ...rowToItem(row), distance: h.distance, dhash_distance: dhashOf && row.dhash ? hamming(dhashOf, row.dhash) : null };
    });
  return { items, phash: hash, method };
}

export async function duplicates(env: Env, postId: number, maxDistance: number, limit: number, r: SearchRequest | null = null) {
  const me = await env.DB.prepare("SELECT phash, dhash FROM artworks WHERE post_id = ?").bind(postId).first<{ phash: string | null; dhash: string | null }>();
  if (!me?.phash) return { items: [], phash: null as string | null, method: "bands" as const };
  return duplicatesOfHash(env, me.phash, maxDistance, limit, postId, me.dhash, r);
}

// ---- hybrid similarity ----------------------------------------------------------------------------

export const SIMILAR_WEIGHTS = { semantic: 0.65, phash: 0.15, dhash: 0.08, color: 0.07, geometry: 0.05 };

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);

/** SigLIP image↔image cosine → 0..1 (on Pixagram: median pair 0.57, 99th percentile 0.77, re-post ~0.99). */
export function imageSimilarity(cos: number): number {
  return clamp01((cos - 0.55) / 0.4);
}

/** Histogram intersection of the named-colour buckets. */
export function colorSimilarity(a: Array<{ name: string; weight: number }>, b: Array<{ name: string; weight: number }>): number {
  const m = new Map(a.map((x) => [x.name, x.weight]));
  let s = 0;
  for (const x of b) s += Math.min(m.get(x.name) ?? 0, x.weight);
  return clamp01(s);
}

export function geometrySimilarity(a: { aspect: number | null; width: number | null; height: number | null }, b: { aspect: number | null; width: number | null; height: number | null }): number {
  const ra = a.aspect ?? (a.width && a.height ? a.width / a.height : 1);
  const rb = b.aspect ?? (b.width && b.height ? b.width / b.height : 1);
  const shape = clamp01(1 - Math.abs(Math.log(ra / rb)) / Math.log(3));
  const sa = Math.max(a.width ?? 0, a.height ?? 0);
  const sb = Math.max(b.width ?? 0, b.height ?? 0);
  const size = sa && sb ? clamp01(1 - Math.abs(Math.log(sa / sb)) / Math.log(8)) : 0.5;
  return 0.7 * shape + 0.3 * size;
}

export interface SimilarScore {
  visual: number;
  semantic: number | null;
  /** shared concepts (0..1): the semantic part's stand-in when the artwork has no vector */
  theme?: number;
  phash: number | null;
  dhash: number | null;
  color: number;
  geometry: number;
}

/**
 * `cos`: SigLIP cosine with the artwork, null without a vector. `theme`: the share of the
 * artwork's concepts the other one carries (conceptNeighbours), used in the semantic part's
 * place when there is no cosine; null when the artwork has no concepts either.
 */
export function similarityScore(cos: number | null, me: any, other: any, theme: number | null = null): SimilarScore {
  const pd = me.phash && other.phash ? hamming(me.phash, other.phash) : null;
  const dd = me.dhash && other.dhash ? hamming(me.dhash, other.dhash) : null;
  const sem = cos === null ? null : imageSimilarity(cos);
  const th = sem === null && theme !== null ? clamp01(theme) : null;
  const ph = pd === null ? null : clamp01(1 - pd / 32);
  const dh = dd === null ? null : clamp01(1 - dd / 32);
  const color = colorSimilarity(parse(me.buckets_json), parse(other.buckets_json));
  const geometry = geometrySimilarity(me, other);
  const parts: Array<[number, number | null]> = [
    [SIMILAR_WEIGHTS.semantic, sem ?? th],
    [SIMILAR_WEIGHTS.phash, ph],
    [SIMILAR_WEIGHTS.dhash, dh],
    [SIMILAR_WEIGHTS.color, color],
    [SIMILAR_WEIGHTS.geometry, geometry],
  ];
  const wsum = parts.reduce((s, [w, v]) => s + (v === null ? 0 : w), 0) || 1;
  const visual = parts.reduce((s, [w, v]) => s + (v === null ? 0 : w * v), 0) / wsum;
  const out: SimilarScore = { visual, semantic: sem, phash: ph, dhash: dh, color, geometry };
  if (th !== null) out.theme = th;
  return out;
}

function parse(s: string | null): Array<{ name: string; weight: number }> {
  try {
    const v = JSON.parse(s ?? "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

/**
 * Artworks about the same things as `postId`, by the concepts they share (artwork_concepts: tags,
 * title, description and the AI description), each weighted by how rare it is (idf), so that a
 * concept on nearly every artwork counts for little. Score 0..1: the share of the artwork's own
 * concept weight the other one carries. The semantic part's stand-in when there is no vector.
 * Scored in SQL over every artwork that shares a concept: nothing is cut before the weighting.
 */
export async function conceptNeighbours(env: Env, postId: number, limit: number): Promise<Map<number, number>> {
  const mine =
    (await env.DB.prepare("SELECT concept, confidence FROM artwork_concepts WHERE post_id = ? ORDER BY confidence DESC, concept LIMIT 60").bind(postId).all<{ concept: string; confidence: number }>()).results ?? [];
  if (!mine.length) return new Map();
  const [total, df] = await env.DB.batch<{ n: number; concept?: string }>([
    env.DB.prepare("SELECT COUNT(*) AS n FROM posts WHERE deleted = 0 AND type = 'artwork'"),
    env.DB.prepare(`SELECT concept, COUNT(*) AS n FROM artwork_concepts WHERE concept IN (${mine.map(() => "?").join(",")}) GROUP BY concept`).bind(...mine.map((m) => m.concept)),
  ]);
  const n = Math.max(1, Number(total.results?.[0]?.n ?? 1));
  const dfOf = new Map((df.results ?? []).map((x) => [String(x.concept), Number(x.n)]));
  const idf = (c: string) => Math.log(1 + n / Math.max(1, dfOf.get(c) ?? 1));
  // the most telling concepts: rare and confident ones (3 values each: 24 stay under D1's 100)
  const keep = mine.map((m) => ({ ...m, idf: idf(m.concept) })).sort((a, b) => b.idf * b.confidence - a.idf * a.confidence).slice(0, 24);
  const norm = keep.reduce((s, m) => s + m.idf * m.confidence, 0) || 1;
  const rows =
    (
      await env.DB
        .prepare(
          `WITH w(concept, conf, idf) AS (VALUES ${keep.map(() => "(?, ?, ?)").join(", ")})
           SELECT c.post_id AS id, SUM(w.idf * MIN(w.conf, c.confidence)) AS s
           FROM w JOIN artwork_concepts c ON c.concept = w.concept JOIN posts p ON p.id = c.post_id
           WHERE p.deleted = 0 AND p.type = 'artwork' AND c.post_id != ?
           GROUP BY c.post_id ORDER BY s DESC, c.post_id LIMIT ?`,
        )
        .bind(...keep.flatMap((k) => [k.concept, k.confidence, k.idf]), postId, limit)
        .all<{ id: number; s: number }>()
    ).results ?? [];
  return new Map(rows.map((x) => [Number(x.id), Math.min(1, Number(x.s) / norm)] as [number, number]));
}

/** Artworks with the most colour in common with `postId` (histogram intersection of the named buckets). */
async function colourNeighbours(env: Env, postId: number, limit: number): Promise<number[]> {
  const rows =
    (
      await env.DB
        .prepare(
          `SELECT b.post_id AS id, SUM(MIN(a.weight, b.weight)) AS inter FROM artwork_colors a
           JOIN artwork_colors b ON b.bucket = a.bucket AND b.post_id != a.post_id
           JOIN posts p ON p.id = b.post_id
           WHERE a.post_id = ? AND p.deleted = 0 AND p.type = 'artwork'
           GROUP BY b.post_id ORDER BY inter DESC, b.post_id LIMIT ?`,
        )
        .bind(postId, limit)
        .all<{ id: number }>()
    ).results ?? [];
  return rows.map((x) => Number(x.id));
}

/**
 * How the neighbours were found. "hybrid": by the artwork's SigLIP vector (`vector` says whether
 * it was the stored one or computed for this request), with pHash; "features": no vector could be
 * had (the Space is down), so by shared concepts, colours, shape and pHash; "none": nothing found.
 */
export type SimilarMethod = "hybrid" | "features" | "none";

export async function similar(
  env: Env,
  postId: number,
  limit: number,
  r: SearchRequest | null,
  o: { onDemand?: boolean; allowCompute?: () => Promise<boolean> } = {},
): Promise<{ items: Array<SearchItem & { similarity?: SimilarScore }>; method: SimilarMethod; vector: "stored" | "computed" | "none"; note?: string }> {
  const meRows = await hydrateRows(env.DB, [postId], null);
  const me = meRows.get(postId);
  if (!me || me.type !== "artwork") return { items: [], method: "none", vector: "none" };
  let indexDown: string | null = null;
  let got: Awaited<ReturnType<VectorizeIndex["getByIds"]>> = [];
  try {
    got = await env.VEC.getByIds([vectorId(postId)]);
  } catch (e) {
    indexDown = e instanceof Error ? e.message : String(e);
  }
  let vec = got[0]?.values ? Array.from(got[0].values as ArrayLike<number>) : null;
  let vector: "stored" | "computed" | "none" = vec ? "stored" : "none";
  let note: string | undefined;
  if (indexDown) {
    // the vector index is failing: its neighbours could not be read either
    note = `vector index unavailable: ${indexDown}`;
  } else if (!vec && o.onDemand !== false) {
    // `allowCompute`: the caller's rate limit, asked only when a vector is about to be computed
    const fresh = await imageVectorNow(env, postId, { allow: o.allowCompute }).catch((e: unknown) => ({ vector: null, reason: e instanceof Error ? e.message : String(e) }) as const);
    if (fresh.vector) (vec = fresh.vector), (vector = "computed");
    else note = `no image vector: ${fresh.reason}`;
  }
  const cos = new Map<number, number>();
  if (vec) for (const h of await knn(env, "image", vec, r, 100)) cos.set(h.id, h.score);
  // without a vector: the artworks about the same things, and those with the same colours
  const [near, themes, colours] = await Promise.all([
    me.phash ? phashNeighbours(env, me.phash, 12, 100, postId).then((x) => x.hits) : Promise.resolve([] as DuplicateHit[]),
    vec ? Promise.resolve(new Map<number, number>()) : conceptNeighbours(env, postId, 100),
    vec ? Promise.resolve([] as number[]) : colourNeighbours(env, postId, 50),
  ]);
  const ids = [...new Set([...cos.keys(), ...near.map((x) => x.id), ...themes.keys(), ...colours])].filter((id) => id !== postId);
  const base = { vector, ...(note ? { note } : {}) };
  if (!ids.length) return { items: [], method: "none", ...base };
  const rows = await hydrateRows(env.DB, ids, r);
  // A pHash neighbour outside the vector top-100 has a cosine at most the 100th one: use that floor.
  const floor = cos.size >= 100 ? Math.min(...cos.values()) : null;
  const scored = [...rows.values()]
    .filter((row) => row.type === "artwork")
    .map((row) => ({ row, s: vec ? similarityScore(cos.get(row.id) ?? floor, me, row) : similarityScore(null, me, row, themes.size ? (themes.get(row.id) ?? 0) : null) }))
    .sort((a, b) => b.s.visual - a.s.visual || a.row.id - b.row.id)
    .slice(0, limit);
  const items = scored.map(({ row, s }) => {
    const it: SearchItem & { similarity?: SimilarScore } = rowToItem(row);
    it.score = { fused: Math.round(s.visual * 10000) / 10000, ranks: {} };
    it.similarity = Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v === null ? null : Math.round((v as number) * 1000) / 1000])) as unknown as SimilarScore;
    return it;
  });
  return { items, method: vec ? "hybrid" : "features", ...base };
}
