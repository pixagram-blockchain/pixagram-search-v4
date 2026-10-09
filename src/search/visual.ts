// Visual similarity and near-duplicates.
//
// /duplicates: exact pHash search. Band index (4 × 16 bits) for radius ≤ 15 — guaranteed to find
// every hash within the radius — and an in-SQL popcount scan beyond that. Each hit also reports
// its dHash distance as a second opinion.
//
// /similar: hybrid score over the union of the SigLIP neighbours and the pHash neighbourhood:
//   0.65 · semantic + 0.15 · pHash + 0.08 · dHash + 0.07 · colour + 0.05 · geometry
// (weights renormalised when the artwork has no vector yet).

import type { Env } from "../env";
import { bandProbes, hamming, MAX_INDEXED_RADIUS, sqlHamming } from "../enrich/phash";
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
  phash: number | null;
  dhash: number | null;
  color: number;
  geometry: number;
}

export function similarityScore(cos: number | null, me: any, other: any): SimilarScore {
  const pd = me.phash && other.phash ? hamming(me.phash, other.phash) : null;
  const dd = me.dhash && other.dhash ? hamming(me.dhash, other.dhash) : null;
  const sem = cos === null ? null : imageSimilarity(cos);
  const ph = pd === null ? null : clamp01(1 - pd / 32);
  const dh = dd === null ? null : clamp01(1 - dd / 32);
  const color = colorSimilarity(parse(me.buckets_json), parse(other.buckets_json));
  const geometry = geometrySimilarity(me, other);
  const parts: Array<[number, number | null]> = [
    [SIMILAR_WEIGHTS.semantic, sem],
    [SIMILAR_WEIGHTS.phash, ph],
    [SIMILAR_WEIGHTS.dhash, dh],
    [SIMILAR_WEIGHTS.color, color],
    [SIMILAR_WEIGHTS.geometry, geometry],
  ];
  const wsum = parts.reduce((s, [w, v]) => s + (v === null ? 0 : w), 0) || 1;
  const visual = parts.reduce((s, [w, v]) => s + (v === null ? 0 : w * v), 0) / wsum;
  return { visual, semantic: sem, phash: ph, dhash: dh, color, geometry };
}

function parse(s: string | null): Array<{ name: string; weight: number }> {
  try {
    const v = JSON.parse(s ?? "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

export async function similar(
  env: Env,
  postId: number,
  limit: number,
  r: SearchRequest | null,
): Promise<{ items: Array<SearchItem & { similarity?: SimilarScore }>; method: "hybrid" | "phash" | "none" }> {
  const meRows = await hydrateRows(env.DB, [postId], null);
  const me = meRows.get(postId);
  if (!me || me.type !== "artwork") return { items: [], method: "none" };
  const got = await env.VEC.getByIds([vectorId(postId)]).catch(() => []);
  const vec = got[0]?.values ? Array.from(got[0].values as ArrayLike<number>) : null;
  const cos = new Map<number, number>();
  if (vec) for (const h of await knn(env, "image", vec, r, 100)) cos.set(h.id, h.score);
  const near = me.phash ? (await phashNeighbours(env, me.phash, 12, 100, postId)).hits : [];
  const ids = [...new Set([...cos.keys(), ...near.map((n) => n.id)])].filter((id) => id !== postId);
  if (!ids.length) return { items: [], method: "none" };
  const rows = await hydrateRows(env.DB, ids, r);
  // A pHash neighbour outside the vector top-100 has a cosine at most the 100th one: use that floor.
  const floor = cos.size >= 100 ? Math.min(...cos.values()) : null;
  const scored = [...rows.values()]
    .filter((row) => row.type === "artwork")
    .map((row) => ({ row, s: similarityScore(cos.get(row.id) ?? (vec ? floor : null), me, row) }))
    .sort((a, b) => b.s.visual - a.s.visual || a.row.id - b.row.id)
    .slice(0, limit);
  const items = scored.map(({ row, s }) => {
    const it: SearchItem & { similarity?: SimilarScore } = rowToItem(row);
    it.score = { fused: Math.round(s.visual * 10000) / 10000, ranks: {} };
    it.similarity = Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v === null ? null : Math.round((v as number) * 1000) / 1000])) as unknown as SimilarScore;
    return it;
  });
  return { items, method: vec ? "hybrid" : "phash" };
}
