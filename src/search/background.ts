// Background sample for semantic score normalisation.
//
// SigLIP cosines (and its own calibrated probabilities) are not comparable across queries on
// pixel art: "a cat" peaks at p = 0.01 for real cats while "mount fuji" reaches 0.3. What is
// stable is how far a match stands out from the corpus: z = (cos − median) / (1.4826 · MAD), the
// median and MAD taken over a fixed random sample of the index (256 vectors, refreshed by cron).
// On the live corpus the real subject scores z ≈ 4–9 and unrelated artworks stay below ≈ 3.5.
// Vectorize only returns the top 100, so the sample is what makes the distribution visible.

import type { Env } from "../env";
import { base64Decode, base64Encode } from "../lib/bytes";
import { vectorId } from "../db/posts";
import { dot } from "../enrich/embed";
import type { VectorKind } from "./vectors";

export interface Norm {
  median: number;
  scale: number; // 1.4826 · MAD
}

const KEY = (env: Env, kind: VectorKind) => `bg:${kind}:${env.EMBED_MODEL ?? "m"}`;
const memo = new Map<string, { at: number; vectors: Float32Array[] }>();

export async function loadBackground(env: Env, kind: VectorKind): Promise<Float32Array[] | null> {
  const key = KEY(env, kind);
  const m = memo.get(key);
  if (m && Date.now() - m.at < 10 * 60_000) return m.vectors;
  const raw = (await env.CACHE.get(key, "json").catch(() => null)) as { dim: number; b64: string } | null;
  if (!raw?.b64) return null;
  const bytes = base64Decode(raw.b64);
  const all = new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
  const vectors: Float32Array[] = [];
  for (let i = 0; i + raw.dim <= all.length; i += raw.dim) vectors.push(all.slice(i, i + raw.dim));
  memo.set(key, { at: Date.now(), vectors });
  return vectors;
}

/** Re-sample the background: up to n random vectors of the index (cron, or on first use). */
export async function refreshBackground(env: Env, kind: VectorKind, n = 256): Promise<number> {
  const index = kind === "image" ? env.VEC : env.VEC_TEXT;
  if (!index) return 0;
  const sql =
    kind === "image"
      ? "SELECT a.post_id AS id FROM artworks a JOIN posts p ON p.id = a.post_id WHERE p.deleted = 0 AND a.embed_hash = a.content_hash ORDER BY RANDOM() LIMIT ?"
      : "SELECT id FROM posts WHERE deleted = 0 AND text_hash IS NOT NULL ORDER BY RANDOM() LIMIT ?";
  const ids = ((await env.DB.prepare(sql).bind(n).all<{ id: number }>()).results ?? []).map((r) => vectorId(r.id));
  const vecs: number[][] = [];
  for (let i = 0; i < ids.length; i += 20) {
    const got = await index.getByIds(ids.slice(i, i + 20));
    for (const v of got) if (v.values) vecs.push(Array.from(v.values as ArrayLike<number>));
  }
  if (vecs.length < 16) return vecs.length; // too few to estimate a distribution
  const dim = vecs[0].length;
  const flat = new Float32Array(vecs.length * dim);
  vecs.forEach((v, k) => flat.set(v, k * dim));
  await env.CACHE.put(KEY(env, kind), JSON.stringify({ dim, b64: base64Encode(new Uint8Array(flat.buffer)), at: Date.now() }), { expirationTtl: 3 * 24 * 3600 });
  memo.delete(KEY(env, kind));
  return vecs.length;
}

export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Median and robust scale of the query's cosines against the background sample. */
export function normFor(query: ArrayLike<number>, sample: Float32Array[]): Norm | null {
  if (sample.length < 16) return null;
  const cos = sample.map((v) => dot(query, v));
  const med = median(cos);
  const mad = median(cos.map((c) => Math.abs(c - med)));
  return { median: med, scale: Math.max(1e-4, 1.4826 * mad) };
}

export function zScore(cos: number, n: Norm): number {
  return (cos - n.median) / n.scale;
}

/** Background norm for a query vector, or null when no sample exists yet (refreshed in the background). */
export async function queryNorm(env: Env, kind: VectorKind, query: number[] | null, exec?: ExecutionContext): Promise<Norm | null> {
  if (!query) return null;
  const sample = await loadBackground(env, kind).catch(() => null);
  if (!sample) {
    exec?.waitUntil(refreshBackground(env, kind).catch(() => 0));
    return null;
  }
  return normFor(query, sample);
}
