// Cross-encoder reranking (spec §13): a second stage that reads the question and each candidate
// together, after the cheap feature ranker.
//
//   100–300 candidates ─► feature ranker (ranker.ts) ─► top SEARCH_RERANK_K (50) ─► cross-encoder ─► order
//
// The cross-encoder sees text only (title, tags, the author's description, the AI caption, concepts,
// author, date): it cannot see the image, so for /search its score is blended with the feature
// ranker's (which carries the image similarity) rather than replacing it. It never changes a
// candidate's metadata, only its order.
//
// Models: Workers AI @cf/baai/bge-reranker-base (SEARCH_RERANKER_MODEL, default), or "http": a
// Text Embeddings Inference style endpoint at SEARCH_RERANK_URL (POST {query, texts, raw_scores}
// → [{index, score}]), e.g. the same BAAI/bge-reranker-base served locally for the offline
// evaluation (eval/offline/rerank_server.py). Scores are logits; sigmoid gives [0, 1].

import type { Env } from "../env";
import { int, num } from "../env";
import { aiReply } from "../lib/ai";
import { sha256Hex, utf8 } from "../lib/bytes";
import { modelFor } from "../llm/router";
import { parseList } from "./retrieval";
import type { Row } from "./operators/types";

/** How a candidate is written for the cross-encoder; part of the cache key and of the recorded versions. */
export const REPRESENTATION_VERSION = "r1";

export interface RerankerModel {
  id: string;
  /** relevance of each text to the query, in [0, 1], in the texts' order */
  score(query: string, texts: string[]): Promise<number[]>;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const MAX_TEXT = 1200;

/** The candidate as the cross-encoder reads it. */
export function candidateText(row: Row): string {
  const parts: string[] = [];
  if (row.title) parts.push(String(row.title));
  const tags = parseList(row.tags_json);
  if (tags.length) parts.push(`tags: ${tags.slice(0, 12).join(", ")}`);
  if (row.description && String(row.description).trim().length > 2 && row.description !== row.title) parts.push(String(row.description).slice(0, 300));
  if (row.ai_caption) parts.push(String(row.ai_caption).slice(0, 400));
  const aiTags = parseList(row.ai_tags_json);
  if (aiTags.length) parts.push(`keywords: ${aiTags.slice(0, 12).join(", ")}`);
  parts.push(`by @${row.author}`);
  return parts.join(". ").slice(0, MAX_TEXT);
}

class WorkersAiReranker implements RerankerModel {
  constructor(private readonly env: Env, readonly id: string) {}
  async score(query: string, texts: string[]): Promise<number[]> {
    const ai = this.env.AI as unknown as { run: (m: string, i: unknown) => Promise<unknown> };
    const r = await aiReply(await ai.run(this.id, { query: query.slice(0, 500), contexts: texts.map((t) => ({ text: t || " " })), top_k: texts.length }));
    const rows: Array<{ id: number; score: number }> = Array.isArray(r?.response) ? r.response : Array.isArray(r) ? r : [];
    const out = new Array<number>(texts.length).fill(0);
    let seen = 0;
    for (const x of rows) if (Number.isInteger(x?.id) && x.id >= 0 && x.id < texts.length && Number.isFinite(x.score)) (out[x.id] = sigmoid(x.score), seen++);
    if (!seen) throw new Error(`reranker ${this.id}: no scores in the reply`);
    return out;
  }
}

class HttpReranker implements RerankerModel {
  readonly id: string;
  constructor(private readonly url: string) {
    this.id = `http:${new URL(url).host}`;
  }
  async score(query: string, texts: string[]): Promise<number[]> {
    const res = await fetch(this.url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ query: query.slice(0, 500), texts, raw_scores: true, truncate: true }), signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`reranker ${this.id}: HTTP ${res.status}`);
    const rows = (await res.json()) as Array<{ index: number; score: number }>;
    const out = new Array<number>(texts.length).fill(0);
    for (const x of rows) if (Number.isInteger(x?.index) && x.index >= 0 && x.index < texts.length) out[x.index] = sigmoid(x.score);
    return out;
  }
}

/** The configured cross-encoder, or null when reranking is off or none is available. */
export function rerankerModel(env: Env): RerankerModel | null {
  const id = modelFor(env, "reranker");
  if (id === "off" || id === "none") return null;
  if (id === "http") return env.SEARCH_RERANK_URL ? new HttpReranker(env.SEARCH_RERANK_URL) : null;
  return env.AI ? new WorkersAiReranker(env, id) : null;
}

export function rerankerVersion(env: Env): string | null {
  const m = rerankerModel(env);
  return m ? `${m.id}@${REPRESENTATION_VERSION}` : null;
}

/** Cross-encoder scores of texts for a query, in their order, cached (spec §50) by model, representation and content. */
export async function rerankTexts(env: Env, query: string, texts: string[], opts: { ttl?: number; representation?: string } = {}): Promise<{ scores: number[]; model: string; cached: boolean; ms: number } | null> {
  const model = rerankerModel(env);
  if (!model || !texts.length || !query.trim()) return null;
  const t0 = Date.now();
  const key = `rr4:${(await sha256Hex(utf8(JSON.stringify([model.id, opts.representation ?? REPRESENTATION_VERSION, query.trim().toLowerCase(), texts])))).slice(0, 40)}`;
  const hit = (await env.CACHE.get(key, "json").catch(() => null)) as number[] | null;
  let scores = Array.isArray(hit) && hit.length === texts.length ? hit : null;
  const cached = !!scores;
  if (!scores) {
    scores = await model.score(query, texts);
    await env.CACHE.put(key, JSON.stringify(scores), { expirationTtl: Math.max(60, opts.ttl ?? 3600) }).catch(() => {});
  }
  return { scores, model: model.id, cached, ms: Date.now() - t0 };
}

/** Cross-encoder scores of candidate rows for a query, by post id. */
export async function rerankRows(env: Env, query: string, rows: Row[], opts: { ttl?: number } = {}): Promise<{ scores: Map<number, number>; model: string; cached: boolean; ms: number } | null> {
  const r = await rerankTexts(env, query, rows.map(candidateText), opts);
  return r ? { ...r, scores: new Map(rows.map((row, i) => [row.id as number, r.scores[i]])) } : null;
}

/** How many candidates the cross-encoder reads (SEARCH_RERANK_K) and the /search blend weight (SEARCH_RERANK_BLEND). */
export const rerankDepth = (env: Env) => Math.min(100, Math.max(5, int(env.SEARCH_RERANK_K, 50)));
// Measured offline on the 160 judged queries (README-V4): bge-reranker-base alone ranks far below
// the feature ranker (nDCG@10 0.74–0.79 vs 0.953: it reads no image, and little besides English);
// blended at 0.1–0.2 it is neutral (0.953), from 0.3 on it costs (0.951, 0.948 at 0.5).
export const rerankBlend = (env: Env) => Math.min(1, Math.max(0, num(env.SEARCH_RERANK_BLEND, 0.2)));

/**
 * Blend the feature ranker's order with the cross-encoder: the top `k` items get
 * (1 − β) · score / best score + β · cross-encoder score, and keep their place above the rest.
 */
export function blendOrder<T extends { id: number; final: number }>(ranked: T[], ce: Map<number, number>, beta: number, k: number): Array<T & { rerank?: number; blended?: number }> {
  const head = ranked.slice(0, k);
  const tail = ranked.slice(k);
  const best = Math.max(1e-9, ...head.map((r) => r.final));
  const scored = head.map((r) => {
    const c = ce.get(r.id);
    const blended = c === undefined ? r.final / best : (1 - beta) * (r.final / best) + beta * c;
    return { ...r, rerank: c, blended };
  });
  scored.sort((a, b) => b.blended! - a.blended! || a.id - b.id);
  return [...scored, ...tail];
}
