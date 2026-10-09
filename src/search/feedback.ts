// Learning-to-rank plumbing. Every /search answer carries a query_id; the client reports what the
// user did with a result (POST /feedback). rank_log holds the features of what was shown, so
// (features, action) pairs can be exported and turned into ranker weights offline
// (scripts/train-ranker.py), then installed with POST /admin/ranker/weights. No model has to run
// in the Worker: the learned weights replace the defaults of the same linear scorer.

import type { Env } from "../env";
import { now } from "../env";
import { setSetting, getSetting } from "../db/posts";
import { DEFAULT_WEIGHTS, FEATURES, mergeWeights } from "./ranker";
import { reasoningMessages } from "../llm/prompts";

export const ACTIONS = ["click", "open", "like", "save", "similar", "dwell"] as const;
export type Action = (typeof ACTIONS)[number];

/** Graded label per action (the strongest action of a (query, post) pair wins). */
export const LABEL: Record<Action, number> = { click: 1, open: 1, dwell: 2, similar: 2, save: 3, like: 3 };

export async function recordFeedback(env: Env, body: unknown): Promise<{ ok: boolean; error?: string }> {
  const b = (body ?? {}) as Record<string, unknown>;
  const postId = Number(b.post_id);
  const action = String(b.action ?? "click") as Action;
  if (!Number.isFinite(postId) || postId <= 0) return { ok: false, error: "post_id required" };
  if (!ACTIONS.includes(action)) return { ok: false, error: `action must be one of ${ACTIONS.join(", ")}` };
  const qid = typeof b.query_id === "string" ? b.query_id.slice(0, 32) : "";
  if (!qid) return { ok: false, error: "query_id required (from the /search answer)" };
  const rank = Number.isFinite(Number(b.rank)) ? Math.max(1, Math.min(1000, Number(b.rank))) : null;
  const dwell = Number.isFinite(Number(b.dwell_ms)) ? Math.max(0, Math.min(3_600_000, Number(b.dwell_ms))) : null;
  // Only for a real, recent query and a post it returned, and a bounded number per query: the
  // endpoint is public and its rows feed the ranker's training data.
  const q = await env.DB
    .prepare("SELECT top_json, (SELECT COUNT(*) FROM feedback f WHERE f.qid = l.qid) AS n FROM query_log l WHERE l.qid = ? AND l.at >= ?")
    .bind(qid, now() - 2 * 86400)
    .first<{ top_json: string | null; n: number }>();
  if (!q) return { ok: false, error: "unknown or expired query_id" };
  let shown: number[] = [];
  try {
    shown = JSON.parse(q.top_json ?? "[]");
  } catch {
    shown = [];
  }
  if (!shown.includes(postId)) {
    const logged = await env.DB.prepare("SELECT 1 FROM rank_log WHERE qid = ? AND post_id = ?").bind(qid, postId).first();
    if (!logged) return { ok: false, error: "post_id was not a result of that query" };
  }
  if (q.n >= 50) return { ok: true }; // enough said about this query
  await env.DB.prepare("INSERT INTO feedback (qid, post_id, rank, action, dwell_ms, at) VALUES (?, ?, ?, ?, ?, ?)").bind(qid, postId, rank, action, dwell, now()).run();
  return { ok: true };
}

/**
 * Training rows: every shown result of a logged query with its features and a label (0 = shown,
 * not used; else the strongest action). Only queries with at least one action are exported —
 * a query nobody reacted to says nothing about the order.
 */
export async function exportTraining(env: Env, days: number, limit = 20000): Promise<Array<{ qid: string; q: string; post_id: number; rank: number; label: number; features: Record<string, number> }>> {
  const since = now() - days * 86400;
  const rows = await env.DB
    .prepare(
      `SELECT r.qid, l.q, r.post_id, r.rank, r.features,
              (SELECT MAX(CASE f.action WHEN 'like' THEN 3 WHEN 'save' THEN 3 WHEN 'dwell' THEN 2 WHEN 'similar' THEN 2 ELSE 1 END)
                 FROM feedback f WHERE f.qid = r.qid AND f.post_id = r.post_id) AS label
       FROM rank_log r JOIN query_log l ON l.qid = r.qid
       WHERE r.at >= ? AND EXISTS (SELECT 1 FROM feedback f2 WHERE f2.qid = r.qid)
       ORDER BY r.qid, r.rank LIMIT ?`,
    )
    .bind(since, limit)
    .all<{ qid: string; q: string; post_id: number; rank: number; features: string; label: number | null }>();
  return (rows.results ?? []).map((r) => ({ qid: r.qid, q: r.q, post_id: r.post_id, rank: r.rank, label: r.label ?? 0, features: JSON.parse(r.features) }));
}

export async function getWeights(env: Env) {
  const s = await getSetting(env.DB, "ranker:weights");
  return { active: mergeWeights(s ? JSON.parse(s) : {}), defaults: DEFAULT_WEIGHTS, features: FEATURES };
}

export async function setWeights(env: Env, body: unknown): Promise<{ ok: boolean; weights?: unknown; error?: string }> {
  if (body && typeof body === "object" && (body as any).reset === true) {
    await setSetting(env.DB, "ranker:weights", "{}");
    await env.CACHE.delete("ctx:weights").catch(() => {});
    return { ok: true, weights: DEFAULT_WEIGHTS };
  }
  const merged = mergeWeights(body);
  if (Object.values(merged.w).some((v) => v < 0 || v > 5)) return { ok: false, error: "feature weights must be within 0..5" };
  await setSetting(env.DB, "ranker:weights", JSON.stringify(merged));
  await env.CACHE.delete("ctx:weights").catch(() => {});
  return { ok: true, weights: merged };
}

/**
 * Pairwise preferences for learning to rank (spec §36): within a query, a result the user acted on
 * is preferred over every result shown above it that they skipped ("skip-above"), and over the
 * next one below it. Each pair carries both feature vectors, ready for a pairwise trainer.
 */
export async function exportPairs(env: Env, days: number, limit = 20000): Promise<Array<{ qid: string; q: string; better: number; worse: number; label_gap: number; better_features: Record<string, number>; worse_features: Record<string, number> }>> {
  const rows = await exportTraining(env, days, limit);
  const byQ = new Map<string, typeof rows>();
  for (const r of rows) byQ.set(r.qid, [...(byQ.get(r.qid) ?? []), r]);
  const out: Array<{ qid: string; q: string; better: number; worse: number; label_gap: number; better_features: Record<string, number>; worse_features: Record<string, number> }> = [];
  for (const list of byQ.values()) {
    const ranked = [...list].sort((a, b) => a.rank - b.rank);
    for (const [i, x] of ranked.entries()) {
      if (x.label <= 0) continue;
      const worse = [...ranked.slice(0, i).filter((y) => y.label < x.label), ...ranked.slice(i + 1, i + 2).filter((y) => y.label < x.label)];
      for (const y of worse) out.push({ qid: x.qid, q: x.q, better: x.post_id, worse: y.post_id, label_gap: x.label - y.label, better_features: x.features, worse_features: y.features });
      if (out.length >= limit) return out;
    }
  }
  return out;
}

// ---- answers --------------------------------------------------------------------------------------

export const ANSWER_REASONS = ["wrong", "unsupported", "incomplete", "other"] as const;

/** A vote on an /ask answer: {query_id, rating: 1 | -1 (or "up" / "down"), reason?}. */
export async function recordAnswerFeedback(env: Env, body: unknown): Promise<{ ok: boolean; error?: string }> {
  const b = (body ?? {}) as Record<string, unknown>;
  const qid = typeof b.query_id === "string" ? b.query_id.slice(0, 32) : "";
  if (!qid) return { ok: false, error: "query_id required (from the /ask answer)" };
  const raw = b.rating ?? b.vote;
  const rating = raw === 1 || raw === "1" || raw === "up" || raw === "helpful" || raw === true ? 1 : raw === -1 || raw === "-1" || raw === "down" || raw === "not_helpful" || raw === false ? -1 : 0;
  if (!rating) return { ok: false, error: "rating must be 1 (helpful) or -1 (not helpful)" };
  const reason = typeof b.reason === "string" && (ANSWER_REASONS as readonly string[]).includes(b.reason) ? b.reason : null;
  // only for a real, recent answer, and a few votes per answer: the endpoint is public
  const q = await env.DB
    .prepare("SELECT (SELECT COUNT(*) FROM answer_feedback f WHERE f.qid = l.qid) AS n FROM ask_log l WHERE l.qid = ? AND l.at >= ?")
    .bind(qid, now() - 7 * 86400)
    .first<{ n: number }>();
  if (!q) return { ok: false, error: "unknown or expired query_id" };
  if (q.n >= 20) return { ok: true };
  await env.DB.prepare("INSERT INTO answer_feedback (qid, rating, reason, at) VALUES (?, ?, ?, ?)").bind(qid, rating, reason, now()).run();
  return { ok: true };
}

/**
 * Fine-tuning examples (spec §36): the sampled /ask answers whose claims were all verified and that
 * nobody voted down, as {messages, target}. The messages are rebuilt from the exact cards and
 * context the model was given; the target is what was kept of its reply after claim verification.
 */
export async function exportSft(env: Env, days: number, opts: { minEgs?: number; limit?: number } = {}): Promise<Array<{ qid: string; question: string; model: string | null; egs: number | null; rating: number; messages: Array<{ role: string; content: string }>; target: unknown }>> {
  const rows = await env.DB
    .prepare(
      `SELECT l.qid, l.q, l.model, l.egs, l.trace, COALESCE((SELECT SUM(f.rating) FROM answer_feedback f WHERE f.qid = l.qid), 0) AS rating
       FROM ask_log l WHERE l.at >= ? AND l.status = 'answered' AND l.model IS NOT NULL AND l.trace IS NOT NULL AND COALESCE(l.egs, 0) >= ?
       ORDER BY l.at DESC LIMIT ?`,
    )
    .bind(now() - days * 86400, opts.minEgs ?? 1, Math.min(5000, opts.limit ?? 1000))
    .all<{ qid: string; q: string; model: string | null; egs: number | null; trace: string; rating: number }>();
  const out: Array<{ qid: string; question: string; model: string | null; egs: number | null; rating: number; messages: Array<{ role: string; content: string }>; target: unknown }> = [];
  for (const r of rows.results ?? []) {
    if (r.rating < 0) continue;
    let t: any;
    try {
      t = JSON.parse(r.trace);
    } catch {
      continue;
    }
    // only what was shown: a reply verification rejected is not a target
    if (!t?.model_input?.cards || !t?.model_output || t.model_output.used === false || !t.model_output.answer) continue;
    const messages = reasoningMessages({ task: "answer", question: r.q, cards: t.model_input.cards, lang: t.model_input.lang ?? "en", context: t.model_input.context ?? [] });
    out.push({ qid: r.qid, question: r.q, model: r.model, egs: r.egs, rating: r.rating, messages, target: t.model_output });
  }
  return out;
}

/** Keep the logs bounded (nightly cron). */
export async function pruneLogs(env: Env, keepDays = 120): Promise<void> {
  const since = now() - keepDays * 86400;
  await env.DB.batch([
    env.DB.prepare("DELETE FROM rank_log WHERE at < ?").bind(since),
    env.DB.prepare("DELETE FROM feedback WHERE at < ?").bind(since),
    env.DB.prepare("DELETE FROM query_log WHERE at < ?").bind(since),
    env.DB.prepare("DELETE FROM help_log WHERE at < ?").bind(since),
    env.DB.prepare("DELETE FROM ask_log WHERE at < ?").bind(since),
    env.DB.prepare("DELETE FROM answer_feedback WHERE at < ?").bind(since),
    // who ran what, for the popular searches: 30 days; the daily salts: today's and yesterday's
    // only, so that no older value can be traced back to an address (search/suggest.ts)
    env.DB.prepare("DELETE FROM query_people WHERE day < ?").bind(Math.floor(now() / 86400) - 30),
    env.DB.prepare("DELETE FROM settings WHERE k LIKE 'suggest:salt:%' AND CAST(substr(k, 14) AS INTEGER) < ?").bind(Math.floor(now() / 86400) - 1),
  ]);
}
