// What /healthz reports beyond "the Worker answers": whether the embedding Space can embed now,
// and how many artworks and posts are waiting for their vectors.
//
// v4.8.2: the v4 Space stopped answering on 29 October 2026 and was found asleep on 3 November;
// all the while /healthz answered {"ok": true, "semantic": true}. 298 artworks got no vectors and nothing said so. Now /healthz
// says "degraded" with the reasons, and /healthz?strict=1 answers 503 then, for an uptime monitor.

//
// v4.8.3: and whether new posts reach the index at all. The tail dropped posts Hivemind had not
// processed yet, about two or three Pixagram posts a day, and every Pixtopia post was filtered
// out, and nothing here said so: /healthz now reports the chain tail (stopped, stalled, behind,
// posts waiting for Hivemind) and what the reconciliation had to recover.

import type { Env } from "../env";
import { bool, now } from "../env";
import { GIVE_UP_AFTER } from "../db/posts";
import { embedLabel, embeddingEnabled, spaceStatus, type SpaceStatus } from "./embed";
import { staleImageVector, staleTextVector } from "./sweeper";
import { indexerStub } from "../chain/indexer-do";
import { reconcileRecord } from "../chain/reconcile";

/** A vector missing for longer than this is stuck, not on its way (the queue embeds a new artwork within minutes). */
export const STUCK_AFTER_S = 3600;

export interface VectorBacklog {
  /** live artworks whose image is indexed and whose embedding is not skipped for good */
  artworks: number;
  /** of those, without a current image vector (none, or one of another image, model, views or patch budget) */
  image_missing: number;
  /** of those, untouched for over an hour */
  image_stuck: number;
  /** when the oldest of them last changed (unix seconds) */
  image_oldest: number | null;
  /** live posts without a current text vector, and those of them untouched for over an hour (null: text vectors off) */
  text_missing: number | null;
  text_stuck: number | null;
  /** failed jobs at the give-up cap, by stage: the sweeper no longer re-drives them */
  given_up: Record<string, number>;
}

const BACKLOG = new WeakMap<object, { at: number; v: VectorBacklog }>();
/** One count of the backlog stands for a minute per isolate: /healthz polled often costs D1 little. */
const BACKLOG_TTL_S = 60;

export async function vectorBacklog(env: Env): Promise<VectorBacklog> {
  const hit = BACKLOG.get(env.DB);
  if (hit && now() - hit.at < BACKLOG_TTL_S) return hit.v;
  const v = await countBacklog(env);
  BACKLOG.set(env.DB, { at: now(), v });
  return v;
}

async function countBacklog(env: Env): Promise<VectorBacklog> {
  const model = env.EMBED_MODEL ?? "";
  const cutoff = now() - STUCK_AFTER_S;
  const img = staleImageVector("?1", "?2", !!model);
  const txt = staleTextVector("?1", !!model);
  const textOn = bool(env.TEXT_VECTORS, true) && !!env.VEC_TEXT;
  const stmts = [
    // ?1 model, ?2 views label, ?3 cutoff (bound even when the model clause is left out: D1 wants
    // as many values as the highest placeholder)
    env.DB
      .prepare(
        `SELECT COUNT(*) AS n, SUM(CASE WHEN ${img} THEN 1 ELSE 0 END) AS missing,
                SUM(CASE WHEN (${img}) AND a.updated < ?3 THEN 1 ELSE 0 END) AS stuck,
                MIN(CASE WHEN ${img} THEN a.updated END) AS oldest
         FROM artworks a JOIN posts p ON p.id = a.post_id
         WHERE p.deleted = 0 AND p.type = 'artwork' AND a.stats_hash = a.content_hash
           AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.post_id = a.post_id AND j.stage = 'embed' AND j.status = 'skipped')`,
      )
      .bind(model, embedLabel(env), cutoff),
    env.DB
      .prepare(
        `SELECT j.stage, COUNT(*) AS n FROM jobs j JOIN posts p ON p.id = j.post_id
         WHERE p.deleted = 0 AND j.status = 'failed' AND j.attempts >= ? GROUP BY j.stage ORDER BY j.stage`,
      )
      .bind(GIVE_UP_AFTER),
  ];
  if (textOn) {
    // ?1 model, ?2 cutoff
    stmts.push(
      env.DB
        .prepare(
          `SELECT SUM(CASE WHEN ${txt} THEN 1 ELSE 0 END) AS missing,
                  SUM(CASE WHEN (${txt}) AND p.indexed_at < ?2 THEN 1 ELSE 0 END) AS stuck
           FROM posts p WHERE p.deleted = 0
             AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.post_id = p.id AND j.stage = 'text' AND j.status = 'skipped')`,
        )
        .bind(model, cutoff),
    );
  }
  const [image, gave, text] = await env.DB.batch<Record<string, number | string | null>>(stmts);
  const i = image.results?.[0] ?? {};
  const t = text?.results?.[0];
  const num = (v: unknown) => Number(v ?? 0) || 0;
  return {
    artworks: num(i.n),
    image_missing: num(i.missing),
    image_stuck: num(i.stuck),
    image_oldest: i.oldest === null || i.oldest === undefined ? null : Number(i.oldest),
    text_missing: t ? num(t.missing) : null,
    text_stuck: t ? num(t.stuck) : null,
    given_up: Object.fromEntries((gave.results ?? []).map((x) => [String(x.stage), num(x.n)])),
  };
}

/** The chain tail as /healthz shows it (src/chain/indexer-do.ts, src/chain/reconcile.ts). */
export interface TailHealth {
  running: boolean;
  cursor: number | null;
  lib: number | null;
  /** irreversible blocks not read yet (3 s each) */
  behind_blocks: number | null;
  last_tick_age_s: number | null;
  /** posts whose op the tail has read but Hivemind does not show yet, and those waiting for over an hour */
  deferred: number;
  deferred_over_hour: number;
  /** posts set aside because their ingestion kept failing */
  parked: number;
  /** the reconciliation's last run, and what it has recovered (posts the tail missed) */
  reconcile: { last_run_age_s: number; last_missing: number; last_stale: number; recovered_total: number; last_recovery_at: number | null } | null;
  /** the indexer could not be asked */
  error?: string;
}

/** A tick every few seconds is normal; this long without one is a stalled tail. */
const TAIL_STALLED_S = 600;
/** About an hour of blocks. */
const TAIL_BEHIND_BLOCKS = 1200;

export async function tailHealth(env: Env): Promise<TailHealth | null> {
  // no Durable Object namespace (tests, local tools): nothing to ask
  if (typeof (env.INDEXER as { idFromName?: unknown } | undefined)?.idFromName !== "function") return null;
  const t = now();
  const rec = await reconcileRecord(env).catch(() => null);
  const reconcile = rec
    ? { last_run_age_s: t - rec.last.at, last_missing: rec.last.missing, last_stale: rec.last.stale, recovered_total: rec.recovered_total, last_recovery_at: rec.last_recovery?.at ?? null }
    : null;
  try {
    const s = await indexerStub(env).status();
    const hourAgo = Date.now() - 3_600_000;
    return {
      running: s.running,
      cursor: s.cursor,
      lib: s.lib,
      behind_blocks: s.cursor !== null && s.lib !== null ? Math.max(0, s.lib - s.cursor + 1) : null,
      last_tick_age_s: s.lastTickAt ? Math.round((Date.now() - s.lastTickAt) / 1000) : null,
      deferred: s.deferredWaiting ?? 0,
      deferred_over_hour: (s.deferred ?? []).filter((d) => d.since < hourAgo).length,
      parked: s.parked?.length ?? 0,
      reconcile,
    };
  } catch (e) {
    return { running: false, cursor: null, lib: null, behind_blocks: null, last_tick_age_s: null, deferred: 0, deferred_over_hour: 0, parked: 0, reconcile, error: e instanceof Error ? e.message : String(e) };
  }
}

export interface HealthReport {
  /** the Worker and its database answer */
  ok: true;
  /** "ok": search works with every leg; "degraded": see problems */
  status: "ok" | "degraded";
  problems: string[];
  posts: number;
  /** semantic search works now: the Space is configured and ready (before v4.8.2: configured) */
  semantic: boolean;
  text_vectors: boolean;
  /** the Space's /health, checked at most once a minute (null when HF_EMBED_URL is not set) */
  embedding: SpaceStatus | null;
  vectors: VectorBacklog | null;
  /** v4.8.3: whether new posts reach the index (null without the indexer binding) */
  tail: TailHealth | null;
}

export async function healthReport(env: Env): Promise<HealthReport> {
  const on = embeddingEnabled(env);
  const [row, space, vectors, tail] = await Promise.all([
    env.DB.prepare("SELECT COUNT(*) AS n FROM posts WHERE deleted = 0").first<{ n: number }>(),
    on ? spaceStatus(env) : Promise.resolve(null),
    on ? vectorBacklog(env) : Promise.resolve(null),
    tailHealth(env).catch(() => null),
  ]);
  const problems: string[] = [];
  const s = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  if (space && !space.ready) problems.push(`embedding Space not ready: ${space.problem}`);
  if (vectors?.image_stuck) problems.push(`${s(vectors.image_stuck, "artwork", "artworks")} without an image vector for over an hour`);
  if (vectors?.text_stuck) problems.push(`${s(vectors.text_stuck, "post", "posts")} without a text vector for over an hour`);
  if (tail?.error) problems.push(`chain indexer unreachable: ${tail.error}`);
  else if (tail) {
    if (!tail.running) problems.push("chain indexer stopped: new posts are not indexed (POST /admin/indexer/start)");
    else if (tail.last_tick_age_s !== null && tail.last_tick_age_s > TAIL_STALLED_S) problems.push(`chain indexer stalled: no tick for ${Math.round(tail.last_tick_age_s / 60)} min`);
    if (tail.behind_blocks !== null && tail.behind_blocks > TAIL_BEHIND_BLOCKS) problems.push(`chain indexer ${tail.behind_blocks} blocks (about ${Math.round((tail.behind_blocks * 3) / 360) / 10} h) behind the chain`);
    if (tail.deferred_over_hour) problems.push(`${s(tail.deferred_over_hour, "new post", "new posts")} waiting for the node's Hivemind for over an hour`);
  }
  return {
    ok: true,
    status: problems.length ? "degraded" : "ok",
    problems,
    posts: row?.n ?? 0,
    semantic: !!space?.ready,
    text_vectors: !!env.VEC_TEXT,
    embedding: space,
    vectors,
    tail,
  };
}
