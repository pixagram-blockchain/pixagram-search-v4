// What /healthz reports beyond "the Worker answers": whether the embedding Space can embed now,
// and how many artworks and posts are waiting for their vectors.
//
// v4.8.2: the v4 Space stopped answering on 29 October 2026 and was found asleep on 3 November;
// all the while /healthz answered {"ok": true, "semantic": true}. 298 artworks got no vectors and nothing said so. Now /healthz
// says "degraded" with the reasons, and /healthz?strict=1 answers 503 then, for an uptime monitor.

import type { Env } from "../env";
import { bool, now } from "../env";
import { GIVE_UP_AFTER } from "../db/posts";
import { embedLabel, embeddingEnabled, spaceStatus, type SpaceStatus } from "./embed";
import { staleImageVector, staleTextVector } from "./sweeper";

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
}

export async function healthReport(env: Env): Promise<HealthReport> {
  const on = embeddingEnabled(env);
  const [row, space, vectors] = await Promise.all([
    env.DB.prepare("SELECT COUNT(*) AS n FROM posts WHERE deleted = 0").first<{ n: number }>(),
    on ? spaceStatus(env) : Promise.resolve(null),
    on ? vectorBacklog(env) : Promise.resolve(null),
  ]);
  const problems: string[] = [];
  const s = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  if (space && !space.ready) problems.push(`embedding Space not ready: ${space.problem}`);
  if (vectors?.image_stuck) problems.push(`${s(vectors.image_stuck, "artwork", "artworks")} without an image vector for over an hour`);
  if (vectors?.text_stuck) problems.push(`${s(vectors.text_stuck, "post", "posts")} without a text vector for over an hour`);
  return {
    ok: true,
    status: problems.length ? "degraded" : "ok",
    problems,
    posts: row?.n ?? 0,
    semantic: !!space?.ready,
    text_vectors: !!env.VEC_TEXT,
    embedding: space,
    vectors,
  };
}
