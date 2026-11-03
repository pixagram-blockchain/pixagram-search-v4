// Sweeper: re-drives enrichment that fell through the cracks. The queue retries a message 8 times
// and then parks it in the dead-letter queue; when the Space was down for longer than that, the
// artwork stayed without a vector forever (21 of 133 on the v2 stack in Oct 2026). Every run picks
// a bounded batch of artworks whose stage is missing or stale and enqueues just those stages.
//
// v4.8.2: the embedding stages (embed, text) are re-driven only while the Space is ready (its
// /health, src/enrich/embed.ts spaceStatus): while it is asleep or down they would only fail again.
// Their failures on the Space's side no longer count toward the give-up cap (src/db/posts.ts), so
// everything it missed is re-driven once it is back, as many per run as its CPUs embed in a minute
// or two (sweepBudget).

import type { Env, Stage } from "../env";
import { bool, now } from "../env";
import { embedLabel, embeddingEnabled, spaceSlots, spaceStatus } from "./embed";
import { GIVE_UP_AFTER, setJobsMany } from "../db/posts";
import { paphEnabled } from "../paph/shards";
import { currentEngine } from "../paph/copies";

/**
 * Artworks whose image vector is missing or stale (another image, model, views or patch budget),
 * as an SQL condition on `a` (artworks). `model` and `views` are the placeholders bound to
 * EMBED_MODEL and embedLabel(env); the model clause is left out when EMBED_MODEL is unset.
 */
export function staleImageVector(model: string, views: string, modelSet: boolean): string {
  return `a.embed_hash IS NULL OR a.embed_hash != a.content_hash OR COALESCE(a.embed_views, 'xbrz') != ${views}${modelSet ? ` OR a.embed_model IS NOT ${model}` : ""}`;
}

/** Posts (`p`) whose text vector is missing or of another model; `model` is the placeholder bound to EMBED_MODEL. */
export function staleTextVector(model: string, modelSet: boolean): string {
  return `p.text_hash IS NULL${modelSet ? ` OR p.text_model IS NOT ${model}` : ""}`;
}

/**
 * Posts one ten-minute run re-drives at most: twelve per embedding the Space computes at once
 * (its /health "concurrency", one per CPU), at least 25 (the 2-vCPU Space's old budget), at most
 * 200. On an 8-vCPU Space that is 96 a run: a backlog of a few hundred is gone within the hour.
 */
export async function sweepBudget(env: Env): Promise<number> {
  const slots = embeddingEnabled(env) ? await spaceSlots(env).catch(() => 2) : 2;
  return Math.min(200, Math.max(25, slots * 12));
}

export async function sweep(env: Env, max = 50): Promise<{ enqueued: number; byStage: Record<string, number>; held?: string }> {
  const byStage: Record<string, number> = {};
  const cutoff = now() - 30 * 60; // leave fresh jobs to the queue
  const want = new Map<number, { author: string; permlink: string; stages: Set<Stage> }>();
  const add = (rows: Array<{ id: number; author: string; permlink: string }>, stage: Stage) => {
    for (const r of rows) {
      const w = want.get(r.id) ?? { author: r.author, permlink: r.permlink, stages: new Set<Stage>() };
      w.stages.add(stage);
      want.set(r.id, w);
      byStage[stage] = (byStage[stage] ?? 0) + 1;
    }
  };
  // ?1 = cutoff, ?2 = max, ?3 = EMBED_MODEL, ?4 = views label (EMBED_VIEWS, EMBED_PATCHES), ?5 = the
  // copy-detection identity. SQLite counts parameters up
  // to the highest number used, and D1 rejects a bind count that differs, so bind exactly that many.
  const model = env.EMBED_MODEL ?? "";
  const views = embedLabel(env);
  // copy detection: the identity a completed check must carry (a release or policy change re-checks)
  const paphEngine = paphEnabled(env) ? await currentEngine(env).catch((e: unknown) => (console.warn("sweeper: paph engine unavailable", e), null)) : null;
  const params = [cutoff, max, model, views, paphEngine ?? ""];
  const q = (sql: string, stage: Stage) => {
    const text = sql.replace("$NOTBUSY", notBusy(stage));
    const used = Math.max(0, ...[...text.matchAll(/\?(\d+)/g)].map((m) => Number(m[1])));
    return env.DB.prepare(text).bind(...params.slice(0, used)).all<{ id: number; author: string; permlink: string }>().then((r) => r.results ?? []);
  };
  // Not queued recently, not given up on (skipped = permanent, e.g. license or unsupported image), not hopeless.
  const notBusy = (stage: Stage) => `NOT EXISTS (SELECT 1 FROM jobs j WHERE j.post_id = p.id AND j.status = 'queued' AND j.updated >= ?1)
                   AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.post_id = p.id AND j.stage = '${stage}' AND j.status = 'skipped')
                   AND COALESCE((SELECT j.attempts FROM jobs j WHERE j.post_id = p.id AND j.stage = '${stage}'), 0) < ${GIVE_UP_AFTER}`;
  // The embedding stages only while the Space can do them: asleep or down, they would fail again
  // (uncounted, but each delivery still waits for the Space). The other stages go on.
  let held: string | undefined;
  let spaceReady = false;
  if (embeddingEnabled(env)) {
    const s = await spaceStatus(env).catch((e: unknown) => ({ ready: false, problem: e instanceof Error ? e.message : String(e) }));
    spaceReady = s.ready;
    if (!s.ready) held = `embed and text held back, the Space is not ready: ${s.problem}`;
  }

  add(await q(`SELECT p.id, p.author, p.permlink FROM posts p LEFT JOIN artworks a ON a.post_id = p.id
               WHERE p.deleted = 0 AND p.type = 'artwork' AND (a.post_id IS NULL OR a.stats_hash IS NULL OR a.features_hash IS NULL OR a.features_hash != a.content_hash) AND $NOTBUSY LIMIT ?2`, "stats"), "stats");
  if (paphEngine) {
    // copy detection not completed for the current image under the current identity (stats first:
    // the stage reads its hash)
    add(await q(`SELECT p.id, p.author, p.permlink FROM posts p JOIN artworks a ON a.post_id = p.id
                 WHERE p.deleted = 0 AND p.type = 'artwork' AND a.stats_hash = a.content_hash
                   AND (a.paph_hash IS NULL OR a.paph_hash != a.content_hash OR a.paph_engine IS NOT ?5) AND $NOTBUSY LIMIT ?2`, "paph"), "paph");
  }
  if (spaceReady) {
    // missing, image changed, other model (EMBED_MODEL changed), or other views (EMBED_VIEWS changed)
    add(await q(`SELECT p.id, p.author, p.permlink FROM posts p JOIN artworks a ON a.post_id = p.id
                 WHERE p.deleted = 0 AND p.type = 'artwork' AND (${staleImageVector("?3", "?4", !!model)}) AND $NOTBUSY LIMIT ?2`, "embed"), "embed");
  }
  if ((env.VLM_BACKEND ?? "gemma") !== "off") {
    add(await q(`SELECT p.id, p.author, p.permlink FROM posts p JOIN artworks a ON a.post_id = p.id
                 WHERE p.deleted = 0 AND p.type = 'artwork' AND (a.describe_hash IS NULL OR a.describe_hash != a.content_hash OR COALESCE(a.ai_caption, '') = '') AND $NOTBUSY LIMIT ?2`, "describe"), "describe");
  }
  add(await q(`SELECT p.id, p.author, p.permlink FROM posts p JOIN artworks a ON a.post_id = p.id
               WHERE p.deleted = 0 AND p.type = 'artwork' AND a.concepts_hash IS NULL AND $NOTBUSY LIMIT ?2`, "concepts"), "concepts");
  if (spaceReady && bool(env.TEXT_VECTORS, true) && env.VEC_TEXT) {
    add(await q(`SELECT p.id, p.author, p.permlink FROM posts p WHERE p.deleted = 0 AND (${staleTextVector("?3", !!model)}) AND $NOTBUSY LIMIT ?2`, "text"), "text");
  }

  const msgs = [...want.entries()].slice(0, max).map(([postId, w]) => ({ body: { postId, author: w.author, permlink: w.permlink, stages: [...w.stages], reason: "sweeper" } }));
  for (let i = 0; i < msgs.length; i += 100) await env.ENRICH_QUEUE.sendBatch(msgs.slice(i, i + 100));
  await setJobsMany(env.DB, msgs.map((m) => ({ postId: m.body.postId, stages: m.body.stages })), "queued");
  return held ? { enqueued: msgs.length, byStage, held } : { enqueued: msgs.length, byStage };
}
