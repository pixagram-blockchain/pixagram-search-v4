// Shards of the copy-detection index, and checks that run in all of them at once.
//
// Works are sharded by post-id range: shard i holds posts [i·S, (i+1)·S), S = PAPH_SHARD_SIZE
// (default 100,000: about 3–5 GB of a Durable Object's 10 GB at Pixagram's sizes). The corpus
// grows into new shards by itself — nothing is ever moved — and a check asks every shard at once,
// each bounded by the same budget, so latency stays that of one shard while capacity grows with
// their number. Kept apart from shard-do.ts so modules that only call shards never import
// `cloudflare:workers`.
//
// S is part of each shard's name: changing it starts empty shards (re-run the paph stage, see
// README-V4). Choose it once.

import type { Env } from "../env";
import { bool, int } from "../env";
import type { PaphShard } from "./shard-do";
import { byStrength, type Candidate, type Checked, type FindOptions, type FindResult, type Match, type ShardStats, type WireInput } from "./shard-store";

export function paphEnabled(env: Env): boolean {
  return !!env.PAPH && bool(env.PAPH_ENABLED, true);
}

export function shardSize(env: Pick<Env, "PAPH_SHARD_SIZE">): number {
  return Math.max(1000, int(env.PAPH_SHARD_SIZE, 100_000));
}

export function shardOf(env: Pick<Env, "PAPH_SHARD_SIZE">, postId: number): number {
  return Math.floor(Math.max(0, postId) / shardSize(env));
}

export type ShardStub = DurableObjectStub<PaphShard>;

export function shardStub(env: Env, shard: number): ShardStub {
  if (!env.PAPH) throw new Error("paph: no PAPH Durable Object binding");
  return env.PAPH.get(env.PAPH.idFromName(`paph:${shardSize(env)}:${shard}`)) as unknown as ShardStub;
}

export function homeShard(env: Env, postId: number): ShardStub {
  return shardStub(env, shardOf(env, postId));
}

let counted: { at: number; size: number; n: number } | null = null;

/**
 * How many shards exist: up to the highest post id. Fresh for the enrichment stage (a work must
 * be checked against the newest shard too); interactive checks may use a count up to a minute old.
 */
export async function shardCount(env: Env, maxAgeMs = 0): Promise<number> {
  const size = shardSize(env);
  if (maxAgeMs > 0 && counted && counted.size === size && Date.now() - counted.at <= maxAgeMs) return counted.n;
  const r = await env.DB.prepare("SELECT MAX(id) AS m FROM posts").first<{ m: number | null }>();
  const n = Math.floor(Math.max(0, r?.m ?? 0) / size) + 1;
  counted = { at: Date.now(), size, n };
  return n;
}

/** Forget the cached shard count (tests). */
export function resetShardCount(): void {
  counted = null;
}

class ShardTimeout extends Error {}

export type ShardCall<T> = { shard: number; ok: true; value: T } | { shard: number; ok: false; error: string; timedOut: boolean };

/** Call shards in parallel, each within `timeoutMs`; failures and timeouts come back as values. */
export async function fanOut<T>(env: Env, shards: number[], call: (stub: ShardStub, shard: number) => Promise<T>, timeoutMs: number): Promise<Array<ShardCall<T>>> {
  return Promise.all(
    shards.map(async (shard): Promise<ShardCall<T>> => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const value = await Promise.race([
          call(shardStub(env, shard), shard),
          new Promise<never>((_, reject) => {
            timer = setTimeout(() => reject(new ShardTimeout(`shard ${shard} did not answer within ${timeoutMs} ms`)), timeoutMs);
          }),
        ]);
        return { shard, ok: true, value };
      } catch (e) {
        return { shard, ok: false, error: e instanceof Error ? e.message : String(e), timedOut: e instanceof ShardTimeout };
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }
    }),
  );
}

/** A shard whose answer is not used: it failed, did not answer in time, or runs another engine. */
export interface ShardFailure {
  shard: number;
  error: string;
  timedOut: boolean;
  /** the identity it answered under, when that was the reason (a deploy under way) */
  engine?: string;
}

export interface MergedFind {
  matches: Match[];
  checked: Checked[];
  stats: Omit<ShardStats, "ms" | "partial">;
  /**
   * shards asked; those whose answers were used, and of those the ones that answered in full;
   * the others; the slowest shard's own time and the fan-out's wall time
   */
  shards: { asked: number; answered: number; complete: number[]; failed: ShardFailure[]; slowestMs: number; wallMs: number };
  /** the verdicts' identity: the one asked for. stats.waitedMs: the longest a shard's check waited for interactive ones */
  identity: string;
  /** some shard stopped at its deadline, failed, did not answer or runs another engine: not every candidate was compared */
  partial: boolean;
}

export interface FindEverywhere extends Omit<FindOptions, "extra"> {
  /**
   * The identity the caller's verdicts carry (engine, profiles, policy). A shard that answers
   * under another — a deploy reaches the Worker and the shards at different moments — counts as
   * failed: its verdicts would mean something else, and it is not complete for the stage.
   */
  identity: string;
  extra?: Candidate[];
  /** only the shards holding `extra` candidates (a pass over given candidates, no nomination) */
  extraShardsOnly?: boolean;
  /** only these shards (a retry of the ones a check did not finish); default every shard */
  shards?: number[];
  /** how long to wait for a shard beyond its deadline (RPC, a cold start) */
  graceMs?: number;
  /** shard-count staleness allowed (0: read it now) */
  countAgeMs?: number;
}

/** Check a fingerprint in every shard at once and merge the answers. */
export async function findEverywhere(env: Env, query: WireInput, o: FindEverywhere): Promise<MergedFind> {
  const byShard = new Map<number, Candidate[]>();
  for (const c of o.extra ?? []) {
    if (!Number.isSafeInteger(c.id) || c.id <= 0) continue;
    const s = shardOf(env, c.id);
    const list = byShard.get(s) ?? [];
    list.push(c);
    byShard.set(s, list);
  }
  const shards = o.extraShardsOnly
    ? [...byShard.keys()].sort((a, b) => a - b)
    : o.shards ?? Array.from({ length: await shardCount(env, o.countAgeMs ?? 0) }, (_, i) => i);
  const { extra: _e, extraShardsOnly: _x, graceMs, countAgeMs: _c, shards: _s, identity: _i, ...opts } = o;
  const timeout = Math.max(100, o.budget.deadlineMs) + (graceMs ?? 400);
  const t0 = Date.now();
  // a shard stops at its deadline, waiting included (a check whose turn comes too late does no
  // work); the grace covers the last slice and the way back
  const calls = await fanOut(env, shards, (stub, s) => stub.find(query, { ...opts, extra: byShard.get(s) ?? [] }) as Promise<FindResult>, timeout);
  const merged: MergedFind = {
    matches: [],
    checked: [],
    stats: {
      nominated: { total: 0 },
      verified: 0, compared: 0, rejected: 0, capped: 0, pending: 0, unreadable: 0, legacy: 0,
      postings: { keys: 0, si: 0 },
      skipped: { keys: 0, si: 0 },
      waitedMs: 0,
    },
    shards: { asked: shards.length, answered: 0, complete: [], failed: [], slowestMs: 0, wallMs: Date.now() - t0 },
    identity: o.identity,
    partial: false,
  };
  for (const c of calls) {
    if (!c.ok) {
      merged.shards.failed.push({ shard: c.shard, error: c.error, timedOut: c.timedOut });
      merged.partial = true;
      continue;
    }
    const r = c.value;
    if (r.identity !== o.identity) {
      merged.shards.failed.push({ shard: c.shard, error: `shard ${c.shard} runs another engine`, timedOut: false, engine: r.identity });
      merged.partial = true;
      continue;
    }
    merged.shards.answered++;
    if (!r.stats.partial) merged.shards.complete.push(c.shard);
    merged.shards.slowestMs = Math.max(merged.shards.slowestMs, r.stats.ms);
    merged.matches.push(...r.matches);
    merged.checked.push(...r.checked);
    if (r.stats.partial) merged.partial = true;
    const s = merged.stats;
    for (const [k, v] of Object.entries(r.stats.nominated)) s.nominated[k] = (s.nominated[k] ?? 0) + v;
    s.verified += r.stats.verified;
    s.compared += r.stats.compared;
    s.rejected += r.stats.rejected;
    s.capped += r.stats.capped;
    s.pending += r.stats.pending;
    s.unreadable += r.stats.unreadable;
    s.legacy += r.stats.legacy ?? 0;
    s.postings.keys += r.stats.postings.keys;
    s.postings.si += r.stats.postings.si;
    s.skipped.keys += r.stats.skipped.keys;
    s.skipped.si += r.stats.skipped.si;
    s.waitedMs = Math.max(s.waitedMs, r.stats.waitedMs);
  }
  merged.matches.sort(byStrength);
  return merged;
}
