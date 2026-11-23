// ChainIndexer: a single Durable Object that tails the Pixa chain.
//
// One named instance ("main") holds the block cursor and re-arms its own alarm. It reads only
// irreversible blocks (no reorg handling needed), picks top-level comment ops and marketplace
// custom_json ops out of each block, fetches the post's current state with get_content and
// upserts it. Everything else (votes, follows, transfers) is ignored.
//
// A post whose ingestion keeps failing is set aside ("parked") after a minute of failed ticks, so
// one post can never hold the whole chain back; the ten-minute cron tries parked posts again.
//
// v4.8.3: get_content is answered by Hivemind, which can lag behind the blocks the tail reads
// (about three seconds after they are produced). A Hivemind still on an earlier block answers
// "Post a/p does not exist" for a post created after it, and the previous version for a post
// edited after it. The tail took the first answer for a deletion and dropped the post for good:
// 46 Pixagram posts between 28 October and 15 November 2026, seven of them in one hour on the
// evening of 12 November, found only by a manual backfill. Now each op carries the state
// Hivemind must show once it has processed it (its `last_update`, see OpRef.prevAt); a post it
// has not caught up with is "deferred" and asked again on the next ticks, with backoff, for up to
// TAIL_DEFER_HOURS. src/chain/reconcile.ts is the safety net behind it.
//
// v4.8.4: from 18 Nov 13:24 to 19 Nov 12:32 UTC (2026) no post reached the index, and none came
// back: what fits is the primary node's Hivemind stalled while answering "does not exist". Every
// node runs a Hivemind of its own, so the retries of a deferred (and of a parked) post now ask
// the next node first. And nothing but an explicit stop() halts the tail: a step of a tick other
// than a post's ingestion (the edit history, marketplace ops) that keeps failing on the same
// blocks is skipped after TAIL_STEP_SKIP_AFTER ticks, logged and listed in status() (parking
// covered posts only, so such a step held the cursor for good); a block range that cannot be read
// is read in smaller pieces; and the ten-minute watchdog starts a tail that was never started and
// re-arms one whose alarm is gone or which has not ticked for a quarter of an hour.

import { DurableObject } from "cloudflare:workers";
import type { Env } from "../env";
import { int, num } from "../env";
import { chainTime, normalizeOp, type Block, type ChainRpc, type CommentOp, type CondenserPost, type CustomJsonOp } from "./rpc";
import { ingestPost, rpcFor, type IngestOutcome } from "./ingest";
import { handleMarketOp, marketIds } from "./market";
import { artworksWithHashes, refreshImageHistory, relabelKinds, versionsFromBlocks, writeVersions } from "./versions";

interface PostRef {
  author: string;
  permlink: string;
  block: number;
}

/** A post a block mentions, and what Hivemind must show once it has processed the op. */
export interface OpRef extends PostRef {
  /** the post's last operation in the blocks read: a comment (create or edit) or a delete_comment */
  op: "comment" | "delete";
  /**
   * The `last_update` Hivemind gives the post once it has processed the op (unix seconds): the
   * time of the block *before* the op's, because hived stamps a post with the head block time,
   * which moves to a block's own time only after that block's transactions are applied. Measured
   * on api.pixagram.com, Nov 2026, on creations and edits alike: an op in the block of 03:00:12
   * gives last_update 03:00:09. 0: any state will do (a parked post retried by the cron).
   */
  prevAt: number;
}

/** A post the tail set aside because ingesting it kept failing; retryParked() tries it again. */
export interface ParkedPost extends PostRef {
  error: string;
  parkedAt: number;
  /** tries by retryParked() so far */
  attempts: number;
  lastTryAt: number;
  /** the op that brought it in (absent on posts parked before v4.8.3) */
  op?: "comment" | "delete";
}

/** A post whose op the tail has read but whose new state Hivemind does not show yet. */
export interface DeferredPost extends OpRef {
  /** what Hivemind answered last */
  why: string;
  /** v4.8.4: the node that answered last (each retry asks the next node first) */
  node?: string | null;
  /** when it was first deferred, last tried and is due again (milliseconds) */
  since: number;
  lastTryAt: number;
  nextAt: number;
  tries: number;
}

interface IndexerState {
  running: boolean;
  cursor: number | null; // next block to process
  lastProcessed: number | null;
  /** the timestamp of block lastProcessed (unix seconds): the prevAt of the next block's ops */
  lastBlockTime: number | null;
  lib: number | null;
  head: number | null;
  lastTickAt: number | null;
  lastError: string | null;
  lastErrorAt: number | null;
  postsSeen: number;
  ticks: number;
  /** bumped by start()/stop(): a tick that sees it change does not overwrite their decision */
  epoch: number;
  /** the post the last failed ticks stopped on, and how many ticks in a row stopped there */
  stuck: { ref: string; block: number; ticks: number; error: string } | null;
  /** posts deferred so far (Hivemind behind), and those of them given up on */
  deferredTotal: number;
  deferredGivenUp: number;
  /** v4.8.4: when stop() was called (null: never, or started since); the watchdog leaves a stopped tail alone */
  stoppedAt: number | null;
  /** v4.8.4: the step other than a post's ingestion that the last failed ticks stopped on, and how many ticks in a row */
  stuckStep: { step: TickStep; cursor: number; ticks: number; error: string } | null;
  /** v4.8.4: steps skipped after failing TAIL_STEP_SKIP_AFTER ticks in a row (the last SKIPPED_MAX), and how many in all */
  skipped: SkippedStep[];
  skippedTotal: number;
  /** v4.8.4: block ranges are read 2^readShrink times smaller after reads that failed (0: TAIL_BLOCKS_PER_TICK) */
  readShrink: number;
  /** successful reads since readShrink last changed */
  readOk: number;
}

/** The steps of a tick that are not a post's ingestion (which parking covers). */
export type TickStep = "versions" | "market";

export interface SkippedStep {
  step: TickStep;
  from: number;
  to: number;
  error: string;
  at: number;
}

const DEFAULT_STATE: IndexerState = {
  running: false,
  cursor: null,
  lastProcessed: null,
  lastBlockTime: null,
  lib: null,
  head: null,
  lastTickAt: null,
  lastError: null,
  lastErrorAt: null,
  postsSeen: 0,
  ticks: 0,
  epoch: 0,
  stuck: null,
  deferredTotal: 0,
  deferredGivenUp: 0,
  stoppedAt: null,
  stuckStep: null,
  skipped: [],
  skippedTotal: 0,
  readShrink: 0,
  readOk: 0,
};

/** Failed ticks in a row on one step of the same blocks before that step is skipped for them (about a minute). */
const STEP_SKIP_AFTER = 6;
/** Skipped steps listed in the state (the newest). */
const SKIPPED_MAX = 20;
/** Smallest read: TAIL_BLOCKS_PER_TICK / 2^7 (100 → 1 block). */
const READ_SHRINK_MAX = 7;
/** Successful reads before a shrunk range doubles again. */
const READ_GROW_AFTER = 20;
/** The watchdog re-arms a running tail that has not ticked for this long, alarm or not. */
export const STALL_MS = 15 * 60_000;

/** Failed ticks in a row on one post before it is parked: about a minute (a failed tick re-arms in 10 s). */
const PARK_AFTER = 6;
/** Posts parked at most (one storage value); past that the longest-parked goes, logged (a backfill of its author brings it back). */
const PARKED_MAX = 200;
/** Parked posts tried again per cron run. */
const RETRY_PER_RUN = 10;

/**
 * Deferred posts kept at most, and the bytes of their storage value: one value, which the storage
 * API caps at 128 KiB (as for the parked list). Past either, the oldest go, logged; the
 * reconciliation still finds them once Hivemind lists them.
 */
export const DEFERRED_MAX = 200;
const DEFERRED_MAX_BYTES = 100_000;
/** Deferred posts asked again per tick, the longest-waiting first. */
const DEFER_PER_TICK = 10;
/** How long a deferred post is asked again before the tail gives up on it (TAIL_DEFER_HOURS). */
const DEFER_HOURS = 24;
/** Wait before the n-th retry: 6 s, 12 s, 24 s … then every ten minutes. */
export const deferBackoffMs = (tries: number): number => Math.min(600_000, 3_000 * 2 ** Math.min(Math.max(1, tries), 10));
/** The deferred posts status() lists (the oldest first); deferredWaiting counts them all. */
const STATUS_DEFERRED = 50;
/** Characters of Hivemind's last answer kept on a deferred post. */
const WHY_MAX = 160;

const refKey = (r: { author: string; permlink: string }): string => `${r.author}/${r.permlink}`;
const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e)).slice(0, 240);
const iso = (t: number): string => new Date(t * 1000).toISOString().slice(0, 19);

/** When Hivemind last changed the post (its last edit, else its creation), unix seconds; 0 unknown. */
function hivemindTime(p: CondenserPost): number {
  return chainTime(p.last_update ?? p.updated ?? p.created);
}

/** What asking Hivemind about one op came to. */
export type OpOutcome =
  | { kind: "ingested"; r: IngestOutcome }
  /** deleted on chain: marked deleted if it is in the index */
  | { kind: "gone" }
  /** Hivemind has not processed the op yet; `stale` is the older state it does show, if any; `node` the node asked */
  | { kind: "deferred"; why: string; stale: CondenserPost | null; node?: string | null };

async function markDeleted(env: Env, ref: { author: string; permlink: string }): Promise<void> {
  // delete_comment on a post without votes really removes it from state.
  await env.DB.prepare("UPDATE posts SET deleted = 1 WHERE author = ? AND permlink = ?").bind(ref.author, ref.permlink).run();
}

/**
 * Ingest the state an op left a post in, once Hivemind shows it. "does not exist" after a comment
 * op is Hivemind being behind (the op proves the post exists), never a deletion; "was deleted",
 * or any absence after a delete_comment, is final. A state older than the op is Hivemind behind
 * on an edit (or on the delete), unless prevAt is 0.
 */
export async function ingestOp(env: Env, ref: OpRef, reason: string, rpc: ChainRpc = rpcFor(env), o: { node?: number } = {}): Promise<OpOutcome> {
  // v4.8.4: `node` asks the node-th configured node first (a retry of a deferred or parked post):
  // every node runs a Hivemind of its own, and one that has stalled must not hold a post back
  const s = await rpc.getContentState(ref.author, ref.permlink, o.node === undefined ? {} : { from: o.node });
  if ("absent" in s) {
    if (ref.op === "delete" || s.absent === "deleted") {
      await markDeleted(env, ref);
      return { kind: "gone" };
    }
    return { kind: "deferred", why: "Hivemind does not hold it yet", stale: null, node: rpc.lastNode };
  }
  const t = hivemindTime(s.post);
  if (ref.prevAt > 0 && t > 0 && t < ref.prevAt) {
    return { kind: "deferred", why: `Hivemind shows the state of ${iso(t)}, the op needs ${iso(ref.prevAt)}`, stale: s.post, node: rpc.lastNode };
  }
  return { kind: "ingested", r: await ingestPost(env, s.post, ref.block, reason) };
}

export class ChainIndexer extends DurableObject<Env> {
  private async load(): Promise<IndexerState> {
    return { ...DEFAULT_STATE, ...((await this.ctx.storage.get<Partial<IndexerState>>("state")) ?? {}) };
  }

  private async save(s: IndexerState): Promise<void> {
    await this.ctx.storage.put("state", s);
  }

  private async parkedList(): Promise<ParkedPost[]> {
    return (await this.ctx.storage.get<ParkedPost[]>("parked")) ?? [];
  }

  private async park(ref: PostRef & { op?: "comment" | "delete" }, error: string): Promise<void> {
    const now = Date.now();
    const list = (await this.parkedList()).filter((p) => refKey(p) !== refKey(ref));
    list.push({ author: ref.author, permlink: ref.permlink, block: ref.block, op: ref.op, error, parkedAt: now, attempts: 0, lastTryAt: now });
    while (list.length > PARKED_MAX) {
      const i = list.reduce((oldest, p, j) => (p.parkedAt < list[oldest].parkedAt ? j : oldest), 0);
      console.error("indexer dropped a parked post (too many parked)", refKey(list[i]), list[i].error);
      list.splice(i, 1);
    }
    await this.ctx.storage.put("parked", list);
  }

  private async deferredMap(): Promise<Map<string, DeferredPost>> {
    return new Map(((await this.ctx.storage.get<DeferredPost[]>("deferred")) ?? []).map((d) => [refKey(d), d]));
  }

  private async saveDeferred(m: Map<string, DeferredPost>): Promise<void> {
    const list = [...m.values()].sort((a, b) => a.since - b.since);
    while (list.length > DEFERRED_MAX || (list.length && JSON.stringify(list).length > DEFERRED_MAX_BYTES)) {
      const d = list.shift()!;
      m.delete(refKey(d));
      console.error("indexer dropped a deferred post (too many deferred; the reconciliation or a backfill brings it back)", refKey(d), "block", d.block, d.why);
    }
    await this.ctx.storage.put("deferred", list);
  }

  // ---- control API (called by the Worker's admin routes) ------------------------

  async status(): Promise<IndexerState & { alarmAt: number | null; parked: ParkedPost[]; deferredWaiting: number; deferred: DeferredPost[] }> {
    const s = await this.load();
    const deferred = [...(await this.deferredMap()).values()].sort((a, b) => a.since - b.since);
    return { ...s, alarmAt: await this.ctx.storage.getAlarm(), parked: await this.parkedList(), deferredWaiting: deferred.length, deferred: deferred.slice(0, STATUS_DEFERRED) };
  }

  /**
   * Try parked posts again (the ten-minute cron), from the front of the list: the list is kept in
   * order of last activity — a post is parked at the back, and one tried and still failing goes
   * back there, its attempt counted — so no post waits behind ones that keep failing. One that
   * ingests now (or turns out deleted) leaves the list. One Hivemind still does not hold after a
   * comment op stays (it exists: the op says so). The list is re-read before it is written: a
   * tick may park another post while this awaits the chain.
   */
  async retryParked(limit = RETRY_PER_RUN): Promise<{ retried: number; ingested: number; left: number }> {
    const batch = (await this.parkedList()).slice(0, Math.max(0, limit));
    if (!batch.length) return { retried: 0, ingested: 0, left: 0 };
    const done = new Set<string>();
    const failed = new Map<string, { error: string; at: number }>();
    const rpc = rpcFor(this.env);
    for (const p of batch) {
      try {
        // any state Hivemind shows will do (prevAt 0): the post has waited long enough. Each try
        // starts with another node (v4.8.4): the one the tail kept failing on was the preferred one.
        const out = await ingestOp(this.env, { author: p.author, permlink: p.permlink, block: p.block, op: p.op ?? "delete", prevAt: 0 }, "parked", rpc, { node: p.attempts + 1 });
        if (out.kind === "deferred") failed.set(refKey(p), { error: out.why, at: Date.now() });
        else done.add(refKey(p));
      } catch (e) {
        failed.set(refKey(p), { error: errorText(e), at: Date.now() });
      }
    }
    const now = await this.parkedList();
    const stay = now.filter((p) => !done.has(refKey(p)) && !failed.has(refKey(p)));
    const again = now
      .filter((p) => failed.has(refKey(p)))
      .map((p) => {
        const f = failed.get(refKey(p))!;
        return { ...p, attempts: p.attempts + 1, error: f.error, lastTryAt: f.at };
      });
    const next = [...stay, ...again];
    await this.ctx.storage.put("parked", next);
    return { retried: batch.length, ingested: done.size, left: next.length };
  }

  /** Start tailing. `from` replays from that block; otherwise continue, or begin at the irreversible head. */
  async start(from?: number): Promise<IndexerState> {
    const s = await this.load();
    if (typeof from === "number" && from > 0) s.cursor = from;
    if (s.cursor === null) {
      const dgp = await rpcFor(this.env).getDynamicGlobalProperties();
      s.cursor = dgp.last_irreversible_block_num;
    }
    s.running = true;
    s.stoppedAt = null;
    s.lastError = null;
    s.stuck = null;
    s.stuckStep = null;
    s.readShrink = 0;
    s.readOk = 0;
    s.epoch++;
    await this.save(s);
    await this.ctx.storage.setAlarm(Date.now() + 10);
    return s;
  }

  async stop(): Promise<IndexerState> {
    const s = await this.load();
    s.running = false;
    s.stoppedAt = Date.now();
    s.epoch++;
    await this.save(s);
    await this.ctx.storage.deleteAlarm();
    return s;
  }

  /** Watchdog (before v4.8.4): if running but the alarm is gone, re-arm it. Kept for callers of the old name. */
  async ensureAlarm(): Promise<boolean> {
    return (await this.watchdog()).action !== "none";
  }

  /**
   * The ten-minute watchdog. A tail that was never started (a fresh deployment: no cursor, no
   * stop) is started at the irreversible head; one stopped by stop() is left alone (/healthz says
   * so). A running tail whose alarm is gone (it happens after failed deploys, or once an alarm
   * exhausted its retries) or that has not ticked for STALL_MS is re-armed. A tick in progress is
   * not disturbed: it sets its own alarm when it ends, and alarm handlers never overlap.
   */
  async watchdog(): Promise<{ action: "none" | "started" | "rearmed"; why?: string }> {
    const s = await this.load();
    if (!s.running) {
      if (s.cursor !== null || s.stoppedAt !== null) return { action: "none" };
      await this.start();
      return { action: "started", why: "never started" };
    }
    const alarm = await this.ctx.storage.getAlarm();
    const since = Date.now() - (s.lastTickAt ?? 0);
    if (alarm === null || since > STALL_MS) {
      const why = alarm === null ? "no alarm" : `no tick for ${Math.round(since / 60_000)} min (alarm at ${new Date(alarm).toISOString()})`;
      await this.ctx.storage.setAlarm(Date.now() + 10);
      return { action: "rearmed", why };
    }
    return { action: "none" };
  }

  // ---- the loop ---------------------------------------------------------------------

  /**
   * Persist a tick's outcome and re-arm, unless start()/stop() ran meanwhile: alarm() awaits RPC
   * and D1 calls, and a Durable Object delivers other calls during those awaits. Then their
   * decision (running, cursor, alarm) stands and only the tick's counters are merged in.
   */
  private async commit(s: IndexerState, epoch: number, nextAlarmMs: number): Promise<void> {
    const cur = await this.load();
    if (cur.epoch !== epoch) {
      await this.save({
        ...cur, ticks: s.ticks, postsSeen: s.postsSeen, lib: s.lib, head: s.head, lastTickAt: s.lastTickAt,
        deferredTotal: s.deferredTotal, deferredGivenUp: s.deferredGivenUp, skipped: s.skipped, skippedTotal: s.skippedTotal,
      });
      return;
    }
    await this.save(s);
    await this.ctx.storage.setAlarm(Date.now() + nextAlarmMs);
  }

  /**
   * The time of the block before `cursor` (the prevAt of the ops in the first block of a range):
   * from the last tick when it processed that block, else read from the chain. Should even that
   * fail, three seconds (one block) before the first block's own time.
   */
  private async timeBefore(s: IndexerState, rpc: ChainRpc, firstBlockTime: number): Promise<number> {
    if (s.cursor !== null && s.lastProcessed === s.cursor - 1 && s.lastBlockTime) return s.lastBlockTime;
    try {
      const [b] = await rpc.getBlockRange(s.cursor! - 1, 1);
      const t = chainTime(b?.timestamp);
      if (t > 0) return t;
    } catch (e) {
      console.warn("indexer: could not read the block before", s.cursor, errorText(e));
    }
    return firstBlockTime - 3;
  }

  /**
   * Ask Hivemind again about the deferred posts that are due, the longest-waiting first, at most
   * `limit`. One it now shows is ingested and leaves the list; past TAIL_DEFER_HOURS the tail gives
   * up — keeping the older state Hivemind does show, if any (better than nothing), else dropping
   * the post, logged (the reconciliation or a backfill can still bring it in).
   */
  private async retryDeferred(m: Map<string, DeferredPost>, limit: number): Promise<{ tried: number; gaveUp: number }> {
    const t = Date.now();
    const due = [...m.values()].filter((d) => d.nextAt <= t).sort((a, b) => a.nextAt - b.nextAt).slice(0, limit);
    let gaveUp = 0;
    if (!due.length) return { tried: 0, gaveUp };
    const maxAgeMs = Math.max(0, num(this.env.TAIL_DEFER_HOURS, DEFER_HOURS)) * 3_600_000;
    const rpc = rpcFor(this.env);
    for (const d of due) {
      const key = refKey(d);
      try {
        // the n-th retry asks the n-th node first (the first ask went to the preferred one)
        const out = await ingestOp(this.env, d, "tail-deferred", rpc, { node: d.tries });
        if (out.kind !== "deferred") {
          m.delete(key);
          if (t - d.since > 60_000) console.log("indexer: Hivemind caught up with a deferred post", key, "block", d.block, `after ${Math.round((t - d.since) / 1000)} s`);
          continue;
        }
        if (t - d.since >= maxAgeMs) {
          if (out.stale) {
            await ingestPost(this.env, out.stale, d.block, "tail-deferred-expired");
            console.warn("indexer gave up waiting for Hivemind and kept the state it shows", key, "block", d.block, out.why);
          } else {
            console.error("indexer gave up on a post Hivemind never returned", key, "block", d.block, out.why);
          }
          m.delete(key);
          gaveUp++;
          continue;
        }
        m.set(key, { ...d, why: out.why.slice(0, WHY_MAX), node: out.node ?? null, tries: d.tries + 1, lastTryAt: t, nextAt: t + deferBackoffMs(d.tries + 1) });
      } catch (e) {
        m.set(key, { ...d, why: errorText(e).slice(0, WHY_MAX), tries: d.tries + 1, lastTryAt: t, nextAt: t + deferBackoffMs(d.tries + 1) });
      }
    }
    return { tried: due.length, gaveUp };
  }

  async alarm(): Promise<void> {
    const s = await this.load();
    if (!s.running || s.cursor === null) return;
    const epoch = s.epoch;
    const perTick = int(this.env.TAIL_BLOCKS_PER_TICK, 200);
    const idleMs = int(this.env.TAIL_IDLE_SECONDS, 3) * 1000;
    const deferPerTick = Math.max(0, int(this.env.TAIL_DEFER_PER_TICK, DEFER_PER_TICK));
    const rpc = rpcFor(this.env);

    try {
      const dgp = await rpc.getDynamicGlobalProperties();
      s.lib = dgp.last_irreversible_block_num;
      s.head = dgp.head_block_number;
      s.ticks++;
      s.lastTickAt = Date.now();

      if (s.cursor > s.lib) {
        // between blocks: the posts Hivemind had not caught up with get their turn
        const deferred = await this.deferredMap();
        const r = await this.retryDeferred(deferred, deferPerTick);
        if (r.tried) {
          await this.saveDeferred(deferred);
          s.deferredGivenUp += r.gaveUp;
        }
        await this.commit(s, epoch, idleMs);
        return;
      }

      // A range that could not be read is read in smaller pieces (a node timing out on a large
      // reply); it grows back after READ_GROW_AFTER good reads.
      const count = Math.min(Math.max(1, perTick >> s.readShrink), s.lib - s.cursor + 1);
      let blocks: Block[];
      try {
        blocks = await rpc.getBlockRange(s.cursor, count);
      } catch (e) {
        s.readShrink = Math.min(READ_SHRINK_MAX, s.readShrink + 1);
        s.readOk = 0;
        throw e;
      }
      if (s.readShrink > 0 && ++s.readOk >= READ_GROW_AFTER) {
        s.readShrink--;
        s.readOk = 0;
      }
      if (!blocks.length) {
        // a node behind the one that gave the irreversible head: nothing to read yet, wait as when idle
        await this.commit(s, epoch, idleMs);
        return;
      }
      const times = blocks.map((b) => chainTime(b.timestamp));
      let before: number | null = null;
      const prevAt = async (i: number): Promise<number> => (i > 0 ? times[i - 1] : (before ??= await this.timeBefore(s, rpc, times[0])));
      const posts = new Map<string, { author: string; permlink: string; block: number; i: number; op: "comment" | "delete" }>();
      const market: CustomJsonOp[] = [];
      const ids = marketIds(this.env);

      blocks.forEach((block, i) => {
        const blockNum = s.cursor! + i;
        for (const tx of block.transactions ?? []) {
          for (const raw of tx.operations ?? []) {
            const op = normalizeOp(raw);
            if (op.type === "comment") {
              const c = op.value as CommentOp;
              if (c.parent_author === "") posts.set(`${c.author}/${c.permlink}`, { author: c.author, permlink: c.permlink, block: blockNum, i, op: "comment" });
            } else if (op.type === "delete_comment") {
              // Rare on Pixagram (posts are "deleted" by editing the body) but cheap to honour.
              const v = op.value as { author: string; permlink: string };
              posts.set(`${v.author}/${v.permlink}`, { author: v.author, permlink: v.permlink, block: blockNum, i, op: "delete" });
            } else if (op.type === "custom_json" && ids.has((op.value as CustomJsonOp).id)) {
              market.push(op.value as CustomJsonOp);
            }
          }
        }
      });

      const tracked = new Set<string>();
      const parked = new Set((await this.parkedList()).map(refKey));
      const deferred = await this.deferredMap();
      const deferredBefore = deferred.size;
      let deferredChanged = false;
      let newlyDeferred = 0;
      const parkAfter = Math.max(1, int(this.env.TAIL_PARK_AFTER, PARK_AFTER));
      for (const p of posts.values()) {
        const key = refKey(p);
        // parked earlier: retryParked() reads its current state (its edits in these blocks are
        // not recorded as versions, so its image date may only be a lower bound)
        if (parked.has(key)) continue;
        const ref: OpRef = { author: p.author, permlink: p.permlink, block: p.block, op: p.op, prevAt: await prevAt(p.i) };
        let out: OpOutcome;
        try {
          out = await ingestOp(this.env, ref, "tail", rpc);
        } catch (e) {
          // The tick fails and is retried, as any failure is: a network blip passes. But one post
          // must not hold the whole chain back — a node asserting on deleted posts once stalled
          // the tail for nine days — so after `parkAfter` ticks in a row stopped on this same
          // post, it is parked and the tail moves on without it.
          const error = errorText(e);
          const ticks = s.stuck?.ref === key && s.stuck.block === ref.block ? s.stuck.ticks + 1 : 1;
          if (ticks < parkAfter) {
            s.stuck = { ref: key, block: ref.block, ticks, error };
            throw e;
          }
          await this.park(ref, error);
          parked.add(key);
          if (deferred.delete(key)) deferredChanged = true; // parked now: retryParked's from here on
          s.stuck = null;
          console.error("indexer parked a post", key, "block", ref.block, error);
          continue;
        }
        if (out.kind === "deferred") {
          // Hivemind is behind: asked again from the next tick on. The op's version is recorded
          // below all the same (the post exists: the block says so).
          const t = Date.now();
          const was = deferred.get(key);
          if (!was) newlyDeferred++;
          deferred.set(key, { ...ref, why: out.why.slice(0, WHY_MAX), node: out.node ?? null, since: was?.since ?? t, lastTryAt: t, nextAt: t + deferBackoffMs(1), tries: (was?.tries ?? 0) + 1 });
          deferredChanged = true;
          tracked.add(key);
        } else {
          if (deferred.delete(key)) deferredChanged = true; // a later op settled it
          if (out.kind === "ingested" && out.r.postId !== null) tracked.add(key);
        }
        s.postsSeen++;
      }
      s.stuck = null; // every post of the range is done with (ingested, deferred or parked)
      if (deferred.size > deferredBefore) console.warn("indexer deferred posts Hivemind does not show yet", deferred.size - deferredBefore, "waiting:", deferred.size);

      // The other steps of the tick: a failure is retried with the tick, as any failure is, but after
      // `stepSkipAfter` ticks in a row failing on the same step of the same blocks, that step is
      // skipped for them (logged, listed in status().skipped) and the tail moves on. The posts are in
      // already; what is lost is their exact edit history for these blocks (the snapshot remains).
      const stepSkipAfter = Math.max(1, int(this.env.TAIL_STEP_SKIP_AFTER, STEP_SKIP_AFTER));
      const first = s.cursor;
      const runStep = async (step: TickStep, fn: () => Promise<void>): Promise<void> => {
        try {
          await fn();
        } catch (e) {
          const error = errorText(e);
          const ticks = s.stuckStep?.step === step && s.stuckStep.cursor === first ? s.stuckStep.ticks + 1 : 1;
          if (ticks < stepSkipAfter) {
            s.stuckStep = { step, cursor: first, ticks, error };
            throw e;
          }
          s.stuckStep = null;
          s.skipped = [...(s.skipped ?? []), { step, from: first, to: first + blocks.length - 1, error, at: Date.now() }].slice(-SKIPPED_MAX);
          s.skippedTotal = (s.skippedTotal ?? 0) + 1;
          console.error(`indexer skipped the ${step} step of blocks ${first}-${first + blocks.length - 1} after ${ticks} failed ticks`, error);
        }
      };
      // Every version of a tracked (or deferred) post, with its exact block, transaction and image hash.
      await runStep("versions", async () => {
        const versions = await versionsFromBlocks(blocks, first, (a, p) => tracked.has(`${a}/${p}`));
        if (!versions.length) return;
        await writeVersions(this.env.DB, versions);
        for (const author of new Set(versions.map((v) => v.author))) await relabelKinds(this.env, author);
        // Artworks already hashed by the consumer pick up the exact times now; ones still in the
        // queue read them when their stats stage runs.
        for (const id of await artworksWithHashes(this.env.DB, versions.map((v) => v.content_hash ?? ""))) await refreshImageHistory(this.env.DB, id);
      });
      if (market.length) await runStep("market", async () => {
        for (const op of market) await handleMarketOp(this.env, op);
      });
      s.stuckStep = null; // both steps done (or skipped) for these blocks

      // the posts deferred on earlier ticks that are due (not those deferred just now: they are due later)
      const retried = await this.retryDeferred(deferred, deferPerTick);
      if (deferredChanged || retried.tried) await this.saveDeferred(deferred);
      // counted once the list they describe is saved (a tick that fails is done again whole)
      s.deferredTotal += newlyDeferred;
      s.deferredGivenUp += retried.gaveUp;

      s.lastProcessed = s.cursor + blocks.length - 1;
      s.cursor = s.cursor + blocks.length;
      if (times.length) s.lastBlockTime = times[times.length - 1];
      s.lastError = null;
      s.stuck = null;
      const behind = s.cursor <= s.lib;
      await this.commit(s, epoch, behind ? 25 : idleMs);
    } catch (e) {
      s.lastError = e instanceof Error ? e.message : String(e);
      s.lastErrorAt = Date.now();
      console.error("indexer tick failed", s.lastError);
      await this.commit(s, epoch, 10_000);
    }
  }
}

/** The single indexer instance. */
export function indexerStub(env: Env) {
  const id = env.INDEXER.idFromName("main");
  return env.INDEXER.get(id) as unknown as DurableObjectStub<ChainIndexer>;
}
