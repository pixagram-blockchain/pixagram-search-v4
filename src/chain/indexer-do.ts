// ChainIndexer: a single Durable Object that tails the Pixa chain.
//
// One named instance ("main") holds the block cursor and re-arms its own alarm. It reads only
// irreversible blocks (no reorg handling needed), picks top-level comment ops and marketplace
// custom_json ops out of each block, fetches the post's current state with get_content and
// upserts it. Everything else (votes, follows, transfers) is ignored.
//
// A post whose ingestion keeps failing is set aside ("parked") after a minute of failed ticks, so
// one post can never hold the whole chain back; the ten-minute cron tries parked posts again.

import { DurableObject } from "cloudflare:workers";
import type { Env } from "../env";
import { int } from "../env";
import { normalizeOp, type CommentOp, type CustomJsonOp } from "./rpc";
import { ingestPostRef, rpcFor, type IngestOutcome } from "./ingest";
import { handleMarketOp, marketIds } from "./market";
import { artworksWithHashes, refreshImageHistory, relabelKinds, versionsFromBlocks, writeVersions } from "./versions";

interface PostRef {
  author: string;
  permlink: string;
  block: number;
}

/** A post the tail set aside because ingesting it kept failing; retryParked() tries it again. */
export interface ParkedPost extends PostRef {
  error: string;
  parkedAt: number;
  /** tries by retryParked() so far */
  attempts: number;
  lastTryAt: number;
}

interface IndexerState {
  running: boolean;
  cursor: number | null; // next block to process
  lastProcessed: number | null;
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
}

const DEFAULT_STATE: IndexerState = {
  running: false,
  cursor: null,
  lastProcessed: null,
  lib: null,
  head: null,
  lastTickAt: null,
  lastError: null,
  lastErrorAt: null,
  postsSeen: 0,
  ticks: 0,
  epoch: 0,
  stuck: null,
};

/** Failed ticks in a row on one post before it is parked: about a minute (a failed tick re-arms in 10 s). */
const PARK_AFTER = 6;
/** Posts parked at most (one storage value); past that the longest-parked goes, logged (a backfill of its author brings it back). */
const PARKED_MAX = 200;
/** Parked posts tried again per cron run. */
const RETRY_PER_RUN = 10;

const refKey = (r: { author: string; permlink: string }): string => `${r.author}/${r.permlink}`;
const errorText = (e: unknown): string => (e instanceof Error ? e.message : String(e)).slice(0, 240);

/** Ingest a post the chain mentioned; one the chain no longer holds is marked deleted. */
async function ingestRef(env: Env, ref: PostRef, reason: string): Promise<IngestOutcome> {
  const r = await ingestPostRef(env, ref.author, ref.permlink, ref.block, reason);
  if (r.action === "missing") {
    // delete_comment on a post without votes really removes it from state.
    await env.DB.prepare("UPDATE posts SET deleted = 1 WHERE author = ? AND permlink = ?").bind(ref.author, ref.permlink).run();
  }
  return r;
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

  private async park(ref: PostRef, error: string): Promise<void> {
    const now = Date.now();
    const list = (await this.parkedList()).filter((p) => refKey(p) !== refKey(ref));
    list.push({ author: ref.author, permlink: ref.permlink, block: ref.block, error, parkedAt: now, attempts: 0, lastTryAt: now });
    while (list.length > PARKED_MAX) {
      const i = list.reduce((oldest, p, j) => (p.parkedAt < list[oldest].parkedAt ? j : oldest), 0);
      console.error("indexer dropped a parked post (too many parked)", refKey(list[i]), list[i].error);
      list.splice(i, 1);
    }
    await this.ctx.storage.put("parked", list);
  }

  // ---- control API (called by the Worker's admin routes) ------------------------

  async status(): Promise<IndexerState & { alarmAt: number | null; parked: ParkedPost[] }> {
    const s = await this.load();
    return { ...s, alarmAt: await this.ctx.storage.getAlarm(), parked: await this.parkedList() };
  }

  /**
   * Try parked posts again (the ten-minute cron), from the front of the list: the list is kept in
   * order of last activity — a post is parked at the back, and one tried and still failing goes
   * back there, its attempt counted — so no post waits behind ones that keep failing. One that
   * ingests now (or turns out deleted) leaves the list. The list is re-read before it is written:
   * a tick may park another post while this awaits the chain.
   */
  async retryParked(limit = RETRY_PER_RUN): Promise<{ retried: number; ingested: number; left: number }> {
    const batch = (await this.parkedList()).slice(0, Math.max(0, limit));
    if (!batch.length) return { retried: 0, ingested: 0, left: 0 };
    const done = new Set<string>();
    const failed = new Map<string, { error: string; at: number }>();
    for (const p of batch) {
      try {
        await ingestRef(this.env, p, "parked");
        done.add(refKey(p));
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
    s.lastError = null;
    s.stuck = null;
    s.epoch++;
    await this.save(s);
    await this.ctx.storage.setAlarm(Date.now() + 10);
    return s;
  }

  async stop(): Promise<IndexerState> {
    const s = await this.load();
    s.running = false;
    s.epoch++;
    await this.save(s);
    await this.ctx.storage.deleteAlarm();
    return s;
  }

  /** Watchdog: if running but the alarm is gone (it happens after failed deploys), re-arm it. */
  async ensureAlarm(): Promise<boolean> {
    const s = await this.load();
    if (!s.running) return false;
    if ((await this.ctx.storage.getAlarm()) === null) {
      await this.ctx.storage.setAlarm(Date.now() + 10);
      return true;
    }
    return false;
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
      await this.save({ ...cur, ticks: s.ticks, postsSeen: s.postsSeen, lib: s.lib, head: s.head, lastTickAt: s.lastTickAt });
      return;
    }
    await this.save(s);
    await this.ctx.storage.setAlarm(Date.now() + nextAlarmMs);
  }

  async alarm(): Promise<void> {
    const s = await this.load();
    if (!s.running || s.cursor === null) return;
    const epoch = s.epoch;
    const perTick = int(this.env.TAIL_BLOCKS_PER_TICK, 200);
    const idleMs = int(this.env.TAIL_IDLE_SECONDS, 3) * 1000;
    const rpc = rpcFor(this.env);

    try {
      const dgp = await rpc.getDynamicGlobalProperties();
      s.lib = dgp.last_irreversible_block_num;
      s.head = dgp.head_block_number;
      s.ticks++;
      s.lastTickAt = Date.now();

      if (s.cursor > s.lib) {
        await this.commit(s, epoch, idleMs);
        return;
      }

      const count = Math.min(perTick, s.lib - s.cursor + 1);
      const blocks = await rpc.getBlockRange(s.cursor, count);
      const posts = new Map<string, { author: string; permlink: string; block: number }>();
      const market: CustomJsonOp[] = [];
      const ids = marketIds(this.env);

      blocks.forEach((block, i) => {
        const blockNum = s.cursor! + i;
        for (const tx of block.transactions ?? []) {
          for (const raw of tx.operations ?? []) {
            const op = normalizeOp(raw);
            if (op.type === "comment") {
              const c = op.value as CommentOp;
              if (c.parent_author === "") posts.set(`${c.author}/${c.permlink}`, { author: c.author, permlink: c.permlink, block: blockNum });
            } else if (op.type === "delete_comment") {
              // Rare on Pixagram (posts are "deleted" by editing the body) but cheap to honour.
              const v = op.value as { author: string; permlink: string };
              posts.set(`${v.author}/${v.permlink}`, { author: v.author, permlink: v.permlink, block: blockNum });
            } else if (op.type === "custom_json" && ids.has((op.value as CustomJsonOp).id)) {
              market.push(op.value as CustomJsonOp);
            }
          }
        }
      });

      const tracked = new Set<string>();
      const parked = new Set((await this.parkedList()).map(refKey));
      const parkAfter = Math.max(1, int(this.env.TAIL_PARK_AFTER, PARK_AFTER));
      for (const ref of posts.values()) {
        const key = refKey(ref);
        // parked earlier: retryParked() reads its current state (its edits in these blocks are
        // not recorded as versions, so its image date may only be a lower bound)
        if (parked.has(key)) continue;
        let r: IngestOutcome;
        try {
          r = await ingestRef(this.env, ref, "tail");
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
          s.stuck = null;
          console.error("indexer parked a post", key, "block", ref.block, error);
          continue;
        }
        if (r.postId !== null) tracked.add(key);
        s.postsSeen++;
      }
      // Every version of a tracked post, with its exact block, transaction and image hash.
      const versions = await versionsFromBlocks(blocks, s.cursor, (a, p) => tracked.has(`${a}/${p}`));
      if (versions.length) {
        await writeVersions(this.env.DB, versions);
        for (const author of new Set(versions.map((v) => v.author))) await relabelKinds(this.env, author);
        // Artworks already hashed by the consumer pick up the exact times now; ones still in the
        // queue read them when their stats stage runs.
        for (const id of await artworksWithHashes(this.env.DB, versions.map((v) => v.content_hash ?? ""))) await refreshImageHistory(this.env.DB, id);
      }
      for (const op of market) await handleMarketOp(this.env, op);

      s.lastProcessed = s.cursor + blocks.length - 1;
      s.cursor = s.cursor + blocks.length;
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
