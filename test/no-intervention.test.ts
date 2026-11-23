// v4.8.4: the index keeps itself complete without anyone stepping in.
//
// On 23 November 2026 the live index lacked 18 of the 319 Pixagram posts of the previous four days
// (all never edited: the tail had asked Hivemind before it held them) and all 916 Pixtopia posts:
// the v4.8.3 fixes for both had been lost in a merge on 16 November, and nothing but a backfill run
// by hand brought missed posts back. These tests cover what now does it without a hand: the
// rolling pass over the whole listing, the watchdog, a tail that cannot be held by one step or one
// unreadable range, and an RPC client that uses every node.

import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { ChainIndexer, STALL_MS } from "../src/chain/indexer-do";
import { fullRecord, reconcileFull, reconcileRecent } from "../src/chain/reconcile";
import { ChainRpc, PREFER_MS, resetPreferredNodes, type CondenserPost } from "../src/chain/rpc";
import { ingestPost } from "../src/chain/ingest";
import { healthReport } from "../src/enrich/health";
import { makeEnv, type TestEnv } from "./harness/fakes";

class FakeStorage {
  m = new Map<string, any>();
  alarm: number | null = null;
  async get(k: string) {
    return structuredClone(this.m.get(k));
  }
  async put(k: string, v: any) {
    this.m.set(k, structuredClone(v));
  }
  async getAlarm() {
    return this.alarm;
  }
  async setAlarm(t: number) {
    this.alarm = t;
  }
  async deleteAlarm() {
    this.alarm = null;
  }
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  resetPreferredNodes();
});

const NOW = Date.UTC(2026, 10, 23, 11, 0, 0);
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
const ago = (minutes: number) => iso(NOW / 1000 - minutes * 60);

function clockAt(ms: number) {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(ms);
}

type Reply = { result: unknown } | { error: { code: number; message: string; data?: unknown } };
const doesNotExist = (a: string, p: string): Reply => ({ error: { code: -32602, message: `Assert Exception:Post ${a}/${p} does not exist` } });

/** A post as Hivemind lists it: created `minutesAgo`, an artwork unless the body says otherwise. */
function listed(author: string, permlink: string, minutesAgo: number, o: { app?: string; body?: string; updatedMinutesAgo?: number } = {}): CondenserPost {
  return {
    author,
    permlink,
    parent_author: "",
    parent_permlink: "pixelart",
    category: "pixelart",
    title: permlink,
    body: o.body ?? `Some words about ${permlink}`,
    body_length: (o.body ?? `Some words about ${permlink}`).length,
    json_metadata: JSON.stringify({ app: o.app ?? "pixagram/3.0.2", tags: ["pixelart"] }),
    created: ago(minutesAgo),
    last_update: ago(o.updatedMinutesAgo ?? minutesAgo),
    depth: 0,
    children: 0,
    net_votes: 0,
  };
}

/**
 * A Pixa node for the reconciliation: Hivemind's listing (newest first, the next page after the
 * `start` post, as Pixa's does), get_content from the same posts, and an optional hook to fail a page.
 */
function listingNode(chain: CondenserPost[], o: { failPage?: (q: any) => Reply | null } = {}) {
  const calls: any[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id, params } = JSON.parse(init.body);
      const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
      if (method === "condenser_api.get_discussions_by_created") {
        const q = params[0];
        calls.push(q);
        const fail = o.failPage?.(q);
        if (fail) return send(fail);
        const all = [...chain].sort((a, b) => b.created.localeCompare(a.created));
        let from = 0;
        if (q.start_author) {
          const i = all.findIndex((p) => p.author === q.start_author && p.permlink === q.start_permlink);
          if (i < 0) return send(doesNotExist(q.start_author, q.start_permlink));
          from = i + 1;
        }
        const page = all.slice(from, from + q.limit).map((p) => ({ ...p, body: (p.body ?? "").slice(0, q.truncate_body ?? undefined) }));
        return send({ result: page });
      }
      if (method === "condenser_api.get_content") {
        const p = chain.find((x) => x.author === params[0] && x.permlink === params[1]);
        return send(p ? { result: p } : doesNotExist(params[0], params[1]));
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return calls;
}

async function postsIn(env: TestEnv) {
  const r = await env.DB.prepare("SELECT author, permlink, deleted FROM posts ORDER BY author, permlink").all<{ author: string; permlink: string; deleted: number }>();
  return (r.results ?? []).map((x) => `${x.author}/${x.permlink}${x.deleted ? " (deleted)" : ""}`);
}

// ---- the rolling pass ---------------------------------------------------------------------------

describe("reconcileFull: a rolling pass over the whole listing", () => {
  it("brings back what the index lacks at any age, a few pages a run, then starts again from the newest", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    // 75 posts over two months, one every 20 hours; the index lacks five of them, three of them
    // far older than the 48 hours the nightly reconciliation looks at
    const chain = Array.from({ length: 75 }, (_, i) => listed(`artist${i}`, `work-${i}`, 10 + i * 20 * 60, { app: i % 4 === 0 ? "pixtopia/0.1.0" : "pixagram/3.0.2" }));
    const lost = new Set([1, 30, 52, 60, 74]);
    for (const [i, p] of chain.entries()) if (!lost.has(i)) await ingestPost(env, p, null, "test");
    const calls = listingNode(chain);

    const r1 = await reconcileFull(env, { pages: 2 });
    expect(r1).toMatchObject({ pages: 2, listed: 40, missing: 2, ingested: 2, pass_done: false }); // posts 1 and 30
    let rec = (await fullRecord(env))!;
    expect(rec.current).toMatchObject({ pages: 2, start: { author: "artist39", permlink: "work-39" } });
    expect(calls[1]).toMatchObject({ start_author: "artist19", start_permlink: "work-19", truncate_body: 16 });

    const r2 = await reconcileFull(env, { pages: 2 });
    expect(r2).toMatchObject({ pages: 2, missing: 3, ingested: 3, pass_done: true }); // 52, 60, 74: the end of the listing
    rec = (await fullRecord(env))!;
    expect(rec).toMatchObject({ passes: 1, recovered_total: 5, last_pass: { pages: 4, listed: 75, missing: 5, ingested: 5 } });
    expect(rec.current).toMatchObject({ start: null, pages: 0 });
    expect(await postsIn(env)).toHaveLength(75);
    expect(env._queue.sent.filter((m) => m.reason === "reconcile-full")).toHaveLength(5);

    // the next pass starts from the newest and finds nothing left
    const r3 = await reconcileFull(env, { pages: 10 });
    expect(r3).toMatchObject({ pages: 4, missing: 0, ingested: 0, pass_done: true });
    expect(calls[calls.length - 4].start_author).toBeUndefined();
  });

  it("checks its ingest budget between pages: the page in hand is finished, the next run goes on after it", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const chain = Array.from({ length: 40 }, (_, i) => listed(`a${i}`, `p${i}`, 10 + i));
    listingNode(chain);
    const r1 = await reconcileFull(env, { pages: 10, maxIngest: 5 });
    expect(r1).toMatchObject({ pages: 1, ingested: 20, pass_done: false }); // over the budget within the page, then stops
    const r2 = await reconcileFull(env, { pages: 10, maxIngest: 5 });
    expect(r2).toMatchObject({ pages: 1, ingested: 20 });
    expect(await postsIn(env)).toHaveLength(40);
  });

  it("leaves the newest five minutes to the tail, and the apps APP_PREFIXES leaves out", async () => {
    const env = makeEnv({ APP_PREFIXES: "pixagram" } as any);
    clockAt(NOW);
    const chain = [listed("jesus", "calisthenics", 2), listed("conayi", "flowers", 30, { app: "pixtopia/0.1.0" }), listed("ismeris", "cat", 40)];
    listingNode(chain);
    const r = await reconcileFull(env, { pages: 1 });
    expect(r).toMatchObject({ listed: 2, ingested: 1, pass_done: true });
    expect((await fullRecord(env))!.last_pass).toMatchObject({ skipped_app: 1 });
    expect(await postsIn(env)).toEqual(["ismeris/cat"]);
  });

  it("revives a post the index marked deleted while the chain shows it; leaves posts deleted by an edit alone", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const alive = listed("magareigns", "little-friends", 60);
    const gone = listed("cayitus63", "felices-fiestas", 70, { body: "deleted" });
    const goneUnknown = listed("someone", "deleted-before-the-index-saw-it", 80, { body: "deleted" });
    await ingestPost(env, alive, null, "test");
    await ingestPost(env, gone, null, "test");
    // what the old tail did when a lagging Hivemind said "does not exist" about an indexed post
    await env.DB.prepare("UPDATE posts SET deleted = 1 WHERE author = 'magareigns'").run();
    listingNode([alive, gone, goneUnknown]);

    const r = await reconcileFull(env, { pages: 5 });
    expect(r).toMatchObject({ revived: 1, missing: 0, ingested: 1 });
    expect(await postsIn(env)).toEqual(["cayitus63/felices-fiestas (deleted)", "magareigns/little-friends"]);
    // and it is enriched again (stages queued) like any post coming back
    expect(env._queue.sent.map((m) => `${m.author}:${m.reason}`)).toContain("magareigns:reconcile-full");
  });

  it("the window reconciliation revives such a post too", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const alive = listed("purpleidy", "market-view", 30);
    await ingestPost(env, alive, null, "test");
    await env.DB.prepare("UPDATE posts SET deleted = 1").run();
    listingNode([alive]);
    expect(await reconcileRecent(env, { hours: 3 })).toMatchObject({ revived: 1, ingested: 1 });
    expect(await postsIn(env)).toEqual(["purpleidy/market-view"]);
  });

  it("starts over when the post it went on from is gone from the listing", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const chain = Array.from({ length: 30 }, (_, i) => listed(`a${i}`, `p${i}`, 10 + i));
    listingNode(chain);
    await reconcileFull(env, { pages: 1 });
    expect((await fullRecord(env))!.current.start).toMatchObject({ author: "a19" });
    chain.splice(19, 1); // a19 deleted (delete_comment): Hivemind no longer pages after it
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = await reconcileFull(env, { pages: 1 });
    expect(r).toMatchObject({ restarted: true, pages: 0 });
    expect((await fullRecord(env))!.current).toMatchObject({ start: null, pages: 0 });
    expect(warn.mock.calls.some((c) => String(c[0]).includes("starting over"))).toBe(true);
  });

  it("a page that fails run after run does not hold the pass: it starts over after six runs", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const chain = Array.from({ length: 30 }, (_, i) => listed(`a${i}`, `p${i}`, 10 + i));
    let broken = false;
    listingNode(chain, { failPage: (q) => (broken && q.start_author ? { error: { code: -32603, message: "Internal Error", data: "statement timeout" } } : null) });
    await reconcileFull(env, { pages: 1 });
    broken = true;
    vi.spyOn(console, "error").mockImplementation(() => {});
    for (let i = 1; i < 6; i++) {
      const r = await reconcileFull(env, { pages: 1 });
      expect(r!.error).toMatch(/statement timeout/);
      expect((await fullRecord(env))!.failed_runs).toBe(i);
    }
    const r = await reconcileFull(env, { pages: 1 });
    expect(r).toMatchObject({ restarted: true });
    expect((await fullRecord(env))!).toMatchObject({ failed_runs: 0, current: { start: null } });
  });

  it("a node that ignores the start post is an error, not a pass that reads its first page for ever", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const chain = Array.from({ length: 30 }, (_, i) => listed(`a${i}`, `p${i}`, 10 + i));
    let ignoreStart = false;
    const first = [...chain].sort((a, b) => b.created.localeCompare(a.created)).slice(0, 20);
    listingNode(chain, { failPage: (q) => (ignoreStart && q.start_author ? { result: first } : null) });
    await reconcileFull(env, { pages: 1 });
    ignoreStart = true;
    const r = await reconcileFull(env, { pages: 3 });
    expect(r!.error).toMatch(/start_author ignored/);
    expect((await fullRecord(env))!.current).toMatchObject({ pages: 1, start: { author: "a19" } });
  });

  it("is off with RECONCILE_FULL_PAGES=0", async () => {
    const env = makeEnv({ RECONCILE_FULL_PAGES: "0" } as any);
    expect(await reconcileFull(env)).toBeNull();
    expect(await fullRecord(env)).toBeNull();
  });
});

// ---- the tail -----------------------------------------------------------------------------------

const BASE = 2_300_000;
const T0 = Date.UTC(2026, 10, 23, 10, 0, 0) / 1000;
const timeOf = (n: number) => T0 + 3 * (n - BASE);

/** A node for the tail: blocks with their real spacing, every post shown at once by Hivemind. */
function tailNode(o: { lib: number; ops?: Record<number, unknown[]>; failRead?: (from: number, count: number) => boolean }) {
  const calls: Array<{ method: string; params: any }> = [];
  const n = { lib: o.lib, calls };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id, params } = JSON.parse(init.body);
      calls.push({ method, params });
      const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
      if (method === "database_api.get_dynamic_global_properties") return send({ result: { head_block_number: n.lib, last_irreversible_block_num: n.lib, time: iso(timeOf(n.lib)) } });
      if (method === "block_api.get_block_range") {
        const { starting_block_num: from, count } = params;
        if (o.failRead?.(from, count)) return new Response("upstream timed out", { status: 504 });
        const blocks = [];
        for (let b = from; b < from + count && b <= n.lib; b++) blocks.push({ timestamp: iso(timeOf(b)), transactions: (o.ops?.[b] ?? []).map((op) => ({ operations: [op] })) });
        return send({ result: { blocks } });
      }
      if (method === "condenser_api.get_content") {
        return send({
          result: {
            author: params[0], permlink: params[1], parent_author: "", parent_permlink: "pixelart", category: "pixelart", title: params[1], body: "words",
            json_metadata: JSON.stringify({ app: "pixagram/3.0.2" }), created: iso(T0), last_update: iso(T0 + 86_400), depth: 0, children: 0,
          },
        });
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return n;
}

const commentOp = (author: string, permlink: string) => ({
  type: "comment_operation",
  value: { parent_author: "", parent_permlink: "pixelart", author, permlink, title: permlink, body: "words", json_metadata: JSON.stringify({ app: "pixagram/3.0.2" }) },
});

function indexer(over: Record<string, string> = {}) {
  const storage = new FakeStorage();
  const env = makeEnv({ APP_PREFIXES: "", TAIL_BLOCKS_PER_TICK: "100", ...over } as any);
  const doo = new ChainIndexer({ storage } as any, env as any);
  return { storage, env, doo };
}

describe("the watchdog", () => {
  it("starts a tail that was never started (a fresh deployment) at the irreversible head", async () => {
    const { storage, doo } = indexer();
    tailNode({ lib: BASE + 50 });
    expect(await doo.watchdog()).toEqual({ action: "started", why: "never started" });
    const s = await doo.status();
    expect(s).toMatchObject({ running: true, cursor: BASE + 50, stoppedAt: null });
    expect(storage.alarm).not.toBeNull();
  });

  it("leaves a tail stopped by stop() alone, and /healthz says who stopped it", async () => {
    const { storage, doo } = indexer();
    tailNode({ lib: BASE + 50 });
    await doo.start();
    await doo.stop();
    expect(await doo.watchdog()).toEqual({ action: "none" });
    expect((await doo.status()).running).toBe(false);
    expect(storage.alarm).toBeNull();
    const env = makeEnv();
    (env as any).INDEXER = { idFromName: () => "main", get: () => ({ status: () => doo.status() }) };
    expect((await healthReport(env)).problems).toEqual(["chain indexer stopped by POST /admin/indexer/stop: new posts are not indexed (POST /admin/indexer/start)"]);
  });

  it("re-arms a running tail whose alarm is gone, or that has not ticked for a quarter of an hour", async () => {
    const { storage, doo } = indexer();
    tailNode({ lib: BASE + 50 });
    const t0 = Date.now();
    clockAt(t0);
    await storage.put("state", { running: true, cursor: BASE + 51, lastTickAt: t0 });
    expect(await doo.watchdog()).toMatchObject({ action: "rearmed", why: "no alarm" });
    expect(storage.alarm).toBe(t0 + 10);

    // healthy: an alarm, a recent tick
    expect(await doo.watchdog()).toEqual({ action: "none" });

    // an alarm that is set but never fires: no tick for STALL_MS
    storage.alarm = t0 + 3_600_000;
    clockAt(t0 + STALL_MS + 60_000);
    expect(await doo.watchdog()).toMatchObject({ action: "rearmed", why: expect.stringContaining("no tick for 16 min") });
    expect(storage.alarm).toBe(t0 + STALL_MS + 60_000 + 10);
    expect(await doo.ensureAlarm()).toBe(true); // the old name still works (and still stalled)
  });
});

describe("one step or one range cannot hold the tail", () => {
  it("a history step that keeps failing on the same blocks is skipped after TAIL_STEP_SKIP_AFTER ticks; the posts are in", async () => {
    const { storage, env, doo } = indexer({ TAIL_STEP_SKIP_AFTER: "3" });
    tailNode({ lib: BASE + 5, ops: { [BASE + 2]: [commentOp("anti", "whopper")] } });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    // post_versions refuses every write (a D1 limit, a schema problem: the same failure every time)
    const prepare = env.DB.prepare.bind(env.DB);
    vi.spyOn(env.DB, "prepare").mockImplementation((sql: string) => {
      if (/INSERT INTO post_versions/.test(sql)) throw new Error("D1_ERROR: string or blob too big");
      return prepare(sql);
    });
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});

    await doo.alarm();
    await doo.alarm();
    let s = await doo.status();
    expect(s.cursor).toBe(BASE + 1); // retried, as any failure is
    expect(s.stuckStep).toMatchObject({ step: "versions", cursor: BASE + 1, ticks: 2 });
    expect(s.lastError).toMatch(/string or blob too big/);

    await doo.alarm(); // the third: skipped, the tail moves on
    s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.cursor).toBe(BASE + 6);
    expect(s.stuckStep).toBeNull();
    expect(s.skippedTotal).toBe(1);
    expect(s.skipped).toMatchObject([{ step: "versions", from: BASE + 1, to: BASE + 5, error: "D1_ERROR: string or blob too big" }]);
    expect(await postsIn(env)).toEqual(["anti/whopper"]);
    expect(errors.mock.calls.some((c) => String(c[0]).includes("skipped the versions step"))).toBe(true);
  });

  it("a step that fails once is retried, not skipped", async () => {
    const { storage, env, doo } = indexer({ TAIL_STEP_SKIP_AFTER: "2" });
    tailNode({ lib: BASE + 3, ops: { [BASE + 2]: [commentOp("anti", "once")] } });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    const prepare = env.DB.prepare.bind(env.DB);
    let fail = 1;
    vi.spyOn(env.DB, "prepare").mockImplementation((sql: string) => {
      if (/INSERT INTO post_versions/.test(sql) && fail-- > 0) throw new Error("D1_ERROR: network connection lost");
      return prepare(sql);
    });
    vi.spyOn(console, "error").mockImplementation(() => {});
    await doo.alarm();
    await doo.alarm();
    const s = await doo.status();
    expect(s).toMatchObject({ cursor: BASE + 4, skippedTotal: 0, stuckStep: null, lastError: null });
    const v = await env.DB.prepare("SELECT COUNT(*) AS n FROM post_versions WHERE author = 'anti'").first<{ n: number }>();
    expect(v!.n).toBe(1); // the history made it on the retry
  });

  it("a range that cannot be read is read in smaller pieces, and grows back after good reads", async () => {
    const { storage, doo } = indexer();
    const n = tailNode({ lib: BASE + 2000, failRead: (_f, count) => count > 25 });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    vi.spyOn(console, "error").mockImplementation(() => {});
    for (let i = 0; i < 3; i++) await doo.alarm(); // 100, 50: fail; 25: read
    const reads = () => n.calls.filter((c) => c.method === "block_api.get_block_range").map((c) => c.params.count);
    // every node is tried for each failed read: here there is one (rpc.test), tried three times
    expect([...new Set(reads())]).toEqual([100, 50, 25]);
    let s = await doo.status();
    expect(s).toMatchObject({ cursor: BASE + 26, readShrink: 2, lastError: null });
    for (let i = 0; i < 19; i++) await doo.alarm();
    s = await doo.status();
    expect(s).toMatchObject({ readShrink: 1, cursor: BASE + 26 + 19 * 25 }); // twenty good reads: 50 is tried again…
    await doo.alarm();
    s = await doo.status();
    expect(reads().at(-1)).toBe(50);
    expect(s).toMatchObject({ readShrink: 2, cursor: BASE + 26 + 19 * 25 }); // …and, still too much for this node, halved again
  });

  it("a node behind the one that gave the head returns no blocks: the tail waits as when idle, it does not spin", async () => {
    const { storage, doo } = indexer({ TAIL_IDLE_SECONDS: "3" });
    tailNode({ lib: BASE + 10 });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: unknown, init: any) => {
        const { method, id } = JSON.parse(init.body);
        const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
        if (method === "database_api.get_dynamic_global_properties") return send({ result: { head_block_number: BASE + 10, last_irreversible_block_num: BASE + 10, time: iso(timeOf(BASE + 10)) } });
        return send({ result: { blocks: [] } });
      }),
    );
    await storage.put("state", { running: true, cursor: BASE + 5 });
    const t0 = Date.now();
    clockAt(t0);
    await doo.alarm();
    expect((await doo.status()).cursor).toBe(BASE + 5);
    expect(storage.alarm).toBe(t0 + 3_000);
  });
});

describe("a node whose Hivemind has stalled (18–19 Nov 2026: 23 hours of posts lost)", () => {
  // From block 2,160,277 (18 Nov, 13:24 UTC) to 2,188,033 (19 Nov, 12:32 UTC) the live index got
  // no post at all, and none came back: ids 2443 and 2444 are consecutive. A Hivemind that stops
  // syncing keeps answering, from its old state: "does not exist" for every new post. Every node
  // runs a Hivemind of its own, so a deferred post's retries ask the other nodes.
  it("a deferred post's retry asks the next node, which holds it; the preferred node stays the primary", async () => {
    const primary = "https://api.pixagram.test";
    const second = "https://merlion.test";
    const { storage, env, doo } = indexer({ RPC_URL: primary, RPC_FALLBACK_URLS: second } as any);
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string, init: any) => {
        const { method, id, params } = JSON.parse(init.body);
        const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
        if (method === "database_api.get_dynamic_global_properties") {
          asked.push(`${u} dgp`);
          return send({ result: { head_block_number: BASE + 2, last_irreversible_block_num: BASE + 2, time: iso(timeOf(BASE + 2)) } });
        }
        if (method === "block_api.get_block_range") {
          const blocks = [];
          for (let b = params.starting_block_num; b < params.starting_block_num + params.count && b <= BASE + 2; b++) {
            blocks.push({ timestamp: iso(timeOf(b)), transactions: b === BASE + 1 ? [{ operations: [commentOp("germandude", "perkz-in-gta6-1795012809893")] }] : [] });
          }
          return send({ result: { blocks } });
        }
        if (method === "condenser_api.get_content") {
          asked.push(`${u} get_content`);
          if (u === primary) return send(doesNotExist(params[0], params[1])); // stalled at 13:24
          return send({
            result: {
              author: params[0], permlink: params[1], parent_author: "", parent_permlink: "pixelart", category: "pixelart", title: "Perkz in GTA6", body: "words",
              json_metadata: JSON.stringify({ app: "pixagram/3.0.2" }), created: iso(timeOf(BASE)), last_update: iso(timeOf(BASE)), depth: 0, children: 0,
            },
          });
        }
        throw new Error(`unexpected ${method}`);
      }),
    );
    await storage.put("state", { running: true, cursor: BASE + 1 });
    const t0 = Date.now();
    clockAt(t0);
    await doo.alarm();
    let s = await doo.status();
    expect(s.deferred[0]).toMatchObject({ author: "germandude", node: primary, tries: 1 });
    expect(await postsIn(env)).toEqual([]);

    clockAt(t0 + 6_000); // the first retry: the second node first
    await doo.alarm();
    s = await doo.status();
    expect(s.deferredWaiting).toBe(0);
    expect(await postsIn(env)).toEqual(["germandude/perkz-in-gta6-1795012809893"]);
    expect(asked.filter((a) => a.endsWith("get_content"))).toEqual([`${primary} get_content`, `${second} get_content`]);
    // the rotated retry did not make the second node the preferred one: the next tick asks the primary
    await doo.alarm();
    expect(asked.at(-1)).toBe(`${primary} dgp`);
  });

  it("a parked post's retries start with another node than the one the tail failed on", async () => {
    const nodes = ["https://a.test", "https://b.test", "https://c.test"];
    const rpc = new ChainRpc({ url: nodes[0], fallbacks: nodes.slice(1) });
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string, init: any) => {
        asked.push(u);
        const { id } = JSON.parse(init.body);
        return new Response(JSON.stringify({ jsonrpc: "2.0", id, result: { author: "a", permlink: "p" } }));
      }),
    );
    for (const from of [0, 1, 2, 3, 4]) await rpc.getContentState("a", "p", { from });
    expect(asked).toEqual([nodes[0], nodes[1], nodes[2], nodes[0], nodes[1]]);
    expect(rpc.nodes()[0]).toBe(nodes[0]);
  });
});

// ---- the RPC client -----------------------------------------------------------------------------

describe("ChainRpc uses every node", () => {
  const nodes = ["https://api.pixagram.test", "https://merlion.test", "https://blockforge.test", "https://pixarex.test", "https://boitata.test", "https://dubai.test"];
  function nodesUp(up: Set<string>) {
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string, init: any) => {
        asked.push(u);
        if (!up.has(u)) throw new TypeError("fetch failed");
        const { id } = JSON.parse(init.body);
        return new Response(JSON.stringify({ jsonrpc: "2.0", id, result: { node: u } }));
      }),
    );
    return asked;
  }
  const rpc = () => new ChainRpc({ url: nodes[0], fallbacks: nodes.slice(1), timeoutMs: 1_000 });

  it("goes past the first two fallbacks (three nodes down used to stop the tail while three answered)", async () => {
    const asked = nodesUp(new Set([nodes[3]]));
    expect(await rpc().call("x")).toEqual({ node: nodes[3] });
    expect(asked).toEqual(nodes.slice(0, 4));
  });

  it("asks the node that answered first for ten minutes, then the primary again", async () => {
    const asked = nodesUp(new Set([nodes[3]]));
    await rpc().call("x");
    asked.length = 0;
    await rpc().call("y"); // a new client (each tick makes one): the preference is per isolate
    expect(asked).toEqual([nodes[3]]);
    asked.length = 0;
    clockAt(Date.now() + PREFER_MS + 1);
    await rpc().call("z");
    expect(asked).toEqual(nodes.slice(0, 4)); // the primary first again: still down
  });

  it("retries: 0 (a request path's single try, /similar's chain read) stays one try, on the node that answered lately", async () => {
    const asked = nodesUp(new Set([nodes[3]]));
    const once = () => new ChainRpc({ url: nodes[0], fallbacks: nodes.slice(1), timeoutMs: 1_000, retries: 0 });
    await expect(once().call("x")).rejects.toThrow(/fetch failed/);
    expect(asked).toEqual([nodes[0]]);
    await rpc().call("x"); // the tail finds the node that answers
    asked.length = 0;
    expect(await once().call("y")).toEqual({ node: nodes[3] });
    expect(asked).toEqual([nodes[3]]);
  });

  it("every node down: the error says so, after one try each", async () => {
    const asked = nodesUp(new Set());
    await expect(rpc().call("x")).rejects.toThrow(/fetch failed/);
    expect(asked).toEqual(nodes);
  });

  it("the timeout covers the reply's body: a node that stalls after its headers is left for the next", async () => {
    let first = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: string, init: any) => {
        const { id } = JSON.parse(init.body);
        if (first) {
          first = false;
          // headers at once, then a body that never ends (until the request is aborted)
          return new Response(new ReadableStream({ start: (c) => init.signal.addEventListener("abort", () => c.error(new DOMException("The operation was aborted", "AbortError"))) }));
        }
        return new Response(JSON.stringify({ jsonrpc: "2.0", id, result: "ok" }));
      }),
    );
    const t = Date.now();
    expect(await new ChainRpc({ url: nodes[0], fallbacks: [nodes[1]], timeoutMs: 200 }).call("x")).toBe("ok");
    expect(Date.now() - t).toBeLessThan(5_000);
  });

  it("an assert is an answer: not retried on other nodes", async () => {
    const asked: string[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (u: string, init: any) => {
        asked.push(u);
        const { id } = JSON.parse(init.body);
        return new Response(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32602, message: "Assert Exception:Post a/b does not exist" } }));
      }),
    );
    await expect(rpc().call("condenser_api.get_content", ["a", "b"])).rejects.toThrow(/does not exist/);
    expect(asked).toEqual([nodes[0]]);
  });
});

// ---- /healthz and the deployed configuration ----------------------------------------------------

describe("/healthz: the rolling pass and the cron", () => {
  function withIndexer(env: TestEnv) {
    (env as any).INDEXER = { idFromName: () => "main", get: () => ({ status: async () => ({ running: true, cursor: 101, lib: 100, lastTickAt: Date.now(), parked: [], deferredWaiting: 0, deferred: [], skipped: [], skippedTotal: 0, stoppedAt: null }) }) };
    return env;
  }

  it("reports the pass in progress and the last complete one", async () => {
    const env = withIndexer(makeEnv({ APP_PREFIXES: "" } as any));
    clockAt(NOW);
    listingNode([listed("a", "one", 30), listed("b", "two", 40)]);
    await reconcileRecent(env, { hours: 3 });
    await reconcileFull(env, { pages: 5 });
    const r = await healthReport(env);
    expect(r.status).toBe("ok");
    // the window reconciliation brought both in; the full pass that followed found nothing left
    expect(r.tail?.full_pass).toMatchObject({ passes: 1, pages: 0, last_pass_pages: 1, last_pass_ingested: 0, recovered_total: 0, last_run_age_s: 0, last_error: null });
    expect(r.tail?.reconcile).toMatchObject({ recovered_total: 2 });
  });

  it("degraded when the ten-minute cron has stopped running (nothing would restart a tail that stops)", async () => {
    const env = withIndexer(makeEnv({ APP_PREFIXES: "" } as any));
    clockAt(NOW);
    listingNode([listed("a", "one", 30)]);
    await reconcileRecent(env, { hours: 3 });
    await reconcileFull(env, { pages: 5 });
    clockAt(NOW + 45 * 60_000);
    expect((await healthReport(env)).problems).toEqual(["the ten-minute cron has not run for 45 min (watchdog, sweeper, reconciliation)"]);
  });
});

describe("the deployed configuration", () => {
  const text = readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8");
  it("has one limits key (a second one silently replaced the first)", () => {
    expect(text.match(/^\s*"limits"\s*:/gm)).toHaveLength(1);
    expect(text).toMatch(/"limits":\s*\{\s*"cpu_ms":\s*300000,\s*"subrequests":\s*1000000\s*\}/);
  });
  it("runs the rolling pass and indexes every front end", () => {
    expect(Number(text.match(/"RECONCILE_FULL_PAGES":\s*"(\d+)"/)?.[1])).toBeGreaterThan(0);
    expect(text.match(/"APP_PREFIXES":\s*"([^"]*)"/)?.[1]).toBe("");
  });
});
