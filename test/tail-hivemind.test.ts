// v4.8.3: new posts must reach the index. Two leaks, found on 16 November 2026 when "search stops
// working every four days" turned out to mean "new posts are missing":
//
// 1. The tail read each irreversible block about three seconds after it was produced and asked
//    get_content for every post in it. Hivemind, which answers get_content, had sometimes not
//    processed that block yet and said "Post a/p does not exist": the tail took that for a
//    deletion and dropped the post for good. 46 Pixagram posts between 28 October and 15 November,
//    seven of them in one hour on the evening of 12 November (20:08–21:16), restored only by
//    backfills run by hand. Hivemind stamps a post with the time of the block *before* its op
//    (measured on api.pixagram.com: an op in the block of 03:00:12 gives last_update 03:00:09),
//    which is what tells a current answer from a stale one.
// 2. APP_PREFIXES was "pixagram", and Pixtopia (app "pixtopia/0.1.0") started posting on
//    1 November: by the 15th it wrote 160 posts a day to Pixagram's 47, none of them indexed.

import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { ChainIndexer, DEFERRED_MAX, deferBackoffMs } from "../src/chain/indexer-do";
import { reconcileRecent, reconcileRecord } from "../src/chain/reconcile";
import { ingestPost } from "../src/chain/ingest";
import { healthReport } from "../src/enrich/health";
import { makeEnv, type TestEnv } from "./harness/fakes";
import type { CondenserPost } from "../src/chain/rpc";

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
});

const BASE = 2_000_000;
/** 12 Nov 2026 20:08:00 UTC: block BASE. One block every three seconds. */
const T = Date.UTC(2026, 10, 12, 20, 8, 0) / 1000;
const timeOf = (n: number) => T + 3 * (n - BASE);
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);

type Reply = { result: unknown } | { error: { code: number; message: string; data?: unknown } };
const doesNotExist = (a: string, p: string): Reply => ({
  error: { code: -32602, message: `Assert Exception:Post ${a}/${p} does not exist`, data: { code: 10, name: "assert_exception", message: "Assert Exception", stack: [] } },
});
const wasDeleted = (a: string, p: string): Reply => ({ error: { code: -31999, message: "Invalid parameters", data: `Post ${a}/${p} was deleted 1 time(s)` } });

/** A post as Hivemind shows it after the op of block `opBlock`: last_update is the block before's time. */
function post(author: string, permlink: string, o: { createdBlock: number; opBlock?: number; title?: string; app?: string }): CondenserPost {
  return {
    author,
    permlink,
    parent_author: "",
    parent_permlink: "pixelart",
    category: "pixelart",
    title: o.title ?? permlink,
    body: `A few words about ${o.title ?? permlink}`,
    json_metadata: JSON.stringify({ app: o.app ?? "pixagram/3.0.2", tags: ["pixelart"] }),
    created: iso(timeOf(o.createdBlock) - 3),
    last_update: iso(timeOf(o.opBlock ?? o.createdBlock) - 3),
    depth: 0,
    children: 0,
    net_votes: 0,
    pending_payout_value: "0.000 PXS",
    total_payout_value: "0.000 PXS",
    curator_payout_value: "0.000 PXS",
  };
}

const commentOp = (p: CondenserPost) => ({
  type: "comment_operation",
  value: { parent_author: "", parent_permlink: p.parent_permlink, author: p.author, permlink: p.permlink, title: p.title, body: p.body, json_metadata: p.json_metadata },
});
const deleteOp = (a: string, p: string) => ({ type: "delete_comment_operation", value: { author: a, permlink: p } });

/**
 * A Pixa node: blocks with their real spacing, get_content answered by `content` (changed by the
 * test as Hivemind catches up), and Hivemind's listing of the newest posts, which (like Pixa's)
 * starts the next page after the `start` post.
 */
function node(o: { lib: number; ops?: Record<number, unknown[]>; content?: (a: string, p: string) => Reply; listing?: CondenserPost[] }) {
  const calls: Array<{ method: string; params: any }> = [];
  const n = { lib: o.lib, ops: o.ops ?? {}, content: o.content ?? ((a: string, p: string) => doesNotExist(a, p)), listing: o.listing ?? [], calls };
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id, params } = JSON.parse(init.body);
      calls.push({ method, params });
      const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
      if (method === "database_api.get_dynamic_global_properties") return send({ result: { head_block_number: n.lib, last_irreversible_block_num: n.lib, time: iso(timeOf(n.lib)) } });
      if (method === "block_api.get_block_range") {
        const { starting_block_num: from, count } = params;
        const blocks = [];
        for (let b = from; b < from + count && b <= n.lib; b++) {
          blocks.push({ timestamp: iso(timeOf(b)), transactions: (n.ops[b] ?? []).map((op) => ({ operations: [op] })) });
        }
        return send({ result: { blocks } });
      }
      if (method === "condenser_api.get_content") return send(n.content(params[0], params[1]));
      if (method === "condenser_api.get_discussions_by_created") {
        const q = params[0];
        const all = [...n.listing].sort((a, b) => b.created.localeCompare(a.created));
        const from = q.start_author ? all.findIndex((p) => p.author === q.start_author && p.permlink === q.start_permlink) + 1 : 0;
        return send({ result: all.slice(from, from + q.limit) });
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return n;
}

async function postsIn(env: TestEnv) {
  const r = await env.DB.prepare("SELECT author, permlink, title, deleted FROM posts ORDER BY id").all<{ author: string; permlink: string; title: string; deleted: number }>();
  return (r.results ?? []).map((x) => `${x.author}/${x.permlink}${x.deleted ? " (deleted)" : ""}`);
}

function indexer(over: Record<string, string> = {}) {
  const storage = new FakeStorage();
  const env = makeEnv({ APP_PREFIXES: "", ...over } as any);
  const doo = new ChainIndexer({ storage } as any, env as any);
  return { storage, env, doo };
}

/** Move the clock (Date only: the RPC client's timers stay real). */
function clockAt(ms: number) {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(ms);
}

describe("the tail and a Hivemind behind it", () => {
  it("a post Hivemind does not hold yet is deferred, not taken for deleted, and ingested once it shows (the 12 Nov losses)", async () => {
    const { storage, env, doo } = indexer();
    const sarix = post("sarix", "mi-bolita-de-pelo-2026-1794514132083", { createdBlock: BASE + 1 });
    let caughtUp = false;
    const n = node({ lib: BASE + 2, ops: { [BASE + 1]: [commentOp(sarix)] }, content: (a, p) => (caughtUp ? { result: sarix } : doesNotExist(a, p)) });
    await storage.put("state", { running: true, cursor: BASE });
    const t0 = Date.now();
    clockAt(t0);

    await doo.alarm();
    let s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.cursor).toBe(BASE + 3); // the tail moved on
    expect(await postsIn(env)).toEqual([]); // not ingested, and not recorded as deleted either
    expect(s.deferredWaiting).toBe(1);
    expect(s.deferred[0]).toMatchObject({ author: "sarix", block: BASE + 1, op: "comment", prevAt: timeOf(BASE), why: "Hivemind does not hold it yet", tries: 1 });
    // its version is recorded from the block all the same (exact block and time)
    const v = await env.DB.prepare("SELECT block_num, source, at FROM post_versions WHERE author = 'sarix'").all();
    expect(v.results).toEqual([{ block_num: BASE + 1, source: "tail", at: timeOf(BASE + 1) }]);

    // Hivemind catches up; the next tick after the first backoff asks again
    caughtUp = true;
    clockAt(t0 + deferBackoffMs(1));
    n.lib = BASE + 2; // nothing new on the chain: the tail is idle and works on the deferred posts
    await doo.alarm();
    s = await doo.status();
    expect(s.deferredWaiting).toBe(0);
    expect(await postsIn(env)).toEqual(["sarix/mi-bolita-de-pelo-2026-1794514132083"]);
    expect(env._queue.sent.map((m) => `${m.author}/${m.permlink}:${m.reason}`)).toEqual(["sarix/mi-bolita-de-pelo-2026-1794514132083:tail-deferred"]);
    expect(s.deferredTotal).toBe(1);
  });

  it("a deferred post is not asked again before it is due", async () => {
    const { storage, doo } = indexer();
    const p = post("wendy", "s-per-yako-1794516980412", { createdBlock: BASE + 1 });
    const n = node({ lib: BASE + 1, ops: { [BASE + 1]: [commentOp(p)] } });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    clockAt(Date.now());
    await doo.alarm();
    const asked = () => n.calls.filter((c) => c.method === "condenser_api.get_content").length;
    expect(asked()).toBe(1);
    await doo.alarm(); // three seconds later in real life: not due yet (6 s)
    expect(asked()).toBe(1);
  });

  it("Hivemind's last_update is the block before the op's: a post it already shows goes in at once, the first block of a range included", async () => {
    const { storage, env, doo } = indexer({ TAIL_BLOCKS_PER_TICK: "2" });
    const a = post("ladybug", "heart-for-love-1794514513975", { createdBlock: BASE + 10 });
    const b = post("yesenia", "trinitarias-1794517128464", { createdBlock: BASE + 12 });
    const n = node({ lib: BASE + 13, ops: { [BASE + 10]: [commentOp(a)], [BASE + 12]: [commentOp(b)] }, content: (au) => ({ result: au === "ladybug" ? a : b }) });
    await storage.put("state", { running: true, cursor: BASE + 10 });

    await doo.alarm(); // blocks +10, +11: the op is in the range's first block, whose predecessor is read
    expect(n.calls.filter((c) => c.method === "block_api.get_block_range").map((c) => c.params.starting_block_num)).toEqual([BASE + 10, BASE + 9]);
    expect(await postsIn(env)).toEqual(["ladybug/heart-for-love-1794514513975"]);

    await doo.alarm(); // blocks +12, +13: the op is in the first block again; the last tick's time serves
    expect(n.calls.filter((c) => c.method === "block_api.get_block_range").map((c) => c.params.starting_block_num)).toEqual([BASE + 10, BASE + 9, BASE + 12]);
    expect(await postsIn(env)).toEqual(["ladybug/heart-for-love-1794514513975", "yesenia/trinitarias-1794517128464"]);
    expect((await doo.status()).deferredWaiting).toBe(0);
  });

  it("an edit Hivemind has not applied yet is deferred until its last_update reaches the op", async () => {
    const { storage, env, doo } = indexer();
    const before = post("saavedraa", "flowers-1794518212770", { createdBlock: BASE, title: "Flowers" });
    const after = post("saavedraa", "flowers-1794518212770", { createdBlock: BASE, opBlock: BASE + 20, title: "Desert flowers" });
    await ingestPost(env, before, BASE, "test");
    let shows = before;
    node({ lib: BASE + 20, ops: { [BASE + 20]: [commentOp(after)] }, content: () => ({ result: shows }) });
    await storage.put("state", { running: true, cursor: BASE + 20 });
    const t0 = Date.now();
    clockAt(t0);

    await doo.alarm();
    const title = async () => (await env.DB.prepare("SELECT title FROM posts").first<{ title: string }>())!.title;
    expect(await title()).toBe("Flowers"); // the older state was not written over the edit
    const d = (await doo.status()).deferred[0];
    expect(d.why).toBe(`Hivemind shows the state of ${iso(timeOf(BASE) - 3)}, the op needs ${iso(timeOf(BASE + 19))}`);

    shows = after;
    clockAt(t0 + deferBackoffMs(1));
    await doo.alarm();
    expect(await title()).toBe("Desert flowers");
    expect((await doo.status()).deferredWaiting).toBe(0);
  });

  it("delete_comment: still shown is Hivemind behind (deferred); absent is deleted; shown newer is a post created again", async () => {
    const { storage, env, doo } = indexer();
    const cat = post("raidhiap", "black-cat-1794594802143", { createdBlock: BASE });
    await ingestPost(env, cat, BASE, "test");
    let reply: Reply = { result: cat };
    node({ lib: BASE + 30, ops: { [BASE + 30]: [deleteOp(cat.author, cat.permlink)] }, content: () => reply });
    await storage.put("state", { running: true, cursor: BASE + 30 });
    const t0 = Date.now();
    clockAt(t0);

    await doo.alarm();
    expect(await postsIn(env)).toEqual(["raidhiap/black-cat-1794594802143"]);
    expect((await doo.status()).deferred[0]).toMatchObject({ op: "delete", block: BASE + 30 });

    reply = wasDeleted(cat.author, cat.permlink);
    clockAt(t0 + deferBackoffMs(1));
    await doo.alarm();
    expect(await postsIn(env)).toEqual(["raidhiap/black-cat-1794594802143 (deleted)"]);
    expect((await doo.status()).deferredWaiting).toBe(0);

    // "was deleted" is final even after a comment op (Hivemind ahead: deleted in a later block)
    const { storage: s2, env: e2, doo: d2 } = indexer();
    const gone = post("tetiana", "colors-of-italy-1794428051828", { createdBlock: BASE + 40 });
    node({ lib: BASE + 40, ops: { [BASE + 40]: [commentOp(gone)] }, content: (a, p) => wasDeleted(a, p) });
    await s2.put("state", { running: true, cursor: BASE + 40 });
    await d2.alarm();
    expect((await d2.status()).deferredWaiting).toBe(0);
    expect(await postsIn(e2)).toEqual([]);
  });

  it("gives up after TAIL_DEFER_HOURS: keeps the older state Hivemind shows, drops a post it never returned (logged)", async () => {
    const { storage, env, doo } = indexer({ TAIL_DEFER_HOURS: "1" });
    const never = post("knightrider", "gta-vi-with-no-disk-1794509406258", { createdBlock: BASE + 1 });
    const old = post("mistertwister", "the-beach-1794514386334", { createdBlock: BASE, title: "The beach" });
    const edit = post("mistertwister", "the-beach-1794514386334", { createdBlock: BASE, opBlock: BASE + 2, title: "The beach at night" });
    node({
      lib: BASE + 2,
      ops: { [BASE + 1]: [commentOp(never)], [BASE + 2]: [commentOp(edit)] },
      content: (a, p) => (a === "mistertwister" ? { result: old } : doesNotExist(a, p)),
    });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    const warns = vi.spyOn(console, "warn").mockImplementation(() => {});
    const t0 = Date.now();
    clockAt(t0);
    await doo.alarm();
    expect((await doo.status()).deferredWaiting).toBe(2);

    clockAt(t0 + 3_600_000 + 1);
    await doo.alarm();
    const s = await doo.status();
    expect(s.deferredWaiting).toBe(0);
    expect(s.deferredGivenUp).toBe(2);
    expect(await postsIn(env)).toEqual(["mistertwister/the-beach-1794514386334"]); // the state Hivemind had
    expect(errors.mock.calls.some((c) => String(c[0]).includes("never returned") && c[1] === "knightrider/gta-vi-with-no-disk-1794509406258")).toBe(true);
    expect(warns.mock.calls.some((c) => String(c[0]).includes("kept the state it shows"))).toBe(true);
  });

  it("the deferred list stays bounded (one storage value): the oldest go, logged", async () => {
    const { storage, doo } = indexer({ TAIL_BLOCKS_PER_TICK: "300" });
    const ops: Record<number, unknown[]> = {};
    for (let k = 0; k < DEFERRED_MAX + 5; k++) ops[BASE + 1 + k] = [commentOp(post(`artist${k}`, `a-rather-long-permlink-of-the-kind-pixtopia-writes-${k}-${"x".repeat(60)}`, { createdBlock: BASE + 1 + k }))];
    node({ lib: BASE + DEFERRED_MAX + 5, ops });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await doo.alarm();
    const s = await doo.status();
    expect(s.deferredWaiting).toBe(DEFERRED_MAX);
    expect(JSON.stringify(storage.m.get("deferred")).length).toBeLessThan(128 * 1024);
    expect(errors.mock.calls.filter((c) => String(c[0]).includes("dropped a deferred post"))).toHaveLength(5);
    expect(s.deferredTotal).toBe(DEFERRED_MAX + 5);
  });

  it("a tick that fails after deferring a post counts it once, when the tick that succeeds saves it", async () => {
    const { storage, doo } = indexer({ TAIL_PARK_AFTER: "5" });
    const a = post("anna", "late", { createdBlock: BASE + 1 });
    const b = post("bert", "flaky", { createdBlock: BASE + 2 });
    let flaky = true;
    node({
      lib: BASE + 2,
      ops: { [BASE + 1]: [commentOp(a)], [BASE + 2]: [commentOp(b)] },
      content: (au, p) => (au === "anna" ? doesNotExist(au, p) : flaky ? { error: { code: -32603, message: "Internal Error" } } : { result: b }),
    });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    await doo.alarm(); // anna deferred, then bert fails: the tick is done again
    let s = await doo.status();
    expect(s.lastError).toBe("Internal Error");
    expect(s.deferredTotal).toBe(0);
    flaky = false;
    await doo.alarm();
    s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.deferredTotal).toBe(1);
    expect(s.deferredWaiting).toBe(1);
  });

  it("retryParked keeps a post Hivemind does not hold after a comment op (it exists), and settles a deleted one", async () => {
    const { storage, env, doo } = indexer({ TAIL_PARK_AFTER: "1" });
    const p = post("ramisey", "good-night-1794623408783", { createdBlock: BASE + 1 });
    let reply: Reply = { error: { code: -32603, message: "Internal Error", data: "hivemind: statement timeout" } };
    node({ lib: BASE + 1, ops: { [BASE + 1]: [commentOp(p)] }, content: () => reply });
    await storage.put("state", { running: true, cursor: BASE + 1 });
    vi.spyOn(console, "error").mockImplementation(() => {});
    await doo.alarm();
    expect((await doo.status()).parked).toMatchObject([{ author: "ramisey", op: "comment" }]);

    reply = doesNotExist(p.author, p.permlink);
    expect(await doo.retryParked()).toEqual({ retried: 1, ingested: 0, left: 1 });
    expect((await doo.status()).parked[0]).toMatchObject({ attempts: 1, error: "Hivemind does not hold it yet" });

    reply = { result: p };
    expect(await doo.retryParked()).toEqual({ retried: 1, ingested: 1, left: 0 });
    expect(await postsIn(env)).toEqual(["ramisey/good-night-1794623408783"]);

    // a post parked before v4.8.3 (no op recorded): absent settles it, as it always did
    await storage.put("parked", [{ author: "x", permlink: "y", block: 1, error: "e", parkedAt: 0, attempts: 0, lastTryAt: 0 }]);
    reply = doesNotExist("x", "y");
    expect(await doo.retryParked()).toEqual({ retried: 1, ingested: 1, left: 0 });
  });
});

// ---- the reconciliation -------------------------------------------------------------------------

describe("reconcileRecent: Hivemind's newest posts against the index", () => {
  const NOW = Date.UTC(2026, 10, 16, 3, 0, 0);
  const at = (minutesAgo: number) => iso(NOW / 1000 - minutesAgo * 60);
  function listed(author: string, permlink: string, createdMinutesAgo: number, o: { updatedMinutesAgo?: number; app?: string; title?: string } = {}): CondenserPost {
    return {
      ...post(author, permlink, { createdBlock: BASE, app: o.app, title: o.title }),
      created: at(createdMinutesAgo),
      last_update: at(o.updatedMinutesAgo ?? createdMinutesAgo),
    };
  }

  it("ingests what the index lacks or holds in an older state, from every front end, within the window", async () => {
    const env = makeEnv({ APP_PREFIXES: "" } as any);
    clockAt(NOW);
    const inSync = listed("benchy", "hidden-starfish", 30);
    const edited = listed("irita", "muy-friend-sonic", 40, { updatedMinutesAgo: 20, title: "My friend Sonic" });
    const lost = listed("ismeris", "a-walk-with-my-cat", 50);
    const pixtopia = listed("conayi", "full-of-flowers-da5e4794", 60, { app: "pixtopia/0.1.0" });
    const tooNew = listed("jesus", "calisthenics-9010f76f", 2); // the tail's (it may be asking right now)
    const tooOld = listed("anoma", "a-shady-walkway-d6e8a47b", 400); // outside a three-hour window
    // pad the listing so it takes several pages (Pixa's Hivemind does not repeat the start post)
    const filler = Array.from({ length: 45 }, (_, i) => listed(`artist${i}`, `p${i}`, 70 + i));
    await ingestPost(env, inSync, null, "test");
    await ingestPost(env, { ...edited, title: "Sonic", last_update: at(40) }, null, "test");
    for (const f of filler) await ingestPost(env, f, null, "test");
    const chain = [inSync, edited, lost, pixtopia, tooNew, tooOld, ...filler];
    const n = node({ lib: BASE, listing: chain, content: (a, p) => ({ result: chain.find((x) => x.author === a && x.permlink === p) }) });

    const r = await reconcileRecent(env, { hours: 3 });
    expect(r).toMatchObject({ missing: 2, stale: 1, ingested: 3, failed: 0, skipped_app: 0, truncated: false });
    expect(r.recovered.sort()).toEqual(["conayi/full-of-flowers-da5e4794", "irita/muy-friend-sonic", "ismeris/a-walk-with-my-cat"]);
    expect(r.listed).toBe(4 + 45); // the two-minute-old and the 400-minute-old ones left out
    expect(r.pages).toBe(3); // stops at the page that reaches past the window
    const all = await postsIn(env);
    expect(all).toContain("conayi/full-of-flowers-da5e4794");
    expect(all).not.toContain("jesus/calisthenics-9010f76f");
    expect(all).not.toContain("anoma/a-shady-walkway-d6e8a47b");
    expect((await env.DB.prepare("SELECT title FROM posts WHERE author = 'irita'").first<{ title: string }>())!.title).toBe("My friend Sonic");
    // every page after the first started after the previous page's last post
    const pages = n.calls.filter((c) => c.method === "condenser_api.get_discussions_by_created").map((c) => c.params[0]);
    expect(pages[0]).toMatchObject({ tag: "", limit: 20, truncate_body: 1 });
    expect(pages[1].start_permlink).toBe(chain.sort((a, b) => b.created.localeCompare(a.created))[19].permlink);

    // recorded for /healthz and /admin/stats; a second run finds nothing left
    const again = await reconcileRecent(env, { hours: 3 });
    expect(again).toMatchObject({ missing: 0, stale: 0, ingested: 0 });
    const rec = await reconcileRecord(env);
    expect(rec).toMatchObject({ recovered_total: 3, last: { ingested: 0 }, last_recovery: { missing: 2, stale: 1 } });
  });

  it("leaves out the apps APP_PREFIXES leaves out, and stops at maxIngest (truncated: the next run goes on)", async () => {
    const env = makeEnv({ APP_PREFIXES: "pixagram" } as any);
    clockAt(NOW);
    const chain = [listed("conayi", "flowers", 30, { app: "pixtopia/0.1.0" }), listed("a", "one", 31), listed("b", "two", 32), listed("c", "three", 33)];
    node({ lib: BASE, listing: chain, content: (a, p) => ({ result: chain.find((x) => x.author === a && x.permlink === p) }) });
    const r = await reconcileRecent(env, { hours: 3, maxIngest: 2 });
    expect(r).toMatchObject({ skipped_app: 1, ingested: 2, truncated: true });
    expect((await reconcileRecent(env, { hours: 3, maxIngest: 2 })).ingested).toBe(1);
    expect(await postsIn(env)).toHaveLength(3);
  });

  it("is off with RECONCILE_HOURS=0, and reports a node error instead of throwing", async () => {
    const env = makeEnv({ RECONCILE_HOURS: "0" } as any);
    expect(await reconcileRecent(env)).toMatchObject({ pages: 0, listed: 0 });
    const env2 = makeEnv();
    vi.stubGlobal("fetch", vi.fn(async () => new Response("bad gateway", { status: 502 })));
    const r = await reconcileRecent(env2, { hours: 3 });
    expect(r.error).toMatch(/HTTP 502/);
  });
});

// ---- every front end ----------------------------------------------------------------------------

describe("front ends", () => {
  it("the deployed configuration indexes every front end (APP_PREFIXES empty)", () => {
    const text = readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8");
    expect(text.match(/"APP_PREFIXES":\s*"([^"]*)"/)?.[1]).toBe("");
  });

  it("a Pixtopia artwork goes in with APP_PREFIXES empty, and is left out with \"pixagram\"", async () => {
    const pix = { ...post("conayi", "full-of-flowers-da5e4794", { createdBlock: BASE, app: "pixtopia/0.1.0" }), body: "data:image/webp;base64,UklGRg==", json_metadata: JSON.stringify({ app: "pixtopia/0.1.0", nsfw: false, tags: ["pixelart"], image: [], format: "image", license: { type: "ALL_RIGHTS_RESERVED" }, description: "" }) };
    const all = makeEnv({ APP_PREFIXES: "" } as any);
    expect(await ingestPost(all, pix, null, "test")).toMatchObject({ action: "inserted", enqueued: true });
    const only = makeEnv({ APP_PREFIXES: "pixagram" } as any);
    expect(await ingestPost(only, pix, null, "test")).toMatchObject({ action: "skipped-app" });
  });
});

// ---- /healthz -----------------------------------------------------------------------------------

describe("/healthz reports the chain tail", () => {
  function withIndexer(status: Record<string, unknown>) {
    const env = makeEnv();
    (env as any).INDEXER = { idFromName: () => "main", get: () => ({ status: async () => status }) };
    return env;
  }
  const base = { running: true, cursor: 101, lib: 100, lastTickAt: Date.now(), parked: [], deferredWaiting: 0, deferred: [] };

  it("ok with the tail at the head and nothing waiting", async () => {
    const r = await healthReport(withIndexer(base));
    expect(r.status).toBe("ok");
    expect(r.tail).toMatchObject({ running: true, behind_blocks: 0, deferred: 0, deferred_over_hour: 0, parked: 0 });
  });

  it("degraded: posts waiting for Hivemind for over an hour, a stopped or stalled tail, a tail far behind", async () => {
    const old = { author: "a", permlink: "b", since: Date.now() - 2 * 3_600_000 };
    expect((await healthReport(withIndexer({ ...base, deferredWaiting: 3, deferred: [old, { ...old, since: Date.now() }] }))).problems).toEqual([
      "1 new post waiting for the node's Hivemind for over an hour",
    ]);
    expect((await healthReport(withIndexer({ ...base, running: false }))).problems).toEqual(["chain indexer stopped: new posts are not indexed (POST /admin/indexer/start)"]);
    expect((await healthReport(withIndexer({ ...base, lastTickAt: Date.now() - 20 * 60_000 }))).problems).toEqual(["chain indexer stalled: no tick for 20 min"]);
    expect((await healthReport(withIndexer({ ...base, cursor: 1, lib: 2400 }))).problems).toEqual(["chain indexer 2400 blocks (about 2 h) behind the chain"]);
  });

  it("says so when the indexer cannot be asked", async () => {
    const env = makeEnv();
    (env as any).INDEXER = { idFromName: () => "main", get: () => ({ status: async () => { throw new Error("Durable Object reset"); } }) };
    const r = await healthReport(env);
    expect(r.status).toBe("degraded");
    expect(r.problems).toEqual(["chain indexer unreachable: Durable Object reset"]);
  });
});
