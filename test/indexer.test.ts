// The chain tail's control calls against a tick in flight: a Durable Object delivers other calls
// while alarm() awaits the chain, so start()/stop() must win over the tick's stale state.
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChainIndexer } from "../src/chain/indexer-do";
import { makeEnv } from "./harness/fakes";

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

/** A chain whose get_block_range waits until released, so a control call can land mid-tick. */
function gatedChain() {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  let entered!: () => void;
  const inRange = new Promise<void>((r) => (entered = r));
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id } = JSON.parse(init.body);
      const ok = (result: unknown) => new Response(JSON.stringify({ jsonrpc: "2.0", id, result }));
      if (method === "database_api.get_dynamic_global_properties") return ok({ head_block_number: 2000, last_irreversible_block_num: 1990, time: "2026-10-04T00:00:00" });
      if (method === "block_api.get_block_range") {
        entered();
        await gate;
        return ok({ blocks: [{ timestamp: "2026-10-04T00:00:00", transactions: [] }] });
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return { release, inRange };
}

afterEach(() => vi.unstubAllGlobals());

describe("ChainIndexer", () => {
  it("a stop() during a tick stays stopped, with no alarm", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release, inRange } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    const tick = doo.alarm();
    await inRange;
    await doo.stop();
    release();
    await tick;
    const after = await doo.status();
    expect(after.running).toBe(false);
    expect(after.alarmAt).toBeNull();
    expect(after.ticks).toBe(1); // the tick's counters still count
  });

  it("a start(from) replay during a tick keeps its cursor", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release, inRange } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    const tick = doo.alarm();
    await inRange;
    await doo.start(500);
    release();
    await tick;
    const after = await doo.status();
    expect(after.cursor).toBe(500);
    expect(after.running).toBe(true);
    expect(after.alarmAt).not.toBeNull();
  });

  it("an undisturbed tick advances the cursor and re-arms", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    release();
    await doo.alarm();
    const after = await doo.status();
    expect(after.cursor).toBe(1001);
    expect(after.alarmAt).not.toBeNull();
  });
});

// ---- the tail and posts the node will not return ----------------------------------------------
//
// 13 Oct 2026: @anti deleted a reply (delete_comment, block 1130742). The tail asks get_content for
// every post a block mentions, and the node (Hivemind) answers a deleted post with an assert —
// {"code":-31999,"message":"Invalid parameters","data":"Post anti/… was deleted 1 time(s)"} — where
// legacy nodes answered an empty object. The tick threw, re-armed in 10 s, threw again: the tail
// stalled there for nine days and every post since was unknown (the originality drawer showed
// "Not checked yet" on all of them).

import { ChainRpc, RpcError, isMissingPost } from "../src/chain/rpc";

const DELETED_REPLY = { author: "anti", permlink: "re-anti-germany-berlin-reichstag-1791821292552-mv71sldf" };

type Reply = { result: unknown } | { error: { code: number; message: string; data?: unknown } };

const deletedAssert = (a: string, p: string): Reply => ({ error: { code: -31999, message: "Invalid parameters", data: `Post ${a}/${p} was deleted 1 time(s)` } });
const missingAssert = (a: string, p: string): Reply => ({
  error: { code: -32602, message: `Assert Exception:Post ${a}/${p} does not exist`, data: { code: 10, name: "assert_exception", message: "Assert Exception", stack: [] } },
});

function topLevel(author: string, permlink: string, title: string) {
  return {
    author,
    permlink,
    parent_author: "",
    parent_permlink: "cat",
    category: "cat",
    title,
    body: `A few words about ${title}`,
    json_metadata: JSON.stringify({ app: "pixagram/3.0.2", tags: ["cat"] }),
    created: "2026-10-13T19:18:18",
    last_update: "2026-10-13T19:18:18",
    depth: 0,
    children: 0,
    net_votes: 0,
    pending_payout_value: "0.000 PXS",
    total_payout_value: "0.000 PXS",
    curator_payout_value: "0.000 PXS",
  };
}

const commentOp = (p: ReturnType<typeof topLevel>) => ({
  type: "comment_operation",
  value: { parent_author: "", parent_permlink: p.parent_permlink, author: p.author, permlink: p.permlink, title: p.title, body: p.body, json_metadata: p.json_metadata },
});
const deleteOp = (a: string, p: string) => ({ type: "delete_comment_operation", value: { author: a, permlink: p } });

/** A chain: `ops` by block number, get_content answered by `content` (called per request). */
function chain(o: { lib: number; ops: Record<number, unknown[]>; content: (author: string, permlink: string) => Reply }) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id, params } = JSON.parse(init.body);
      const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
      if (method === "database_api.get_dynamic_global_properties") return send({ result: { head_block_number: o.lib, last_irreversible_block_num: o.lib, time: "2026-10-13T19:20:00" } });
      if (method === "block_api.get_block_range") {
        const { starting_block_num: from, count } = params;
        const blocks = [];
        for (let n = from; n < from + count && n <= o.lib; n++) {
          blocks.push({ timestamp: "2026-10-13T19:12:39", transactions: (o.ops[n] ?? []).map((op) => ({ operations: [op] })) });
        }
        return send({ result: { blocks } });
      }
      if (method === "condenser_api.get_content") {
        calls.push(`${params[0]}/${params[1]}`);
        return send(o.content(params[0], params[1]));
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return calls;
}

async function postsIn(env: ReturnType<typeof makeEnv>) {
  const r = await env.DB.prepare("SELECT author, permlink, deleted FROM posts ORDER BY id").all<{ author: string; permlink: string; deleted: number }>();
  return (r.results ?? []).map((x) => `${x.author}/${x.permlink}${x.deleted ? " (deleted)" : ""}`);
}

describe("get_content on a post the node does not hold", () => {
  const rpc = () => new ChainRpc({ url: "https://rpc.test", retries: 0 });

  it("a deleted post is null, as the empty object of legacy nodes was", async () => {
    chain({ lib: 1, ops: {}, content: (a, p) => deletedAssert(a, p) });
    expect(await rpc().getContent(DELETED_REPLY.author, DELETED_REPLY.permlink)).toBeNull();
  });

  it("a post never created is null", async () => {
    chain({ lib: 1, ops: {}, content: (a, p) => missingAssert(a, p) });
    expect(await rpc().getContent("anti", "no-such-post")).toBeNull();
  });

  it("other refusals still throw, with the node's reason in the message", async () => {
    chain({ lib: 1, ops: {}, content: () => ({ error: { code: -31999, message: "Invalid parameters", data: "permlink cannot be blank" } }) });
    const e = await rpc().getContent("anti", "").catch((x) => x);
    expect(e).toBeInstanceOf(RpcError);
    expect(e.message).toBe("Invalid parameters: permlink cannot be blank");
  });

  it("an assert about another post is not this post's absence", () => {
    const e = new RpcError("Invalid parameters", "condenser_api.get_content", -31999, "Post anti/p10 was deleted 1 time(s)");
    expect(isMissingPost(e, "anti", "p10")).toBe(true);
    expect(isMissingPost(e, "anti", "p1")).toBe(false);
    expect(isMissingPost(new Error("Post anti/p10 was deleted 1 time(s)"), "anti", "p10")).toBe(false);
  });
});

describe("ChainIndexer: posts it cannot ingest", () => {
  it("a deleted reply in a block no longer stops the tail (the 13 Oct stall)", async () => {
    const storage = new FakeStorage();
    const env = makeEnv();
    const doo = new ChainIndexer({ storage } as any, env as any);
    const silva = topLevel("silva", "miau-cat-1791919097727", "Miau cat");
    chain({
      lib: 1130860,
      ops: { 1130742: [deleteOp(DELETED_REPLY.author, DELETED_REPLY.permlink)], 1130856: [commentOp(silva)] },
      content: (a, p) => (a === silva.author && p === silva.permlink ? { result: silva } : deletedAssert(a, p)),
    });
    await storage.put("state", { running: true, cursor: 1130700 });
    await doo.alarm();
    const s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.cursor).toBe(1130861);
    expect(s.parked).toEqual([]);
    expect(await postsIn(env)).toEqual(["silva/miau-cat-1791919097727"]);
    expect(env._queue.sent.map((m) => `${m.author}/${m.permlink}`)).toEqual(["silva/miau-cat-1791919097727"]);
  });

  it("a deleted top-level post is marked deleted", async () => {
    const storage = new FakeStorage();
    const env = makeEnv();
    const doo = new ChainIndexer({ storage } as any, env as any);
    const cat = topLevel("silva", "gone-cat", "Gone cat");
    let deleted = false;
    chain({
      lib: 2002,
      ops: { 2000: [commentOp(cat)], 2002: [deleteOp(cat.author, cat.permlink)] },
      content: (a, p) => (deleted ? deletedAssert(a, p) : { result: cat }),
    });
    await storage.put("state", { running: true, cursor: 2000 });
    // first tick: the post exists (the range stops before the delete)
    (doo as any).env.TAIL_BLOCKS_PER_TICK = "2";
    await doo.alarm();
    expect(await postsIn(env)).toEqual(["silva/gone-cat"]);
    deleted = true;
    await doo.alarm();
    expect((await doo.status()).lastError).toBeNull();
    expect(await postsIn(env)).toEqual(["silva/gone-cat (deleted)"]);
  });

  it("a post that keeps failing is parked after TAIL_PARK_AFTER ticks; the tail moves on", async () => {
    const storage = new FakeStorage();
    const env = makeEnv({ TAIL_PARK_AFTER: "3" } as any);
    const doo = new ChainIndexer({ storage } as any, env as any);
    const bad = topLevel("mike", "mountain-high", "Mountain high");
    const good = topLevel("whale", "whale1", "Whale");
    let healed = false;
    const calls = chain({
      lib: 3005,
      ops: { 3001: [commentOp(bad)], 3003: [commentOp(good)] },
      content: (a, p) =>
        a === bad.author && !healed ? { error: { code: -32603, message: "Internal Error", data: "hivemind: statement timeout" } } : { result: a === bad.author ? bad : good },
    });
    await storage.put("state", { running: true, cursor: 3000 });

    await doo.alarm();
    await doo.alarm();
    let s = await doo.status();
    expect(s.cursor).toBe(3000); // two failed ticks: still retrying
    expect(s.stuck).toMatchObject({ ref: "mike/mountain-high", block: 3001, ticks: 2 });
    expect(s.lastError).toBe("Internal Error: hivemind: statement timeout");
    expect(s.parked).toEqual([]);

    await doo.alarm(); // the third: parked, and the rest of the range goes in
    s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.stuck).toBeNull();
    expect(s.cursor).toBe(3006);
    expect(s.parked).toMatchObject([{ author: "mike", permlink: "mountain-high", block: 3001, attempts: 0, error: "Internal Error: hivemind: statement timeout" }]);
    expect(await postsIn(env)).toEqual(["whale/whale1"]);

    // the cron tries it again: still failing, it stays (attempt counted); then the node answers
    expect(await doo.retryParked()).toEqual({ retried: 1, ingested: 0, left: 1 });
    expect((await doo.status()).parked[0].attempts).toBe(1);
    healed = true;
    expect(await doo.retryParked()).toEqual({ retried: 1, ingested: 1, left: 0 });
    expect(await postsIn(env)).toEqual(["whale/whale1", "mike/mountain-high"]);
    expect(calls.filter((c) => c === "mike/mountain-high")).toHaveLength(5);
  });

  it("a ticks-in-a-row count does not carry over once a tick gets past the post", async () => {
    const storage = new FakeStorage();
    const env = makeEnv({ TAIL_PARK_AFTER: "3" } as any);
    const doo = new ChainIndexer({ storage } as any, env as any);
    const p = topLevel("mike", "flaky", "Flaky");
    let fail = 1;
    chain({ lib: 4001, ops: { 4000: [commentOp(p)] }, content: () => (fail-- > 0 ? { error: { code: -32603, message: "Internal Error" } } : { result: p }) });
    await storage.put("state", { running: true, cursor: 4000 });
    await doo.alarm(); // fails once
    await doo.alarm(); // goes through
    const s = await doo.status();
    expect(s.stuck).toBeNull();
    expect(s.parked).toEqual([]);
    expect(s.cursor).toBe(4002);
    expect(await postsIn(env)).toEqual(["mike/flaky"]);
  });

  it("two failing posts in one range: a parked post is not retried by the tail (no see-saw)", async () => {
    const storage = new FakeStorage();
    const env = makeEnv({ TAIL_PARK_AFTER: "2" } as any);
    const doo = new ChainIndexer({ storage } as any, env as any);
    const a = topLevel("anna", "a", "A");
    const b = topLevel("bert", "b", "B");
    const c = topLevel("cleo", "c", "C");
    chain({
      lib: 5003,
      ops: { 5001: [commentOp(a)], 5002: [commentOp(b)], 5003: [commentOp(c)] },
      content: (au) => (au === "cleo" ? { result: c } : { error: { code: -32603, message: `Internal Error ${au}` } }),
    });
    await storage.put("state", { running: true, cursor: 5000 });
    for (let i = 0; i < 4; i++) await doo.alarm();
    const s = await doo.status();
    expect(s.lastError).toBeNull();
    expect(s.cursor).toBe(5004);
    expect(s.parked.map((x) => `${x.author}/${x.permlink}`)).toEqual(["anna/a", "bert/b"]);
    expect(await postsIn(env)).toEqual(["cleo/c"]);
  });

  it("start() forgets a count in progress", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    chain({ lib: 10, ops: {}, content: () => ({ result: {} }) });
    await storage.put("state", { running: true, cursor: 5, stuck: { ref: "a/b", block: 5, ticks: 4, error: "x" } });
    await doo.start();
    expect((await doo.status()).stuck).toBeNull();
  });
});

describe("ChainIndexer.retryParked", () => {
  /** Park every post in blocks 6001.. (TAIL_PARK_AFTER=1: the first failed tick parks). */
  async function parkedIndexer(names: string[]) {
    const storage = new FakeStorage();
    const env = makeEnv({ TAIL_PARK_AFTER: "1" } as any);
    const doo = new ChainIndexer({ storage } as any, env as any);
    const posts = names.map((n) => topLevel(n, `p-${n}`, n));
    const ops: Record<number, unknown[]> = {};
    posts.forEach((p, i) => (ops[6001 + i] = [commentOp(p)]));
    const answer = { ok: new Set<string>() };
    chain({ lib: 6000 + names.length, ops, content: (a) => (answer.ok.has(a) ? { result: posts.find((p) => p.author === a) } : { error: { code: -32603, message: `Internal Error ${a}` } }) });
    await storage.put("state", { running: true, cursor: 6001 });
    await doo.alarm();
    return { doo, env, answer };
  }

  it("tries the least recently tried first, keeps every post it did not get to", async () => {
    const { doo, answer } = await parkedIndexer(["ann", "bob", "cid"]);
    expect((await doo.status()).parked.map((p) => p.author)).toEqual(["ann", "bob", "cid"]);
    expect(await doo.retryParked(1)).toEqual({ retried: 1, ingested: 0, left: 3 });
    expect(await doo.retryParked(1)).toEqual({ retried: 1, ingested: 0, left: 3 });
    const s = await doo.status();
    expect(s.parked.find((p) => p.author === "cid")!.attempts).toBe(0); // its turn is next
    answer.ok.add("cid");
    expect(await doo.retryParked(1)).toEqual({ retried: 1, ingested: 1, left: 2 });
  });

  it("a post parked while it awaits the chain is kept", async () => {
    const { doo, env } = await parkedIndexer(["ann"]);
    // the retry's get_content waits; meanwhile a tick parks another post
    let release!: () => void;
    const gate = new Promise<void>((r) => (release = r));
    let entered!: () => void;
    const inRetry = new Promise<void>((r) => (entered = r));
    const late = topLevel("dan", "p-dan", "dan");
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_u: unknown, init: any) => {
        const { method, id, params } = JSON.parse(init.body);
        const send = (r: Reply) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...r }));
        if (method === "database_api.get_dynamic_global_properties") return send({ result: { head_block_number: 6002, last_irreversible_block_num: 6002, time: "2026-10-13T19:20:00" } });
        if (method === "block_api.get_block_range") return send({ result: { blocks: [{ timestamp: "2026-10-13T19:20:00", transactions: [{ operations: [commentOp(late)] }] }] } });
        if (params[0] === "ann") {
          entered();
          await gate;
          return send({ result: topLevel("ann", "p-ann", "ann") });
        }
        return send({ error: { code: -32603, message: "Internal Error dan" } });
      }),
    );
    const retry = doo.retryParked();
    await inRetry;
    await doo.alarm(); // parks dan
    release();
    expect(await retry).toEqual({ retried: 1, ingested: 1, left: 1 });
    expect((await doo.status()).parked.map((p) => p.author)).toEqual(["dan"]);
    expect(await postsIn(env)).toEqual(["ann/p-ann"]);
  });
});
