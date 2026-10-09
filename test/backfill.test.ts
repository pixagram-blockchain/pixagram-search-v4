import { describe, expect, it, vi } from "vitest";
import { addCounters, backfillAuthor, emptyCounters } from "../src/chain/backfill-author";
import { RpcError, type BridgePost } from "../src/chain/rpc";

const post = (author: string, i: number): BridgePost =>
  ({ author, permlink: `p${i}`, parent_author: "", parent_permlink: "art", title: `t${i}`, body: "", json_metadata: "{}", created: "2026-09-10T00:00:00", depth: 0, children: 0 }) as BridgePost;

// What api.pixagram.com answers for accounts that Hivemind does not know (seen 2026-09-30).
const unknownAccount = (name: string) =>
  new RpcError(`Assert Exception:Account ${name} does not exist`, "bridge.get_account_posts", -32602);

/** Fake Hivemind: `alice` has 45 posts (pages of 20, 20, 5), plus one reblog in the first page. */
function fakeRpc() {
  const calls: Array<{ author: string; start?: { author: string; permlink: string } }> = [];
  const getAccountPosts = vi.fn(async (author: string, start?: { author: string; permlink: string }, limit = 20) => {
    calls.push({ author, start });
    if (author === "pixa.omnibus") throw unknownAccount(author);
    if (author === "broken") throw new RpcError("Invalid parameters", "bridge.get_account_posts", -32602);
    if (author === "flaky") throw new Error("HTTP 502 from https://api.pixagram.com");
    if (author !== "alice") return [];
    const all = Array.from({ length: 45 }, (_, i) => post("alice", i));
    const from = start ? all.findIndex((p) => p.permlink === start.permlink) + 1 : 0;
    const page = all.slice(from, from + limit);
    if (!start) page[3] = post("bob", 999); // a reblog: must not be counted or ingested
    return page;
  });
  return { rpc: { getAccountPosts }, calls };
}

describe("backfillAuthor", () => {
  it("skips an account that Hivemind does not know, instead of failing the backfill", async () => {
    const { rpc } = fakeRpc();
    const ingest = vi.fn();
    const c = emptyCounters(1);
    await backfillAuthor(rpc, "pixa.omnibus", ingest, c);
    expect(c.missing).toBe(1);
    expect(c.posts).toBe(0);
    expect(ingest).not.toHaveBeenCalled();
  });

  it("walks every page and ingests each own post", async () => {
    const { rpc, calls } = fakeRpc();
    const ingest = vi.fn(async (p: BridgePost) => ({ action: p.permlink === "p0" ? "updated" : "inserted", enqueued: p.permlink !== "p0" }));
    const c = emptyCounters(1);
    await backfillAuthor(rpc, "alice", ingest, c);
    expect(calls.map((x) => x.start?.permlink ?? null)).toEqual([null, "p19", "p39"]);
    expect(c.posts).toBe(44); // 45 minus the reblog that replaced p3
    expect(c.inserted).toBe(43);
    expect(c.updated).toBe(1);
    expect(c.enqueued).toBe(43);
    expect(c.missing).toBe(0);
    expect(ingest.mock.calls.every(([p]) => p.author === "alice")).toBe(true);
  });

  it("still fails on other RPC errors and on transient errors, so Workflows retries the step", async () => {
    const { rpc } = fakeRpc();
    await expect(backfillAuthor(rpc, "broken", vi.fn(), emptyCounters(1))).rejects.toThrow("Invalid parameters");
    await expect(backfillAuthor(rpc, "flaky", vi.fn(), emptyCounters(1))).rejects.toThrow("HTTP 502");
  });

  it("adds step counters into the total, authors excluded", () => {
    const total = emptyCounters(78);
    addCounters(total, { authors: 5, posts: 3, inserted: 2, updated: 1, enqueued: 2, skipped: 0, missing: 1 });
    addCounters(total, { authors: 5, posts: 1, inserted: 1, updated: 0, enqueued: 1, skipped: 0, missing: 0 });
    expect(total).toEqual({ authors: 78, posts: 4, inserted: 3, updated: 1, enqueued: 3, skipped: 0, missing: 1 });
  });
});
