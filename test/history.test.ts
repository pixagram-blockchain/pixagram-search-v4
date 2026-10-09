import { afterEach, describe, expect, it, vi } from "vitest";
import { ChainRpc } from "../src/chain/rpc";
import { walkCommentHistory, bodyKind } from "../src/chain/versions";

// A node that behaves like account_history_api with operation_filter_low: it scans at most 2000
// operations below `start`, asserts start >= limit - 1, and when the filter finds nothing in the
// window it answers with the assert that tells where to continue.
function strictNode(ops: Array<{ type: string; value: any }>) {
  const calls: any[] = [];
  vi.stubGlobal("fetch", async (_url: string, init: RequestInit) => {
    const { id, params } = JSON.parse(String(init.body));
    calls.push(params);
    const reply = (x: object) => new Response(JSON.stringify({ jsonrpc: "2.0", id, ...x }));
    const fail = (message: string) => reply({ error: { code: -32003, message, data: { name: "assert_exception" } } });
    if (params.start >= 0 && params.start < params.limit - 1) return fail("Assert Exception:args.start >= args.limit-1: start must be greater than or equal to limit-1 (start is 0-based index)");
    let s = params.start < 0 ? ops.length - 1 : Math.min(params.start, ops.length - 1);
    const out: Array<[number, any]> = [];
    let scanned = 0;
    while (s >= 0 && scanned < 2000 && out.length < params.limit) {
      const op = ops[s];
      if (op.type === "comment_operation") out.push([s, { trx_id: `t${s}`, block: 1000 + s, trx_in_block: 0, op_in_trx: 0, virtual_op: false, timestamp: new Date(Date.UTC(2026, 8, 1) + s * 60_000).toISOString().slice(0, 19), op }]);
      s--;
      scanned++;
    }
    if (!out.length && s >= 0) return fail(`Assert Exception:false: Could not find filtered operation in 2000 operations, to continue searching, set start=${s}.`);
    return reply({ result: { history: out.reverse() } });
  });
  return calls;
}

const post = (permlink: string, body: string, extra: object = {}) => ({ type: "comment_operation", value: { parent_author: "", parent_permlink: "pixagram", author: "w", permlink, title: permlink, body, json_metadata: "{}", ...extra } });
const reward = { type: "producer_reward_operation", value: {} };

afterEach(() => vi.unstubAllGlobals());

describe("account history walk", () => {
  it("continues past the 2000-operation scan window and through the start >= limit-1 assert", async () => {
    // a witness: thousands of producer rewards between its posts
    const ops = [post("first", "hello"), ...Array(4500).fill(reward), post("first", "hello again"), ...Array(2500).fill(reward), post("second", "deleted"), post("reply", "hi", { parent_author: "x" })];
    const calls = strictNode(ops);
    const rpc = new ChainRpc({ url: "https://rpc.test", retries: 0 });
    const { rows, result } = await walkCommentHistory(rpc, "w", { maxCalls: 50, pageSize: 100 });
    expect(result.complete).toBe(true);
    expect(rows.map((r) => [r.permlink, r.kind])).toEqual([
      ["first", "create"],
      ["first", "edit"],
      ["second", "delete"],
    ]); // the reply is not a version
    expect(calls.length).toBeGreaterThan(3); // several "set start=" continuations
    expect(calls.every((p) => p.operation_filter_low === 2)).toBe(true);
    expect(calls.every((p) => p.start < 0 || p.start >= p.limit - 1)).toBe(true);
  });

  it("an incomplete walk (call budget spent) labels nothing as a creation", async () => {
    const ops = [post("a", "x"), ...Array(6000).fill(reward), post("a", "y")];
    strictNode(ops);
    const rpc = new ChainRpc({ url: "https://rpc.test", retries: 0 });
    const { rows, result } = await walkCommentHistory(rpc, "w", { maxCalls: 2, pageSize: 100 });
    expect(result.complete).toBe(false);
    expect(rows.map((r) => r.kind)).toEqual(["edit"]); // relabelKinds() fixes this from posts.created
  });

  it("classifies bodies", () => {
    expect(bodyKind("deleted")).toBe("deleted");
    expect(bodyKind(" Deleted ")).toBe("deleted");
    expect(bodyKind("@@ -1,3 +1,3 @@")).toBe("patch");
    expect(bodyKind("data:image/webp;base64,UklGRg==")).toBe("image");
    expect(bodyKind("# a blog post")).toBe("text");
  });
});
