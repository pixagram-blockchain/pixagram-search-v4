// fetch() stub for tests: a fake Pixa chain (JSON-RPC) and a fake embedding endpoint.

import { vi } from "vitest";
import type { CondenserPost } from "../../src/chain/rpc";

export interface ChainFixture {
  posts: CondenserPost[];
  /** account history entries per account, newest last (as account_history_api returns them) */
  history?: Record<string, Array<[number, any]>>;
  blocks?: Record<number, any>;
  dgp?: { head_block_number: number; last_irreversible_block_num: number; time: string };
}

export type EmbedFn = (kind: "image" | "text", input: string) => number[];

export function installFetch(opts: { rpcUrl: string; embedUrl?: string; chain: ChainFixture; embed?: EmbedFn; model?: string; calibration?: { logit_scale: number; logit_bias: number }; spaceConcurrency?: number }) {
  // looked up on every call, so a test can edit or delete a post on the "chain" mid-test
  const byRef = { get: (ref: string) => opts.chain.posts.find((p) => `${p.author}/${p.permlink}` === ref) };
  const calls: Array<{ url: string; body: any }> = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ url, body });
    if (url.startsWith(opts.rpcUrl)) {
      const { method, params, id } = body;
      const ok = (result: unknown) => new Response(JSON.stringify({ jsonrpc: "2.0", id, result }), { status: 200 });
      const err = (message: string) => new Response(JSON.stringify({ jsonrpc: "2.0", id, error: { code: -32000, message } }), { status: 200 });
      if (method === "condenser_api.get_content") return ok(byRef.get(`${params[0]}/${params[1]}`) ?? { author: "", permlink: "" });
      if (method === "bridge.get_account_posts") {
        const all = opts.chain.posts.filter((p) => p.author === params.account && !p.parent_author).sort((a, b) => b.created.localeCompare(a.created));
        const from = params.start_permlink ? all.findIndex((p) => p.permlink === params.start_permlink) + 1 : 0;
        return ok(all.slice(from, from + (params.limit ?? 20)));
      }
      if (method === "condenser_api.lookup_accounts") return ok([...new Set(opts.chain.posts.map((p) => p.author))].sort());
      if (method === "account_history_api.get_account_history") {
        const h = (opts.chain.history?.[params.account] ?? []).filter(([seq]) => params.start < 0 || seq <= params.start);
        return ok({ history: h.slice(-params.limit) });
      }
      if (method === "database_api.get_dynamic_global_properties") return ok(opts.chain.dgp ?? { head_block_number: 100, last_irreversible_block_num: 99, time: "2026-10-04T00:00:00" });
      if (method === "block_api.get_block_range") {
        const blocks = [];
        for (let b = params.starting_block_num; b < params.starting_block_num + params.count; b++) if (opts.chain.blocks?.[b]) blocks.push(opts.chain.blocks[b]);
        return ok({ blocks });
      }
      return err(`unknown method ${method}`);
    }
    if (opts.embedUrl && url.startsWith(opts.embedUrl) && url.endsWith("/health")) {
      return new Response(JSON.stringify({ ok: true, ready: true, model: opts.model ?? "stub", concurrency: opts.spaceConcurrency ?? 2 }), { status: 200 });
    }
    if (opts.embedUrl && url.startsWith(opts.embedUrl)) {
      const inputs = body.inputs;
      const embeddings = [...(inputs.images ?? []).map((b: string) => opts.embed!("image", b)), ...(inputs.texts ?? []).map((t: string) => opts.embed!("text", t))];
      return new Response(JSON.stringify({ model: opts.model ?? "stub", dim: embeddings[0]?.length ?? 0, embeddings, calibration: opts.calibration }), { status: 200 });
    }
    throw new Error(`unexpected fetch ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { calls, fetchMock };
}

/** Unit vector from a seed (same idea as the Space's EMBED_STUB). */
export function seededVector(seed: string, dim: number): number[] {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) h = Math.imul(h ^ seed.charCodeAt(i), 16777619) >>> 0;
  const v: number[] = [];
  for (let i = 0; i < dim; i++) {
    h = Math.imul(h ^ (h >>> 15), 2246822507) >>> 0;
    h = Math.imul(h ^ (h >>> 13), 3266489909) >>> 0;
    v.push(((h >>> 0) / 4294967296) * 2 - 1);
  }
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

/** Mix unit vectors with weights and normalise (to build "similar" fixtures). */
export function mix(parts: Array<[number[], number]>): number[] {
  const out = new Array<number>(parts[0][0].length).fill(0);
  for (const [v, w] of parts) v.forEach((x, i) => (out[i] += x * w));
  const n = Math.hypot(...out) || 1;
  return out.map((x) => x / n);
}
