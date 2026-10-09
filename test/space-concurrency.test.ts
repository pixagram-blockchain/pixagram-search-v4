// The queue consumer works on as many messages at once as the Space computes embeddings at once
// (its /health "concurrency": one per CPU, hf/siglip.py), so a Space on bigger hardware indexes
// faster without a deploy.

import { afterEach, describe, expect, it, vi } from "vitest";
import { codecsReady } from "./helpers";
import { makeEnv, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, type ChainFixture } from "./harness/net";
import { base64Encode } from "../src/lib/bytes";
import { encodePng, type RgbaImage } from "../src/enrich/decode";
import { ingestPost } from "../src/chain/ingest";
import { handleEnrichBatch } from "../src/enrich/consumer";
import { spaceSlots } from "../src/enrich/embed";
import type { EnrichMessage } from "../src/env";

const DIM = 16;
const T0 = Date.UTC(2026, 8, 1) / 1000;
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);

afterEach(() => vi.unstubAllGlobals());

function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number, number]): RgbaImage {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
  return { width: w, height: h, data };
}

function chainPost(permlink: string, created: number, body: string, title: string) {
  return {
    author: "alice", permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title, body,
    json_metadata: JSON.stringify({ app: "pixagram/3.0.2", format: "image", tags: ["art"] }),
    created: iso(created), last_update: iso(created), depth: 0, children: 0, net_votes: 0,
    pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
  };
}

/** Six artworks on the chain, a Space that answers /health with `health` (or not at all), and the embeddings in flight at once. */
async function setup(health: { concurrency?: number; cpus?: number } | "down", n = 6) {
  await codecsReady();
  const env: TestEnv = makeEnv({ HF_EMBED_URL: "https://embed.test/embed", PLANNER_BACKEND: "rules" });
  const posts = [];
  for (let i = 0; i < n; i++) {
    const png = await encodePng(paint(8, 8, (x, y) => [i * 40, x * 30, y * 30, 255]));
    posts.push(chainPost(`art-${i}`, T0 + i * 3600, `data:image/png;base64,${base64Encode(png)}`, `Artwork ${i}`));
  }
  const chain: ChainFixture = { posts: posts as any };
  installFetch({ rpcUrl: "https://rpc.test", embedUrl: "https://embed.test", chain, embed: (kind, input) => seededVector(`${kind}:${input.slice(-40)}`, DIM) });
  const stub = globalThis.fetch;
  const seen = { now: 0, max: 0, health: 0 };
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url === "https://embed.test/health") {
      seen.health++;
      if (health === "down") throw new TypeError("network error");
      return new Response(JSON.stringify({ ok: true, ...health }), { status: 200 });
    }
    if (url !== "https://embed.test/embed") return stub(input, init);
    seen.now++;
    seen.max = Math.max(seen.max, seen.now);
    try {
      await new Promise((r) => setTimeout(r, 15)); // the Space at work
      return await stub(input, init);
    } finally {
      seen.now--;
    }
  });
  for (const p of posts) await ingestPost(env, p as any, null, "test");
  return { env, seen };
}

async function consume(env: TestEnv) {
  const acks: string[] = [];
  const msgs: EnrichMessage[] = env._queue.drain() as any;
  const batch = {
    queue: "q",
    messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => acks.push("ack"), retry: () => acks.push("retry") })),
    ackAll() {},
    retryAll() {},
  };
  await handleEnrichBatch(batch as any, env);
  return { acks, n: msgs.length };
}

describe("the consumer keeps every CPU of the Space busy", () => {
  it("a Space computing three at once gets three at once, and every artwork is indexed", async () => {
    const { env, seen } = await setup({ concurrency: 3 });
    const { acks, n } = await consume(env);
    expect(n).toBe(6);
    expect(acks).toEqual(Array(6).fill("ack"));
    expect(seen.max).toBe(3);
    const done = await env.DB.prepare("SELECT COUNT(*) AS n FROM artworks WHERE embed_hash = content_hash").first<{ n: number }>();
    expect(done!.n).toBe(6);
  });

  it("an older Space reports its CPUs; an unreachable one counts as two; one at a time without a Space", async () => {
    const old = await setup({ cpus: 4 });
    await consume(old.env);
    expect(old.seen.max).toBe(4);
    const down = await setup("down");
    await consume(down.env);
    expect(down.seen.max).toBe(2);
    expect(await spaceSlots(makeEnv({ HF_EMBED_URL: "" }))).toBe(1);
  });

  it("asked every ten minutes, not for every batch: an upgrade shows within ten minutes", async () => {
    const { env, seen } = await setup({ concurrency: 8 });
    expect(await spaceSlots(env)).toBe(8);
    expect(await spaceSlots(env)).toBe(8);
    expect(seen.health).toBe(1);
    expect(await env.CACHE.get("space:slots")).toBe("8");
  });
});
