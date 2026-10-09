// Regressions for the defects an independent review found before the first deployment. Each test
// reproduces the original failure scenario and checks the fixed behaviour.

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { codecsReady } from "./helpers";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, type ChainFixture } from "./harness/net";
import { base64Encode } from "../src/lib/bytes";
import { encodePng, type RgbaImage } from "../src/enrich/decode";
import { handleEnrichBatch } from "../src/enrich/consumer";
import { ingestPost, rpcFor } from "../src/chain/ingest";
import { relabelKinds, walkCommentHistory, writeVersions } from "../src/chain/versions";
import { sweep } from "../src/enrich/sweeper";
import { knn } from "../src/search/vectors";
import { emptyRequest } from "../src/search/params";
import { app } from "../src/api";
import type { EnrichMessage } from "../src/env";

const DIM = 16;
const T0 = Date.UTC(2026, 8, 1) / 1000;
const DAY = 86400;
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);

function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number, number]): RgbaImage {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
  return { width: w, height: h, data };
}
const uri = (mime: string, b: Uint8Array) => `data:${mime};base64,${base64Encode(b)}`;
const meta = (o: object) => JSON.stringify({ app: "pixagram/3.0.2", format: "image", ...o });
function chainPost(author: string, permlink: string, created: number, body: string, o: { title: string; tags?: string[]; updated?: number; votes?: number; blog?: boolean; description?: string }) {
  return {
    author, permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title: o.title, body,
    json_metadata: o.blog ? JSON.stringify({ app: "pixagram/3.0.2", format: "markdown", tags: o.tags ?? [] }) : meta({ tags: o.tags ?? [], description: o.description ?? "" }),
    created: iso(created), last_update: iso(o.updated ?? created), depth: 0, children: 0, net_votes: o.votes ?? 0,
    pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
  };
}
function hop(seq: number, at: number, author: string, permlink: string, title: string, body: string): [number, any] {
  return [seq, { trx_id: `${author}-${seq}`, block: 1000 + seq, trx_in_block: 0, op_in_trx: 0, virtual_op: false, timestamp: iso(at), op: { type: "comment_operation", value: { parent_author: "", parent_permlink: "pixagram", author, permlink, title, body, json_metadata: "{}" } } }];
}

async function setup(chain: ChainFixture, over: Record<string, string> = {}) {
  await codecsReady();
  const env = makeEnv({ HF_EMBED_URL: "https://embed.test/embed", EMBED_MODEL: "stub", EMBED_DIM: String(DIM), VLM_BACKEND: "off", ADMIN_TOKEN: "t", PLANNER_BACKEND: "rules", ...over } as any);
  installFetch({ rpcUrl: "https://rpc.test", embedUrl: "https://embed.test", chain, embed: (kind, input) => seededVector(`${kind}:${input.slice(-64)}`, DIM) });
  return { env, exec: new FakeExec() };
}
async function drain(env: TestEnv, msgs: EnrichMessage[] = env._queue.drain()) {
  const acks: string[] = [];
  await handleEnrichBatch({ queue: "q", messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => acks.push(`ack:${body.postId}`), retry: () => acks.push(`retry:${body.postId}`) })), ackAll() {}, retryAll() {} } as any, env);
  return acks;
}
async function call(env: TestEnv, exec: FakeExec, path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const res = await app.fetch(new Request(`https://search.test${path}`, init), env, exec as unknown as ExecutionContext);
  await exec.settle();
  const text = await res.text();
  let body: any = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* not JSON */
  }
  return { status: res.status, body };
}
const askQ = async (env: TestEnv, exec: FakeExec, question: string, extra: object = {}) =>
  (await call(env, exec, "/ask", { method: "POST", body: JSON.stringify({ question, ...extra }), headers: { "content-type": "application/json" } })).body;
const clearCache = (env: TestEnv) => {
  for (const k of [...env._kv.m.keys()]) if (k.startsWith("s3:")) env._kv.m.delete(k);
};
const checker = (c1: [number, number, number]) => paint(16, 16, (x, y) => ((x ^ y) & 1 ? [...c1, 255] : [20, 20, 20, 255]));

/** A PNG header that declares a huge image (nothing after IHDR): rejected before any decoding. */
function hugePngHeader(w: number, h: number): Uint8Array {
  const b = new Uint8Array(33);
  b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52]);
  new DataView(b.buffer).setUint32(16, w);
  new DataView(b.buffer).setUint32(20, h);
  b.set([8, 6, 0, 0, 0], 24);
  return b;
}

beforeAll(() => codecsReady());
afterEach(() => vi.unstubAllGlobals());

describe("ingestion", () => {
  it("an image edit that keeps the byte length is re-enriched", async () => {
    const red = await encodePng(paint(32, 32, () => [220, 20, 20, 255]));
    const blue = await encodePng(paint(32, 32, () => [20, 20, 220, 255]));
    expect(red.length).toBe(blue.length);
    const v1 = chainPost("alice", "art", T0, uri("image/png", red), { title: "Square" });
    const chain = { posts: [v1] as any[] };
    const { env } = await setup(chain);
    const r1 = await ingestPost(env, v1 as any, null, "test");
    await drain(env);
    const v2 = chainPost("alice", "art", T0, uri("image/png", blue), { title: "Square", updated: T0 + DAY });
    chain.posts[0] = v2;
    const r2 = await ingestPost(env, v2 as any, 123, "tail");
    expect(r2.enqueued).toBe(true);
    await drain(env);
    const after = await env.DB.prepare("SELECT primary_color FROM artworks WHERE post_id = ?").bind(r1.postId).first<any>();
    expect(after.primary_color).toBe("blue");
  });

  it("an older snapshot does not bring a deleted post back", async () => {
    const live = chainPost("alice", "dragon", T0, uri("image/png", await encodePng(checker([200, 30, 30]))), { title: "Dragon" });
    const chain = { posts: [live] as any[] };
    const { env, exec } = await setup(chain);
    await ingestPost(env, live as any, null, "backfill");
    await drain(env);
    const stale = structuredClone(live);
    const deleted = chainPost("alice", "dragon", T0, "deleted", { title: "Dragon", updated: T0 + DAY });
    chain.posts[0] = deleted;
    await ingestPost(env, deleted as any, 500, "tail");
    const r = await ingestPost(env, stale as any, null, "backfill");
    expect(r.enqueued).toBe(false);
    await drain(env);
    clearCache(env);
    expect((await env.DB.prepare("SELECT deleted FROM posts WHERE permlink = 'dragon'").first<any>()).deleted).toBe(1);
    expect((await call(env, exec, "/search?q=dragon")).body.items).toEqual([]);
  });

  it("a markdown post with an inline picture stays a blog post", async () => {
    const post = chainPost("carol", "tips", T0, `Shading tips.\n\n![step](${uri("image/png", await encodePng(checker([1, 2, 3])))})\n\nMore text.`, { title: "Tips", blog: true });
    const { env } = await setup({ posts: [post] as any });
    const r = await ingestPost(env, post as any, null, "test");
    expect((await env.DB.prepare("SELECT type, body FROM posts WHERE id = ?").bind(r.postId).first<any>())).toMatchObject({ type: "blog" });
  });
});

describe("enrichment stages record success last", () => {
  it("a transient R2 error during stats is redone on redelivery", async () => {
    const post = chainPost("alice", "art", T0, uri("image/png", await encodePng(checker([200, 30, 30]))), { title: "Art" });
    const { env } = await setup({ posts: [post] as any });
    const realPut = env._r2.put.bind(env._r2);
    let fail = true;
    (env._r2 as any).put = async (k: string, b: any, o: any) => {
      if (fail && k.startsWith("orig/")) {
        fail = false;
        throw new Error("R2 internal error (transient)");
      }
      return realPut(k, b, o);
    };
    await ingestPost(env, post as any, null, "test");
    const msgs = [...env._queue.sent];
    await drain(env);
    await drain(env, msgs); // redelivery
    const a = await env.DB.prepare("SELECT stats_hash = content_hash AS done, r2_orig_key, first_seen FROM artworks WHERE post_id = 1").first<any>();
    expect(a.done).toBe(1);
    expect(env._r2.m.has(a.r2_orig_key)).toBe(true);
    expect(a.first_seen).not.toBeNull();
  });

  it("a transient D1 error while writing the caption to full text leaves describe undone (the sweeper re-drives it)", async () => {
    const post = chainPost("alice", "art", T0, uri("image/png", await encodePng(checker([200, 30, 30]))), { title: "Untitled" });
    const { env, exec } = await setup({ posts: [post] as any }, { VLM_BACKEND: "moondream", HF_EMBED_URL: "" });
    env._ai.handler = (_m, input) =>
      input.task === "query" ? { answer: JSON.stringify({ caption: "A lighthouse on a rocky coast at night.", subjects: ["lighthouse"], objects: ["rocks"], tags: ["lighthouse", "coast"], style: "landscape", mood: "calm", nsfw: 0 }) } : { caption: "" };
    const db: any = env._db;
    const realPrepare = db.prepare.bind(db);
    let fail = true;
    db.prepare = (sql: string) => {
      if (fail && /^UPDATE search_docs SET ai_caption/.test(sql.trim())) {
        fail = false;
        throw new Error("D1_ERROR: Network connection lost.");
      }
      return realPrepare(sql);
    };
    await ingestPost(env, post as any, null, "test");
    await drain(env);
    expect((await env.DB.prepare("SELECT describe_hash IS NULL AS undone FROM artworks WHERE post_id = 1").first<any>()).undone).toBe(1);
    await env.DB.prepare("UPDATE jobs SET updated = updated - 3600").run(); // past the sweeper's 30-minute grace
    expect((await sweep(env, 10)).byStage.describe).toBe(1);
    await drain(env);
    expect((await call(env, exec, "/search?q=lighthouse")).body.items.length).toBe(1);
  });

  it("the AI NSFW estimate reaches the image vector although Vectorize reads lag behind writes", async () => {
    const post = chainPost("alice", "art", T0, uri("image/png", await encodePng(checker([200, 150, 130]))), { title: "Untitled" });
    const { env } = await setup({ posts: [post] as any }, { VLM_BACKEND: "moondream" });
    env._ai.handler = (_m, input) =>
      input.task === "query" ? { answer: JSON.stringify({ caption: "A nude figure lying on a bed.", subjects: ["nude"], objects: [], tags: ["nude", "figure"], style: "portrait", mood: "", nsfw: 0.95 }) } : { caption: "" };
    const vec: any = env._vec;
    const pending: any[] = [];
    const realUpsert = vec.upsert.bind(vec);
    vec.upsert = async (vs: any[]) => {
      pending.push(vs);
      return { mutationId: "m", count: vs.length };
    };
    await ingestPost(env, post as any, null, "test");
    await drain(env);
    for (const vs of pending) await realUpsert(vs); // applied later, in order
    expect(vec.v.get("1").metadata.nsfw).toBe(true);
    expect((await knn(env, "image", vec.v.get("1").values, emptyRequest({ nsfw: "only" }), 10)).length).toBe(1);
    expect(env._vecText.v.get("1")?.metadata?.nsfw).toBe(true);
  });

  it("an image too large to decode is skipped for good, before any pixel is allocated", async () => {
    const post = chainPost("mallory", "huge", T0, uri("image/png", hugePngHeader(4096, 4096)), { title: "Huge" });
    const { env } = await setup({ posts: [post] as any });
    await ingestPost(env, post as any, null, "test");
    const acks = await drain(env);
    expect(acks).toEqual(["ack:1"]);
    expect(await env.DB.prepare("SELECT status FROM jobs WHERE post_id = 1 AND stage = 'stats'").first("status")).toBe("skipped");
    expect((await sweep(env, 10)).enqueued).toBe(0);
  });
});

describe("history", () => {
  it("a recolour by the same author is a new artwork, not a near-duplicate", async () => {
    const slime = (body: [number, number, number]) => (x: number, y: number): [number, number, number, number] => {
      const dx = x - 12, dy = y - 14;
      return dx * dx + 1.6 * dy * dy < 90 ? [...body, 255] : [245, 245, 245, 255];
    };
    const green = uri("image/png", await encodePng(paint(24, 24, slime([40, 170, 60]))));
    const blue = uri("image/png", await encodePng(paint(24, 24, slime([40, 60, 200]))));
    const posts = [chainPost("eve", "green-slime", T0, green, { title: "Green slime", tags: ["slime"] }), chainPost("eve", "blue-slime", T0 + 10 * DAY, blue, { title: "Blue slime", tags: ["slime"] })];
    const history = { eve: [hop(0, T0 + 3, "eve", "green-slime", "Green slime", green), hop(1, T0 + 10 * DAY + 3, "eve", "blue-slime", "Blue slime", blue)] };
    const { env, exec } = await setup({ posts: posts as any, history }, { HF_EMBED_URL: "" });
    for (const p of posts) await ingestPost(env, p as any, null, "test");
    const { rows } = await walkCommentHistory(rpcFor(env), "eve", { maxCalls: 5 });
    await writeVersions(env.DB, rows);
    await relabelKinds(env, "eve");
    await drain(env);
    const row = await env.DB.prepare("SELECT a.first_seen_match, a.first_seen FROM artworks a JOIN posts p ON p.id = a.post_id WHERE p.permlink = 'blue-slime'").first<any>();
    expect(row.first_seen_match).toBe("self");
    expect((await askQ(env, exec, "When was the first blue slime posted?")).answer).toBe("2026-09-11");
  });
});

describe("/ask", () => {
  it("tone words filter first, last and most-liked questions too", async () => {
    const bright = await encodePng(paint(24, 24, (x, y) => ((x + y) % 2 ? [250, 240, 200, 255] : [230, 230, 120, 255])));
    const dark = await encodePng(paint(24, 24, (x, y) => ((x + y) % 2 ? [10, 10, 30, 255] : [40, 20, 20, 255])));
    const posts = [chainPost("alice", "sunny", T0, uri("image/png", bright), { title: "Sunny", votes: 9 }), chainPost("bob", "night", T0 + 5 * DAY, uri("image/png", dark), { title: "Night", votes: 1 })];
    const { env, exec } = await setup({ posts: posts as any }, { HF_EMBED_URL: "" });
    for (const p of posts) await ingestPost(env, p as any, null, "test");
    await drain(env);
    expect((await askQ(env, exec, "Who posted the first dark artwork?")).answer).toBe("bob");
    expect((await askQ(env, exec, "What is the most liked dark artwork?")).answer).toBe("/@bob/night");
    expect((await askQ(env, exec, "How many dark artworks are there?")).answer).toBe(1);
  });

  it("callers cannot force the LLM planner, and questions are capped", async () => {
    const { env, exec } = await setup({ posts: [] }, { PLANNER_BACKEND: "rules", HF_EMBED_URL: "" });
    env._ai.handler = () => ({ response: "{}" });
    const r = await askQ(env, exec, `cat ${"x".repeat(5_000)}`, { planner: "llm" });
    expect(env._ai.calls.length).toBe(0);
    expect(r.question.length).toBeLessThanOrEqual(300);
    const big = await call(env, exec, "/ask", { method: "POST", body: JSON.stringify({ question: "cat " + "x".repeat(200_000) }), headers: { "content-type": "application/json" } });
    expect(big.status).toBe(413);
  });
});

describe("public API", () => {
  it("deleted posts and their images are no longer served", async () => {
    const live = chainPost("alice", "selfie", T0, uri("image/png", await encodePng(checker([200, 30, 30]))), { title: "Me at the beach" });
    const chain = { posts: [live] as any[] };
    const { env, exec } = await setup(chain);
    await ingestPost(env, live as any, null, "test");
    await drain(env);
    const before = await call(env, exec, "/posts/1");
    const img = before.body.artwork.images.original;
    expect((await call(env, exec, img)).status).toBe(200);
    const del = chainPost("alice", "selfie", T0, "deleted", { title: "Me at the beach", updated: T0 + DAY });
    chain.posts[0] = del;
    await ingestPost(env, del as any, 9, "tail");
    expect((await call(env, exec, "/posts/1")).status).toBe(404);
    expect((await call(env, exec, "/posts/alice/selfie")).status).toBe(404);
    expect((await call(env, exec, img)).status).toBe(404);
    expect((await call(env, exec, "/history/1")).body.deleted).toBe(true); // the record stays readable
  });

  it("a query that is only @author or #tag browses with those filters", async () => {
    const posts = [
      chainPost("carol", "a", T0, "Drawing a dragon, step by step.", { title: "Dragon tutorial", tags: ["dragon"], blog: true }),
      chainPost("dan", "b", T0 + DAY, "My red dragon.", { title: "Red dragon", tags: ["dragon"], blog: true }),
    ];
    const { env, exec } = await setup({ posts: posts as any }, { HF_EMBED_URL: "" });
    for (const p of posts) await ingestPost(env, p as any, null, "test");
    const tag = await call(env, exec, `/search?q=${encodeURIComponent("#dragon")}`);
    expect(tag.body.mode).toBe("browse");
    expect(tag.body.items.map((i: any) => i.author)).toEqual(["dan", "carol"]);
    expect((await call(env, exec, `/search?q=${encodeURIComponent("@carol")}`)).body.items.map((i: any) => i.author)).toEqual(["carol"]);
  });

  it("facets describe the results, whichever leg found them", async () => {
    const posts = [
      chainPost("carol", "a", T0, "Drawing a dragon, step by step.", { title: "Dragon tutorial", blog: true }),
      chainPost("dan", "b", T0 + DAY, "My red dragon.", { title: "Red dragon", blog: true }),
    ];
    const { env, exec } = await setup({ posts: posts as any }, { HF_EMBED_URL: "" });
    for (const p of posts) await ingestPost(env, p as any, null, "test");
    for (const q of ["dargon", "dragon tutorial red"]) {
      const s = await call(env, exec, `/search?q=${encodeURIComponent(q)}&facets=1`);
      const n = s.body.facets.author.reduce((t: number, x: any) => t + x.n, 0);
      expect(n, q).toBe(s.body.items.length);
    }
  });

  it("an uploaded image is size-checked on its header before decoding", async () => {
    const { env, exec } = await setup({ posts: [] });
    const r = await call(env, exec, "/search-by-image", { method: "POST", body: JSON.stringify({ image: base64Encode(hugePngHeader(4096, 4096)) }), headers: { "content-type": "application/json" } });
    expect(r.status).toBe(413);
    const tooBig = await call(env, exec, "/search-by-image", { method: "POST", body: JSON.stringify({ image: "A".repeat(4 * 1024 * 1024) }), headers: { "content-type": "application/json" } });
    expect(tooBig.status).toBe(413);
  });

  it("anonymous searches write a bounded number of log rows; feedback needs a real query", async () => {
    const posts = Array.from({ length: 60 }, (_, i) => chainPost("carol", `post-${i}`, T0 + i * DAY, `Notes on pixel shading number ${i}.`, { title: `Shading ${i}`, blog: true }));
    const { env, exec } = await setup({ posts: posts as any }, { HF_EMBED_URL: "" });
    for (const p of posts) await ingestPost(env, p as any, null, "test");
    env._queue.drain();
    const N = 20;
    for (let i = 0; i < N; i++) await call(env, exec, "/search?q=shading&limit=50");
    for (let i = 0; i < N; i++) await call(env, exec, "/feedback", { method: "POST", body: JSON.stringify({ post_id: 1 + i, action: "like", query_id: "x" }) });
    const rows = (await env.DB.prepare("SELECT COUNT(*) AS n FROM rank_log").first<any>()).n;
    expect(rows).toBeLessThanOrEqual(N * 24);
    expect(rows).toBeLessThan(N * 50 * 0.6); // sampled (expected 25 %), and at most 24 rows each
    expect((await env.DB.prepare("SELECT COUNT(*) AS n FROM feedback").first<any>()).n).toBe(0);
  });
});
