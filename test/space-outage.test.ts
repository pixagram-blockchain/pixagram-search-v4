// v4.8.2: a Space outage must not leave artworks without vectors for good, and /similar and
// /healthz must cope with one and say so. The v4 Space (free CPU) stopped answering on
// 29 October 2026 and was found asleep on 3 November: every embed and text failure counted toward the sweeper's 40-attempt cap, 298
// artworks reached it and stayed without vectors after the Space woke up, /similar returned
// nothing for them, and /healthz kept answering {"ok": true, "semantic": true}.

import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { codecsReady } from "./helpers";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, mix, type ChainFixture } from "./harness/net";
import { base64Encode } from "../src/lib/bytes";
import { encodePng, type RgbaImage } from "../src/enrich/decode";
import { ingestPost } from "../src/chain/ingest";
import { handleEnrichBatch, viewPng } from "../src/enrich/consumer";
import { sweep, sweepBudget } from "../src/enrich/sweeper";
import { spaceStatus } from "../src/enrich/embed";
import { vectorId } from "../src/db/posts";
import { app } from "../src/api";
import type { EnrichMessage } from "../src/env";

const DIM = 16;
/** A Space host: hf/app.py there always answers /health with JSON, so anything else means trouble. */
const SPACE = "https://primerz-pixagram-test.hf.space";
const T0 = Date.UTC(2026, 9, 29) / 1000; // 2026-10-29, the day the v4 Space fell asleep
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
const axis = (k: number) => Array.from({ length: DIM }, (_, i) => (i === k ? 1 : 0));

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number]): RgbaImage {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set([...f(x, y), 255], (y * w + x) * 4);
  return { width: w, height: h, data };
}

const ORANGE: [number, number, number] = [235, 140, 40];
const DARK: [number, number, number] = [40, 30, 20];
const ART = {
  // two cats, a dragon, and a sunset over the sea that shares the cats' orange but not their subject
  tabby: { title: "Tabby", tags: ["cat"], img: paint(16, 16, (x, y) => ((x - 8) ** 2 + (y - 9) ** 2 < 30 ? DARK : ORANGE)), meaning: mix([[axis(0), 1], [seededVector("tabby", DIM), 0.3]]) },
  kitten: { title: "Kitten", tags: ["cat", "kitten"], img: paint(16, 16, (x, y) => (x > 4 && x < 12 && y > 3 ? DARK : ORANGE)), meaning: mix([[axis(0), 1], [seededVector("kitten", DIM), 0.3]]) },
  dragon: { title: "Red dragon", tags: ["dragon"], img: paint(16, 16, (x, y) => ((x + y) % 4 < 2 ? [200, 30, 30] : [30, 160, 60])), meaning: mix([[axis(1), 1], [seededVector("dragon", DIM), 0.3]]) },
  // (colour buckets describe the foreground: the cats' dark shapes, the sea and a dark boat here)
  sunset: { title: "Sunset at sea", tags: ["sunset", "sea"], img: paint(16, 16, (x, y) => (y < 9 ? ORANGE : x > 4 && x < 11 && y < 12 ? DARK : [30, 60, 180])), meaning: mix([[axis(2), 1], [seededVector("sunset", DIM), 0.3]]) },
};
type Name = keyof typeof ART;

/**
 * What the Space does: "up" answers; "asleep" answers every route with Hugging Face's HTML page;
 * "down" does not answer; "refuse" answers /embed for images with 422 (it read the request and
 * refused the image, the one failure that is the input's).
 */
type Mode = "up" | "asleep" | "down" | "refuse";

async function world(o: { concurrency?: number; enrich?: boolean } = {}) {
  await codecsReady();
  const env: TestEnv = makeEnv({ HF_EMBED_URL: `${SPACE}/embed`, PLANNER_BACKEND: "rules", ADMIN_TOKEN: "t" });
  const exec = new FakeExec();
  const views = new Map<string, number[]>();
  const posts = [];
  let i = 0;
  for (const [name, a] of Object.entries(ART)) {
    const png = await encodePng(a.img);
    views.set(base64Encode(await viewPng("xbrz", a.img, env)), a.meaning);
    posts.push({
      author: `artist${i}`, permlink: name, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title: a.title,
      body: `data:image/png;base64,${base64Encode(png)}`,
      json_metadata: JSON.stringify({ app: "pixagram/3.0.2", format: "image", tags: a.tags }),
      created: iso(T0 + i * 3600), last_update: iso(T0 + i * 3600), depth: 0, children: 0, net_votes: 0,
      pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
    });
    i++;
  }
  const chain: ChainFixture = { posts: posts as any };
  installFetch({
    rpcUrl: "https://rpc.test",
    embedUrl: SPACE,
    chain,
    spaceConcurrency: o.concurrency ?? 8,
    embed: (kind, input) => (kind === "image" ? (views.get(input) ?? seededVector(input.slice(-48), DIM)) : seededVector(`t:${input}`, DIM)),
  });
  const space = { mode: "up" as Mode, health: 0, embeds: 0 };
  const inner = globalThis.fetch;
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url.startsWith(SPACE)) {
      const health = url.endsWith("/health");
      if (health) space.health++;
      else space.embeds++;
      if (space.mode === "asleep") return new Response("<!DOCTYPE html><html><body>This Space is sleeping due to inactivity.</body></html>", { status: 200, headers: { "content-type": "text/html" } });
      if (space.mode === "down") throw new TypeError("fetch failed");
      if (space.mode === "refuse" && !health && JSON.parse(String(init?.body ?? "{}")).inputs?.images) {
        return new Response(JSON.stringify({ detail: "UnidentifiedImageError: cannot identify image file" }), { status: 422 });
      }
    }
    return inner(input, init);
  });
  const ids: Record<Name, number> = {} as Record<Name, number>;
  for (const p of posts) ids[p.permlink as Name] = (await ingestPost(env, p as any, null, "test")).postId!;
  const pending: EnrichMessage[] = env._queue.drain();
  if (o.enrich !== false) await deliver(env, pending);
  const call = async (path: string) => {
    const res = await app.fetch(new Request(`https://search.test${path}`), env, exec as unknown as ExecutionContext);
    await exec.settle();
    return { status: res.status, body: (await res.json()) as any };
  };
  return { env, space, ids, pending, call };
}

/** One delivery of these messages, as the queue makes it (attempts: which delivery this is). */
async function deliver(env: TestEnv, msgs: EnrichMessage[], attempts = 1) {
  const acks: string[] = [];
  const batch = {
    queue: "q",
    messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts, body, ack: () => acks.push("ack"), retry: () => acks.push("retry") })),
    ackAll() {},
    retryAll() {},
  };
  await handleEnrichBatch(batch as any, env);
  return acks;
}

const job = (env: TestEnv, id: number, stage: string) =>
  env.DB.prepare("SELECT status, attempts, error FROM jobs WHERE post_id = ? AND stage = ?").bind(id, stage).first<{ status: string; attempts: number; error: string | null }>();

/** Time passes (Date only: timers keep running), past the minute a Space check stands for. */
function later(seconds: number) {
  vi.useFakeTimers({ toFake: ["Date"], now: Date.now() + seconds * 1000 });
}

/** The artwork as the outage left it: no vector in the index, embed job failed at the cap. */
async function strand(env: TestEnv, id: number, o: { updatedAgo?: number } = {}) {
  env._vec.v.delete(vectorId(id));
  await env.DB.prepare("UPDATE artworks SET embed_hash = NULL, embed_model = NULL, updated = ? WHERE post_id = ?").bind(Math.floor(Date.now() / 1000) - (o.updatedAgo ?? 0), id).run();
  await env.DB.prepare("UPDATE jobs SET status = 'failed', attempts = 40, error = 'embedding endpoint returned non-JSON (text/html)' WHERE post_id = ? AND stage = 'embed'").bind(id).run();
}

describe("an outage does not use up a post's attempts", () => {
  it("the Space asleep for 45 deliveries: the embed and text jobs fail, uncounted, and the other stages are done", async () => {
    const { env, space, ids, pending } = await world({ enrich: false });
    space.mode = "asleep";
    for (let d = 1; d <= 45; d++) expect(await deliver(env, pending, d)).toEqual(Array(4).fill("retry"));
    for (const id of Object.values(ids)) {
      expect(await job(env, id, "embed")).toMatchObject({ status: "failed", attempts: 0 });
      expect((await job(env, id, "embed"))!.error).toMatch(/non-JSON/);
      expect(await job(env, id, "text")).toMatchObject({ status: "failed", attempts: 0 });
      expect(await job(env, id, "stats")).toMatchObject({ status: "done" });
      expect(await job(env, id, "concepts")).toMatchObject({ status: "done" });
    }
  });

  it("down (no answer) is the endpoint's failure too; a 422 for the image is the input's and counts", async () => {
    const { env, space, ids, pending } = await world({ enrich: false });
    space.mode = "down";
    await deliver(env, pending);
    expect(await job(env, ids.tabby, "embed")).toMatchObject({ status: "failed", attempts: 0 });
    space.mode = "refuse";
    await deliver(env, pending, 2);
    await deliver(env, pending, 3);
    expect(await job(env, ids.tabby, "embed")).toMatchObject({ status: "failed", attempts: 2 });
    // the text vector does not depend on the image: done as soon as the Space answers
    expect(await job(env, ids.tabby, "text")).toMatchObject({ status: "done", attempts: 0 });
  });

  it("migration 0007 gives the embed and text jobs that failed before v4.8.2 a fresh count, and nothing else", async () => {
    const env = makeEnv();
    await env.DB.batch(
      [
        [1, "embed", "failed", 40], [1, "text", "failed", 41], [1, "describe", "failed", 40], [2, "embed", "done", 0], [3, "text", "failed", 3],
      ].map(([id, stage, status, n]) => env.DB.prepare("INSERT INTO jobs (post_id, stage, status, attempts, error, updated) VALUES (?, ?, ?, ?, 'x', 0)").bind(id, stage, status, n)),
    );
    await env.DB.prepare(readFileSync(new URL("../migrations/0007_outage_attempts.sql", import.meta.url), "utf8")).run();
    const rows = (await env.DB.prepare("SELECT post_id, stage, status, attempts FROM jobs ORDER BY post_id, stage").all()).results;
    expect(rows).toEqual([
      { post_id: 1, stage: "describe", status: "failed", attempts: 40 },
      { post_id: 1, stage: "embed", status: "failed", attempts: 0 },
      { post_id: 1, stage: "text", status: "failed", attempts: 0 },
      { post_id: 2, stage: "embed", status: "done", attempts: 0 },
      { post_id: 3, stage: "text", status: "failed", attempts: 0 },
    ]);
  });
});

describe("the sweeper", () => {
  it("holds the embedding stages back while the Space sleeps, and re-drives everything it missed once it is back", async () => {
    const { env, space, ids, pending } = await world({ enrich: false });
    space.mode = "asleep";
    await deliver(env, pending);
    const held = await sweep(env, 50);
    expect(held.enqueued).toBe(0);
    expect(held.byStage.embed).toBeUndefined();
    expect(held.held).toMatch(/held back, the Space is not ready: \/health answered with a page/);
    // the Space wakes up; the check of a minute ago no longer stands
    space.mode = "up";
    later(61);
    const r = await sweep(env, 50);
    expect(r.held).toBeUndefined();
    expect(r.byStage).toMatchObject({ embed: 4, text: 4 });
    const msgs = env._queue.drain();
    expect(msgs.map((m) => m.postId).sort()).toEqual(Object.values(ids).sort());
    expect(await deliver(env, msgs)).toEqual(Array(4).fill("ack"));
    const done = await env.DB.prepare("SELECT COUNT(*) AS n FROM artworks WHERE embed_hash = content_hash").first<{ n: number }>();
    expect(done!.n).toBe(4);
    expect(env._vec.v.size).toBe(4);
  });

  it("still gives up on a post whose own failures reached the cap", async () => {
    const { env, ids } = await world();
    await strand(env, ids.tabby);
    await env.DB.prepare("UPDATE jobs SET error = 'embedding endpoint HTTP 422: cannot identify image file' WHERE post_id = ? AND stage = 'embed'").bind(ids.tabby).run();
    expect((await sweep(env, 50)).byStage.embed).toBeUndefined();
    await env.DB.prepare("UPDATE jobs SET attempts = 0 WHERE post_id = ? AND stage = 'embed'").bind(ids.tabby).run();
    expect((await sweep(env, 50)).byStage.embed).toBe(1);
  });

  it("re-drives twelve posts a run per embedding the Space computes at once: 96 on 8 vCPU, 25 at least, 200 at most", async () => {
    expect(await sweepBudget((await world({ concurrency: 8, enrich: false })).env)).toBe(96);
    expect(await sweepBudget((await world({ concurrency: 2, enrich: false })).env)).toBe(25);
    expect(await sweepBudget((await world({ concurrency: 32, enrich: false })).env)).toBe(200);
    expect(await sweepBudget(makeEnv())).toBe(25);
  });
});

describe("the Space's state", () => {
  const probe = async (health: Record<string, unknown> | number | "page", over: Record<string, string> = {}) => {
    const env = makeEnv({ HF_EMBED_URL: `${SPACE}/embed`, EMBED_MODEL: "google/siglip2-base-patch16-naflex", EMBED_DIM: "768", EMBED_PATCHES: "576", ...over });
    let calls = 0;
    vi.stubGlobal("fetch", async () => {
      calls++;
      if (health === "page") return new Response("<html>Building</html>", { status: 200 });
      if (typeof health === "number") return new Response("not found", { status: health });
      return new Response(JSON.stringify(health), { status: 200 });
    });
    const s = await spaceStatus(env);
    return { s, env, calls: () => calls };
  };
  const good = { ok: true, ready: true, model: "google/siglip2-base-patch16-naflex", dim: 768, max_num_patches: 576, concurrency: 8, threads: 1, cpus: 8, uptime_s: 30 };

  it("ready when it serves the model, dimension and patch budget the indexes hold; checked once a minute", async () => {
    const { s, env, calls } = await probe(good);
    expect(s).toMatchObject({ configured: true, reachable: true, ready: true, problem: null, concurrency: 8, cpus: 8, threads: 1 });
    await spaceStatus(env);
    await spaceStatus(env);
    expect(calls()).toBe(1);
    later(61);
    await spaceStatus(env);
    expect(calls()).toBe(2);
  });

  it("not ready while loading, or with another model or patch budget; not reachable when it answers with a page", async () => {
    expect((await probe({ ...good, ready: false, dim: 0 })).s).toMatchObject({ reachable: true, ready: false, problem: "loading the model" });
    expect((await probe({ ...good, model: "google/siglip2-base-patch16-256" })).s.problem).toBe("serves google/siglip2-base-patch16-256 but EMBED_MODEL is google/siglip2-base-patch16-naflex");
    expect((await probe({ ...good, max_num_patches: 256 })).s.problem).toBe("256 patches per image but EMBED_PATCHES is 576");
    expect((await probe("page")).s).toMatchObject({ reachable: false, ready: false, problem: "/health answered with a page, not JSON (asleep, building or restarting)" });
    expect((await probe(503)).s).toMatchObject({ reachable: false, ready: false, problem: "/health HTTP 503" });
    // a private Space reached without a valid token, or a deleted one, answers 404
    expect((await probe(404)).s).toMatchObject({ reachable: false, ready: false, problem: "/health HTTP 404 (a private Space without a valid token, or no such Space)" });
  });

  it("an endpoint that is not a Space (an Inference Endpoint) may have no /health, or a plain one: nothing to check there", async () => {
    const other = { HF_EMBED_URL: "https://xyz.us-east-1.aws.endpoints.huggingface.cloud" };
    expect((await probe(404, other)).s).toMatchObject({ reachable: true, ready: true, problem: null });
    expect((await probe("page", other)).s).toMatchObject({ reachable: true, ready: true, problem: null });
    expect((await probe(503, other)).s).toMatchObject({ ready: false, problem: "/health HTTP 503" });
  });
});

describe("/healthz", () => {
  it("ok with the Space ready and every vector in place", async () => {
    const { call } = await world();
    const { status, body } = await call("/healthz");
    expect(status).toBe(200);
    expect(body).toMatchObject({ ok: true, status: "ok", problems: [], posts: 4, semantic: true, embedding: { ready: true, concurrency: 8 } });
    expect(body.vectors).toMatchObject({ artworks: 4, image_missing: 0, image_stuck: 0, text_missing: 0, text_stuck: 0, given_up: {} });
  });

  it("degraded while the Space sleeps: semantic false, and 503 for a monitor asking strict=1", async () => {
    const { call, space } = await world();
    space.mode = "asleep";
    later(61);
    const { status, body } = await call("/healthz");
    expect(status).toBe(200);
    expect(body).toMatchObject({ ok: true, status: "degraded", semantic: false, embedding: { reachable: false, ready: false } });
    expect(body.problems).toEqual(["embedding Space not ready: /health answered with a page, not JSON (asleep, building or restarting)"]);
    expect((await call("/healthz?strict=1")).status).toBe(503);
  });

  it("degraded when an artwork has been without its vector for over an hour, and says how many have given up", async () => {
    const { env, ids, call } = await world();
    await strand(env, ids.tabby, { updatedAgo: 7200 });
    await strand(env, ids.kitten);
    const { body } = await call("/healthz");
    expect(body.status).toBe("degraded");
    expect(body.problems).toEqual(["1 artwork without an image vector for over an hour"]);
    expect(body.vectors).toMatchObject({ image_missing: 2, image_stuck: 1, given_up: { embed: 2 } });
    expect((await call("/healthz?strict=1")).status).toBe(503);
  });
});

describe("/similar for an artwork without a vector", () => {
  it("embeds it on demand, finds its neighbours by that vector, caches it and queues the stage that stores it", async () => {
    const { env, space, ids, call } = await world();
    await strand(env, ids.tabby);
    const before = space.embeds;
    const health = space.health;
    const r = await call(`/similar/${ids.tabby}`);
    expect(r.status).toBe(200);
    expect(r.body).toMatchObject({ method: "hybrid", vector: "computed" });
    expect(space.health).toBe(health); // a request does not wait for /health
    expect(r.body.items.map((i: any) => i.permlink)[0]).toBe("kitten");
    expect(space.embeds).toBe(before + 1);
    const queued = env._queue.drain();
    expect(queued).toEqual([{ postId: ids.tabby, author: "artist0", permlink: "tabby", stages: ["embed"], reason: "similar" }]);
    expect(await job(env, ids.tabby, "embed")).toMatchObject({ status: "queued" });
    // again: the cached vector, no call to the Space, not queued twice
    const again = await call(`/similar/${ids.tabby}`);
    expect(again.body).toMatchObject({ method: "hybrid", vector: "computed" });
    expect(again.body.items[0].permlink).toBe("kitten");
    expect(space.embeds).toBe(before + 1);
    expect(env._queue.drain()).toEqual([]);
    // the queued stage stores it, without asking the Space again; /similar then reads the stored vector
    expect(await deliver(env, queued)).toEqual(["ack"]);
    expect(await job(env, ids.tabby, "embed")).toMatchObject({ status: "done", attempts: 0 });
    expect(space.embeds).toBe(before + 1);
    expect(env._vec.v.has(vectorId(ids.tabby))).toBe(true);
    expect((await call(`/similar/${ids.tabby}`)).body).toMatchObject({ method: "hybrid", vector: "stored" });
  });

  it("with the Space down, ranks by shared concepts, then colours, shape and pHash, instead of returning nothing", async () => {
    const { env, space, ids, call } = await world();
    await strand(env, ids.tabby);
    space.mode = "down";
    later(61);
    const r = await call(`/similar/${ids.tabby}`);
    expect(r.status).toBe(200);
    expect(r.body).toMatchObject({ method: "features", vector: "none" });
    expect(r.body.note).toBe("no image vector: embedding endpoint unreachable: fetch failed");
    const items = r.body.items as any[];
    expect(items[0].permlink).toBe("kitten");
    expect(items[0].similarity.theme).toBeGreaterThan(0.5);
    // the sunset shares a colour (the dark boat), not the subject: after the other cat; the dragon shares neither
    const sunset = items.find((i) => i.permlink === "sunset");
    expect(sunset.similarity.theme).toBe(0);
    expect(sunset.similarity.color).toBeGreaterThan(0);
    expect(items.map((i) => i.permlink)).toEqual(["kitten", "sunset"]);
    expect(env._queue.drain()).toEqual([]);
    // the next request in this isolate does not wait for the Space again
    const calls = space.embeds;
    expect((await call(`/similar/${ids.tabby}`)).body.note).toBe("no image vector: the Space is not ready: embedding endpoint unreachable: fetch failed");
    expect(space.embeds).toBe(calls);
    // /ask "similar to N" goes the same way
    const ask = await app.fetch(
      new Request("https://search.test/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question: `similar to ${ids.tabby}` }) }),
      env,
      new FakeExec() as unknown as ExecutionContext,
    );
    expect(((await ask.json()) as any).answer).toBeGreaterThan(0);
  });

  it("an image the Space refuses is not sent again for hours; computing is charged to the heavy rate limit", async () => {
    const { env, space, ids, call } = await world();
    await strand(env, ids.tabby);
    space.mode = "refuse";
    const before = space.embeds;
    const first = await call(`/similar/${ids.tabby}`);
    expect(first.body).toMatchObject({ method: "features", vector: "none" });
    expect(first.body.note).toMatch(/^no image vector: embedding endpoint HTTP 422/);
    expect(space.embeds).toBe(before + 1);
    const again = await call(`/similar/${ids.tabby}`);
    expect(again.body.note).toMatch(/^no image vector: embedding endpoint HTTP 422/);
    expect(again.body.items[0].permlink).toBe("kitten");
    expect(space.embeds).toBe(before + 1);
    // another artwork, with the heavy budget used up: the fallback, and the Space is not asked
    space.mode = "up";
    await strand(env, ids.kitten);
    env.RL_HEAVY = { limit: async () => ({ success: false }) } as any;
    const limited = await call(`/similar/${ids.kitten}`);
    expect(limited.body).toMatchObject({ method: "features", vector: "none", note: "no image vector: rate limited (computing a vector is a heavy request)" });
    expect(space.embeds).toBe(before + 1);
    env.RL_HEAVY = undefined;
    expect((await call(`/similar/${ids.kitten}`)).body).toMatchObject({ method: "hybrid", vector: "computed" });
  });

  it("a vector the index lost while the database says it is current: computed, and queued with force so the stage restores it", async () => {
    const { env, space, ids, call } = await world();
    env._vec.v.delete(vectorId(ids.sunset));
    const r = await call(`/similar/${ids.sunset}`);
    expect(r.body).toMatchObject({ method: "hybrid", vector: "computed" });
    const queued = env._queue.drain();
    expect(queued).toEqual([{ postId: ids.sunset, author: "artist3", permlink: "sunset", stages: ["embed"], force: true, reason: "similar" }]);
    const before = space.embeds;
    expect(await deliver(env, queued)).toEqual(["ack"]);
    expect(space.embeds).toBe(before + 1); // forced: computed again rather than taken from the cache
    expect(env._vec.v.has(vectorId(ids.sunset))).toBe(true);
  });

  it("with the Space known down, a cached vector queues the embed stage only (it needs no Space), not the text stage", async () => {
    const { env, space, ids, call } = await world();
    await strand(env, ids.tabby);
    await env.DB.prepare("UPDATE posts SET text_hash = NULL WHERE id = ?").bind(ids.tabby).run();
    expect((await call(`/similar/${ids.tabby}`)).body.vector).toBe("computed");
    expect(env._queue.drain()[0].stages).toEqual(["embed", "text"]);
    // half an hour later the queued mark has lapsed; the Space is now down, and /healthz has seen it
    space.mode = "down";
    later(31 * 60);
    expect((await call("/healthz")).body.status).toBe("degraded");
    const r = await call(`/similar/${ids.tabby}`);
    expect(r.body).toMatchObject({ method: "hybrid", vector: "computed" });
    const queued = env._queue.drain();
    expect(queued[0].stages).toEqual(["embed"]);
    // the embed stage stores the cached vector with the Space still down
    expect(await deliver(env, queued)).toEqual(["ack"]);
    expect(await job(env, ids.tabby, "embed")).toMatchObject({ status: "done" });
    expect(env._vec.v.has(vectorId(ids.tabby))).toBe(true);
  });

  it("a failing vector index: the fallback, without computing a vector it could not compare anyway", async () => {
    const { env, space, ids, call } = await world();
    const getByIds = env._vec.getByIds.bind(env._vec);
    env._vec.getByIds = async () => {
      throw new Error("VECTOR_GET_ERROR (code = 40008): internal error");
    };
    const before = space.embeds;
    const r = await call(`/similar/${ids.tabby}`);
    expect(r.body).toMatchObject({ method: "features", vector: "none", note: "vector index unavailable: VECTOR_GET_ERROR (code = 40008): internal error" });
    expect(r.body.items[0].permlink).toBe("kitten");
    expect(space.embeds).toBe(before);
    env._vec.getByIds = getByIds;
  });

  it("an artwork whose embedding is skipped for good gets the fallback without asking the Space; one with nothing in common, none", async () => {
    const { env, space, ids, call } = await world();
    await strand(env, ids.kitten);
    await env.DB.prepare("UPDATE jobs SET status = 'skipped', error = 'ai-training=false blocks embeddings' WHERE post_id = ? AND stage = 'embed'").bind(ids.kitten).run();
    const before = space.embeds;
    const r = await call(`/similar/${ids.kitten}`);
    expect(r.body).toMatchObject({ method: "features", vector: "none", note: "no image vector: embedding skipped: ai-training=false blocks embeddings" });
    expect(r.body.items[0].permlink).toBe("tabby");
    expect(space.embeds).toBe(before);
    // the dragon shares no concept, colour or shape hash with anything
    await strand(env, ids.dragon);
    await env.DB.prepare("UPDATE jobs SET status = 'skipped', error = 'unsupported image' WHERE post_id = ? AND stage = 'embed'").bind(ids.dragon).run();
    expect((await call(`/similar/${ids.dragon}`)).body).toMatchObject({ method: "none", vector: "none", items: [] });
  });
});
