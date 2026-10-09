import { describe, expect, it } from "vitest";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { seededVector, mix } from "./harness/net";
import { parseSearchRequest, requestKey } from "../src/search/params";
import { rankRetrieval, retrieve, search } from "../src/search/service";
import { loadContext } from "../src/search/context";
import { adaptiveKnn } from "../src/search/vectors";
import { writeSearchDoc } from "../src/db/posts";
import { sha256Hex } from "../src/lib/bytes";

const NOW = Math.floor(Date.now() / 1000);
const DAY = 86400;

async function addPost(env: TestEnv, o: { author: string; permlink: string; title: string; tags?: string[]; nsfw?: number; created?: number }): Promise<number> {
  const created = o.created ?? NOW - 30 * DAY;
  const r = await env.DB
    .prepare("INSERT INTO posts (author, permlink, type, title, description, tags_json, nsfw, created, updated, indexed_at) VALUES (?, ?, 'artwork', ?, '', ?, ?, ?, ?, ?)")
    .bind(o.author, o.permlink, o.title, JSON.stringify(o.tags ?? []), o.nsfw ?? 0, created, created, created)
    .run();
  const id = r.meta.last_row_id;
  await env.DB.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, updated) VALUES (?, ?, 'image/png', 100, ?)").bind(id, `h${id}`, created).run();
  await writeSearchDoc(env.DB, id, { author: o.author, title: o.title, description: "", body: "", tags: o.tags ?? [] });
  return id;
}

describe("full-text legs", () => {
  it("an exact all-words match keeps the AND label and outranks a prefix-only match", async () => {
    const env = makeEnv();
    const words = ["tree", "house", "sun", "moon", "river", "boat", "car", "girl", "sword", "castle", "flower", "star", "sky", "sea"];
    for (let i = 0; i < 120; i++) await addPost(env, { author: `u${i}`, permlink: `p${i}`, title: `${words[i % words.length]} ${words[(i * 7) % words.length]}` });
    for (let i = 0; i < 10; i++) await addPost(env, { author: `b${i}`, permlink: `b${i}`, title: `black ${words[i]}` });
    for (let i = 0; i < 10; i++) await addPost(env, { author: `c${i}`, permlink: `c${i}`, title: `${words[i]} cat`, tags: ["cat"] });
    for (let i = 0; i < 10; i++) await addPost(env, { author: `k${i}`, permlink: `k${i}`, title: `catalog ${words[i]}` });
    const exact = await addPost(env, { author: "x", permlink: "exact", title: "black cat", tags: ["cat"] });
    const prefix = await addPost(env, { author: "y", permlink: "prefix", title: "black catalog" });
    const ctx = await loadContext(env);
    const ret = await retrieve(env, parseSearchRequest(new URLSearchParams("q=black+cat")), ctx, { need: 24 });
    expect(ret.cands.get(exact)!.bm25Leg).toBe("and");
    expect(ret.cands.get(prefix)!.bm25Leg).toBe("and"); // "cat"* also matches catalog (type-ahead)
    expect(ret.cands.get(exact + 0)!.ranks.fts_or).toBeDefined(); // it is in the OR leg too
    const ranked = await rankRetrieval(env, ret, ctx);
    const pos = (id: number) => ranked.findIndex((r) => r.id === id);
    expect(pos(exact)).toBe(0);
    expect(pos(exact)).toBeLessThan(pos(prefix));
    const f = (id: number) => ranked.find((r) => r.id === id)!.features;
    expect(f(exact).lexical).toBeGreaterThan(f(prefix).lexical);
  });
});

describe("result cache", () => {
  it("serves a cached ranking only to the exact request, and re-filters the page", async () => {
    const env = makeEnv();
    const exec = new FakeExec();
    const safe = await addPost(env, { author: "alice", permlink: "cat", title: "a cat" });
    const nsfw = await addPost(env, { author: "mallory", permlink: "nsfw-cat", title: "cat", nsfw: 1 });
    const r = parseSearchRequest(new URLSearchParams("q=cat"));
    const key = `s3:${(await sha256Hex(new TextEncoder().encode(requestKey(r)))).slice(0, 40)}`;
    const poisoned = { ranked: [{ id: nsfw, f: 1, r: 1, ranks: {} }], mode: "text", notes: [], plan: {}, legs: {}, matchExpr: null };
    // an entry written for another request (a forged hash collision) is ignored...
    await env.CACHE.put(key, JSON.stringify({ ...poisoned, key: requestKey(parseSearchRequest(new URLSearchParams("q=cat&nsfw=only"))) }));
    let res = await search(env, r, exec as unknown as ExecutionContext);
    expect(res.items.map((i) => i.id)).toEqual([safe]);
    // ...and even a matching entry cannot put a filtered-out post on the page
    await env.CACHE.put(key, JSON.stringify({ ...poisoned, key: requestKey(r) }));
    res = await search(env, r, exec as unknown as ExecutionContext);
    expect(res.items.some((i) => i.id === nsfw)).toBe(false);
  });
});

describe("adaptive time slices (/ask)", () => {
  async function corpus(env: TestEnv, dim: number) {
    const axis = Array.from({ length: dim }, (_, i) => (i === 0 ? 1 : 0));
    let n = 0;
    const add = async (created: number, values: number[]) => {
      n++;
      await env._vec.upsert([{ id: String(n), values, metadata: { created, nsfw: false } }]);
    };
    // 1000 unrelated artworks over 100 days, then 150 cats within a few hours
    for (let i = 0; i < 1000; i++) await add(NOW - 100 * DAY + Math.floor((i / 1000) * 99 * DAY), seededVector(`r${i}`, dim));
    for (let i = 0; i < 150; i++) await add(NOW - DAY + i * 60, mix([[axis, 1], [seededVector(`c${i}`, dim), 0.15]]));
    return axis;
  }

  it("splits full slices until every cat is found, and says when the budget ran out", async () => {
    const env = makeEnv();
    const q = await corpus(env, 32);
    const r = await adaptiveKnn(env, "image", q, null, NOW - 100 * DAY, NOW + 1, { slices: 15, budget: 48 });
    const cats = r.hits.filter((h) => Number(h.id) > 1000);
    expect(cats.length).toBe(150);
    expect(r.truncated).toBe(false);
    expect(r.queries).toBeLessThanOrEqual(48);
    const small = await adaptiveKnn(env, "image", q, null, NOW - 100 * DAY, NOW + 1, { slices: 15, budget: 16 });
    expect(small.truncated).toBe(true); // a count over these hits is a lower bound
    expect(small.hits.filter((h) => Number(h.id) > 1000).length).toBeLessThan(150);
  });
});
