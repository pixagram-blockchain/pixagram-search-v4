// End to end through the Worker's own code: chain → ingestion → edit history → enrichment
// (stats, features, hashes, vectors, VLM description, concepts, text vectors) → the HTTP API
// (/search, /ask, /duplicates, /similar, /history, /feedback, admin) → deletion. SQLite stands in
// for D1 (with the real migrations), in-memory fakes for KV/R2/Queues/Vectorize/Workers AI, and
// fetch() is a fake chain plus a fake embedding endpoint (test/harness).

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { codecsReady, fixture } from "./helpers";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, mix, type ChainFixture } from "./harness/net";
import { base64Encode } from "../src/lib/bytes";
import { decodeImage, encodePng, type RgbaImage } from "../src/enrich/decode";
import { ingestPost, rpcFor } from "../src/chain/ingest";
import { artworksWithHashes, refreshImageHistory, relabelKinds, walkCommentHistory, writeVersions } from "../src/chain/versions";
import { handleEnrichBatch, viewPng } from "../src/enrich/consumer";
import { matchConcepts } from "../src/concepts";
import { app } from "../src/api";
import type { EnrichMessage } from "../src/env";
import { vi } from "vitest";

const DIM = 16;
const T0 = Date.UTC(2026, 8, 1) / 1000; // 2026-09-01
const DAY = 86400;
const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
const TOKEN = "admin-secret";

// Orthogonal "meaning" axes so the fake embedder is unambiguous in 16 dimensions.
const axis = (k: number) => Array.from({ length: DIM }, (_, i) => (i === k ? 1 : 0));
const MEANING: Record<string, number[]> = { cat: axis(0), dragon: axis(1), woman: axis(2), sunset: axis(3) };

let env: TestEnv;
let exec: FakeExec;
const ids: Record<string, number> = {};

async function call(path: string, init?: RequestInit & { admin?: boolean }): Promise<any> {
  const headers = new Headers(init?.headers);
  if (init?.admin) headers.set("authorization", `Bearer ${TOKEN}`);
  const res = await app.fetch(new Request(`https://search.test${path}`, { ...init, headers }), env, exec as unknown as ExecutionContext);
  await exec.settle();
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number, number]): RgbaImage {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
  return { width: w, height: h, data };
}

const meta = (o: object) => JSON.stringify({ app: "pixagram/3.0.2", format: "image", ...o });
function chainPost(author: string, permlink: string, created: number, body: string, o: { title: string; tags?: string[]; updated?: number; votes?: number; blog?: boolean; description?: string }) {
  return {
    author, permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title: o.title, body,
    json_metadata: o.blog ? JSON.stringify({ app: "pixagram/3.0.2", format: "markdown", tags: o.tags ?? [] }) : meta({ tags: o.tags ?? [], description: o.description ?? "" }),
    created: iso(created), last_update: iso(o.updated ?? created), depth: 0, children: 0, net_votes: o.votes ?? 0,
    pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
  };
}
function historyOp(seq: number, at: number, op: { author: string; permlink: string; title: string; body: string; parent_author?: string }): [number, any] {
  return [seq, { trx_id: `${op.author}-${seq}`, block: 1000 + seq, trx_in_block: 0, op_in_trx: 0, virtual_op: false, timestamp: iso(at), op: { type: "comment_operation", value: { parent_author: "", parent_permlink: "pixagram", json_metadata: meta({}), ...op } } }];
}

beforeAll(async () => {
  await codecsReady();
  const webpA = fixture("small.webp"); // the "cat" artwork
  const webpB = fixture("second.webp"); // the "dragon" artwork
  const imgA = await decodeImage(webpA);
  const imgB = await decodeImage(webpB);
  const pngA = await encodePng(imgA); // the same pixels re-encoded: a re-upload
  const imgC = paint(24, 40, (x, y) => { const v = (x * 7 + y * 3) % 2 ? 235 : 15; return [v, v, v, 255]; }); // black and white, portrait
  const pngC = await encodePng(imgC);
  const uri = (mime: string, b: Uint8Array) => `data:${mime};base64,${base64Encode(b)}`;
  const A = uri("image/webp", webpA), A2 = uri("image/png", pngA), B = uri("image/webp", webpB), C = uri("image/png", pngC);

  const posts = [
    chainPost("alice", "cat-1", T0, "deleted", { title: "The King of the Cats", tags: ["cat"], updated: T0 + 10 * DAY }),
    chainPost("alice", "cat-2", T0 + 11 * DAY, A2, { title: "The King of the Cats", tags: ["cat", "kitty"], votes: 3 }),
    chainPost("bob", "my-cat", T0 + 20 * DAY, A, { title: "My cat", tags: ["pet"], votes: 1 }),
    chainPost("dave", "red-dragon", T0 + 5 * DAY, B, { title: "Red Dragon", tags: ["dragon", "fantasy"], votes: 7 }),
    chainPost("dave", "noir", T0 + 7 * DAY, C, { title: "Noir", tags: ["noir"] }),
    chainPost("carol", "why-i-draw-cats", T0 + 6 * DAY, "I love drawing **cats** in pixels. Here is why.", { title: "Why I draw cats", tags: ["blog"], blog: true }),
  ];
  const chain: ChainFixture = {
    posts: posts as any,
    history: {
      alice: [
        historyOp(0, T0 + 3, { author: "alice", permlink: "cat-1", title: "The King of the Cats", body: A }),
        historyOp(1, T0 + 2 * DAY, { author: "alice", permlink: "cat-1", title: "The King of the Cats!", body: A }),
        historyOp(2, T0 + 3 * DAY, { author: "alice", permlink: "re-dave", title: "", body: "nice", parent_author: "dave" }),
        historyOp(3, T0 + 10 * DAY, { author: "alice", permlink: "cat-1", title: "The King of the Cats", body: "deleted" }),
        historyOp(4, T0 + 11 * DAY + 3, { author: "alice", permlink: "cat-2", title: "The King of the Cats", body: A2 }),
      ],
      bob: [historyOp(0, T0 + 20 * DAY + 3, { author: "bob", permlink: "my-cat", title: "My cat", body: A })],
      dave: [historyOp(0, T0 + 5 * DAY + 3, { author: "dave", permlink: "red-dragon", title: "Red Dragon", body: B }), historyOp(1, T0 + 7 * DAY + 3, { author: "dave", permlink: "noir", title: "Noir", body: C })],
    },
  };

  env = makeEnv({ HF_EMBED_URL: "https://embed.test/embed", EMBED_MODEL: "stub", EMBED_DIM: String(DIM), VLM_BACKEND: "moondream", ADMIN_TOKEN: TOKEN, PLANNER_BACKEND: "rules", RANK_LOG_SAMPLE: "1" });
  exec = new FakeExec();

  // The fake embedder: images by what they show, texts by the concepts they mention.
  const views = new Map<string, number[]>();
  views.set(base64Encode(await viewPng("xbrz", imgA, env)), mix([[MEANING.cat, 1], [seededVector("imgA", DIM), 0.25]]));
  views.set(base64Encode(await viewPng("xbrz", imgB, env)), mix([[MEANING.dragon, 1], [seededVector("imgB", DIM), 0.25]]));
  views.set(base64Encode(await viewPng("xbrz", imgC, env)), mix([[MEANING.woman, 0.6], [seededVector("imgC", DIM), 0.8]]));
  installFetch({
    rpcUrl: "https://rpc.test",
    embedUrl: "https://embed.test",
    chain,
    embed: (kind, input) => {
      if (kind === "image") return views.get(input) ?? seededVector(input.slice(-64), DIM);
      const parts: Array<[number[], number]> = [[seededVector(`t:${input}`, DIM), 0.2]];
      for (const m of matchConcepts(input)) if (MEANING[m.concept]) parts.push([MEANING[m.concept], 1]);
      return mix(parts);
    },
  });
  // The VLM describes by title (the prompt carries it).
  env._ai.handler = (_model, input) => {
    const q = String(input.question ?? "");
    const caption = q.includes("King of the Cats") || q.includes("My cat") ? "A crowned cat sitting on a golden throne." : q.includes("Dragon") ? "A red dragon breathing fire over a castle." : "A woman in a hat, in black and white.";
    return input.task === "query" ? { answer: JSON.stringify({ caption, subjects: [], objects: ["throne"], tags: caption.toLowerCase().replace(/[^a-z ]/g, "").split(" ").filter((w) => w.length > 3), style: "portrait", mood: "calm", nsfw: 0 }) } : { caption };
  };

  // ---- ingestion, as the backfill does it -----------------------------------------------------------
  for (const p of posts) {
    const r = await ingestPost(env, p as any, null, "test");
    if (r.postId) ids[`${p.author}/${p.permlink}`] = r.postId;
  }
  // ---- edit history -----------------------------------------------------------------------------------
  for (const author of Object.keys(chain.history!)) {
    const { rows } = await walkCommentHistory(rpcFor(env), author, { maxCalls: 10 });
    await writeVersions(env.DB, rows);
    await relabelKinds(env, author);
  }
  // ---- enrichment, through the queue consumer -------------------------------------------------------
  const acks: string[] = [];
  const batch = (msgs: EnrichMessage[]) => ({
    queue: "q",
    messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => acks.push(`ack:${body.postId}`), retry: () => acks.push(`retry:${body.postId}`) })),
    ackAll() {},
    retryAll() {},
  });
  await handleEnrichBatch(batch(env._queue.drain()) as any, env);
  expect(acks.every((a) => a.startsWith("ack:"))).toBe(true);
  const hashes = ((await env.DB.prepare("SELECT content_hash FROM artworks").all<{ content_hash: string }>()).results ?? []).map((r) => r.content_hash);
  for (const id of await artworksWithHashes(env.DB, hashes)) await refreshImageHistory(env.DB, id);
}, 120_000);

afterAll(() => vi.unstubAllGlobals());

describe("ingestion and enrichment", () => {
  it("indexes live posts, not the one already deleted", async () => {
    expect(Object.keys(ids).sort()).toEqual(["alice/cat-1", "alice/cat-2", "bob/my-cat", "carol/why-i-draw-cats", "dave/noir", "dave/red-dragon"]);
    const cat1 = await env.DB.prepare("SELECT deleted FROM posts WHERE id = ?").bind(ids["alice/cat-1"]).first<{ deleted: number }>();
    expect(cat1?.deleted).toBe(1);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM search_docs WHERE post_id = ?").bind(ids["alice/cat-1"]).first("n")).toBe(0);
  });

  it("every stage ran for every artwork; the blog got its text vector only", async () => {
    const a = await env.DB
      .prepare(`SELECT COUNT(*) AS n, SUM(embed_hash = content_hash) AS e, SUM(features_hash = content_hash) AS f, SUM(phash_hi IS NOT NULL AND dhash IS NOT NULL) AS h,
                SUM(COALESCE(ai_caption, '') != '') AS d, SUM(ai_status = 'ok') AS ok, SUM(concepts_hash IS NOT NULL) AS c FROM artworks`)
      .first<any>();
    expect(a).toEqual({ n: 4, e: 4, f: 4, h: 4, d: 4, ok: 4, c: 4 });
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM phash_bands").first("n")).toBe(16);
    expect(env._vec.v.size).toBe(4);
    expect(env._vecText.v.size).toBe(5);
    expect(env._vec.v.get(String(ids["dave/noir"]))!.metadata).toMatchObject({ author: "dave", orientation: "portrait", transparent: false, listed: false, ai_training: true });
    expect(env._vecText.v.get(String(ids["carol/why-i-draw-cats"]))!.metadata).toMatchObject({ type: "blog" });
    const failed = await env.DB.prepare("SELECT COUNT(*) AS n FROM jobs WHERE status = 'failed'").first("n");
    expect(failed).toBe(0);
    // originals and previews in R2, content-addressed
    expect([...env._r2.m.keys()].filter((k) => k.startsWith("orig/")).length).toBe(4);
    expect([...env._r2.m.keys()].every((k) => /^(orig|up)\/[0-9a-f]{64}\.(webp|png)$/.test(k))).toBe(true);
  });

  it("features: the black-and-white portrait is greyscale and portrait", async () => {
    const r = await env.DB.prepare("SELECT colorfulness, monochrome, orientation FROM artworks WHERE post_id = ?").bind(ids["dave/noir"]).first<any>();
    expect(r.colorfulness).toBeLessThan(5);
    expect(r.monochrome).toBe(1);
    expect(r.orientation).toBe("portrait");
  });

  it("concepts come from tags, title and the AI description, in any language", async () => {
    const c = await env.DB.prepare("SELECT concept, source FROM artwork_concepts WHERE post_id = ? ORDER BY confidence DESC").bind(ids["bob/my-cat"]).all<any>();
    const concepts = c.results.map((x: any) => x.concept);
    expect(concepts).toEqual(expect.arrayContaining(["cat", "pet", "animal"]));
  });
});

describe("edit history", () => {
  it("records create, edit and delete-by-edit; replies are not versions", async () => {
    const v = await env.DB.prepare("SELECT permlink, kind, body_kind FROM post_versions WHERE author = 'alice' AND source = 'history' ORDER BY at").all<any>();
    expect(v.results.map((x: any) => `${x.permlink}:${x.kind}`)).toEqual(["cat-1:create", "cat-1:edit", "cat-1:delete", "cat-2:create"]);
  });

  it("first_seen follows an image to its first appearance: exact bytes, or a re-encoded re-upload", async () => {
    const q = (k: string) => env.DB.prepare("SELECT first_seen, first_seen_author, first_seen_permlink, first_seen_match, image_since, history_exact FROM artworks WHERE post_id = ?").bind(ids[k]).first<any>();
    // bob posted alice's (since deleted) bytes
    expect(await q("bob/my-cat")).toMatchObject({ first_seen: T0 + 3, first_seen_author: "alice", first_seen_permlink: "cat-1", first_seen_match: "exact", image_since: T0 + 20 * DAY });
    // alice re-uploaded her own deleted artwork, re-encoded as PNG: same pHash
    expect(await q("alice/cat-2")).toMatchObject({ first_seen: T0 + 3, first_seen_permlink: "cat-1", first_seen_match: "near", image_since: T0 + 11 * DAY, history_exact: 1 });
    expect(await q("dave/red-dragon")).toMatchObject({ first_seen: T0 + 5 * DAY, first_seen_match: "self" });
  });

  it("GET /history/:id", async () => {
    const { status, body } = await call(`/history/${ids["bob/my-cat"]}`);
    expect(status).toBe(200);
    expect(body.same_image_in.map((x: any) => `${x.author}/${x.permlink}`)).toEqual(["alice/cat-1", "bob/my-cat"]);
    expect(body.first_seen).toBe(T0 + 3);
  });
});

describe("search", () => {
  it("exact title first, with a query id, explain features and the plan", async () => {
    const { body } = await call("/search?q=the+king+of+the+cats&explain=1");
    expect(body.mode).toBe("hybrid");
    expect(`${body.items[0].author}/${body.items[0].permlink}`).toBe("alice/cat-2");
    expect(body.items[0].features.title).toBe(1);
    expect(body.query_id).toMatch(/^[0-9a-z]{8,}$/);
    expect(body.plan.concepts).toContain("cat");
    expect(body.items.slice(0, 2).map((i: any) => i.permlink).sort()).toEqual(["cat-2", "my-cat"]);
  });

  it("finds what only the AI description mentions", async () => {
    const { body } = await call("/search?q=throne");
    expect(body.items.slice(0, 2).map((i: any) => i.author).sort()).toEqual(["alice", "bob"]);
  });

  it("multilingual: French 'chat' finds the cats through concepts and vectors", async () => {
    const { body } = await call("/search?q=chat&type=artwork");
    expect(body.items.slice(0, 2).map((i: any) => i.permlink).sort()).toEqual(["cat-2", "my-cat"]);
  });

  it("typos: did_you_mean and the corrected variant", async () => {
    const { body } = await call("/search?q=dargon");
    expect(body.did_you_mean).toBe("dragon");
    expect(body.items[0].permlink).toBe("red-dragon");
  });

  it("colour-led and tone queries", async () => {
    // The fixture cat is a 4-colour grey image: greyscale too. All three greyscale artworks tie
    // on tone; noir is the most relevant (its caption says "black and white"), and the votes of
    // the cat may still break that near-tie (quality is bounded to ±10 %).
    const { body } = await call("/search?q=black+and+white&type=artwork&semantic=0&explain=1");
    expect(body.plan.hints.tones).toEqual(["greyscale"]);
    expect(body.items.slice(0, 3).map((i: any) => i.permlink).sort()).toEqual(["cat-2", "my-cat", "noir"]);
    const best = [...body.items].sort((a: any, b: any) => b.score.relevance - a.score.relevance)[0];
    expect(best.permlink).toBe("noir");
    expect(body.items.find((i: any) => i.permlink === "my-cat").score.duplicate_of).toBe(ids["alice/cat-2"]); // stepped down below its copy
    const k = body.items.findIndex((i: any) => i.permlink === "red-dragon");
    expect(k === -1 || k >= 3).toBe(true); // the colourful one is not a greyscale match
    const lime = await call("/search?q=lime&type=artwork&semantic=0");
    expect(lime.body.items[0].permlink).toBe("red-dragon"); // lime on black
  });

  it("filters, facets, browse and v2's ranking for comparison", async () => {
    const f = await call("/search?q=cat&author=bob&facets=1");
    expect(f.body.items.map((i: any) => i.author)).toEqual(["bob"]);
    expect(f.body.facets.author).toEqual([{ key: "bob", n: 1 }]);
    const o = await call("/search?q=noir&orientation=landscape");
    expect(o.body.items.some((i: any) => i.permlink === "noir")).toBe(false); // noir is portrait
    const b = await call("/search?type=artwork");
    expect(b.body.mode).toBe("browse");
    expect(b.body.items[0].permlink).toBe("my-cat"); // newest first
    const rrf = await call("/search?q=dragon&rank=rrf");
    expect(rrf.body.rank).toBe("rrf");
    expect(rrf.body.items[0].permlink).toBe("red-dragon");
  });

  it("logs ranked results, accepts feedback and exports training rows", async () => {
    const s = await call("/search?q=dragon+fire");
    const qid = s.body.query_id;
    const top = s.body.items[0];
    expect((await call("/feedback", { method: "POST", body: JSON.stringify({ query_id: qid, post_id: top.id, rank: 1, action: "like" }) })).status).toBe(204);
    expect((await call("/feedback", { method: "POST", body: JSON.stringify({ query_id: qid, post_id: top.id, action: "steal" }) })).status).toBe(400);
    // only for a real query and one of its results
    expect((await call("/feedback", { method: "POST", body: JSON.stringify({ query_id: "nope", post_id: top.id, action: "click" }) })).status).toBe(400);
    const notShown = Object.values(ids).find((id) => !s.body.items.some((i: any) => i.id === id))!;
    expect((await call("/feedback", { method: "POST", body: JSON.stringify({ query_id: qid, post_id: notShown, action: "click" }) })).status).toBe(400);
    expect((await call("/feedback", { method: "POST", body: "x".repeat(5000) })).status).toBe(413);
    const ex = await call("/admin/ltr/export?days=1", { admin: true });
    const rows = ex.body.filter((r: any) => r.qid === qid);
    expect(rows.length).toBe(s.body.items.length);
    expect(rows.find((r: any) => r.post_id === top.id).label).toBe(3);
    expect(rows[0].features).toHaveProperty("color_led");
  });
});

describe("visual", () => {
  it("/duplicates finds the re-upload and the stolen copy, by bands and by scan", async () => {
    const bands = await call(`/duplicates/${ids["bob/my-cat"]}`);
    expect(bands.body.method).toBe("bands");
    expect(bands.body.items.map((i: any) => [i.permlink, i.distance])).toEqual([["cat-2", 0]]);
    const scan = await call(`/duplicates/${ids["bob/my-cat"]}?max_distance=20`);
    expect(scan.body.method).toBe("scan");
    expect(scan.body.items[0].permlink).toBe("cat-2");
  });

  it("/similar puts the duplicate first", async () => {
    const { body } = await call(`/similar/${ids["bob/my-cat"]}`);
    expect(body.items[0].permlink).toBe("cat-2");
  });

  it("/similar, /duplicates and /ask similar exclude NSFW unless asked, like /search", async () => {
    await env.DB.prepare("UPDATE posts SET nsfw = 1 WHERE id = ?").bind(ids["alice/cat-2"]).run();
    try {
      const me = ids["bob/my-cat"];
      expect((await call(`/similar/${me}`)).body.items.some((i: any) => i.permlink === "cat-2")).toBe(false);
      expect((await call(`/similar/${me}?nsfw=include`)).body.items[0].permlink).toBe("cat-2");
      expect((await call(`/duplicates/${me}`)).body.items).toEqual([]);
      expect((await call(`/duplicates/${me}?nsfw=include`)).body.items.map((i: any) => i.permlink)).toEqual(["cat-2"]);
      const a = (await call("/ask", { method: "POST", body: JSON.stringify({ question: `duplicates of ${me}` }), headers: { "content-type": "application/json" } })).body;
      expect(a.intent).toBe("duplicate");
      expect(a.items).toEqual([]);
    } finally {
      await env.DB.prepare("UPDATE posts SET nsfw = 0 WHERE id = ?").bind(ids["alice/cat-2"]).run();
    }
  });

  it("/search-by-image goes through the same view pipeline as indexing", async () => {
    const { body } = await call("/search-by-image", { method: "POST", body: JSON.stringify({ image: base64Encode(fixture("second.webp")) }), headers: { "content-type": "application/json" } });
    expect(body.method).toBe("vector");
    expect(body.items[0].permlink).toBe("red-dragon");
    expect(body.duplicates[0].permlink).toBe("red-dragon");
    // the same as a multipart upload
    const fd = new FormData();
    fd.append("image", new Blob([new Uint8Array(fixture("second.webp"))], { type: "image/webp" }), "dragon.webp");
    const multi = await call("/search-by-image", { method: "POST", body: fd });
    expect(multi.status).toBe(200);
    expect(multi.body.items[0].permlink).toBe("red-dragon");
  });
});

describe("/ask", () => {
  const askQ = async (question: string) => (await call("/ask", { method: "POST", body: JSON.stringify({ question }), headers: { "content-type": "application/json" } })).body;

  it("who posted the first cat: the deleted original's author, with the repost noted", async () => {
    const r = await askQ("Who posted the first cat?");
    expect(r.answer).toBe("alice");
    expect(r.evidence[0].first_seen_post).toBe("/@alice/cat-1");
    expect(r.answer_text).toContain("2026-09-01");
  });

  it("a question that names an author or a period is about those posts, not the image's first appearance", async () => {
    const bob = await askQ("When did @bob first post a cat?");
    expect(bob.answer).toBe("2026-09-21"); // bob's own post, although the image dates from alice's
    const after = await askQ("Who posted the first cat after 2026-09-15?");
    expect(after.answer).toBe("bob");
    expect(after.answer_text).toContain("2026-09-21");
    expect(after.answer_text).toContain("/@alice/cat-1"); // the repost note still says where it came from
  });

  it("in French and German", async () => {
    expect((await askQ("Qui a posté le premier chat ?")).answer).toBe("alice");
    expect((await askQ("Wer hat die erste Katze gepostet?")).answer).toBe("alice");
  });

  it("counts, metadata questions and most liked", async () => {
    expect((await askQ("How many dragon artworks are there?")).answer).toBe(1);
    expect((await askQ("How many artworks did dave post?")).answer).toBe(2);
    expect((await askQ("What is the most liked cat artwork?")).answer).toBe("/@alice/cat-2");
    expect((await askQ("How many black and white artworks are there?")).answer).toBe(3); // noir and the grey cat (twice)
  });

  it("says so when nothing matches", async () => {
    const r = await askQ("Who posted the first submarine?");
    expect(r.answer).toBeNull();
    expect(r.evidence).toEqual([]);
  });
});

describe("/query: the single search box", () => {
  const q = async (text: string, extra = "") => (await call(`/query?q=${encodeURIComponent(text)}${extra}`)).body;

  it("words and titles are searches, even a title that starts like a question", async () => {
    const r = await q("red dragon");
    expect(r).toMatchObject({ route: "search", reason: "not a question" });
    expect(r.results.items[0].permlink).toBe("red-dragon");
    const t = await q("Why I draw cats");
    expect(t).toMatchObject({ route: "search", reason: "the title of a post" });
    expect(t.results.items.map((i: any) => i.permlink)).toContain("why-i-draw-cats");
  });

  it("questions about the artworks are answered, with the posts behind the answer", async () => {
    const r = await q("Who posted the first cat?");
    expect(r.route).toBe("ask");
    expect(r.answer.answer).toBe("alice");
    expect(r.answer.items[0]).toMatchObject({ author: "alice", type: "artwork" });
    expect(r.answer.items[0].artwork.images.original).toMatch(/^\/img\/orig\//);
    expect(r.results).toBeUndefined();
    const n = await q("how many artists posted cats?");
    expect(n).toMatchObject({ route: "ask", answer: { answer: 2, answer_text: "2 artists posted cat artworks." } });
    const none = await q("who posted the first unicorn?");
    expect(none.route).toBe("ask");
    expect(none.answer.answer).toBeNull();
    expect(Array.isArray(none.results.items)).toBe(true); // nothing verified: the box shows a search too
  });

  it("questions about the platform go to help; with no documentation yet the box still shows results", async () => {
    const r = await q("How do I mint an artwork?");
    expect(r.route).toBe("help");
    expect(r.answer.status).toBe("no_docs");
    expect(Array.isArray(r.results.items)).toBe(true);
  });

  it("route= forces a destination; a cursor is always the next page of a search", async () => {
    expect((await q("red dragon", "&route=ask")).route).toBe("ask");
    expect((await q("Who posted the first cat?", "&route=search")).route).toBe("search");
    const first = await q("cat", "&limit=1");
    expect(first.results.next_cursor).toBeTruthy();
    const next = await q("cat", `&limit=1&cursor=${encodeURIComponent(first.results.next_cursor)}`);
    expect(next).toMatchObject({ route: "search", reason: "next page" });
    expect(next.results.items[0].id).not.toBe(first.results.items[0].id);
  });

  it("when the client's answer budget is spent, questions get results instead", async () => {
    env.RL_HEAVY = { limit: async () => ({ success: false }) } as any;
    try {
      const r = await q("Who posted the first cat?");
      expect(r.route).toBe("search");
      expect(r.notes[0]).toContain("answer budget is spent");
      expect(r.results.items.length).toBeGreaterThan(0);
    } finally {
      env.RL_HEAVY = undefined;
    }
  });
});

describe("admin", () => {
  it("is token-protected", async () => {
    expect((await call("/admin/stats")).status).toBe(401);
    const s = await call("/admin/stats", { admin: true });
    expect(s.body.artworks).toMatchObject({ n: 4, embedded: 4, described: 4 });
  });

  it("the sweeper finds nothing to do on a complete index, and re-embeds when EMBED_VIEWS changes", async () => {
    expect((await call("/admin/sweep", { method: "POST", admin: true })).body.enqueued).toBe(0);
    env.EMBED_VIEWS = "nearest";
    const r = await call("/admin/sweep", { method: "POST", admin: true });
    expect(r.body.byStage.embed).toBe(4);
    env._queue.drain();
    env.EMBED_VIEWS = undefined;
    await env.DB.prepare("UPDATE jobs SET status = 'done'").run();
  });

  it("ranker weights can be read, set and reset", async () => {
    const set = await call("/admin/ranker/weights", { method: "POST", admin: true, body: JSON.stringify({ w: { title: 0.6 }, version: "t1" }) });
    expect(set.body.weights.w.title).toBe(0.6);
    expect((await call("/admin/ranker/weights", { admin: true })).body.active.version).toBe("t1");
    expect((await call("/admin/ranker/weights", { method: "POST", admin: true, body: JSON.stringify({ w: { title: 9 } }) })).status).toBe(400);
    await call("/admin/ranker/weights", { method: "POST", admin: true, body: JSON.stringify({ reset: true }) });
    expect((await call("/admin/ranker/weights", { admin: true })).body.active.version).toBe("default-1");
  });
});

describe("deletion and return", () => {
  it("a delete-by-edit leaves every index; editing it back restores them", async () => {
    const id = ids["dave/red-dragon"];
    const deleted = chainPost("dave", "red-dragon", T0 + 5 * DAY, "deleted", { title: "Red Dragon", tags: ["dragon"], updated: T0 + 30 * DAY });
    await ingestPost(env, deleted as any, null, "test");
    expect(env._vec.v.has(String(id))).toBe(false);
    expect(env._vecText.v.has(String(id))).toBe(false);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM phash_bands WHERE post_id = ?").bind(id).first("n")).toBe(0);
    for (const k of [...env._kv.m.keys()]) if (k.startsWith("s3:")) env._kv.m.delete(k); // the 60 s result cache
    const s = await call("/search?q=red+dragon+fire&semantic=0");
    expect(s.body.items.some((i: any) => i.id === id)).toBe(false);
    expect((await call("/search?q=dargon")).body.did_you_mean).toBeUndefined(); // the word left the vocabulary

    // edited back to the image
    const back = chainPost("dave", "red-dragon", T0 + 5 * DAY, `data:image/webp;base64,${base64Encode(fixture("second.webp"))}`, { title: "Red Dragon", tags: ["dragon"], updated: T0 + 31 * DAY });
    const r = await ingestPost(env, back as any, null, "test");
    expect(r.stages).toEqual(expect.arrayContaining(["stats", "embed", "concepts", "text"]));
    await handleEnrichBatch({ queue: "q", messages: env._queue.drain().map((body) => ({ id: "x", timestamp: new Date(), attempts: 1, body, ack() {}, retry() {} })), ackAll() {}, retryAll() {} } as any, env);
    expect(env._vec.v.has(String(id))).toBe(true);
    expect(env._vecText.v.has(String(id))).toBe(true);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM phash_bands WHERE post_id = ?").bind(id).first("n")).toBe(4);
    expect(await env.DB.prepare("SELECT COUNT(*) AS n FROM artwork_concepts WHERE post_id = ? AND concept = 'dragon'").bind(id).first("n")).toBe(1);
  });
});

describe("vocabulary", () => {
  it("incremental maintenance (inserts, AI captions, deletion, return) equals a full rebuild", async () => {
    const snap = async () =>
      Object.fromEntries(((await env.DB.prepare("SELECT term, df FROM vocab WHERE df > 0 ORDER BY term").all<{ term: string; df: number }>()).results ?? []).map((r) => [r.term, r.df]));
    const incremental = await snap();
    const r = await call("/admin/vocab/rebuild", { method: "POST", admin: true });
    expect(r.body.docs).toBe(5);
    expect(await snap()).toEqual(incremental);
    expect(incremental.dragon).toBe(1);
  });
});

describe("VLM replies the AI binding leaves unparsed (what Moondream's replies looked like in production)", () => {
  // The binding parses a reply only when its content-type is exactly application/json; otherwise
  // run() returns the body stream, which looked like {} and failed every description.
  const enrich = async (msg: EnrichMessage) => {
    const acks: string[] = [];
    const m = { id: "1", timestamp: new Date(), attempts: 1, body: msg, ack: () => acks.push("ack"), retry: () => acks.push("retry") };
    await handleEnrichBatch({ queue: "q", messages: [m], ackAll() {}, retryAll() {} } as any, env);
    return acks;
  };
  const describeJob = async (id: number) => env.DB.prepare("SELECT status, attempts, error FROM jobs WHERE post_id = ? AND stage = 'describe'").bind(id).first<any>();

  it("a stream reply is read; an empty description is recorded but not retried through the queue", async () => {
    const id = ids["dave/red-dragon"];
    const msg: EnrichMessage = { postId: id, author: "dave", permlink: "red-dragon", stages: ["describe"], force: true, reason: "test" };
    const original = env._ai.handler;
    const before = await env.DB.prepare("SELECT ai_caption FROM artworks WHERE post_id = ?").bind(id).first<{ ai_caption: string }>();
    try {
      env._ai.handler = (_m, input) =>
        new Response(JSON.stringify(input.task === "query" ? { answer: JSON.stringify({ caption: "A red dragon guarding a pile of gold coins.", tags: ["dragon", "gold"] }), finish_reason: "stop" } : { caption: "unused" }), {
          headers: { "content-type": "application/json; charset=utf-8" },
        }).body;
      expect(await enrich(msg)).toEqual(["ack"]);
      expect(await env.DB.prepare("SELECT ai_caption, ai_status FROM artworks WHERE post_id = ?").bind(id).first<any>()).toEqual({ ai_caption: "A red dragon guarding a pile of gold coins.", ai_status: "ok" });
      expect((await describeJob(id)).status).toBe("done");

      env._ai.handler = () => new Response("").body; // a reply with nothing in it
      expect(await enrich(msg)).toEqual(["ack"]); // the sweeper redoes it later, not the queue now
      expect(await describeJob(id)).toMatchObject({ status: "failed", attempts: 1, error: "empty reply" });
      env._ai.handler = () => ({ answer: null, caption: "" }); // parsed, but nothing usable
      expect(await enrich(msg)).toEqual(["ack"]);
      const j = await describeJob(id);
      expect(j).toMatchObject({ status: "failed", attempts: 2 });
      expect(j.error).toContain("moondream returned no usable caption");
      expect(env._ai.calls.filter((c) => c.model.includes("moondream")).length).toBeGreaterThan(0);
    } finally {
      env._ai.handler = original;
      await enrich(msg);
    }
    expect((await env.DB.prepare("SELECT ai_caption FROM artworks WHERE post_id = ?").bind(id).first<{ ai_caption: string }>())!.ai_caption).toBe(before!.ai_caption);
  });
});
