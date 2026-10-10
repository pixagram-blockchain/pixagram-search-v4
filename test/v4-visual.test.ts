// Questions about an uploaded image (spec §46-47) through the real pipeline: chain posts ingested
// and enriched (decode, pHash, dHash, colours, a stub embedding), then POST /ask with an image.
// Exact identity (the same bytes), perceptual identity (a rescaled copy: same shapes and colours),
// and an image that only looks similar are told apart; the first appearance is the index's.

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { codecsReady } from "./helpers";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, type ChainFixture } from "./harness/net";
import { base64Encode } from "../src/lib/bytes";
import { encodePng, type RgbaImage } from "../src/enrich/decode";
import { handleEnrichBatch } from "../src/enrich/consumer";
import { ingestPost } from "../src/chain/ingest";
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
/** A 16x16 sprite: a red heart-ish blob on dark blue. */
const sprite = (scale = 1) =>
  paint(16 * scale, 16 * scale, (x, y) => {
    const X = Math.floor(x / scale);
    const Y = Math.floor(y / scale);
    const d = (X - 7.5) ** 2 + (Y - 8) ** 2;
    return d < 20 || (Y < 6 && (Math.abs(X - 4) < 3 || Math.abs(X - 11) < 3)) ? [220, 30, 40, 255] : [20, 30, 90, 255];
  });
const stripes = () => paint(16, 16, (x) => (x % 4 < 2 ? [240, 230, 40, 255] : [30, 160, 60, 255]));

const uri = (b: Uint8Array) => `data:image/png;base64,${base64Encode(b)}`;
function chainPost(author: string, permlink: string, created: number, body: string, title: string) {
  return {
    author, permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title, body,
    json_metadata: JSON.stringify({ app: "pixagram/3.0.2", format: "image", tags: ["heart"] }),
    created: iso(created), last_update: iso(created), depth: 0, children: 0, net_votes: 1,
    pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
  };
}

async function drain(env: TestEnv, msgs: EnrichMessage[] = env._queue.drain()) {
  await handleEnrichBatch({ queue: "q", messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack() {}, retry() {} })), ackAll() {}, retryAll() {} } as any, env);
}

async function call(env: TestEnv, path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const exec = new FakeExec();
  const res = await app.fetch(new Request(`https://search.test${path}`, init), env, exec as unknown as ExecutionContext);
  await exec.settle();
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

const askImage = (env: TestEnv, question: string, png: Uint8Array, extra: object = {}) =>
  call(env, "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question, image: base64Encode(png), ...extra }) });

let heart: Uint8Array;
let heartBig: Uint8Array;
let other: Uint8Array;

async function gallery(): Promise<TestEnv> {
  const posts = [
    chainPost("alice", "heart", T0, uri(heart), "Heart"),
    chainPost("bob", "my-heart", T0 + 3 * DAY, uri(heart), "My heart"), // the same bytes, later
    chainPost("carol", "field", T0 + DAY, uri(other), "Field"),
  ];
  const chain: ChainFixture = { posts: posts as any[] };
  const env = makeEnv({ HF_EMBED_URL: "https://embed.test/embed", EMBED_MODEL: "stub", EMBED_DIM: String(DIM), VLM_BACKEND: "off", PLANNER_BACKEND: "rules" } as any);
  installFetch({ rpcUrl: "https://rpc.test", embedUrl: "https://embed.test", chain, embed: (kind, input) => seededVector(`${kind}:${input.slice(-64)}`, DIM) });
  for (const p of posts) await ingestPost(env, p as any, null, "test");
  await drain(env);
  return env;
}

beforeAll(async () => {
  await codecsReady();
  heart = await encodePng(sprite());
  heartBig = await encodePng(sprite(3)); // the same picture at 3x: other bytes, same pHash and colours
  other = await encodePng(stripes());
});
afterEach(() => vi.unstubAllGlobals());

describe("questions about an uploaded image", () => {
  it("the same bytes: exact identity, and the first post that showed them", async () => {
    const env = await gallery();
    const r = await askImage(env, "Who posted this first?", heart);
    expect(r.status).toBe(200);
    expect(r.body).toMatchObject({ status: "answered", class: "VISUAL", answer: "alice", answer_type: "author" });
    expect(r.body.answer_text).toBe("This image was first posted by @alice on 2026-09-01, in “Heart”. It appears in 1 other post too.");
    expect(r.body.image).toMatchObject({ task: "origin", identity: "exact_identity", first_seen: { author: "alice", post: "/@alice/heart", match: "exact" } });
    const i1 = r.body.cards.find((c: any) => c.evidence_id === "I1");
    expect(i1).toMatchObject({ type: "query_image", width: 16, height: 16 });
    expect(i1.sha256).toMatch(/^[0-9a-f]{64}$/);
    const bob = r.body.cards.find((c: any) => c.path === "/@bob/my-heart");
    expect(bob.identity).toMatchObject({ exact: true, historical: "same_origin" });
    expect(env._ai.calls.filter((c) => !c.model.includes("embed"))).toEqual([]); // no model: identity is computed
  });

  it("a rescaled copy: perceptual identity (same shapes, same colours), never mistaken for the same bytes", async () => {
    const env = await gallery();
    const r = await askImage(env, "Who made this?", heartBig);
    expect(r.body.image.identity).toBe("perceptual_identity");
    expect(r.body.answer).toBe("alice");
    expect(r.body.answer_text.startsWith("A near-identical version of this image (same shapes and colours) was first posted by @alice on 2026-09-01")).toBe(true);
    const m = r.body.image.matches.find((x: any) => x.path === "/@alice/heart");
    expect(m.identity.exact).toBe(false);
    expect(m.identity.perceptual.near_identical).toBe(true);
    expect(m.identity.perceptual.phash_distance).toBeLessThanOrEqual(4);
  });

  it("an image no artwork shows: not on Pixagram, whatever looks similar", async () => {
    const env = await gallery();
    const novel = await encodePng(paint(16, 16, (x, y) => ((x * y) % 7 < 3 ? [250, 250, 250, 255] : [0, 0, 0, 255])));
    const yes = await askImage(env, "Is this on Pixagram?", heart);
    expect(yes.body).toMatchObject({ answer: true, answer_type: "boolean" });
    expect(yes.body.answer_text.startsWith("Yes: this image is on Pixagram. This image was first posted by @alice")).toBe(true);
    const no = await askImage(env, "Is this on Pixagram?", novel);
    expect(no.body).toMatchObject({ answer: false, image: { identity: expect.not.stringMatching(/identity$/) } });
    expect(no.body.answer_text.startsWith("No: no indexed artwork shows this image")).toBe(true);
    const who = await askImage(env, "Who posted this first?", novel);
    expect(who.body).toMatchObject({ status: "no_match", answer: null });
    // look-alikes are evidence of similarity only
    for (const c of who.body.cards.filter((x: any) => x.identity)) expect(c.identity.exact || c.identity.perceptual?.near_identical).toBeFalsy();
  });

  it("similar artworks: the identical ones first, flagged; the rest only similar", async () => {
    const env = await gallery();
    const r = await askImage(env, "Find artworks similar to this one", heart);
    expect(r.body.image.task).toBe("similar");
    expect(r.body.answer_type).toBe("count");
    expect(r.body.image.matches[0].identity.exact).toBe(true);
  });

  it("multipart uploads work too; mode=v3 has no image questions; size limits hold", async () => {
    const env = await gallery();
    const form = new FormData();
    form.set("question", "Who posted this first?");
    form.set("image", new Blob([heart as Uint8Array<ArrayBuffer>], { type: "image/png" }), "heart.png");
    const m = await call(env, "/ask", { method: "POST", body: form });
    expect(m.status).toBe(200);
    expect(m.body.answer).toBe("alice");
    expect((await askImage(env, "Who posted this first?", heart, { mode: "v3" })).status).toBe(400);
    const big = await encodePng(paint(1200, 1000, () => [1, 2, 3, 255]));
    expect((await askImage(env, "Who posted this first?", big)).status).toBe(413);
    expect((await askImage(env, "", heart)).status).toBe(400);
  });
});

describe("v4.8: a rich answer about an uploaded image", () => {
  it("carries the facts of the posts that show the image, caveats about similarity, and follow-ups about them; the model is not needed", async () => {
    const env = await gallery();
    const r = await askImage(env, "Who posted this first?", heart, { style: "rich", mode: "fast" });
    expect(r.status).toBe(200);
    expect(r.body.style).toBe("rich");
    expect(r.body.answer_text).toBe("This image was first posted by @alice on 2026-09-01, in “Heart”. It appears in 1 other post too.");
    expect(r.body.sections.facts[0]).toMatch(/^“Heart” by @alice: posted on 2026-09-01; tagged heart/);
    expect(r.body.sections.facts.some((f: string) => f.startsWith("“My heart” by @bob"))).toBe(true);
    expect(r.body.answer_full).toContain("From the index:");
    expect(r.body.suggestions.follow_ups.map((f: any) => f.text)).toEqual(expect.arrayContaining(["Was “Heart” edited?", "Was “Heart” reposted?"]));
    expect(r.body.elaboration).toEqual({ status: "none", reason: "fast mode: no model" });
    expect(env._ai.calls.filter((c) => !c.model.includes("embed"))).toEqual([]);
  });
});
