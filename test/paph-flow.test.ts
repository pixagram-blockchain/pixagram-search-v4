// Copy detection across shards and through the Worker: range sharding and parallel checks, the
// paph stage and its verdicts in D1, the API with its day-long cache, and the whole pipeline from
// the chain (ingest → enrichment → /copies → deletion). D1 is SQLite with the real migrations, the
// PAPH shards are real ShardStores behind stubs that clone like RPC (test/harness/paph.ts).

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { codecsReady, fixture, paphReady } from "./helpers";
import { fakePaph, putAs112, type FakePaph } from "./harness/paph";
import { makeEnv, FakeExec, type TestEnv } from "./harness/fakes";
import { installFetch } from "./harness/net";
import { art, crop, mirror, paste, swapRedBlue, upscale } from "./harness/images";
import { decodeImage, encodePng, type RgbaImage } from "../src/enrich/decode";
import { base64Encode } from "../src/lib/bytes";
import { STAGE_BUDGET } from "../src/paph/budget";
import { STATE, fingerprint, type Fingerprint, type PaphRuntime } from "../src/paph/engine";
import { findEverywhere, homeShard, resetShardCount, shardOf } from "../src/paph/shards";
import { copiesOf, copiesOfImage, currentEngine, earlier, liveCopiesOf, pairReport, paphGc, paphGcPass, paphHealWire3, paphStage, paphStaleVerdicts, paphStatus, paphVectorPass, replaceMatches, resetLegacyStore, storeMayHoldLegacy, STAGE_GRACE_MS, STAGE_MIN_MS } from "../src/paph/copies";
import type { Checked, Match } from "../src/paph/shard-store";
import { sweep } from "../src/enrich/sweeper";
import { app } from "../src/api";
import { ingestPost } from "../src/chain/ingest";
import { RUN_MS, handleEnrichBatch } from "../src/enrich/consumer";
import { removeFromIndexes } from "../src/db/posts";
import type { EnrichMessage } from "../src/env";

let rt: PaphRuntime;
let small: RgbaImage;
let second: RgbaImage;
let fpSmall: Fingerprint;

beforeAll(async () => {
  await codecsReady();
  rt = await paphReady();
  small = await decodeImage(fixture("small.webp"));
  second = await decodeImage(fixture("second.webp"));
  fpSmall = await fingerprint(small);
});

afterAll(() => vi.unstubAllGlobals());

const wires = (f: Fingerprint) => ({ t1: f.t1, t2: f.t2 });
const TOKEN = "admin-secret";
/** the identity this Worker's verdicts carry (policy "safe") */
const ID = () => `${rt.identity.id} safe`;

/** D1 with the migrations, copy detection on 1000-id shards, and helpers to add posts. */
function world(over: Record<string, string> = {}) {
  resetShardCount();
  const paph = fakePaph(rt);
  const env = makeEnv({ PAPH: paph.ns, PAPH_SHARD_SIZE: "1000", ADMIN_TOKEN: TOKEN, ...over } as any);
  const db = env._db.raw;
  const post = (id: number, author: string, created: number, hash: string, o: { nsfw?: boolean; phash?: string } = {}) => {
    db.prepare("INSERT INTO posts (id, author, permlink, type, title, created, updated, indexed_at, nsfw) VALUES (?, ?, ?, 'artwork', ?, ?, ?, ?, ?)").run(id, author, `p${id}`, `work ${id}`, created, created, created, o.nsfw ? 1 : 0);
    db.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, updated, phash) VALUES (?, ?, 'image/webp', 1, ?, ?)").run(id, hash, created, o.phash ?? null);
  };
  const run = (id: number, hash: string, im: RgbaImage | (() => RgbaImage), force = false, o: { endsAt?: number } = {}) => {
    const row = db.prepare("SELECT paph_hash, paph_engine, phash FROM artworks WHERE post_id = ?").get(id) as any;
    return paphStage(env, { postId: id, hash, phash: row?.phash ?? null, paphHash: row?.paph_hash ?? null, paphEngine: row?.paph_engine ?? null, force }, async () => (typeof im === "function" ? im() : im), o);
  };
  const pairs = () => (db.prepare("SELECT a, b, verdict FROM paph_matches ORDER BY a, b").all() as any[]).map((r) => `${r.a}-${r.b}:${r.verdict}`);
  return { env, db, paph, post, run, pairs };
}

async function call(env: TestEnv, path: string, init?: RequestInit & { admin?: boolean }): Promise<{ status: number; body: any }> {
  const exec = new FakeExec();
  const headers = new Headers(init?.headers);
  if (init?.admin) headers.set("authorization", `Bearer ${TOKEN}`);
  const res = await app.fetch(new Request(`https://search.test${path}`, { ...init, headers }), env, exec as unknown as ExecutionContext);
  await exec.settle();
  const text = await res.text();
  return { status: res.status, body: text ? JSON.parse(text) : null };
}

const finds = (p: FakePaph) => p.calls.filter((c) => c.endsWith(".find")).length;

describe("shards", () => {
  it("hold works by post-id range and answer a check together", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "dave", 150, "h1500");
    w.post(2500, "erin", 160, "h2500");
    expect([shardOf(w.env, 1), shardOf(w.env, 1500), shardOf(w.env, 2500)]).toEqual([0, 1, 2]);
    await homeShard(w.env, 1).put({ postId: 1, contentHash: "h1", ...wires(fpSmall) });
    await homeShard(w.env, 1500).put({ postId: 1500, contentHash: "h1500", ...wires(await fingerprint(second)) });
    await homeShard(w.env, 2500).put({ postId: 2500, contentHash: "h2500", ...wires(await fingerprint(swapRedBlue(small))) });
    expect([...w.paph.shards.keys()].sort()).toEqual(["paph:1000:0", "paph:1000:1", "paph:1000:2"]);
    const r = await findEverywhere(w.env, wires(await fingerprint(mirror(small))), { budget: STAGE_BUDGET, identity: ID() });
    expect(r.shards).toMatchObject({ asked: 3, answered: 3, failed: [] });
    expect(r.matches.filter((m) => m.state >= STATE.Copy).map((m) => m.id).sort((a, b) => a - b)).toEqual([1, 2500]);
    expect(r.partial).toBe(false);
    expect(r.identity).toBe(`${rt.identity.id} safe`);
  });

  it("send each outside candidate to its own shard only", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(2500, "erin", 160, "h2500");
    await homeShard(w.env, 1).put({ postId: 1, contentHash: "h1", ...wires(fpSmall) });
    await homeShard(w.env, 2500).put({ postId: 2500, contentHash: "h2500", ...wires(await fingerprint(swapRedBlue(small))) });
    w.paph.calls.length = 0;
    const r = await findEverywhere(w.env, wires(fpSmall), { budget: STAGE_BUDGET, identity: ID(), nominate: false, extraShardsOnly: true, exclude: [1], extra: [{ id: 2500, via: "vector" }] });
    expect(w.paph.calls).toEqual(["paph:1000:2.find"]);
    expect(r.checked).toEqual([{ id: 2500, contentHash: "h2500" }]);
    expect(r.matches[0]).toMatchObject({ id: 2500, via: ["vector"] });
  });

  it("report a shard that fails or does not answer in time, and keep the others' answers", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "dave", 150, "h1500");
    await homeShard(w.env, 1).put({ postId: 1, contentHash: "h1", ...wires(fpSmall) });
    await homeShard(w.env, 1500).put({ postId: 1500, contentHash: "h1500", ...wires(await fingerprint(mirror(small))) });
    w.paph.failing.add("paph:1000:0");
    const failed = await findEverywhere(w.env, wires(fpSmall), { budget: STAGE_BUDGET, identity: ID() });
    expect(failed.partial).toBe(true);
    expect(failed.shards.failed).toEqual([{ shard: 0, error: "shard paph:1000:0 unavailable", timedOut: false }]);
    expect(failed.matches.map((m) => m.id)).toEqual([1500]);
    w.paph.failing.clear();
    w.paph.delays.set("paph:1000:1", 400);
    const slow = await findEverywhere(w.env, wires(fpSmall), { budget: { ...STAGE_BUDGET, deadlineMs: 100 }, identity: ID(), graceMs: 50 });
    expect(slow.partial).toBe(true);
    expect(slow.shards.failed).toMatchObject([{ shard: 1, timedOut: true }]);
    expect(slow.matches.map((m) => m.id)).toEqual([1]);
    w.paph.delays.clear();
    // a shard still on the previous release (a deploy under way): its answer is not used
    w.paph.engines.set("paph:1000:1", "paph-x/1.0.9 previous safe");
    const mixed = await findEverywhere(w.env, wires(fpSmall), { budget: STAGE_BUDGET, identity: ID() });
    expect(mixed).toMatchObject({ partial: true, identity: ID(), shards: { asked: 2, answered: 1, complete: [0] } });
    expect(mixed.shards.failed).toEqual([{ shard: 1, error: "shard 1 runs another engine", timedOut: false, engine: "paph-x/1.0.9 previous safe" }]);
    expect(mixed.matches.map((m) => m.id)).toEqual([1]);
    expect(mixed.checked.map((c) => c.id)).toEqual([1]);
  });
});

describe("verdicts in D1", () => {
  const identity = `${"x".repeat(10)} safe`;
  const copy = (id: number, extra: Partial<Match> = {}): Match => ({
    id, contentHash: `h${id}`, verdict: "Copy", state: 3, execution: "FAST", certifiable: true, certificate: false,
    structuralLo: 6500, structuralHi: 7000, geometry: 10000, inliers: 100, mirrored: false, swapped: false,
    route: { local: 10, band: 5, global: 200, class: "FAST" }, via: ["codes"], ...extra,
  });
  const chk = (...ids: number[]): Checked[] => ids.map((id) => ({ id, contentHash: `h${id}` }));
  function seed() {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 300, "h2");
    w.post(3, "other", 200, "h3");
    return w;
  }
  const mark = (db: any, id: number) => db.prepare("SELECT paph_hash, paph_engine FROM artworks WHERE post_id = ?").get(id);

  it("stores a pair as (earlier, later) about both images, with its identity, and marks the stage complete", async () => {
    const { env, db, pairs } = seed();
    expect(await replaceMatches(env.DB, 2, "h2", { matches: [copy(1, { mirrored: true, via: ["codes", "si"] })], checked: chk(1, 3), identity }, identity)).toBe(1);
    expect(pairs()).toEqual(["1-2:Copy"]);
    expect(db.prepare("SELECT a_hash, b_hash, engine, same_author, mirrored, via, structural_lo, structural_hi, execution, rescued FROM paph_matches").get()).toMatchObject({
      a_hash: "h1", b_hash: "h2", engine: identity, same_author: 0, mirrored: 1, via: "codes,si", structural_lo: 6500, structural_hi: 7000, execution: "FAST",
      rescued: 0, // the first release's column, kept
    });
    expect(mark(db, 2)).toMatchObject({ paph_hash: "h2", paph_engine: identity });
  });

  it("replaces only the pairs a check compared — a concurrent check's verdict survives", async () => {
    const { env, pairs } = seed();
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1)], checked: chk(1), identity }, identity);
    // the original's own check ran before 2 was indexed: it compared only 3
    await replaceMatches(env.DB, 1, "h1", { matches: [], checked: chk(3), identity }, identity);
    expect(pairs()).toEqual(["1-2:Copy"]);
    // a later check of 1 that compares 2 and no longer finds a copy withdraws the verdict
    await replaceMatches(env.DB, 1, "h1", { matches: [], checked: chk(2, 3), identity }, identity);
    expect(pairs()).toEqual([]);
  });

  it("lets no check that raced an edit erase what a newer check established", async () => {
    const { env, db, pairs } = seed();
    // 2 was edited (h2 → h2b) and its new image's check stored the pair
    db.prepare("UPDATE artworks SET content_hash = 'h2b' WHERE post_id = 2").run();
    await replaceMatches(env.DB, 2, "h2b", { matches: [copy(1)], checked: chk(1), identity }, identity);
    expect(pairs()).toEqual(["1-2:Copy"]);
    // the check of 2's old image finishes late: it compared 1 and found nothing — it erases nothing
    await replaceMatches(env.DB, 2, "h2", { matches: [], checked: chk(1), identity }, null);
    // nor does a check of 1 that compared 2's old wires
    await replaceMatches(env.DB, 1, "h1", { matches: [], checked: [{ id: 2, contentHash: "h2" }], identity }, null);
    expect(pairs()).toEqual(["1-2:Copy"]);
    expect(mark(db, 2)).toMatchObject({ paph_hash: "h2b" });
    // a check of 1 that compared 2's current image does replace it
    await replaceMatches(env.DB, 1, "h1", { matches: [], checked: [{ id: 2, contentHash: "h2b" }], identity }, null);
    expect(pairs()).toEqual([]);
  });

  it("never writes a verdict about an image that is gone, or for a post that is gone or no artwork", async () => {
    const { env, db, pairs } = seed();
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1, { contentHash: "h1-before" })], checked: chk(1), identity }, identity);
    expect(pairs()).toEqual([]);
    db.prepare("UPDATE posts SET deleted = 1 WHERE id = 3").run();
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(3)], checked: chk(3), identity }, identity);
    expect(pairs()).toEqual([]);
    db.prepare("UPDATE posts SET type = 'blog' WHERE id = 1").run(); // edited into a text post meanwhile
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1)], checked: chk(1), identity }, identity);
    expect(pairs()).toEqual([]);
    db.prepare("UPDATE posts SET type = 'artwork' WHERE id = 1").run();
    db.prepare("UPDATE artworks SET paph_hash = NULL, paph_engine = NULL").run();
    // this work's own image moved on while it was being checked: nothing written, no mark
    await replaceMatches(env.DB, 2, "h2-before", { matches: [copy(1)], checked: chk(1), identity }, identity);
    expect(pairs()).toEqual([]);
    expect(mark(db, 2)).toMatchObject({ paph_hash: null });
    // deleted while it was being checked: no mark either
    db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2").run();
    await replaceMatches(env.DB, 2, "h2", { matches: [], checked: chk(1), identity }, identity);
    expect(mark(db, 2)).toMatchObject({ paph_hash: null });
    db.prepare("UPDATE posts SET deleted = 0 WHERE id = 2").run();
    // a pass that is not the stage (the vector pass) never marks it complete
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1)], checked: chk(1), identity }, null);
    expect(pairs()).toEqual(["1-2:Copy"]);
    expect(mark(db, 2)).toMatchObject({ paph_hash: null });
  });

  it("lists copies from both sides with the listed work's relation, and counts what it hides", async () => {
    const { env, db, post } = seed();
    post(4, "nsfw-copier", 400, "h4", { nsfw: true });
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1)], checked: chk(1), identity }, identity);
    await replaceMatches(env.DB, 4, "h4", { matches: [copy(1, { verdict: "Suspected", state: 2 })], checked: chk(1), identity }, identity);
    const fromOriginal = await copiesOf(env, 1, {});
    expect(fromOriginal.items.map((it) => [it.id, it.copy.relation, it.copy.same_author, it.copy.engine])).toEqual([[2, "later", false, identity]]);
    const fromCopy = await copiesOf(env, 2, {});
    expect(fromCopy.items.map((it) => [it.id, it.copy.relation])).toEqual([[1, "earlier"]]);
    // marked under another identity than the engine's: not "indexed" for the current one
    expect(fromCopy.indexed).toBe(false);
    expect(fromOriginal.indexed).toBe(false);
    // Suspected only when asked for; the nsfw post is hidden by the default filter, and counted;
    // nothing below Suspected is a threshold
    const all = await copiesOf(env, 1, { min: "suspected" });
    expect(all.items.map((it) => it.id)).toEqual([2]);
    expect(all.hidden).toBe(1);
    expect((await copiesOf(env, 1, { min: "unrelated" })).min).toBe("Suspected");
    expect((await copiesOf(env, 1, { min: "identical" })).items).toEqual([]);
    // a pair whose image changed since is about an image no longer shown: hidden, and counted
    db.prepare("UPDATE artworks SET content_hash = 'h2-edited' WHERE post_id = 2").run();
    const edited = await copiesOf(env, 1, {});
    expect(edited.items).toEqual([]);
    expect(edited.hidden).toBe(1);
  });

  it("binds only the compared works that have a stored verdict, however many it compared", async () => {
    const { env, pairs } = seed();
    const many = Array.from({ length: 5000 }, (_, i) => ({ id: 100_000 + i, contentHash: "h".repeat(64) }));
    const deletes = () => env._db.queries.filter((q) => q.startsWith("DELETE FROM paph_matches")).length;
    await replaceMatches(env.DB, 2, "h2", { matches: [], checked: many, identity }, null);
    expect(deletes()).toBe(0);
    await replaceMatches(env.DB, 2, "h2", { matches: [copy(1)], checked: chk(1), identity }, null);
    expect(pairs()).toEqual(["1-2:Copy"]);
    await replaceMatches(env.DB, 2, "h2", { matches: [], checked: [...many, ...chk(1)], identity }, null);
    expect(deletes()).toBe(1);
    expect(pairs()).toEqual([]);
  });

  it("counts as hidden only what it passed over, not what the limit left out", async () => {
    const { env, post } = seed();
    for (let i = 0; i < 30; i++) {
      post(100 + i, `copier${i}`, 500 + i, `h${100 + i}`);
      await replaceMatches(env.DB, 100 + i, `h${100 + i}`, { matches: [copy(1)], checked: chk(1), identity }, null);
    }
    const r = await copiesOf(env, 1, { limit: 5 });
    expect(r.items.length).toBe(5);
    expect(r.hidden).toBe(0);
  });

  it("orders a pair by chain time, then id", () => {
    expect(earlier({ id: 5, created: 100, author: "a" }, { id: 2, created: 200, author: "b" })).toBe(true);
    expect(earlier({ id: 5, created: 200, author: "a" }, { id: 2, created: 100, author: "b" })).toBe(false);
    expect(earlier({ id: 2, created: 100, author: "a" }, { id: 5, created: 100, author: "b" })).toBe(true);
  });
});

describe("the paph stage", () => {
  it("indexes works in arrival order across shards, finds each pair once, and skips an unchanged image", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1003, "other", 150, "h1003");
    w.post(2002, "copycat", 200, "h2002");
    expect((await w.run(1, "h1", small)).status).toBe("done");
    expect((await w.run(1003, "h1003", second)).status).toBe("done");
    const r = await w.run(2002, "h2002", mirror(small));
    expect(r).toMatchObject({ status: "done", complete: true, incomplete: [], stored: 1, reused: false });
    expect(w.pairs()).toEqual(["1-2002:Copy"]);
    const engine = await currentEngine(w.env);
    expect(w.db.prepare("SELECT paph_hash, paph_engine FROM artworks ORDER BY post_id").all()).toEqual([
      { paph_hash: "h1", paph_engine: engine }, { paph_hash: "h1003", paph_engine: engine }, { paph_hash: "h2002", paph_engine: engine },
    ]);
    expect((await w.run(2002, "h2002", mirror(small))).status).toBe("unchanged");
  });

  it("re-checks a work from its stored wires: no image fetched, nothing hashed", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    await w.run(2, "h2", swapRedBlue(small));
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    const noImage = (): RgbaImage => {
      throw new Error("the image was fetched");
    };
    const forced = await w.run(2, "h2", noImage, true);
    expect(forced).toMatchObject({ status: "done", reused: true, fit: null });
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    // keys and signatures of another derivation are re-derived by the same path
    w.paph.shard("paph:1000:0").db.prepare("UPDATE works SET derivation = 'paph-x/0.9 old'").run();
    expect((await w.run(1, "h1", noImage)).status).toBe("done");
  });

  it("re-checks every work after a release or a policy change, without force; the purge waits for it", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    // every other stage is done: what the sweeper finds is copy detection's alone
    w.db.prepare("UPDATE artworks SET stats_hash = content_hash, features_hash = content_hash, concepts_hash = 'c'").run();
    await w.run(1, "h1", small);
    await w.run(2, "h2", mirror(small));
    const before = await currentEngine(w.env);
    expect(w.db.prepare("SELECT engine FROM paph_matches").get()).toEqual({ engine: before });
    // the policy changes (as a release changes the engine's identity)
    w.env.PAPH_POLICY = "exact";
    const after = await currentEngine(w.env);
    expect(after).not.toBe(before);
    // the sweeper sees both works as not checked under the current identity
    const swept = await sweep(w.env, 10);
    expect(swept.byStage.paph).toBe(2);
    expect(w.env._queue.drain().map((m) => [m.postId, m.stages])).toEqual(expect.arrayContaining([[1, ["paph"]], [2, ["paph"]]]));
    // purging the old verdicts now would lose them: refused
    expect(await paphStaleVerdicts(w.env, { purge: true })).toMatchObject({ stale: 1, awaiting_recheck: 2, purged: 0, refused: expect.any(String) });
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    // the re-check (not forced) does run, from the stored wires, and replaces the verdict under the new identity
    const noImage = (): RgbaImage => {
      throw new Error("the image was fetched");
    };
    expect(await w.run(2, "h2", noImage)).toMatchObject({ status: "done", complete: true, reused: true });
    expect(await w.run(1, "h1", noImage)).toMatchObject({ status: "done", complete: true, reused: true });
    expect(w.db.prepare("SELECT engine FROM paph_matches").all()).toEqual([{ engine: after }]);
    expect(await paphStaleVerdicts(w.env, { purge: true })).toMatchObject({ stale: 0, awaiting_recheck: 0, purged: 0 });
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    expect((await w.run(2, "h2", noImage)).status).toBe("unchanged");
  });

  it("withdraws a changed image's verdicts and finds what the new image copies", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(3, "other", 150, "h3");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    await w.run(3, "h3", second);
    await w.run(2, "h2", mirror(small));
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    w.db.prepare("UPDATE artworks SET content_hash = 'h2b' WHERE post_id = 2").run();
    expect((await w.run(2, "h2b", mirror(second))).status).toBe("done");
    expect(w.pairs()).toEqual(["3-2:Copy"]);
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2b");
  });

  it("stands down when a newer image of the same post has overtaken the run", async () => {
    const w = world();
    w.post(1, "original", 100, "h1-newer");
    expect((await w.run(1, "h1", small)).status).toBe("superseded");
    expect(await homeShard(w.env, 1).info(1)).toBeNull();
  });

  it("leaves no trace of a post deleted while it was being checked", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    w.post(3, "late-copycat", 300, "h3");
    await w.run(1, "h1", small);
    // deleted while its image was being hashed
    const deleting = (): RgbaImage => {
      w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2").run();
      return mirror(small);
    };
    expect((await w.run(2, "h2", deleting)).status).toBe("gone");
    expect(await homeShard(w.env, 2).info(2)).toBeNull();
    // deleted just after its shard stored it (the deletion's own removal ran before that)
    w.paph.after.push((_shard, method) => {
      if (method === "putIf") w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 3").run();
    });
    expect((await w.run(3, "h3", mirror(small))).status).toBe("gone");
    expect(await homeShard(w.env, 3).info(3)).toBeNull();
    expect(w.pairs()).toEqual([]);
    expect(w.db.prepare("SELECT post_id, paph_hash FROM artworks WHERE post_id IN (2, 3) ORDER BY post_id").all()).toEqual([
      { post_id: 2, paph_hash: null }, { post_id: 3, paph_hash: null },
    ]);
  });

  it("writes what the answering shards found but leaves the stage incomplete when one did not answer", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1001, "copycat", 200, "h1001");
    w.post(2001, "recolorist", 300, "h2001");
    await w.run(1, "h1", small);
    await w.run(1001, "h1001", mirror(small));
    w.paph.failing.add("paph:1000:0");
    const r = await w.run(2001, "h2001", swapRedBlue(small));
    expect(r).toMatchObject({ status: "done", complete: false, incomplete: ["shards 0 (failed)"] });
    expect(w.pairs()).toEqual(["1-1001:Copy", "1001-2001:Copy"]);
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2001").get() as any).paph_hash).toBeNull();
    // the shards that answered in full are remembered: the retry asks only the one that did not
    expect(w.db.prepare("SELECT done FROM paph_progress WHERE post_id = 2001").get()).toEqual({ done: "[1,2]" });
    w.paph.failing.clear();
    w.paph.calls.length = 0;
    expect(await w.run(2001, "h2001", swapRedBlue(small))).toMatchObject({ status: "done", complete: true, reused: true });
    expect(w.paph.calls.filter((c) => c.endsWith(".find"))).toEqual(["paph:1000:0.find"]);
    expect(w.pairs()).toEqual(["1-1001:Copy", "1-2001:Copy", "1001-2001:Copy"]);
    expect(w.db.prepare("SELECT COUNT(*) AS n FROM paph_progress").get()).toEqual({ n: 0 });
    // a forced run starts over: every shard again
    w.paph.calls.length = 0;
    await w.run(2001, "h2001", swapRedBlue(small), true);
    expect(w.paph.calls.filter((c) => c.endsWith(".find")).sort()).toEqual(["paph:1000:0.find", "paph:1000:1.find", "paph:1000:2.find"]);
  });

  it("keeps no progress for a post deleted during a partial check: restored, it is checked in full", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2500, "copycat", 300, "h2500");
    await w.run(1, "h1", small);
    // shard 0 finds the original, shard 1 fails, and the post is deleted meanwhile
    w.paph.failing.add("paph:1000:1");
    let deleted = false;
    w.paph.after.push(async (shard, method) => {
      if (deleted || shard !== "paph:1000:0" || method !== "find") return;
      deleted = true;
      w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2500").run();
      await removeFromIndexes(w.env, 2500);
    });
    expect((await w.run(2500, "h2500", mirror(small))).status).toBe("gone");
    expect(w.pairs()).toEqual([]);
    expect(w.db.prepare("SELECT COUNT(*) AS n FROM paph_progress").get()).toEqual({ n: 0 });
    // the author restores it: no shard is taken as done, every one is asked, the copy is found
    w.paph.failing.clear();
    w.db.prepare("UPDATE posts SET deleted = 0 WHERE id = 2500").run();
    w.paph.calls.length = 0;
    expect(await w.run(2500, "h2500", mirror(small))).toMatchObject({ status: "done", complete: true });
    expect(w.paph.calls.filter((c) => c.endsWith(".find")).sort()).toEqual(["paph:1000:0.find", "paph:1000:1.find", "paph:1000:2.find"]);
    expect(w.pairs()).toEqual(["1-2500:Copy"]);
  });

  it("does not take shards as done once the progress a retry continued from was reset", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1001, "copycat", 200, "h1001");
    w.post(2001, "recolorist", 300, "h2001");
    await w.run(1, "h1", small);
    await w.run(1001, "h1001", mirror(small));
    w.paph.failing.add("paph:1000:0");
    await w.run(2001, "h2001", swapRedBlue(small));
    expect(w.db.prepare("SELECT done, shard_size FROM paph_progress WHERE post_id = 2001").get()).toEqual({ done: "[1,2]", shard_size: 1000 });
    w.paph.failing.clear();
    // while the retry asks shard 0, the post is deleted and restored: its verdicts and progress went
    let once = false;
    w.paph.after.push(async (shard, method) => {
      if (once || shard !== "paph:1000:0" || method !== "find") return;
      once = true;
      w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2001").run();
      await removeFromIndexes(w.env, 2001);
      w.db.prepare("UPDATE posts SET deleted = 0 WHERE id = 2001").run();
    });
    const r = await w.run(2001, "h2001", swapRedBlue(small));
    expect(r).toMatchObject({ status: "done", complete: false, incomplete: ["the progress this run continued from was reset: every shard is asked again"] });
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2001").get() as any).paph_hash).toBeNull();
    expect(w.db.prepare("SELECT COUNT(*) AS n FROM paph_progress").get()).toEqual({ n: 0 });
    // what shard 0 compared this time is stored; shard 1's verdict went with the deletion
    expect(w.pairs()).toEqual(["1-1001:Copy", "1-2001:Copy"]);
    w.paph.calls.length = 0;
    expect(await w.run(2001, "h2001", swapRedBlue(small))).toMatchObject({ status: "done", complete: true });
    expect(w.paph.calls.filter((c) => c.endsWith(".find")).sort()).toEqual(["paph:1000:0.find", "paph:1000:1.find", "paph:1000:2.find"]);
    expect(w.pairs()).toEqual(["1-1001:Copy", "1-2001:Copy", "1001-2001:Copy"]);
  });

  it("starts over when the shard size changed since a partial check", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1500, "copycat", 200, "h1500");
    w.post(2500, "recolorist", 300, "h2500");
    await w.run(1, "h1", small);
    await w.run(1500, "h1500", mirror(small));
    w.paph.failing.add("paph:1000:1");
    await w.run(2500, "h2500", swapRedBlue(small));
    expect(w.db.prepare("SELECT done, shard_size FROM paph_progress WHERE post_id = 2500").get()).toEqual({ done: "[0,2]", shard_size: 1000 });
    expect(w.pairs()).toEqual(["1-1500:Copy", "1-2500:Copy"]);
    w.paph.failing.clear();
    // re-sharded, 2000 ids a shard: "shard 0 done" meant ids 0–999, the new shard 0 holds 1500 too
    w.env.PAPH_SHARD_SIZE = "2000";
    resetShardCount();
    for (const [id, im] of [[1, small], [1500, mirror(small)]] as const) {
      await homeShard(w.env, id).put({ postId: id, contentHash: `h${id}`, ...wires(await fingerprint(im)) });
    }
    w.paph.calls.length = 0;
    expect(await w.run(2500, "h2500", swapRedBlue(small))).toMatchObject({ status: "done", complete: true });
    expect(w.paph.calls.filter((c) => c.endsWith(".find")).sort()).toEqual(["paph:2000:0.find", "paph:2000:1.find"]);
    expect(w.pairs()).toEqual(["1-1500:Copy", "1-2500:Copy", "1500-2500:Copy"]);
  });

  it("counts a shard answering under another engine as not done, and stores nothing it found", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1001, "copycat", 200, "h1001");
    w.post(2001, "recolorist", 300, "h2001");
    await w.run(1, "h1", small);
    await w.run(1001, "h1001", mirror(small));
    // a deploy under way: shard 0 still runs the previous release
    w.paph.engines.set("paph:1000:0", "paph-x/1.0.9 previous safe");
    const r = (await w.run(2001, "h2001", swapRedBlue(small))) as any;
    expect(r).toMatchObject({ status: "done", complete: false, incomplete: ["shards 0 (runs paph-x/1.0.9 previous safe)"] });
    const engine = await currentEngine(w.env);
    expect(w.pairs()).toEqual(["1-1001:Copy", "1001-2001:Copy"]);
    expect(w.db.prepare("SELECT DISTINCT engine FROM paph_matches").all()).toEqual([{ engine }]);
    expect(w.db.prepare("SELECT done FROM paph_progress WHERE post_id = 2001").get()).toEqual({ done: "[1,2]" });
    // once the deploy is through, the retry asks shard 0 alone and completes
    w.paph.engines.clear();
    w.paph.calls.length = 0;
    expect(await w.run(2001, "h2001", swapRedBlue(small))).toMatchObject({ status: "done", complete: true });
    expect(w.paph.calls.filter((c) => c.endsWith(".find"))).toEqual(["paph:1000:0.find"]);
    expect(w.pairs()).toEqual(["1-1001:Copy", "1-2001:Copy", "1001-2001:Copy"]);
    expect(await paphStaleVerdicts(w.env, {})).toMatchObject({ stale: 0, awaiting_recheck: 0 });
  });

  it("fits its waits in the consumer's run: the deadline is cut, a late check deferred, a late message sent again", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    w.post(3, "other", 300, "h3");
    for (const [id, im] of [[1, small], [2, mirror(small)], [3, second]] as const) {
      await homeShard(w.env, id).put({ postId: id, contentHash: `h${id}`, ...wires(await fingerprint(im)) });
    }
    // a check gets what is left of the run, less the grace
    w.paph.inputs.length = 0;
    expect((await w.run(1, "h1", small, false, { endsAt: Date.now() + 40_000 })).status).toBe("done");
    const asked = (w.paph.inputs.find((c) => c.method === "find")!.args[1] as any).budget.deadlineMs;
    expect(asked).toBeLessThanOrEqual(40_000 - STAGE_GRACE_MS);
    expect(asked).toBeGreaterThan(40_000 - STAGE_GRACE_MS - 5_000);
    // too little left: not even started (no shard asked, the mark untouched)
    w.paph.calls.length = 0;
    expect(await w.run(2, "h2", mirror(small), false, { endsAt: Date.now() + STAGE_GRACE_MS + 1_000 })).toEqual({ status: "deferred" });
    expect(w.paph.calls).toEqual([]);
    // a batch whose run is nearly over: the check that would end too late is deferred, the next
    // message is not started, both come back
    let clock = Date.now();
    const spy = vi.spyOn(Date, "now").mockImplementation(() => clock);
    try {
      w.paph.after.push((_shard, method) => {
        if (method === "putIf") clock += RUN_MS;
      });
      w.paph.calls.length = 0;
      const acks: string[] = [];
      const msgs: EnrichMessage[] = [2, 3].map((postId) => ({ postId, author: "x", permlink: `p${postId}`, stages: ["paph", "concepts"], reason: "test" }));
      await handleEnrichBatch(
        { queue: "q", messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => acks.push(`ack:${body.postId}`), retry: (o?: { delaySeconds?: number }) => acks.push(`retry:${body.postId}:${o?.delaySeconds}`) })), ackAll() {}, retryAll() {} } as any,
        w.env,
      );
      // the deferred check retries its message; the message not started is sent again as a new one
      // (it was not tried: it does not use up a delivery)
      expect(acks).toEqual(["retry:2:30", "ack:3"]);
      expect(w.env._queue.drain()).toEqual([msgs[1]]);
      expect(w.env._queue.delays).toEqual([30]);
      expect(finds(w.paph)).toBe(0);
      // the deferred message's other stages ran; the message not started did nothing
      expect(w.db.prepare("SELECT post_id, stage, status FROM jobs WHERE post_id IN (2, 3) ORDER BY post_id, stage").all()).toEqual([{ post_id: 2, stage: "concepts", status: "done" }]);
    } finally {
      spy.mockRestore();
    }
  });

  it("checks under a short configured stage deadline; only the run's end defers a check", async () => {
    const w = world({ PAPH_BUDGET_STAGE: "deadline_ms:700" });
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    w.paph.inputs.length = 0;
    expect(await w.run(2, "h2", mirror(small))).toMatchObject({ status: "done", complete: true });
    expect((w.paph.inputs.find((c) => c.method === "find")!.args[1] as any).budget.deadlineMs).toBe(700);
    expect(w.pairs()).toEqual(["1-2:Copy"]);
  });

  it("does not wait on a slow home shard past what the run can spare", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    await w.run(2, "h2", mirror(small));
    // the home shard is slow from the start: the run gives up on it at once
    w.paph.delays.set("paph:1000:0", 2_000);
    let t0 = Date.now();
    expect(await w.run(2, "h2", mirror(small), true, { endsAt: Date.now() + STAGE_GRACE_MS + STAGE_MIN_MS + 300 })).toEqual({ status: "deferred" });
    expect(Date.now() - t0).toBeLessThan(1_500);
    expect(w.pairs()).toEqual(["1-2:Copy"]); // nothing was undone
    // it turns slow once the run has begun (its mark cleared): the run still ends in time
    w.paph.delays.clear();
    w.paph.after.push((shard, method) => {
      if (shard === "paph:1000:0" && method === "info") w.paph.delays.set("paph:1000:0", 2_000);
    });
    t0 = Date.now();
    expect(await w.run(2, "h2", mirror(small), true, { endsAt: Date.now() + STAGE_GRACE_MS + STAGE_MIN_MS + 300 })).toEqual({ status: "deferred" });
    expect(Date.now() - t0).toBeLessThan(1_500);
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2").get() as any).paph_hash).toBeNull();
    // the vector pass gives up on it the same way
    t0 = Date.now();
    expect(await paphVectorPass(w.env, 2, "h2", [1, 0, 0], { endsAt: Date.now() + STAGE_GRACE_MS + STAGE_MIN_MS + 300 })).toBe(0);
    expect(Date.now() - t0).toBeLessThan(1_500);
  });

  it("cannot put an older image back over a newer one's entry", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2old");
    await w.run(1, "h1", small);
    // run A, for the old image, has seen the post show it; before A puts, the post is edited and
    // run B, for the new image, runs to completion
    const d1 = w.env.DB as any;
    const prepare = d1.prepare.bind(d1);
    let armed = true;
    d1.prepare = (sql: string) => {
      const st = prepare(sql);
      if (!/^SELECT a\.content_hash FROM artworks a JOIN posts p/.test(sql)) return st;
      return new Proxy(st, {
        get: (t, k) =>
          k !== "bind"
            ? Reflect.get(t, k)
            : (...args: unknown[]) => {
                const b = t.bind(...args);
                return new Proxy(b, {
                  get: (bt, bk) =>
                    bk !== "first"
                      ? Reflect.get(bt, bk)
                      : async () => {
                          const r = await bt.first();
                          if (armed && args[0] === 2) {
                            armed = false;
                            w.db.prepare("UPDATE artworks SET content_hash = 'h2new' WHERE post_id = 2").run();
                            expect(await w.run(2, "h2new", mirror(small))).toMatchObject({ status: "done", complete: true });
                          }
                          return r;
                        },
                });
              },
      });
    };
    expect(await w.run(2, "h2old", second)).toEqual({ status: "superseded" });
    d1.prepare = prepare;
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2new");
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    // the other order: an older image's entry lands after the newer run read the entry: the newer
    // run, whose image the post shows, puts its own over it
    w.db.prepare("UPDATE artworks SET content_hash = 'h2newer' WHERE post_id = 2").run();
    let once = false;
    w.paph.after.push(async (shard, method) => {
      if (once || shard !== "paph:1000:0" || method !== "info") return;
      once = true;
      await homeShard(w.env, 2).put({ postId: 2, contentHash: "h2old", ...wires(await fingerprint(second)) });
    });
    expect(await w.run(2, "h2newer", swapRedBlue(small))).toMatchObject({ status: "done", complete: true });
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2newer");
  });

  it("drops the mark of a work whose entry shows another image, so the stage indexes it again", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    await w.run(2, "h2", mirror(small));
    // what a reverted edit with both runs in flight could leave: the entry shows the other image
    await homeShard(w.env, 2).put({ postId: 2, contentHash: "h2-reverted", ...wires(await fingerprint(second)) });
    expect(await paphGc(w.env)).toMatchObject({ removed: 0, reset: 1 });
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2").get() as any).paph_hash).toBeNull();
    expect(w.env._queue.drain()).toEqual([{ postId: 2, author: "copycat", permlink: "p2", stages: ["paph"], reason: "paph-gc" }]);
    expect((w.db.prepare("SELECT status FROM jobs WHERE post_id = 2 AND stage = 'paph'").get() as any).status).toBe("queued");
    expect(await paphGc(w.env)).toMatchObject({ reset: 0 }); // a work whose check is pending is its stage's
    expect(await w.run(2, "h2", mirror(small))).toMatchObject({ status: "done", complete: true });
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2");
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    expect(await paphGc(w.env)).toMatchObject({ reset: 0 });
  });

  it("sends a work marked complete but missing from its shard back to the stage", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    w.post(1500, "other", 250, "h1500");
    await w.run(1, "h1", small);
    await w.run(2, "h2", mirror(small));
    await w.run(1500, "h1500", second);
    // what a removal racing a restore could leave: the mark, without the entry
    await homeShard(w.env, 2).remove(2);
    w.env._queue.drain();
    expect(await paphGc(w.env)).toMatchObject({ removed: 0, reset: 1 });
    expect(w.env._queue.drain().map((m) => m.postId)).toEqual([2]);
    expect(await w.run(2, "h2", mirror(small))).toMatchObject({ status: "done", complete: true });
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2");
    expect(await paphGc(w.env)).toMatchObject({ reset: 0 });
  });

  it("reconciles at most a hundred works a page, in batches the queue takes, resuming where it stopped", async () => {
    const w = world();
    // 150 works marked complete whose shard lost them (as a new shard size would leave them)
    for (let id = 10; id < 160; id++) w.post(id, `a${id}`, 100 + id, `h${id}`);
    w.db.prepare("UPDATE artworks SET paph_hash = content_hash, paph_engine = 'some engine'").run();
    expect(w.env.ENRICH_QUEUE).toBe(w.env._queue);
    const sizes: number[] = [];
    const sendBatch = w.env._queue.sendBatch.bind(w.env._queue);
    w.env._queue.sendBatch = async (msgs: Array<{ body: EnrichMessage }>) => {
      sizes.push(msgs.length);
      return sendBatch(msgs);
    };
    expect(await paphGc(w.env, 500)).toMatchObject({ reset: 100, wrapped: [] });
    expect(await paphGc(w.env, 500)).toMatchObject({ reset: 50, wrapped: [0] });
    expect(sizes).toEqual([100, 50]);
    expect(w.env._queue.drain().map((m) => m.postId)).toEqual(Array.from({ length: 150 }, (_, i) => 10 + i));
    expect(w.db.prepare("SELECT COUNT(*) AS n FROM artworks WHERE paph_hash IS NOT NULL").get()).toEqual({ n: 0 });
    expect(await paphGc(w.env, 500)).toMatchObject({ reset: 0, wrapped: [0] });
  });

  it("leaves a newer run's entry and mark alone when an older run of the same post fails", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2a");
    await w.run(1, "h1", small);
    await w.run(2, "h2a", second);
    // edited twice in quick succession: the run for the first edit fails (its image moved on on
    // chain) just after the run for the second one has put its entry
    w.db.prepare("UPDATE artworks SET content_hash = 'h2b' WHERE post_id = 2").run();
    let release!: () => void;
    const newerPut = new Promise<void>((r) => (release = r));
    w.paph.after.push((shard, method) => {
      if (shard === "paph:1000:0" && method === "putIf") release();
    });
    const older = w.run(2, "h2b", (async () => {
      await newerPut;
      throw new Error("superseded: the image changed since the stats stage");
    }) as any);
    await new Promise((r) => setTimeout(r, 20)); // the older run is hashing
    w.db.prepare("UPDATE artworks SET content_hash = 'h2c' WHERE post_id = 2").run();
    const newer = w.run(2, "h2c", mirror(small));
    await expect(older).rejects.toThrow(/^superseded:/);
    expect(await newer).toMatchObject({ status: "done", complete: true });
    expect((await homeShard(w.env, 2).info(2))!.contentHash).toBe("h2c");
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2").get() as any).paph_hash).toBe("h2c");
    expect(w.pairs()).toEqual(["1-2:Copy"]);
  });

  it("near the end of a run, takes a check the mark says is complete as it is, and defers the others", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(2, "copycat", 200, "h2");
    await w.run(1, "h1", small);
    w.paph.calls.length = 0;
    const late = () => ({ endsAt: Date.now() + STAGE_GRACE_MS + 1_000 });
    expect(await w.run(1, "h1", small, false, late())).toEqual({ status: "unchanged" });
    expect(await w.run(1, "h1", small, true, late())).toEqual({ status: "deferred" });
    expect(await w.run(2, "h2", mirror(small), false, late())).toEqual({ status: "deferred" });
    expect(w.paph.calls).toEqual([]);
  });

  it("records no job for a run a newer image overtook, so the sweeper still finds the work", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    await homeShard(w.env, 1).put({ postId: 1, contentHash: "h1", ...wires(fpSmall) });
    w.db.prepare("UPDATE artworks SET stats_hash = content_hash, features_hash = content_hash, concepts_hash = 'c'").run();
    // the post is edited while the run asks its shard
    w.paph.after.push((_shard, method) => {
      if (method === "info") w.db.prepare("UPDATE artworks SET content_hash = 'h1-newer', stats_hash = 'h1-newer', features_hash = 'h1-newer' WHERE post_id = 1").run();
    });
    const acks: string[] = [];
    const body: EnrichMessage = { postId: 1, author: "original", permlink: "p1", stages: ["paph"], reason: "test" };
    await handleEnrichBatch({ queue: "q", messages: [{ id: "0", timestamp: new Date(), attempts: 1, body, ack: () => acks.push("ack"), retry: () => acks.push("retry") }], ackAll() {}, retryAll() {} } as any, w.env);
    expect(acks).toEqual(["ack"]);
    expect(w.db.prepare("SELECT COUNT(*) AS n FROM jobs WHERE post_id = 1 AND stage = 'paph'").get()).toEqual({ n: 0 });
    w.paph.after.length = 0;
    expect((await sweep(w.env, 10)).byStage.paph).toBe(1);
  });

  it("lets two deliveries of the same retry both end complete, and settles the job", async () => {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1001, "copycat", 200, "h1001");
    w.post(2001, "recolorist", 300, "h2001");
    await w.run(1, "h1", small);
    await w.run(1001, "h1001", mirror(small));
    w.paph.failing.add("paph:1000:0");
    await w.run(2001, "h2001", swapRedBlue(small));
    w.paph.failing.clear();
    w.paph.calls.length = 0;
    const [a, b] = await Promise.all([w.run(2001, "h2001", swapRedBlue(small)), w.run(2001, "h2001", swapRedBlue(small))]);
    expect(w.paph.calls.filter((c) => c === "paph:1000:0.find")).toHaveLength(2); // both continued from the progress
    expect([a, b]).toMatchObject([{ status: "done", complete: true, incomplete: [] }, { status: "done", complete: true, incomplete: [] }]);
    expect(w.pairs()).toEqual(["1-1001:Copy", "1-2001:Copy", "1001-2001:Copy"]);
    // a delivery that finds the check complete settles a job an earlier one left failed
    w.db.prepare("INSERT INTO jobs (post_id, stage, status, attempts, error, updated) VALUES (2001, 'paph', 'failed', 3, 'incomplete', 0)").run();
    const acks: string[] = [];
    const body: EnrichMessage = { postId: 2001, author: "recolorist", permlink: "p2001", stages: ["paph"], reason: "test" };
    await handleEnrichBatch({ queue: "q", messages: [{ id: "0", timestamp: new Date(), attempts: 2, body, ack: () => acks.push("ack"), retry: () => acks.push("retry") }], ackAll() {}, retryAll() {} } as any, w.env);
    expect(acks).toEqual(["ack"]);
    expect(w.db.prepare("SELECT status, attempts, error FROM jobs WHERE post_id = 2001 AND stage = 'paph'").get()).toEqual({ status: "done", attempts: 0, error: null });
  });

  it("leaves the stage incomplete when a nominating channel was unavailable", async () => {
    const w = world();
    w.post(1, "original", 100, "h1", { phash: "0123456789abcdef" });
    w.db.exec("DROP TABLE phash_bands"); // the pHash channel's index cannot be read
    const r = await w.run(1, "h1", small);
    expect(r).toMatchObject({ status: "done", complete: false, incomplete: ["pHash neighbours unavailable"] });
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 1").get() as any).paph_hash).toBeNull();
  });

  it("checks the fresh embedding's neighbours afterwards, without touching the stage's mark", async () => {
    const w = world({ HF_EMBED_URL: "https://embed.test/embed" });
    w.post(1, "original", 100, "h1");
    w.post(2, "recolorist", 300, "h2");
    await homeShard(w.env, 1).put({ postId: 1, contentHash: "h1", ...wires(fpSmall) });
    await homeShard(w.env, 2).put({ postId: 2, contentHash: "h2", ...wires(await fingerprint(swapRedBlue(small))) });
    await w.env._vec.upsert([{ id: "1", values: [1, 0, 0], metadata: {} }]);
    expect(await paphVectorPass(w.env, 2, "h2", [1, 0, 0])).toBe(1);
    expect(w.pairs()).toEqual(["1-2:Copy"]);
    expect((w.db.prepare("SELECT via FROM paph_matches").get() as any).via).toBe("vector");
    expect((w.db.prepare("SELECT paph_hash FROM artworks WHERE post_id = 2").get() as any).paph_hash).toBeNull();
  });
});

describe("interactive checks", () => {
  async function indexed() {
    const w = world();
    w.post(1, "original", 100, "h1");
    w.post(1500, "other", 150, "h1500");
    w.post(2500, "copycat", 200, "h2500");
    await w.run(1, "h1", small);
    await w.run(1500, "h1500", second);
    await w.run(2500, "h2500", mirror(small));
    for (let i = 0; i < 12; i++) {
      const id = 10 + i;
      w.post(id, "filler", 50 + i, `hf${i}`);
      await w.run(id, `hf${i}`, art(3000 + i));
    }
    return w;
  }

  it("answer an upload in every shard within the time budget, then from the day-long cache", async () => {
    const w = await indexed();
    w.paph.calls.length = 0;
    const t0 = Date.now();
    const first = await copiesOfImage(w.env, crop(small, 10, 13, 86, 107), { startedAt: t0 });
    expect(first.cached).toBe(false);
    expect(first.partial).toBe(false);
    expect(first.items.map((it) => it.id).sort((a, b) => a - b)).toEqual([1, 2500]);
    expect(first.items[0].copy.relation).toBeNull();
    expect(first.shards).toMatchObject({ asked: 3, answered: 3, failed: 0 });
    expect(first.took_ms).toBeLessThan(1000);
    expect(finds(w.paph)).toBe(3);
    const [key] = [...w.env._kv.m.keys()].filter((k) => k.startsWith("paph:img:"));
    expect(Math.round((w.env._kv.expires.get(key)! - Date.now()) / 1000)).toBeGreaterThan(86_390);
    // the same pixels, whatever the encoding: from the cache, no shard asked
    const again = await copiesOfImage(w.env, crop(small, 10, 13, 86, 107), {});
    expect(again).toMatchObject({ cached: true, as_of: first.as_of });
    expect(again.items.map((it) => it.id)).toEqual(first.items.map((it) => it.id));
    expect(finds(w.paph)).toBe(3);
    // the listing is rebuilt per request: a copy deleted since drops out of the cached answer
    w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2500").run();
    const later = await copiesOfImage(w.env, crop(small, 10, 13, 86, 107), {});
    expect(later.items.map((it) => it.id)).toEqual([1]);
    expect(later.hidden).toBe(1);
  });

  it("keep an answer with a shard missing for ten minutes only", async () => {
    const w = await indexed();
    w.paph.failing.add("paph:1000:2");
    const r = await copiesOfImage(w.env, upscale(small, 2), {});
    expect(r.partial).toBe(true);
    expect(r.items.map((it) => it.id)).toEqual([1]);
    const [key] = [...w.env._kv.m.keys()].filter((k) => k.startsWith("paph:img:"));
    expect(Math.round((w.env._kv.expires.get(key)! - Date.now()) / 1000)).toBeLessThanOrEqual(600);
  });

  it("answer within the request's time budget even when a shard is slow, and say what is missing", async () => {
    const w = await indexed();
    w.env.PAPH_QUERY_MS = "400";
    w.paph.delays.set("paph:1000:2", 1_500);
    const t0 = Date.now();
    const r = await copiesOfImage(w.env, paste(small, second, 60, 45), { startedAt: t0 });
    expect(r.partial).toBe(true);
    expect(r.shards).toMatchObject({ asked: 3, answered: 2, failed: 1 });
    expect(r.items.map((it) => it.id)).toContain(1);
    expect(Date.now() - t0).toBeLessThan(1_000);
  });

  it("re-check a stored work live, and report a pair with comparator 42 beside PAPH-X, cached", async () => {
    const w = await indexed();
    const live = await liveCopiesOf(w.env, 1, {});
    expect(live.method).toBe("live");
    expect(live.items.map((it) => [it.id, it.copy.relation])).toEqual([[2500, "later"]]);
    const rep = (await pairReport(w.env, 1, 2500, true))!;
    expect(rep.cached).toBe(false);
    expect((rep.report as any).verdict).toBe("Copy");
    expect((rep.report as any).comparator).toBe(50);
    expect((rep.report as any).fallback.comparator).toBe(42);
    expect((rep.wires as any).a.t1).toBe(base64Encode(fpSmall.t1));
    expect((await pairReport(w.env, 1, 2500, true))!.cached).toBe(true);
    expect(await pairReport(w.env, 1, 99_999, false)).toBeNull();
    // never about a deleted post, or an image its post no longer shows
    w.db.prepare("UPDATE artworks SET content_hash = 'h1500-edited' WHERE post_id = 1500").run();
    expect(await pairReport(w.env, 1, 1500, false)).toBeNull();
    w.db.prepare("UPDATE posts SET deleted = 1 WHERE id = 2500").run();
    expect(await pairReport(w.env, 1, 2500, false)).toBeNull();
  });

  it("compute the embedding of a semantic upload only when the answer is not cached", async () => {
    const w = await indexed();
    let embeds = 0;
    const embed = async () => (embeds++, [1, 0, 0]);
    const first = await copiesOfImage(w.env, mirror(small), { semantic: true, embed });
    const again = await copiesOfImage(w.env, mirror(small), { semantic: true, embed });
    expect([first.cached, again.cached, embeds]).toEqual([false, true, 1]);
    // a failed embedding: answered without its neighbours, and kept ten minutes only
    const failing = await copiesOfImage(w.env, crop(small, 5, 5, 90, 110), { semantic: true, embed: async () => Promise.reject(new Error("Space asleep")) });
    expect(failing.partial).toBe(true);
    expect(failing.notes).toEqual(["semantic candidates unavailable: Space asleep"]);
    expect(failing.items.map((it) => it.id)).toContain(1);
  });

  it("clean the shards of works whose posts were deleted, a page at a time", async () => {
    const w = await indexed();
    w.db.prepare("UPDATE posts SET deleted = 1 WHERE id IN (11, 13, 2500)").run(); // deleted, their removal from the shards lost
    const shard0 = w.paph.shard("paph:1000:0").store;
    const before = shard0.list(0, 100).length;
    const first = await paphGc(w.env, 5);
    expect(first.failed).toBe(0);
    let removed = first.removed;
    for (let i = 0; i < 5; i++) removed += (await paphGc(w.env, 5)).removed;
    expect(removed).toBe(3);
    expect(shard0.list(0, 100).length).toBe(before - 2);
    expect(await homeShard(w.env, 2500).info(2500)).toBeNull();
    expect(await homeShard(w.env, 1).info(1)).not.toBeNull();
    // the nightly pass goes through every shard once, whatever its size
    w.db.prepare("UPDATE posts SET deleted = 1 WHERE id IN (12, 1500)").run();
    const pass = await paphGcPass(w.env, 4);
    expect(pass).toMatchObject({ removed: 2, failed: [], unfinished: [] });
    // a shard that fails stops alone; the others are gone through
    w.db.prepare("UPDATE posts SET deleted = 1 WHERE id IN (14, 20)").run();
    w.paph.failing.add("paph:1000:1");
    expect(await paphGcPass(w.env, 4)).toMatchObject({ removed: 2, failed: [1] });
    w.paph.failing.clear();
    expect(await homeShard(w.env, 1500).info(1500)).toBeNull();
  });

  it("serve the routes: stored, live (admin only), uploads, reports, index facts — and 503 when off", async () => {
    const w = await indexed();
    const stored = await call(w.env, "/copies/1");
    expect(stored.status).toBe(200);
    expect(stored.body.items.map((it: any) => [it.id, it.copy.verdict, it.copy.relation])).toEqual([[2500, "Copy", "later"]]);
    expect((await call(w.env, "/copies/1?live=1")).status).toBe(403);
    expect((await call(w.env, "/copies/1?live=1", { admin: true })).body.method).toBe("live");
    const png = await encodePng(mirror(small));
    const form = new FormData();
    form.set("image", new Blob([png.slice().buffer as ArrayBuffer], { type: "image/png" }), "x.png");
    const up = await call(w.env, "/copies-by-image", { method: "POST", body: form });
    expect(up.status).toBe(200);
    expect(up.body.items.map((it: any) => it.id).sort((a: number, b: number) => a - b)).toEqual([1, 2500]);
    expect(up.body.items.find((it: any) => it.id === 1).copy.mirrored).toBe(true);
    const json = await call(w.env, "/copies-by-image?min=identical", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ image: base64Encode(await encodePng(small)) }) });
    expect(json.body.items.map((it: any) => [it.id, it.copy.verdict])).toEqual([[1, "Identical"]]);
    const rep = await call(w.env, "/copies/1/report/2500");
    expect(rep.body.report.verdict).toBe("Copy");
    expect((await call(w.env, "/paph/2500")).body).toMatchObject({ shard: 2, indexed: true, complete: true, image_current: true, derivation_current: true });
    // ids D1 does not know never reach a shard (no Durable Object is created for them)
    const shards = [...w.paph.shards.keys()].sort();
    expect((await call(w.env, "/paph/123456789012")).status).toBe(404);
    expect((await call(w.env, "/copies/5/report/333333333333")).status).toBe(404);
    expect([...w.paph.shards.keys()].sort()).toEqual(shards);
    expect((await call(w.env, "/copies/1?min=unrelated")).body.min).toBe("Suspected");
    const off = makeEnv({});
    expect((await call(off, "/copies/1")).status).toBe(503);
    expect((await call(off, "/copies-by-image", { method: "POST", body: form })).status).toBe(503);
  });
});

describe("moving the store to wire 4 (PAPH-X 1.2.0)", () => {
  /** what 1.1.2 marked its works with */
  const OLD = "paph-x/1.1.2 c50 CAL-004-PROPOSED:91b545f801f5a095 X2-PROVISIONAL:27993afaaca76d11 si:dfe99f33b6a8dfc0 k1 w3 safe";
  /** the identity of a verdict this release reached on wire-3 sides */
  const W3 = () => ID().replace(/ w4 /, " w3 ");
  /**
   * A work as 1.1.2 left it: its entry as that release wrote it (wire 3, its derivation, SI3
   * signature and postings), its stage marked done under 1.1.2's identity.
   */
  async function legacy(w: ReturnType<typeof world>, id: number, hash: string, im: RgbaImage) {
    const fp3 = await fingerprint(im, undefined, 3);
    putAs112(w.paph.shard(`paph:1000:${shardOf(w.env, id)}`).db, rt, id, hash, { t1: fp3.t1, t2: fp3.t2 });
    w.db.prepare("UPDATE artworks SET paph_hash = ?, paph_engine = ? WHERE post_id = ?").run(hash, OLD, id);
  }
  /** for each check sent to a shard: whether it brought the query's wire-3 twin */
  const twins = (p: FakePaph) => p.inputs.filter((x) => x.method === "find").map((x) => !!(x.args[1] as { twin?: unknown }).twin);
  const verdicts = (w: ReturnType<typeof world>) => w.db.prepare("SELECT a, b, engine FROM paph_matches ORDER BY a, b").all();
  // a new isolate: the checks have not heard from the shards yet
  beforeEach(() => resetLegacyStore());
  afterEach(() => resetLegacyStore());

  it("checks a new work against works not re-hashed yet through its wire-3 twin, and the re-hash puts the verdict on wire 4", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "carol", 150, "h1500");
    w.post(2, "bob", 200, "h2");
    await legacy(w, 1, "h1", small);
    await legacy(w, 1500, "h1500", second);
    // a copy posted after the deploy: hashed in both formats, it finds the original at once
    const o = await w.run(2, "h2", mirror(small));
    expect(o).toMatchObject({ status: "done", complete: true, twin: true, stats: { unreadable: 0, legacy: 2 } });
    expect(twins(w.paph)).toEqual([true, true]);
    // reached on wire 3, and named so: stale until the original's own re-check
    expect(verdicts(w)).toEqual([{ a: 1, b: 2, engine: W3() }]);
    expect(await paphStaleVerdicts(w.env)).toMatchObject({ stale: 0, wire3: 1 });
    expect((await copiesOf(w.env, 2, {})).items.map((it) => [it.id, it.copy.verdict === "Copy" || it.copy.verdict === "Identical", it.copy.engine])).toEqual([[1, true, W3()]]);
    expect(await paphStatus(w.env)).toMatchObject({ wire: { format: 4, legacy_works: 2, legacy_shards: 2, twin: true } });
    // the original's re-check: its stored wires are wire 3, so it is hashed again, and its previous
    // partner is compared on wire 4
    expect(await w.run(1, "h1", small)).toMatchObject({ status: "done", complete: true, reused: false, twin: true });
    expect(await homeShard(w.env, 1).info(1)).toMatchObject({ wire: 4, current: true });
    expect(verdicts(w)).toEqual([{ a: 1, b: 2, engine: ID() }]);
    expect(storeMayHoldLegacy()).toBe(true);
    // the last one: no shard holds wire 3 any more, and checks stop bringing a twin
    await w.run(1500, "h1500", second);
    expect(storeMayHoldLegacy()).toBe(false);
    expect(await paphStatus(w.env)).toMatchObject({ wire: { legacy_works: 0, legacy_shards: 0, twin: false } });
    w.paph.inputs.length = 0;
    const up = await copiesOfImage(w.env, crop(small, 10, 13, 86, 107), {});
    expect(up.partial).toBe(false);
    expect(up.items.map((it) => [it.id, it.copy.engine])).toEqual([[1, ID()], [2, ID()]]);
    expect(twins(w.paph)).toEqual([false, false]);
    expect(await paphStaleVerdicts(w.env)).toMatchObject({ stale: 0, wire3: 0 });
    // and the stage hashes a new work once
    w.post(4, "dave", 300, "h4");
    expect(await w.run(4, "h4", swapRedBlue(small))).toMatchObject({ status: "done", complete: true, twin: false, stats: { unreadable: 0, legacy: 0 } });
  });

  it("never lets a verdict reached on wire 3 replace one its stored side's own re-check reached meanwhile on wire 4", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "bob", 150, "h1500");
    await legacy(w, 1, "h1", small);
    await legacy(w, 1500, "h1500", mirror(small));
    // work 1's re-check reaches shard 1 while 1500 is still on wire 3 (compared through 1's twin);
    // before that answer is back, 1500's whole re-check runs: re-hashed, it compares 1 on wire 4
    let raced = false;
    w.paph.after.push(async (shard, method) => {
      if (raced || shard !== "paph:1000:1" || method !== "find") return;
      raced = true;
      expect(await w.run(1500, "h1500", mirror(small))).toMatchObject({ status: "done", complete: true });
      expect(verdicts(w)).toEqual([{ a: 1, b: 1500, engine: ID() }]);
    });
    expect(await w.run(1, "h1", small)).toMatchObject({ status: "done", complete: true, twin: true });
    expect(raced).toBe(true);
    // work 1's comparison on wire 3 came back last: it replaced nothing
    expect(verdicts(w)).toEqual([{ a: 1, b: 1500, engine: ID() }]);
    expect(await paphStaleVerdicts(w.env)).toMatchObject({ stale: 0, wire3: 0 });
  });

  it("compares again on wire 4 a verdict reached on wire 3 that no re-check will replace, and never purges one", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(2, "bob", 200, "h2");
    await w.run(1, "h1", small);
    await w.run(2, "h2", mirror(small));
    expect(verdicts(w)).toEqual([{ a: 1, b: 2, engine: ID() }]);
    // as a race may leave it: the pair's verdict reached on wire 3, written after both works' checks
    w.db.prepare("UPDATE paph_matches SET engine = ?, verdict = 'Suspected', state = 2").run(W3());
    expect(await paphStaleVerdicts(w.env)).toMatchObject({ stale: 0, wire3: 1 });
    expect(await paphStaleVerdicts(w.env, { purge: true })).toMatchObject({ purged: 0 });
    // not while one side still waits for its own re-check: that re-check compares it
    w.db.prepare("UPDATE artworks SET paph_engine = ? WHERE post_id = 2").run(OLD);
    expect(await paphHealWire3(w.env)).toEqual({ found: 0, healed: 0, unresolved: 0, failed: 0 });
    w.db.prepare("UPDATE artworks SET paph_engine = ? WHERE post_id = 2").run(ID());
    expect(await paphHealWire3(w.env)).toEqual({ found: 1, healed: 1, unresolved: 0, failed: 0 });
    expect(w.db.prepare("SELECT a, b, verdict, engine FROM paph_matches").all()).toEqual([{ a: 1, b: 2, verdict: expect.stringMatching(/^(Copy|Identical)$/), engine: ID() }]);
    expect(await paphHealWire3(w.env)).toEqual({ found: 0, healed: 0, unresolved: 0, failed: 0 });
  });

  it("brings the twin when a retry reuses the stored wires, so the shard the first run missed still compares works on wire 3", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "bob", 150, "h1500");
    await legacy(w, 1, "h1", small);
    // the copy's first run: shard 0, which holds the original, does not answer
    w.paph.failing.add("paph:1000:0");
    expect(await w.run(1500, "h1500", mirror(small))).toMatchObject({ status: "done", complete: false, reused: false, twin: true });
    w.paph.failing.clear();
    w.paph.inputs.length = 0;
    // the retry reuses the stored wires and asks shard 0 alone — with the twin, decoded for it
    expect(await w.run(1500, "h1500", mirror(small))).toMatchObject({ status: "done", complete: true, reused: true, twin: true, stats: { unreadable: 0 } });
    expect(twins(w.paph)).toEqual([true]);
    expect(verdicts(w)).toEqual([{ a: 1, b: 1500, engine: W3() }]);
  });

  it("answers an upload in full while works wait for their re-hash, and keeps one that met them without a twin ten minutes", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(1500, "carol", 150, "h1500");
    await w.run(1, "h1", small);
    await w.run(1500, "h1500", second);
    expect(storeMayHoldLegacy()).toBe(false);
    // a work on wire 3 the checks have not heard of (say, restored from before the re-hash)
    w.post(3, "dave", 120, "h3");
    await legacy(w, 3, "h3", second);
    const blind = await copiesOfImage(w.env, mirror(second), {});
    expect(blind.items.map((it) => it.id)).toEqual([1500]);
    expect(blind.partial).toBe(true);
    expect(blind.notes).toEqual(["1 nominated work was not compared (stored on another wire format, or unreadable)"]);
    const [key] = [...w.env._kv.m.keys()].filter((k) => k.startsWith("paph:img:"));
    expect(Math.round((w.env._kv.expires.get(key)! - Date.now()) / 1000)).toBeLessThanOrEqual(600);
    // the shards said so: from now on an upload brings its twin, and is answered in full
    expect(storeMayHoldLegacy()).toBe(true);
    const full = await copiesOfImage(w.env, crop(second, 20, 20, 180, 180), {});
    expect(full.partial).toBe(false);
    expect(full.items.map((it) => [it.id, it.copy.engine]).sort()).toEqual([[1500, ID()], [3, W3()]].sort());
  });

  it("re-checks live a work not re-hashed yet against the works of its own format, and says what it could not compare", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(2, "bob", 200, "h2");
    w.post(3, "erin", 210, "h3");
    await legacy(w, 1, "h1", small);
    await legacy(w, 3, "h3", upscale(small, 2));
    await w.run(2, "h2", mirror(small));
    const live = await liveCopiesOf(w.env, 1, { min: "copy" });
    // its wire-3 copy is compared; the re-hashed one (a previous partner) cannot be, and is counted
    expect(live.items.map((it) => it.id)).toEqual([3]);
    expect(live.partial).toBe(true);
    expect(live.note).toBe("1 nominated work was not compared: stored on another wire format than this work's (wire 3), which the paph stage re-hashes to wire 4, or unreadable");
  });

  it("reports a pair of two formats as refused, says why, and reports it anew once both are re-hashed", async () => {
    const w = world();
    w.post(1, "alice", 100, "h1");
    w.post(2, "bob", 200, "h2");
    await legacy(w, 1, "h1", small);
    await w.run(2, "h2", mirror(small));
    const mixed = (await pairReport(w.env, 1, 2, false))!;
    expect(mixed).toMatchObject({ wire_formats: { a: 3, b: 4 }, cached: false });
    expect((mixed.report as { verdict: string }).verdict).toBe("Indeterminate");
    expect(mixed.note).toBe("the stored wires are of two formats (3 and 4): the pair is compared once both are re-hashed");
    expect((await pairReport(w.env, 1, 2, false))!.cached).toBe(true);
    await w.run(1, "h1", small);
    // the same images, new wires: a new report, not the cached refusal
    const after = (await pairReport(w.env, 1, 2, false))!;
    expect(after).toMatchObject({ wire_formats: { a: 4, b: 4 }, cached: false, engine: ID() });
    expect((after.report as { verdict: string }).verdict).toMatch(/^(Copy|Identical)$/);
    expect(after.note).toBeUndefined();
  });
});

describe("the pipeline, from the chain", () => {
  it("fingerprints every new artwork, finds the copy, lists it from both sides, and forgets a deleted one", async () => {
    resetShardCount();
    const paph = fakePaph(rt);
    const env = makeEnv({ PAPH: paph.ns, PAPH_SHARD_SIZE: "1000", ADMIN_TOKEN: TOKEN } as any);
    const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
    const T0 = Date.UTC(2026, 8, 1) / 1000;
    const uri = (b: Uint8Array, mime = "image/png") => `data:${mime};base64,${base64Encode(b)}`;
    const meta = JSON.stringify({ app: "pixagram/3.0.2", format: "image", tags: [] });
    const chainPost = (author: string, permlink: string, created: number, body: string, title: string) => ({
      author, permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title, body, json_metadata: meta,
      created: iso(created), last_update: iso(created), depth: 0, children: 0, net_votes: 0,
      pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
    });
    const posts = [
      chainPost("alice", "anonymous", T0, uri(fixture("small.webp"), "image/webp"), "Anonymous"),
      chainPost("dave", "invader", T0 + 3600, uri(fixture("second.webp"), "image/webp"), "Space Invader"),
      chainPost("bob", "my-anonymous", T0 + 86400, uri(await encodePng(mirror(small))), "Mine"),
    ];
    const net = installFetch({ rpcUrl: "https://rpc.test", chain: { posts: posts as any } });
    const ids: Record<string, number> = {};
    for (const p of posts) ids[p.author] = (await ingestPost(env, p as any, null, "test")).postId!;
    const sent = env._queue.drain();
    expect(sent.every((m) => m.stages!.includes("paph"))).toBe(true);
    const drain = async (msgs: EnrichMessage[]) => {
      const acks: string[] = [];
      await handleEnrichBatch({ queue: "q", messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => acks.push("ack"), retry: () => acks.push("retry") })), ackAll() {}, retryAll() {} } as any, env);
      return acks;
    };
    expect(await drain(sent)).toEqual(["ack", "ack", "ack"]);
    const jobs = env._db.raw.prepare("SELECT post_id, status FROM jobs WHERE stage = 'paph' ORDER BY post_id").all() as any[];
    expect(jobs.map((j) => j.status)).toEqual(["done", "done", "done"]);
    const pairs = () => (env._db.raw.prepare("SELECT a, b, verdict FROM paph_matches").all() as any[]).map((r) => `${r.a}-${r.b}:${r.verdict}`);
    expect(pairs()).toEqual([`${ids.alice}-${ids.bob}:Copy`]);
    const fromOriginal = await call(env, `/copies/${ids.alice}`);
    expect(fromOriginal.body.items.map((it: any) => [it.author, it.copy.relation, it.copy.mirrored])).toEqual([["bob", "later", true]]);
    expect((await call(env, `/copies/${ids.bob}`)).body.items.map((it: any) => [it.author, it.copy.relation])).toEqual([["alice", "earlier"]]);
    // a paph-only re-check (after a release) reuses the stored wires: the chain is not asked
    const rpcBefore = net.calls.length;
    expect(await drain([{ postId: ids.bob, author: "bob", permlink: "my-anonymous", stages: ["paph"], force: true, reason: "test" }])).toEqual(["ack"]);
    expect(net.calls.length).toBe(rpcBefore);
    expect(pairs()).toEqual([`${ids.alice}-${ids.bob}:Copy`]);
    // bob deletes his post on chain: the copy leaves the index and the listings
    posts[2].body = "deleted";
    await ingestPost(env, posts[2] as any, null, "test");
    expect(pairs()).toEqual([]);
    expect(await homeShard(env, ids.bob).info(ids.bob)).toBeNull();
    expect((await call(env, `/copies/${ids.alice}`)).body.items).toEqual([]);
  });
});
