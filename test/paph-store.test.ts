// Copy detection, the engine and one shard: PAPH-X (the real WebAssembly of the installed release)
// on real Pixagram artworks and procedural distractors, the shard's SQLite on node:sqlite.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { codecsReady, fixture, paphReady } from "./helpers";
import { putAs112, sqliteStorage } from "./harness/paph";
import { decodeImage, type RgbaImage } from "../src/enrich/decode";
import { art, crop, mirror, paste, swapRedBlue, upscale } from "./harness/images";
import { DEFAULT_MAX_PIXELS, PAPH_X_VERSION, STATE, fingerprint, fitForHash, onWire, stateOf, upscaleFactor, wireFormat, type Fingerprint, type PaphRuntime } from "../src/paph/engine";
import { KIND_BAND, KIND_CODE, KIND_SI, SQL, ShardBusy, ShardStore, Turns, packKeys, rarestFirst, topK, unpackKeys, type ShardStoreOptions } from "../src/paph/shard-store";
import { QUERY_BUDGET, STAGE_BUDGET, budgetLabel, parseBudget, type Budget } from "../src/paph/budget";

const require = createRequire(import.meta.url);

// ---- fixtures ---------------------------------------------------------------------------

let rt: PaphRuntime;
let small: RgbaImage; // "Anonymous", 108×134, transparent backdrop
let second: RgbaImage; // "Space Invader", 224×224
let fpSmall: Fingerprint;
let fpSecond: Fingerprint;
let distractors: Fingerprint[];

beforeAll(async () => {
  await codecsReady();
  rt = await paphReady();
  small = await decodeImage(fixture("small.webp"));
  second = await decodeImage(fixture("second.webp"));
  fpSmall = await fingerprint(small);
  fpSecond = await fingerprint(second);
  distractors = [];
  for (let i = 0; i < 24; i++) distractors.push(await fingerprint(art(1000 + i)));
});

const wires = (f: Fingerprint) => ({ t1: f.t1, t2: f.t2 });
const budget = (over: Partial<Budget> = {}): Budget => ({ ...STAGE_BUDGET, ...over });

describe("the engine", () => {
  it("is the installed @pixagram/paph-x, and names everything a verdict depends on", () => {
    // the package exports no ./package.json: read it next to the module it ships
    const pkg = JSON.parse(readFileSync(join(dirname(require.resolve("@pixagram/paph-x/wasm/paph.wasm")), "..", "package.json"), "utf8"));
    expect(pkg.name).toBe("@pixagram/paph-x");
    expect(PAPH_X_VERSION).toBe(pkg.version);
    expect(rt.identity.package).toBe(pkg.version);
    expect(rt.identity.comparator).toBe(50);
    // 1.2.0's defaults (CHANGELOG): CAL-007 (its name field holds 16 characters), X3 bound to it,
    // SI4 bound to X3, wire 4 — pinned, so a release that moves one is noticed here
    expect(rt.identity.calibration).toBe("CAL-007-PROVISIO");
    expect(rt.identity.calibrationId.slice(0, 16)).toBe("741afad9252f2ccb");
    expect(rt.identity.xprofile).toBe("X3-PROVISIONAL");
    expect(rt.identity.xprofileId).toBe("8af84dd0abb12192");
    expect(rt.identity.siprofileId).toBe("aa8ce6d311f4ce56");
    expect(rt.identity.wire).toBe(4);
    expect(rt.identity.id).toBe(
      `paph-x/${pkg.version} c50 CAL-007-PROVISIO:741afad9252f2ccb X3-PROVISIONAL:8af84dd0abb12192 si:aa8ce6d311f4ce56 k1 w4`,
    );
    // the shipped SI profile is bound to the shipped X profile: SI takes part
    expect(rt.si).not.toBeNull();
    expect(rt.siOff).toBeNull();
    expect(rt.derivation).toContain(`si:${rt.identity.siprofileId}`);
  });

  it("hashes a 3952-byte Tier 1 and a 32 + 40·kp Tier 2, deterministically", async () => {
    expect(fpSmall.t1.length).toBe(3952);
    expect(fpSmall.t2.length).toBe(32 + 40 * fpSmall.kp);
    expect(fpSmall.kp).toBeGreaterThan(50);
    expect([fpSmall.width, fpSmall.height]).toEqual([108, 134]);
    const again = await fingerprint(small);
    expect(Buffer.from(again.t1).equals(Buffer.from(fpSmall.t1))).toBe(true);
    expect(Buffer.from(again.t2).equals(Buffer.from(fpSmall.t2))).toBe(true);
  });

  it("hashes wire 4, and wire 3 on request: Tier 1's byte 4 says which; keypoints and keys are the same in both; a mixed pair is refused", async () => {
    expect([fpSmall.wire, wireFormat(fpSmall.t1)]).toEqual([4, 4]);
    const fp3 = await fingerprint(small, DEFAULT_MAX_PIXELS, 3);
    expect([fp3.wire, wireFormat(fp3.t1), fp3.t1.length, fp3.kp]).toEqual([3, 3, 3952, fpSmall.kp]);
    expect(Buffer.from(fp3.t1).equals(Buffer.from(fpSmall.t1))).toBe(false);
    // what wire 4 does not resample is byte for byte the same (SPEC-W4 §9): Tier 2's keypoint
    // records, and so the index keys, which nominate across a store being re-hashed
    expect(Buffer.from(fp3.t2.subarray(32)).equals(Buffer.from(fpSmall.t2.subarray(32)))).toBe(true);
    expect(rt.engine.indexKeys(wires(fp3))).toEqual(rt.engine.indexKeys(wires(fpSmall)));
    // one image's two formats are never compared: Indeterminate, WIRE_MISMATCH
    const mixed = rt.engine.xcompare(wires(fp3), wires(fpSmall), { profile: rt.x });
    expect(mixed.verdict).toBe("Indeterminate");
    expect(JSON.stringify([mixed.reason, mixed.reasons])).toContain("WIRE_MISMATCH");
    // either format against itself is compared as before
    expect(rt.engine.xcompare(wires(fp3), wires(await fingerprint(mirror(small), DEFAULT_MAX_PIXELS, 3)), { profile: rt.x }).verdict).toMatch(/Copy|Identical/);
    // a verdict reached on wire 3 says so in its identity
    const id = `${rt.identity.id} safe`;
    expect(onWire(id, 4)).toBe(id);
    expect(onWire(id, 3)).toBe(`${rt.identity.id.replace(/ w4$/, " w3")} safe`);
  });

  it("hashes an image within the budget as it is, and brings a larger one inside it", async () => {
    expect(fpSmall.fit).toEqual({ divided: 1, boxed: 1 });
    expect(upscaleFactor(small)).toBe(1);
    const up8 = upscale(small, 8);
    expect(upscaleFactor(up8)).toBe(8);
    expect(fitForHash(up8, DEFAULT_MAX_PIXELS)).toMatchObject({ divided: 1, boxed: 1 });
    // a 20x blow-up is too large even to hand over: divided here, which gives back the original's wire
    const up20 = await fingerprint(upscale(small, 20));
    expect(up20.fit).toEqual({ divided: 20, boxed: 1 });
    expect(Buffer.from(up20.t1).equals(Buffer.from(fpSmall.t1))).toBe(true);
    // a large image that is no blow-up is box-filtered by the smallest factor that fits
    const big = paste(small, upscale(second, 5), 300, 300);
    const fit = fitForHash(big, DEFAULT_MAX_PIXELS);
    expect([fit.divided, fit.boxed]).toEqual([1, 2]);
    expect(fit.img.width * fit.img.height).toBeLessThanOrEqual(DEFAULT_MAX_PIXELS);
  });

  it("reads verdict thresholds by name or number, never Indeterminate or NotCopy", () => {
    expect(stateOf("copy", 0)).toBe(STATE.Copy);
    expect(stateOf("Suspected", 0)).toBe(STATE.Suspected);
    expect(stateOf("4", 0)).toBe(STATE.Identical);
    expect(stateOf("indeterminate", 2)).toBe(2);
    expect(stateOf("notcopy", 3)).toBe(3);
    expect(stateOf(undefined, 3)).toBe(3);
  });
});

describe("budgets", () => {
  it("override a preset by name, clamped, and ignore what they do not know", () => {
    const b = parseBudget(QUERY_BUDGET, "verify:128, deadline_ms:10, si_top:x, nonsense:5, si_min_score:150, rescue:3");
    expect(b).toMatchObject({ verify: 128, deadlineMs: 50, siTop: QUERY_BUDGET.siTop, siMinScore: 150 });
    expect(b).not.toHaveProperty("rescue"); // 1.1.0's ungated second pass is gone (X2's gate drops no copy)
    expect(parseBudget(QUERY_BUDGET, "si_min_score:profile").siMinScore).toBeNull();
    expect(parseBudget(QUERY_BUDGET, undefined)).toEqual(QUERY_BUDGET);
    // the label leaves the deadline out: a different deadline finds the same copies, or says it is partial
    expect(budgetLabel({ ...QUERY_BUDGET, deadlineMs: 1 })).toBe(budgetLabel(QUERY_BUDGET));
    expect(budgetLabel({ ...QUERY_BUDGET, verify: 1 })).not.toBe(budgetLabel(QUERY_BUDGET));
  });

  it("read the rarest keys first, within the df cap and the posting budget", () => {
    const keys = [{ kind: 1, key: 1, n: 50 }, { kind: 1, key: 2, n: 1 }, { kind: 2, key: 3, n: 300 }, { kind: 2, key: 4, n: 7 }, { kind: 1, key: 5, n: 20 }];
    expect(rarestFirst(keys, 256, 30)).toEqual({ keys: [keys[1], keys[3], keys[4]], read: 28 });
    expect(rarestFirst(keys, 256, 1000).keys.map((k) => k.key)).toEqual([2, 4, 5, 1]); // 300 > df cap
    expect(rarestFirst(keys, 256, 0)).toEqual({ keys: [], read: 0 });
  });
});

describe("a shard", () => {
  function open(opts: ShardStoreOptions = {}) {
    const s = sqliteStorage();
    const store = new ShardStore(s.sql, s.transact, rt, opts);
    store.migrate();
    return { ...s, store };
  }
  const count = (db: any, q: string, ...b: unknown[]) => Number((db.prepare(q).get(...b) as any).n);
  const put = (store: ShardStore, postId: number, fp: Fingerprint) => store.put({ postId, contentHash: `h${postId}`, t1: fp.t1, t2: fp.t2 });
  const ids = (r: { checked: Array<{ id: number }> }) => r.checked.map((c) => c.id);
  function corpus(opts: ShardStoreOptions = {}) {
    const o = open(opts);
    put(o.store, 1, fpSmall);
    put(o.store, 2, fpSecond);
    distractors.forEach((fp, i) => put(o.store, 100 + i, fp));
    return o;
  }

  /** A clock that moves only when a slice yields (as on Cloudflare, where it moves only between turns). */
  function steppedClock(stepMs: number) {
    let t = 1_000_000;
    vi.spyOn(Date, "now").mockImplementation(() => t);
    return { pause: async () => void (t += stepMs), slices: () => (t - 1_000_000) / stepMs };
  }
  afterEach(() => vi.restoreAllMocks());

  it("packs a work's keys exactly, and keeps the k best in order", () => {
    const k = { c: [0, 1, 2 ** 53 - 1, 1234567890123], b: [0, 16777216 * 9 + 5], s: [7, 2 ** 27 - 1] };
    expect(unpackKeys(packKeys(k))).toEqual(k);
    expect(() => unpackKeys(packKeys(k).subarray(0, 20))).toThrow(/truncated/);
    const items = [5, 1, 9, 3, 9, 7, 2].map((score, i) => ({ id: 10 - i, score }));
    expect(topK(items, 3, (x) => x.score).map((x) => x.id)).toEqual([6, 8, 5]);
    expect(topK(items, 0, (x) => x.score)).toEqual([]);
    expect(topK(items, 99, (x) => x.score).length).toBe(7);
    // the one-pass path (k ≤ 64) and the sort path agree with a full sort, ties by id
    const many = Array.from({ length: 500 }, (_, i) => ({ id: (i * 7919) % 500, score: (i * 31) % 17 }));
    const sorted = [...many].sort((a, b) => b.score - a.score || a.id - b.id);
    expect(topK(many, 40, (x) => x.score)).toEqual(sorted.slice(0, 40));
    expect(topK(many, 200, (x) => x.score)).toEqual(sorted.slice(0, 200));
  });

  it("indexes a work under its local codes, descriptor bands and PAPH-SI keys, with document frequencies and running counts", () => {
    const { store, db } = open();
    const r = put(store, 1, fpSmall);
    expect(r.replaced).toBe(false);
    expect(r.keys.codes).toBeGreaterThan(50);
    expect(r.keys.bands).toBeGreaterThan(200);
    expect(r.keys.si).toBeGreaterThan(20);
    expect(r.keys.si).toBeLessThanOrEqual(54);
    for (const [kind, n] of [[KIND_CODE, r.keys.codes], [KIND_BAND, r.keys.bands], [KIND_SI, r.keys.si]]) {
      expect(count(db, "SELECT COUNT(*) AS n FROM postings WHERE kind = ?", kind)).toBe(n);
    }
    expect(count(db, "SELECT COUNT(*) AS n FROM df WHERE n <> 1")).toBe(0);
    expect(store.stats()).toMatchObject({ works: 1, current: 1, stale: 0, rehash: 0, postings: { codes: r.keys.codes, bands: r.keys.bands, si: r.keys.si }, queued: { interactive: 0, background: 0 } });
    expect(store.info(1)).toMatchObject({ postId: 1, contentHash: "h1", kp: fpSmall.kp, width: 108, height: 134, wire: 4, current: true });
    const w = store.wires(1)!;
    expect(Buffer.from(w.t1).equals(Buffer.from(fpSmall.t1))).toBe(true);
    expect(Buffer.from(w.t2!).equals(Buffer.from(fpSmall.t2))).toBe(true);
    expect((db.prepare("SELECT length(si_sig) AS n FROM works").get() as any).n).toBe(104);
    // small columns first: the scans of status reads never walk the wires
    const cols = (db.prepare("PRAGMA table_info(works)").all() as any[]).map((c) => c.name);
    expect(cols.indexOf("derivation")).toBeLessThan(cols.indexOf("keys"));
    expect(cols.slice(-2)).toEqual(["t1", "t2"]);
  });

  it("finds the original of each kind of copy among distractors, through more than one nominator", async () => {
    const { store } = corpus();
    const copies: Array<[string, RgbaImage]> = [
      ["mirrored", mirror(small)],
      ["cropped", crop(small, 10, 13, 86, 107)],
      ["upscaled", upscale(small, 2)],
      ["channel-swapped", swapRedBlue(small)],
      ["pasted", paste(small, second, 60, 45)],
    ];
    for (const [name, im] of copies) {
      const r = await store.find(wires(await fingerprint(im)), { budget: budget() });
      const hit = r.matches.find((m) => m.id === 1);
      expect(hit, name).toBeDefined();
      expect(hit!.state, name).toBeGreaterThanOrEqual(STATE.Copy);
      expect(hit!.via.length, `${name}: ${hit!.via}`).toBeGreaterThanOrEqual(1);
      expect(r.matches.filter((m) => m.state >= STATE.Copy && m.id >= 100), name).toEqual([]);
      expect(r.checked, name).toContainEqual({ id: 1, contentHash: "h1" });
      expect(r.stats.partial).toBe(false);
      expect(r.identity).toBe(`${rt.identity.id} safe`);
    }
  });

  it("nominates with PAPH-SI beside the keys: a copy is reached by both", async () => {
    const { store } = corpus();
    const q = wires(await fingerprint(mirror(small)));
    const r = await store.find(q, { budget: budget() });
    expect(r.stats.nominated.si).toBeGreaterThan(0);
    expect(r.stats.postings.si).toBeGreaterThan(0);
    const hit = r.matches.find((m) => m.id === 1)!;
    expect(hit.via).toEqual(expect.arrayContaining(["si"]));
    expect(hit.via.some((v) => v === "codes" || v === "bands")).toBe(true);
    // SI off by budget: the keys still find it
    const keysOnly = await store.find(q, { budget: budget({ siTop: 0 }) });
    expect(keysOnly.stats.nominated.si).toBeUndefined();
    expect(keysOnly.matches.find((m) => m.id === 1)!.state).toBeGreaterThanOrEqual(STATE.Copy);
  });

  it("reads its postings in chunks and merges them into the same nominations", async () => {
    const whole = corpus();
    const sliced = corpus({ chunkPostings: 7 });
    for (const im of [mirror(small), paste(small, second, 60, 45), art(77)]) {
      const q = wires(await fingerprint(im));
      const a = await whole.store.find(q, { budget: budget() });
      const b = await sliced.store.find(q, { budget: budget() });
      expect(b.stats.nominated).toEqual(a.stats.nominated);
      expect(b.stats.postings).toEqual(a.stats.postings);
      expect(b.matches.map((m) => [m.id, m.verdict, m.via.slice().sort()])).toEqual(a.matches.map((m) => [m.id, m.verdict, m.via.slice().sort()]));
    }
  });

  it("is bounded: posting reads and the verify cap", async () => {
    const { store } = corpus();
    const q = wires(await fingerprint(mirror(small)));
    const tight = await store.find(q, { budget: budget({ keyPostings: 40, siPostings: 30 }) });
    expect(tight.stats.postings.keys).toBeLessThanOrEqual(40);
    expect(tight.stats.postings.si).toBeLessThanOrEqual(30);
    expect(tight.stats.skipped.keys).toBeGreaterThan(0);
    const capped = await store.find(q, { budget: budget({ verify: 2 }) });
    expect(capped.stats.verified).toBeLessThanOrEqual(2);
    expect(capped.stats.capped).toBe(capped.stats.nominated.total - 2);
    expect(capped.stats.partial).toBe(false); // the cap is the budget, not a failure
    // the best candidate comes first, so even two comparisons find the original
    expect(capped.matches.map((m) => m.id)).toContain(1);
  });

  it("keeps its deadline on a clock that moves only between slices, and says what it did not compare", async () => {
    const clock = steppedClock(100);
    const { store } = corpus({ pause: clock.pause });
    const q = wires(await fingerprint(mirror(small)));
    const full = await store.find(q, { budget: budget({ deadlineMs: 1_000_000 }) });
    const slices = clock.slices();
    expect(full.stats.partial).toBe(false);
    expect(full.stats.ms).toBe((slices - 1) * 100); // the first pause, before the check, is not its running time
    // a deadline that the nomination's slices alone use up: nothing compared, everything pending
    const late = await store.find(q, { budget: budget({ deadlineMs: 150 }) });
    expect(late.stats.partial).toBe(true);
    expect(late.checked).toEqual([]);
    expect(late.stats.pending).toBe(late.stats.nominated.total - late.stats.capped);
  });

  it("runs concurrent checks one at a time, in arrival order, each answering by its own deadline", async () => {
    const clock = steppedClock(100);
    const { store } = corpus({ pause: clock.pause });
    const q = wires(await fingerprint(mirror(small)));
    const before = clock.slices();
    const solo = await store.find(q, { budget: budget({ deadlineMs: 1_000_000 }) });
    const slices = clock.slices() - before;
    // a deadline (from arrival, waiting included) that covers two checks' turns: of four at once,
    // the first two complete exactly as alone; the last two, whose turn comes too late, do no work
    const deadline = 2 * slices * 100 + 50;
    const runs = await Promise.all([1, 2, 3, 4].map(() => store.find(q, { budget: budget({ deadlineMs: deadline }) })));
    expect(runs.map((r) => r.stats.partial)).toEqual([false, false, true, true]);
    for (const r of runs.slice(0, 2)) {
      expect(r.stats.ms).toBe(solo.stats.ms);
      expect(r.matches.map((m) => [m.id, m.verdict])).toEqual(solo.matches.map((m) => [m.id, m.verdict]));
    }
    for (const r of runs.slice(2)) expect(r.stats).toMatchObject({ verified: 0, postings: { keys: 0, si: 0 } });
    expect(runs[1].stats.waitedMs).toBeGreaterThanOrEqual(solo.stats.ms);
  });

  it("does no work for a caller that has given up, and refuses work past a full queue", async () => {
    const slow = () => new Promise<void>((resolve) => setTimeout(resolve, 5));
    const { store } = corpus({ pause: slow, maxQueued: 1 });
    const q = wires(await fingerprint(mirror(small)));
    const first = store.find(q, { budget: budget() });
    const late = store.find(q, { budget: budget({ deadlineMs: 0 }) }); // queued behind the first; its caller stops waiting at once
    await expect(store.find(q, { budget: budget() })).rejects.toBeInstanceOf(ShardBusy); // the queue (1) is full
    const [a, b] = await Promise.all([first, late]);
    expect(a.stats.partial).toBe(false);
    expect(b.stats).toMatchObject({ partial: true, verified: 0, postings: { keys: 0, si: 0 } });
    expect(b.checked).toEqual([]);
  });

  it("gives the turn on when its holder's request was dropped", async () => {
    const turns = new Turns(1_000, 64, 4, 20);
    await turns.acquire("interactive"); // granted, never released: its request went away
    expect(turns.watching).toBe(false); // no one waits behind it: no timer
    const t0 = Date.now();
    const next = await turns.acquire("interactive");
    expect(next.lane).toBe("interactive");
    expect(Date.now() - t0).toBeGreaterThanOrEqual(20);
    expect(turns.watching).toBe(false);
    turns.release(next);
    expect(turns.queued).toEqual({ interactive: 0, background: 0 });
  });

  it("keeps one lease timer however many wait, and none once they are served", async () => {
    const set = vi.spyOn(globalThis, "setTimeout");
    try {
      const turns = new Turns(1_000, 64, 4, 30_000);
      let t = await turns.acquire("interactive");
      const waiting = Array.from({ length: 50 }, () => turns.acquire("interactive"));
      expect(turns.watching).toBe(true);
      for (const w of waiting) {
        turns.release(t);
        t = await w;
      }
      expect(turns.watching).toBe(false); // the last one holds the turn with no one behind it
      turns.release(t);
      expect(turns.watching).toBe(false);
      expect(set.mock.calls.filter(([, ms]) => (ms ?? 0) > 1_000)).toHaveLength(1);
    } finally {
      set.mockRestore();
    }
  });

  it("lets background work progress under a stream of interactive checks", async () => {
    const slow = () => new Promise<void>((resolve) => setTimeout(resolve, 2));
    const { store } = corpus({ pause: slow, ageMs: 20 });
    const q = wires(await fingerprint(mirror(small)));
    let backgroundDone = false;
    const background = store.find(q, { budget: budget(), priority: "background" }).then((r) => ((backgroundDone = true), r));
    let interactive = 0;
    const stream = async () => {
      while (!backgroundDone && interactive < 400) {
        await store.find(q, { budget: budget() });
        interactive++;
      }
    };
    await Promise.all([stream(), stream(), background]);
    expect(backgroundDone).toBe(true);
    expect(interactive).toBeLessThan(400);
    expect((await background).stats.partial).toBe(false);
  });

  it("lets an interactive check overtake a background check at its next slice", async () => {
    const slow = () => new Promise<void>((resolve) => setTimeout(resolve, 5));
    const { store } = corpus({ pause: slow });
    const q = wires(await fingerprint(mirror(small)));
    const done: string[] = [];
    const background = store.find(q, { budget: budget(), priority: "background" }).then((r) => (done.push("background"), r));
    await new Promise((resolve) => setTimeout(resolve, 6)); // the background check is under way
    const interactive = store.find(q, { budget: budget() }).then((r) => (done.push("interactive"), r));
    const [b, i] = await Promise.all([background, interactive]);
    expect(done).toEqual(["interactive", "background"]);
    expect(b.stats.waitedMs).toBeGreaterThan(0);
    expect(i.stats.waitedMs).toBeLessThan(b.stats.waitedMs); // at most the slice under way
    // the same verdicts either way
    expect(b.matches.map((m) => [m.id, m.verdict])).toEqual(i.matches.map((m) => [m.id, m.verdict]));
    // and a background check arriving while an interactive one runs waits for it before starting
    const order: string[] = [];
    const i2 = store.find(q, { budget: budget() }).then(() => order.push("interactive"));
    const b2 = store.find(q, { budget: budget(), priority: "background" }).then(() => order.push("background"));
    await Promise.all([i2, b2]);
    expect(order).toEqual(["interactive", "background"]);
  });

  it("relies on a gate that drops no copy: what it rejects, the ungated comparison never calls a copy", async () => {
    // why a check makes one gated pass (1.1.0's gate dropped a few channel swaps and palette
    // shuffles, which a second, ungated pass caught; under X2 the gate's exits ask the structural
    // channels first): every work against every query, gated and not
    const { engine, x } = rt;
    const works = [fpSmall, fpSecond, ...distractors];
    const queries = [mirror(small), crop(small, 10, 13, 86, 107), upscale(small, 2), swapRedBlue(small), paste(small, second, 60, 45), swapRedBlue(second)];
    const sides = works.map((w) => engine.xprepare(w.t1, w.t2, { strict: true, profile: x }));
    let dropped = 0;
    try {
      for (const im of queries) {
        const fq = await fingerprint(im);
        const qs = engine.xprepare(fq.t1, fq.t2, { strict: true, profile: x });
        try {
          const gated = engine.xrank(qs, sides, { profile: x, policy: "safe", gate: true, scope: "copy" });
          const open = engine.xrank(qs, sides, { profile: x, policy: "safe", gate: false, scope: "copy" });
          gated.forEach((g, i) => {
            if (g.state !== -1) return expect(g.state, `query ${queries.indexOf(im)}, work ${i}`).toBe(open[i].state);
            dropped++;
            // (it may drop a pair comparator 42 reads Suspected — 8 of 450 such queries on the
            // chain's works, PAPH-X 1.1.2 — never a Copy)
            expect([STATE.Copy, STATE.Identical], `query ${queries.indexOf(im)}, work ${i}: dropped at the gate`).not.toContain(open[i].state);
          });
          // and the copies are found gated
          expect(gated[queries.indexOf(im) < 5 ? 0 : 1].state).toBeGreaterThanOrEqual(STATE.Copy);
        } finally {
          qs.free();
        }
      }
    } finally {
      for (const s of sides) s.free();
    }
    expect(dropped).toBeGreaterThan(0); // the gate does drop pairs: it is a screen
  });

  it("checks a newly indexed work against the shard, never itself", async () => {
    const { store } = corpus();
    const pasted = await fingerprint(paste(small, second, 60, 45));
    put(store, 3, pasted);
    const r = await store.find(wires(pasted), { budget: budget(), exclude: [3] });
    expect(r.matches.map((m) => m.id)).toEqual(expect.arrayContaining([1, 2]));
    expect(ids(r)).not.toContain(3);
  });

  it("verifies candidates nominated elsewhere, tags them, and skips what it does not hold", async () => {
    const { store } = corpus();
    const r = await store.find(wires(fpSmall), { budget: budget(), exclude: [1], extra: [{ id: 2, via: "phash" }, { id: 101, via: "vector" }, { id: 999, via: "phash" }] });
    expect(r.matches.find((m) => m.id === 1)).toBeUndefined();
    expect(ids(r)).not.toContain(999);
    const only = await store.find(wires(fpSmall), { budget: budget(), nominate: false, exclude: [1], extra: [{ id: 2, via: "vector" }, { id: 105, via: "vector" }] });
    expect(only.stats.nominated.total).toBe(2);
    expect(only.stats.verified).toBeLessThanOrEqual(2 + 2);
    const self = await store.find(wires(fpSmall), { budget: budget(), extra: [{ id: 1, via: "phash", rank: 1 }] });
    const hit = self.matches.find((m) => m.id === 1)!;
    expect(hit.verdict).toBe("Identical");
    expect(hit.via).toEqual(expect.arrayContaining(["phash"]));
  });

  it("replaces a work's postings on re-index, and removal leaves no trace", () => {
    const { store, db } = corpus();
    const before = count(db, "SELECT COUNT(*) AS n FROM postings");
    const dfBefore = count(db, "SELECT SUM(n) AS n FROM df");
    expect(dfBefore).toBe(before);
    expect(put(store, 1, fpSmall).replaced).toBe(true);
    expect(count(db, "SELECT COUNT(*) AS n FROM postings")).toBe(before);
    expect(count(db, "SELECT SUM(n) AS n FROM df")).toBe(dfBefore);
    expect(store.remove(1)).toBe(true);
    expect(store.remove(1)).toBe(false);
    expect(count(db, "SELECT COUNT(*) AS n FROM postings WHERE post_id = 1")).toBe(0);
    expect(count(db, "SELECT COUNT(*) AS n FROM df WHERE n <= 0")).toBe(0);
    expect(count(db, "SELECT SUM(n) AS n FROM df")).toBe(count(db, "SELECT COUNT(*) AS n FROM postings"));
    const p = store.stats().postings;
    expect(p.codes + p.bands + p.si).toBe(count(db, "SELECT COUNT(*) AS n FROM postings"));
    expect(store.info(1)).toBeNull();
    expect(store.list(0, 3).map((w) => w.postId)).toEqual([2, 100, 101]);
    expect(store.list(101, 2).map((w) => w.postId)).toEqual([102, 103]);
  });

  it("re-derives keys and signatures from stored wires after a release, and marks refused wires for re-hashing", async () => {
    const { store, db } = corpus();
    const postings = count(db, "SELECT COUNT(*) AS n FROM postings");
    db.prepare("UPDATE works SET derivation = 'paph-x/1.0.0 k1 x:old si:old' WHERE post_id IN (1, 2, 100)").run();
    db.prepare("UPDATE works SET wire = 2 WHERE post_id = 100").run();
    expect(store.stats()).toMatchObject({ stale: 3, rehash: 0 });
    expect(store.info(1)!.current).toBe(false);
    // a stale signature does not score: SI does not nominate the original until re-derived
    const stale = await store.find(wires(fpSmall), { budget: budget(), exclude: [] });
    expect(stale.matches.find((m) => m.id === 1)!.via).not.toContain("si");
    expect(await store.rederive(1)).toEqual({ rederived: 1, rehash: 0, remaining: 2 });
    expect(await store.rederive()).toEqual({ rederived: 1, rehash: 1, remaining: 0 });
    expect(store.stats()).toMatchObject({ stale: 0, rehash: 1 });
    expect(store.info(100)!.derivation).toBe("rehash");
    expect(count(db, "SELECT COUNT(*) AS n FROM postings")).toBe(postings);
    // a wire of another version is never compared
    const r = await store.find(wires(distractors[0]), { budget: budget(), extra: [{ id: 100, via: "phash" }] });
    expect(ids(r)).not.toContain(100);
    expect(r.stats.unreadable).toBeGreaterThanOrEqual(1);
  });

  it("records each work's wire format from its Tier 1, and says whether it holds works on another (hashed before 1.2.0)", async () => {
    const { store } = open();
    put(store, 2, fpSecond);
    expect(store.holdsLegacy()).toBe(false);
    expect(store.stats()).toMatchObject({ wires: { "4": 1 }, legacy: 0 });
    // a work as 1.1.2 stored it: indexed by its keys (the same in both formats), no PAPH-SI
    // signature (SI4 holds wire 4's), waiting for its re-hash
    const r = put(store, 1, await fingerprint(small, DEFAULT_MAX_PIXELS, 3));
    expect(r.keys).toMatchObject({ si: 0 });
    expect(r.keys.codes + r.keys.bands).toBeGreaterThan(0);
    expect(store.info(1)).toMatchObject({ wire: 3, derivation: "rehash", current: false });
    expect(store.holdsLegacy()).toBe(true);
    expect(store.stats()).toMatchObject({ works: 2, current: 1, rehash: 1, stale: 0, wires: { "3": 1, "4": 1 }, legacy: 1 });
    expect(await store.rederive()).toEqual({ rederived: 0, rehash: 0, remaining: 0 });
    // re-hashed: on the current format, signed, current
    expect(put(store, 1, fpSmall).keys.si).toBeGreaterThan(0);
    expect(store.info(1)).toMatchObject({ wire: 4, current: true });
    expect(store.holdsLegacy()).toBe(false);
    expect(store.stats()).toMatchObject({ wires: { "4": 2 }, legacy: 0 });
    store.remove(1);
    store.remove(2);
    expect(store.stats()).toMatchObject({ works: 0, wires: {}, legacy: 0 });
  });

  it("compares each stored side with the query of its own format while works wait for their re-hash (SPEC-W4 §9)", async () => {
    const { store } = open();
    // the original as 1.1.2 stored it (wire 3); another original and the distractors re-hashed
    put(store, 1, await fingerprint(small, DEFAULT_MAX_PIXELS, 3));
    put(store, 2, fpSecond);
    distractors.forEach((fp, i) => put(store, 100 + i, fp));
    const q = wires(await fingerprint(mirror(small)));
    const twin = wires(await fingerprint(mirror(small), DEFAULT_MAX_PIXELS, 3));
    // without the twin, the wire-3 original is nominated (by its keys) but never compared
    const alone = await store.find(q, { budget: budget() });
    expect(alone.stats.legacy).toBe(1);
    expect(alone.stats.unreadable).toBe(1);
    expect(ids(alone)).not.toContain(1);
    expect(alone.matches.find((m) => m.id === 1)).toBeUndefined();
    // with it, the original is compared on wire 3, everything else on wire 4
    const both = await store.find(q, { budget: budget(), twin });
    const hit = both.matches.find((m) => m.id === 1)!;
    expect(hit).toMatchObject({ wire: 3 });
    expect(hit.state).toBeGreaterThanOrEqual(STATE.Copy);
    expect(hit.via.some((v) => v === "codes" || v === "bands")).toBe(true);
    expect(both.checked).toContainEqual({ id: 1, contentHash: "h1", wire: 3 });
    expect(both.checked.filter((c) => c.id !== 1).every((c) => c.wire === undefined)).toBe(true);
    expect(both.stats.unreadable).toBe(0);
    expect(both.matches.filter((m) => m.state >= STATE.Copy && m.id >= 100)).toEqual([]);
    expect(both.matches.filter((m) => m.id !== 1).every((m) => m.wire === 4)).toBe(true);
    // a copy of the re-hashed original: wire 4, twin or not
    const q2 = await store.find(wires(await fingerprint(mirror(second))), { budget: budget(), twin: wires(await fingerprint(mirror(second), DEFAULT_MAX_PIXELS, 3)) });
    expect(q2.matches.find((m) => m.id === 2)).toMatchObject({ wire: 4 });
    // a query in wire 3 (a work not re-hashed yet, re-checked live from its stored wires): only
    // wire-3 works are compared, and PAPH-SI (wire 4's signatures) is not probed
    const old = await store.find(twin, { budget: budget(), extra: [{ id: 2, via: "phash" }] });
    expect(old.matches.find((m) => m.id === 1)).toMatchObject({ wire: 3 });
    expect(old.stats.postings.si).toBe(0);
    expect(ids(old)).not.toContain(2);
    expect(old.stats.unreadable).toBeGreaterThanOrEqual(1);
    // a twin the engine refuses leaves the wire-3 works unread, and the check goes on
    const broken = { t1: twin.t1.slice(), t2: twin.t2 };
    broken.t1[100] ^= 0xff;
    const refused = await store.find(q, { budget: budget(), twin: broken });
    expect(ids(refused)).not.toContain(1);
    expect(refused.stats.unreadable).toBe(1);
    expect(refused.stats.partial).toBe(false);
  });

  it("re-derives a store 1.1.2 left: its wire-3 entries keep their keys, lose their SI3 signatures and postings, and wait for their re-hash", async () => {
    const { store, db } = open();
    putAs112(db, rt, 1, "h1", wires(await fingerprint(small, DEFAULT_MAX_PIXELS, 3)));
    putAs112(db, rt, 2, "h2", wires(await fingerprint(second, DEFAULT_MAX_PIXELS, 3)));
    put(store, 3, distractors[0]);
    const si = () => count(db, "SELECT COUNT(*) AS n FROM postings WHERE kind = 3 AND post_id IN (1, 2)");
    const keyRows = () => count(db, "SELECT COUNT(*) AS n FROM postings WHERE kind <> 3 AND post_id IN (1, 2)");
    const keysBefore = keyRows();
    expect(si()).toBeGreaterThan(0);
    expect(store.stats()).toMatchObject({ works: 3, current: 1, stale: 2, rehash: 0, wires: { "3": 2, "4": 1 }, legacy: 2 });
    // a wire-4 query reads their SI3 postings, which SI4's probes share a key space with, for nothing
    const q = wires(await fingerprint(mirror(small)));
    const twin = wires(await fingerprint(mirror(small), DEFAULT_MAX_PIXELS, 3));
    const before = await store.find(q, { budget: budget(), twin });
    expect(await store.rederive()).toEqual({ rederived: 0, rehash: 2, remaining: 0 });
    expect(si()).toBe(0);
    expect(keyRows()).toBe(keysBefore);
    expect(count(db, "SELECT COUNT(*) AS n FROM works WHERE si_sig IS NOT NULL AND post_id IN (1, 2)")).toBe(0);
    expect(store.stats()).toMatchObject({ stale: 0, rehash: 2, legacy: 2 });
    // the running counts still agree with the postings
    expect(store.stats().postings.si).toBe(count(db, "SELECT COUNT(*) AS n FROM postings WHERE kind = 3"));
    // still reached by their keys, compared through the twin, and fewer SI rows read for it
    const after = await store.find(q, { budget: budget(), twin });
    expect(after.matches.find((m) => m.id === 1)).toMatchObject({ wire: 3 });
    expect(after.stats.postings.si).toBeLessThan(before.stats.postings.si);
  });

  it("does not send a work to be re-hashed for an error that is not about its wires", async () => {
    const { store, db } = corpus();
    db.prepare("UPDATE works SET derivation = 'old' WHERE post_id = 2").run();
    const spy = vi.spyOn(store, "put").mockImplementation(() => {
      throw new Error("out of memory");
    });
    await expect(store.rederive()).rejects.toThrow(/out of memory/);
    spy.mockRestore();
    expect(store.info(2)!.derivation).toBe("old");
    expect(await store.rederive()).toMatchObject({ rederived: 1, rehash: 0, remaining: 0 });
  });

  it("runs every statement of a check and of an index write on the primary keys", () => {
    const { db } = corpus();
    const keys = JSON.stringify({ c: [1, 2, 3], b: [4, 5], s: [6] });
    const chosen = JSON.stringify([[1, 1, 3], [2, 4, 1]]);
    const plans: Record<string, unknown[]> = {
      insertPostings: [keys, 1],
      incDf: [keys],
      deletePostings: [keys, 1],
      decDf: [keys],
      dropDf: [keys],
      countPostings: [1, 2, 3],
      dfOf: [keys],
      sumKeys: [chosen, "[]"],
      reachSi: [JSON.stringify([[6, 2]]), "[]"],
      loadSigs: ["[1, 2]", "d"],
      loadWires: ["[1, 2]"],
      legacy: [4],
    };
    // every statement is held to it
    expect(Object.keys(plans).sort()).toEqual(Object.keys(SQL).sort());
    for (const [name, args] of Object.entries(plans)) {
      const plan = (db.prepare(`EXPLAIN QUERY PLAN ${(SQL as Record<string, string>)[name]}`).all(...(args as any[])) as Array<{ detail: string }>).map((r) => r.detail);
      const scans = plan.filter((d) => /^SCAN (postings|df|works|meta|p|d)\b/.test(d));
      expect(scans, `${name}: ${plan.join(" | ")}`).toEqual([]);
    }
  });
});
