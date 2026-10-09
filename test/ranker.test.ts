import { describe, expect, it } from "vitest";
import {
  DEFAULT_WEIGHTS,
  freshnessScore,
  isColorLed,
  mergeWeights,
  qualityScore,
  rankCandidates,
  relevance,
  toneScore,
  type RankContext,
  type RankInput,
} from "../src/search/ranker";
import { planQuery } from "../src/search/planner";
import { reciprocalRankFusion } from "../src/search/rrf";

const NOW = Date.UTC(2026, 9, 4) / 1000;
const DAY = 86400;

function item(id: number, over: Partial<RankInput> = {}): RankInput {
  return {
    id, type: "artwork", author: "someone", title: "", description: "", tags: [], aiCaption: null, aiTags: [],
    created: NOW - 30 * DAY, netVotes: 0, phash: null, buckets: [], brightness: 0.5, contrast: 0.2, saturation: 0.5,
    colorfulness: 50, monochrome: false, colorCount: 64, paletteEntropy: 4, orientation: "square", backgroundName: null,
    bm25: null, bm25Leg: null, cosImage: [null], cosText: [null], concepts: new Map(), ...over,
  };
}

function ctx(q: string, over: Partial<RankContext> = {}): RankContext {
  const plan = planQuery(q, { mode: "search", authors: new Set(["laura", "retro"]), now: NOW });
  return {
    plan, weights: DEFAULT_WEIGHTS, calibration: null, now: NOW, voteRate: 1,
    variants: [{ text: plan.text, factor: 1, imageNorm: { median: 0.05, scale: 0.02 }, textNorm: null, cosImageFloor: null, cosTextFloor: null }],
    ...over,
  };
}

describe("ranker features", () => {
  it("an exact title beats a partial one by far (title is squared)", () => {
    const r = rankCandidates([item(1, { title: "Ride Into the Sunset" }), item(2, { title: "Sunset" }), item(3, { title: "Ride" })], ctx("ride into the sunset"));
    expect(r.map((x) => x.id)).toEqual([1, 2, 3]);
    expect(r[0].features.title).toBe(1);
    expect(r[2].features.title).toBeLessThan(0.25);
  });

  it("semantic score is a sigmoid of the z-score against the background sample", () => {
    const c = ctx("a cat");
    const [hi, mid, lo] = rankCandidates([item(1, { cosImage: [0.05 + 0.02 * 5] }), item(2, { cosImage: [0.05 + 0.02 * 2.5] }), item(3, { cosImage: [0.05] })], c);
    expect(hi.features.sem).toBeCloseTo(1 / (1 + Math.exp(-1.5 * (5 - 2.5))), 6); // slope 1.5, z0 2.5
    expect(mid.features.sem).toBeCloseTo(0.5, 5); // z = semZ0
    expect(lo.features.sem).toBeLessThan(0.03);
    expect(hi.raw?.zImage).toBeCloseTo(5, 5);
  });

  it("a candidate the image leg did not return gets the leg's floor, not zero information", () => {
    const c = ctx("a cat");
    c.variants[0].cosImageFloor = 0.05 + 0.02 * 2.5;
    const [r] = rankCandidates([item(1, { cosImage: [null] })], c);
    expect(r.features.sem).toBeCloseTo(0.5, 5);
  });

  it("a spelling-corrected variant counts its factor", () => {
    const c = ctx("elodrado");
    c.variants.push({ text: "eldorado", factor: 0.9, imageNorm: { median: 0.05, scale: 0.02 }, textNorm: null, cosImageFloor: null, cosTextFloor: null });
    const [r] = rankCandidates([item(1, { title: "Eldorado", cosImage: [0.05, 0.05 + 0.02 * 8] })], c);
    expect(r.features.title).toBeCloseTo(0.81, 5); // (0.9 · 1)²
    expect(r.features.sem).toBeCloseTo(0.9, 2);
  });

  it("author: explicit or the whole query = 1, a bare name among other words = 0.3", () => {
    expect(rankCandidates([item(1, { author: "laura" })], ctx("cat by laura"))[0].features.author).toBe(1);
    expect(rankCandidates([item(1, { author: "laura" })], ctx("laura"))[0].features.author).toBe(1);
    expect(rankCandidates([item(1, { author: "retro" })], ctx("retro car"))[0].features.author).toBe(0.3);
  });

  it("colour-led queries rank by palette share, not by the word in a title", () => {
    const c = ctx("blue");
    expect(isColorLed(c.plan)).toBe(true);
    expect(isColorLed(ctx("blue dragon").plan)).toBe(false);
    expect(isColorLed(ctx("black and white").plan)).toBe(true);
    const r = rankCandidates(
      [item(1, { title: "Blue", buckets: [{ name: "red", weight: 0.9 }], bm25: -5, bm25Leg: "and" }), item(2, { title: "Sea", buckets: [{ name: "blue", weight: 0.7 }] })],
      c,
    );
    expect(r[0].id).toBe(2);
  });

  it("tones are continuous versions of the SQL predicates", () => {
    expect(toneScore("greyscale", { colorfulness: 2 } as any)).toBe(1);
    expect(toneScore("greyscale", { colorfulness: 30 } as any)).toBe(0);
    expect(toneScore("dark", { brightness: 0.1 } as any)).toBe(1);
    expect(toneScore("light", { brightness: 0.1 } as any)).toBe(0);
    expect(toneScore("monochrome", { monochrome: true, colorfulness: 60 } as any)).toBe(1);
    expect(toneScore("minimal", { colorCount: 6, paletteEntropy: 4 } as any)).toBe(1);
  });

  it("concept feature is the mean confidence over the query's concepts", () => {
    const c = ctx("cat on the beach");
    const r = rankCandidates([item(1, { concepts: new Map([["cat", 0.95], ["beach", 0.65]]) })], c);
    expect(r[0].features.concept).toBeCloseTo(0.8, 5);
  });
});

describe("ranker scoring", () => {
  it("quality is a bounded Bayesian vote rate: popularity breaks near-ties, never relevance", () => {
    // two votes in the first hour vs forty over a week
    const early = qualityScore(2, NOW - 3600, NOW, 1);
    const steady = qualityScore(40, NOW - 7 * DAY, NOW, 1);
    expect(steady).toBeGreaterThan(early);
    expect(qualityScore(0, NOW - 30 * DAY, NOW, 1)).toBeGreaterThan(0);
    const w = DEFAULT_WEIGHTS;
    expect(w.quality[1] / w.quality[0]).toBeLessThan(1.25);
    const r = rankCandidates([item(1, { title: "Cat", netVotes: 0 }), item(2, { title: "Cats and dogs", netVotes: 500 })], ctx("cat"));
    expect(r[0].id).toBe(1);
  });

  it("freshness decays exponentially", () => {
    expect(freshnessScore(NOW, NOW, 180)).toBe(1);
    expect(freshnessScore(NOW - 180 * DAY, NOW, 180)).toBeCloseTo(Math.exp(-1), 6);
  });

  it("near-duplicates step down below their better copy", () => {
    const a = item(1, { title: "Cat", phash: "ffff0000ffff0000" });
    const b = item(2, { title: "Cat", phash: "ffff0000ffff0001" });
    const c = item(3, { title: "Cat", phash: "0000ffff0000ffff" });
    const r = rankCandidates([a, b, c], ctx("cat"));
    expect(r.find((x) => x.id === 2)!.duplicateOf).toBe(1);
    expect(r.map((x) => x.id)).toEqual([1, 3, 2]);
  });

  it("relevance applies the colour-led multipliers", () => {
    const f = { title: 1, lexical: 0, coverage: 0, tag: 0, author: 0, concept: 0, sem: 0, sem_txt: 0, color: 1, tone: 0, orientation: 0 };
    expect(relevance(f, DEFAULT_WEIGHTS, false)).toBeCloseTo(0.4 + 0.2, 6);
    expect(relevance(f, DEFAULT_WEIGHTS, true)).toBeCloseTo(0.4 * 0.3 + 0.2 * 5, 6);
  });

  it("mergeWeights keeps defaults for missing or invalid values", () => {
    const w = mergeWeights({ w: { title: 0.7, sem: "x" }, quality: [1.2, 0.8], colorBoost: 3, version: "t" });
    expect(w.w.title).toBe(0.7);
    expect(w.w.sem).toBe(DEFAULT_WEIGHTS.w.sem);
    expect(w.quality).toEqual(DEFAULT_WEIGHTS.quality); // inverted bounds are rejected
    expect(w.colorBoost).toBe(3);
    expect(w.version).toBe("t");
    expect(mergeWeights(null)).toEqual(DEFAULT_WEIGHTS);
  });

  it("v2's fusion is still available for comparison", () => {
    const fused = reciprocalRankFusion([{ name: "fts", ids: [1, 2] }, { name: "vec", ids: [2, 3] }]);
    expect(fused[0].id).toBe(2);
  });
});
