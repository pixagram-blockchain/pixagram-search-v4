import { beforeAll, describe, expect, it } from "vitest";
import { DatabaseSync } from "node:sqlite";
import { bandNeighbours, bandProbes, dhash, halves, hamming, MAX_INDEXED_RADIUS, phash, phashBands, sqlHamming } from "../src/enrich/phash";
import { computeFeatures, orientationOf } from "../src/enrich/features";
import { computeStats } from "../src/enrich/stats";
import { decodeImage, type RgbaImage } from "../src/enrich/decode";
import { codecsReady, fixture } from "./helpers";

beforeAll(() => codecsReady());

// deterministic PRNG for reproducible random hashes
function rng(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s ^ (s >>> 15), 2246822507) + 0x9e3779b9) >>> 0) / 4294967296);
}
const hex64 = (r: () => number) => [0, 1, 2, 3].map(() => Math.floor(r() * 65536).toString(16).padStart(4, "0")).join("");
function flip(hash: string, bits: number[]): string {
  let [hi, lo] = halves(hash);
  for (const b of bits) {
    if (b < 32) hi = (hi ^ (1 << (31 - b))) >>> 0;
    else lo = (lo ^ (1 << (63 - b))) >>> 0;
  }
  return hi.toString(16).padStart(8, "0") + lo.toString(16).padStart(8, "0");
}

describe("multi-index hashing", () => {
  it("bandNeighbours enumerates exactly the values within the radius", () => {
    const n2 = bandNeighbours(0x1234, 2);
    expect(n2.length).toBe(1 + 16 + 120);
    expect(new Set(n2).size).toBe(n2.length);
    expect(bandNeighbours(0x1234, 3).length).toBe(1 + 16 + 120 + 560);
    const pop = (x: number) => x.toString(2).replace(/0/g, "").length;
    for (const v of n2) expect(pop(v ^ 0x1234)).toBeLessThanOrEqual(2);
  });

  it("band probes find every hash within the radius (pigeonhole), compared with brute force", () => {
    const r = rng(7);
    const q = hex64(r);
    // a corpus of random hashes plus planted neighbours at every distance 0..20
    const corpus: string[] = Array.from({ length: 3000 }, () => hex64(r));
    for (let d = 0; d <= 20; d++) {
      const bits = new Set<number>();
      while (bits.size < d) bits.add(Math.floor(r() * 64));
      corpus.push(flip(q, [...bits]));
    }
    for (const radius of [0, 4, 8, 12, MAX_INDEXED_RADIUS]) {
      const probes = bandProbes(q, radius);
      const viaBands = new Set(
        corpus.filter((h) => {
          const b = phashBands(h);
          return probes.some((p) => p.values.includes(b[p.band]));
        }),
      );
      const truth = corpus.filter((h) => hamming(q, h) <= radius);
      for (const h of truth) expect(viaBands.has(h), `radius ${radius} missed ${h} at ${hamming(q, h)}`).toBe(true);
    }
    expect(() => bandProbes(q, MAX_INDEXED_RADIUS + 1)).toThrow();
  });

  it("the SQL popcount equals the JS Hamming distance (no multiplication, no float overflow)", () => {
    const db = new DatabaseSync(":memory:");
    db.exec("CREATE TABLE h (id INTEGER PRIMARY KEY, hex TEXT, hi INTEGER, lo INTEGER)");
    const ins = db.prepare("INSERT INTO h (hex, hi, lo) VALUES (?, ?, ?)");
    const r = rng(11);
    const hashes = [...Array.from({ length: 300 }, () => hex64(r)), "0000000000000000", "ffffffffffffffff", "8000000000000001"];
    for (const x of hashes) {
      const [hi, lo] = halves(x);
      ins.run(x, hi, lo);
    }
    for (const q of [hashes[0], "ffffffffffffffff", "0000000000000000", "0123456789abcdef"]) {
      const rows = db.prepare(`SELECT hex, ${sqlHamming("hi", "lo", q)} AS d FROM h`).all() as Array<{ hex: string; d: number }>;
      for (const row of rows) expect(row.d, `${q} vs ${row.hex}`).toBe(hamming(q, row.hex));
    }
    db.close();
  });

  it("phash and dhash are stable, and an upscaled copy stays close on both", async () => {
    const img = await decodeImage(fixture("small.webp"));
    const big = upscale3(img);
    expect(hamming(phash(img), phash(big))).toBeLessThanOrEqual(6);
    expect(hamming(dhash(img), dhash(big))).toBeLessThanOrEqual(8);
    expect(dhash(img)).toMatch(/^[0-9a-f]{16}$/);
    const other = await decodeImage(fixture("second.webp"));
    expect(hamming(dhash(img), dhash(other))).toBeGreaterThan(10);
  });
});

function upscale3(img: RgbaImage): RgbaImage {
  const W = img.width * 3, H = img.height * 3, data = new Uint8Array(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data.set(img.data.subarray(((Math.floor(y / 3) * img.width + Math.floor(x / 3)) * 4), ((Math.floor(y / 3) * img.width + Math.floor(x / 3)) * 4) + 4), (y * W + x) * 4);
  return { width: W, height: H, data };
}

function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number, number]): RgbaImage {
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
  return { width: w, height: h, data };
}
const feats = (img: RgbaImage) => {
  const st = computeStats(img);
  return computeFeatures(img, { backgroundHex: st.backgroundHex, transparentShare: st.transparentShare, clusters: st.clusters });
};

describe("image features", () => {
  it("greyscale art has near-zero colourfulness and counts as monochrome", () => {
    const f = feats(paint(32, 32, (x, y) => { const v = ((x >> 2) + (y >> 2)) % 2 ? 230 : 20; return [v, v, v, 255]; }));
    expect(f.colorfulness).toBeLessThan(1);
    expect(f.monochrome).toBe(true);
    expect(f.saturation).toBeLessThan(0.01);
    expect(f.contrast).toBeGreaterThan(0.3);
  });

  it("a sepia (one-hue) image is monochrome, a rainbow is colourful", () => {
    // the classic sepia matrix applied to a grey ramp
    const sepia = feats(paint(32, 32, (x) => { const v = 20 + 5 * x; return [Math.min(255, Math.round(1.351 * v)), Math.min(255, Math.round(1.203 * v)), Math.round(0.937 * v), 255]; }));
    expect(sepia.monochrome).toBe(true);
    expect(sepia.colorfulness).toBeGreaterThan(8); // not greyscale: the tone is in the hue
    const hues: Array<[number, number, number]> = [[230, 30, 30], [30, 200, 40], [30, 60, 230], [240, 220, 20], [200, 30, 200], [20, 210, 210]];
    const rainbow = feats(paint(36, 36, (x) => [...hues[Math.floor(x / 6)], 255] as [number, number, number, number]));
    expect(rainbow.monochrome).toBe(false);
    expect(rainbow.colorfulness).toBeGreaterThan(60);
  });

  it("brightness, orientation, symmetry and the foreground on a flat backdrop", () => {
    const dark = feats(paint(40, 20, () => [10, 10, 30, 255]));
    expect(dark.brightness).toBeLessThan(0.1);
    expect(dark.orientation).toBe("landscape");
    // a centred symmetric figure on a flat blue background
    const fig = feats(paint(30, 30, (x, y) => (Math.abs(x - 14.5) < 5 && Math.abs(y - 14.5) < 8 ? [250, 200, 40, 255] : [30, 60, 200, 255])));
    expect(fig.backgroundName).toBe("blue");
    expect(fig.symmetryX).toBeGreaterThan(0.95);
    expect(fig.centerX).toBeCloseTo(0.5, 1);
    expect(fig.foregroundShare).toBeGreaterThan(0.1);
    expect(fig.foregroundShare).toBeLessThan(0.3);
    expect(orientationOf(100, 200)).toBe("portrait");
    expect(orientationOf(100, 110)).toBe("square");
  });
});
