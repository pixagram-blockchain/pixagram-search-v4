// Perceptual hashes for near-duplicate detection, and the multi-index layout that makes the
// duplicate search exact.
//
//   pHash  64-bit DCT hash (v2, unchanged): robust to rescaling and re-encoding
//   dHash  64-bit gradient hash: cheap second opinion, sensitive to different structure
//
// Both are computed on the native image composited over white.
//
// Index layout: four 16-bit bands. If d(x, y) <= r then some band differs in at most floor(r/4)
// bits (pigeonhole over 4 bands), so looking up, for every band, the stored value and all values
// within floor(r/4) bits of it finds every hash within distance r. v2's eight 8-bit chunks with a
// single equality lookup only guaranteed r <= 7, while /duplicates defaulted to r = 8.

import type { RgbaImage } from "./decode";

const N = 32; // DCT size
const K = 8; // low-frequency block

/** Grayscale w×h via box filter (area average), transparent pixels composited over white. */
function downscaleGray(img: RgbaImage, w: number, h: number): Float64Array {
  const out = new Float64Array(w * h);
  const { width, height, data } = img;
  for (let oy = 0; oy < h; oy++) {
    const y0 = Math.floor((oy * height) / h);
    const y1 = Math.max(y0 + 1, Math.floor(((oy + 1) * height) / h));
    for (let ox = 0; ox < w; ox++) {
      const x0 = Math.floor((ox * width) / w);
      const x1 = Math.max(x0 + 1, Math.floor(((ox + 1) * width) / w));
      let sum = 0;
      let cnt = 0;
      for (let y = y0; y < y1 && y < height; y++) {
        for (let x = x0; x < x1 && x < width; x++) {
          const i = (y * width + x) * 4;
          const a = data[i + 3] / 255;
          const r = data[i] * a + 255 * (1 - a);
          const g = data[i + 1] * a + 255 * (1 - a);
          const b = data[i + 2] * a + 255 * (1 - a);
          sum += 0.299 * r + 0.587 * g + 0.114 * b;
          cnt++;
        }
      }
      out[oy * w + ox] = cnt ? sum / cnt : 255;
    }
  }
  return out;
}

const COS = (() => {
  const t = new Float64Array(N * N);
  for (let u = 0; u < N; u++) for (let x = 0; x < N; x++) t[u * N + x] = Math.cos(((2 * x + 1) * u * Math.PI) / (2 * N));
  return t;
})();

/** Top-left K×K block of the 2D DCT-II (separable). */
function dctLow(px: Float64Array): Float64Array {
  const rows = new Float64Array(N * K); // rows[y*K + u]
  for (let y = 0; y < N; y++) {
    for (let u = 0; u < K; u++) {
      let s = 0;
      for (let x = 0; x < N; x++) s += px[y * N + x] * COS[u * N + x];
      rows[y * K + u] = s;
    }
  }
  const out = new Float64Array(K * K); // out[v*K + u]
  for (let v = 0; v < K; v++) {
    for (let u = 0; u < K; u++) {
      let s = 0;
      for (let y = 0; y < N; y++) s += rows[y * K + u] * COS[v * N + y];
      out[v * K + u] = s;
    }
  }
  return out;
}

function bitsToHex(bits: ArrayLike<number>): string {
  let hi = 0;
  let lo = 0;
  for (let i = 0; i < 64; i++) {
    if (i < 32) hi = (hi << 1) | bits[i];
    else lo = (lo << 1) | bits[i];
  }
  return (hi >>> 0).toString(16).padStart(8, "0") + (lo >>> 0).toString(16).padStart(8, "0");
}

/** pHash, 16 hex chars (identical to v2). */
export function phash(img: RgbaImage): string {
  const px = downscaleGray(img, N, N);
  const dct = dctLow(px);
  const ac = Array.from(dct.subarray(1)).sort((a, b) => a - b);
  const median = ac.length % 2 ? ac[(ac.length - 1) / 2] : (ac[ac.length / 2 - 1] + ac[ac.length / 2]) / 2;
  const bits = new Uint8Array(64);
  for (let i = 0; i < 64; i++) bits[i] = dct[i] > median ? 1 : 0;
  return bitsToHex(bits);
}

/** dHash (horizontal gradient on a 9×8 thumbnail), 16 hex chars. */
export function dhash(img: RgbaImage): string {
  const px = downscaleGray(img, 9, 8);
  const bits = new Uint8Array(64);
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) bits[y * 8 + x] = px[y * 9 + x + 1] > px[y * 9 + x] ? 1 : 0;
  return bitsToHex(bits);
}

function popcount32(x: number): number {
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

export function hamming(a: string, b: string): number {
  return popcount32((parseInt(a.slice(0, 8), 16) ^ parseInt(b.slice(0, 8), 16)) >>> 0) + popcount32((parseInt(a.slice(8, 16), 16) ^ parseInt(b.slice(8, 16), 16)) >>> 0);
}

/** The two unsigned 32-bit halves of a 64-bit hex hash (stored as phash_hi / phash_lo). */
export function halves(hash: string): [number, number] {
  return [parseInt(hash.slice(0, 8), 16) >>> 0, parseInt(hash.slice(8, 16), 16) >>> 0];
}

// ---- multi-index hashing -------------------------------------------------------------------

export const BANDS = 4;
export const BAND_BITS = 16;
/** Largest radius the band index answers exactly with a bounded number of lookups (≤ 3 flipped bits per band). */
export const MAX_INDEXED_RADIUS = 15;

/** Four 16-bit bands, most significant first. */
export function phashBands(hash: string): number[] {
  const out: number[] = [];
  for (let i = 0; i < 16; i += 4) out.push(parseInt(hash.slice(i, i + 4), 16));
  return out;
}

/** Every 16-bit value within `radius` bits of `v` (including v): Σ C(16, k), k ≤ radius. */
export function bandNeighbours(v: number, radius: number): number[] {
  const out: number[] = [v];
  const rec = (cur: number, start: number, left: number) => {
    for (let b = start; b < BAND_BITS; b++) {
      const next = cur ^ (1 << b);
      out.push(next);
      if (left > 1) rec(next, b + 1, left - 1);
    }
  };
  if (radius > 0) rec(v, 0, radius);
  return out;
}

/** Per-band lookup lists that together are guaranteed to contain every hash within `radius`. */
export function bandProbes(hash: string, radius: number): Array<{ band: number; values: number[] }> {
  if (radius > MAX_INDEXED_RADIUS) throw new Error(`radius ${radius} > ${MAX_INDEXED_RADIUS}: use the full scan`);
  const per = Math.floor(radius / BANDS);
  return phashBands(hash).map((v, band) => ({ band, values: bandNeighbours(v, per) }));
}

/**
 * SQLite expression for the Hamming distance between the stored halves (columns hiCol, loCol) and
 * a query hash, without multiplication (SQLite turns integer overflow into floating point):
 * XOR as (a | b) - (a & b), then a SWAR popcount on each 32-bit half.
 */
export function sqlHamming(hiCol: string, loCol: string, hash: string): string {
  const [qh, ql] = halves(hash);
  const pop = (x: string) => {
    const s1 = `((${x} & 1431655765) + ((${x} >> 1) & 1431655765))`; // 0x55555555
    const s2 = `((${s1} & 858993459) + ((${s1} >> 2) & 858993459))`; // 0x33333333
    const s3 = `((${s2} & 252645135) + ((${s2} >> 4) & 252645135))`; // 0x0f0f0f0f
    const s4 = `((${s3} & 16711935) + ((${s3} >> 8) & 16711935))`; // 0x00ff00ff
    return `((${s4} & 65535) + ((${s4} >> 16) & 65535))`;
  };
  const xor = (col: string, q: number) => `((${col} | ${q}) - (${col} & ${q}))`;
  return `(${pop(xor(hiCol, qh))} + ${pop(xor(loCol, ql))})`;
}
