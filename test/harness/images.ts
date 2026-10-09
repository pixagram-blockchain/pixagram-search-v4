// Image transforms for the copy-detection tests (RGBA, row-major): the copies an artist meets —
// mirrored, cropped, upscaled, pasted into another work, channel-swapped — and procedural pixel
// art to serve as distractors.

import type { RgbaImage } from "../../src/enrich/decode";

const px = (im: RgbaImage, x: number, y: number) => im.data.subarray((y * im.width + x) * 4, (y * im.width + x) * 4 + 4);

export function mirror(im: RgbaImage): RgbaImage {
  const out = new Uint8Array(im.data.length);
  for (let y = 0; y < im.height; y++) for (let x = 0; x < im.width; x++) out.set(px(im, x, y), (y * im.width + (im.width - 1 - x)) * 4);
  return { width: im.width, height: im.height, data: out };
}

export function crop(im: RgbaImage, x0: number, y0: number, w: number, h: number): RgbaImage {
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) out.set(im.data.subarray(((y0 + y) * im.width + x0) * 4, ((y0 + y) * im.width + x0 + w) * 4), y * w * 4);
  return { width: w, height: h, data: out };
}

export function upscale(im: RgbaImage, k: number): RgbaImage {
  const w = im.width * k, h = im.height * k, out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out.set(px(im, (x / k) | 0, (y / k) | 0), (y * w + x) * 4);
  return { width: w, height: h, data: out };
}

/** Composite `guest` (alpha ≥ 128) onto `host` at (ox, oy). */
export function paste(guest: RgbaImage, host: RgbaImage, ox: number, oy: number): RgbaImage {
  const out = host.data.slice();
  for (let y = 0; y < guest.height; y++) for (let x = 0; x < guest.width; x++) {
    const p = px(guest, x, y);
    if (p[3] >= 128) out.set(p, ((oy + y) * host.width + ox + x) * 4);
  }
  return { width: host.width, height: host.height, data: out };
}

export function swapRedBlue(im: RgbaImage): RgbaImage {
  const out = im.data.slice();
  for (let i = 0; i < out.length; i += 4) [out[i], out[i + 2]] = [out[i + 2], out[i]];
  return { width: im.width, height: im.height, data: out };
}

/** Procedural pixel art — blocks and dithers in an 8-colour palette — as distractors. */
export function art(seed: number): RgbaImage {
  let s = seed;
  const r = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const w = 64 + ((r() * 6) | 0) * 24, h = 64 + ((r() * 6) | 0) * 24;
  const pal = Array.from({ length: 8 }, () => [(r() * 256) | 0, (r() * 256) | 0, (r() * 256) | 0]);
  const data = new Uint8Array(w * h * 4);
  const bayer = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const c = pal[bayer[((y & 3) << 2) | (x & 3)] < (y * 16) / h ? 0 : 1];
    data.set([c[0], c[1], c[2], 255], (y * w + x) * 4);
  }
  for (let i = 0; i < 14; i++) {
    const bw = 4 + ((r() * 18) | 0), bh = 4 + ((r() * 18) | 0), bx = (r() * (w - bw)) | 0, by = (r() * (h - bh)) | 0;
    const c = pal[2 + ((r() * 6) | 0)];
    for (let y = by; y < by + bh; y++) for (let x = bx; x < bx + bw; x++) if ((x + y) % 7 || i % 2) data.set([c[0], c[1], c[2], 255], (y * w + x) * 4);
  }
  return { width: w, height: h, data };
}
