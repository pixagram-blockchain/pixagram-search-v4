// Cheap global image descriptors on the native artwork (no upscaling), used by the planner
// ("dark", "black and white", "portrait orientation", "minimal"), by the ranker (colour/tone
// match) and by visual similarity (geometry and histograms). A 430 px artwork takes a few ms.

import { nearestNamed, parseHex, rgbToLab, type Cluster } from "./color";
import type { RgbaImage } from "./decode";

export type Orientation = "portrait" | "landscape" | "square";

export interface ImageFeatures {
  brightness: number; // mean luminance of opaque pixels, 0..1
  contrast: number; // std of luminance, 0..0.5
  saturation: number; // mean HSV saturation of opaque pixels, 0..1
  colorfulness: number; // Hasler–Süsstrunk M on the 0..255 scale (<10 ≈ greyscale, >60 very colourful)
  monochrome: boolean; // greyscale, or nearly all colour in one narrow hue band (sepia, duotone)
  hueConcentration: number; // share of chromatic pixels in the dominant 30° hue window
  edgeDensity: number; // share of neighbouring pixel pairs that differ visibly
  symmetryX: number; // left/right mirror agreement over the foreground, 0..1
  symmetryY: number;
  foregroundShare: number; // opaque non-background pixels / all pixels
  centerX: number; // foreground centre of mass, 0..1
  centerY: number;
  aspect: number; // width / height
  orientation: Orientation;
  paletteEntropy: number; // bits over the ΔE-clustered palette
  backgroundName: string | null; // named colour of the backdrop, or "transparent"
  backgroundShare: number;
  lab: { L: number; a: number; b: number }; // mean CIELAB of counted pixels
  hue12: number[]; // saturation-weighted hue histogram, 12 bins of 30°, sums to 1 (or all 0)
  lum8: number[]; // luminance histogram, 8 bins, sums to 1
}

export function orientationOf(width: number, height: number): Orientation {
  const r = width / Math.max(1, height);
  return r > 1.15 ? "landscape" : r < 1 / 1.15 ? "portrait" : "square";
}

const ALPHA = 8;
const EDGE_L1 = 48; // RGB L1 distance that counts as a visible edge / mirror mismatch

export function computeFeatures(
  img: RgbaImage,
  ctx: { backgroundHex: string | null; transparentShare: number; clusters: Cluster[] },
): ImageFeatures {
  const { width: W, height: H, data } = img;
  const n = W * H;
  const bg = ctx.backgroundHex ? parseHex(ctx.backgroundHex) : null;
  const isBg = (i: number) => (bg ? data[i] === bg[0] && data[i + 1] === bg[1] && data[i + 2] === bg[2] : false);

  let opaque = 0;
  let lumSum = 0;
  let lumSq = 0;
  let satSum = 0;
  let rgSum = 0, rgSq = 0, ybSum = 0, ybSq = 0;
  let fg = 0, cx = 0, cy = 0, bgCount = 0;
  const hue = new Float64Array(36); // 10° bins for the concentration measure
  let chromatic = 0;
  const lum8 = new Float64Array(8);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      if (data[i + 3] < ALPHA) continue;
      opaque++;
      const r = data[i], g = data[i + 1], b = data[i + 2];
      const L = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
      lumSum += L;
      lumSq += L * L;
      lum8[Math.min(7, Math.floor(L * 8))] += 1;
      const mx = Math.max(r, g, b);
      const mn = Math.min(r, g, b);
      const s = mx === 0 ? 0 : (mx - mn) / mx;
      satSum += s;
      const rg = r - g;
      const yb = 0.5 * (r + g) - b;
      rgSum += rg;
      rgSq += rg * rg;
      ybSum += yb;
      ybSq += yb * yb;
      if (s > 0.15 && mx > 40) {
        let h: number;
        const d = mx - mn;
        if (mx === r) h = ((g - b) / d) % 6;
        else if (mx === g) h = (b - r) / d + 2;
        else h = (r - g) / d + 4;
        h = (h * 60 + 360) % 360;
        hue[Math.floor(h / 10) % 36] += s;
        chromatic += s;
      }
      if (isBg(i)) {
        bgCount++;
      } else {
        fg++;
        cx += x;
        cy += y;
      }
    }
  }

  const op = Math.max(1, opaque);
  const meanL = lumSum / op;
  const std = (sq: number, sum: number) => Math.sqrt(Math.max(0, sq / op - (sum / op) ** 2));
  const colorfulness = Math.hypot(std(rgSq, rgSum), std(ybSq, ybSum)) + 0.3 * Math.hypot(rgSum / op, ybSum / op);

  // dominant 30° window (3 adjacent 10° bins, circular)
  let bestWin = 0;
  for (let k = 0; k < 36; k++) bestWin = Math.max(bestWin, hue[k] + hue[(k + 1) % 36] + hue[(k + 2) % 36]);
  const hueConcentration = chromatic > 0 ? bestWin / chromatic : 1;
  const saturation = satSum / op;
  const monochrome = colorfulness < 8 || (hueConcentration > 0.9 && saturation < 0.35);

  const hue12 = new Array<number>(12).fill(0);
  if (chromatic > 0) for (let k = 0; k < 36; k++) hue12[Math.floor(k / 3)] += hue[k] / chromatic;

  // Edges and mirror symmetry, over opaque pixels.
  let pairs = 0, edges = 0;
  let symXPairs = 0, symXOk = 0, symYPairs = 0, symYOk = 0;
  const l1 = (i: number, j: number) => Math.abs(data[i] - data[j]) + Math.abs(data[i + 1] - data[j + 1]) + Math.abs(data[i + 2] - data[j + 2]);
  const op1 = (i: number) => data[i + 3] >= ALPHA;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4;
      const oi = op1(i);
      if (x + 1 < W) {
        const j = i + 4;
        if (oi && op1(j)) {
          pairs++;
          if (l1(i, j) > EDGE_L1) edges++;
        }
      }
      if (y + 1 < H) {
        const j = i + W * 4;
        if (oi && op1(j)) {
          pairs++;
          if (l1(i, j) > EDGE_L1) edges++;
        }
      }
      if (x < W >> 1) {
        const j = (y * W + (W - 1 - x)) * 4;
        const oj = op1(j);
        if (oi || oj) {
          symXPairs++;
          if (oi === oj && (!oi || l1(i, j) <= EDGE_L1)) symXOk++;
        }
      }
      if (y < H >> 1) {
        const j = ((H - 1 - y) * W + x) * 4;
        const oj = op1(j);
        if (oi || oj) {
          symYPairs++;
          if (oi === oj && (!oi || l1(i, j) <= EDGE_L1)) symYOk++;
        }
      }
    }
  }

  // Mean Lab and entropy from the clustered palette (weights are shares of counted pixels).
  let wsum = 0;
  const lab = { L: 0, a: 0, b: 0 };
  let entropy = 0;
  for (const c of ctx.clusters) {
    wsum += c.weight;
    lab.L += c.lab.L * c.weight;
    lab.a += c.lab.a * c.weight;
    lab.b += c.lab.b * c.weight;
  }
  if (wsum > 0) {
    lab.L /= wsum;
    lab.a /= wsum;
    lab.b /= wsum;
    for (const c of ctx.clusters) {
      const p = c.weight / wsum;
      if (p > 0) entropy -= p * Math.log2(p);
    }
  }

  let backgroundName: string | null = null;
  let backgroundShare = 0;
  if (ctx.transparentShare >= 0.2) {
    backgroundName = "transparent";
    backgroundShare = ctx.transparentShare;
  } else if (bg) {
    backgroundName = nearestNamed(rgbToLab(bg[0], bg[1], bg[2])).name;
    backgroundShare = bgCount / Math.max(1, n);
  }

  const r4 = (v: number) => Math.round(v * 10000) / 10000;
  return {
    brightness: r4(meanL),
    contrast: r4(std(lumSq, lumSum)),
    saturation: r4(saturation),
    colorfulness: Math.round(colorfulness * 100) / 100,
    monochrome,
    hueConcentration: r4(hueConcentration),
    edgeDensity: r4(pairs ? edges / pairs : 0),
    symmetryX: r4(symXPairs ? symXOk / symXPairs : 1),
    symmetryY: r4(symYPairs ? symYOk / symYPairs : 1),
    foregroundShare: r4(fg / Math.max(1, n)),
    centerX: r4(fg ? cx / fg / Math.max(1, W - 1) : 0.5),
    centerY: r4(fg ? cy / fg / Math.max(1, H - 1) : 0.5),
    aspect: r4(W / Math.max(1, H)),
    orientation: orientationOf(W, H),
    paletteEntropy: Math.round(entropy * 1000) / 1000,
    backgroundName,
    backgroundShare: r4(backgroundShare),
    lab: { L: Math.round(lab.L * 100) / 100, a: Math.round(lab.a * 100) / 100, b: Math.round(lab.b * 100) / 100 },
    hue12: hue12.map(r4),
    lum8: Array.from(lum8, (v) => r4(v / op)),
  };
}
