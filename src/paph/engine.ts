// PAPH-X — the perceptual-hash engine behind copy detection (@pixagram/paph-x, WebAssembly).
//
// One instance per isolate, shared by the Worker (hashing uploads and new artworks, pair reports)
// and by the PaphShard Durable Objects that run in it (indexing and comparing). The module is
// registered by the entry point: Wrangler turns `import x from "….wasm"` into a compiled
// WebAssembly.Module (Workers cannot compile WebAssembly from bytes at run time); tests register
// one compiled from node_modules.
//
// Everything a verdict depends on — the package, the comparator-42 calibration, the X (route)
// profile, the SI profile, the key derivation, the wire version, the policy — is named by
// `identity`, stored with every verdict, every indexed work and every cached answer, so that a
// new release (1.1.2 moved the X profile to X2 and the SI profile to SI3) is a dependency bump
// followed by a re-derivation, never a silent change of what stored results mean.
//
// 1.2.0 changed the wire itself: hash() writes format 4 (docs/SPEC-W4-paph-wire4.md in the
// package), whose sampling commutes with mirrors and quarter turns, under CAL-007, X3 and SI4. A
// wire-3 side and a wire-4 side are never compared, so stored works are re-hashed (the identity
// moved: the stage re-checks every work, and hashes again a work whose stored wires are wire 3).
// Until a shard holds no wire-3 work, a query is hashed in both formats and each stored side is
// compared with the query of its own format (SPEC-W4 §9, shard-store.ts and copies.ts).

import {
  init,
  KEYS_VERSION,
  STATES,
  WIRE_VERSION,
  type Engine,
  type HashResult,
  type Policy,
  type SIProfile,
  type XProfile,
} from "@pixagram/paph-x/wasm";
import type { RgbaImage } from "../enrich/decode";

export type { Engine, Policy, SIProfile, XProfile, XRankRecord, XSide, XReport, RouteReading, SIPlan } from "@pixagram/paph-x/wasm";
export { KEYS_VERSION, STATES, WIRE_VERSION };

/**
 * The installed @pixagram/paph-x version. Bump it with the dependency (a test compares it with
 * node_modules): it is part of every identity, so results of another release never pass for
 * this one's.
 */
export const PAPH_X_VERSION = "1.2.0";

let mod: WebAssembly.Module | null = null;
let artefacts: { base?: Uint8Array; x?: Uint8Array; si?: Uint8Array } = {};
let ready: Promise<PaphRuntime> | null = null;

/** Register the compiled module. Safe to call repeatedly; the first call wins. */
export function setPaphModule(m: WebAssembly.Module): void {
  if (!mod) mod = m;
}

/**
 * Run other profile artefacts than the package's defaults (a comparator-42 .pcal, an X profile, an
 * SI profile), e.g. a provisional X2 shipped as a file. Call before the first use; the identity
 * follows, so stored works are re-derived and verdicts re-checked (README-V4, "Copy detection").
 */
export function setPaphProfiles(p: { base?: Uint8Array; x?: Uint8Array; si?: Uint8Array }): void {
  if (ready) throw new Error("paph: profiles must be set before the engine is first used");
  artefacts = { ...p };
}

/** Verdict states, as PAPH-X numbers them (6 = NotCopy: copy scope, not lifted above Related). */
export const STATE = { Unrelated: 0, Related: 1, Suspected: 2, Copy: 3, Identical: 4, Indeterminate: 5, NotCopy: 6 } as const;

/** "copy" | "Copy" | "3" → 3; anything else → the fallback. Only Unrelated…Identical are thresholds. */
export function stateOf(v: string | undefined | null, fallback: number): number {
  if (!v) return fallback;
  const n = Number(v);
  if (Number.isInteger(n) && n >= 0 && n <= 4) return n;
  const i = STATES.findIndex((s) => s.toLowerCase() === v.trim().toLowerCase());
  return i >= 0 && i <= 4 ? i : fallback;
}

export interface Identity {
  /** one line naming everything below: stored with verdicts and works, part of cache keys */
  id: string;
  package: string;
  /** PAPH-X's comparator number, and the comparator-42 calibration it reproduces */
  comparator: number;
  calibration: string;
  calibrationId: string;
  /** the X (route) profile */
  xprofile: string;
  xprofileId: string;
  /** the SI profile, or null when the shipped one is bound to another X profile (SI then sits out) */
  siprofileId: string | null;
  keys: number;
  wire: number;
}

export interface PaphRuntime {
  engine: Engine;
  x: XProfile;
  si: SIProfile | null;
  /** why SI sits out, when it does */
  siOff: string | null;
  identity: Identity;
  /** what a stored work's derived data (keys, SI signature) depends on: re-derive when it changes */
  derivation: string;
}

/** The engine, its profiles and identity, created once per isolate. */
export function paphRuntime(): Promise<PaphRuntime> {
  if (!ready) {
    if (!mod) return Promise.reject(new Error("paph: wasm module not registered (setPaphModule)"));
    const m = mod;
    ready = init(m).then((engine) => build(engine));
    ready.catch(() => {
      ready = null;
    });
  }
  return ready;
}

/** The engine alone (hashing). */
export async function paphEngine(): Promise<Engine> {
  return (await paphRuntime()).engine;
}

function build(engine: Engine): PaphRuntime {
  const x = engine.xprofile({ base: artefacts.base, x: artefacts.x });
  const status = x.status();
  // a mismatched or unsupported X profile makes every comparison Indeterminate: refuse to run
  // rather than replace stored verdicts with nothing
  if (status !== "ok") throw new Error(`paph: the X profile cannot be used (status ${status})`);
  let si: SIProfile | null = null;
  let siOff: string | null = null;
  const xid = x.id();
  try {
    const sp = engine.siprofile(artefacts.si);
    if (sp.xid() === xid) si = sp;
    else {
      siOff = `the SI profile ${sp.id().slice(0, 16)} is bound to the X profile ${sp.xid().slice(0, 16)}, not ${xid.slice(0, 16)}`;
      sp.free();
    }
  } catch (e) {
    siOff = `no SI profile: ${e instanceof Error ? e.message : String(e)}`;
  }
  // the comparator and calibration, read from a report: a 16×16 two-colour checkerboard
  const data = new Uint8Array(16 * 16 * 4);
  for (let i = 0; i < 256; i++) data.set((i >> 4) % 2 === (i & 15) % 2 ? [255, 255, 255, 255] : [20, 40, 160, 255], i * 4);
  const fp = engine.hash({ data, width: 16, height: 16 });
  const r = engine.xcompare(fp, fp, { profile: x });
  const siId = si ? si.id().slice(0, 16) : null;
  const identity: Identity = {
    id: [
      `paph-x/${PAPH_X_VERSION}`,
      `c${r.comparator}`,
      `${r.calibration}:${r.calibrationId.slice(0, 16)}`,
      `${r.xcalibration}:${xid.slice(0, 16)}`,
      `si:${siId ?? "-"}`,
      `k${KEYS_VERSION}`,
      `w${WIRE_VERSION}`,
    ].join(" "),
    package: PAPH_X_VERSION,
    comparator: r.comparator,
    calibration: r.calibration,
    calibrationId: r.calibrationId,
    xprofile: r.xcalibration,
    xprofileId: xid.slice(0, 16),
    siprofileId: siId,
    keys: KEYS_VERSION,
    wire: WIRE_VERSION,
  };
  return { engine, x, si, siOff, identity, derivation: `paph-x/${PAPH_X_VERSION} k${KEYS_VERSION} x:${xid.slice(0, 16)} si:${siId ?? "-"}` };
}

/** The identity a verdict carries: the engine's, and the policy the cascade ran under. */
export function verdictIdentity(rt: Pick<PaphRuntime, "identity">, policy: Policy): string {
  return `${rt.identity.id} ${policy}`;
}

/** 1.0–1.1's wire format: stored works hashed before 1.2.0, until they are re-hashed. */
export const WIRE_3 = 3;

/** The format a wire is in: Tier 1's byte 4 (3 or 4; 0 for a Tier 1 too short to say). */
export function wireFormat(t1: Uint8Array): number {
  return t1.length > 4 ? t1[4] : 0;
}

/**
 * The identity of a verdict reached on wire-`wire` sides: `identity` with its wire named as it
 * was. A stored work not re-hashed yet is compared with the query's wire-3 twin (SPEC-W4 §9): the
 * verdict says so, counts as stale until that work's own re-check after its re-hash replaces it,
 * and never passes for one reached on the current wire.
 */
export function onWire(identity: string, wire: number): string {
  return wire === WIRE_VERSION ? identity : identity.replace(/(^| )w\d+(?= |$)/, `$1w${wire}`);
}

// ---- hashing ----------------------------------------------------------------------------------

export interface Fingerprint {
  t1: Uint8Array;
  t2: Uint8Array;
  /** the wire format (3 or 4) */
  wire: number;
  width: number;
  height: number;
  kp: number;
  /** how the image was reduced before hashing (1, 1 = it was not: the wire is the image's own) */
  fit: { divided: number; boxed: number };
}

/**
 * Pixel budget for hashing in a Worker. The hasher's working memory is ~60 bytes per pixel and
 * WebAssembly memory never shrinks, while an isolate has 128 MB for everything: 768² (~37 MB
 * peak) leaves room for the codecs and the JavaScript heap. Pixagram artworks are typically
 * under 430 px on the long side, far inside it.
 */
export const DEFAULT_MAX_PIXELS = 768 * 768;

/** k such that the image is an exact k-fold nearest blow-up: gcd(w, h, every colour-change position). */
export function upscaleFactor(img: RgbaImage): number {
  const { width: w, height: h } = img;
  const d = img.data.byteOffset % 4 === 0 ? img.data : img.data.slice();
  const px = new Uint32Array(d.buffer, d.byteOffset, w * h);
  const gcd = (a: number, b: number): number => {
    while (b) [a, b] = [b, a % b];
    return a;
  };
  let g = gcd(w, h);
  for (let y = 0; y < h && g > 1; y++) {
    const row = y * w;
    for (let x = 1; x < w && g > 1; x++) if (px[row + x] !== px[row + x - 1]) g = gcd(g, x);
    if (y > 0 && g > 1) {
      for (let x = 0; x < w; x++) {
        if (px[row + x] !== px[row - w + x]) {
          g = gcd(g, y);
          break;
        }
      }
    }
  }
  return g;
}

/** Every k-th pixel of every k-th row: the exact inverse of a k-fold blow-up. */
function nearestDivide(img: RgbaImage, k: number): RgbaImage {
  const w = img.width / k, h = img.height / k;
  const d = img.data.byteOffset % 4 === 0 ? img.data : img.data.slice();
  const src = new Uint32Array(d.buffer, d.byteOffset, img.width * img.height);
  const out = new Uint32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = src[y * k * img.width + x * k];
  return { width: w, height: h, data: new Uint8Array(out.buffer) };
}

/** Integer box filter: each output pixel is the rounded mean of an f×f block (clipped at the edges). */
function boxDown(img: RgbaImage, f: number): RgbaImage {
  const w = Math.ceil(img.width / f), h = Math.ceil(img.height / f);
  const out = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    const y1 = Math.min(img.height, (y + 1) * f);
    for (let x = 0; x < w; x++) {
      const x1 = Math.min(img.width, (x + 1) * f);
      let r = 0, g = 0, b = 0, a = 0, n = 0;
      for (let yy = y * f; yy < y1; yy++) {
        for (let xx = x * f; xx < x1; xx++) {
          const i = (yy * img.width + xx) * 4;
          r += img.data[i];
          g += img.data[i + 1];
          b += img.data[i + 2];
          a += img.data[i + 3];
          n++;
        }
      }
      const o = (y * w + x) * 4, half = n >> 1;
      out[o] = ((r + half) / n) | 0;
      out[o + 1] = ((g + half) / n) | 0;
      out[o + 2] = ((b + half) / n) | 0;
      out[o + 3] = ((a + half) / n) | 0;
    }
  }
  return { width: w, height: h, data: out };
}

/**
 * Bring an image inside the budget. Within it — after dividing out an exact blow-up, which the
 * hasher does itself — the image is hashed as it is, so its wire is the one anyone computes from
 * the same pixels. Beyond it, the blow-up is divided here, and what is still too large is box-
 * filtered by the smallest integer factor that fits (deterministic: a re-upload of the same large
 * image hashes to the same wire).
 */
export function fitForHash(img: RgbaImage, maxPixels: number): { img: RgbaImage; divided: number; boxed: number } {
  const k = upscaleFactor(img);
  const effective = (img.width / k) * (img.height / k);
  if (effective <= maxPixels && img.width * img.height <= 4 * maxPixels) return { img, divided: 1, boxed: 1 };
  let out = k > 1 ? nearestDivide(img, k) : img;
  let boxed = 1;
  if (out.width * out.height > maxPixels) {
    boxed = Math.max(2, Math.ceil(Math.sqrt((out.width * out.height) / maxPixels)));
    while (Math.ceil(out.width / boxed) * Math.ceil(out.height / boxed) > maxPixels) boxed++;
    out = boxDown(out, boxed);
  }
  return { img: out, divided: k, boxed };
}

/**
 * Hash decoded RGBA pixels: Tier 1 (3952 B) and Tier 2 (32 + 40·kp B), byte-identical to the
 * JavaScript engine for any image within the budget (see fitForHash for the rest). In the
 * current format (4), or with `wire` 3 in 1.0–1.1's, byte for byte: the twin of a query that is
 * compared with stored works not re-hashed yet.
 */
export async function fingerprint(img: RgbaImage, maxPixels = DEFAULT_MAX_PIXELS, wire?: 3 | 4): Promise<Fingerprint> {
  const fit = fitForHash(img, Math.max(1, maxPixels));
  const engine = await paphEngine();
  const px = { data: fit.img.data, width: fit.img.width, height: fit.img.height };
  const r: HashResult = wire && wire !== WIRE_VERSION ? engine.hash(px, { wire }) : engine.hash(px);
  return { t1: r.t1, t2: r.t2, wire: wireFormat(r.t1), width: r.width, height: r.height, kp: r.kpCount, fit: { divided: fit.divided, boxed: fit.boxed } };
}
