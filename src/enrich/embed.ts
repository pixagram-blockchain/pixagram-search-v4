// Client for the SigLIP embedding service (hf/app.py on a Hugging Face Space, or hf/handler.py on an
// Inference Endpoint — same contract). HF_EMBED_URL is the full URL of the embed route, e.g.
// https://<owner>-<space>.hf.space/embed
// One model, two towers: images for the pipeline, texts for queries. Vectors are L2-normalised
// by the handler so cosine similarity is a dot product.
//
// v3: the Space also reports the model's logit scale and bias. SigLIP was trained with a sigmoid
// loss, so sigmoid(scale · cos + bias) is a calibrated "this text matches this image" probability,
// which the ranker and /ask use instead of raw cosines whose scale means nothing on its own.

import type { Env } from "../env";
import { int, now } from "../env";

export interface Calibration {
  logit_scale: number; // exp(logit_scale) of the model, ~110 for SigLIP base
  logit_bias: number; // ~ -12 .. -17
}

export interface EmbedResult {
  model: string;
  dim: number;
  embeddings: number[][];
  calibration: Calibration | null;
}

/**
 * Who a failed call says something about. "input": the endpoint read the request and refused this
 * image or text (HTTP 400, 413, 422): trying the same post again is likely to fail again. "endpoint":
 * everything else (asleep, starting, unreachable, overloaded, a 5xx, a wrong token, another model or
 * patch budget): the post is fine and will embed once the endpoint is. The jobs table counts only the
 * first kind toward the sweeper's give-up cap (src/db/posts.ts).
 */
export type EmbedBlame = "endpoint" | "input";

export class EmbedUnavailable extends Error {
  constructor(msg: string, public readonly retryable: boolean, public readonly blame: EmbedBlame = "endpoint") {
    super(msg);
    this.name = "EmbedUnavailable";
  }
}

/** True when an error says the embedding endpoint was the problem, not the post (see EmbedBlame). */
export function endpointFault(e: unknown): boolean {
  return e instanceof EmbedUnavailable && e.blame === "endpoint";
}

/** Measured with hf/siglip.py (transformers 5, Oct 2026): model.logit_scale.exp(), model.logit_bias. */
export const KNOWN_CALIBRATION: Record<string, Calibration> = {
  "google/siglip2-base-patch16-256": { logit_scale: 112.901, logit_bias: -16.7718 },
  "google/siglip2-base-patch16-naflex": { logit_scale: 115.3019, logit_bias: -16.777 },
};

export function embeddingEnabled(env: Env): boolean {
  return !!env.HF_EMBED_URL;
}

const SLOTS = new WeakMap<object, { at: number; n: number }>();
const SLOTS_KEY = "space:slots";

/**
 * How many embeddings the Space computes at once: its /health "concurrency" (one per CPU, see
 * hf/siglip.py), an older Space's "cpus", else 2. Read every ten minutes (isolate, then KV), so a
 * Space moved to bigger hardware is used fully without a deploy.
 */
export async function spaceSlots(env: Env): Promise<number> {
  if (!env.HF_EMBED_URL) return 1;
  const hit = SLOTS.get(env.DB);
  if (hit && Date.now() - hit.at < 10 * 60_000) return hit.n;
  let n = Math.trunc(Number(await env.CACHE.get(SLOTS_KEY).catch(() => null))) || 0;
  if (n <= 0) {
    try {
      const headers: Record<string, string> = env.HF_TOKEN ? { authorization: `Bearer ${env.HF_TOKEN}` } : {};
      const res = await fetch(env.HF_EMBED_URL.replace(/\/embed\/?$/, "") + "/health", { headers, signal: AbortSignal.timeout(5_000) });
      const h = res.ok ? ((await res.json()) as { concurrency?: unknown; cpus?: unknown }) : null;
      n = Math.trunc(Number(h?.concurrency ?? h?.cpus)) || 0;
    } catch {
      n = 0;
    }
    n = n > 0 ? Math.min(n, 32) : 2;
    await env.CACHE.put(SLOTS_KEY, String(n), { expirationTtl: 600 }).catch(() => {});
  }
  SLOTS.set(env.DB, { at: Date.now(), n });
  return n;
}

// ---- the Space's state -----------------------------------------------------------------------------

/** What the embedding endpoint's /health says, checked against this Worker's configuration. */
export interface SpaceStatus {
  /** HF_EMBED_URL is set */
  configured: boolean;
  /** /health answered with JSON */
  reachable: boolean;
  /** reachable, its weights loaded, and serving the model, dimension and patch budget the indexes hold */
  ready: boolean;
  model: string | null;
  dim: number | null;
  max_num_patches: number | null;
  /** embeddings computed at once, PyTorch threads per embedding and the CPUs it may use (hf/siglip.py) */
  concurrency: number | null;
  threads: number | null;
  cpus: number | null;
  uptime_s: number | null;
  /** why it is not ready; null when it is */
  problem: string | null;
  /** when it was checked (unix seconds), and how long /health took */
  checked_at: number;
  ms: number | null;
}

const STATUS = new WeakMap<object, SpaceStatus>();
const STATUS_KEY = "space:status";
/** One check stands for a minute (KV keeps a key at least 60 s). */
const STATUS_TTL_S = 60;

function healthUrl(env: Env): string {
  return (env.HF_EMBED_URL ?? "").replace(/\/embed\/?$/, "") + "/health";
}

function blankStatus(env: Env): SpaceStatus {
  return {
    configured: !!env.HF_EMBED_URL, reachable: false, ready: false, model: null, dim: null, max_num_patches: null,
    concurrency: null, threads: null, cpus: null, uptime_s: null, problem: env.HF_EMBED_URL ? null : "HF_EMBED_URL not configured", checked_at: now(), ms: null,
  };
}

/** How long a /health check may take. */
const PROBE_TIMEOUT_MS = 5_000;

/** The last check if it is less than `maxAgeS` old (this isolate, then KV); null otherwise. Never asks the Space. */
export async function cachedSpaceStatus(env: Env, maxAgeS = STATUS_TTL_S): Promise<SpaceStatus | null> {
  if (!env.HF_EMBED_URL) return blankStatus(env);
  const t = now();
  const hit = STATUS.get(env.DB);
  if (hit && t - hit.checked_at < maxAgeS) return hit;
  if (maxAgeS <= 0) return null;
  const kv = (await env.CACHE.get(STATUS_KEY, "json").catch(() => null)) as SpaceStatus | null;
  if (kv && typeof kv.checked_at === "number" && t - kv.checked_at < maxAgeS) {
    STATUS.set(env.DB, kv);
    return kv;
  }
  return null;
}

/**
 * The Space's state from its /health, checked at most once a minute per isolate and shared through
 * KV, so a monitor polling /healthz or a burst of requests costs the Space one request a minute.
 * A Space that is asleep, building or restarting answers with an HTML page or not at all: not
 * reachable. `maxAgeS: 0` checks now. Only these full checks are shared; a failed call noted by
 * noteEndpointFault stays in its isolate.
 */
export async function spaceStatus(env: Env, o: { maxAgeS?: number } = {}): Promise<SpaceStatus> {
  const cached = await cachedSpaceStatus(env, o.maxAgeS ?? STATUS_TTL_S);
  if (cached) return cached;
  const s = await probeSpace(env, PROBE_TIMEOUT_MS);
  STATUS.set(env.DB, s);
  await env.CACHE.put(STATUS_KEY, JSON.stringify(s), { expirationTtl: STATUS_TTL_S }).catch(() => {});
  return s;
}

async function probeSpace(env: Env, timeoutMs: number): Promise<SpaceStatus> {
  const s = blankStatus(env);
  const started = Date.now();
  // hf/app.py on a Space always serves /health as JSON: anything else there means the Space is not
  // serving it (asleep, building, private without a valid token, deleted). Elsewhere (an Inference
  // Endpoint running hf/handler.py) /health may be missing or plain text: nothing to check, the
  // calls will tell, and /healthz's backlog check still catches vectors that do not arrive.
  let ours = true;
  try {
    ours = /\.hf\.space$/i.test(new URL(healthUrl(env)).hostname);
  } catch {
    // not a URL: the fetch below says so
  }
  let h: Record<string, unknown>;
  try {
    const headers: Record<string, string> = { accept: "application/json" };
    if (env.HF_TOKEN) headers.authorization = `Bearer ${env.HF_TOKEN}`;
    const res = await fetch(healthUrl(env), { headers, signal: AbortSignal.timeout(timeoutMs) });
    const text = await res.text();
    s.ms = Date.now() - started;
    if (!res.ok) {
      if (!ours && (res.status === 404 || res.status === 405)) return { ...s, reachable: true, ready: true };
      return { ...s, problem: ours && res.status === 404 ? "/health HTTP 404 (a private Space without a valid token, or no such Space)" : `/health HTTP ${res.status}` };
    }
    let parsed: unknown = null;
    try {
      parsed = JSON.parse(text);
    } catch {
      // not JSON: handled below, like JSON that is not an object
    }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      if (!ours) return { ...s, reachable: true, ready: true };
      return { ...s, problem: "/health answered with a page, not JSON (asleep, building or restarting)" };
    }
    h = parsed as Record<string, unknown>;
  } catch (e) {
    const name = (e as { name?: string })?.name;
    s.ms = Date.now() - started;
    s.problem = name === "TimeoutError" || name === "AbortError" ? `/health timed out after ${timeoutMs / 1000} s` : `unreachable: ${e instanceof Error ? e.message : String(e)}`;
    return s;
  }
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  Object.assign(s, {
    reachable: true, model: typeof h.model === "string" ? h.model : null, dim: num(h.dim), max_num_patches: num(h.max_num_patches),
    concurrency: num(h.concurrency), threads: num(h.threads), cpus: num(h.cpus), uptime_s: num(h.uptime_s),
  });
  // the same checks as every /embed reply (call() below): vectors of another model, dimension or
  // patch budget would not belong in the indexes
  const dim = int(env.EMBED_DIM, 768);
  const patches = int(env.EMBED_PATCHES, 0);
  if (h.ok === false) s.problem = "/health says ok: false";
  else if (h.ready === false) s.problem = "loading the model";
  else if (env.EMBED_MODEL && s.model && s.model !== "stub" && s.model !== env.EMBED_MODEL) s.problem = `serves ${s.model} but EMBED_MODEL is ${env.EMBED_MODEL}`;
  else if (s.dim && s.dim !== dim) s.problem = `dimension ${s.dim} but EMBED_DIM is ${dim}`;
  else if (patches > 0 && s.max_num_patches !== null && s.max_num_patches !== patches) s.problem = `${s.max_num_patches} patches per image but EMBED_PATCHES is ${patches}`;
  s.ready = s.problem === null;
  return s;
}

/**
 * A call just failed on the endpoint's side (endpointFault): until the next check, this isolate
 * treats the Space as not ready, so the requests behind it fall back at once instead of each
 * waiting for the same timeout.
 */
export function noteEndpointFault(env: Env, e: unknown): void {
  if (!endpointFault(e) || !env.HF_EMBED_URL) return;
  const prev = STATUS.get(env.DB) ?? blankStatus(env);
  STATUS.set(env.DB, { ...prev, ready: false, problem: (e as Error).message, checked_at: now() });
}

/** Query texts wait at most this long (a hung Space must not stall /search); images, for indexing, longer. */
const TEXT_TIMEOUT_MS = 10_000;
const IMAGE_TIMEOUT_MS = 120_000;

async function call(env: Env, inputs: Record<string, unknown>, timeoutMs: number): Promise<EmbedResult> {
  if (!env.HF_EMBED_URL) throw new EmbedUnavailable("HF_EMBED_URL not configured", false);
  const headers: Record<string, string> = {
    "content-type": "application/json",
    accept: "application/json",
    // Inference Endpoints: hold the request while a scaled-to-zero replica wakes up (ignored by Spaces).
    "x-scale-up-timeout": "600",
  };
  if (env.HF_TOKEN) headers.authorization = `Bearer ${env.HF_TOKEN}`;
  let res: Response;
  try {
    res = await fetch(env.HF_EMBED_URL, { method: "POST", headers, body: JSON.stringify({ inputs }), signal: AbortSignal.timeout(timeoutMs) });
  } catch (e) {
    const name = (e as { name?: string })?.name;
    if (name === "TimeoutError" || name === "AbortError") throw new EmbedUnavailable(`embedding endpoint timed out after ${timeoutMs / 1000} s`, true);
    throw new EmbedUnavailable(`embedding endpoint unreachable: ${e instanceof Error ? e.message : String(e)}`, true);
  }
  if (res.status === 503 || res.status === 502 || res.status === 429) {
    throw new EmbedUnavailable(`embedding endpoint ${res.status}`, true);
  }
  if (!res.ok) {
    // 400 / 413 / 422: hf/app.py read the request and refused this input (a bad image, too many
    // items). Anything else (401 wrong token, 404 wrong URL, 5xx) is the endpoint's.
    const input = res.status === 400 || res.status === 413 || res.status === 422;
    throw new EmbedUnavailable(`embedding endpoint HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`, res.status >= 500, input ? "input" : "endpoint");
  }
  // A Space that is sleeping, building or restarting answers with an HTML page: treat as transient.
  const text = await res.text();
  let json: any;
  try {
    json = JSON.parse(text);
  } catch {
    throw new EmbedUnavailable(`embedding endpoint returned non-JSON (${res.headers.get("content-type") ?? "?"}): ${text.slice(0, 120)}`, true);
  }
  const embeddings: number[][] = Array.isArray(json?.embeddings) ? json.embeddings : Array.isArray(json) ? json : [];
  if (!embeddings.length) throw new EmbedUnavailable("embedding endpoint returned no vectors", false);
  // Vectors from two models are not comparable even at the same dimension, so insist that the
  // Space serves EMBED_MODEL. Retryable: during a rollout the other side catches up in minutes.
  const served = typeof json?.model === "string" ? json.model : null;
  if (env.EMBED_MODEL && served && served !== "stub" && served !== env.EMBED_MODEL) {
    throw new EmbedUnavailable(`embedding endpoint serves ${served} but EMBED_MODEL is ${env.EMBED_MODEL}`, true);
  }
  const expected = int(env.EMBED_DIM, 768);
  if (embeddings[0].length !== expected) {
    throw new EmbedUnavailable(`embedding dim ${embeddings[0].length} != EMBED_DIM ${expected} (Vectorize index must match)`, false);
  }
  // NaFlex: the patch budget shapes image vectors, so it must be the one the index was built with.
  // (Text vectors do not depend on it: queries keep working during a rollout.)
  const patches = Number(json?.max_num_patches);
  if (inputs.images && env.EMBED_PATCHES && Number.isFinite(patches) && patches !== int(env.EMBED_PATCHES, 0)) {
    throw new EmbedUnavailable(`embedding endpoint uses ${patches} patches per image but EMBED_PATCHES is ${env.EMBED_PATCHES}`, true);
  }
  const c = json?.calibration;
  const calibration: Calibration | null =
    c && Number.isFinite(c.logit_scale) && Number.isFinite(c.logit_bias) ? { logit_scale: Number(c.logit_scale), logit_bias: Number(c.logit_bias) } : null;
  return { model: json?.model ?? env.EMBED_MODEL ?? "unknown", dim: embeddings[0].length, embeddings, calibration };
}

/**
 * Embed PNG/WebP images given as base64 strings (no data: prefix needed). Indexing waits up to two
 * minutes (a Space restarting after a deploy); a request path passes a shorter `timeoutMs`.
 */
export function embedImages(env: Env, imagesBase64: string[], timeoutMs = IMAGE_TIMEOUT_MS): Promise<EmbedResult> {
  return call(env, { images: imagesBase64 }, timeoutMs);
}

export function embedTexts(env: Env, texts: string[]): Promise<EmbedResult> {
  return call(env, { texts }, TEXT_TIMEOUT_MS);
}

const CAL_KEY = (model: string) => `calib:${model}`;

/** The model's sigmoid calibration: cached from the Space's replies, else the measured table. */
export async function getCalibration(env: Env): Promise<Calibration | null> {
  const model = env.EMBED_MODEL ?? "";
  const hit = (await env.CACHE.get(CAL_KEY(model), "json").catch(() => null)) as Calibration | null;
  if (hit && Number.isFinite(hit.logit_scale)) return hit;
  return KNOWN_CALIBRATION[model] ?? null;
}

async function rememberCalibration(env: Env, r: EmbedResult): Promise<void> {
  if (!r.calibration || r.model === "stub") return;
  await env.CACHE.put(CAL_KEY(r.model), JSON.stringify(r.calibration), { expirationTtl: 30 * 24 * 3600 }).catch(() => {});
}

/** Normalise a query for the embedding cache key (and the Space lowercases anyway). */
export function normQuery(text: string): string {
  return text.trim().toLowerCase().replace(/\s+/g, " ").slice(0, 256);
}

/** Query-text embeddings are cached in KV (same text → same vector) to keep search latency low. */
export async function embedQueryCached(env: Env, text: string): Promise<number[]> {
  const norm = normQuery(text);
  const key = `qemb:${env.EMBED_MODEL ?? "m"}:${norm}`;
  const hit = await env.CACHE.get(key, "json").catch(() => null);
  if (Array.isArray(hit)) return hit as number[];
  const r = await embedTexts(env, [norm]);
  await Promise.all([
    env.CACHE.put(key, JSON.stringify(r.embeddings[0]), { expirationTtl: 7 * 24 * 3600 }).catch(() => {}),
    rememberCalibration(env, r),
  ]);
  return r.embeddings[0];
}

/** Several query texts at once (one round trip for the uncached ones). */
export async function embedQueriesCached(env: Env, texts: string[]): Promise<number[][]> {
  const norms = texts.map(normQuery);
  const keys = norms.map((n) => `qemb:${env.EMBED_MODEL ?? "m"}:${n}`);
  const hits = await Promise.all(keys.map((k) => env.CACHE.get(k, "json").catch(() => null)));
  const missing = norms.map((n, i) => (Array.isArray(hits[i]) ? null : n)).filter((x): x is string => x !== null);
  if (missing.length) {
    const r = await embedTexts(env, [...new Set(missing)]);
    const byText = new Map([...new Set(missing)].map((t, i) => [t, r.embeddings[i]]));
    await rememberCalibration(env, r);
    norms.forEach((n, i) => {
      if (!Array.isArray(hits[i])) {
        hits[i] = byText.get(n)!;
        env.CACHE.put(keys[i], JSON.stringify(hits[i]), { expirationTtl: 7 * 24 * 3600 }).catch(() => {});
      }
    });
  }
  return hits as number[][];
}

export async function embedAndRemember(env: Env, kind: "images" | "texts", inputs: string[], timeoutMs?: number): Promise<EmbedResult> {
  const r = kind === "images" ? await embedImages(env, inputs, timeoutMs) : await embedTexts(env, inputs);
  await rememberCalibration(env, r);
  return r;
}

// ---- views and vector maths -----------------------------------------------------------------

export type ViewName = "xbrz" | "nearest" | "native";
export interface ViewWeight {
  view: ViewName;
  weight: number;
}

/** EMBED_VIEWS: "xbrz" (default), "nearest", "native", or "nearest:0.65,xbrz:0.35". */
export function parseViews(v: string | undefined): ViewWeight[] {
  const out: ViewWeight[] = [];
  for (const part of (v ?? "xbrz").split(",")) {
    const [name, w] = part.trim().split(":");
    if (name === "xbrz" || name === "nearest" || name === "native") out.push({ view: name, weight: w === undefined ? 1 : Math.max(0, Number(w) || 0) });
  }
  return out.length && out.some((x) => x.weight > 0) ? out.filter((x) => x.weight > 0) : [{ view: "xbrz", weight: 1 }];
}

export function viewsLabel(views: ViewWeight[]): string {
  return views.length === 1 ? views[0].view : views.map((v) => `${v.view}:${v.weight}`).join(",");
}

/**
 * What an image vector was made from: the views, plus the NaFlex patch budget when set
 * ("xbrz@256"). Stored per artwork; a different label makes the sweeper re-embed.
 */
export function embedLabel(env: Pick<Env, "EMBED_VIEWS" | "EMBED_PATCHES">): string {
  const v = viewsLabel(parseViews(env.EMBED_VIEWS));
  const p = int(env.EMBED_PATCHES, 0);
  return p > 0 ? `${v}@${p}` : v;
}

export function normalize(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

/** Weighted sum of unit vectors, re-normalised. */
export function mixVectors(vs: number[][], weights: number[]): number[] {
  const out = new Array<number>(vs[0].length).fill(0);
  vs.forEach((v, k) => v.forEach((x, i) => (out[i] += x * weights[k])));
  return normalize(out);
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
