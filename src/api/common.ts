// What every route module shares: the bindings type, capped request bodies, rate limits, the admin
// check, and the edge cache.

import type { Context, Next } from "hono";
import type { Env } from "../env";

export type Bindings = { Bindings: Env; Variables: { ctx: ExecutionContext } };

// ---- request bodies -------------------------------------------------------------------------
// Bodies are read with a byte cap (checked on Content-Length first, then while streaming), so a
// public caller cannot make the Worker buffer an arbitrarily large upload.

export const MAX_IMAGE_UPLOAD = 3 * 1024 * 1024; // a 2 MB image as base64 JSON; artworks are far smaller
export const MAX_SMALL_BODY = 16 * 1024;
/** Uploaded images are decoded only up to 1024x1024: enough for any artwork, bounded in memory. */
export const MAX_QUERY_IMAGE_PIXELS = 1024 * 1024;

export class BodyTooLarge extends Error {}

/**
 * A request body, at most `max` bytes. With `large`, a body may grow past `max` up to
 * `large.max` only when `large.allow` accepts its first `max` bytes (a JSON question carrying an
 * image): anything else is refused as soon as it passes `max`, without buffering more.
 */
export async function readBody(req: Request, max: number, large?: { max: number; allow: (prefix: Uint8Array) => boolean }): Promise<Uint8Array> {
  const cap = large ? Math.max(max, large.max) : max;
  const declared = Number(req.headers.get("content-length") ?? "");
  if (Number.isFinite(declared) && declared > cap) throw new BodyTooLarge();
  if (!req.body) return new Uint8Array();
  const reader = req.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  let allowed = !large;
  const join = (n: number) => {
    const out = new Uint8Array(n);
    let o = 0;
    for (const ch of chunks) {
      if (o >= n) break;
      const part = ch.byteLength > n - o ? ch.subarray(0, n - o) : ch;
      out.set(part, o);
      o += part.byteLength;
    }
    return out;
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    chunks.push(value);
    if (total > max && !allowed) {
      allowed = !!large && large.allow(join(max));
      if (!allowed) {
        await reader.cancel().catch(() => {});
        throw new BodyTooLarge();
      }
    }
    if (total > cap) {
      await reader.cancel().catch(() => {});
      throw new BodyTooLarge();
    }
  }
  return join(total);
}

export async function readJson(req: Request, max = MAX_SMALL_BODY): Promise<unknown> {
  const bytes = await readBody(req, max);
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch {
    return null;
  }
}

/** A POST body as an object (anything else as {}). */
export async function bodyObject(c: Context<Bindings>, max = MAX_SMALL_BODY): Promise<Record<string, unknown>> {
  if (c.req.method !== "POST") return {};
  const j = await readJson(c.req.raw, max);
  return j && typeof j === "object" && !Array.isArray(j) ? (j as Record<string, unknown>) : {};
}

// ---- rate limits ----------------------------------------------------------------------------
// Per client IP and Cloudflare location (the binding is approximate by design). Without the
// bindings (tests, wrangler dev) nothing is limited.

export function rateLimited(kind: "public" | "heavy" | "suggest") {
  return async (c: Context<Bindings>, next: Next) => {
    // suggestions: a binding of their own (RL_SUGGEST, a larger budget: one request per pause in
    // typing), else the public one under a key of their own
    const limiter = kind === "heavy" ? c.env.RL_HEAVY : kind === "suggest" ? (c.env.RL_SUGGEST ?? c.env.RL_PUBLIC) : c.env.RL_PUBLIC;
    if (limiter && !isAdmin(c)) {
      const ip = c.req.header("cf-connecting-ip") ?? "unknown";
      const { success } = await limiter.limit({ key: `${kind}:${ip}` }).catch(() => ({ success: true }));
      if (!success) return c.json({ error: "too many requests, slow down" }, 429, { "retry-after": "60" });
    }
    await next();
  };
}

/** /query spends the /ask budget only when it answers (a search stays on the public budget). */
export async function heavyAllowed(c: Context<Bindings>): Promise<boolean> {
  if (!c.env.RL_HEAVY || isAdmin(c)) return true;
  const ip = c.req.header("cf-connecting-ip") ?? "unknown";
  const { success } = await c.env.RL_HEAVY.limit({ key: `heavy:${ip}` }).catch(() => ({ success: true }));
  return success;
}

/** The request carries the admin token (constant-time comparison). */
export function isAdmin(c: Context<Bindings>): boolean {
  const token = c.env.ADMIN_TOKEN;
  const auth = c.req.header("authorization") ?? "";
  if (!token) return false;
  const want = `Bearer ${token}`;
  if (auth.length !== want.length) return false;
  let d = 0;
  for (let i = 0; i < want.length; i++) d |= auth.charCodeAt(i) ^ want.charCodeAt(i);
  return d === 0;
}

/**
 * A JSON answer kept in the location's cache (caches.default) for `maxAge` seconds, under a key
 * built from the normalised parameters, so that everybody typing "dra" in a location costs one
 * lookup a minute. Without the Cache API (tests, local runs) it is computed every time.
 */
export async function edgeCached(c: Context<Bindings>, key: string, maxAge: number, make: () => Promise<unknown>): Promise<Response> {
  const cache = (globalThis as { caches?: { default?: Cache } }).caches?.default;
  const req = new Request(`https://edge-cache.pixagram-search.internal/${key}`);
  const hit = cache ? await cache.match(req).catch(() => undefined) : undefined;
  if (hit) return new Response(hit.body, hit);
  const res = new Response(JSON.stringify(await make()), {
    headers: { "content-type": "application/json; charset=UTF-8", "cache-control": `public, max-age=${maxAge}` },
  });
  if (cache) {
    try {
      c.executionCtx.waitUntil(cache.put(req, res.clone()).catch(() => {}));
    } catch {
      // no execution context: nothing kept
    }
  }
  return res;
}

/** The execution context, or a stand-in that runs waitUntil work immediately (no context: tests). */
export function execOf(c: Context<Bindings>): ExecutionContext {
  try {
    return c.executionCtx as unknown as ExecutionContext;
  } catch {
    return { waitUntil: (p: Promise<unknown>) => void p.catch(() => {}), passThroughOnException() {} } as unknown as ExecutionContext;
  }
}
