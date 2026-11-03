// The images SigLIP sees for an artwork, and its image vector from them: shared by the embed stage
// (consumer.ts), /search-by-image, copy detection's uploads and the on-demand vector of /similar
// (on-demand.ts), so that every image goes through exactly the same pipeline as the indexed
// artworks. A module of its own so that the search side can use it without importing the queue
// consumer (which reaches src/search through copy detection).

import type { Env } from "../env";
import { int } from "../env";
import { base64Encode } from "../lib/bytes";
import { encodePng, type RgbaImage } from "./decode";
import { factorFor, upscale, upscaleNearest } from "./upscale";
import { embedAndRemember, embedLabel, mixVectors, parseViews, type ViewName } from "./embed";

/**
 * The PNG SigLIP sees for one view. Shared by the embed stage and /search-by-image so a query
 * image goes through exactly the same pipeline as the indexed artworks. `preview` is the cached
 * R2 preview (xBRZ at UPSCALE_TARGET when SCALER is xbrz), reused instead of upscaling twice.
 */
export async function viewPng(view: ViewName, im: RgbaImage, env: Pick<Env, "SCALER" | "UPSCALE_TARGET">, preview?: () => Promise<Uint8Array>): Promise<Uint8Array> {
  if (view === "xbrz") {
    if (preview && (env.SCALER ?? "xbrz") !== "nearest") return preview();
    return encodePng(upscale(im, factorFor(im.width, im.height, int(env.UPSCALE_TARGET, 800)), "xbrz"));
  }
  if (view === "nearest") return encodePng(upscaleNearest(im, factorFor(im.width, im.height, 512)));
  return encodePng(im);
}

/**
 * SigLIP image vector of a decoded image per EMBED_VIEWS (one view, or the weighted mix of several).
 * `timeoutMs`: how long the Space may take (indexing waits two minutes; a request path less).
 */
export async function embedImageViews(env: Env, im: RgbaImage, preview?: () => Promise<Uint8Array>, o: { timeoutMs?: number } = {}): Promise<{ vector: number[]; model: string; label: string }> {
  const views = parseViews(env.EMBED_VIEWS);
  const images: string[] = [];
  for (const v of views) images.push(base64Encode(await viewPng(v.view, im, env, preview)));
  const r = await embedAndRemember(env, "images", images, o.timeoutMs);
  const vector = r.embeddings.length === 1 ? r.embeddings[0] : mixVectors(r.embeddings, views.map((v) => v.weight));
  return { vector, model: r.model, label: embedLabel(env) };
}

// ---- a vector computed for a request, reused by the embed stage ----------------------------------

const VECTOR_TTL_S = 24 * 3600;
const vectorKey = (env: Env, hash: string) => `ivec:${env.EMBED_MODEL ?? "m"}:${embedLabel(env)}:${hash}`;

/**
 * The image vector computed in the last day for this image (content hash), model, views and patch
 * budget: what /similar computed on demand (on-demand.ts). The embed stage stores it instead of
 * asking the Space for the same vector again.
 */
export async function cachedImageVector(env: Env, hash: string): Promise<number[] | null> {
  const hit = await env.CACHE.get(vectorKey(env, hash), "json").catch(() => null);
  return Array.isArray(hit) && hit.length === int(env.EMBED_DIM, 768) ? (hit as number[]) : null;
}

export async function rememberImageVector(env: Env, hash: string, vector: number[]): Promise<void> {
  await env.CACHE.put(vectorKey(env, hash), JSON.stringify(vector), { expirationTtl: VECTOR_TTL_S }).catch(() => {});
}
