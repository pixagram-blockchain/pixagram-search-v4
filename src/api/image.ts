// Uploaded images (/search-by-image, /ask with an image): read from a multipart form or a JSON body,
// checked, and turned into what the index compares images by.

import type { Env } from "../env";
import { int } from "../env";
import { base64Decode, base64Encode, sha256Hex } from "../lib/bytes";
import { decodeImage, encodePng, sniff } from "../enrich/decode";
import { dhash, phash } from "../enrich/phash";
import { computeStats } from "../enrich/stats";
import { embeddingEnabled } from "../enrich/embed";
import { embedImageViews } from "../enrich/consumer";
import { factorFor, upscale } from "../enrich/upscale";
import { describeImage, isVlmBackend } from "../enrich/describe";
import type { QueryImage } from "../search/image-question";
import { MAX_QUERY_IMAGE_PIXELS } from "./common";

export type Upload = { bytes: Uint8Array; info: ReturnType<typeof sniff>; fields: Record<string, string> } | { error: string; status: 400 | 413 | 415 };

/**
 * The image of a request body: multipart field "image" (other text fields come back in `fields`),
 * or JSON {"image": "<base64 or data URI>", …}.
 */
export async function readUploadedImage(url: string, contentType: string, raw: Uint8Array): Promise<Upload> {
  let bytes: Uint8Array | null = null;
  const fields: Record<string, string> = {};
  if (contentType.includes("multipart/form-data")) {
    const body = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
    const form = await new Request(url, { method: "POST", headers: { "content-type": contentType }, body }).formData().catch(() => null);
    const f = form?.get("image");
    if (f && typeof f !== "string") bytes = new Uint8Array(await f.arrayBuffer());
    form?.forEach((v, k) => {
      if (typeof v === "string" && k !== "image") fields[k] = v.slice(0, 2000);
    });
  } else {
    let body: Record<string, unknown> | null = null;
    try {
      const j = JSON.parse(new TextDecoder().decode(raw));
      body = j && typeof j === "object" && !Array.isArray(j) ? j : null;
    } catch {
      body = null;
    }
    if (typeof body?.image === "string") {
      try {
        bytes = base64Decode(body.image.replace(/^data:[^,]*,/, ""));
      } catch {
        bytes = null;
      }
    }
    for (const [k, v] of Object.entries(body ?? {})) if (k !== "image" && (typeof v === "string" || typeof v === "number" || typeof v === "boolean")) fields[k] = String(v).slice(0, 2000);
  }
  if (!bytes || bytes.length === 0) return { error: "no image", status: 400 };
  const info = sniff(bytes);
  if (info.format === "unknown") return { error: "unsupported format (webp or png)", status: 415 };
  return { bytes, info, fields };
}

/**
 * What /ask compares an uploaded image by: its sha-256 (the content hash of the same file), pHash,
 * dHash, colours and, when the embedding endpoint is on, its SigLIP vector (the views the index
 * uses). describe() asks the vision model what it shows, on demand.
 */
export async function queryImageOf(env: Env, bytes: Uint8Array, info: ReturnType<typeof sniff>, notes: string[]): Promise<QueryImage & { describe: () => Promise<QueryImage["description"] | null> }> {
  if (!info.width || !info.height || info.width * info.height > MAX_QUERY_IMAGE_PIXELS) throw new ImageTooLarge(`image too large: ${info.width ?? "?"}x${info.height ?? "?"} (1024x1024 max)`);
  const img = await decodeImage(bytes, info, MAX_QUERY_IMAGE_PIXELS);
  const stats = computeStats(img, { lossy: info.lossy });
  let vector: number[] | null = null;
  if (embeddingEnabled(env)) {
    try {
      vector = (await embedImageViews(env, img)).vector;
    } catch (e) {
      notes.push(`image embedding unavailable: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return {
    sha256: await sha256Hex(bytes),
    phash: phash(img),
    dhash: dhash(img),
    buckets: stats.buckets,
    width: img.width,
    height: img.height,
    format: info.format,
    vector,
    describe: async () => {
      const configured = (env.VLM_BACKEND ?? "gemma").toLowerCase();
      const backend = isVlmBackend(configured) ? configured : "gemma";
      const png = await encodePng(upscale(img, factorFor(img.width, img.height, int(env.VLM_TARGET, 512)), (env.SCALER ?? "xbrz") === "nearest" ? "nearest" : "xbrz"));
      const r = await describeImage(env, backend, `data:image/png;base64,${base64Encode(png)}`, {});
      const d = r.description;
      return { caption: d.caption, subjects: d.subjects, tags: d.tags, text_in_image: d.text_in_image, model: r.model };
    },
  };
}

export class ImageTooLarge extends Error {}
