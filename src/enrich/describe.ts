// AI description of an artwork via Workers AI.
//
//   chat       the vision chat model named by VLM_MODEL (v4.8.1: @cf/zai-org/glm-5.3-flash on the
//              deployed stack), with a JSON schema; any model the table (src/llm/model.ts) knows as
//              vision-capable, or one SEARCH_MODEL_OVERRIDES declares so
//   gemma      @cf/google/gemma-4-26b-a4b-it with a JSON schema, reasoning off (per token the
//              cheapest of the table's vision models on Workers AI in October 2026)
//   scout      @cf/meta/llama-4-scout-17b-16e-instruct with a JSON schema
//   moondream  @cf/moondream/moondream3.1-9B-A2B: "query" task asking for JSON, then the
//              "caption" task as a fallback when the JSON is unusable
//   caption    Moondream "caption" task only (cheapest; caption without tags)
//
// The stage's staleness is by content hash (artworks.describe_hash), not by model: a backend or
// model change describes new and edited images with the new model and leaves the rest as they are
// (`scripts/admin.sh reindex-all describe` redoes them all).
//
// v2 bug fixed here: when Moondream's reply had no usable `answer`, v2 fell back to
// JSON.stringify(reply) and parsed the *envelope* ({finish_reason, metrics, answer: null, ...}) as a
// description with every field empty, which then counted as "done". On the live stacks every
// artwork ended up with an empty caption. Now the reply field is read explicitly per task, the
// result is validated (a caption or tags are required) and an empty result is an error.

import type { Env } from "../env";
import { complete } from "../llm/provider";
import { isModelId, modelSpec } from "../llm/model";
import { MOONDREAM_MODEL, moondreamCaption, moondreamQuery, replyText } from "../llm/adapters/moondream";

export interface Description {
  caption: string;
  subjects: string[];
  objects: string[];
  style: string;
  mood: string;
  text_in_image: string;
  tags: string[];
  nsfw: number; // 0..1
}

export { MOONDREAM_MODEL, replyText };
export const SCOUT_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct";
export const GEMMA_MODEL = "@cf/google/gemma-4-26b-a4b-it";

export const VLM_BACKENDS = ["chat", "gemma", "scout", "moondream", "caption"] as const;
export type VlmBackend = (typeof VLM_BACKENDS)[number];
export const isVlmBackend = (x: string): x is VlmBackend => (VLM_BACKENDS as readonly string[]).includes(x);

/** Chat models that take the image as a message part and answer in JSON-schema mode. */
const CHAT_VLM: Partial<Record<VlmBackend, string>> = { scout: SCOUT_MODEL, gemma: GEMMA_MODEL };

/**
 * Why the "chat" backend cannot run as configured (no VLM_MODEL, not a model id, not a vision
 * model), else null. The consumer skips the stage with this reason instead of failing every image;
 * the other backends have nothing to configure.
 */
export function vlmConfigError(env: Pick<Env, "VLM_MODEL" | "SEARCH_LLM_ENDPOINTS" | "SEARCH_MODEL_OVERRIDES">, backend: VlmBackend): string | null {
  if (backend !== "chat") return null;
  const id = (env.VLM_MODEL ?? "").trim();
  if (!id) return "VLM_BACKEND=chat needs VLM_MODEL (a vision chat model id)";
  if (!isModelId(id, env as Env)) return `VLM_MODEL "${id}" is not a model id this engine can call`;
  if (!modelSpec(id, env as Env).vision) return `VLM_MODEL "${id}" is not known as a vision model (SEARCH_MODEL_OVERRIDES can declare {"${id}": {"vision": true}})`;
  return null;
}

/** The chat model a backend describes with (null for the Moondream tasks). Throws on a misconfigured "chat" backend. */
export function chatVlmModel(env: Env, backend: VlmBackend): string | null {
  if (backend !== "chat") return CHAT_VLM[backend] ?? null;
  const err = vlmConfigError(env, backend);
  if (err) throw new Error(err);
  return env.VLM_MODEL!.trim();
}

export class EmptyDescription extends Error {
  constructor(msg: string, public readonly raw: string) {
    super(msg);
    this.name = "EmptyDescription";
  }
}

const SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["caption", "subjects", "objects", "style", "mood", "text_in_image", "tags", "nsfw"],
  properties: {
    caption: { type: "string", description: "One or two sentences describing what the image shows." },
    subjects: { type: "array", items: { type: "string" }, description: "Main subjects, e.g. 'girl', 'dragon', 'city skyline'." },
    objects: { type: "array", items: { type: "string" }, description: "Notable objects and elements." },
    style: { type: "string", description: "Art style beyond 'pixel art', e.g. 'portrait', 'isometric', 'retro game sprite', 'landscape', 'abstract'." },
    mood: { type: "string", description: "Overall mood in one or two words." },
    text_in_image: { type: "string", description: "Any legible text in the image, else empty string." },
    tags: { type: "array", items: { type: "string" }, description: "5-12 short lowercase search keywords." },
    nsfw: { type: "number", description: "Probability from 0 to 1 that the image is sexual or graphically violent." },
  },
} as const;

export function buildPrompt(ctx: { title?: string; tags?: string[]; description?: string }): string {
  const hints: string[] = [];
  if (ctx.title) hints.push(`Title: "${ctx.title.slice(0, 120)}"`);
  if (ctx.description && ctx.description.trim().length > 2) hints.push(`Author's description: "${ctx.description.slice(0, 300)}"`);
  if (ctx.tags?.length) hints.push(`Author's tags: ${ctx.tags.slice(0, 10).join(", ")}`);
  return [
    "This is a pixel-art image from a social network for pixel artists, shown upscaled so you can see it clearly.",
    "Do not comment on it being pixel art or low resolution; describe the content: who or what is depicted,",
    "the setting, notable objects, the artistic style (portrait, landscape, isometric, sprite, abstract...), the mood,",
    "and any legible text. Write in English even if the title is in another language.",
    hints.length ? `Context from the author (may help, may be wrong): ${hints.join("; ")}.` : "",
    "Respond with a single JSON object with exactly these keys:",
    '{"caption": string, "subjects": string[], "objects": string[], "style": string, "mood": string, "text_in_image": string, "tags": string[] (5-12 lowercase keywords), "nsfw": number 0..1}',
    "No markdown, no explanation, JSON only.",
  ]
    .filter(Boolean)
    .join(" ");
}

/** Pull the first JSON object out of a model reply (handles code fences and chatter). */
export function parseDescription(text: string): Description | null {
  if (!text || typeof text !== "string") return null;
  let s = text.trim().replace(/^```(?:json)?/i, "").replace(/```$/, "").trim();
  const start = s.indexOf("{");
  const end = s.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  s = s.slice(start, end + 1);
  let obj: any;
  try {
    obj = JSON.parse(s);
  } catch {
    return null;
  }
  return normalizeDescription(obj);
}

export function normalizeDescription(obj: any): Description | null {
  if (!obj || typeof obj !== "object" || Array.isArray(obj)) return null;
  const str = (v: unknown, max = 1000) => (typeof v === "string" ? v.trim().slice(0, max) : "");
  const arr = (v: unknown, max = 24) =>
    Array.isArray(v)
      ? [...new Set(v.filter((x) => typeof x === "string").map((x: string) => x.trim().toLowerCase()).filter(Boolean))].slice(0, max)
      : typeof v === "string" && v.trim()
        ? [...new Set(v.split(/[,;]/).map((x) => x.trim().toLowerCase()).filter(Boolean))].slice(0, max)
        : [];
  const nsfw = typeof obj.nsfw === "number" ? Math.min(1, Math.max(0, obj.nsfw)) : typeof obj.nsfw === "boolean" ? (obj.nsfw ? 1 : 0) : 0;
  return {
    caption: str(obj.caption ?? obj.description, 600),
    subjects: arr(obj.subjects),
    objects: arr(obj.objects),
    style: noPixelArt(str(obj.style, 80).toLowerCase()),
    mood: str(obj.mood, 60).toLowerCase(),
    text_in_image: str(obj.text_in_image ?? obj.text, 300),
    tags: arr(obj.tags ?? obj.keywords, 16).filter((t) => !PIXEL_ART_TAG.test(t)),
    nsfw,
  };
}

/**
 * Every artwork is pixel art, which the prompt asks the model not to say. When it says it anyway
 * ("pixel art sprite composition"), the words are dropped: in every description they would only
 * add noise to full-text search.
 */
const PIXEL_ART_TAG = /^(pixel[- ]?art|pixels?|pixelated|pixel[- ]?art style)$/;
export function noPixelArt(style: string): string {
  return style
    .replace(/\bpixel[- ]?art\b|\bpixelated\b/g, "")
    .replace(/\s+([,;])/g, "$1")
    .replace(/([,;])[\s,;]*[,;]/g, "$1")
    .replace(/\s{2,}/g, " ")
    .replace(/^[\s,;:/-]+|[\s,;:/-]+$/g, "")
    .trim();
}

/** A description is usable when it says something: a real caption, or subjects/tags. */
export function isUsable(d: Description | null): d is Description {
  if (!d) return false;
  return d.caption.replace(/[^\p{L}]/gu, "").length >= 8 || d.subjects.length + d.tags.length >= 2;
}

/** Keywords for a caption-only description: its content words (no stopwords), deduplicated. */
export function captionKeywords(caption: string, max = 12): string[] {
  const stop = new Set(
    "a an the and or of to in on at by for with from is are was were be it its this that as into over under image picture shows showing depicts depicting pixel art style there their his her he she they wearing which while has have".split(
      " ",
    ),
  );
  const out: string[] = [];
  for (const w of caption.toLowerCase().split(/[^\p{L}\p{N}-]+/u)) {
    if (w.length < 3 || stop.has(w) || out.includes(w)) continue;
    out.push(w);
    if (out.length >= max) break;
  }
  return out;
}

export type DescribeStatus = "ok" | "caption_only";

export async function describeImage(
  env: Env,
  backend: VlmBackend,
  pngDataUri: string,
  ctx: { title?: string; tags?: string[]; description?: string },
): Promise<{ model: string; description: Description; status: DescribeStatus; raw: string }> {
  const prompt = buildPrompt(ctx);
  const raws: string[] = [];

  const chat = chatVlmModel(env, backend);
  if (chat) {
    // through the model layer (src/llm): the request shape per model lives in its adapter
    const r = await complete(env, {
      model: chat,
      messages: [
        { role: "system", content: "You describe images precisely and answer only with JSON." },
        { role: "user", content: [{ type: "text", text: prompt }, { type: "image", dataUri: pngDataUri }] },
      ],
      json: { name: "artwork_description", schema: SCHEMA },
      reasoning: "none",
      maxOutputTokens: 700,
      temperature: 0.2,
    });
    const raw = r.text ?? "";
    raws.push(raw);
    const d = parseDescription(raw);
    if (isUsable(d)) return { model: chat, description: d, status: "ok", raw };
    throw new EmptyDescription(`${backend} returned no usable description: ${raw.slice(0, 160) || r.notes.join("; ") || "empty reply"}`, raw);
  }

  if (backend === "moondream") {
    const q = await moondreamQuery(env, pngDataUri, prompt);
    raws.push(q.text?.trim() ? q.text : `<no answer: ${q.shape}>`);
    const d = q.text ? parseDescription(q.text) : null;
    if (isUsable(d)) return { model: MOONDREAM_MODEL, description: d, status: "ok", raw: q.text! };
    // fall through to the caption task
  }

  // Moondream's dedicated caption task: always prose, never JSON.
  const c = await moondreamCaption(env, pngDataUri);
  const caption = (c.text ?? "").trim();
  raws.push(caption || `<no caption: ${c.shape}>`);
  const d: Description = { caption: caption.slice(0, 600), subjects: [], objects: [], style: "", mood: "", text_in_image: "", tags: captionKeywords(caption), nsfw: 0 };
  if (!isUsable(d)) throw new EmptyDescription(`moondream returned no usable caption (${raws.join(" | ").slice(0, 200)})`, raws.join("\n---\n"));
  return { model: MOONDREAM_MODEL, description: d, status: backend === "caption" ? "ok" : "caption_only", raw: raws.join("\n---\n") };
}
