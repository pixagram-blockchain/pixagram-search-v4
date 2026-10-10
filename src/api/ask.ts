// /ask: questions answered from the index's evidence (search/ask.ts), and votes on the answers.
//
//   GET  /ask?q=…&mode=…&reasoning=…&style=rich|brief&text=short|full&length=short|medium|long&defer=1
//   POST /ask {"question": "…", "mode": "auto|fast|balanced|deep|expert|v3", "reasoning": "auto|none|low|medium|high",
//              "model": "…", "max_output_tokens": 1500, "trace": true, "graph": true, "image": "<base64 or data URI>",
//              "style": "rich", "text": "short", "length": "medium", "defer": false}
//   POST /ask (multipart: image + question + the same fields) — a question about an uploaded image
//   GET  /ask/elaboration/:query_id — the model's part of an answer asked with defer=1 (README "Rich answers")
//   POST /ask/feedback {"query_id": "…", "rating": 1 | -1, "reason": "wrong|unsupported|incomplete|other"}
//
// Public callers: modes up to SEARCH_MAX_MODE (default deep), reasoning up to medium, the models of
// SEARCH_PUBLIC_MODELS, traces when SEARCH_TRACE is on. The admin token lifts all four.

import type { Context, Hono } from "hono";
import { cleanText } from "../search/params";
import { ask, elaborate, type AskRequestV4 } from "../search/ask";
import { isAnswerStyle, isLengthRequest } from "../search/compose";
import { isModeRequest } from "../search/query-router";
import { isReasoningLevel } from "../llm/provider";
import { recordAnswerFeedback } from "../search/feedback";
import { BodyTooLarge, execOf, isAdmin, MAX_IMAGE_UPLOAD, MAX_SMALL_BODY, readBody, readJson, type Bindings } from "./common";
import { ImageTooLarge, queryImageOf, readUploadedImage } from "./image";

const truthy = (v: unknown) => v === true || (typeof v === "string" && /^(1|true|yes|on)$/i.test(v)) || v === 1;

/** The long-form options of a request (/ask and /query): style, text, length, defer. */
export function richOptions(get: (k: string) => unknown): Pick<AskRequestV4, "style" | "text" | "length" | "defer"> {
  const style = get("style");
  const text = get("text");
  const length = get("length");
  return {
    style: isAnswerStyle(style) ? style : undefined,
    text: text === "full" || text === "short" ? text : undefined,
    length: isLengthRequest(length) ? length : undefined,
    defer: truthy(get("defer")) || undefined,
  };
}

/** The /ask request of a GET or POST, with what the caller may ask for. */
function askRequest(c: Context<Bindings>, body: Record<string, unknown>, notes: string[]): AskRequestV4 | { error: string } {
  const sp = new URL(c.req.url).searchParams;
  const get = (k: string) => (body[k] !== undefined && body[k] !== null ? body[k] : sp.get(k));
  const question = cleanText(String(body.question ?? sp.get("q") ?? sp.get("question") ?? "")).slice(0, 300);
  if (!question) return { error: 'give a question: POST {"question": "..."} or GET /ask?q=...' };
  const admin = isAdmin(c);
  const type = get("type");
  const nsfw = String(get("nsfw") ?? "exclude");
  const mode = get("mode");
  let reasoning = get("reasoning");
  if (reasoning === "high" && !admin) {
    reasoning = "medium";
    notes.push("reasoning=high needs the admin token: medium used");
  }
  const maxTokens = Number(get("max_output_tokens"));
  return {
    question,
    type: type === "artwork" || type === "blog" ? type : undefined,
    limit: Number(get("limit") ?? 10) || 10,
    nsfw: nsfw === "include" || nsfw === "only" ? nsfw : "exclude",
    threshold: Number(get("threshold")) || undefined,
    // Callers may turn the LLM planner off, never force it on (it is the expensive path).
    planner: String(get("planner") ?? "") === "rules" ? "rules" : undefined,
    mode: isModeRequest(mode) ? mode : undefined,
    reasoning: reasoning === "auto" || isReasoningLevel(reasoning) ? (reasoning as AskRequestV4["reasoning"]) : undefined,
    model: typeof get("model") === "string" ? String(get("model")).slice(0, 120) : undefined,
    max_output_tokens: Number.isFinite(maxTokens) && maxTokens > 0 ? Math.floor(maxTokens) : undefined,
    trace: truthy(get("trace")),
    graph: truthy(get("graph")),
    admin,
    ...richOptions(get),
  };
}

export function registerAsk(app: Hono<Bindings>): void {
  /** Natural-language questions answered from evidence: POST {"question": "..."} or GET ?q=, optionally about an image. */
  app.on(["GET", "POST"], "/ask", async (c) => {
    const notes: string[] = [];
    let body: Record<string, unknown> = {};
    let image: AskRequestV4["image"];
    let upload: { bytes: Uint8Array; info: Parameters<typeof queryImageOf>[2] } | null = null;
    if (c.req.method === "POST") {
      const ct = c.req.header("content-type") ?? "";
      const multipart = ct.includes("multipart/form-data");
      // a question alone stays small (16 KB, as in v3); only an upload (multipart, or JSON whose
      // first 16 KB carry an "image" field) may be larger, up to MAX_IMAGE_UPLOAD
      const raw = multipart
        ? await readBody(c.req.raw, MAX_IMAGE_UPLOAD)
        : await readBody(c.req.raw, MAX_SMALL_BODY, { max: MAX_IMAGE_UPLOAD, allow: (prefix) => /"image"\s*:\s*"/.test(new TextDecoder().decode(prefix)) });
      let json: unknown = null;
      if (!multipart) {
        try {
          json = JSON.parse(new TextDecoder().decode(raw));
        } catch {
          json = null;
        }
      }
      const obj = json && typeof json === "object" && !Array.isArray(json) ? (json as Record<string, unknown>) : {};
      if (!multipart && typeof obj.image === "string") {
        // the question and the mode are checked before the image is even read
        const pre = askRequest(c, obj, []);
        if ("error" in pre) return c.json({ error: pre.error }, 400);
        if (pre.mode === "v3") return c.json({ error: "mode=v3 has no questions about images" }, 400);
      }
      if (multipart || typeof obj.image === "string") {
        const up = await readUploadedImage(c.req.url, ct, raw);
        if ("error" in up) return c.json({ error: up.error }, up.status);
        body = up.fields;
        upload = { bytes: up.bytes, info: up.info };
      } else {
        if (raw.length > MAX_SMALL_BODY) throw new BodyTooLarge();
        body = obj;
      }
    }
    // the question and its options are checked before the image is decoded, hashed or embedded
    const a = askRequest(c, body, notes);
    if ("error" in a) return c.json({ error: a.error }, 400);
    if (upload && a.mode === "v3") return c.json({ error: "mode=v3 has no questions about images" }, 400);
    if (upload) {
      try {
        image = await queryImageOf(c.env, upload.bytes, upload.info, notes);
      } catch (e) {
        if (e instanceof ImageTooLarge) return c.json({ error: e.message }, 413);
        throw e;
      }
    }
    const res = await ask(c.env, { ...a, ...(image ? { image } : {}) }, execOf(c));
    if (notes.length) res.notes = [...new Set([...notes, ...res.notes])];
    return c.json(res);
  });

  /**
   * The model's part of a rich answer asked with defer=1: {status: pending | ready | failed |
   * unknown, answer: the fields it changed}. Runs the model the first time it is asked (so the
   * caller waits here instead of on /ask); a failed call is tried again by the next GET.
   */
  app.get("/ask/elaboration/:qid", async (c) => {
    const r = await elaborate(c.env, c.req.param("qid"));
    c.header("cache-control", "no-store");
    return c.json(r, r.status === "unknown" ? 404 : 200);
  });

  /** A vote on an answer: {"query_id": "…", "rating": 1 | -1, "reason"?}. */
  app.post("/ask/feedback", async (c) => {
    const r = await recordAnswerFeedback(c.env, await readJson(c.req.raw, 2048));
    return r.ok ? c.body(null, 204) : c.json({ error: r.error }, 400);
  });
}
