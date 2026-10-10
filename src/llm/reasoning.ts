// The reasoning model (spec §17-18, §21-22): interprets the evidence, connects it, says whether it
// is sufficient, and writes an answer with structured claims. It never decides facts: its claims
// are checked against the evidence afterwards (search/claims.ts, search/compose.ts) and the
// deterministic results always win.
//
// ReasoningModel is the interface every caller uses; reasoningModel() builds one for any model id
// the model layer can serve, so swapping models changes nothing else (spec §31: a benchmark gives
// several models exactly the same question, evidence and instructions).
//
// Two answer tasks: "answer" (v4: one to three sentences) and "compose" (v4.8: the same direct
// answer plus a long body, a written reasoning trail, caveats, follow-up questions and searches;
// see prompts.ts). A compose reply that was cut off at the token limit is salvaged: the fields
// that were written whole are kept, the body up to its last complete sentence.

import type { Env } from "../env";
import { int } from "../env";
import type { Lang } from "../lib/text";
import { sha256Hex, utf8 } from "../lib/bytes";
import { callCost, effectiveReasoning, modelSpec } from "./model";
import { complete, LlmError, parseJsonObject, replyObject, type ReasoningLevel, type Usage } from "./provider";
import { PROMPT_VERSION, reasoningMessages, schemaFor, type ReasoningTask } from "./prompts";

/** Any evidence card (search/evidence.ts): an id, a type, and its fields. */
export interface EvidenceCardLike {
  evidence_id: string;
  type: string;
  [field: string]: unknown;
}

export interface ReasoningRequest {
  question: string;
  evidence: EvidenceCardLike[];
  reasoning: ReasoningLevel;
  /** tokens for the visible answer */
  maxTokens: number;
  temperature?: number;
  responseFormat?: "text" | "json";
  task?: ReasoningTask;
  lang?: Lang;
  /** what the model must know about the evidence ("counts are lower bounds") */
  context?: string[];
  /** compose: the body's length, in words */
  words?: number;
}

export interface Claim {
  text: string;
  evidence: string[];
  confidence?: number;
  kind?: "fact" | "inference";
}

export type ReasoningStatus = "answered" | "insufficient_evidence" | "conflict";

export interface ReasoningResponse {
  answer: string;
  claims: Claim[];
  confidence?: number;
  model: string;
  usage?: Usage;
  status: ReasoningStatus;
  rationale?: string;
  conflicts?: Array<{ evidence: string[]; about: string }>;
  /** compose: the long answer in Markdown, as the model wrote it (verified sentence by sentence afterwards) */
  body?: string;
  /** compose: the written reasoning trail, one step per entry */
  thinking?: string[];
  caveats?: string[];
  followUps?: string[];
  searches?: string[];
  /** the reply was cut off at the token limit and salvaged (compose) */
  salvaged?: boolean;
  /** the reasoning level the model ran at (a level it lacks is mapped, model.ts) */
  reasoning: ReasoningLevel;
  promptVersion: string;
  latencyMs: number;
  costUsd: number | null;
  finishReason: string | null;
  notes: string[];
}

export interface ReasoningModel {
  id: string;
  generate(request: ReasoningRequest): Promise<ReasoningResponse>;
}

/** The reply could not be read as an answer (empty, cut off, not JSON). */
export class UnusableReply extends Error {
  constructor(message: string, public readonly model: string) {
    super(message);
    this.name = "UnusableReply";
  }
}

const MAX_CLAIMS = 24;
const MAX_TEXT = 600;
/** The long body of a compose reply, in characters (about 1,500 words). */
export const MAX_BODY = 12_000;
const MAX_STEPS = 10;
const MAX_LIST = 8;
/** Evidence ids: a capital letter and a number (E12 an artwork, R1 a result, D3 a documentation excerpt…). */
const ID = /^[A-Z]\d{1,9}$/;

/** "E12", "[E12]", "e12", "12" (help: a doc number) → the evidence id as given in the cards, or null. */
export function normalizeEvidenceId(raw: unknown, valid: Set<string>, task: ReasoningTask = "answer"): string | null {
  if (typeof raw !== "string" && typeof raw !== "number") return null;
  let s = String(raw).trim().replace(/^\[|\]$/g, "").toUpperCase();
  if (/^\d{1,4}$/.test(s) && task === "help") s = `D${s}`;
  return ID.test(s) && valid.has(s) ? s : null;
}

const clamp01 = (x: unknown): number | undefined => (typeof x === "number" && Number.isFinite(x) ? Math.min(1, Math.max(0, x)) : undefined);
const text = (x: unknown, max = MAX_TEXT): string => (typeof x === "string" ? x.replace(/\s+/g, " ").trim().slice(0, max) : "");
/** A Markdown text: line breaks kept (paragraphs, lists), runs of blank lines reduced. */
const markdown = (x: unknown, max = MAX_BODY): string =>
  typeof x === "string"
    ? x
        .replace(/\r\n?/g, "\n")
        .replace(/[ \t]+\n/g, "\n")
        .replace(/\n{3,}/g, "\n\n")
        .trim()
        .slice(0, max)
    : "";
/** A list of short texts (steps, caveats, questions): strings only, numbered prefixes removed. */
function texts(x: unknown, max: number, each: number): string[] {
  if (!Array.isArray(x)) return [];
  const out: string[] = [];
  for (const v of x) {
    const t = text(typeof v === "string" ? v : v && typeof v === "object" && typeof (v as any).text === "string" ? (v as any).text : "", each).replace(/^\s*(?:\d{1,2}[.)]|[-*•])\s+/, "");
    if (t) out.push(t);
    if (out.length >= max) break;
  }
  return out;
}

/**
 * A model's reply as a ReasoningResponse body. Unknown evidence ids are dropped (the claim keeps its
 * text, so claim verification sees it unsupported), claims are bounded in number and length.
 */
export function parseReasoningReply(
  o: Record<string, unknown>,
  validIds: Set<string>,
  task: ReasoningTask = "answer",
): Pick<ReasoningResponse, "answer" | "claims" | "status" | "rationale" | "confidence" | "conflicts" | "body" | "thinking" | "caveats" | "followUps" | "searches"> {
  const status: ReasoningStatus = o.status === "insufficient_evidence" || o.status === "conflict" ? o.status : "answered";
  const claims: Claim[] = [];
  for (const c of Array.isArray(o.claims) ? o.claims.slice(0, MAX_CLAIMS * 2) : []) {
    if (!c || typeof c !== "object") continue;
    const t = text((c as any).text);
    if (!t) continue;
    const ev = [...new Set((Array.isArray((c as any).evidence) ? (c as any).evidence : [(c as any).evidence]).map((x: unknown) => normalizeEvidenceId(x, validIds, task)).filter((x: string | null): x is string => !!x))] as string[];
    claims.push({ text: t, evidence: ev, kind: (c as any).kind === "inference" ? "inference" : "fact", ...(clamp01((c as any).confidence) !== undefined ? { confidence: clamp01((c as any).confidence) } : {}) });
    if (claims.length >= MAX_CLAIMS) break;
  }
  const conflicts = (Array.isArray(o.conflicts) ? o.conflicts : [])
    .slice(0, 6)
    .map((c: any) => ({ evidence: (Array.isArray(c?.evidence) ? c.evidence : []).map((x: unknown) => normalizeEvidenceId(x, validIds, task)).filter(Boolean) as string[], about: text(c?.about, 300) }))
    .filter((c) => c.about || c.evidence.length);
  const out: ReturnType<typeof parseReasoningReply> = {
    status,
    answer: text(o.answer, 1500),
    claims,
    rationale: text(o.rationale, 400) || undefined,
    confidence: clamp01(o.confidence),
    ...(conflicts.length ? { conflicts } : {}),
  };
  if (task === "compose") {
    const body = markdown(o.body);
    if (body) out.body = body;
    const thinking = texts(o.thinking, MAX_STEPS, 400);
    if (thinking.length) out.thinking = thinking;
    const caveats = texts(o.caveats, MAX_LIST, 300);
    if (caveats.length) out.caveats = caveats;
    const followUps = texts(o.follow_ups ?? o.followUps, MAX_LIST, 200);
    if (followUps.length) out.followUps = followUps;
    const searches = texts(o.searches, MAX_LIST, 80);
    if (searches.length) out.searches = searches;
  }
  return out;
}

/**
 * Text mode: sentences are the claims, with the ids written in them ("… [E12][R1]"). For models
 * without a JSON mode; the JSON reply is the default.
 */
export function parseTextReply(t: string, validIds: Set<string>, task: ReasoningTask = "answer"): Pick<ReasoningResponse, "answer" | "claims" | "status"> {
  const answer = t.replace(/\s+/g, " ").trim().slice(0, 1500);
  const sentences = answer.match(/[^.!?。！？]+[.!?。！？]*/g) ?? [];
  const claims: Claim[] = [];
  for (const s of sentences.slice(0, MAX_CLAIMS)) {
    const ids = [...s.matchAll(/\[([A-Za-z]?\d{1,9})\]/g)].map((m) => normalizeEvidenceId(m[1], validIds, task)).filter((x): x is string => !!x);
    const clean = s.replace(/\s*\[[A-Za-z]?\d{1,9}\]/g, "").trim();
    if (clean) claims.push({ text: clean, evidence: [...new Set(ids)], kind: "fact" });
  }
  const insufficient = /insufficient evidence|pas assez d.éléments|nicht genug|evidencia insuficiente|prove insufficienti/i.test(answer);
  return { answer, claims, status: insufficient ? "insufficient_evidence" : "answered" };
}

/**
 * The top-level fields of a JSON object written as far as `src` goes: complete values parsed, a
 * string cut off mid-way returned under `partial`, everything after the cut ignored. Reads the
 * object's own keys only (a "status" inside the body's text is never mistaken for the field).
 */
export function partialJsonObject(src: string): { fields: Record<string, unknown>; partial: { key: string; text: string } | null } {
  const fields: Record<string, unknown> = {};
  let partial: { key: string; text: string } | null = null;
  const n = src.length;
  let i = src.indexOf("{");
  if (i < 0) return { fields, partial };
  i++;
  const ws = () => {
    while (i < n && /\s/.test(src[i])) i++;
  };
  // a string literal from the opening quote at i: its value, and whether it was closed
  const str = (): { value: string; complete: boolean } => {
    let out = "";
    i++;
    while (i < n) {
      const ch = src[i];
      if (ch === '"') {
        i++;
        return { value: out, complete: true };
      }
      if (ch === "\\") {
        const e = src[i + 1];
        if (e === undefined) break;
        if (e === "n") out += "\n";
        else if (e === "t") out += "\t";
        else if (e === "r") out += "\r";
        else if (e === "u" && /^[0-9a-fA-F]{4}$/.test(src.slice(i + 2, i + 6))) {
          out += String.fromCharCode(parseInt(src.slice(i + 2, i + 6), 16));
          i += 4;
        } else out += e;
        i += 2;
        continue;
      }
      out += ch;
      i++;
    }
    return { value: out, complete: false };
  };
  // the end of a balanced [...] or {...} starting at i (strings respected), or -1 when cut
  const balanced = (): number => {
    let depth = 0;
    let inString = false;
    for (let k = i; k < n; k++) {
      const ch = src[k];
      if (inString) {
        if (ch === "\\") k++;
        else if (ch === '"') inString = false;
        continue;
      }
      if (ch === '"') inString = true;
      else if (ch === "[" || ch === "{") depth++;
      else if (ch === "]" || ch === "}") {
        depth--;
        if (depth === 0) return k;
      }
    }
    return -1;
  };
  while (i < n) {
    ws();
    if (src[i] === "}") break;
    if (src[i] === ",") {
      i++;
      continue;
    }
    if (src[i] !== '"') break;
    const key = str();
    if (!key.complete) break;
    ws();
    if (src[i] !== ":") break;
    i++;
    ws();
    const ch = src[i];
    if (ch === undefined) break;
    if (ch === '"') {
      const v = str();
      if (!v.complete) {
        partial = { key: key.value, text: v.value };
        break;
      }
      fields[key.value] = v.value;
    } else if (ch === "[" || ch === "{") {
      const end = balanced();
      if (end < 0) break;
      try {
        fields[key.value] = JSON.parse(src.slice(i, end + 1));
      } catch {
        break;
      }
      i = end + 1;
    } else {
      let k = i;
      while (k < n && !/[,}\s]/.test(src[k])) k++;
      const raw = src.slice(i, k);
      if (k >= n) break; // a number or literal cut off: unknown
      try {
        fields[key.value] = JSON.parse(raw);
      } catch {
        break;
      }
      i = k;
    }
  }
  return { fields, partial };
}

/**
 * What can be kept of a compose reply cut off at the token limit: the answer when it was written
 * whole, the body up to its last complete sentence, the lists that were closed. Null when not even
 * the answer is there, or when the model's status (when it was written) was not "answered": an
 * "insufficient evidence" reply cut off is no answer at all.
 */
export function salvageComposeReply(src: string | null | undefined): Record<string, unknown> | null {
  if (typeof src !== "string" || !src.includes("{")) return null;
  const { fields, partial } = partialJsonObject(src.trim().replace(/^```(?:json)?\s*/i, ""));
  if (typeof fields.status === "string" && fields.status !== "answered") return null;
  const answer = typeof fields.answer === "string" ? fields.answer : null;
  if (!answer?.trim()) return null;
  const out: Record<string, unknown> = { status: "answered", answer, claims: Array.isArray(fields.claims) ? fields.claims : [], salvaged: true };
  let body = typeof fields.body === "string" ? fields.body : partial?.key === "body" ? partial.text : "";
  if (partial?.key === "body") {
    // up to the last sentence end (with its citations), so no half sentence is shown
    const m = /[\s\S]*[.!?…](?:\s*\[[A-Za-z]?\d{1,9}\])*/.exec(body);
    body = m ? m[0] : "";
  }
  if (body.trim()) out.body = body;
  for (const key of ["thinking", "caveats", "follow_ups", "searches", "conflicts"]) if (Array.isArray(fields[key])) out[key] = fields[key];
  if (typeof fields.rationale === "string") out.rationale = fields.rationale;
  if (typeof fields.confidence === "number") out.confidence = fields.confidence;
  return out;
}

/** A ReasoningModel for any model id the model layer can serve. */
export function reasoningModel(env: Env, id: string, opts: { timeoutMs?: number } = {}): ReasoningModel {
  const spec = modelSpec(id, env);
  return {
    id: spec.id,
    async generate(req: ReasoningRequest): Promise<ReasoningResponse> {
      const task = req.task ?? "answer";
      const validIds = new Set(req.evidence.map((c) => c.evidence_id));
      const messages = reasoningMessages({ task, question: req.question, cards: req.evidence, lang: req.lang ?? "en", context: req.context, words: req.words });
      const json = (req.responseFormat ?? "json") === "json";
      const r = await complete(env, {
        model: spec.id,
        messages,
        json: json ? { name: task === "help" ? "help_answer" : task === "compose" ? "composed_answer" : "evidence_answer", schema: schemaFor(task) } : undefined,
        reasoning: req.reasoning,
        maxOutputTokens: req.maxTokens,
        temperature: req.temperature ?? 0,
        timeoutMs: opts.timeoutMs ?? int(env.SEARCH_LLM_TIMEOUT_MS, 45_000),
      });
      const notes = [...r.notes];
      let body: ReturnType<typeof parseReasoningReply>;
      let salvaged = false;
      if (json) {
        let o = replyObject(r) ?? parseJsonObject(r.text);
        if (!o && task === "compose") {
          o = salvageComposeReply(r.text);
          if (o) {
            salvaged = true;
            notes.push(`${spec.id}: the reply was cut off${r.finishReason === "length" ? " at the token limit" : ""}; what was written whole is kept`);
          }
        }
        if (!o) throw new UnusableReply(`${spec.id}: reply is not a JSON answer${r.finishReason === "length" ? " (cut off at the token limit)" : ""}: ${(r.text ?? "").slice(0, 120)}`, spec.id);
        body = parseReasoningReply(o, validIds, task);
      } else {
        if (!r.text?.trim()) throw new UnusableReply(`${spec.id}: empty reply`, spec.id);
        body = parseTextReply(r.text, validIds, task);
      }
      if (!body.answer && body.status === "answered") throw new UnusableReply(`${spec.id}: the reply has no answer`, spec.id);
      const usage = r.usage ?? undefined;
      return {
        ...body,
        ...(salvaged ? { salvaged } : {}),
        model: spec.id,
        usage,
        reasoning: effectiveReasoning(spec, req.reasoning),
        promptVersion: PROMPT_VERSION,
        latencyMs: r.latencyMs,
        costUsd: usage ? callCost(spec, usage.inputTokens, usage.outputTokens) : null,
        finishReason: r.finishReason,
        notes,
      };
    },
  };
}

/** Cache key of a reasoning call (spec §50): the model, the prompt version, the budget and the exact context. */
export async function reasoningCacheKey(model: string, req: ReasoningRequest): Promise<string> {
  const payload = JSON.stringify([model, PROMPT_VERSION, req.task ?? "answer", req.reasoning, req.maxTokens, req.lang ?? "en", req.question, req.evidence, req.context ?? [], req.words ?? null]);
  return `llm4:${(await sha256Hex(utf8(payload))).slice(0, 40)}`;
}

/** generate() with the KV cache in front (answers at temperature 0 on the same evidence are reused). */
export async function generateCached(env: Env, model: ReasoningModel, req: ReasoningRequest, ttl: number): Promise<ReasoningResponse & { cached?: boolean }> {
  if (ttl <= 0) return model.generate(req);
  const key = await reasoningCacheKey(model.id, req);
  const hit = (await env.CACHE.get(key, "json").catch(() => null)) as ReasoningResponse | null;
  if (hit && typeof hit.answer === "string" && Array.isArray(hit.claims)) return { ...hit, cached: true, latencyMs: 0, costUsd: 0 };
  const r = await model.generate(req);
  await env.CACHE.put(key, JSON.stringify(r), { expirationTtl: Math.max(60, ttl) }).catch(() => {});
  return r;
}

export { LlmError };
