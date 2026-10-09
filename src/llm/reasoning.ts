// The reasoning model (spec §17-18, §21-22): interprets the evidence, connects it, says whether it
// is sufficient, and writes a concise answer with structured claims. It never decides facts: its
// claims are checked against the evidence afterwards (search/claims.ts) and the deterministic
// results always win.
//
// ReasoningModel is the interface every caller uses; reasoningModel() builds one for any model id
// the model layer can serve, so swapping models changes nothing else (spec §31: a benchmark gives
// several models exactly the same question, evidence and instructions).

import type { Env } from "../env";
import { int } from "../env";
import type { Lang } from "../lib/text";
import { sha256Hex, utf8 } from "../lib/bytes";
import { callCost, effectiveReasoning, modelSpec } from "./model";
import { complete, LlmError, parseJsonObject, replyObject, type ReasoningLevel, type Usage } from "./provider";
import { PROMPT_VERSION, REPLY_SCHEMA, reasoningMessages, type ReasoningTask } from "./prompts";

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

const MAX_CLAIMS = 12;
const MAX_TEXT = 600;
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

/**
 * A model's reply as a ReasoningResponse body. Unknown evidence ids are dropped (the claim keeps its
 * text, so claim verification sees it unsupported), claims are bounded in number and length.
 */
export function parseReasoningReply(o: Record<string, unknown>, validIds: Set<string>, task: ReasoningTask = "answer"): Pick<ReasoningResponse, "answer" | "claims" | "status" | "rationale" | "confidence" | "conflicts"> {
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
  return {
    status,
    answer: text(o.answer, 1500),
    claims,
    rationale: text(o.rationale, 400) || undefined,
    confidence: clamp01(o.confidence),
    ...(conflicts.length ? { conflicts } : {}),
  };
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

/** A ReasoningModel for any model id the model layer can serve. */
export function reasoningModel(env: Env, id: string, opts: { timeoutMs?: number } = {}): ReasoningModel {
  const spec = modelSpec(id, env);
  return {
    id: spec.id,
    async generate(req: ReasoningRequest): Promise<ReasoningResponse> {
      const task = req.task ?? "answer";
      const validIds = new Set(req.evidence.map((c) => c.evidence_id));
      const messages = reasoningMessages({ task, question: req.question, cards: req.evidence, lang: req.lang ?? "en", context: req.context });
      const json = (req.responseFormat ?? "json") === "json";
      const r = await complete(env, {
        model: spec.id,
        messages,
        json: json ? { name: task === "help" ? "help_answer" : "evidence_answer", schema: REPLY_SCHEMA } : undefined,
        reasoning: req.reasoning,
        maxOutputTokens: req.maxTokens,
        temperature: req.temperature ?? 0,
        timeoutMs: opts.timeoutMs ?? int(env.SEARCH_LLM_TIMEOUT_MS, 45_000),
      });
      const notes = [...r.notes];
      let body: Pick<ReasoningResponse, "answer" | "claims" | "status" | "rationale" | "confidence" | "conflicts">;
      if (json) {
        const o = replyObject(r) ?? parseJsonObject(r.text);
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
  const payload = JSON.stringify([model, PROMPT_VERSION, req.task ?? "answer", req.reasoning, req.maxTokens, req.lang ?? "en", req.question, req.evidence, req.context ?? []]);
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
