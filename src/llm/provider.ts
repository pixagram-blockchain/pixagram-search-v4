// The one door to language models (spec §18). Every call — the reasoning model, the planner, help
// answers, image descriptions — goes through a provider here, so a model is changed by
// configuration and nothing outside src/llm/adapters/ knows how a given model wants its request.
//
//   caller ──CompletionRequest──► provider (adapters/workers-ai.ts, adapters/openai-compatible.ts)
//          ◄─CompletionResult───  visible text only: a model's reasoning is read and dropped here,
//                                 never returned (spec §21: no chain-of-thought leaves the engine)
//
// A request states how much the model may reason (none … high) and how long its visible answer
// may be, separately (spec §21). The adapter turns that into whatever the model takes: a template
// flag, an effort field, a system-prompt line, and a token limit that leaves room for both.

import type { Env } from "../env";
import { modelSpec, type ModelSpec } from "./model";
import { WorkersAiProvider } from "./adapters/workers-ai";
import { OpenAiCompatibleProvider } from "./adapters/openai-compatible";

export type ReasoningLevel = "none" | "low" | "medium" | "high";
export const REASONING_LEVELS: ReasoningLevel[] = ["none", "low", "medium", "high"];
export const isReasoningLevel = (x: unknown): x is ReasoningLevel => typeof x === "string" && (REASONING_LEVELS as string[]).includes(x);

export type ContentPart = { type: "text"; text: string } | { type: "image"; dataUri: string };

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string | ContentPart[];
}

export interface CompletionRequest {
  /** the configured model id: "@cf/openai/gpt-oss-120b", "openai:gpt-4.1-mini", "local:qwen2.5-3b" … */
  model: string;
  messages: ChatMessage[];
  /** the reply must be JSON following this schema (name + JSON Schema) */
  json?: { name: string; schema: object };
  /** how much the model may think before it answers */
  reasoning: ReasoningLevel;
  /** tokens for the visible answer; the adapter adds the reasoning allowance for models that reason */
  maxOutputTokens: number;
  temperature?: number;
  timeoutMs?: number;
}

export interface Usage {
  inputTokens: number;
  outputTokens: number;
  /** output tokens spent reasoning, when the provider reports them (included in outputTokens) */
  reasoningTokens?: number;
}

export interface CompletionResult {
  /** the answer as text (JSON text in JSON mode), never the model's reasoning; null when empty */
  text: string | null;
  /** the reply already parsed, when the provider returned an object for a JSON request */
  json?: unknown;
  model: string;
  provider: string;
  usage: Usage | null;
  finishReason: string | null;
  /** wall time of the provider call (spec §54: measured apart from the engine's own latency) */
  latencyMs: number;
  /** what the adapter had to adapt (a reasoning level the model lacks, a retried request …) */
  notes: string[];
}

export interface LlmProvider {
  readonly id: string;
  complete(req: CompletionRequest, spec: ModelSpec): Promise<CompletionResult>;
}

/** A model call that failed: transient (retry later) or not (the request or the model is wrong). */
export class LlmError extends Error {
  constructor(message: string, public readonly model: string, public readonly retryable: boolean) {
    super(message);
    this.name = "LlmError";
  }
}

const PROVIDERS = new WeakMap<object, Map<string, LlmProvider>>();

/** The provider that serves a model id, per Worker environment (one instance per binding set). */
export function providerFor(env: Env, spec: ModelSpec): LlmProvider {
  const key = env as unknown as object;
  let m = PROVIDERS.get(key);
  if (!m) PROVIDERS.set(key, (m = new Map()));
  const id = spec.provider === "workers-ai" ? "workers-ai" : `openai-compatible:${spec.endpoint ?? ""}`;
  let p = m.get(id);
  if (!p) {
    p = spec.provider === "workers-ai" ? new WorkersAiProvider(env) : new OpenAiCompatibleProvider(env, spec.endpoint ?? "");
    m.set(id, p);
  }
  return p;
}

/** One completion with any configured model. */
export async function complete(env: Env, req: CompletionRequest): Promise<CompletionResult> {
  const spec = modelSpec(req.model, env);
  return providerFor(env, spec).complete(req, spec);
}

/**
 * The JSON object in a model's text: the whole text, or the outermost {...} inside it (code fences
 * and a sentence around it are common). Null when there is none.
 */
export function parseJsonObject(text: string | null | undefined): Record<string, unknown> | null {
  if (typeof text !== "string") return null;
  const t = text.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
  for (const candidate of [t, t.slice(t.indexOf("{"), t.lastIndexOf("}") + 1)]) {
    if (!candidate.startsWith("{")) continue;
    try {
      const o = JSON.parse(candidate);
      if (o && typeof o === "object" && !Array.isArray(o)) return o as Record<string, unknown>;
    } catch {
      // try the next form
    }
  }
  return null;
}

/** The reply of a JSON-mode completion as an object, whichever form the provider gave it in. */
export function replyObject(r: CompletionResult): Record<string, unknown> | null {
  if (r.json && typeof r.json === "object" && !Array.isArray(r.json)) return r.json as Record<string, unknown>;
  return parseJsonObject(r.text);
}
