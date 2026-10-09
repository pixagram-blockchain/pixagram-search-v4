// What the engine knows about each model: which provider serves it, which request style it takes,
// how its reasoning is controlled, its context window and its price (for cost accounting, spec §53).
// The adapters read this; nothing else branches on a model's name.
//
// Model ids:
//   @cf/<author>/<model>, @hf/<author>/<model>   Workers AI, through the AI binding
//   <endpoint>:<model>                            an OpenAI-compatible HTTP endpoint configured in
//                                                 SEARCH_LLM_ENDPOINTS (e.g. "openai:gpt-4.1-mini",
//                                                 "local:qwen2.5-3b-instruct")
//
// Workers AI request styles (checked against the models' input schemas, October 2026):
//   openai-chat  Nemotron 3, Gemma 4, Kimi K2.6: OpenAI chat completions; json_schema = {name, schema};
//                reasoning through chat_template_kwargs (Nemotron also has a low-effort mode) or,
//                for Kimi, reasoning_effort ("none" | "high"); replies as choices[0].message
//   legacy-chat  Llama 3.3, Llama 4 Scout, Qwen 3, Mistral: messages; json_schema = the schema itself;
//                replies as {response, usage}
//   gpt-oss      gpt-oss-120b / -20b: messages like legacy-chat; always reasons; the effort is set
//                with a "Reasoning: low|medium|high" line in the system prompt (OpenAI's convention
//                for these models; the binding's schema has no field for it)
// SEARCH_MODEL_OVERRIDES (JSON) describes models this table does not know, or corrects it.

import type { Env } from "../env";
import type { ReasoningLevel } from "./provider";

export type RequestStyle = "openai-chat" | "legacy-chat" | "gpt-oss";

/** How a model's reasoning is switched and sized. */
export type ReasoningControl =
  | "none" //               the model does not reason: every level means "answer directly"
  | "chat_template" //      chat_template_kwargs.enable_thinking on/off
  | "chat_template_low" //  …plus low_effort for the "low" level (Nemotron 3)
  | "kimi_effort" //        reasoning_effort "none" | "high" (other levels map to "high")
  | "system_prompt" //      "Reasoning: low|medium|high" in the system message (gpt-oss: always on)
  | "reasoning_effort"; //  OpenAI-style reasoning_effort "low" | "medium" | "high" (HTTP endpoints)

export interface ModelSpec {
  /** the id as configured */
  id: string;
  provider: "workers-ai" | "openai-compatible";
  /** SEARCH_LLM_ENDPOINTS key, for openai-compatible models */
  endpoint?: string;
  /** the name the provider knows the model by */
  name: string;
  style: RequestStyle;
  reasoning: ReasoningControl;
  /** whether "none" really turns reasoning off (gpt-oss cannot: none becomes low) */
  canDisableReasoning: boolean;
  contextTokens: number;
  /** US dollars per million tokens */
  price?: { input: number; output: number; cachedInput?: number };
  vision: boolean;
  /** known to this table (false: guessed from the id) */
  known: boolean;
}

type Known = Omit<ModelSpec, "id" | "provider" | "name" | "known" | "endpoint">;

const OPENAI_CHAT = (o: Partial<Known>): Known => ({ style: "openai-chat", reasoning: "chat_template", canDisableReasoning: true, contextTokens: 128_000, vision: false, ...o });
const LEGACY = (o: Partial<Known>): Known => ({ style: "legacy-chat", reasoning: "none", canDisableReasoning: true, contextTokens: 32_000, vision: false, ...o });
const GPT_OSS = (o: Partial<Known>): Known => ({ style: "gpt-oss", reasoning: "system_prompt", canDisableReasoning: false, contextTokens: 128_000, vision: false, ...o });

/** Workers AI models with their facts as published in October 2026 (prices: Workers AI model pages). */
export const KNOWN_MODELS: Record<string, Known> = {
  "@cf/openai/gpt-oss-120b": GPT_OSS({ price: { input: 0.35, output: 0.75 } }),
  "@cf/openai/gpt-oss-20b": GPT_OSS({ price: { input: 0.2, output: 0.3 } }),
  "@cf/nvidia/nemotron-3-120b-a12b": OPENAI_CHAT({ reasoning: "chat_template_low", contextTokens: 256_000, price: { input: 0.5, output: 1.5 } }),
  "@cf/google/gemma-4-26b-a4b-it": OPENAI_CHAT({ contextTokens: 256_000, vision: true, price: { input: 0.1, output: 0.3 } }),
  "@cf/moonshotai/kimi-k2.6": OPENAI_CHAT({ reasoning: "kimi_effort", contextTokens: 262_144, vision: true, price: { input: 0.95, output: 4.0, cachedInput: 0.16 } }),
  "@cf/meta/llama-3.3-70b-instruct-fp8-fast": LEGACY({ contextTokens: 24_000, price: { input: 0.293, output: 2.253 } }),
  "@cf/meta/llama-4-scout-17b-16e-instruct": LEGACY({ contextTokens: 131_000, vision: true, price: { input: 0.27, output: 0.85 } }),
  "@cf/meta/llama-3.1-8b-instruct-fp8": LEGACY({ contextTokens: 32_000, price: { input: 0.152, output: 0.287 } }),
  "@cf/qwen/qwen3-30b-a3b-fp8": LEGACY({ contextTokens: 32_768, price: { input: 0.0509, output: 0.335 } }),
  "@cf/mistralai/mistral-small-3.1-24b-instruct": LEGACY({ contextTokens: 128_000, price: { input: 0.351, output: 0.555 } }),
};

/** Non-chat Workers AI models the engine prices (spec §53): the reranker and the embeddings. */
export const KNOWN_PRICES: Record<string, { input: number; output?: number }> = {
  "@cf/baai/bge-reranker-base": { input: 0.00311 },
  "@cf/baai/bge-m3": { input: 0.0118 },
};

export interface EndpointConfig {
  /** e.g. https://api.openai.com/v1, http://127.0.0.1:8080/v1 */
  base_url: string;
  /** name of the secret (env var) holding the bearer token; none for a local server */
  api_key_env?: string;
  /** how a JSON reply is asked for: OpenAI's json_schema (default), json_object, or nothing */
  json?: "json_schema" | "json_object" | "none";
  /** how reasoning is set: OpenAI's reasoning_effort, or not at all (default) */
  reasoning?: "reasoning_effort" | "none";
  context_tokens?: number;
  price?: { input: number; output: number };
}

function parseJsonVar<T>(v: unknown): T | null {
  if (v && typeof v === "object") return v as T; // wrangler can pass JSON vars as objects
  if (typeof v !== "string" || !v.trim()) return null;
  try {
    return JSON.parse(v) as T;
  } catch {
    return null;
  }
}

/** The OpenAI-compatible endpoints configured for this Worker (SEARCH_LLM_ENDPOINTS). */
export function endpoints(env: Env): Record<string, EndpointConfig> {
  const raw = parseJsonVar<Record<string, EndpointConfig>>(env.SEARCH_LLM_ENDPOINTS) ?? {};
  const out: Record<string, EndpointConfig> = {};
  for (const [k, v] of Object.entries(raw)) if (v && typeof v.base_url === "string" && /^https?:\/\//.test(v.base_url)) out[k.toLowerCase()] = v;
  return out;
}

const MEMO = new WeakMap<object, Map<string, ModelSpec>>();

/** Everything the adapters need about a model id. Unknown Workers AI models are taken as legacy chat models. */
export function modelSpec(id: string, env?: Env): ModelSpec {
  const model = id.trim();
  const memo = env ? (MEMO.get(env as unknown as object) ?? new Map<string, ModelSpec>()) : null;
  if (env && memo && !MEMO.has(env as unknown as object)) MEMO.set(env as unknown as object, memo);
  const hit = memo?.get(model);
  if (hit) return hit;
  const overrides = env ? (parseJsonVar<Record<string, Partial<Known>>>(env.SEARCH_MODEL_OVERRIDES) ?? {}) : {};
  let spec: ModelSpec;
  if (/^@(cf|hf)\//.test(model)) {
    const known = KNOWN_MODELS[model];
    spec = { id: model, provider: "workers-ai", name: model, known: !!known, ...(known ?? LEGACY({})), ...(overrides[model] ?? {}) };
  } else {
    const k = model.indexOf(":");
    const endpoint = k > 0 ? model.slice(0, k).toLowerCase() : "";
    const name = k > 0 ? model.slice(k + 1) : model;
    const ep = env ? endpoints(env)[endpoint] : undefined;
    spec = {
      id: model,
      provider: "openai-compatible",
      endpoint,
      name,
      style: "openai-chat",
      reasoning: ep?.reasoning === "reasoning_effort" ? "reasoning_effort" : "none",
      canDisableReasoning: true,
      contextTokens: ep?.context_tokens ?? 32_000,
      price: ep?.price,
      vision: false,
      known: !!ep,
      ...(overrides[model] ?? {}),
    };
  }
  memo?.set(model, spec);
  return spec;
}

/** Whether a string is a model id this engine can call (Workers AI, or a configured endpoint). */
export function isModelId(id: string, env: Env): boolean {
  const m = id.trim();
  if (/^@(cf|hf)\/[\w.-]+\/[\w.-]+$/.test(m)) return true;
  const k = m.indexOf(":");
  return k > 0 && /^[\w.\-/:]+$/.test(m.slice(k + 1)) && !!endpoints(env)[m.slice(0, k).toLowerCase()];
}

/** Tokens a model may spend reasoning, per level, on top of the visible answer (SEARCH_REASONING_TOKENS). */
export function reasoningAllowance(env: Env | undefined, level: ReasoningLevel): number {
  const dflt: Record<ReasoningLevel, number> = { none: 0, low: 1024, medium: 4096, high: 12_288 };
  const raw = env?.SEARCH_REASONING_TOKENS ?? "";
  for (const part of String(raw).split(",")) {
    const [k, v] = part.split(":").map((s) => s.trim());
    if (k === level && Number.isFinite(Number(v)) && Number(v) >= 0) return Math.floor(Number(v));
  }
  return dflt[level];
}

/** The level a model actually runs at: one it cannot honour is mapped to the nearest it can. */
export function effectiveReasoning(spec: ModelSpec, level: ReasoningLevel): ReasoningLevel {
  if (spec.reasoning === "none") return "none";
  if (level === "none" && !spec.canDisableReasoning) return "low";
  if (spec.reasoning === "kimi_effort" && level !== "none") return "high";
  if (spec.reasoning === "chat_template" && level !== "none") return "medium"; // on/off only: one "on" level
  return level;
}

/** US dollars for a call (null when the price is unknown). */
export function callCost(spec: Pick<ModelSpec, "price">, inputTokens: number, outputTokens: number): number | null {
  if (!spec.price) return null;
  return (inputTokens * spec.price.input + outputTokens * spec.price.output) / 1e6;
}
