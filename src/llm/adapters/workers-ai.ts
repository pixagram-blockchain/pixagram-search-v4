// Workers AI (the AI binding). The only place that knows how each Workers AI model wants its
// request and what its reply looks like (model.ts says which style a model has).
//
//   style        request                                                   reply
//   openai-chat  messages, response_format.json_schema = {name, schema},   choices[0].message.content
//                chat_template_kwargs / reasoning_effort for reasoning
//   legacy-chat  messages, response_format.json_schema = the schema         {response, usage}
//   gpt-oss      as legacy-chat, plus "Reasoning: <level>" in the system    either of the above, or the
//                message (it always reasons; "none" runs as "low")         Responses API's output[]
//
// Reasoning text (reasoning_content, reasoning items) is never read into the result.
// The binding returns the raw body stream when a reply's content type is not exactly
// application/json; lib/ai.ts reads those (JSON or server-sent events).

import type { Env } from "../../env";
import { aiReply, replyShape } from "../../lib/ai";
import { effectiveReasoning, reasoningAllowance, type ModelSpec } from "../model";
import { LlmError, type ChatMessage, type CompletionRequest, type CompletionResult, type LlmProvider, type ReasoningLevel, type Usage } from "../provider";

type AiRunner = { run: (model: string, input: unknown) => Promise<unknown> };

/** response_format for a JSON schema, in the shape the model's input schema takes. */
export function jsonSchemaFormat(spec: Pick<ModelSpec, "style">, name: string, schema: object): { type: "json_schema"; json_schema: object } {
  return { type: "json_schema", json_schema: spec.style === "openai-chat" ? { name, schema } : schema };
}

/**
 * Request fields that set a model's reasoning. Sent only to the models that take them: the others'
 * input schemas do not have these fields.
 */
export function reasoningFields(spec: Pick<ModelSpec, "reasoning">, level: ReasoningLevel): Record<string, unknown> {
  switch (spec.reasoning) {
    case "chat_template":
      return { chat_template_kwargs: { enable_thinking: level !== "none" } };
    case "chat_template_low":
      if (level === "none") return { chat_template_kwargs: { enable_thinking: false } };
      return { chat_template_kwargs: level === "low" ? { enable_thinking: true, low_effort: true } : { enable_thinking: true } };
    case "kimi_effort":
      return { reasoning_effort: level === "none" ? "none" : "high" };
    default:
      return {};
  }
}

/** Messages in the binding's form: images as image_url parts; gpt-oss gets its reasoning line. */
export function bindingMessages(spec: Pick<ModelSpec, "reasoning">, messages: ChatMessage[], level: ReasoningLevel): Array<{ role: string; content: unknown }> {
  const out = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? { type: "text", text: p.text } : { type: "image_url", image_url: { url: p.dataUri } })),
  }));
  if (spec.reasoning === "system_prompt") {
    const line = `Reasoning: ${level === "none" ? "low" : level}`;
    const sys = out.find((m) => m.role === "system");
    if (sys && typeof sys.content === "string") sys.content = `${sys.content}\n${line}`;
    else out.unshift({ role: "system", content: line });
  }
  return out;
}

/** The binding input for a completion request. */
export function bindingInput(spec: ModelSpec, req: CompletionRequest, env?: Env): Record<string, unknown> {
  const level = effectiveReasoning(spec, req.reasoning);
  const reasons = spec.reasoning !== "none" && level !== "none";
  return {
    messages: bindingMessages(spec, req.messages, level),
    ...(req.json ? { response_format: jsonSchemaFormat(spec, req.json.name, req.json.schema) } : {}),
    max_tokens: Math.max(16, Math.floor(req.maxOutputTokens) + (reasons ? reasoningAllowance(env, level) : 0)),
    temperature: req.temperature ?? 0,
    ...reasoningFields(spec, level),
  };
}

/** Text parts of a content value (string, or an array of {type, text} parts). */
function contentText(c: unknown): string | null {
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    const parts = c.map((p) => (p && typeof p === "object" && typeof (p as any).text === "string" && (p as any).type !== "reasoning" ? (p as any).text : "")).filter(Boolean);
    return parts.length ? parts.join("") : null;
  }
  return null;
}

/**
 * The visible answer of a reply in any of the shapes Workers AI uses. An object `response`
 * (JSON mode on legacy models) is kept as the parsed reply too.
 */
export function replyContent(r: unknown): { text: string | null; json?: unknown; finishReason: string | null } {
  if (typeof r === "string") return { text: r, finishReason: null };
  if (!r || typeof r !== "object") return { text: null, finishReason: null };
  const o = r as Record<string, any>;
  const finish = (o.choices?.[0]?.finish_reason ?? o.finish_reason ?? o.status ?? null) as string | null;
  const res = o.response ?? o.result?.response;
  if (typeof res === "string") return { text: res, finishReason: finish };
  if (res && typeof res === "object") return { text: JSON.stringify(res), json: res, finishReason: finish };
  const msg = o.choices?.[0]?.message;
  if (msg) {
    const t = contentText(msg.content);
    if (t !== null) return { text: t, finishReason: finish };
  }
  if (typeof o.output_text === "string") return { text: o.output_text, finishReason: finish };
  if (Array.isArray(o.output)) {
    // Responses API: reasoning items are skipped, message items carry output_text parts
    const texts = o.output.filter((x: any) => x?.type === "message").map((x: any) => contentText(x.content)).filter(Boolean);
    if (texts.length) return { text: texts.join("\n"), finishReason: finish };
  }
  return { text: null, finishReason: finish };
}

/** Token counts in either the chat-completions or the Responses API form. */
export function replyUsage(r: unknown): Usage | null {
  const u = (r as any)?.usage;
  if (!u || typeof u !== "object") return null;
  const input = Number(u.prompt_tokens ?? u.input_tokens);
  const output = Number(u.completion_tokens ?? u.output_tokens);
  if (!Number.isFinite(input) && !Number.isFinite(output)) return null;
  const reasoning = Number(u.completion_tokens_details?.reasoning_tokens ?? u.output_tokens_details?.reasoning_tokens);
  return { inputTokens: Number.isFinite(input) ? input : 0, outputTokens: Number.isFinite(output) ? output : 0, ...(Number.isFinite(reasoning) ? { reasoningTokens: reasoning } : {}) };
}

/** An error the binding raises for an input its schema refuses (as opposed to capacity or network). */
const INPUT_REFUSED = /\b(5006|5007|invalid input|input validation|oneOf|additional propert|unknown (field|propert)|not allowed|required propert)/i;
/** Transient: capacity, rate limits, timeouts, 5xx. */
const TRANSIENT = /\b(3040|3043|capacity|rate limit|too many|timed? ?out|temporar|unavailable|overloaded|502|503|504)\b/i;

export class WorkersAiProvider implements LlmProvider {
  readonly id = "workers-ai";
  constructor(private readonly env: Env) {}

  async complete(req: CompletionRequest, spec: ModelSpec): Promise<CompletionResult> {
    const ai = this.env.AI as unknown as AiRunner;
    if (!ai?.run) throw new LlmError("the AI binding is not configured", spec.id, false);
    const notes: string[] = [];
    const level = effectiveReasoning(spec, req.reasoning);
    if (level !== req.reasoning && spec.reasoning !== "none") notes.push(`${spec.id} runs reasoning "${req.reasoning}" as "${level}"`);
    const input = bindingInput(spec, req, this.env);
    const t0 = Date.now();
    let raw: unknown;
    try {
      raw = await withTimeout(ai.run(spec.name, input), req.timeoutMs, spec.id);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      // A model whose input schema changed may refuse a reasoning field: once more without it.
      const optional = Object.keys(reasoningFields(spec, level));
      if (optional.length && INPUT_REFUSED.test(msg)) {
        for (const k of optional) delete (input as Record<string, unknown>)[k];
        notes.push(`${spec.id} refused ${optional.join(", ")}; asked again without (${msg.slice(0, 120)})`);
        try {
          raw = await withTimeout(ai.run(spec.name, input), req.timeoutMs, spec.id);
        } catch (e2) {
          throw toLlmError(e2, spec.id);
        }
      } else throw toLlmError(e, spec.id);
    }
    let reply: unknown;
    try {
      reply = await aiReply(raw);
    } catch (e) {
      throw new LlmError(`${spec.id}: ${e instanceof Error ? e.message : String(e)}`, spec.id, true);
    }
    const c = replyContent(reply);
    if (c.text === null && c.json === undefined) notes.push(`${spec.id} reply had no answer (${replyShape(reply)})`);
    return { text: c.text, json: c.json, model: spec.id, provider: this.id, usage: replyUsage(reply), finishReason: c.finishReason, latencyMs: Date.now() - t0, notes };
  }
}

function toLlmError(e: unknown, model: string): LlmError {
  if (e instanceof LlmError) return e;
  const msg = e instanceof Error ? e.message : String(e);
  return new LlmError(`${model}: ${msg.slice(0, 300)}`, model, TRANSIENT.test(msg));
}

async function withTimeout<T>(p: Promise<T>, ms: number | undefined, model: string): Promise<T> {
  if (!ms || ms <= 0) return p;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([p, new Promise<never>((_, reject) => (timer = setTimeout(() => reject(new LlmError(`${model}: no reply within ${ms} ms`, model, true)), ms)))]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}
