// Any OpenAI-compatible chat-completions endpoint (OpenAI, a hosted open-model provider, or a local
// server such as llama.cpp or vLLM), so the engine is not tied to one provider (spec §3, §18).
//
// Configured in SEARCH_LLM_ENDPOINTS, a JSON object keyed by endpoint name:
//   {"openai": {"base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY", "reasoning": "reasoning_effort"},
//    "local":  {"base_url": "http://127.0.0.1:8080/v1", "json": "json_object"}}
// and used as "<endpoint>:<model>", e.g. SEARCH_REASONING_MODEL=openai:gpt-4.1-mini. The key is read
// from the named secret; it is never part of the configuration itself.

import type { Env } from "../../env";
import { endpoints, reasoningAllowance, type EndpointConfig, type ModelSpec } from "../model";
import { LlmError, type ChatMessage, type CompletionRequest, type CompletionResult, type LlmProvider } from "../provider";
import { replyContent, replyUsage } from "./workers-ai";

export function openAiMessages(messages: ChatMessage[]): Array<{ role: string; content: unknown }> {
  return messages.map((m) => ({
    role: m.role,
    content: typeof m.content === "string" ? m.content : m.content.map((p) => (p.type === "text" ? { type: "text", text: p.text } : { type: "image_url", image_url: { url: p.dataUri } })),
  }));
}

export function openAiBody(spec: ModelSpec, ep: EndpointConfig, req: CompletionRequest, env?: Env): Record<string, unknown> {
  const reasons = ep.reasoning === "reasoning_effort" && req.reasoning !== "none";
  const body: Record<string, unknown> = {
    model: spec.name,
    messages: openAiMessages(req.messages),
    max_tokens: Math.max(16, Math.floor(req.maxOutputTokens) + (reasons ? reasoningAllowance(env, req.reasoning) : 0)),
    temperature: req.temperature ?? 0,
  };
  if (reasons) body.reasoning_effort = req.reasoning;
  if (req.json && ep.json !== "none") {
    body.response_format = ep.json === "json_object" ? { type: "json_object" } : { type: "json_schema", json_schema: { name: req.json.name, schema: req.json.schema, strict: false } };
  }
  return body;
}

export class OpenAiCompatibleProvider implements LlmProvider {
  readonly id: string;
  constructor(private readonly env: Env, private readonly endpoint: string) {
    this.id = `openai-compatible:${endpoint}`;
  }

  async complete(req: CompletionRequest, spec: ModelSpec): Promise<CompletionResult> {
    const ep = endpoints(this.env)[this.endpoint];
    if (!ep) throw new LlmError(`no endpoint "${this.endpoint}" in SEARCH_LLM_ENDPOINTS`, spec.id, false);
    const key = ep.api_key_env ? (this.env as unknown as Record<string, unknown>)[ep.api_key_env] : undefined;
    if (ep.api_key_env && typeof key !== "string") throw new LlmError(`secret ${ep.api_key_env} is not set`, spec.id, false);
    const headers: Record<string, string> = { "content-type": "application/json", accept: "application/json" };
    if (typeof key === "string" && key) headers.authorization = `Bearer ${key}`;
    const t0 = Date.now();
    let res: Response;
    try {
      res = await fetch(`${ep.base_url.replace(/\/+$/, "")}/chat/completions`, {
        method: "POST",
        headers,
        body: JSON.stringify(openAiBody(spec, ep, req, this.env)),
        signal: AbortSignal.timeout(req.timeoutMs && req.timeoutMs > 0 ? req.timeoutMs : 120_000),
      });
    } catch (e) {
      throw new LlmError(`${spec.id}: ${e instanceof Error ? e.message : String(e)}`, spec.id, true);
    }
    const text = await res.text();
    if (!res.ok) throw new LlmError(`${spec.id}: HTTP ${res.status} ${text.slice(0, 200)}`, spec.id, res.status === 429 || res.status >= 500);
    let reply: unknown;
    try {
      reply = JSON.parse(text);
    } catch {
      throw new LlmError(`${spec.id}: reply is not JSON (${text.slice(0, 120)})`, spec.id, true);
    }
    const c = replyContent(reply);
    return { text: c.text, json: c.json, model: spec.id, provider: this.id, usage: replyUsage(reply), finishReason: c.finishReason, latencyMs: Date.now() - t0, notes: [] };
  }
}
