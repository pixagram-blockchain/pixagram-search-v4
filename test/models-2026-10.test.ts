// v4.8.1: the models the deployed configuration moved to (GLM-5.3 and GLM-5.3-Flash for reasoning
// and help, Qwen3-Embedding for the documentation vectors) and the ones the table learnt alongside
// (Qwen3.8, DeepSeek V4): their request shapes (reasoning_effort in each model's own levels, the
// {name, schema} JSON shape), the levels they can and cannot honour, their prices, the query-side
// instruction of Qwen3-Embedding, and the "chat" description backend (VLM_MODEL).
//
// FakeAI does not validate inputs: the shapes asserted here are those of the models' Workers AI
// input schemas (October 2026). What is not verified live is listed in README-V4 "Models,
// October 2026".

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { codecsReady } from "./helpers";
import { FakeExec, FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installFetch, seededVector, type ChainFixture } from "./harness/net";
import { installGitHub, type FakeRepo } from "./harness/github";
import { base64Encode } from "../src/lib/bytes";
import { encodePng, type RgbaImage } from "../src/enrich/decode";
import { handleEnrichBatch } from "../src/enrich/consumer";
import { ingestPost } from "../src/chain/ingest";
import { chatVlmModel, describeImage, VLM_BACKENDS, vlmConfigError } from "../src/enrich/describe";
import { bindingInput, jsonSchemaFormat, reasoningFields } from "../src/llm/adapters/workers-ai";
import { callCost, effectiveReasoning, KNOWN_MODELS, KNOWN_PRICES, modelSpec, reasoningAllowance } from "../src/llm/model";
import { complete } from "../src/llm/provider";
import { configuredReasoningModels, modelFor } from "../src/llm/router";
import { DEFAULT_DOCS_QUERY_INSTRUCTION, docsEmbedInput, embedQuestion } from "../src/docs/vectors";
import { syncDocs } from "../src/docs/sync";
import { retrieveDocs } from "../src/help/retrieve";
import { app } from "../src/api";
import type { EnrichMessage } from "../src/env";

const GLM = "@cf/zai-org/glm-5.3";
const GLM_FLASH = "@cf/zai-org/glm-5.3-flash";
const QWEN38 = "@cf/qwen/qwen3.8-27b";
const DS_FLASH = "@cf/deepseek-ai/deepseek-v4-flash-0731";
const DS_PRO = "@cf/deepseek-ai/deepseek-v4-pro-0813";
const QWEN_EMBED = "@cf/qwen/qwen3-embedding-0.6b";
const BGE = "@cf/baai/bge-m3";
const GEMMA = "@cf/google/gemma-4-26b-a4b-it";
const IMG = "data:image/png;base64,AAAA";
const TOKEN = "admin-secret";
const DIM = 32;

/** OpenAI-style reply, as every openai-chat model answers on Workers AI. */
const chat = (content: unknown, extra: object = {}) => ({ choices: [{ message: { role: "assistant", content: typeof content === "string" ? content : JSON.stringify(content) }, finish_reason: "stop" }], ...extra });

afterEach(() => vi.unstubAllGlobals());

describe("request shapes of the October 2026 models", () => {
  it("GLM-5.3 and GLM-5.3-Flash: reasoning_effort low | high | max, never off (the engine's 'none' runs as low), {name, schema} JSON", () => {
    for (const id of [GLM, GLM_FLASH]) {
      const spec = modelSpec(id);
      expect(spec).toMatchObject({ provider: "workers-ai", style: "openai-chat", reasoning: "effort", canDisableReasoning: false, known: true });
      expect(reasoningFields(spec, "none")).toEqual({ reasoning_effort: "low" });
      expect(reasoningFields(spec, "low")).toEqual({ reasoning_effort: "low" });
      expect(reasoningFields(spec, "medium")).toEqual({ reasoning_effort: "high" });
      expect(reasoningFields(spec, "high")).toEqual({ reasoning_effort: "max" });
      expect(effectiveReasoning(spec, "none")).toBe("low");
      expect(effectiveReasoning(spec, "medium")).toBe("medium");
      expect(jsonSchemaFormat(spec, "x", { type: "object" })).toEqual({ type: "json_schema", json_schema: { name: "x", schema: { type: "object" } } });
    }
    expect(modelSpec(GLM_FLASH).vision).toBe(true);
    expect(modelSpec(GLM).vision).toBe(false);
    expect(modelSpec(GLM).contextTokens).toBeGreaterThanOrEqual(1_000_000);
  });

  it("DeepSeek V4: reasoning_effort none | low | high | max ('none' really turns it off)", () => {
    for (const id of [DS_FLASH, DS_PRO]) {
      const spec = modelSpec(id);
      expect(spec).toMatchObject({ style: "openai-chat", reasoning: "effort", canDisableReasoning: true, known: true });
      expect(reasoningFields(spec, "none")).toEqual({ reasoning_effort: "none" });
      expect(reasoningFields(spec, "low")).toEqual({ reasoning_effort: "low" });
      expect(reasoningFields(spec, "medium")).toEqual({ reasoning_effort: "high" });
      expect(reasoningFields(spec, "high")).toEqual({ reasoning_effort: "max" });
      expect(effectiveReasoning(spec, "none")).toBe("none");
    }
  });

  it("Qwen3.8: reasoning_effort low | medium | xhigh, off through the chat template; a vision model", () => {
    const spec = modelSpec(QWEN38);
    expect(spec).toMatchObject({ style: "openai-chat", reasoning: "effort", canDisableReasoning: true, vision: true, known: true });
    expect(reasoningFields(spec, "none")).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(reasoningFields(spec, "low")).toEqual({ reasoning_effort: "low" });
    expect(reasoningFields(spec, "medium")).toEqual({ reasoning_effort: "medium" });
    expect(reasoningFields(spec, "high")).toEqual({ reasoning_effort: "xhigh" });
    expect(effectiveReasoning(spec, "none")).toBe("none");
  });

  it("the binding input of a GLM call at 'none' budgets the reasoning it cannot switch off (the low allowance) and carries the images", () => {
    const env = makeEnv({ SEARCH_REASONING_TOKENS: "low:2048,medium:6144,high:16384" } as any);
    const req = {
      model: GLM_FLASH,
      messages: [{ role: "system" as const, content: "sys" }, { role: "user" as const, content: [{ type: "text" as const, text: "describe" }, { type: "image" as const, dataUri: IMG }] }],
      json: { name: "artwork_description", schema: { type: "object" } },
      reasoning: "none" as const,
      maxOutputTokens: 700,
      temperature: 0.2,
    };
    const input = bindingInput(modelSpec(GLM_FLASH, env), req, env);
    expect(input).toMatchObject({ max_tokens: 700 + 2048, temperature: 0.2, reasoning_effort: "low", response_format: { type: "json_schema", json_schema: { name: "artwork_description", schema: { type: "object" } } } });
    expect((input.messages as any[])[1].content).toEqual([{ type: "text", text: "describe" }, { type: "image_url", image_url: { url: IMG } }]);
    expect(reasoningAllowance(env, "high")).toBe(16384);
    // a model that does switch off at 'none' budgets the visible answer only
    expect(bindingInput(modelSpec(DS_FLASH, env), req, env).max_tokens).toBe(700);
    expect(bindingInput(modelSpec(QWEN38, env), req, env)).toMatchObject({ max_tokens: 700, chat_template_kwargs: { enable_thinking: false } });
  });

  it("a GLM reply through the provider: the note that 'none' ran as 'low', the usage priced at the model's rate", async () => {
    const env = makeEnv();
    env._ai.handler = (model) => (model === GLM_FLASH ? chat("Hello.", { usage: { prompt_tokens: 1000, completion_tokens: 500, completion_tokens_details: { reasoning_tokens: 120 } } }) : {});
    const r = await complete(env, { model: GLM_FLASH, messages: [{ role: "user", content: "hi" }], reasoning: "none", maxOutputTokens: 100 });
    expect(r.text).toBe("Hello.");
    expect(r.notes).toContain(`${GLM_FLASH} runs reasoning "none" as "low"`);
    expect(r.usage).toEqual({ inputTokens: 1000, outputTokens: 500, reasoningTokens: 120 });
    expect(env._ai.calls[0].input.reasoning_effort).toBe("low");
    expect(callCost(modelSpec(GLM_FLASH), 1000, 500)).toBeCloseTo((1000 * 0.15 + 500 * 0.5) / 1e6, 12);
  });

  it("prices (Workers AI model pages, October 2026): the GLMs, DeepSeek V4, Qwen3.8 and the embedding model", () => {
    expect(KNOWN_MODELS[GLM].price).toEqual({ input: 1.4, output: 4.4, cachedInput: 0.26 });
    expect(KNOWN_MODELS[GLM_FLASH].price).toEqual({ input: 0.15, output: 0.5, cachedInput: 0.03 });
    expect(KNOWN_MODELS[QWEN38].price).toEqual({ input: 0.45, output: 3.2, cachedInput: 0.05 });
    expect(KNOWN_MODELS[DS_FLASH].price).toEqual({ input: 0.44, output: 1.32, cachedInput: 0.014 });
    expect(KNOWN_MODELS[DS_PRO].price).toEqual({ input: 1.32, output: 3.96, cachedInput: 0.044 });
    expect(KNOWN_PRICES[QWEN_EMBED]).toEqual({ input: 0.0118 });
    // GLM-5.3-Flash is the cheaper reasoning model: below gpt-oss-120b on both sides
    expect(KNOWN_MODELS[GLM_FLASH].price!.input).toBeLessThan(KNOWN_MODELS["@cf/openai/gpt-oss-120b"].price!.input);
    expect(KNOWN_MODELS[GLM_FLASH].price!.output).toBeLessThan(KNOWN_MODELS["@cf/openai/gpt-oss-120b"].price!.output);
  });

  it("the models route lists them with their facts", async () => {
    const env = makeEnv({ ADMIN_TOKEN: TOKEN } as any);
    const res = await app.fetch(new Request("https://search.test/admin/models", { headers: { authorization: `Bearer ${TOKEN}` } }), env, new FakeExec() as unknown as ExecutionContext);
    expect(res.status).toBe(200);
    const body = (await res.json()) as any;
    const known: any[] = body.known ?? body.models ?? [];
    const glm = known.find((m: any) => m.id === GLM_FLASH);
    expect(glm).toMatchObject({ reasoning: "effort", can_disable_reasoning: false, vision: true, price: { input: 0.15, output: 0.5 } });
  });
});

describe("the deployed configuration (wrangler.jsonc, v4.8.1)", () => {
  const vars = async () => {
    const { readFileSync } = await import("node:fs");
    const text = readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8");
    const out: Record<string, string> = {};
    for (const m of text.matchAll(/"([A-Z0-9_]+)":\s*"([^"]*)"/g)) out[m[1]] = m[2];
    return out;
  };

  it("reasoning on GLM-5.3-Flash, the deep band on GLM-5.3, help on GLM-5.3-Flash, planning on Gemma 4", async () => {
    const v = await vars();
    const env = makeEnv(v as any);
    expect(modelFor(env, "planner")).toBe(GEMMA);
    expect(modelFor(env, "help")).toBe(GLM_FLASH);
    expect(configuredReasoningModels(env)).toEqual({ simple: GLM_FLASH, normal: GLM_FLASH, complex: GLM_FLASH, deep: GLM });
    // every deployed chat model is one the table knows, so its request shape is its own
    for (const id of [modelFor(env, "planner"), modelFor(env, "help"), ...Object.values(configuredReasoningModels(env))]) expect(modelSpec(id, env).known).toBe(true);
    // the help model reasons a little in every mode: its 'none' is 'low', budgeted by the raised allowance
    expect(reasoningFields(modelSpec(GLM_FLASH, env), "none")).toEqual({ reasoning_effort: "low" });
    expect(reasoningAllowance(env, "low")).toBe(2048);
    expect(reasoningAllowance(env, "medium")).toBe(6144);
    expect(reasoningAllowance(env, "high")).toBe(16384);
    expect(Number(v.SEARCH_LLM_TIMEOUT_MS)).toBeGreaterThanOrEqual(90_000);
  });

  it("the documentation vectors on Qwen3-Embedding-0.6B, 1024 dimensions as VEC_DOCS; descriptions still on Gemma 4 with the chat model ready", async () => {
    const v = await vars();
    expect(v.DOCS_EMBED_MODEL).toBe(QWEN_EMBED);
    expect(v.DOCS_EMBED_DIM).toBe("1024");
    expect(Number(v.DOCS_MIN_SCORE)).toBeLessThan(0.5); // re-read on the new model: lower than bge-m3's 0.5
    expect(v.VLM_BACKEND).toBe("gemma");
    expect(v.VLM_MODEL).toBe(GLM_FLASH);
    expect(vlmConfigError({ VLM_MODEL: v.VLM_MODEL }, "chat")).toBeNull();
  });
});

describe("Qwen3-Embedding embeds questions with the retrieval instruction", () => {
  /** A bag-of-words embedder that answers both input shapes of the embedding models. */
  function bow(text: string): number[] {
    const v = new Array<number>(DIM).fill(0);
    for (const w of String(text).toLowerCase().split(/[^a-z]+/)) {
      if (w.length < 3) continue;
      let h = 7;
      for (const ch of w) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
      v[h % DIM] += 1;
    }
    const n = Math.hypot(...v) || 1;
    return v.map((x) => x / n);
  }
  const embedder = (_model: string, input: any) => {
    const texts: string[] = input.queries ?? input.text ?? input.documents;
    return { shape: [texts.length, DIM], data: texts.map(bow) };
  };

  it("the input per model and kind: {queries, instruction} for a question on Qwen3, {text} for chunks and for everything on bge-m3", () => {
    expect(docsEmbedInput({ DOCS_EMBED_MODEL: QWEN_EMBED }, ["how do fees work"], "query")).toEqual({ queries: ["how do fees work"], instruction: DEFAULT_DOCS_QUERY_INSTRUCTION });
    expect(docsEmbedInput({ DOCS_EMBED_MODEL: QWEN_EMBED, DOCS_QUERY_INSTRUCTION: "Find the page" }, ["q"], "query")).toEqual({ queries: ["q"], instruction: "Find the page" });
    expect(docsEmbedInput({ DOCS_EMBED_MODEL: QWEN_EMBED }, ["Fees\nSelling\nA sale pays 5%"], "document")).toEqual({ text: ["Fees\nSelling\nA sale pays 5%"] });
    expect(docsEmbedInput({ DOCS_EMBED_MODEL: BGE }, ["how do fees work"], "query")).toEqual({ text: ["how do fees work"] });
    expect(docsEmbedInput({}, ["q"], "query")).toEqual({ text: ["q"] }); // the default model is bge-m3
    expect(DEFAULT_DOCS_QUERY_INSTRUCTION).toMatch(/retrieve/i);
  });

  it("through the index: chunks go as documents, the question as a query, cached under a versioned key; retrieval finds the page", async () => {
    const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", DOCS_EMBED_MODEL: QWEN_EMBED, DOCS_EMBED_DIM: String(DIM), ADMIN_TOKEN: TOKEN } as any);
    (env as any).VEC_DOCS = new FakeVectorize([]);
    const repo: FakeRepo = {
      owner: "pixa",
      repo: "info",
      head: "a".repeat(40),
      files: { "guides/fees.md": "# Fees\n\nMinting an artwork is free.\n\n## Selling\n\nA sale on the marketplace pays a 5% fee to the platform.", "guides/wallet.md": "# Wallet\n\nThe wallet keeps your PXS and your artworks." },
      calls: [],
    };
    installGitHub(repo);
    env._ai.handler = embedder;
    const r = await syncDocs(env);
    expect(r.status).toBe("synced");
    expect(r.embedded).toBeGreaterThan(0);
    const chunkCalls = env._ai.calls.filter((c) => c.model === QWEN_EMBED);
    expect(chunkCalls.length).toBeGreaterThan(0);
    for (const c of chunkCalls) {
      expect(c.input.text).toBeInstanceOf(Array);
      expect(c.input.queries).toBeUndefined();
      expect(c.input.instruction).toBeUndefined();
    }
    env._ai.calls.length = 0;
    const hits = await retrieveDocs(env, "what fee does a sale pay?");
    expect(hits.hits[0].path).toBe("guides/fees.md");
    expect(hits.degraded).toBe(false);
    const q = env._ai.calls.find((c) => c.model === QWEN_EMBED);
    expect(q!.input).toEqual({ queries: ["what fee does a sale pay?"], instruction: DEFAULT_DOCS_QUERY_INSTRUCTION });
    const key = [...env._kv.m.keys()].find((k) => k.startsWith("demb:"));
    expect(key).toBe(`demb:2:${QWEN_EMBED}:what fee does a sale pay?`);
    // the second time the vector comes from KV
    env._ai.calls.length = 0;
    await embedQuestion(env, "What fee does a sale pay?");
    expect(env._ai.calls.length).toBe(0);
  });

  it("the retrieval debug route shows each chunk's lexical coverage, cosine and score (for DOCS_MIN_SCORE)", async () => {
    const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", DOCS_EMBED_MODEL: QWEN_EMBED, DOCS_EMBED_DIM: String(DIM), DOCS_MIN_SCORE: "0.4", ADMIN_TOKEN: TOKEN } as any);
    (env as any).VEC_DOCS = new FakeVectorize([]);
    installGitHub({ owner: "pixa", repo: "info", head: "b".repeat(40), files: { "guides/fees.md": "# Fees\n\nA sale on the marketplace pays a 5% fee to the platform." }, calls: [] });
    env._ai.handler = embedder;
    await syncDocs(env);
    const call = async (path: string) => {
      const res = await app.fetch(new Request(`https://search.test${path}`, { headers: { authorization: `Bearer ${TOKEN}` } }), env, new FakeExec() as unknown as ExecutionContext);
      return { status: res.status, body: (await res.json()) as any };
    };
    expect((await call("/admin/debug/docs")).status).toBe(400);
    const r = await call("/admin/debug/docs?q=marketplace%20fee");
    expect(r.status).toBe(200);
    expect(r.body).toMatchObject({ embed_model: QWEN_EMBED, min_score: 0.4, degraded: false });
    expect(r.body.hits[0]).toMatchObject({ path: "guides/fees.md", lexical: expect.any(Number), cosine: expect.any(Number), score: expect.any(Number) });
    expect(r.body.hits[0].cosine).toBeGreaterThan(0);
  });
});

describe("the chat description backend (VLM_MODEL)", () => {
  it("describes with the configured vision model: image part, {name, schema} JSON, the model's lowest reasoning", async () => {
    const env = makeEnv({ VLM_MODEL: GLM_FLASH } as any);
    env._ai.handler = (model) =>
      model === GLM_FLASH ? chat({ caption: "A lighthouse on a cliff at dusk.", subjects: ["lighthouse"], objects: ["cliff"], style: "landscape", mood: "calm", text_in_image: "", tags: ["lighthouse", "cliff", "dusk"], nsfw: 0 }) : {};
    const r = await describeImage(env, "chat", IMG, { title: "Dusk" });
    expect(r).toMatchObject({ model: GLM_FLASH, status: "ok", description: { caption: "A lighthouse on a cliff at dusk.", subjects: ["lighthouse"], tags: ["lighthouse", "cliff", "dusk"] } });
    const input = env._ai.calls[0].input;
    expect(env._ai.calls[0].model).toBe(GLM_FLASH);
    expect(input.response_format).toEqual({ type: "json_schema", json_schema: { name: "artwork_description", schema: expect.objectContaining({ required: expect.arrayContaining(["caption", "tags"]) }) } });
    expect(input.reasoning_effort).toBe("low");
    expect(input.chat_template_kwargs).toBeUndefined();
    expect(input.messages[1].content).toEqual([expect.objectContaining({ type: "text" }), { type: "image_url", image_url: { url: IMG } }]);
    expect(input.messages[1].content[0].text).toContain('Title: "Dusk"');
    // Qwen3.8 as the chat model: reasoning off through its chat template
    const env2 = makeEnv({ VLM_MODEL: QWEN38 } as any);
    env2._ai.handler = () => chat({ caption: "A lighthouse on a cliff at dusk.", tags: ["lighthouse"] });
    await describeImage(env2, "chat", IMG, {});
    expect(env2._ai.calls[0].input).toMatchObject({ chat_template_kwargs: { enable_thinking: false } });
    expect(env2._ai.calls[0].input.reasoning_effort).toBeUndefined();
  });

  it("the configuration is checked: no VLM_MODEL, not a model id, not a vision model; an override can declare one", () => {
    expect(vlmConfigError({}, "chat")).toBe("VLM_BACKEND=chat needs VLM_MODEL (a vision chat model id)");
    expect(vlmConfigError({ VLM_MODEL: "glm" }, "chat")).toMatch(/not a model id/);
    expect(vlmConfigError({ VLM_MODEL: GLM }, "chat")).toMatch(/not known as a vision model/);
    expect(vlmConfigError({ VLM_MODEL: GLM_FLASH }, "chat")).toBeNull();
    expect(vlmConfigError({ VLM_MODEL: QWEN38 }, "chat")).toBeNull();
    expect(vlmConfigError({ VLM_MODEL: "@cf/acme/new-vlm", SEARCH_MODEL_OVERRIDES: JSON.stringify({ "@cf/acme/new-vlm": { style: "openai-chat", vision: true } }) }, "chat")).toBeNull();
    expect(vlmConfigError({}, "gemma")).toBeNull();
    const env = makeEnv();
    expect(chatVlmModel(env, "gemma")).toBe(GEMMA);
    expect(chatVlmModel(env, "moondream")).toBeNull();
    expect(() => chatVlmModel(env, "chat")).toThrow(/needs VLM_MODEL/);
    expect(VLM_BACKENDS).toContain("chat");
  });

  it("the describe debug route takes ?backend=chat and reports a misconfigured one", async () => {
    const env = makeEnv({ ADMIN_TOKEN: TOKEN } as any);
    const res = await app.fetch(new Request("https://search.test/admin/debug/describe/1?backend=chat", { method: "POST", headers: { authorization: `Bearer ${TOKEN}` } }), env, new FakeExec() as unknown as ExecutionContext);
    expect(res.status).toBe(404); // accepted as a backend; post 1 does not exist here
    expect(((await res.json()) as any).error).toBe("not an artwork");
  });

  describe("in the enrichment pipeline", () => {
    const T0 = Date.UTC(2026, 9, 1) / 1000;
    const iso = (t: number) => new Date(t * 1000).toISOString().slice(0, 19);
    function paint(w: number, h: number, f: (x: number, y: number) => [number, number, number, number]): RgbaImage {
      const data = new Uint8Array(w * h * 4);
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data.set(f(x, y), (y * w + x) * 4);
      return { width: w, height: h, data };
    }
    const checker = () => paint(16, 16, (x, y) => ((x ^ y) & 1 ? [200, 30, 30, 255] : [20, 20, 20, 255]));
    function chainPost(author: string, permlink: string, body: string, title: string) {
      return {
        author, permlink, parent_author: "", parent_permlink: "pixagram", category: "pixagram", title, body,
        json_metadata: JSON.stringify({ app: "pixagram/3.0.2", format: "image", tags: ["test"], description: "" }),
        created: iso(T0), last_update: iso(T0), depth: 0, children: 0, net_votes: 0,
        pending_payout_value: "0.000 PXS", total_payout_value: "0.000 PXS", curator_payout_value: "0.000 PXS",
      };
    }
    async function setup(chain: ChainFixture, over: Record<string, string>) {
      await codecsReady();
      const env = makeEnv({ HF_EMBED_URL: "", EMBED_MODEL: "stub", EMBED_DIM: String(DIM), ADMIN_TOKEN: TOKEN, PLANNER_BACKEND: "rules", ...over } as any);
      installFetch({ rpcUrl: "https://rpc.test", embedUrl: "https://embed.test", chain, embed: (kind, input) => seededVector(`${kind}:${input.slice(-64)}`, DIM) });
      return env;
    }
    async function drain(env: TestEnv, msgs: EnrichMessage[] = env._queue.drain()) {
      await handleEnrichBatch({ queue: "q", messages: msgs.map((body, i) => ({ id: String(i), timestamp: new Date(), attempts: 1, body, ack: () => {}, retry: () => {} })), ackAll() {}, retryAll() {} } as any, env);
    }
    let png: Uint8Array;
    beforeAll(async () => {
      await codecsReady();
      png = await encodePng(checker());
    });

    it("VLM_BACKEND=chat describes with VLM_MODEL and records it on the artwork", async () => {
      const post = chainPost("alice", "art", `data:image/png;base64,${base64Encode(png)}`, "Lighthouse");
      const env = await setup({ posts: [post] as any }, { VLM_BACKEND: "chat", VLM_MODEL: GLM_FLASH });
      env._ai.handler = (model) =>
        model === GLM_FLASH ? chat({ caption: "A lighthouse on a rocky coast at night.", subjects: ["lighthouse"], objects: ["rocks"], tags: ["lighthouse", "coast"], style: "landscape", mood: "calm", text_in_image: "", nsfw: 0 }) : {};
      await ingestPost(env, post as any, null, "test");
      await drain(env);
      const a = await env.DB.prepare("SELECT vlm_model, ai_caption, ai_status, describe_hash IS NOT NULL AS done FROM artworks WHERE post_id = 1").first<any>();
      expect(a).toMatchObject({ vlm_model: GLM_FLASH, ai_caption: "A lighthouse on a rocky coast at night.", ai_status: "ok", done: 1 });
      expect((await env.DB.prepare("SELECT status FROM jobs WHERE post_id = 1 AND stage = 'describe'").first<any>()).status).toBe("done");
      expect(env._ai.calls.filter((c) => c.model === GLM_FLASH).length).toBe(1);
    });

    it("VLM_BACKEND=chat without a usable VLM_MODEL skips the stage with the reason instead of failing every image", async () => {
      const post = chainPost("alice", "art", `data:image/png;base64,${base64Encode(png)}`, "Lighthouse");
      const env = await setup({ posts: [post] as any }, { VLM_BACKEND: "chat", VLM_MODEL: "" });
      env._ai.handler = () => {
        throw new Error("no model should be called");
      };
      await ingestPost(env, post as any, null, "test");
      await drain(env);
      const job = await env.DB.prepare("SELECT status, error FROM jobs WHERE post_id = 1 AND stage = 'describe'").first<any>();
      expect(job).toEqual({ status: "skipped", error: "VLM_BACKEND=chat needs VLM_MODEL (a vision chat model id)" });
      expect((await env.DB.prepare("SELECT ai_caption FROM artworks WHERE post_id = 1").first<any>()).ai_caption).toBeNull();
    });
  });
});
