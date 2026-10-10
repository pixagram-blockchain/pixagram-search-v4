// Swapping Workers AI models: Gemma 4 as the description model, Nemotron 3 / Gemma 4 as the
// help or planner model (their JSON-schema shape, reasoning off, OpenAI-style replies), and the
// admin route that answers a help question with another model for comparison.
//
// FakeAI does not validate inputs, so the request shapes each model's Workers AI input schema
// requires are asserted here (Gemma 4 / Nemotron 3: json_schema = {name, schema}; Llama 3.3 and
// Scout: the schema itself).

import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeExec, FakeVectorize, makeEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { describeImage, GEMMA_MODEL, noPixelArt, normalizeDescription, SCOUT_MODEL, VLM_BACKENDS } from "../src/enrich/describe";
import { jsonSchemaFormat, reasoningFields } from "../src/llm/adapters/workers-ai";
import { modelSpec } from "../src/llm/model";
import { modelFor } from "../src/llm/router";
import { llmPlan } from "../src/search/llm-planner";
import { planQuery } from "../src/search/planner";
import { syncDocs } from "../src/docs/sync";
import { answerHelp } from "../src/help/answer";
import { app } from "../src/api";

const IMG = "data:image/png;base64,AAAA";
const NEMOTRON = "@cf/nvidia/nemotron-3-120b-a12b";
const LLAMA = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const EMBED = "@cf/baai/bge-m3";
const TOKEN = "admin-secret";
const DIM = 32;

/** OpenAI-style reply, as Nemotron 3 and Gemma 4 answer on Workers AI. */
const chat = (content: unknown) => ({ choices: [{ message: { role: "assistant", content: typeof content === "string" ? content : JSON.stringify(content) } }] });

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

afterEach(() => vi.unstubAllGlobals());

// v4: the request shape per model lives in the Workers AI adapter (src/llm/adapters/workers-ai.ts).
const jsonFormat = (model: string, name: string, schema: object) => jsonSchemaFormat(modelSpec(model), name, schema);
const noThinking = (model: string) => reasoningFields(modelSpec(model), "none");

describe("request shapes per model", () => {
  it("JSON schema: {name, schema} for Nemotron 3, Gemma 4 and Kimi K2.6, the schema itself for the others", () => {
    const schema = { type: "object", properties: { a: { type: "string" } } };
    expect(jsonFormat(NEMOTRON, "help_answer", schema)).toEqual({ type: "json_schema", json_schema: { name: "help_answer", schema } });
    expect(jsonFormat("@cf/google/gemma-4-26b-a4b-it", "x", schema)).toEqual({ type: "json_schema", json_schema: { name: "x", schema } });
    expect(jsonFormat("@cf/moonshotai/kimi-k2.6", "x", schema)).toEqual({ type: "json_schema", json_schema: { name: "x", schema } });
    for (const m of [LLAMA, SCOUT_MODEL, "@cf/openai/gpt-oss-120b"]) expect(jsonFormat(m, "x", schema)).toEqual({ type: "json_schema", json_schema: schema });
  });

  it("reasoning off: only the models that take it get chat_template_kwargs or reasoning_effort (the others do not accept it)", () => {
    expect(noThinking(NEMOTRON)).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(noThinking(" @cf/google/gemma-4-26b-a4b-it ")).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(noThinking("@cf/moonshotai/kimi-k2.6")).toEqual({ reasoning_effort: "none" });
    for (const m of [LLAMA, "@cf/openai/gpt-oss-120b", "@cf/moondream/moondream3.1-9B-A2B", SCOUT_MODEL, "@cf/google/gemma-3-12b-it"]) expect(noThinking(m)).toEqual({});
  });
});

describe("the deployed configuration", () => {
  /** A variable as wrangler.jsonc sets it for the deployed Worker. */
  const deployed = async (name: string) => {
    const { readFileSync } = await import("node:fs");
    return readFileSync(new URL("../wrangler.jsonc", import.meta.url), "utf8").match(new RegExp(`"${name}":\\s*"([^"]+)"`))?.[1];
  };

  it("the help and planner models wrangler.jsonc deploys (v4.8.1: GLM-5.3-Flash for help, Gemma 4 for planning) are sent with the least reasoning they allow and the {name, schema} JSON shape", async () => {
    const vars: Record<string, string | undefined> = {};
    for (const name of ["HELP_MODEL", "PLANNER_MODEL", "SEARCH_HELP_MODEL", "SEARCH_PLANNER_MODEL"]) vars[name] = await deployed(name);
    const env = makeEnv(vars as any);
    expect(modelFor(env, "help")).toBe("@cf/zai-org/glm-5.3-flash");
    expect(modelFor(env, "planner")).toBe("@cf/google/gemma-4-26b-a4b-it");
    // the planner plans with reasoning off; GLM cannot switch it off and runs at its lowest effort
    expect(noThinking(modelFor(env, "planner"))).toEqual({ chat_template_kwargs: { enable_thinking: false } });
    expect(noThinking(modelFor(env, "help"))).toEqual({ reasoning_effort: "low" });
    for (const [role, format] of [["help", "help_answer"], ["planner", "search_plan"]] as const) {
      expect(jsonFormat(modelFor(env, role), format, {})).toEqual({ type: "json_schema", json_schema: { name: format, schema: {} } });
    }
    // Nemotron 3, the v4 help model, keeps its shape for a stack that stays on it
    expect(noThinking(NEMOTRON)).toEqual({ chat_template_kwargs: { enable_thinking: false } });
  });

  it("the search box as deployed: a question the rules cannot plan is planned by the deployed planner (v4: Gemma 4), once, its reply read as JSON or as a stream", async () => {
    const q = "any artworks of a samurai drinking tea in japan?";
    const plan = { intent: "search", output: "results", subject_en: "samurai drinking tea", object: "artwork", authors: [], colors: [], tones: [], date_from: "", date_to: "", sort: "none" };
    for (const stream of [false, true]) {
      const env = makeEnv({ PLANNER_BACKEND: await deployed("PLANNER_BACKEND"), PLANNER_MODEL: await deployed("PLANNER_MODEL"), SEARCH_PLANNER_MODEL: await deployed("SEARCH_PLANNER_MODEL") } as any);
      const PLANNER = modelFor(env, "planner");
      env._ai.handler = (model) => {
        if (model !== PLANNER) return {};
        const reply = chat(plan);
        return stream ? new Response(JSON.stringify(reply), { headers: { "content-type": "application/json; charset=utf-8" } }).body : reply;
      };
      const exec = new FakeExec();
      const res = await app.fetch(new Request(`https://search.test/query?q=${encodeURIComponent(q)}`), env, exec as unknown as ExecutionContext);
      await exec.settle();
      const body = (await res.json()) as any;
      expect(body.route).toBe("ask");
      expect(body.answer.plan).toMatchObject({ source: "rules+llm", residual: "samurai drinking tea" });
      expect(body.answer.notes.join(" ")).not.toMatch(/planner unavailable/);
      const calls = env._ai.calls.filter((c) => c.model === PLANNER);
      expect(calls.length).toBe(1);
      expect(calls[0].input).toMatchObject({
        max_tokens: 300,
        temperature: 0,
        chat_template_kwargs: { enable_thinking: false },
        response_format: { type: "json_schema", json_schema: { name: "search_plan", schema: expect.objectContaining({ type: "object" }) } },
      });
    }
  });
});

describe("Gemma 4 describes artworks", () => {
  it("image as a message part, JSON schema, reasoning off; reads the OpenAI-style reply", async () => {
    const env = makeEnv();
    env._ai.handler = (model) =>
      model === GEMMA_MODEL ? chat({ caption: "A knight riding a horse at sunset.", subjects: ["Knight"], objects: [], style: "landscape", mood: "calm", text_in_image: "", tags: ["knight", "horse", "sunset"], nsfw: 0 }) : {};
    const r = await describeImage(env, "gemma", IMG, { title: "Sir Galahad" });
    expect(r).toMatchObject({ model: GEMMA_MODEL, status: "ok", description: { caption: "A knight riding a horse at sunset.", subjects: ["knight"], tags: ["knight", "horse", "sunset"] } });
    const input = env._ai.calls[0].input;
    expect(input.response_format).toEqual({ type: "json_schema", json_schema: { name: "artwork_description", schema: expect.objectContaining({ required: expect.arrayContaining(["caption", "tags"]) }) } });
    expect(input.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(input.messages[1].content).toEqual([expect.objectContaining({ type: "text" }), { type: "image_url", image_url: { url: IMG } }]);
    expect(input.messages[1].content[0].text).toContain('Title: "Sir Galahad"');
  });

  it("Gemma's real reply for artwork 103 (live, 5 October 2026), without the words 'pixel art'", async () => {
    const env = makeEnv();
    // what @cf/google/gemma-4-26b-a4b-it answered for @retro's "Hello Kitty! Hail Satan!"
    const raw = JSON.stringify({
      caption:
        "A central figure of Hello Kitty dressed in black gothic clothing with a skull motif stands atop a rocky cliff, flanked by four silhouetted figures in dark robes, set against a backdrop of a tall white gothic spire and a cloudy purple sky.",
      mood: "dark, eerie, subversive, gothic",
      nsfw: 0,
      objects: ["rocky cliff", "gothic spire", "skull necklace", "pink bow"],
      style: "pixel art sprite composition",
      subjects: ["Hello Kitty", "silhouetted figures"],
      tags: ["gothic", "hello kitty", "dark", "silhouette", "spire", "skull", "subversive", "fantasy", "retro"],
      text_in_image: "",
    }, null, 2);
    env._ai.handler = (model) => (model === GEMMA_MODEL ? chat(raw) : {});
    const r = await describeImage(env, "gemma", IMG, { title: "Hello Kitty! Hail Satan!", tags: ["retro", "kitty", "satan", "hello", "video-game"] });
    expect(r.status).toBe("ok");
    expect(r.description).toMatchObject({ style: "sprite composition", subjects: ["hello kitty", "silhouetted figures"], nsfw: 0 });
    expect(r.description.tags).toContain("hello kitty");
    expect(normalizeDescription({ caption: "x", style: "Pixel-Art", tags: ["pixel art", "Cat", "pixels", "pixelated", "pixel art style"] })).toMatchObject({ style: "", tags: ["cat"] });
    expect(noPixelArt("isometric pixel art, retro")).toBe("isometric, retro");
    expect(noPixelArt("retro, pixel art, isometric")).toBe("retro, isometric");
    expect(noPixelArt("8-bit game sprite")).toBe("8-bit game sprite");
  });

  it("an empty reply is an error, named after the backend; scout is unchanged", async () => {
    const env = makeEnv();
    env._ai.handler = () => chat("");
    await expect(describeImage(env, "gemma", IMG, {})).rejects.toThrow(/^gemma returned no usable description/);
    env._ai.handler = (model) => (model === SCOUT_MODEL ? { response: { caption: "A lighthouse at night by the sea.", tags: ["lighthouse"] } } : {});
    await describeImage(env, "scout", IMG, {});
    expect(env._ai.calls.at(-1)!.model).toBe(SCOUT_MODEL);
    expect(env._ai.calls.at(-1)!.input.chat_template_kwargs).toBeUndefined();
    expect(env._ai.calls.at(-1)!.input.response_format.json_schema.properties.caption).toBeDefined(); // the schema itself
  });

  it("the describe debug route refuses unknown backends; without one it still works when descriptions are off", async () => {
    const post = async (env: ReturnType<typeof makeEnv>, qs: string) => {
      const res = await app.fetch(new Request(`https://search.test/admin/debug/describe/1${qs}`, { method: "POST", headers: { authorization: `Bearer ${TOKEN}` } }), env, new FakeExec() as unknown as ExecutionContext);
      return { status: res.status, body: (await res.json()) as any };
    };
    const env = makeEnv({ ADMIN_TOKEN: TOKEN });
    const bad = await post(env, "?backend=llava");
    expect(bad.status).toBe(400);
    expect(bad.body.error).toBe(`backend must be one of ${VLM_BACKENDS.join(", ")}`);
    expect(VLM_BACKENDS).toContain("gemma");
    // VLM_BACKEND=off and no ?backend: not refused (here it goes on to look for post 1, which does not exist)
    expect(await post(makeEnv({ ADMIN_TOKEN: TOKEN, VLM_BACKEND: "off" }), "")).toEqual({ status: 404, body: { error: "not an artwork" } });
    expect((await post(env, "?backend=GEMMA")).status).toBe(404);
  });
});

describe("Nemotron 3 / Gemma 4 as the help and planner models", () => {
  function docsEnv(over: Record<string, string> = {}) {
    const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", ADMIN_TOKEN: TOKEN, ...over });
    (env as any).VEC_DOCS = new FakeVectorize([]);
    installGitHub({
      owner: "pixa",
      repo: "info",
      head: "1".repeat(40),
      files: { "fees.md": "# Fees\n\nMinting an artwork is free.\n\n## Selling\n\nA sale on the marketplace pays a 5% fee to the platform." },
      calls: [],
    });
    const asked: string[] = [];
    env._ai.handler = (model, input) => {
      if (model === EMBED) return { shape: [input.text.length, DIM], data: input.text.map(bow) };
      asked.push(model);
      return chat({ answerable: true, answer: `A sale pays a 5% fee [1].`, sources: [1] });
    };
    return { env, asked };
  }

  it("HELP_MODEL=Nemotron 3: reasoning off, OpenAI-style reply accepted", async () => {
    const { env, asked } = docsEnv({ HELP_MODEL: NEMOTRON });
    await syncDocs(env);
    const r = await answerHelp(env, "What fee does a sale pay?");
    expect(r).toMatchObject({ status: "answered", answer_text: "A sale pays a 5% fee [1].", model: NEMOTRON });
    expect(asked).toEqual([NEMOTRON]);
    const call = env._ai.calls.find((c) => c.model === NEMOTRON)!;
    expect(call.input.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(call.input.response_format).toEqual({ type: "json_schema", json_schema: { name: "help_answer", schema: expect.objectContaining({ type: "object" }) } });
  });

  it("a help reply the binding leaves as a stream (content-type not exactly application/json) is read", async () => {
    const { env } = docsEnv({ HELP_MODEL: NEMOTRON });
    await syncDocs(env);
    const handler = env._ai.handler;
    env._ai.handler = (model, input) => {
      const r = handler(model, input);
      return model === NEMOTRON ? new Response(JSON.stringify(r), { headers: { "content-type": "application/json; charset=utf-8" } }).body : r;
    };
    expect(await answerHelp(env, "What fee does a sale pay?")).toMatchObject({ status: "answered", answer_text: "A sale pays a 5% fee [1]." });
  });

  it("/admin/debug/help answers with the model asked for, without logging; bad input is refused", async () => {
    const { env, asked } = docsEnv();
    await syncDocs(env);
    const exec = new FakeExec();
    const get = async (qs: string) => {
      const res = await app.fetch(new Request(`https://search.test/admin/debug/help?${qs}`, { headers: { authorization: `Bearer ${TOKEN}` } }), env, exec as unknown as ExecutionContext);
      await exec.settle();
      return { status: res.status, body: (await res.json()) as any };
    };
    const a = await get(`q=${encodeURIComponent("What fee does a sale pay?")}&model=${encodeURIComponent(NEMOTRON)}`);
    expect(a.status).toBe(200);
    expect(a.body).toMatchObject({ status: "answered", model: NEMOTRON });
    const b = await get(`q=${encodeURIComponent("What fee does a sale pay?")}`);
    expect(b.body.model).toBe(LLAMA); // HELP_MODEL's default
    expect(asked).toEqual([NEMOTRON, LLAMA]);
    expect(env._ai.calls.find((c) => c.model === LLAMA)!.input.response_format.json_schema.properties).toBeDefined(); // Llama: the schema itself
    const hf = await get(`q=${encodeURIComponent("Is minting free?")}&model=${encodeURIComponent("@hf/nousresearch/hermes-2-pro-mistral-7b")}`);
    expect(hf.body.model).toBe("@hf/nousresearch/hermes-2-pro-mistral-7b");
    expect((await env.DB.prepare("SELECT COUNT(*) AS n FROM help_log").first<{ n: number }>())!.n).toBe(0);
    expect((await get(`q=fees&model=${encodeURIComponent("https://evil.example/model")}`)).status).toBe(400);
    expect((await get("model=" + encodeURIComponent(NEMOTRON))).status).toBe(400);
    const anon = await app.fetch(new Request(`https://search.test/admin/debug/help?q=fees`), env, exec as unknown as ExecutionContext);
    expect(anon.status).toBe(401);
  });

  it("PLANNER_MODEL=Gemma 4: reasoning off, OpenAI-style plan accepted", async () => {
    const GEMMA = "@cf/google/gemma-4-26b-a4b-it";
    const env = makeEnv({ PLANNER_MODEL: GEMMA });
    env._ai.handler = () => chat('{"intent":"find_first","output":"author","subject_en":"dragon","object":"artwork","authors":[],"colors":[],"tones":[],"date_from":"","date_to":"","sort":"none"}');
    const authors = new Set(["alice"]);
    const rules = planQuery("who posted the first dragon?", { mode: "ask", authors, now: Date.UTC(2026, 9, 1) / 1000 });
    const plan = await llmPlan(env, "who posted the first dragon?", rules, authors);
    expect(plan.source).toBe("rules+llm");
    expect(plan.residual).toBe("dragon");
    expect(env._ai.calls[0].model).toBe(GEMMA);
    expect(env._ai.calls[0].input.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(env._ai.calls[0].input.response_format.json_schema.name).toBe("search_plan");
  });
});
