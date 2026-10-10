// v4 on /search and /help: POST /search with the cross-encoder (blended into the order, counts per
// retrieval family), and /help answers checked sentence by sentence against the documentation
// (a fee the excerpts do not state is removed), help modes, the documentation reranker.

import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeExec, FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { artCorpus, GALLERY } from "./harness/corpus";
import { singular, tokens } from "../src/lib/text";
import { syncDocs } from "../src/docs/sync";
import { answerHelp, answerSegments, checkHelpAnswer, HELP_PROMPT_VERSION, helpMode } from "../src/help/answer";
import type { DocHit } from "../src/help/retrieve";
import { app } from "../src/api";

const DIM = 64;
const EMBED = "@cf/baai/bge-m3";
const HELP = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const NEMOTRON = "@cf/nvidia/nemotron-3-120b-a12b";
const RERANKER = "@cf/baai/bge-reranker-base";
const TOKEN = "admin-secret";

async function call(env: TestEnv, path: string, init: RequestInit & { admin?: boolean } = {}): Promise<{ status: number; body: any }> {
  const exec = new FakeExec();
  const headers = new Headers(init.headers);
  if (init.admin) headers.set("authorization", `Bearer ${TOKEN}`);
  if (init.body && !headers.has("content-type")) headers.set("content-type", "application/json");
  const res = await app.fetch(new Request(`https://search.test${path}`, { ...init, headers }), env, exec as unknown as ExecutionContext);
  await exec.settle();
  return { status: res.status, body: await res.json<any>() };
}

afterEach(() => vi.unstubAllGlobals());

describe("POST /search with the cross-encoder", () => {
  it("the same search as GET, reordered by the reranker when asked, with candidates per retrieval family", async () => {
    const env = await artCorpus(GALLERY);
    const seen: any[] = [];
    env._ai.handler = (model, input) => {
      if (model !== RERANKER) throw new Error(`unexpected model ${model}`);
      seen.push(input);
      // the cross-encoder prefers the nap
      return { response: input.contexts.map((c: { text: string }, id: number) => ({ id, score: c.text.startsWith("Cat nap") ? 6 : -2 })) };
    };
    const plain = await call(env, "/search", { method: "POST", body: JSON.stringify({ query: "cat", limit: 10 }) });
    expect(plain.status).toBe(200);
    expect(plain.body.reranked).toBe(false);
    expect(plain.body.retrieval).toMatchObject({ fts: expect.any(Number) });
    const get = await call(env, "/search?q=cat&limit=10");
    expect(get.body.items.map((i: any) => i.id)).toEqual(plain.body.items.map((i: any) => i.id));
    const rr = await call(env, "/search", { method: "POST", body: JSON.stringify({ query: "cat", limit: 10, rerank: true }) });
    expect(rr.body).toMatchObject({ reranked: true, reranker: RERANKER });
    expect(seen.length).toBe(1);
    expect(seen[0].query).toBe("cat");
    expect(rr.body.items[0].title).toBe("Cat nap");
    expect(rr.body.items[0].score.rerank).toBeGreaterThan(0.9);
    // the same set of results, only the order changes
    expect(new Set(rr.body.items.map((i: any) => i.id))).toEqual(new Set(plain.body.items.map((i: any) => i.id)));
  });

  it("a reranker that fails leaves the feature ranker's order and says so", async () => {
    const env = await artCorpus(GALLERY);
    env._ai.handler = () => {
      throw new Error("capacity exceeded");
    };
    const r = await call(env, "/search", { method: "POST", body: JSON.stringify({ query: "cat", rerank: true }) });
    expect(r.status).toBe(200);
    expect(r.body.reranked).toBe(false);
    expect(r.body.notes.join(" ")).toContain("reranker unavailable");
  });
});

// ---- /help ------------------------------------------------------------------------------------------

function bow(text: string): number[] {
  const v = new Array<number>(DIM).fill(0);
  for (const t of tokens(text, { keepHyphenated: false })) {
    if (t.length < 3) continue;
    let h = 7;
    for (const ch of singular(t)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    v[h % DIM] += 1;
  }
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

const FILES: Record<string, string> = {
  "guides/fees.md": "# Fees\n\nMinting an artwork is free.\n\n## Selling\n\nA sale on the marketplace pays a 5% fee to the platform.\n\n## Royalties\n\nArtists receive royalties on every resale.",
  "guides/mint.md": "# Minting\n\nTo publish an artwork, open the editor and click “Mint”. Minting takes two blocks.",
};

function docsEnv(reply: (input: any, model: string) => unknown, over: Record<string, string> = {}) {
  const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", ADMIN_TOKEN: TOKEN, ...over });
  (env as any).VEC_DOCS = new FakeVectorize([]);
  installGitHub({ owner: "pixa", repo: "info", head: "a".repeat(40), files: { ...FILES }, calls: [] });
  const asked: Array<{ model: string; input: any }> = [];
  env._ai.handler = (model, input) => {
    if (model === EMBED) return { shape: [input.text.length, DIM], data: input.text.map(bow) };
    asked.push({ model, input });
    return reply(input, model);
  };
  return { env, asked };
}
const legacy = (answer: string, sources = [1]) => () => ({ response: { answerable: true, answer, sources } });

const hit = (n: number, text: string, extra: Partial<DocHit> = {}): DocHit => ({ id: n, path: `p${n}.md`, title: `Page ${n}`, heading: "", anchor: "", url: `https://github.com/pixa/info/blob/main/p${n}.md`, text, lang: "en", lexical: 1, cosine: null, score: 0.9, ...extra });

describe("help answers checked sentence by sentence", () => {
  it("segments: sentences and list items, joined back to the text", () => {
    const t = "First [1]. Then [2].\n1. Open it [1]\n2. Pay. [2] Done!";
    const segs = answerSegments(t);
    expect(segs.join("")).toBe(t);
    expect(segs.map((s) => s.trim())).toEqual(["First [1].", "Then [2].", "1. Open it [1]", "2. Pay. [2]", "Done!"]);
  });

  it("a number, a date or a quoted label the excerpts do not state is caught; list numbers and step references are not", () => {
    const hits = [hit(1, "A sale pays a 5% fee to the platform."), hit(2, "Open the editor and click “Mint”. Minting takes two blocks.")];
    const c = checkHelpAnswer("A sale pays a 5% fee [1]. Royalties are 10% on resales [1].", hits);
    expect(c.text).toBe("A sale pays a 5% fee [1].");
    expect(c.claims.map((x) => x.status)).toEqual(["supported", "unsupported"]);
    expect(c.claims[1].problems![0]).toContain("percent 10 is not in the evidence");
    expect(c).toMatchObject({ removed: 1, egs: 0.5, cited: [1] });
    const steps = checkHelpAnswer("1. Open the editor [2]\n2. Click “Mint” [2]\n3. Wait 2 blocks [2]", hits);
    expect(steps.removed).toBe(0);
    expect(steps.text).toBe("1. Open the editor [2]\n2. Click “Mint” [2]\n3. Wait 2 blocks [2]");
    expect(checkHelpAnswer("Step 3: wait for two blocks [2].", hits).removed).toBe(0);
    // a translated label is only qualified (kept), a missing number is not
    const fr = checkHelpAnswer("Ouvrez l'éditeur et cliquez sur « Publier » [2]. Cela coûte 3 PXS [2].", hits);
    expect(fr.claims.map((x) => x.status)).toEqual(["qualified", "unsupported"]);
    expect(fr.text).toBe("Ouvrez l'éditeur et cliquez sur « Publier » [2].");
    // expert: a sentence that cites nothing does not stay
    expect(checkHelpAnswer("Minting takes two blocks [2]. Have fun!", hits).text).toBe("Minting takes two blocks [2]. Have fun!");
    expect(checkHelpAnswer("Minting takes two blocks [2]. Have fun!", hits, { strict: true }).text).toBe("Minting takes two blocks [2].");
  });

  it("end to end: the unsupported sentence is gone, the claims and grounding say why, confidence is discounted", async () => {
    const { env } = docsEnv(legacy("A sale pays a 5% fee [1]. Buyers also pay 2% [1]."));
    await syncDocs(env);
    const r = await answerHelp(env, "What fee does a sale pay?");
    expect(r.status).toBe("answered");
    expect(r.answer_text).toBe("A sale pays a 5% fee [1].");
    expect(r.grounding).toEqual({ egs: 0.5, citation_accuracy: 1, removed: 1 });
    expect(r.claims!.map((c) => c.status)).toEqual(["supported", "unsupported"]);
    expect(r.notes.join(" ")).toContain("1 sentence(s) removed");
    expect(r.confidence).toBeLessThan(r.sources[0].score);
    expect(r).toMatchObject({ mode: "fast", reasoning: "none", versions: { docs_commit: "a".repeat(40), help_model: HELP, prompt: HELP_PROMPT_VERSION, reranker: null } });
    const log = await env.DB.prepare("SELECT status, mode, model, egs FROM help_log ORDER BY id DESC LIMIT 1").first<any>();
    expect(log).toEqual({ status: "answered", mode: "fast", model: HELP, egs: 0.5 });
  });

  it("nothing left once the unsupported sentences are out: the excerpts themselves, not cached", async () => {
    const { env, asked } = docsEnv(legacy("Selling costs 12% [1]."));
    await syncDocs(env);
    const r = await answerHelp(env, "What fee does a sale pay?");
    expect(r.status).toBe("excerpts");
    expect(r.notes.join(" ")).toContain("no sentence of the answer is supported");
    await answerHelp(env, "What fee does a sale pay?");
    expect(asked.length).toBe(2);
  });

  it("modes: auto by length; deep asks the model to reason (Nemotron: low effort) with more room; each mode its own cache entry", async () => {
    const { env, asked } = docsEnv(() => ({ choices: [{ message: { content: JSON.stringify({ answerable: true, answer: "A sale pays a 5% fee [1].", sources: [1] }) } }] }), { HELP_MODEL: NEMOTRON });
    await syncDocs(env);
    expect(helpMode(env, "What fee does a sale pay?")).toBe("fast");
    expect(helpMode(env, "Please explain to me in detail what fee a sale on the marketplace pays to the platform")).toBe("balanced");
    expect(helpMode(env, "fees", "expert")).toBe("expert");
    const fast = await answerHelp(env, "What fee does a sale pay?");
    expect(asked[0].input.chat_template_kwargs).toEqual({ enable_thinking: false });
    expect(asked[0].input.max_tokens).toBe(700);
    const deep = await answerHelp(env, "What fee does a sale pay?", { mode: "deep" });
    expect(deep).toMatchObject({ mode: "deep", reasoning: "low", status: "answered" });
    expect(deep.cached).toBeUndefined();
    expect(asked[1].input.chat_template_kwargs).toEqual({ enable_thinking: true, low_effort: true });
    expect(asked[1].input.max_tokens).toBeGreaterThan(1500);
    expect((await answerHelp(env, "What fee does a sale pay?")).cached).toBe(true);
    expect(fast.mode).toBe("fast");
  });

  it("HELP_RERANK=on: the cross-encoder orders the excerpts of an English question in the deeper modes", async () => {
    const { env, asked } = docsEnv(
      (input, model) => (model === RERANKER ? { response: input.contexts.map((c: { text: string }, id: number) => ({ id, score: c.text.includes("Royalties") ? 5 : -3 })) } : { response: { answerable: true, answer: "Artists receive royalties on every resale [1].", sources: [1] } }),
      { HELP_RERANK: "on" },
    );
    await syncDocs(env);
    const r = await answerHelp(env, "Do artists receive royalties when a sale happens on the marketplace?", { mode: "balanced" });
    expect(asked.some((a) => a.model === RERANKER)).toBe(true);
    expect(r.versions?.reranker).toBe(RERANKER);
    expect(r.sources[0].heading).toBe("Royalties");
    const help = asked.find((a) => a.model === HELP)!;
    expect(help.input.messages[1].content).toContain("[1] Fees — Royalties");
    // fast mode: no reranking
    const before = asked.filter((a) => a.model === RERANKER).length;
    await answerHelp(env, "Do artists receive royalties?", { mode: "fast" });
    expect(asked.filter((a) => a.model === RERANKER).length).toBe(before);
  });

  it("over HTTP: mode and reasoning, expert and reasoning=high for the admin only", async () => {
    const { env } = docsEnv(legacy("Minting is free [1]. Have fun!"));
    await syncDocs(env);
    const pub = await call(env, "/help?q=" + encodeURIComponent("Is minting free?") + "&mode=expert");
    expect(pub.body.mode).toBe("fast"); // expert refused: by length
    expect(pub.body.answer_text).toBe("Minting is free [1]. Have fun!");
    const admin = await call(env, "/help?q=" + encodeURIComponent("Is minting free?") + "&mode=expert", { admin: true });
    expect(admin.body).toMatchObject({ mode: "expert", reasoning: "medium", answer_text: "Minting is free [1]." });
    const post = await call(env, "/help", { method: "POST", body: JSON.stringify({ question: "Is minting free?", mode: "deep", reasoning: "high" }) });
    expect(post.body).toMatchObject({ mode: "deep", reasoning: "low" }); // high refused: the mode's
  });
});
