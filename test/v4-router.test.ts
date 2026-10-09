// The v4 query router (spec §6-8, §20): classes, complexity bands, and the mode a question runs in.

import { describe, expect, it } from "vitest";
import { makeEnv } from "./harness/fakes";
import { artCorpus, GALLERY } from "./harness/corpus";
import { planQuery } from "../src/search/planner";
import { decomposeRules } from "../src/search/query-planner";
import { bandOf, chooseMode, routeQuestion, type QueryRoute } from "../src/search/query-router";
import { ask, type AskResponseV4 } from "../src/search/ask";
import { modelFor } from "../src/llm/router";

const AUTHORS = new Set(["alice", "bob", "carol"]);
const NOW = Date.UTC(2026, 9, 8) / 1000;

function route(q: string, opts: { image?: boolean } = {}): QueryRoute {
  const plan = planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
  const program = decomposeRules(q, plan, { authors: AUTHORS, lang: plan.lang, now: NOW });
  return routeQuestion(q, plan, program, { ...opts, env: makeEnv() });
}

describe("classes", () => {
  it("each question gets the class of what it asks", () => {
    const cases: Array<[string, string]> = [
      ["What is the link of “Swan”?", "EXACT"],
      ["Who posted “Lake”?", "FACTUAL"],
      ["Who posted the first cat artwork?", "FACTUAL"],
      ["red dragons", "SEMANTIC"],
      ["Was “Swan” edited?", "MULTI_HOP"],
      ["How long after “Swan” was “Lake” posted?", "TEMPORAL"],
      ["Did @alice post more cats than @bob?", "COMPARATIVE"],
      ["How many artworks did @alice publish?", "AGGREGATION"],
      ["In which month did @alice post the most?", "AGGREGATION"],
      ["Who posted the first cat artwork and was it later reposted?", "MULTI_HOP"],
      ["Why are cats so popular?", "EXPLANATORY"],
      ["Who posted this first?", "AMBIGUOUS"],
    ];
    for (const [q, c] of cases) expect(route(q).class, q).toBe(c);
    expect(route("Who posted this first?", { image: true }).class).toBe("VISUAL");
  });

  it("“who posted this?” with nothing named: a question back, not an answer about every artwork", async () => {
    const env = await artCorpus(GALLERY);
    const r = (await ask(env, { question: "Who posted this first?" })) as AskResponseV4;
    expect(r).toMatchObject({ status: "clarify", answer: null, answer_text: "Which artwork do you mean? Name it (its title in quotes, or its id), or ask with the image." });
    expect(r.confidence).toBeLessThanOrEqual(0.3);
    expect(((await ask(env, { question: "Qui a posté ça ?" })) as AskResponseV4).answer_text).toContain("De quelle œuvre parlez-vous");
  });

  it("complexity grows with entities, steps and relations; bands are configurable", () => {
    const simple = route("How many artworks did @alice publish?");
    const multi = route("Who posted the first cat artwork and was it later reposted?");
    const compare = route("Did @alice post more cats than @bob?");
    expect(simple.complexity).toBeLessThan(multi.complexity);
    expect(multi.complexity).toBeLessThan(compare.complexity);
    expect(simple.band).toBe("trivial");
    expect(bandOf(0.5, makeEnv({ SEARCH_COMPLEXITY_BANDS: "0.1,0.2,0.3,0.4" }))).toBe("deep");
    expect(bandOf(0.5, makeEnv({ SEARCH_COMPLEXITY_BANDS: "nonsense" }))).toBe("normal");
  });
});

describe("modes", () => {
  it("auto follows the band, one step up for a synthesis; an explicit mode up to the caller's maximum; or none of it", () => {
    const env = makeEnv();
    const r = route("How many artworks did @alice publish?");
    expect(chooseMode(env, "auto", r, { needsSynthesis: false })).toEqual({ mode: "fast", explicit: false });
    expect(chooseMode(env, "auto", r, { needsSynthesis: true })).toEqual({ mode: "balanced", explicit: false });
    expect(chooseMode(env, "expert", r, { needsSynthesis: false, max: "deep" })).toEqual({ mode: "deep", explicit: true });
    // the search box's ceiling is for what auto picks, not for an explicit mode
    expect(chooseMode(env, "deep", r, { needsSynthesis: false, ceiling: "balanced", max: "deep" })).toEqual({ mode: "deep", explicit: true });
    expect(chooseMode(env, "auto", r, { needsSynthesis: true, ceiling: "fast", max: "deep" })).toEqual({ mode: "fast", explicit: false });
    expect(chooseMode(makeEnv({ SEARCH_AUTO_COMPLEXITY: "false", SEARCH_DEFAULT_MODE: "balanced" }), undefined, r, { needsSynthesis: false })).toEqual({ mode: "balanced", explicit: false });
    // a deployment's default mode is not the caller's choice
    expect(chooseMode(makeEnv({ SEARCH_DEFAULT_MODE: "deep" }), undefined, r, { needsSynthesis: false })).toEqual({ mode: "deep", explicit: false });
  });

  it("a public caller gets at most SEARCH_MAX_MODE (deep); the admin may run expert", async () => {
    const env = await artCorpus(GALLERY);
    expect(((await ask(env, { question: "How many artworks did @alice publish?", mode: "expert" })) as AskResponseV4).mode).toBe("deep");
    expect(((await ask(env, { question: "How many artworks did @alice publish?", mode: "expert", admin: true })) as AskResponseV4).mode).toBe("expert");
    const low = await artCorpus(GALLERY, { SEARCH_MAX_MODE: "fast" });
    expect(((await ask(low, { question: "How many artworks did @alice publish?", mode: "deep" })) as AskResponseV4).mode).toBe("fast");
  });

  it("the reasoning model per complexity band (SEARCH_REASONING_MODEL_<BAND>), else SEARCH_REASONING_MODEL", () => {
    const env = makeEnv({ SEARCH_REASONING_MODEL: "@cf/openai/gpt-oss-120b", SEARCH_REASONING_MODEL_SIMPLE: "@cf/openai/gpt-oss-20b", SEARCH_REASONING_MODEL_DEEP: "@cf/moonshotai/kimi-k2.6" });
    expect(modelFor(env, "reasoning", "simple")).toBe("@cf/openai/gpt-oss-20b");
    expect(modelFor(env, "reasoning", "normal")).toBe("@cf/openai/gpt-oss-120b");
    expect(modelFor(env, "reasoning", "deep")).toBe("@cf/moonshotai/kimi-k2.6");
    expect(modelFor(makeEnv(), "reasoning")).toBe("@cf/openai/gpt-oss-120b");
  });
});
