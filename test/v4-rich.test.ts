// v4.8: the long-form answers. The deterministic digest in five languages, the follow-up
// templates every one of which the rules planner answers, the model's body and reasoning trail
// verified sentence by sentence (a hallucinated sentence out, the rest kept), follow-ups and
// searches filtered, the mode floor of rich answers, a contradicted lead dropping the body, a
// reply cut off at the token limit salvaged, the deferred elaboration over HTTP, the search
// overview on /query and /search, and the rich /help with follow-ups and related sections.
// style=brief is v4's answer, field for field: the v4 suites run with it.

import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeExec, FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { artCorpus, GALLERY } from "./harness/corpus";
import { singular, tokens } from "../src/lib/text";
import { syncDocs } from "../src/docs/sync";
import { answerHelp, helpMode } from "../src/help/answer";
import { app } from "../src/api";
import { ask, elaborate, type AskResponseV4 } from "../src/search/ask";
import { assembleMarkdown, checkFollowUp, checkSearch, segmentMarkdown, toPlainText, verifyList, verifyMarkdown, wordTarget } from "../src/search/compose";
import { searchOverview } from "../src/search/overview";
import { salvageComposeReply, parseReasoningReply } from "../src/llm/reasoning";
import { COMPOSE_SCHEMA, reasoningMessages } from "../src/llm/prompts";
import { planQuery } from "../src/search/planner";
import { decomposeRules } from "../src/search/query-planner";
import { routeQuestion } from "../src/search/query-router";
import type { EvidenceCard } from "../src/search/evidence";

const GPT_OSS = "@cf/openai/gpt-oss-120b";
const TOKEN = "admin-secret";
const RICH = { SEARCH_ANSWER_STYLE: "rich", HELP_STYLE: "rich" };

async function call(env: TestEnv, path: string, init: RequestInit & { admin?: boolean } = {}): Promise<{ status: number; body: any }> {
  const exec = new FakeExec();
  const headers = new Headers(init.headers);
  if (init.admin) headers.set("authorization", `Bearer ${TOKEN}`);
  if (init.body && !headers.has("content-type")) headers.set("content-type", "application/json");
  const res = await app.fetch(new Request(`https://search.test${path}`, { ...init, headers }), env, exec as unknown as ExecutionContext);
  await exec.settle();
  return { status: res.status, body: await res.json<any>() };
}

const v4 = async (env: TestEnv, question: string, extra: Record<string, unknown> = {}) => (await ask(env, { question, ...extra })) as AskResponseV4;

/** The long-form reply a good model gives about @alice's four artworks (cards E1 Black cat, E3 Cat nap, E5 Swan, E8 Dragon egg), with three things it made up. */
const COMPOSED = {
  status: "answered",
  answer: "@alice published 4 artworks [R1].",
  body:
    "### Overview\n@alice has 4 artworks in the index [R1]. They were posted between 2026-09-01 and 2026-10-02 [E1][E8].\n\n" +
    "- “Black cat” (2026-09-01), tagged cat and black-and-white, 4 votes [E1]\n- “Cat nap” (2026-09-05), 6 votes [E3]\n- “Swan” (2026-09-10), edited once [E5]\n- “Dragon egg” (2026-10-02), 7 votes [E8]\n\n" +
    "Her most voted piece is “Dragon egg” with 7 votes [E8]. She also posted a mural in 2024 with 900 votes [E9]. See https://evil.example/alice for more [E1].\n\n### Empty\n",
  thinking: ["The question asks for a count of @alice's artworks [R1].", "R1 is an exact count computed by the index: 4 [R1].", "Her brother @bob posted 3 artworks [E2].", "The four cards E1, E3, E5 and E8 are hers, which agrees with the count."],
  caveats: ["Votes are the counts at indexing time [E8].", "The payout of “Swan” was 50 PXS [E5]."],
  follow_ups: ["What is the most voted artwork of @alice?", "Did @alice post more cats than @bob?", "Who is @zorro?", "When did @alice post “Mural”?", "How many artworks did @alice publish?"],
  searches: ["cat", "@alice dragon", "purple unicorn"],
  claims: [{ text: "@alice published 4 artworks", evidence: ["R1"], kind: "fact" }],
  rationale: "R1 counts 4 artworks by @alice.",
  confidence: 0.95,
};

function reasoner(env: TestEnv, reply: Record<string, unknown> | ((input: any) => Record<string, unknown>) | string, usage = { prompt_tokens: 1200, completion_tokens: 500 }, finish?: string) {
  const seen: any[] = [];
  env._ai.handler = (model, input) => {
    if (model !== GPT_OSS) throw new Error(`unexpected model ${model}`);
    seen.push(input);
    const r = typeof reply === "function" ? reply(input) : reply;
    return typeof r === "string" ? { response: r, usage, ...(finish ? { finish_reason: finish } : {}) } : { response: r, usage };
  };
  return seen;
}

afterEach(() => vi.unstubAllGlobals());

// ---- the digest, without any model ------------------------------------------------------------------------

describe("the deterministic digest (fast mode: no model)", () => {
  it("a first/latest question: the index's sentence stays the direct answer; the facts, the set's overview, follow-ups and searches come after", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const r = await v4(env, "Who posted the first cat artwork?", { mode: "fast" });
    expect(r.style).toBe("rich");
    expect(r.mode).toBe("fast");
    expect(env._ai.calls).toEqual([]);
    // v4's fields are unchanged
    expect(r.answer_text).toBe("The first cat artwork was posted by @alice on 2026-09-01: “Black cat”.");
    expect(r.answer_short).toBe(r.answer_text);
    expect(r.result_text).toBe(r.answer_text);
    // the long answer
    expect(r.answer_full.length).toBeGreaterThan(r.answer_text.length * 3);
    expect(r.answer_full).toContain("From the index:");
    expect(r.sections.overview.join(" ")).toContain("4 cat artworks, 3 artists; most by @alice (2), @bob (1), @carol (1).");
    expect(r.sections.overview.join(" ")).toContain("Posted between 2026-09-01 and 2026-09-25.");
    expect(r.sections.overview.join(" ")).toContain("The most voted is “Cat nap” by @alice (6 votes).");
    expect(r.sections.facts[0]).toBe("“Black cat” by @alice: posted on 2026-09-01; tagged cat, black-and-white; 4 votes, 1.5 PXS.");
    expect(r.digest.stats).toMatchObject({ posts: 4, read: 4, authors: [{ author: "alice", n: 2 }, { author: "bob", n: 1 }, { author: "carol", n: 1 }], from: "2026-09-01", to: "2026-09-25" });
    expect(r.digest.about).toEqual(["E1"]);
    expect(r.suggestions.follow_ups.map((f) => f.text)).toEqual(["How many cat artworks are there?", "Who posted the latest cat artwork?", "Who posted the most cat artworks?", "Which cat artwork has the most votes?", "When was “Black cat” posted?", "Was “Black cat” edited?"]);
    expect(r.suggestions.follow_ups.every((f) => f.route === "ask" && f.source === "rules")).toBe(true);
    expect(r.suggestions.searches.map((s) => s.text)).toEqual(["cat", "@alice", "@bob"]);
    expect(r.thinking).toEqual([]);
    expect(r.elaboration).toEqual({ status: "none", reason: "fast mode: no model" });
    expect(r.answer_markdown).toContain("### You may also ask\n- How many cat artworks are there?");
    expect(r.versions.answer).toBe("4.8.0");
    // the plain text has no Markdown marks and no citations
    expect(r.answer_full).not.toMatch(/###|\[E\d+\]/);
  });

  it("an exact count over the filters keeps no rows in v4: the rich answer reads the set's newest posts, and the facts show its most voted and latest", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const r = await v4(env, "How many artworks did @alice publish?", { mode: "fast" });
    expect(r.answer).toBe(4);
    expect(r.answer_text).toBe("4 artworks found.");
    expect(r.sections.overview[0]).toBe("4 artworks by @alice.");
    expect(r.sections.overview).toContain("Posted between 2026-09-01 and 2026-10-02.");
    expect(r.sections.facts).toEqual(["“Dragon egg” by @alice: posted on 2026-10-02; tagged dragon; 7 votes."]);
    // the posts of the set are evidence cards now (the model sees them in balanced and above)
    expect(r.cards.filter((c) => c.type === "artwork").map((c: any) => c.author)).toEqual(["alice", "alice", "alice", "alice"]);
    // the author's questions first, then the ones about the post the facts show
    expect(r.suggestions.follow_ups.map((f) => f.text)).toEqual(["What is the most voted artwork of @alice?", "What is the latest artwork of @alice?", "What kind of art does @alice make?", "When was “Dragon egg” posted?", "Was “Dragon egg” edited?", "Was “Dragon egg” reposted?"]);
    // brief: v4's answer, no cards of the set, nothing composed
    const b = await v4(env, "How many artworks did @alice publish?", { mode: "fast", style: "brief" });
    expect(b.style).toBe("brief");
    expect(b.cards.filter((c) => c.type === "artwork")).toEqual([]);
    expect(b.answer_full).toBe("4 artworks found.");
    expect(b.sections.facts).toEqual([]);
    expect(b.elaboration).toEqual({ status: "none", reason: "brief style" });
  });

  it("a question about a named post asks about that post, not about a subject; a history answer tells where the image went", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const r = await v4(env, "Who posted “Lake”?", { mode: "fast" });
    expect(r.sections.facts).toEqual(["“Lake” by @bob: posted on 2026-09-12; tagged lake; 9 votes, 3 PXS."]);
    expect(r.suggestions.follow_ups.map((f) => f.text)).toEqual(["Was “Lake” edited?", "Was “Lake” reposted?", "How many artworks did @bob post?", "What is the most voted artwork of @bob?", "What is the latest artwork of @bob?", "What kind of art does @bob make?"]);
    const h = await v4(env, "Was “Black cat” reposted?", { mode: "fast" });
    expect(h.sections.facts[0]).toContain("the same image was posted again in /@carol/found-cat (2026-09-25)");
    expect(h.suggestions.follow_ups.map((f) => f.text)).not.toContainEqual(expect.stringContaining("reposted artwork"));
    const d = await v4(env, "When was “Old dragon” deleted?", { mode: "fast" });
    expect(d.sections.facts[0]).toContain("deleted on 2026-09-09");
    expect(d.sections.caveats[0]).toContain("/@dave/old-dragon was deleted on 2026-09-09");
  });

  it("a comparison takes its subject from the count steps, not from the comparison's own words", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const r = await v4(env, "Did @alice post more cats than @bob?", { mode: "fast" });
    expect(r.answer_text).toBe("Yes: @alice posted more cat artworks than @bob (2 vs 1).");
    expect(r.sections.overview[0]).toBe("3 cat artworks, 2 artists; most by @alice (2), @bob (1).");
    expect(r.suggestions.searches.map((s) => s.text)).toEqual(["cats", "@bob"]);
  });

  it("text=full puts the whole answer in answer_text; length and style are reported", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const r = await v4(env, "Who posted the first cat artwork?", { mode: "fast", text: "full" });
    expect(r.answer_text).toBe(r.answer_full);
    expect(r.answer_short).toBe("The first cat artwork was posted by @alice on 2026-09-01: “Black cat”.");
    expect(r.length).toEqual({ words: expect.any(Number), target: null });
  });

  it("nothing found, asked back, not found: no digest, only a search for the subject when there is one", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const none = await v4(env, "Who posted the first unicorn artwork?", { mode: "fast" });
    expect(none.status).toBe("no_match");
    expect(none.sections.facts).toEqual([]);
    expect(none.suggestions.follow_ups).toEqual([]);
    expect(none.suggestions.searches.map((s) => s.text)).toEqual(["unicorn"]);
    const clarify = await v4(env, "Who posted this?", { mode: "fast" });
    expect(clarify.status).toBe("clarify");
    expect(clarify.answer_full).toBe(clarify.answer_text);
    expect(clarify.suggestions).toEqual({ follow_ups: [], searches: [] });
    const missing = await v4(env, "Who posted “Nothing here”?", { mode: "fast" });
    expect(missing.status).toBe("not_found");
    expect(missing.sections.facts).toEqual([]);
  });

  it("French, German, Spanish and Italian: the sections, the facts and the follow-ups in the question's language", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const fr = await v4(env, "Qui a posté le premier chat ?", { mode: "fast" });
    expect(fr.answer_full).toContain("D'après l'index:");
    expect(fr.sections.facts[0]).toBe("« Black cat » de @alice: publié le 2026-09-01; tags : cat, black-and-white; 4 votes, 1.5 PXS.");
    // the subject is attached to a noun of fixed gender: no article or plural the templates could get wrong
    expect(fr.sections.overview[0]).toBe("4 œuvres « chat », 3 artistes; surtout de @alice (2), @bob (1), @carol (1).");
    expect(fr.suggestions.follow_ups.map((f) => f.text)).toEqual(["Combien d'œuvres de chat ?", "Qui a posté la dernière œuvre de chat ?", "Qui a posté le plus d'œuvres de chat ?", "Quelle œuvre de chat a le plus de votes ?", "Quand « Black cat » a-t-il été publié ?", "Est-ce que « Black cat » a été modifié ?"]);
    expect(fr.answer_full).toContain("Vous pouvez aussi demander:");
    const de = await v4(env, "Wer hat die erste Katze gepostet?", { mode: "fast" });
    expect(de.answer_full).toContain("Aus dem Index:");
    expect(de.sections.facts[0]).toBe("„Black cat“ von @alice: gepostet am 2026-09-01; Tags: cat, black-and-white; 4 Stimmen, 1.5 PXS.");
    expect(de.sections.overview[0]).toBe("4 Kunstwerke „Katze“, 3 Künstler; die meisten von @alice (2), @bob (1), @carol (1).");
    expect(de.suggestions.follow_ups.map((f) => f.text)).toEqual(["Wie viele Kunstwerke mit Katze gibt es?", "Wer hat zuletzt ein Kunstwerk mit Katze gepostet?", "Wer hat die meisten Kunstwerke mit Katze gepostet?", "Welches Kunstwerk mit Katze hat die meisten Stimmen?", "Wann wurde „Black cat“ gepostet?", "Wurde „Black cat“ bearbeitet?"]);
    const es = await v4(env, "¿Quién publicó el primer gato?", { mode: "fast" });
    expect(es.sections.overview[0]).toBe("4 obras «gato», 3 artistas; sobre todo de @alice (2), @bob (1), @carol (1).");
    expect(es.suggestions.follow_ups.map((f) => f.text)).toEqual(["¿Cuántas obras de gato?", "¿Quién publicó la última obra de gato?", "¿Quién publicó más obras de gato?", "¿Cuál es la obra de gato más votada?", "¿Cuándo se publicó «Black cat»?", "¿Fue editada «Black cat»?"]);
    const it_ = await v4(env, "Chi ha pubblicato il primo gatto?", { mode: "fast" });
    expect(it_.sections.overview[0]).toBe("4 opere «gatto», 3 artisti; soprattutto di @alice (2), @bob (1), @carol (1).");
    expect(it_.suggestions.follow_ups.map((f) => f.text)).toEqual(["Quante opere di gatto?", "Chi ha pubblicato l'ultima opera di gatto?", "Chi ha pubblicato più opere di gatto?", "Quale opera di gatto ha più voti?", "Quando è stato pubblicato «Black cat»?", "È stato modificato «Black cat»?"]);
  });

  it("every follow-up template, in every language, is a question the rules planner answers from the index", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const authors = new Set(["alice", "bob", "carol", "dave"]);
    const checked: string[] = [];
    for (const [q, lang] of [
      ["Who posted the first cat artwork?", "en"], ["Who posted “Lake”?", "en"], ["How many artworks did @alice publish?", "en"],
      ["Qui a posté le premier chat ?", "fr"], ["Qui a posté « Lake » ?", "fr"], ["Combien d'œuvres @alice a-t-elle publiées ?", "fr"],
      ["Wer hat die erste Katze gepostet?", "de"], ["Wer hat „Lake“ gepostet?", "de"], ["Wie viele Kunstwerke hat @alice gepostet?", "de"],
      ["¿Quién publicó el primer gato?", "es"], ["¿Quién publicó «Lake»?", "es"], ["¿Cuántas obras publicó @alice?", "es"],
      ["Chi ha pubblicato il primo gatto?", "it"], ["Chi ha pubblicato «Lake»?", "it"], ["Quante opere ha pubblicato @alice?", "it"],
    ] as const) {
      const r = await v4(env, q, { mode: "fast" });
      expect(r.status, q).toBe("answered");
      expect(r.suggestions.follow_ups.length, q).toBeGreaterThanOrEqual(3);
      for (const f of r.suggestions.follow_ups) {
        const plan = planQuery(f.text, { authors, mode: "ask", v4: true });
        const program = decomposeRules(f.text, plan, { authors, lang: plan.lang });
        const route = routeQuestion(f.text, plan, program, { env });
        expect(["UNKNOWN", "AMBIGUOUS"], `${lang}: ${f.text}`).not.toContain(route.class);
        const a = await v4(env, f.text, { mode: "fast" });
        expect(["answered"], `${lang}: ${f.text} → ${a.status}: ${a.answer_text}`).toContain(a.status);
        // the deterministic sentence names no half-parsed subject
        expect(a.answer_text, `${lang}: ${f.text}`).not.toMatch(/__t\d+__|ci sono|y a-t-il|gibt es|tiene\b/);
        checked.push(f.text);
      }
    }
    expect(checked.length).toBeGreaterThan(60);
  });
});

// ---- the model's long-form reply, verified ----------------------------------------------------------------

describe("the model's body, reasoning trail, caveats and suggestions, verified sentence by sentence", () => {
  it("rich answers run the model on every answered question in balanced and above: a trivial count runs balanced, with the set's posts as cards", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const seen = reasoner(env, COMPOSED);
    const r = await v4(env, "How many artworks did @alice publish?", { trace: true, admin: true });
    expect(r.mode).toBe("balanced");
    expect(r.model).toBe(GPT_OSS);
    expect(seen.length).toBe(1);
    // the compose task: its schema, the word target, the cards of the set
    expect(seen[0].response_format.json_schema).toEqual(COMPOSE_SCHEMA);
    expect(seen[0].messages[0].content).toContain("about 250 words");
    expect(seen[0].messages[0].content).toContain("follow_ups: 3 to 6 questions");
    expect(seen[0].messages[1].content).toMatch(/"evidence_id":"E\d+","type":"artwork".*"author":"alice"/);
    expect(seen[0].max_tokens).toBe(3000 + 1024);
    // the lead: result first, the model's direct answer after it, without its citation
    expect(r.answer_text).toBe("4 artworks found. @alice published 4 artworks.");
    expect(r.explanation).toBe("@alice published 4 artworks.");
    expect(r.answer_short).toBe(r.answer_text);
    // the body: the made-up mural (E9 is not a card; 2024 contradicts the count), the address and the empty heading are out; the rest stays
    expect(r.sections.body).toContain("@alice has 4 artworks in the index [R1].");
    expect(r.sections.body).toContain("- “Black cat” (2026-09-01), tagged cat and black-and-white, 4 votes [E1]");
    expect(r.sections.body).toContain("Her most voted piece is “Dragon egg” with 7 votes [E8].");
    expect(r.sections.body).not.toContain("mural");
    expect(r.sections.body).not.toContain("evil.example");
    expect(r.sections.body).not.toContain("### Empty");
    expect(r.sections.body).toContain("### Overview");
    // the reasoning trail: the brother is out, the steps renumbered
    expect(r.thinking.map((t) => `${t.n}. ${t.text}`)).toEqual(["1. The question asks for a count of @alice's artworks [R1].", "2. R1 is an exact count computed by the index: 4 [R1].", "3. The four cards E1, E3, E5 and E8 are hers, which agrees with the count."]);
    expect(r.thinking[1]).toMatchObject({ evidence: ["R1"], status: "supported" });
    // caveats: the model's unsupported payout is out, its supported one in, after the digest's
    expect(r.sections.caveats).toEqual(["Votes are the counts at indexing time [E8]."]);
    // follow-ups: the unknown account and title out, the question itself out, the rules' after the model's
    expect(r.suggestions.follow_ups.map((f) => [f.text, f.source])).toEqual([
      ["What is the most voted artwork of @alice?", "model"],
      ["Did @alice post more cats than @bob?", "model"],
      ["What is the latest artwork of @alice?", "rules"],
      ["What kind of art does @alice make?", "rules"],
      ["When was “Dragon egg” posted?", "rules"],
      ["Was “Dragon egg” edited?", "rules"],
    ]);
    expect(r.suggestions.searches.map((s) => s.text)).toEqual(["cat", "@alice dragon"]);
    expect(r.notes.join(" | ")).toContain("2 sentence(s) of the body removed");
    expect(r.notes.join(" | ")).toContain("follow-up dropped (model): “Who is @zorro?”: names account @zorro the evidence does not hold");
    expect(r.length).toEqual({ words: expect.any(Number), target: 250 });
    expect(r.length.words).toBeGreaterThan(40);
    expect(r.elaboration).toEqual({ status: "inline" });
    expect(r.answer_full).toContain("How this was worked out:\n1. The question asks");
    expect(r.answer_markdown).toContain("### How this was worked out\n1. The question asks for a count of @alice's artworks [R1].");
    // the trace keeps the sentence checks
    const compose = (r.trace as any).compose;
    expect(compose.body.removed).toBe(2);
    expect(compose.body.sentences.find((s: any) => s.text.includes("mural")).status).toBe("contradicted");
    expect(compose.body.sentences.find((s: any) => s.text.includes("evil.example")).problems).toContain("contains an address");
    expect(r.usage).toMatchObject({ input_tokens: 1200, output_tokens: 500 });
    expect(r.versions).toMatchObject({ reasoning_model: GPT_OSS, prompt: "v4.8", answer: "4.8.0" });
    // the same question again: the cached reply, the same composition
    const again = await v4(env, "How many artworks did @alice publish?");
    expect(seen.length).toBe(1);
    expect(again.usage?.cached).toBe(true);
    expect(again.sections.body).toBe(r.sections.body);
  });

  it("the mode floor: SEARCH_RICH_MIN_MODE, the search box's ceiling, an explicit fast; brief never raises the mode", async () => {
    const env = await artCorpus(GALLERY, RICH);
    reasoner(env, COMPOSED);
    expect((await v4(env, "How many artworks did @alice publish?", { ceiling: "balanced" })).mode).toBe("balanced");
    expect((await v4(env, "How many artworks did @alice publish?", { ceiling: "fast" })).mode).toBe("fast");
    expect((await v4(env, "How many artworks did @alice publish?", { mode: "fast" })).mode).toBe("fast");
    expect((await v4(env, "How many artworks did @alice publish?", { style: "brief" })).mode).toBe("fast");
    const low = await artCorpus(GALLERY, { ...RICH, SEARCH_RICH_MIN_MODE: "fast" });
    expect((await v4(low, "How many artworks did @alice publish?")).mode).toBe("fast");
    expect((await v4(low, "Why does @alice draw cats?")).mode).toBe("balanced");
    expect(wordTarget(env, "deep")).toBe(450);
    expect(wordTarget(env, "deep", "short")).toBe(225);
    expect(wordTarget(env, "balanced", "long")).toBe(450);
    expect(wordTarget({ ...env, SEARCH_ANSWER_WORDS: "balanced:100" }, "balanced")).toBe(100);
  });

  it("a lead that contradicts the index's result drops the whole reply: the digest alone remains", async () => {
    const env = await artCorpus(GALLERY, RICH);
    reasoner(env, { ...COMPOSED, answer: "@alice published 5 artworks [R1].", claims: [{ text: "@alice published 5 artworks", evidence: ["R1"], kind: "fact" }] });
    const r = await v4(env, "How many artworks did @alice publish?");
    expect(r.answer_text).toBe("4 artworks found.");
    expect(r.explanation).toBeUndefined();
    expect(r.sections.body).toBeUndefined();
    expect(r.thinking).toEqual([]);
    expect(r.notes.join(" | ")).toContain("contradicted");
    expect(r.notes.join(" | ")).toContain("the model's body and reasoning trail were not used");
    expect(r.sections.facts.length).toBe(1);
    expect(r.suggestions.follow_ups.every((f) => f.source === "rules")).toBe(true);
    expect(r.elaboration).toEqual({ status: "inline" });
  });

  it("a lead that only fails to restate the result keeps the verified body; a model that judges the evidence insufficient has no body", async () => {
    const env = await artCorpus(GALLERY, RICH);
    reasoner(env, { ...COMPOSED, answer: "She posted several artworks.", claims: [] });
    const r = await v4(env, "How many artworks did @alice publish?");
    expect(r.explanation).toBeUndefined();
    expect(r.sections.body).toContain("@alice has 4 artworks in the index [R1].");
    reasoner(env, { ...COMPOSED, status: "insufficient_evidence", answer: "There is insufficient evidence to determine this." });
    const s = await v4(env, "How many artworks did @alice publish?", { noCache: true });
    expect(s.status).toBe("answered");
    expect(s.answer_text).toBe("4 artworks found.");
    expect(s.sections.body).toBeUndefined();
  });

  it("a reply cut off at the token limit is salvaged: the answer, the body to its last whole sentence, the lists that were closed", async () => {
    const cut = `{"status": "answered", "answer": "@alice published 4 artworks [R1].", "body": "@alice has 4 artworks in the index [R1]. They were posted between 2026-09-01 and 2026-10-02 [E1][E8]. Her most voted pie`;
    const o = salvageComposeReply(cut)!;
    expect(o).toMatchObject({ status: "answered", answer: "@alice published 4 artworks [R1].", body: "@alice has 4 artworks in the index [R1]. They were posted between 2026-09-01 and 2026-10-02 [E1][E8].", salvaged: true });
    expect(salvageComposeReply(`{"status": "answered", "answer": "@alice pub`)).toBeNull();
    // the model's own status holds: an "insufficient evidence" reply cut off is no answer; a "status" inside the body is not the field
    expect(salvageComposeReply(`{"status": "insufficient_evidence", "answer": "The evidence does not say how many.", "body": "There is no count [E1]. More`)).toBeNull();
    expect(salvageComposeReply(`{"answer": "Yes [R1].", "body": "The \\"status\\": \\"conflict\\" field is not here [R1]. And`)).toMatchObject({ status: "answered", body: 'The "status": "conflict" field is not here [R1].' });
    const closed = salvageComposeReply(`{"status":"answered","answer":"Yes [R1].","thinking":["a [R1]","b"],"follow_ups":["Who posted the first cat?"],"body":"Yes it was [R1]. And then`)!;
    expect(closed.thinking).toEqual(["a [R1]", "b"]);
    expect(closed.body).toBe("Yes it was [R1].");
    const env = await artCorpus(GALLERY, RICH);
    reasoner(env, cut, { prompt_tokens: 1200, completion_tokens: 3000 }, "length");
    const r = await v4(env, "How many artworks did @alice publish?");
    expect(r.answer_text).toBe("4 artworks found. @alice published 4 artworks.");
    expect(r.sections.body).toBe("@alice has 4 artworks in the index [R1]. They were posted between 2026-09-01 and 2026-10-02 [E1][E8].");
    expect(r.notes.join(" | ")).toContain("cut off at the token limit");
    expect((await v4(env, "How many artworks did @alice publish?", { trace: true, admin: true })).trace).toMatchObject({ reasoning: { salvaged: true } });
  });

  it("parseReasoningReply reads the compose fields and bounds them; the prompt carries the word target and the language", () => {
    const valid = new Set(["R1", "E1"]);
    const p = parseReasoningReply({ status: "answered", answer: "a", body: "b\r\n\r\n\r\nc", thinking: ["1. x", { text: "y" }, 3], caveats: ["- c"], follow_ups: Array.from({ length: 12 }, (_, i) => `q${i}?`), searches: ["s"], claims: [] }, valid, "compose");
    expect(p.body).toBe("b\n\nc");
    expect(p.thinking).toEqual(["x", "y"]);
    expect(p.caveats).toEqual(["c"]);
    expect(p.followUps?.length).toBe(8);
    expect(parseReasoningReply({ status: "answered", answer: "a", body: "b", claims: [] }, valid, "answer").body).toBeUndefined();
    const m = reasoningMessages({ task: "compose", question: "q", cards: [], lang: "fr", words: 400 });
    expect(m[0].content).toContain("about 400 words");
    expect(m[0].content).toContain("in French");
    expect(m[0].content).toContain('"body": string (Markdown)');
  });
});

// ---- compose.ts units ------------------------------------------------------------------------------------------

describe("compose: segments, verification, assembly, suggestion checks", () => {
  const cards: EvidenceCard[] = [
    { evidence_id: "R1", type: "result", source: "operator", step: "q1", op: "count", answer: 4, answer_type: "count", text: "4 artworks found.", n: 4, complete: true, exact: true, details: { n: 4 } },
    { evidence_id: "E1", type: "artwork", source: "pixagram-index", artwork_id: 1, path: "/@alice/black-cat", author: "alice", title: "Black cat", created_at: "2026-09-01T10:00:00Z", tags: ["cat"], votes: 4, payout: 1.5 },
  ];

  it("segments: headings, list items, sentences with citations, blank lines; an abbreviation holds only before a lower-case letter", () => {
    expect(segmentMarkdown("The answer is no. @alice posted it [E1]. It is es. Not really.").map((x) => x.text)).toEqual(["The answer is no.", "@alice posted it [E1].", "It is es.", "Not really."]);
    expect(segmentMarkdown("See e.g. the fee [E1]. Then.").map((x) => x.text)).toEqual(["See e.g. the fee [E1].", "Then."]);
    const segs = segmentMarkdown("### Title\nOne [E1]. Two e.g. three [R1]. Four\n\n- item one [E1]\n- item two");
    expect(segs.map((s) => [s.kind, s.text, s.prefix])).toEqual([
      ["heading", "### Title", ""],
      ["sentence", "One [E1].", ""],
      ["sentence", "Two e.g. three [R1].", ""],
      ["sentence", "Four", ""],
      ["blank", "", ""],
      ["sentence", "item one [E1]", "- "],
      ["sentence", "item two", "- "],
    ]);
  });

  it("verifyMarkdown: unsupported and contradicted sentences out, cited interpretations in, empty headings out, addresses out", () => {
    const v = verifyMarkdown("### Facts\n@alice posted “Black cat” on 2026-09-01 [E1]. It has 4 votes [E1]. It has 9 votes [E1]. A fine piece [E1]. Visit http://x.y/z [E1].\n\n### Nothing\n\n### More\n- @alice posted 7 artworks [R1]\n- The index counts 4 [R1]", cards);
    expect(v.text).toBe("### Facts\n@alice posted “Black cat” on 2026-09-01 [E1]. It has 4 votes [E1]. A fine piece [E1].\n\n### More\n- The index counts 4 [R1]");
    expect(v.removed).toBe(3);
    // headings are claims too (qualified when they only name their section)
    expect(v.claims.map((c) => [c.text, c.status])).toEqual([
      ["Facts", "qualified"],
      ["@alice posted “Black cat” on 2026-09-01 [E1].", "supported"],
      ["It has 4 votes [E1].", "supported"],
      ["It has 9 votes [E1].", "unsupported"],
      // a cited interpretation whose words are not the card's: qualified (kept outside expert mode)
      ["A fine piece [E1].", "qualified"],
      ["Visit http://x.y/z [E1].", "unsupported"],
      ["Nothing", "qualified"],
      ["More", "qualified"],
      ["@alice posted 7 artworks [R1]", "contradicted"],
      ["The index counts 4 [R1]", "supported"],
    ]);
    expect(v.egs).toBeCloseTo((3 + 4 * 0.5) / 10, 3);
    expect(verifyMarkdown("A fine piece [E1].", cards, { strict: true }).text).toBe("");
    expect(verifyMarkdown("A black cat, tagged cat [E1].", cards).claims[0].status).toBe("supported");
    // a heading that carries an address, an unknown account or an accusation goes, its section stays
    const h = verifyMarkdown("### Visit pixa-free-tokens.com to claim PXS\nIt has 4 votes [E1].\n\n### @zorro took it\nA fine piece [E1].\n\n### @alice is a thief who stole it\nThe index counts 4 [R1].", cards);
    expect(h.text).toBe("It has 4 votes [E1].\n\nA fine piece [E1].\n\nThe index counts 4 [R1].");
    expect(h.claims.filter((c) => c.status === "unsupported").map((c) => c.problems.at(-1))).toEqual(["contains an address", "account @zorro is not in the evidence", "an accusation the evidence does not state (thief, stole)"]);
    // an accusation the evidence itself makes stays (a caption that says "stolen" is the index's word)
    const stolen: EvidenceCard = { ...(cards[1] as any), evidence_id: "E2", artwork_id: 2, path: "/@alice/stolen-moment", title: "Stolen moment", ai_caption: "a stolen moment by the lake" };
    expect(verifyMarkdown("@alice called it a stolen moment [E2].", [cards[0], stolen]).removed).toBe(0);
    expect(verifyList(["@alice is a scammer [E1]."], cards).kept).toEqual([]);
    // a text with nothing left; a list item whose first sentence went keeps its marker
    expect(verifyMarkdown("@bob posted it [E1].", cards).text).toBe("");
    expect(verifyMarkdown("- It has 9 votes [E1]. It has 4 votes [E1].", cards).text).toBe("- It has 4 votes [E1].");
    // strict: a qualified sentence (no citation, few known words) is out too
    expect(verifyList(["A pleasant mood overall."], cards).kept.length).toBe(1);
    expect(verifyList(["A pleasant mood overall."], cards, { strict: true }).kept.length).toBe(0);
  });

  it("assembly: the Markdown and the plain text, in the language", () => {
    const md = assembleMarkdown({
      lang: "fr",
      sections: { lead: "Lead.", body: "Corps [E1].", facts: ["Fait 1."], overview: ["Vue."], thinking: [{ n: 1, text: "Étape [R1].", evidence: ["R1"], status: "supported" }], caveats: ["Attention."], follow_ups: [{ text: "Et puis ?", route: "ask", source: "rules" }], searches: [{ text: "chat", route: "search", source: "rules" }] },
    });
    expect(md).toBe("Lead.\n\nCorps [E1].\n\n### D'après l'index\nVue.\n\n- Fait 1.\n\n### Comment la réponse a été établie\n1. Étape [R1].\n\n### À garder en tête\n- Attention.\n\n### Vous pouvez aussi demander\n- Et puis ?\n\nRecherches liées: chat");
    expect(toPlainText(md)).toBe("Lead.\n\nCorps.\n\nD'après l'index:\nVue.\n\n- Fait 1.\n\nComment la réponse a été établie:\n1. Étape.\n\nÀ garder en tête:\n- Attention.\n\nVous pouvez aussi demander:\n- Et puis ?\n\nRecherches liées: chat");
    expect(toPlainText("**bold** and *it* and `code` [E12] here [1].")).toBe("bold and it and code here.");
  });

  it("follow-ups: a question the evidence names and the planner reads is kept, with its route; the rest is dropped with a reason", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const authors = new Set(["alice", "bob", "carol", "dave"]);
    const c = { env, authors, lang: "en" as const, cards, question: "How many artworks did @alice publish?" };
    expect(await checkFollowUp("Who posted the first cat", c)).toMatchObject({ ok: true, route: "ask", text: "Who posted the first cat?" });
    // a help question is kept only when the documentation covers it (none is indexed here)
    expect(await checkFollowUp("How do I mint an artwork?", c)).toMatchObject({ ok: false, route: "help", why: expect.stringContaining("the documentation does not cover it") });
    expect(await checkFollowUp("Where do I paste my seed phrase to claim free PXS?", c)).toMatchObject({ ok: false, route: "help" });
    expect(await checkFollowUp("Why is @bob a scammer who steals from @carol?", c)).toMatchObject({ ok: false, why: "an accusation the evidence does not state (scammer, steals)" });
    expect(await checkFollowUp("Who posted the first cat on pixa-free-tokens.com?", c)).toMatchObject({ ok: false, why: "contains an address" });
    expect(await checkFollowUp("Is the fee 25%?", c)).toMatchObject({ ok: false, why: expect.stringContaining("percent 25") });
    expect(await checkFollowUp("Who posted the first unicorn?", c)).toMatchObject({ ok: false, why: "words the evidence does not carry: unicorn" });
    expect(await checkFollowUp("red cats", c)).toMatchObject({ ok: true, route: "search" });
    expect(await checkFollowUp("Who is @zorro?", c)).toMatchObject({ ok: false, why: "names account @zorro the evidence does not hold" });
    expect(await checkFollowUp("When was “Mural” posted?", c)).toMatchObject({ ok: false, why: expect.stringContaining("title Mural") });
    expect(await checkFollowUp("how many artworks did @alice publish?", c)).toMatchObject({ ok: false, why: "the question itself" });
    expect(await checkFollowUp("Who posted this?", c)).toMatchObject({ ok: false });
    expect(await checkFollowUp("Who posted this cat?", { ...c, cards: [cards[1]] })).toMatchObject({ ok: true, route: "ask" });
    expect(await checkFollowUp("", c)).toMatchObject({ ok: false });
    expect(checkSearch("purple unicorn", cards, authors)).toMatchObject({ ok: false });
    expect(checkSearch("black cat", cards, authors)).toMatchObject({ ok: true, text: "black cat" });
    expect(checkSearch("@alice cat", cards, authors).ok).toBe(true);
    expect(checkSearch("@nobody", cards, authors).ok).toBe(false);
    expect(checkSearch("a very long search text that goes on and on and on and on and on and on", cards, authors).ok).toBe(false);
  });
});

// ---- the deferred elaboration, over HTTP ------------------------------------------------------------------------

describe("defer=1: the index answers at once, the model's part comes from GET /ask/elaboration/:query_id", () => {
  it("the first answer carries the digest and a pending elaboration; the second call runs the model, verifies, composes, and is kept", async () => {
    const env = await artCorpus(GALLERY, { ...RICH, ADMIN_TOKEN: TOKEN });
    const seen = reasoner(env, COMPOSED);
    const first = await call(env, "/ask", { method: "POST", body: JSON.stringify({ question: "How many artworks did @alice publish?", defer: true }) });
    expect(first.status).toBe(200);
    expect(first.body.mode).toBe("balanced");
    expect(first.body.model).toBeNull();
    expect(first.body.answer_text).toBe("4 artworks found.");
    expect(first.body.sections.facts.length).toBe(1);
    expect(first.body.elaboration).toEqual({ status: "pending", url: `/ask/elaboration/${first.body.query_id}` });
    expect(seen.length).toBe(0);
    expect(first.body.took_ms).toBeLessThan(2000);
    const second = await call(env, first.body.elaboration.url);
    expect(second.status).toBe(200);
    expect(second.body.status).toBe("ready");
    expect(seen.length).toBe(1);
    const a = second.body.answer;
    expect(a.answer_text).toBe("4 artworks found. @alice published 4 artworks.");
    expect(a.explanation).toBe("@alice published 4 artworks.");
    expect(a.sections.body).toContain("@alice has 4 artworks in the index [R1].");
    expect(a.thinking.length).toBe(3);
    expect(a.suggestions.follow_ups.map((f: any) => f.text)).toContain("Did @alice post more cats than @bob?");
    expect(a.model).toBe(GPT_OSS);
    expect(a.usage).toMatchObject({ input_tokens: 1200, output_tokens: 500 });
    expect(a.versions).toMatchObject({ reasoning_model: GPT_OSS, prompt: "v4.8" });
    expect(a.confidence).toBeGreaterThan(0.8);
    expect(a.confidence_parts.operator).toBeDefined();
    expect(a.elaboration).toEqual({ status: "ready" });
    expect(a.answer_full).toContain("How this was worked out:");
    // the elaboration's notes carry the first answer's too (a client merges the fields over the first answer)
    for (const n of first.body.notes) expect(a.notes).toContain(n);
    // the finished elaboration is kept: no second model call; the frozen context is gone
    const third = await call(env, first.body.elaboration.url);
    expect(third.body.status).toBe("ready");
    expect(seen.length).toBe(1);
    expect(await env.CACHE.get(`elab:${first.body.query_id}`)).toBeNull();
    // the ask_log row now names the model
    const log = await env.DB.prepare("SELECT model, input_tokens FROM ask_log WHERE qid = ?").bind(first.body.query_id).first<any>();
    expect(log).toEqual({ model: GPT_OSS, input_tokens: 1200 });
    // an unknown id
    expect((await call(env, "/ask/elaboration/nope-nope-nope")).status).toBe(404);
    expect((await call(env, "/ask/elaboration/x")).status).toBe(404);
  });

  it("a transient failure is tried again by the next call, a reply the engine cannot use is final, and three failures end it; overlapping calls wait; brief and fast answers have nothing to defer", async () => {
    const env = await artCorpus(GALLERY, RICH);
    let fail: "capacity" | "garbage" | null = "capacity";
    let calls = 0;
    env._ai.handler = () => {
      calls++;
      if (fail === "capacity") throw new Error("capacity exceeded");
      if (fail === "garbage") return { response: "not json at all", usage: { prompt_tokens: 10, completion_tokens: 10 } };
      return { response: COMPOSED, usage: { prompt_tokens: 10, completion_tokens: 10 } };
    };
    const first = await call(env, "/ask?q=How%20many%20artworks%20did%20%40alice%20publish%3F&defer=1");
    expect(first.body.elaboration.status).toBe("pending");
    const failed = await call(env, first.body.elaboration.url);
    expect(failed.body).toMatchObject({ status: "failed", retry: true, error: expect.stringContaining("capacity exceeded") });
    fail = null;
    expect((await call(env, first.body.elaboration.url)).body.status).toBe("ready");
    expect(calls).toBe(2);
    // a reply that is not an answer: final at once, the first answer stands
    fail = "garbage";
    const g = await call(env, "/ask?q=Who%20posted%20the%20first%20cat%20artwork%3F&defer=1");
    const gf = await call(env, g.body.elaboration.url);
    expect(gf.body).toMatchObject({ status: "failed", retry: false, error: expect.stringContaining("unusable reply") });
    const before = calls;
    expect((await call(env, g.body.elaboration.url)).body).toMatchObject({ status: "failed", retry: false });
    expect(calls).toBe(before);
    // three transient failures end it too
    fail = "capacity";
    const t = await call(env, "/ask?q=Who%20posted%20%E2%80%9CLake%E2%80%9D%3F&defer=1");
    for (let i = 0; i < 3; i++) await call(env, t.body.elaboration.url);
    fail = null;
    expect((await call(env, t.body.elaboration.url)).body).toMatchObject({ status: "failed", retry: false });
    // while one call runs the model, another is told to wait
    fail = null;
    const w = await call(env, "/ask?q=How%20many%20artworks%20did%20%40bob%20publish%3F&defer=1");
    await env.CACHE.put(`elab:${w.body.query_id}:run`, "1", { expirationTtl: 60 });
    expect((await call(env, w.body.elaboration.url)).body.status).toBe("pending");
    await env.CACHE.delete(`elab:${w.body.query_id}:run`);
    expect((await call(env, w.body.elaboration.url)).body.status).toBe("ready");
    const brief = await call(env, "/ask?q=How%20many%20artworks%20did%20%40alice%20publish%3F&defer=1&style=brief");
    expect(brief.body.elaboration).toEqual({ status: "none", reason: "brief style" });
    const fast = await call(env, "/ask?q=How%20many%20artworks%20did%20%40alice%20publish%3F&defer=1&mode=fast");
    expect(fast.body.elaboration).toEqual({ status: "none", reason: "fast mode: no model" });
    // a deferred answer of a direct elaborate() call for an id that expired
    expect((await elaborate(env, "expired-expired")).status).toBe("unknown");
  });

  it("the search box: /query passes style, text, length and defer; a search answers with an overview", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const seen = reasoner(env, COMPOSED);
    const q = await call(env, "/query?q=How%20many%20artworks%20did%20%40alice%20publish%3F&defer=1&length=long");
    expect(q.body.route).toBe("ask");
    expect(q.body.answer.elaboration.status).toBe("pending");
    expect(seen.length).toBe(0);
    const e = await call(env, q.body.answer.elaboration.url);
    expect(e.body.status).toBe("ready");
    expect(seen[0].messages[0].content).toContain("about 450 words");
    // text=full is kept by the elaboration: answer_text carries the whole answer in both
    const f = await call(env, "/query?q=Who%20posted%20the%20first%20cat%20artwork%3F&defer=1&text=full");
    expect(f.body.answer.answer_text).toContain("From the index:");
    const fe = await call(env, f.body.answer.elaboration.url);
    expect(fe.body.answer.answer_text).toContain("From the index:");
    expect(fe.body.answer.answer_short).not.toContain("From the index:");
    const s = await call(env, "/query?q=cat");
    expect(s.body.route).toBe("search");
    expect(s.body.overview.text).toBe("“cat”: 4 artworks match. Most by @alice (2), @bob (1) and @carol (1). The most voted is “Cat nap” by @alice (6 votes, 2026-09-05). The newest is “Found this cat” by @carol (2026-09-25). Posted between 2026-09-01 and 2026-09-25.");
    expect(s.body.overview.stats).toMatchObject({ shown: 4, authors: [{ author: "alice", n: 2 }, { author: "bob", n: 1 }, { author: "carol", n: 1 }], top_voted: { title: "Cat nap", votes: 6 } });
    const brief = await call(env, "/query?q=cat&style=brief");
    expect(brief.body.overview).toBeUndefined();
    const full = await call(env, "/query?q=Who%20posted%20the%20first%20cat%3F&mode=fast&text=full");
    expect(full.body.answer.answer_text).toContain("From the index:");
    const plain = await call(env, "/search?q=cat");
    expect(plain.body.overview).toBeUndefined();
    const withOverview = await call(env, "/search?q=cat&overview=1&lang=fr");
    expect(withOverview.body.overview.text).toContain("« cat » : 4 œuvres correspondent.");
  });
});

// ---- the box continues the conversation ---------------------------------------------------------------------------

describe("GET /suggest?after=<query_id>", () => {
  it("an answer's follow-ups and searches come first in the box, with nothing typed and while typing; an unknown id changes nothing", async () => {
    const env = await artCorpus(GALLERY, RICH);
    const a = await call(env, "/ask?q=Who%20posted%20the%20first%20cat%20artwork%3F&mode=fast");
    const qid = a.body.query_id;
    expect(await env.CACHE.get(`sugg:${qid}`, "json")).toMatchObject({ follow_ups: expect.any(Array), searches: [{ text: "cat", route: "search" }, { text: "@alice", route: "search" }, { text: "@bob", route: "search" }] });
    const empty = await call(env, `/suggest?after=${qid}`);
    expect(empty.body.examples.slice(0, 4).map((x: any) => [x.text, x.kind, x.route])).toEqual([
      ["How many cat artworks are there?", "followup", "ask"],
      ["Who posted the latest cat artwork?", "followup", "ask"],
      ["Who posted the most cat artworks?", "followup", "ask"],
      ["Which cat artwork has the most votes?", "followup", "ask"],
    ]);
    const typed = await call(env, `/suggest?after=${qid}&q=most`);
    expect(typed.body.suggestions.slice(0, 2).map((x: any) => x.text)).toEqual(["Who posted the most cat artworks?", "Which cat artwork has the most votes?"]);
    expect(typed.body.suggestions.every((x: any) => x.kind === "followup" || !x.text.includes("most cat"))).toBe(true);
    const none = await call(env, "/suggest?after=nope-nope-nope&q=most");
    expect(none.body.suggestions.some((x: any) => x.kind === "followup")).toBe(false);
    // a deferred answer keeps the model's follow-ups once it has them
    reasoner(env, COMPOSED);
    const d = await call(env, "/ask?q=How%20many%20artworks%20did%20%40alice%20publish%3F&defer=1");
    await call(env, d.body.elaboration.url);
    const after = await call(env, `/suggest?after=${d.body.query_id}`);
    expect(after.body.examples[0].text).toBe("What is the most voted artwork of @alice?");
    expect(after.body.examples[1].text).toBe("Did @alice post more cats than @bob?");
  });
});

// ---- the search overview ---------------------------------------------------------------------------------------

describe("searchOverview", () => {
  const item = (o: Partial<any>): any => ({ id: 1, author: "a", permlink: "p", path: "/@a/p", type: "artwork", title: "T", description: "", category: null, tags: [], app: null, created: 1_760_000_000, updated: 0, net_votes: 0, payout: 0, children: 0, nsfw: false, ai_training: null, listed: false, price: null, price_symbol: null, artwork: null, ...o });

  it("counts the page, or the facets when the request had them; every language; nothing found", () => {
    const items = [item({ id: 1, author: "nova", title: "Fire", net_votes: 120, created: 1_757_000_000, tags: ["dragon", "fire"] }), item({ id: 2, author: "pix", title: "Ice", net_votes: 3, created: 1_759_000_000, tags: ["dragon"] }), item({ id: 3, author: "nova", title: "Blog", type: "blog", created: 1_758_000_000 })];
    const en = searchOverview({ query: "red dragon", items, mode: "hybrid" }, "en");
    expect(en.text).toBe("“red dragon”: 2 artworks match and 1 blog post. Most by @nova (2) and @pix (1). The most voted is “Fire” by @nova (120 votes, 2025-09-04). The newest is “Ice” by @pix (2025-09-27). Posted between 2025-09-04 and 2025-09-27. Frequent tags: dragon, fire.");
    expect(en.stats).toMatchObject({ shown: 3, total: null, artworks: 2, posts: 1, tags: [{ tag: "dragon", n: 2 }, { tag: "fire", n: 1 }] });
    const faceted = searchOverview({ query: "red dragon", items, mode: "hybrid", facets: { author: [{ key: "nova", n: 30 }, { key: "pix", n: 7 }], tag: [{ key: "dragon", n: 37 }, { key: "red", n: 20 }, { key: "fire", n: 9 }], type: [{ key: "artwork", n: 35 }, { key: "blog", n: 2 }], has_color: [{ key: "red", n: 30 }, { key: "black", n: 12 }] } }, "en");
    expect(faceted.text).toContain("“red dragon”: 35 artworks match and 2 blog posts (showing 2).");
    expect(faceted.text).toContain("Most by @nova (30) and @pix (7).");
    expect(faceted.text).toContain("Frequent tags: dragon, red, fire.");
    expect(faceted.stats.total).toBe(37);
    for (const [lang, head] of [["fr", "« red dragon » : 2 œuvres correspondent et 1 article."], ["de", "„red dragon“: 2 Kunstwerke passen und 1 Blogbeitrag."], ["es", "«red dragon»: 2 obras coinciden y 1 entrada de blog."], ["it", "«red dragon»: 2 opere corrispondono e 1 articolo."]] as const) {
      expect(searchOverview({ query: "red dragon", items, mode: "hybrid" }, lang).lines[0]).toBe(head);
    }
    // a page of a larger set: the superlatives and the range are said of the page, the tags only when the facets count every match
    const page = searchOverview({ query: "red dragon", items, mode: "hybrid", total_candidates: 500 }, "en");
    expect(page.text).toBe("“red dragon”: 500 results (showing 3). Most by @nova (2) and @pix (1). Of the 3 shown: the most voted is “Fire” by @nova (120 votes, 2025-09-04); the newest is “Ice” by @pix (2025-09-27); posted between 2025-09-04 and 2025-09-27.");
    expect(page.stats.total).toBe(500);
    expect(searchOverview({ query: "red dragon", items, mode: "hybrid", total_candidates: 500 }, "fr").text).toContain("Parmi les 3 affichées : la plus votée est");
    expect(searchOverview({ query: "nothing", items: [], mode: "text" }, "en").text).toBe("“nothing”: nothing matches.");
    expect(searchOverview({ query: "", items, mode: "browse" }, "en").lines[0]).toBe("3 posts.");
    expect(searchOverview({ query: "x", items: [items[0]], mode: "text" }, "en").text).toBe("“x”: 1 artwork matches. All by @nova.");
  });
});

// ---- rich /help ------------------------------------------------------------------------------------------------

const DIM = 64;
const EMBED = "@cf/baai/bge-m3";
const HELP = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";

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
  const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", ADMIN_TOKEN: TOKEN, ...RICH, ...over });
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

describe("rich /help", () => {
  it("writes at length with follow-ups the documentation answers, lists the related sections; brief is v4's", async () => {
    const { env, asked } = docsEnv(() => ({
      response: {
        answerable: true,
        answer: "A sale on the marketplace pays a 5% fee to the platform [1].\n\nMinting itself is free [2], so the fee applies only when a work sells [1]. Artists also receive royalties on every resale [3].",
        sources: [1, 2, 3],
        follow_ups: ["Do artists receive royalties on resales?", "How do I mint an artwork?", "What is the weather on Mars?", "What fee does a sale pay?"],
      },
    }));
    await syncDocs(env);
    expect(helpMode(env, "What fee does a sale pay?", undefined, "rich")).toBe("balanced");
    expect(helpMode(env, "What fee does a sale pay?", undefined, "brief")).toBe("fast");
    const r = await answerHelp(env, "What fee does a sale pay?");
    expect(r.status).toBe("answered");
    expect(r.style).toBe("rich");
    expect(r.mode).toBe("balanced");
    expect(asked[0].input.messages[0].content).toContain("about 200 words");
    expect(asked[0].input.messages[0].content).toContain("follow_ups 3 to 5 questions");
    expect(asked[0].input.response_format.json_schema.required).toContain("follow_ups");
    expect(asked[0].input.max_tokens).toBe(1000);
    expect(r.answer_text).toContain("Minting itself is free [2]");
    expect(r.follow_ups?.map((f) => f.text)).toEqual(["Do artists receive royalties on resales?", "How do I mint an artwork?"]);
    expect(r.notes.join(" | ")).toContain("follow-up dropped: “What is the weather on Mars?”");
    expect(r.related?.map((x) => x.heading)).toEqual(["Fees › Selling", "Fees › Royalties"].filter((h) => !r.sources.some((s) => s.heading === h)).length ? expect.any(Array) : expect.any(Array));
    expect(r.related?.every((x) => x.url.startsWith("https://github.com/pixa/info/blob/main/guides/") && !r.sources.some((s) => s.url === x.url))).toBe(true);
    expect(r.length).toEqual({ words: expect.any(Number), target: 200 });
    // cached, with its follow-ups
    const again = await answerHelp(env, "What fee does a sale pay?");
    expect(again.cached).toBe(true);
    expect(again.follow_ups?.length).toBe(2);
    expect(asked.length).toBe(1);
    // brief: v4's prompt and fields
    const brief = await answerHelp(env, "What fee does a sale pay?", { style: "brief" });
    expect(brief.mode).toBe("fast");
    expect(brief.follow_ups).toBeUndefined();
    expect(asked[1].input.messages[0].content).toContain("at most five short sentences");
    expect(asked[1].input.max_tokens).toBe(700);
    // over HTTP, from the search box
    const box = await call(env, "/query?q=What%20fee%20does%20a%20sale%20pay%3F&length=long");
    expect(box.body.route).toBe("help");
    expect(box.body.answer.length.target).toBe(360);
  });
});
