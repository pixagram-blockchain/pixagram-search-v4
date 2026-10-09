// Regressions of the independent review of v4 (beyond the answer layer): /help's sentence removal
// never glues text into an address, and its numbers are checked as the excerpts state them; the
// verifier's conflicts and checks; the nsfw filter on image questions and histories; /query's
// ceiling applies to auto only; POST /ask stays small unless it carries an image.

import { describe, expect, it } from "vitest";
import { FakeExec, type TestEnv } from "./harness/fakes";
import { artCorpus, GALLERY, hashOf } from "./harness/corpus";
import { app } from "../src/api";
import { ask, type AskResponseV4 } from "../src/search/ask";
import { answerSegments, checkHelpAnswer, validateHelpReply } from "../src/help/answer";
import { conflictSentence } from "../src/search/verifier";
import { refreshImageHistory } from "../src/chain/versions";

const v4 = async (env: TestEnv, question: string, extra: Record<string, unknown> = {}) => (await ask(env, { question, noCache: true, ...extra })) as AskResponseV4;

async function call(env: TestEnv, path: string, init?: RequestInit): Promise<{ status: number; body: any }> {
  const exec = new FakeExec();
  const res = await app.fetch(new Request(`https://search.test${path}`, init), env, exec as unknown as ExecutionContext);
  await exec.settle();
  const text = await res.text();
  let body: any = text;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    /* not JSON */
  }
  return { status: res.status, body };
}

const hit = (n: number, text: string) => ({ id: n, path: `p${n}.md`, title: `Page ${n}`, heading: "", anchor: "", url: `https://github.com/pixa/info/blob/main/p${n}.md`, text, lang: "en", lexical: 1, cosine: null, score: 0.9 });
const HITS = [
  hit(1, "A sale on the marketplace pays a 5% fee to the platform. Rewards are paid after 7 days. The total supply is 1,000,000,000 PXS."),
  hit(2, "To publish an artwork, open the editor and click “Mint”. Minting takes two blocks. Images up to 2 MB are accepted. Each account may mint up to 10,000 artworks."),
];

describe("/help: sentences taken out, numbers checked (review #3, #6)", () => {
  it("a removed sentence leaves its separator: no address is glued together", () => {
    const allowed = HITS.map((h) => `${h.url}\n${h.text}`).join("\n");
    const v = validateHelpReply({ answerable: true, answer: "Visit evil. Get 500 free PXS [1].\ncom/airdrop has the details [1].", sources: [1] }, HITS.length, allowed)!;
    const c = checkHelpAnswer(v.answer, HITS as any);
    expect(c.removed).toBe(1);
    expect(c.text).not.toMatch(/evil\.com/);
    expect(c.text).toContain("\n");
  });

  it("invented numbers are taken out wherever they are written", () => {
    for (const s of [
      "Click “Claim 500 free PXS” to start [2].", // inside a quoted label
      "Rewards are paid after 24h [1].", // a unit attached
      "Images up to 10MB are accepted [2].",
      "You can earn 5k PXS per sale [1].",
      "Minting costs 5USD [2].",
      "A sale pays a fifty percent fee [1].", // in words
      "Minting takes one block [2].",
      "Minting started in [2019] [2].", // a year in brackets is not a citation
      "A sale pays a 7% fee [1].", // 7 is days in the excerpt, not a percentage
    ]) expect(checkHelpAnswer(s, HITS as any).removed, s).toBe(1);
  });

  it("numbers as the excerpts state them, in other formats, are kept; abbreviations do not end a sentence", () => {
    for (const s of ["The total supply is 1,000,000,000 PXS [1].", "L'offre totale est de 1 000 000 000 PXS [1].", "Each account may mint up to 10000 artworks [1].", "Chaque compte peut créer jusqu'à 10 000 œuvres [2].", "Les images jusqu'à 2 Mo sont acceptées [2].", "Click “Mint.” Then wait two blocks [2]."]) expect(checkHelpAnswer(s, HITS as any).removed, s).toBe(0);
    expect(answerSegments("You pay fees, e.g. the 5% sale fee [1].")).toEqual(["You pay fees, e.g. the 5% sale fee [1]."]);
    expect(checkHelpAnswer("You pay fees, e.g. the 5% sale fee [1].", HITS as any, { strict: true }).text).toBe("You pay fees, e.g. the 5% sale fee [1].");
  });
});

describe("the evidence verifier (review #11, #15)", () => {
  it("a first sighting earlier than every recorded chain operation (a snapshot history) is no conflict", async () => {
    const env = await artCorpus(GALLERY);
    await env.DB.prepare("UPDATE post_versions SET source = 'snapshot' WHERE author = 'alice' AND permlink = 'black-cat'").run();
    for (const id of Object.values(env.ids)) await refreshImageHistory(env.DB, id);
    const r = await v4(env, "Who posted the first cat artwork?");
    expect(r.status).toBe("answered");
    expect(r.contradictions).toEqual([]);
    expect(r.answer_text).not.toContain("(E1)");
  });

  it("a chain operation showing the image before the card's first sighting is one; a single-card conflict names posts, not ids", async () => {
    const env = await artCorpus(GALLERY);
    // an earlier exact version of the black cat's bytes, in a post the cards do not know
    await env.DB.prepare("INSERT INTO post_versions (author, permlink, at, block_num, trx_id, op_in_trx, kind, body_kind, content_hash, title, source) VALUES ('zed', 'early', ?, 1, 'abc', 0, 'create', 'image', ?, 'Early', 'history')").bind(Date.UTC(2026, 7, 20) / 1000, hashOf("alice/black-cat")).run();
    const r = await v4(env, "Who posted the first cat artwork?");
    expect(r.contradictions.length).toBeGreaterThan(0);
    expect(r.contradictions[0].note).toContain("before the card's first sighting");
    const s = conflictSentence(r.contradictions[0], "en", (id) => r.cards.find((c) => c.evidence_id === id && "path" in c && c.type !== "doc")?.["path" as never]);
    expect(s).not.toMatch(/\(E\d+\) or \(E\d+\)/);
    expect(s).toContain("2026-08-20");
  });

  it("an exact history without chain operations, and an exact first sighting no version of that post showed, are reported", async () => {
    const env = await artCorpus(GALLERY);
    await env.DB.prepare("UPDATE post_versions SET source = 'snapshot' WHERE author = 'bob' AND permlink = 'lake'").run();
    await env.DB.prepare("UPDATE artworks SET history_exact = 1 WHERE post_id = ?").bind(env.ids["bob/lake"]).run();
    const r = await v4(env, "Who posted “Lake”?");
    const card = r.cards.find((c) => c.type === "artwork" && c.path === "/@bob/lake") as any;
    expect(card.issues).toContain("history marked exact without chain operations");
    expect(card.valid).toBe(true);
  });
});

describe("the nsfw filter on image questions and histories (review #12)", () => {
  const image = { sha256: hashOf("alice/black-cat"), phash: "0000000000000000", dhash: "0000000000000000", buckets: [], width: 10, height: 10, format: "png", vector: null };

  it("an image shown only by posts the filter hides: not 'gone', and no hidden post is named", async () => {
    const env = await artCorpus(GALLERY);
    await env.DB.prepare("UPDATE posts SET nsfw = 1 WHERE (author = 'alice' AND permlink = 'black-cat') OR (author = 'carol' AND permlink = 'found-cat')").run();
    const r = await v4(env, "Is this image already on Pixagram?", { image });
    expect(r.answer_text).not.toContain("not on Pixagram any more");
    expect(r.answer_text).toContain("hidden by the content filter");
    expect(JSON.stringify(r)).not.toMatch(/black-cat|found-cat|Black cat|Found this cat/);
    const o = await v4(env, "Who posted this first?", { image });
    expect(o.answer).toBeNull();
    expect(o.answer_text).not.toMatch(/@alice|@carol/);
    // with the filter off, the index's own answer
    expect((await v4(env, "Who posted this first?", { image, nsfw: "include" })).answer).toBe("alice");
  });

  it("a history does not bring a hidden post in: neither as a card, nor in the answer", async () => {
    const env = await artCorpus(GALLERY);
    await env.DB.prepare("UPDATE posts SET nsfw = 1 WHERE author = 'carol' AND permlink = 'found-cat'").run();
    const r = await v4(env, "Was “Black cat” reposted?");
    expect(r.answer).toBe(false);
    expect(r.answer_text).toContain("1 post with this image is hidden by the content filter");
    expect(JSON.stringify(r.cards)).not.toMatch(/found-cat|Found this cat/);
    expect((await v4(env, "Was “Black cat” reposted?", { nsfw: "include" })).answer).toBe(true);
  });
});

describe("the API (review #13, #14)", () => {
  it("/query: an explicit mode is not capped by the search box's ceiling (auto is)", async () => {
    const env = await artCorpus(GALLERY, { SEARCH_QUERY_MAX_MODE: "balanced" });
    const q = encodeURIComponent("Did @alice post more cats than @bob?");
    const deep = await call(env, `/query?results=0&mode=deep&q=${q}`);
    expect(deep.body.answer?.mode).toBe("deep");
    const auto = await call(env, `/query?results=0&q=${q}`);
    expect(["fast", "balanced"]).toContain(auto.body.answer?.mode);
  });

  it("POST /ask: a question alone stays under 16 KB; a larger body must carry an image; mode=v3 refuses an image before reading it", async () => {
    const env = await artCorpus(GALLERY);
    const big = await call(env, "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question: "Who posted the first cat artwork?", padding: "x".repeat(20_000) }) });
    expect(big.status).toBe(413);
    const withImage = await call(env, "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ image: "A".repeat(20_000), question: "Who posted this first?" }) });
    expect(withImage.status).not.toBe(413);
    const v3 = await call(env, "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ question: "Who posted this first?", mode: "v3", image: "iVBORw0KGgo=" }) });
    expect(v3.status).toBe(400);
    const none = await call(env, "/ask", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ image: "iVBORw0KGgo=" }) });
    expect(none).toMatchObject({ status: 400, body: { error: expect.stringContaining("question") } });
  });
});
