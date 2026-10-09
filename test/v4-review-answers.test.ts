// Regressions of the independent review of v4 (answer layer): a reasoning model's text never
// replaces the index's answer, and is shown after it only when it states the result's own values
// (spec §27); names without "@", numbers of the wrong kind, numbers and dates written in other
// formats, yes/no in five languages; the confidence of a deterministic answer; status, text and
// evidence that agree; the rationale of a rejected reply.

import { describe, expect, it } from "vitest";
import type { TestEnv } from "./harness/fakes";
import { artCorpus, GALLERY } from "./harness/corpus";
import { ask, type AskResponseV4 } from "../src/search/ask";
import { agreesWithResult, extractAtoms, verifyClaims } from "../src/search/claims";
import type { ArtworkCard, EvidenceCard, ResultCard } from "../src/search/evidence";

const v4 = async (env: TestEnv, question: string, extra: Record<string, unknown> = {}) => (await ask(env, { question, noCache: true, ...extra })) as AskResponseV4;
const reply = (env: TestEnv, r: Record<string, unknown>) => {
  const calls: string[] = [];
  env._ai.handler = (model) => (calls.push(model), { response: r, usage: { prompt_tokens: 10, completion_tokens: 10 } });
  return calls;
};

describe("a model's text never replaces the index's answer (review #1, #7)", () => {
  const wrong: Array<[string, string, Record<string, unknown>]> = [
    ["When was the first cat artwork posted?", "2026-09-01", { answer: "The first cat artwork was posted on 2026-09-03." }],
    ["What is the first cat artwork?", "/@alice/black-cat", { answer: "The first cat artwork is “Cat nap”." }],
    ["Did @alice post more cats than @bob?", "true", { answer: "@bob posted more cat artworks than @alice." }],
    ["Was “Swan” posted before “Lake”?", "true", { answer: "“Lake” (2026-09-12) came before “Swan” (2026-09-10)." }],
    ["Who posted “Lake”?", "bob", { answer: "“Lake” was posted by @alice [R1]." }],
    ["Who posted the first cat artwork?", "alice", { answer: "The first cat artwork was posted by Mallory." }],
    ["Who posted the first cat artwork?", "alice", { answer: "@bob posted the first cat artwork; @alice only reposted it later." }],
    ["How many cat artworks did @alice post?", "2", { answer: "One cat artwork by @alice was found." }],
    ["How many cat artworks did @alice post?", "2", { answer: "@alice a publié 3 œuvres de chat." }],
  ];
  for (const [q, value, r] of wrong) {
    it(`${q} → ${String(r.answer)}`, async () => {
      const env = await artCorpus(GALLERY);
      reply(env, { status: "answered", claims: [], rationale: "R1.", ...r });
      const res = await v4(env, q, { mode: "deep" });
      expect(String(res.answer)).toBe(value);
      expect(res.answer_text).toBe(res.result_text);
      expect(res.explanation).toBeUndefined();
      expect(res.rationale).toBeUndefined();
      expect(res.notes.join(" ")).toContain("the model's explanation was not used");
      // never surer than the operator, whatever the model said
      expect(res.confidence).toBeLessThanOrEqual(res.confidence_parts.operator);
    });
  }

  it("“…did they later repost it?” when the index says no: an answer without its “No” is not shown", async () => {
    const env = await artCorpus(GALLERY);
    reply(env, { status: "answered", answer: "@alice posted it again on 2026-09-25 in /@carol/found-cat.", claims: [], rationale: "R2." });
    const r = await v4(env, "Who posted the first cat artwork and did they later repost it?", { mode: "deep" });
    expect(r.answer).toBe(false);
    expect(r.explanation).toBeUndefined();
    expect(r.notes.join(" ")).toContain("does not say No");
  });

  it("an explanation that states the result is shown after it, the confidence still the operator's", async () => {
    const env = await artCorpus(GALLERY);
    reply(env, { status: "answered", answer: "@alice posted the first cat artwork, “Black cat”, on 2026-09-01 [R1]; @carol later posted the same image [E9].", claims: [], rationale: "R1 is the index's first.", confidence: 0.99 });
    const r = await v4(env, "Who posted the first cat artwork?", { mode: "deep" });
    expect(r.explanation).toContain("@alice posted the first cat artwork");
    expect(r.answer_text.startsWith(r.result_text!)).toBe(true);
    expect(r.answer_text).toContain(r.explanation!);
    expect(r.rationale).toBe("R1 is the index's first.");
    expect(r.confidence).toBeLessThanOrEqual(r.confidence_parts.operator);
  });

  it("no match: a model claiming one is not asked, and not shown", async () => {
    const env = await artCorpus(GALLERY);
    const calls = reply(env, { status: "answered", answer: "@bob posted a unicorn artwork, “Red dragon”, on 2026-09-15.", claims: [], rationale: "E7." });
    const r = await v4(env, "Show me unicorn artworks", { mode: "deep" });
    expect(r.status).toBe("no_match");
    expect(r.answer_text).not.toContain("Red dragon");
    expect(calls).toEqual([]);
  });

  it("a model's “conflict” without a conflict the verifier found does not change the status", async () => {
    const env = await artCorpus(GALLERY);
    reply(env, { status: "conflict", answer: "@alice posted the first cat artwork on 2026-09-01, but the evidence conflicts.", claims: [], rationale: "R1." });
    const r = await v4(env, "Who posted the first cat artwork?", { mode: "deep" });
    expect(r.status).toBe("answered");
    expect(r.contradictions).toEqual([]);
    expect(r.notes.join(" ")).toContain("conflict the evidence verifier did not find");
  });
});

describe("status, answer, text and evidence agree (review #8, #9)", () => {
  it("asked back: no result, evidence, items or steps that answer about every artwork", async () => {
    const env = await artCorpus(GALLERY);
    const r = await v4(env, "Who posted this?");
    expect(r).toMatchObject({ status: "clarify", answer: null, answer_type: "none", evidence: [], items: [], steps: [], cards: [] });
    expect(r.result_text).toBeUndefined();
    expect(r.answer_text).toContain("Which artwork do you mean?");
  });

  it("deciding evidence that failed verification: the model is not asked and nothing is presented as the answer", async () => {
    const env = await artCorpus(GALLERY);
    await env.DB.prepare("UPDATE posts SET created = 1000 WHERE id = ?").bind(env.ids["alice/black-cat"]).run();
    const calls = reply(env, { status: "answered", answer: "@bob posted “Cat in a hat” on 2026-09-03.", claims: [], rationale: "E2." });
    const r = await v4(env, "Who posted the first cat artwork?", { mode: "deep" });
    expect(r).toMatchObject({ status: "insufficient_evidence", answer: null, evidence: [], items: [] });
    expect(r.answer_text).toBe("There is insufficient evidence to determine this.");
    expect(r.result_text).toBeUndefined();
    expect(r.cards.some((c) => c.type === "result")).toBe(false);
    expect(calls).toEqual([]);
  });

  it("a failed step: insufficient evidence, without a model text", async () => {
    const env = await artCorpus(GALLERY);
    const calls = reply(env, { status: "answered", answer: "@alice posted “Black cat” on 2026-09-01.", claims: [], rationale: "E1." });
    const r = await v4(env, "Who posted the first unicorn artwork and was it later reposted?", { mode: "deep" });
    expect(r.status).toBe("insufficient_evidence");
    expect(r.answer_text).not.toContain("Black cat");
    expect(calls).toEqual([]);
  });

  it("the rationale of a rejected answer is not returned; in expert mode an unsupported rationale neither", async () => {
    const env = await artCorpus(GALLERY);
    reply(env, { status: "answered", answer: "@mallory posted the first cat artwork.", claims: [], rationale: "The evidence shows that the first cat artwork was posted by Mallory." });
    expect((await v4(env, "Who posted the first cat artwork?", { mode: "deep" })).rationale).toBeUndefined();
    reply(env, { status: "answered", answer: "@alice posted the first cat artwork [R1].", claims: [], rationale: "The artist was clearly inspired by medieval tapestries, as R1 shows." });
    const x = await v4(env, "Who posted the first cat artwork?", { mode: "expert", admin: true });
    expect(x.explanation).toBe("@alice posted the first cat artwork [R1].");
    expect(x.rationale).toBeUndefined();
  });

  it("a synthesis in another language than the evidence's is read through the concepts", async () => {
    const env = await artCorpus(GALLERY);
    reply(env, { status: "answered", answer: "Elle dessine surtout des chats et des dragons.", claims: [], rationale: "R1." });
    const fr = await v4(env, "Quel genre d'art fait @alice ?", { mode: "deep" });
    expect(fr.explanation).toBe("Elle dessine surtout des chats et des dragons.");
    reply(env, { status: "answered", answer: "Elle dessine surtout des licornes et des robots.", claims: [], rationale: "R1." });
    expect((await v4(env, "Quel genre d'art fait @alice ?", { mode: "deep" })).explanation).toBeUndefined();
  });
});

describe("claim atoms and facts (review #2, #4, #5)", () => {
  const E1: ArtworkCard = { evidence_id: "E12", type: "artwork", source: "pixagram-index", artwork_id: 12, path: "/@alice/cat-1", author: "alice", title: "The King of the Cats", created_at: "2026-09-01T00:00:00Z", first_seen_at: "2026-09-01T00:00:03Z", first_seen_in: "/@alice/cat-1", tags: ["cat"], votes: 3, payout: 14.469 };
  const E2: ArtworkCard = { ...E1, evidence_id: "E30", artwork_id: 30, path: "/@bob/my-cat", author: "bob", title: "My cat", created_at: "2026-09-21T00:00:00Z", votes: 1, payout: 0 };
  const R1: ResultCard = { evidence_id: "R1", type: "result", source: "operator", step: "q1", op: "find_first", question: "who posted the first cat", answer: "alice", answer_type: "author", text: "The first cat artwork was posted by @alice on 2026-09-01: “The King of the Cats”.", n: 2, complete: true, exact: false, details: { first_author: "alice", time: "2026-09-01T00:00:03Z", post: "/@alice/cat-1", title: "The King of the Cats" } };
  const R2: ResultCard = { evidence_id: "R2", type: "result", source: "operator", step: "q2", op: "history", answer: false, answer_type: "boolean", text: "No: the image was not posted again.", complete: true, exact: true };
  const R3: ResultCard = { evidence_id: "R3", type: "result", source: "operator", step: "q1", op: "count", question: "how many cat artworks", answer: 2, answer_type: "count", text: "2 cat artworks found.", complete: true, exact: false };
  const authors = new Set(["alice", "bob"]);
  const status = (answer: string, cards: EvidenceCard[]) => verifyClaims({ answer, claims: [] }, cards, { authors }).answer.status;

  it("names without “@”: after “by”, or posting something, are accounts the evidence must hold; “the first” names its own clause's", () => {
    expect(status("The first cat artwork was posted by Mallory.", [R1, E1, E2])).toBe("contradicted");
    expect(status("Mallory posted a cat artwork on 2026-09-01.", [E1, E2])).toBe("unsupported");
    expect(status("@bob posted the first cat artwork; @alice only reposted it later.", [R1, E1, E2])).toBe("contradicted");
    expect(status("The first cat artwork was posted by @alice; @bob posted his on 2026-09-21.", [R1, E1, E2])).toBe("supported");
    // not names: "inspired by medieval tapestries", "by far"
    expect(extractAtoms("It was inspired by medieval tapestries, by far the best.").filter((a) => a.kind === "account")).toEqual([]);
  });

  it("numbers by kind: an id, a vote count or a date's digits do not support a count; votes belong to their post", () => {
    // E12 is artwork 12; 3 is its votes, not a count of artworks
    expect(status("@alice a publié 3 œuvres de chat.", [R3, E1, E2])).toBe("contradicted");
    expect(status("@alice posted 12 cat artworks.", [E1, E2])).toBe("unsupported");
    expect(status("@alice's cat has 1 vote.", [E1, E2])).toBe("unsupported");
    expect(status("@alice's cat has 3 votes.", [E1, E2])).toBe("supported");
    expect(status("“My cat” earned 14.47 PXS.", [E1, E2])).toBe("unsupported");
    expect(status("“The King of the Cats” earned 14,47 PXS.", [E1, E2])).toBe("supported");
  });

  it("numbers and dates as people write them", () => {
    const kinds = (t: string) => extractAtoms(t).map((a) => `${a.kind}:${a.value}`);
    expect(kinds("1,000,000,000 PXS")).toEqual(["number:1000000000"]);
    expect(kinds("1 234 œuvres")).toEqual(["number:1234"]);
    expect(kinds("1 234 œuvres")).toEqual(["number:1234"]);
    expect(kinds("1'234 Bilder")).toEqual(["number:1234"]);
    expect(kinds("am 01.09.2026")).toContain("date:2026-09-01");
    expect(kinds("on the 21st of September 2026")).toContain("date:2026-09-21");
    expect(kinds("il 21° settembre 2026")).toContain("date:2026-09-21");
    expect(kinds("2 œuvres en 2026.")).toEqual(["number:2", "year:2026"]);
    expect(kinds("twenty-five votes")).toEqual(["number:25"]);
    expect(kinds("two hundred PXS")).toEqual(["number:200"]);
    expect(status("@alice a publié 2 œuvres de chat en 2026.", [R3, E1, E2])).toBe("supported");
  });

  it("yes / no in five languages; “No one …” and “Si …” (if) are not answers", () => {
    const yn = (t: string) => extractAtoms(t).filter((a) => a.kind === "yesno").map((a) => a.value);
    expect(yn("Sí, se publicó de nuevo.")).toEqual(["yes"]);
    expect(yn("Sì: è stata ripubblicata.")).toEqual(["yes"]);
    expect(yn("No se publicó de nuevo.")).toEqual(["no"]);
    expect(yn("Non è stata ripubblicata.")).toEqual(["no"]);
    expect(yn("No one posted a cat before @alice.")).toEqual([]);
    expect(yn("Si l'œuvre existe, elle est là.")).toEqual([]);
    expect(status("Sí: se publicó de nuevo el 2026-09-21 en /@bob/my-cat.", [R1, R2, E1, E2])).toBe("contradicted");
  });

  it("agreement: the result's own value, first", () => {
    expect(agreesWithResult("@alice posted it first.", [R1], [R1, E1, E2]).ok).toBe(true);
    expect(agreesWithResult("The first one was posted by @bob, then by @alice.", [R1], [R1, E1, E2]).ok).toBe(false);
    expect(agreesWithResult("No, it was not posted again.", [R2], [R2, E1]).ok).toBe(true);
    expect(agreesWithResult("It was posted again on 2026-09-21.", [R2], [R2, E1]).ok).toBe(false);
    expect(agreesWithResult("@alice posted 2 of them.", [R3], [R3, E1]).ok).toBe(true);
    expect(agreesWithResult("@alice posted three of them.", [R3], [R3, E1]).ok).toBe(false);
  });
});
