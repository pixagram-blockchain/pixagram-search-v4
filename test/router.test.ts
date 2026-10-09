// The single search box: which destination each kind of text gets (search, ask, help).

import { describe, expect, it } from "vitest";
import { planQuery } from "../src/search/planner";
import { routeQuery, type Route } from "../src/search/router";

const NOW = Date.UTC(2026, 9, 5, 12) / 1000;
const AUTHORS = new Set(["laura", "matus", "matias", "retro", "light", "wang"]);
const TITLES = new Set(["come and fly with me", "how to train your dragon", "why i draw cats"]);

async function route(q: string, docs = 0): Promise<{ route: Route; reason: string }> {
  const plan = planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
  const d = await routeQuery(q, plan, {
    docsScore: async () => docs,
    isTitle: async () => TITLES.has(q.toLowerCase().replace(/[?!.]/g, "").trim()),
  });
  return { route: d.route, reason: d.reason };
}

const CASES: Array<[string, Route, number?]> = [
  // searches: words, names, titles (also titles that start like a question)
  ["red dragon", "search"],
  ["first snow", "search"],
  ["the last samurai", "search"],
  ["top hat", "search"],
  ["best friends", "search"],
  ["@laura cats", "search"],
  ["#japan sunset", "search"],
  ["show me red dragons", "search"],
  ["most liked dragon", "search"],
  ["Come and fly with me", "search"],
  ["How to train your dragon", "search"],
  ["Why I draw cats", "search"],
  ["chat noir", "search"],
  ["Katze", "search"],
  // questions about the artworks and their history
  ["Who posted the first cat?", "ask"],
  ["who posted the first cat", "ask"],
  ["Qui a posté le premier chat ?", "ask"],
  ["Wer hat die erste Katze gepostet?", "ask"],
  ["how many cats", "ask"],
  ["combien de chats ?", "ask"],
  ["how many artists are on pixagram?", "ask"],
  ["how many artworks were posted last week?", "ask"],
  ["who is the most active artist?", "ask"],
  ["who posted the most cats?", "ask"],
  ["qui a posté le plus de chats ?", "ask"],
  ["which artist posted the most travel artworks?", "ask"],
  ["what is the most liked artwork?", "ask"],
  ["most liked dragon?", "ask"],
  ["when was the first sunset posted?", "ask"],
  ["who posted a cat?", "ask"],
  ["what cats wear hats?", "ask"],
  ["is there a dragon with a hat?", "ask"],
  ["who posted first?", "ask"],
  ["how do I find the latest cats?", "ask"],
  ["similar to 42", "ask"],
  ["duplicates of 42", "ask"],
  // questions about the platform
  ["What is Pixagram?", "help"],
  ["How do I mint an NFT?", "help"],
  ["how do royalties work", "help"],
  ["Comment fonctionnent les royalties ?", "help"],
  ["Wie kaufe ich PXS?", "help"],
  ["What is the best way to sell my art?", "help"],
  ["When was Pixagram founded?", "help"],
  ["Who founded Pixagram?", "help"],
  ["how many PXS do I need to mint?", "help"],
  ["what are the fees?", "help"],
  ["can I delete a post?", "help"],
  ["how much does it cost to post?", "help"],
  ["how do I sell my latest artwork?", "help"],
  ["where can I see my royalties?", "help"],
  ["qu'est-ce que PXS ?", "help"],
  ["c'est quoi le staking ?", "help"],
  ["Why I draw cats?", "help"],
  ["why was my post hidden?", "help"],
  ["is pixagram free?", "help"],
  // the documentation decides the open questions
  ["what is a sprite?", "search"],
  ["what is the best time to post?", "search"],
  ["what is a sprite?", "help", 0.8],
  ["who is matias?", "search"],
  ["who is matias?", "help", 0.9],
];

describe("routing the search box", () => {
  for (const [q, want, docs] of CASES) {
    it(`${JSON.stringify(q)}${docs ? ` (documentation match ${docs})` : ""} → ${want}`, async () => {
      const r = await route(q, docs ?? 0);
      expect(r.route, r.reason).toBe(want);
    });
  }

  it("checks titles and the documentation only when a rule needs them", async () => {
    let titles = 0;
    let docs = 0;
    const checks = {
      docsScore: async () => (docs++, 0),
      isTitle: async () => (titles++, false),
    };
    const plan = (q: string) => planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
    await routeQuery("red dragon", plan("red dragon"), checks);
    await routeQuery("Who posted the first cat?", plan("Who posted the first cat?"), checks);
    expect([titles, docs]).toEqual([0, 0]);
    await routeQuery("who posted the first cat", plan("who posted the first cat"), checks);
    expect([titles, docs]).toEqual([1, 0]);
    await routeQuery("what is a sprite", plan("what is a sprite"), checks);
    expect([titles, docs]).toEqual([2, 1]);
  });
});

describe("planner additions for the platform's questions", () => {
  const ask = (q: string) => planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
  it("how many artists counts people; 'on Pixagram' is not a subject", () => {
    expect(ask("how many artists are on pixagram?")).toMatchObject({ intent: "count", countOf: "authors", residual: "" });
    expect(ask("combien d'artistes ont posté des chats ?")).toMatchObject({ intent: "count", countOf: "authors", concepts: ["cat"] });
    expect(ask("wie viele Künstler gibt es?").countOf).toBe("authors");
    expect(ask("how many cats are on pixagram?")).toMatchObject({ intent: "count", residual: "cats" });
    expect(ask("how many cats are on pixagram?").countOf).toBeUndefined();
  });
  it("who posted the most …, the most active artist: which author", () => {
    expect(ask("who posted the most cats?")).toMatchObject({ intent: "compare", output: "author", concepts: ["cat"] });
    expect(ask("who is the most active artist?")).toMatchObject({ intent: "compare", output: "author", residual: "" });
    expect(ask("wer hat die meisten Katzen gepostet?")).toMatchObject({ intent: "compare", concepts: ["cat"] });
    expect(ask("quel est l'artiste le plus actif ?").intent).toBe("compare");
  });
  it("intent words are kept: an intent inferred from 'who' has none", () => {
    expect(ask("who posted the first cat?").intentWords).toEqual(["first"]);
    expect(ask("who posted a cat?")).toMatchObject({ intent: "find_first", intentWords: [] });
  });
});
