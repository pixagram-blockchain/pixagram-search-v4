import { describe, expect, it } from "vitest";
import { parseDates, planQuery } from "../src/search/planner";
import { mergePlans, validateLlmPlan } from "../src/search/llm-planner";

const NOW = Date.UTC(2026, 9, 4, 12) / 1000; // 2026-10-04 12:00 UTC
const AUTHORS = new Set(["laura", "matus", "wang", "retro", "matias"]);
const ask = (q: string) => planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
const search = (q: string) => planQuery(q, { mode: "search", authors: AUTHORS, now: NOW });
const day = (s: string) => Date.parse(`${s}T00:00:00Z`) / 1000;

describe("planner: /ask intents and outputs", () => {
  it("who posted the first X", () => {
    const p = ask("Who posted the first image of a cat?");
    expect(p).toMatchObject({ intent: "find_first", output: "author", residual: "cat", concepts: ["cat"], object: "artwork" });
    expect(p.temporal).toEqual({ operator: "first", field: "image" });
    expect(p.confidence).toBeGreaterThanOrEqual(0.9);
  });

  it("the same question in French and German", () => {
    for (const q of ["Qui a posté le premier chat ?", "Wer hat die erste Katze gepostet?"]) {
      const p = ask(q);
      expect(p.intent, q).toBe("find_first");
      expect(p.output, q).toBe("author");
      expect(p.concepts, q).toEqual(["cat"]);
    }
    expect(ask("Qui a posté le premier chat ?").lang).toBe("fr");
  });

  it("latest, when, how many, most liked, which artist the most", () => {
    expect(ask("Who posted the latest cat picture?")).toMatchObject({ intent: "find_last", output: "author" });
    expect(ask("When was the first butterfly posted?")).toMatchObject({ intent: "find_first", output: "date", concepts: ["butterfly"] });
    expect(ask("How many cat artworks are there?")).toMatchObject({ intent: "count", output: "count", concepts: ["cat"] });
    expect(ask("What is the most liked food artwork?")).toMatchObject({ intent: "top", sort: "votes", output: "post", concepts: ["food"] });
    // "best paid" used to be taken by "best" (most liked)
    for (const q of ["What is the best paid artwork?", "top paid cat", "the most rewarded cat", "le chat le mieux payé", "la mieux payée"]) {
      expect(ask(q).sort, q).toBe("payout");
    }
    expect(ask("top paid cat").residual).toBe("cat");
    expect(ask("Which artist posted the most travel artworks?")).toMatchObject({ intent: "compare", output: "author" });
  });

  it("a bare known author left as the subject becomes a filter; a concept name does not", () => {
    const p = ask("How many artworks did laura post?");
    expect(p.filters.authors).toEqual(["laura"]);
    expect(p.residual).toBe("");
    const r = ask("How many retro artworks are there?"); // "retro" is an author and a style
    expect(r.filters.authors).toBeUndefined();
    expect(r.residual).toBe("retro");
    expect(ask("What was the first artwork posted by wang?").filters.authors).toEqual(["wang"]);
    expect(ask("first cat @matus").filters.authors).toEqual(["matus"]);
  });

  it("colours, backgrounds, tones and orientation become filters in /ask", () => {
    const bw = ask("How many black and white artworks are there?");
    expect(bw.filters.tones).toEqual(["greyscale"]);
    expect(bw.filters.colors).toBeUndefined();
    const bg = ask("first cat on a blue background");
    expect(bg.filters.background).toEqual(["blue"]);
    expect(bg.residual).toBe("cat");
    expect(ask("vertical artworks of dogs").filters.orientation).toEqual(["portrait"]);
    expect(ask("dogs in portrait format").filters.orientation).toEqual(["portrait"]);
    // "portrait" alone is the style (a face), not the format
    expect(ask("first portrait of a dog")).toMatchObject({ concepts: expect.arrayContaining(["portrait", "dog"]) });
    expect(ask("first portrait of a dog").filters.orientation).toBeUndefined();
  });

  it("'post' is also a verb: questions about posting are about artworks, not blog posts", () => {
    expect(ask("Who was the first to post a dragon?")).toMatchObject({ object: "artwork", intent: "find_first", concepts: ["dragon"] });
    expect(ask("How many cats did laura post?")).toMatchObject({ object: "artwork", intent: "count", residual: "cats", concepts: ["cat"], filters: { authors: ["laura"] } });
    expect(ask("What is the most liked post with a cat?").object).toBe("artwork");
    expect(ask("Who wrote the first blog about cats?").object).toBe("blog");
  });

  it("similar and duplicate questions keep their intent and the id", () => {
    expect(ask("similar to 42")).toMatchObject({ intent: "similar", similarTo: { id: 42 } });
    expect(ask("duplicates of 42")).toMatchObject({ intent: "duplicate", similarTo: { id: 42 } });
  });

  it("a question it cannot parse has low confidence (the LLM planner may take over)", () => {
    const p = ask("What would a lonely astronaut paint on a rainy evening?");
    expect(p.confidence).toBeLessThan(0.7);
    // a question mark is enough to mark a question (it used to be looked for after punctuation was removed)
    expect(ask("cats with hats?").confidence).toBeLessThan(0.6);
    expect(ask("cats with hats").confidence).toBeGreaterThanOrEqual(0.9);
  });

  it("'last week', 'le mois dernier', 'letzte Woche' are periods, not 'the latest one'", () => {
    // used to plan as find_last, and answer with one artwork instead of a count
    expect(ask("how many cats last month")).toMatchObject({ intent: "count", output: "count", filters: { from: day("2026-09-01"), to: day("2026-10-01") } });
    for (const q of ["cats posted last week?", "chats postés la semaine dernière", "Katzen letzte Woche"]) {
      const p = ask(q);
      expect(p.intent, q).toBe("search");
      expect(p.filters, q).toMatchObject({ from: day("2026-09-21"), to: day("2026-09-28") });
    }
    // "last" on its own still asks for the latest one, inside the period when one is named
    expect(ask("the last cat posted last month")).toMatchObject({ intent: "find_last", residual: "cat", filters: { from: day("2026-09-01") } });
    expect(ask("cats from last year")).toMatchObject({ intent: "search", residual: "cats", filters: { from: day("2025-01-01"), to: day("2026-01-01") } });
  });

  it("removing filter and intent words never cuts longer words", () => {
    expect(ask("Who posted the first red armored knight?").residual).toBe("armored knight"); // was "armo knight"
    expect(ask("first scared red cat").residual).toBe("scared cat"); // was "sca cat"
    expect(ask("how many lime slime").residual).toBe("slime"); // was "s"
    expect(ask("the last blast").residual).toBe("blast"); // was "b"
  });
});

describe("planner: /search hints", () => {
  it("keeps the query as typed and only hints", () => {
    const p = search("blue dragon by laura");
    expect(p.intent).toBe("search");
    expect(p.text).toBe("blue dragon by laura");
    expect(p.hints.colors).toEqual(["blue"]);
    expect(p.filters.colors).toBeUndefined(); // hints, not filters
    // "by <known author>" names the author: full author feature in the ranking, but no filter
    // (in /search only @ and # are hard filters)
    expect(p.filters.authors).toBeUndefined();
    expect(p.hints.namedAuthors).toEqual(["laura"]);
    expect(search("retro cat").hints.authors).toEqual(["retro"]); // bare name: a weak hint
    expect(search("retro cat").hints.namedAuthors).toEqual([]);
    expect(search("retro cat").filters.authors).toBeUndefined();
  });

  it("ordinary prepositions are not author markers", () => {
    const ctx = { mode: "search" as const, authors: new Set(["soleil", "laura"]), now: NOW };
    const p = planQuery("coucher de soleil", ctx);
    expect(p.filters.authors).toBeUndefined();
    expect(p.hints.namedAuthors).toEqual([]);
    expect(p.concepts).toEqual(["sunset"]);
    // in a question, "de <author>" still works when the name is not a word the vocabulary knows
    const q = planQuery("Quel est le premier chat de laura ?", { ...ctx, mode: "ask" });
    expect(q.filters.authors).toEqual(["laura"]);
    expect(q.residual).toBe("chat");
    expect(planQuery("Qui a posté le premier coucher de soleil ?", { ...ctx, mode: "ask" }).filters.authors).toBeUndefined();
  });

  it("@mentions and #tags are filters", () => {
    const p = search("sunset @wang #japan");
    expect(p.filters).toMatchObject({ authors: ["wang"], tags: ["japan"] });
    expect(p.text).toBe("sunset");
  });

  it("similar to #id", () => {
    expect(search("similar to 42")).toMatchObject({ intent: "similar", similarTo: { id: 42 } });
  });
});

describe("parseDates", () => {
  it("months, days, ISO dates, ranges and relative words", () => {
    expect(parseDates("cats in september", NOW)).toMatchObject({ from: day("2026-09-01"), to: day("2026-10-01"), operator: "between" });
    expect(parseDates("chats en septembre 2025", NOW)).toMatchObject({ from: day("2025-09-01"), to: day("2025-10-01") });
    expect(parseDates("posted on 15 september", NOW)).toMatchObject({ from: day("2026-09-15"), to: day("2026-09-16") });
    expect(parseDates("between 2026-09-10 and 2026-09-12", NOW)).toMatchObject({ from: day("2026-09-10"), to: day("2026-09-13"), operator: "between" });
    expect(parseDates("before 2026-09-10", NOW)).toMatchObject({ to: day("2026-09-10"), operator: "before" });
    expect(parseDates("since september", NOW)).toMatchObject({ from: day("2026-09-01"), operator: "after" });
    expect(parseDates("after 2026-09-10", NOW)).toMatchObject({ from: day("2026-09-11"), operator: "after" }); // the day itself is excluded
    expect(parseDates("since 2026-09-10", NOW)).toMatchObject({ from: day("2026-09-10") });
    expect(parseDates("after september", NOW)).toMatchObject({ from: day("2026-10-01") });
    expect(parseDates("yesterday", NOW)).toMatchObject({ from: day("2026-10-03"), to: day("2026-10-04") });
    expect(parseDates("last year", NOW)).toMatchObject({ from: day("2025-01-01"), to: day("2026-01-01") });
    expect(parseDates("chats de l an dernier", NOW)).toMatchObject({ from: day("2025-01-01"), to: day("2026-01-01") });
    expect(parseDates("le mois passe", NOW)).toMatchObject({ from: day("2026-09-01"), to: day("2026-10-01") });
    // a month later in the year than today means last year's
    expect(parseDates("in november", NOW)).toMatchObject({ from: day("2025-11-01") });
  });

  it("month names that are ordinary words need a year or a preposition", () => {
    expect(parseDates("a pixel painting of mars", NOW)).toBeNull(); // the planet
    expect(parseDates("les sept nains", NOW)).toBeNull(); // seven dwarfs
    expect(parseDates("may the force be with you", NOW)).toBeNull();
    expect(parseDates("en mars", NOW)).toMatchObject({ from: day("2026-03-01"), to: day("2026-04-01") });
    expect(parseDates("mars 2026", NOW)).toMatchObject({ from: day("2026-03-01") });
    expect(parseDates("in may", NOW)).toMatchObject({ from: day("2026-05-01") });
    expect(parseDates("im mai", NOW)).toMatchObject({ from: day("2026-05-01") });
  });
});

describe("LLM planner validation", () => {
  it("accepts only known vocabularies and authors, and keeps rule decisions it was sure of", () => {
    const rules = ask("Who drew the first lonely astronaut eating noodles in the rain?");
    const v = validateLlmPlan(
      '```json\n{"intent":"find_first","output":"author","subject_en":"Astronaut eating noodles!","object":"artwork","authors":["@Laura","nobody"],"colors":["blue","sparkly"],"tones":["dark","weird"],"date_from":"2026-09-01","date_to":"bad","sort":"none"}\n```',
      AUTHORS,
    )!;
    expect(v).toMatchObject({ intent: "find_first", output: "author", subject: "astronaut eating noodles", authors: ["laura"], colors: ["blue"], tones: ["dark"], from: day("2026-09-01") });
    expect(v.to).toBeUndefined();
    expect(v.sort).toBeUndefined();
    const merged = mergePlans(rules, v);
    expect(merged.source).toBe("rules+llm");
    expect(merged.residual).toBe("astronaut eating noodles");
    expect(merged.filters.authors).toEqual(["laura"]);
    expect(merged.temporal).toEqual({ operator: "first", field: "image" });
    expect(validateLlmPlan('{"intent":"destroy","output":"author"}', AUTHORS)).toBeNull();
    expect(validateLlmPlan("not json", AUTHORS)).toBeNull();
  });
});
