// mode=v3 and /search answer exactly as v3 did (spec §56): v4's askV3 and search() against v3's
// own answers, frozen from the last v3 commit on the same test gallery (scripts/v3_fixture.sh →
// test/fixtures/v3-answers.json). /search may only add v4's fields (retrieval, reranked).

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { artCorpus, GALLERY } from "./harness/corpus";
import { askV3 } from "../src/search/ask";
import { search } from "../src/search/service";
import { parseSearchRequest } from "../src/search/params";
import { ask, type AskResponseV4 } from "../src/search/ask";

const fixture = JSON.parse(readFileSync(new URL("./fixtures/v3-answers.json", import.meta.url), "utf8")) as {
  commit: string;
  ask: Array<{ q: string; response: Record<string, unknown> }>;
  search: Array<{ qs: string; response: Record<string, unknown> }>;
};
/** eval/offline/v3-fixture.test.ts runs v3 at this time */
const FIXTURE_NOW = "2026-10-08T12:00:00Z";
const V4_ONLY_SEARCH_FIELDS = ["retrieval", "reranked", "reranker"];
/** a random id per search, on both sides */
const RANDOM = ["query_id"];

const strip = (o: Record<string, unknown>, drop: string[] = []) => {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(o)) if (k !== "took_ms" && !drop.includes(k)) out[k] = v;
  return JSON.parse(JSON.stringify(out));
};

describe(`mode=v3 and /search against v3's own answers (${fixture.commit})`, () => {
  beforeAll(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date(FIXTURE_NOW));
  });
  afterAll(() => {
    vi.useRealTimers();
  });

  it("/ask mode=v3: the same response, field for field", async () => {
    const env = await artCorpus(GALLERY);
    expect(fixture.ask.length).toBeGreaterThanOrEqual(30);
    for (const { q, response } of fixture.ask) expect(strip((await askV3(env, { question: q })) as unknown as Record<string, unknown>), q).toEqual(response);
    // through v4's entry point too
    const via = (await ask(env, { question: fixture.ask[0].q, mode: "v3" })) as unknown as Record<string, unknown>;
    expect(strip(via, ["mode"])).toEqual(fixture.ask[0].response);
  });

  it("/search: the same items, scores and fields (v4 only adds its own)", async () => {
    for (const { qs, response } of fixture.search) {
      const env = await artCorpus(GALLERY);
      const r = (await search(env, parseSearchRequest(new URLSearchParams(qs)))) as unknown as Record<string, unknown>;
      expect(strip(r, [...V4_ONLY_SEARCH_FIELDS, ...RANDOM]), qs).toEqual(strip(response, RANDOM));
    }
  });

  it("v4's own /ask still asks back where v3 answered about every artwork", async () => {
    const env = await artCorpus(GALLERY);
    for (const q of ["Qui a posté ça ?", "¿Quién publicó esto?", "Chi ha pubblicato questo?", "Who posted this?"]) {
      expect(((await ask(env, { question: q })) as AskResponseV4).status, q).toBe("clarify");
    }
  });
});
