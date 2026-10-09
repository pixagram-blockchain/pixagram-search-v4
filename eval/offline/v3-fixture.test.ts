// Freezes v3's answers on the test gallery (test/harness/corpus.ts) into test/fixtures/v3-answers.json,
// for the regression test that keeps mode=v3 and /search exactly v3's (test/v4-v3-parity.test.ts).
// Runs v3's own code from a checkout of the last v3 commit:
//
//   scripts/v3_fixture.sh            (git worktree of 16061e6 → V3_TREE=… npm run eval:v3-fixture)
//
// The questions avoid relative dates ("last week"): their answers depend on the day they run.

import { describe, it, vi } from "vitest";
import { writeFileSync } from "node:fs";
import { artCorpus, GALLERY } from "../../test/harness/corpus";

export const V3_COMMIT = "16061e6";
/** The clock both sides run at: freshness and "now" in the answers are the same whenever the test runs. */
export const FIXTURE_NOW = "2026-10-08T12:00:00Z";

export const ASK_QUESTIONS = [
  "Who posted the first cat artwork?",
  "Qui a posté le premier chat ?",
  "Wer hat die erste Katze gepostet?",
  "¿Quién publicó la primera obra de gato?",
  "Chi ha pubblicato la prima opera con un gatto?",
  "How many cat artworks?",
  "How many artworks did @alice publish?",
  "Who posted the most cats?",
  "What is the latest dragon?",
  "When was the first dragon posted?",
  "most liked cat artwork",
  "Qui a posté ça ?",
  "Who posted this?",
  "¿Quién publicó esto?",
  "Chi ha pubblicato questo?",
  "Wer hat das gepostet?",
  "similar to 3",
  "What is the first artwork from @alice?",
  "Who posted the first lake artwork?",
  "How many dragons were posted in September 2026?",
  "Show me cats",
  "¿Cuántos gatos publicó @alice?",
  "Who posted the first dragon?",
  "Combien de dragons ?",
  "¿Quién publicó el primer dragón?",
  "¡Muéstrame gatos!",
  "Quanti draghi ci sono?",
  "Wie viele Drachen gibt es?",
  "Which artist has posted the most artworks?",
  "Who posted more cats, @alice or @bob?",
  "What is the most voted artwork?",
  "duplicates of 1",
  "Who posted “Lake”?",
  "Was “Swan” edited?",
  "When was the latest cat posted?",
  "Show me red dragons",
  "Combien d'œuvres @alice a-t-elle publiées ?",
  "Est-ce que @alice a posté plus de chats que @bob ?",
];

export const SEARCH_QUERIES = ["q=cat", "q=dragon", "q=cat&sort=newest", "q=lake&author=bob", "q=red dragon", "q=&sort=votes", "q=chat", "q=¿gato?", "q=black cat&explain=1", "q=katze", "q=swan lake", "q=cat&limit=2"];

const strip = (o: Record<string, unknown>) => {
  const { took_ms: _t, ...rest } = o;
  return rest;
};

describe("v3 answers on the test gallery", () => {
  it.skipIf(!process.env.V3_TREE)("frozen into test/fixtures/v3-answers.json", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date(FIXTURE_NOW));
    const tree = process.env.V3_TREE!;
    const { ask } = await import(/* @vite-ignore */ `${tree}/src/search/ask.ts`);
    const { search } = await import(/* @vite-ignore */ `${tree}/src/search/service.ts`);
    const { parseSearchRequest } = await import(/* @vite-ignore */ `${tree}/src/search/params.ts`);
    const out: { commit: string; corpus: string; ask: Array<{ q: string; response: unknown }>; search: Array<{ qs: string; response: unknown }> } = { commit: V3_COMMIT, corpus: "GALLERY", ask: [], search: [] };
    const env = await artCorpus(GALLERY);
    for (const q of ASK_QUESTIONS) out.ask.push({ q, response: strip(await ask(env, { question: q })) });
    for (const qs of SEARCH_QUERIES) {
      const fresh = await artCorpus(GALLERY);
      const r = await search(fresh, parseSearchRequest(new URLSearchParams(qs)));
      out.search.push({ qs, response: strip(r) });
    }
    writeFileSync(new URL("../../test/fixtures/v3-answers.json", import.meta.url), `${JSON.stringify(out, null, 1)}\n`);
  });
});
