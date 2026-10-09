// GET /suggest: completions, questions, titles, documentation sections, popular searches and
// examples for the search box, from a small corpus.

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { FakeExec, FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { upsertPost } from "../src/db/posts";
import { app } from "../src/api";
import { routeOf, suggestExamples, suggestFor, suggestText } from "../src/search/suggest";
import { ask } from "../src/search/ask";
import { addressKey, recordSearcher } from "../src/search/suggest";
import { pruneLogs } from "../src/search/feedback";
import { syncDocs } from "../src/docs/sync";

const T0 = Date.UTC(2026, 8, 1) / 1000;
const NOW = Math.floor(Date.now() / 1000);

afterEach(() => vi.unstubAllGlobals());

/** nsfw: "author" (the post's flag) or "ai" (the AI estimate): hidden by /search by default */
type Row = [author: string, permlink: string, title: string, tags: string[], concepts: string[], color: string | null, votes: number, nsfw?: "author" | "ai"];

async function corpus(rows: Row[], opts: { docs?: Record<string, string>; popular?: boolean } = {}): Promise<TestEnv> {
  const env = makeEnv({ PLANNER_BACKEND: "rules", DOCS_REPO: "pixa/info", ...(opts.popular ? { SUGGEST_POPULAR: "on" } : {}) });
  let i = 0;
  for (const [author, permlink, title, tags, concepts, color, votes, nsfw] of rows) {
    const created = T0 + i++ * 3600;
    const { id } = await upsertPost(
      env,
      { author, permlink, type: "artwork", title, description: "", body: "", bodyLength: 0, category: "pixagram", tags, app: "pixagram/3", nsfw: nsfw === "author", aiTraining: null, licenseJson: null, royaltyPct: null, created, updated: created, deleted: false, netVotes: votes, payout: 0, children: 0, image: null },
      null,
    );
    await env.DB.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, primary_color, r2_orig_key, ai_nsfw, updated) VALUES (?, ?, 'image/png', 10, ?, ?, ?, ?)")
      .bind(id, `h${id}`, color, `orig/h${id}.png`, nsfw === "ai" ? 0.95 : 0.01, created)
      .run();
    for (const c of concepts) await env.DB.prepare("INSERT INTO artwork_concepts (post_id, concept, confidence, source) VALUES (?, ?, 0.9, 'tag')").bind(id, c).run();
  }
  if (opts.docs) {
    (env as any).VEC_DOCS = new FakeVectorize([]);
    installGitHub({ owner: "pixa", repo: "info", head: "5".repeat(40), files: opts.docs, calls: [] });
    env._ai.handler = (model, input) => (model === "@cf/baai/bge-m3" ? { data: input.text.map(() => [1, 0, 0]) } : { response: {} });
    await syncDocs(env);
  }
  return env;
}

const ROWS: Row[] = [
  ["alice", "red-dragon", "Red Dragon", ["dragon"], ["dragon"], "red", 9],
  ["bob", "dragon-two", "Dragon Two", ["dragon"], ["dragon"], "red", 3],
  ["carol", "green-dragon", "Green Dragon", ["dragon"], ["dragon"], "green", 2],
  ["alice", "cat-nap", "Cat nap", ["cat"], ["cat"], "orange", 5],
  ["bob", "black-cat", "Black cat", ["cat"], ["cat"], "black", 4],
  ["carol", "cat-king", "Cat king", ["cat"], ["cat"], "black", 1],
  ["dave", "fox-snow", "Fox in the snow", ["fox"], ["fox"], "white", 1],
  ["dave", "fly", "Come and fly with me", ["sky"], [], "sky", 7],
  // what /search hides by default never shows in a suggestion: not their titles, not their concepts
  ["eve", "naughty-dragon", "Naughty Dragon", ["dragon"], ["dragon"], "red", 99, "author"],
  ["eve", "catwalk", "Catwalk", ["cat"], ["cat"], "pink", 50, "ai"],
  ["eve", "lingerie", "Lingerie study", ["lingerie"], ["fox"], "pink", 60, "author"],
];

const texts = (r: { suggestions: Array<{ text: string }> }) => r.suggestions.map((s) => s.text);

/** Rows of query_people: `people` clients over `days` days (the days just before today), spread in turn. */
async function searchedBy(env: TestEnv, q: string, people: number, days: number) {
  const today = Math.floor(NOW / 86400);
  for (let k = 0; k < people; k++) {
    await env.DB.prepare("INSERT OR IGNORE INTO query_people (qn, q, client, day) VALUES (?, ?, ?, ?)")
      .bind(q.toLowerCase(), q.toLowerCase(), `c${k}`, today - (k % days))
      .run();
  }
}

describe("suggestions while typing", () => {
  let env: TestEnv;
  beforeAll(async () => {
    env = await corpus(ROWS, { popular: true });
    await searchedBy(env, "red dragon", 5, 2);
    await searchedBy(env, "rare", 2, 2); // two people: never suggested
    await searchedBy(env, "red cat", 9, 1); // all on one day: never suggested
    await searchedBy(env, "red pills cheap", 9, 3); // words the index does not have: never suggested
    await searchedBy(env, "red horror", 9, 3); // a concept no safe artwork shows: never suggested
    await searchedBy(env, "red 0791234567", 9, 3); // a number: never suggested
  });

  it("completes the word being typed from the corpus, as typed, with questions about it", async () => {
    const r = await suggestFor(env, "dra", { lang: "en" });
    expect(texts(r)).toEqual(expect.arrayContaining(["dragon", "who posted the first dragon?", "how many dragon artworks?"]));
    // three dragons: the fourth is NSFW
    expect(r.suggestions.find((s) => s.text === "dragon")).toMatchObject({ kind: "complete", route: "search", n: 3 });
    expect(r.completion).toBe("dragon");
    const caps = await suggestFor(env, "Dra", { lang: "en" });
    expect(caps.completion).toBe("Dragon");
  });

  it("completes a question being typed, and proposes whole questions while an opener is typed", async () => {
    const q = await suggestFor(env, "who posted the first dra", { lang: "en" });
    expect(q.suggestions[0]).toMatchObject({ text: "who posted the first dragon?", kind: "question", route: "ask" });
    expect(q.completion).toBe("who posted the first dragon?");
    const opener = await suggestFor(env, "who po", { lang: "en" });
    expect(texts(opener).slice(0, 2)).toEqual(["who posted the first cat?", "who posted the first dragon?"]);
    expect(opener.completion).toBe("who posted the first cat?");
    // the subject reads in the plural where the question wants it
    const fr = await suggestFor(env, "combien de ch", { lang: "fr" });
    expect(texts(fr)[0]).toBe("combien de chats ?");
    const de = await suggestFor(env, "wie viele kat", { lang: "de" });
    expect(texts(de)[0]).toBe("wie viele Katzen?");
    expect(de.completion).toBe("wie viele katzen?"); // the letters typed stay as typed
  });

  it("French templates for the reader's language; none for languages without templates", async () => {
    const fr = await suggestFor(env, "chat", { lang: "fr" });
    expect(texts(fr)).toEqual(expect.arrayContaining(["qui a posté la première œuvre de chat ?", "combien d'œuvres de chat ?"]));
    const it_ = await suggestFor(env, "gatt", { lang: "it" });
    expect(it_.suggestions.filter((s) => s.kind === "question")).toEqual([]);
  });

  it("titles, popular searches (three people on a day, two days, known words, nothing private), and the finished word", async () => {
    const t = await suggestFor(env, "come and f", { lang: "en" });
    expect(t.suggestions[0]).toMatchObject({ text: "Come and fly with me", kind: "title", post: { author: "dave", permlink: "fly", type: "artwork", image: expect.stringMatching(/^\/img\/orig\/h\d+\.png$/) } });
    // the ghost keeps the letters as typed
    expect(t.completion).toBe("come and fly with me");
    const p = await suggestFor(env, "red ", { lang: "en" });
    // n: the most people on one day (three of the five ran it today)
    expect(p.suggestions.filter((s) => s.kind === "popular")).toEqual([expect.objectContaining({ text: "red dragon", n: 3 })]);
    expect(p.completion).toBe("red dragon");
    for (const q of ["rar", "red 07", "red c", "red p", "red h"]) expect((await suggestFor(env, q, { lang: "en" })).suggestions.filter((s) => s.kind === "popular"), q).toEqual([]);
    // the text typed is not suggested back
    expect((await suggestFor(env, "red dragon", { lang: "en" })).suggestions.map((s) => s.text)).not.toContain("red dragon");
  });

  it("a word of one or two letters after others still narrows the titles", async () => {
    expect((await suggestFor(env, "cat ki", { lang: "en" })).suggestions.filter((s) => s.kind === "title").map((s) => s.text)).toEqual(["Cat king"]);
    expect((await suggestFor(env, "cat n", { lang: "en" })).suggestions.filter((s) => s.kind === "title").map((s) => s.text)).toEqual(["Cat nap"]);
    expect((await suggestFor(env, "cat zz", { lang: "en" })).suggestions.filter((s) => s.kind === "title")).toEqual([]);
  });

  it("nothing from what /search hides by default", async () => {
    for (const q of ["naug", "catw", "linge", "naughty d", "Naughty"]) {
      expect(texts(await suggestFor(env, q, { lang: "en" })).filter((x) => /naughty|catwalk|lingerie/i.test(x)), q).toEqual([]);
    }
    // the fox concept's only safe artwork counts; the NSFW one does not
    expect((await suggestFor(env, "fo", { lang: "en" })).suggestions.find((s) => s.text === "fox")).toMatchObject({ n: 1 });
  });

  it("a typo gets a correction when nothing else matches", async () => {
    const r = await suggestFor(env, "dragno", { lang: "en" });
    expect(r.suggestions).toContainEqual(expect.objectContaining({ text: "dragon", kind: "correction" }));
  });

  it("odd input: control characters, a leading sign, a long text", async () => {
    expect(suggestText("\u0000 cat  ")).toBe("cat ");
    expect((await suggestFor(env, "#dra", { lang: "en" })).suggestions.filter((s) => s.kind === "complete")).toEqual([]);
    expect((await suggestFor(env, "x".repeat(500), { lang: "en" })).q.length).toBe(100);
    expect(await suggestFor(env, "   ", { lang: "en" })).toMatchObject({ suggestions: [], completion: null });
  });
});

describe("popular searches: who ran what", () => {
  const search = (env: TestEnv, q: string, ip: string, extra = "") =>
    app.fetch(new Request(`https://search.test/search?q=${encodeURIComponent(q)}${extra}`, { headers: { "cf-connecting-ip": ip } }), env, new FakeExec() as unknown as ExecutionContext);
  const people = async (env: TestEnv) => (await env.DB.prepare("SELECT qn, q, client, day FROM query_people ORDER BY client").all<any>()).results ?? [];

  it("one row per search, person and day; a burst of one person counts once", async () => {
    const env = await corpus(ROWS, { popular: true });
    const exec = new FakeExec();
    for (const extra of ["&type=artwork", "&type=blog", ""]) {
      await app.fetch(new Request(`https://search.test/search?q=Red%20Dragon${extra}`, { headers: { "cf-connecting-ip": "203.0.113.7" } }), env, exec as unknown as ExecutionContext);
    }
    await exec.settle();
    const rows = await people(env);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ qn: "red dragon", q: "red dragon", client: expect.stringMatching(/^[0-9a-f]{6}$/), day: Math.floor(Date.now() / 1000 / 86400) });
    // never the address itself
    expect(JSON.stringify(rows)).not.toContain("203.0.113.7");
  });

  it("people are told apart within a day only: one address on three days is one person, three on a day are three", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    try {
      const env = await corpus(ROWS, { popular: true });
      const day0 = Date.UTC(2026, 9, 1, 12);
      for (let d = 0; d < 3; d++) {
        vi.setSystemTime(day0 + d * 86400_000);
        await recordSearcher(env, "bob black cat", "203.0.113.7");
      }
      vi.setSystemTime(day0 + 3 * 86400_000);
      expect((await suggestFor(env, "bob", { lang: "en" })).suggestions.filter((s) => s.kind === "popular")).toEqual([]);
      // three people on one day, and one of them again the next day
      const env2 = await corpus(ROWS, { popular: true });
      vi.setSystemTime(day0);
      for (const ip of ["198.51.100.1", "198.51.100.2", "2001:db8:5:6:1::1"]) await recordSearcher(env2, "black cat", ip);
      // the same household on IPv6: its /64
      await recordSearcher(env2, "black cat", "2001:db8:5:6:ffff::9");
      vi.setSystemTime(day0 + 86400_000);
      await recordSearcher(env2, "black cat", "198.51.100.1");
      const daily = (await env2.DB.prepare("SELECT day, COUNT(DISTINCT client) AS n FROM query_people GROUP BY day ORDER BY day").all<any>()).results;
      expect(daily.map((r: any) => r.n)).toEqual([3, 1]);
      expect((await suggestFor(env2, "bla", { lang: "en" })).suggestions.filter((s) => s.kind === "popular").map((s) => s.text)).toEqual(["black cat"]);
      // the salts: one per day, random, and only today's and yesterday's are kept
      const salts = (await env2.DB.prepare("SELECT k, v FROM settings WHERE k LIKE 'suggest:salt:%' ORDER BY k").all<any>()).results;
      expect(salts).toHaveLength(2);
      expect(salts[0].v).toMatch(/^[0-9a-f]{32}$/);
      expect(salts[0].v).not.toBe(salts[1].v);
      vi.setSystemTime(day0 + 3 * 86400_000);
      await pruneLogs(env2);
      expect((await env2.DB.prepare("SELECT COUNT(*) AS n FROM settings WHERE k LIKE 'suggest:salt:%'").first<any>()).n).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("addresses: IPv4 as it is, IPv6 by its /64", () => {
    expect(addressKey("203.0.113.7")).toBe("203.0.113.7");
    expect(addressKey("2001:db8:1:2:aaaa:bbbb:cccc:dddd")).toBe("2001:db8:1:2::/64");
    expect(addressKey("2001:DB8:0001:0002::1")).toBe("2001:db8:1:2::/64");
    expect(addressKey("2001:db8::1")).toBe("2001:db8:0:0::/64");
    expect(addressKey("::1")).toBe("0:0:0:0::/64");
    expect(addressKey("::ffff:198.51.100.4")).toBe("198.51.100.4");
  });

  it("only page-one text searches in the default safe mode that found something, and only when it is on", async () => {
    const env = await corpus(ROWS, { popular: true });
    const exec = new FakeExec();
    const run = (path: string) => app.fetch(new Request(`https://search.test${path}`, { headers: { "cf-connecting-ip": "198.51.100.1" } }), env, exec as unknown as ExecutionContext);
    await run("/search?q=dragon&nsfw=include");
    await run("/search?q=zzzqqq");
    await run("/search?q=");
    await exec.settle();
    expect(await people(env)).toEqual([]);
    const off = await corpus(ROWS);
    const r = await search(off, "dragon", "198.51.100.2");
    expect(r.status).toBe(200);
    expect(await people(off)).toEqual([]);
    // and nothing is proposed while it is off, whatever the table holds
    await searchedBy(off, "red dragon", 9, 3);
    expect((await suggestFor(off, "red ", { lang: "en" })).suggestions.filter((s) => s.kind === "popular")).toEqual([]);
  });
});

describe("the cost of a pause in typing", () => {
  it("two letters: no read of the vocabulary or of the titles (the concept names, in memory, complete them)", async () => {
    const env = await corpus(ROWS);
    const seen: string[] = [];
    const prepare = env.DB.prepare.bind(env.DB);
    (env.DB as any).prepare = (sql: string) => (seen.push(sql), prepare(sql));
    const r = await suggestFor(env, "dr", { lang: "en" });
    expect(texts(r)).toContain("dragon");
    expect(seen.filter((q) => /FROM vocab|posts_fts MATCH/.test(q))).toEqual([]);
    seen.length = 0;
    await suggestFor(env, "dra", { lang: "en" });
    expect(seen.filter((q) => /FROM vocab/.test(q))).toHaveLength(1);
  });
});

describe("the questions it proposes get honest answers", () => {
  it("a tie for the most active artist names everyone in it (and what /search hides does not count)", async () => {
    const env = await corpus(ROWS);
    const en = await ask(env, { question: "who is the most active artist?", planner: "rules" });
    // alice, bob, carol and dave have two safe artworks each; eve's three are NSFW
    expect(en.answer_text).toBe("@alice, @bob, @carol and @dave posted the most artworks (2 each).");
    expect(en.confidence).toBeLessThan(0.5);
    const fr = await ask(env, { question: "quel est l'artiste le plus actif ?", planner: "rules" });
    expect(fr.answer_text).toBe("@alice, @bob, @carol et @dave ont posté le plus de « œuvre » (2 chacun).");
    const de = await ask(env, { question: "wer ist der aktivste Künstler?", planner: "rules" });
    expect(de.answer_text).toBe("@alice, @bob, @carol und @dave haben die meisten „Kunstwerk“ gepostet (je 2).");
    // one more artwork for bob: no tie
    const more = await corpus([...ROWS, ["bob", "cat-three", "Cat three", ["cat"], ["cat"], "black", 1]]);
    expect((await ask(more, { question: "who is the most active artist?", planner: "rules" })).answer_text).toBe("@bob posted the most artworks (3).");
  });

  it("a tie of sixty: everyone counted", async () => {
    const rows: Row[] = Array.from({ length: 60 }, (_, i) => [`a${String(i).padStart(2, "0")}x`, `p${i}`, `Piece ${i}`, ["cat"], ["cat"], "black", 1] as Row);
    const env = await corpus(rows);
    expect((await ask(env, { question: "who is the most active artist?", planner: "rules" })).answer_text).toBe("@a00x, @a01x, @a02x and 57 others posted the most artworks (1 each).");
  });

  it("more than four in a tie: the first three and how many others", async () => {
    const rows: Row[] = ["ann", "ben", "cid", "dan", "eva"].map((a, i) => [a, `p${i}`, `Piece ${i}`, ["cat"], ["cat"], "black", 1]);
    const env = await corpus(rows);
    expect((await ask(env, { question: "who is the most active artist?", planner: "rules" })).answer_text).toBe("@ann, @ben, @cid and 2 others posted the most artworks (1 each).");
  });
});

describe("questions and examples lead where they say", () => {
  it("every question template goes to /ask, every help question to /help", async () => {
    const ask = [
      "who posted the first dragon?", "how many dragon artworks?", "who posted the first cat?", "most liked cat?",
      "qui a posté la première œuvre de chat ?", "combien d'œuvres de chat ?", "combien de chats ?", "wie viele Katzen?",
      "who is the most active artist?", "what's the latest artwork?", "what is the most liked artwork?",
      "quel est l'artiste le plus actif ?", "quelle est la dernière œuvre ?", "quelle est l'œuvre la plus aimée ?",
      "wer ist der aktivste Künstler?", "was ist das neueste Bild?", "wer hat die erste Katze gepostet?", "when was the first dragon posted?",
    ];
    for (const q of ask) expect(await routeOf(q), q).toBe("ask");
    const help = [
      "how do I mint an artwork?", "what are the fees?", "how do royalties work?", "what is PXS?", "how do I sell my art?",
      "comment minter une œuvre ?", "quels sont les frais ?", "comment fonctionnent les royalties ?", "qu'est-ce que PXS ?",
      "was ist PXS?", "welche Gebühren gibt es?", "wie funktionieren Lizenzgebühren?",
    ];
    for (const q of help) expect(await routeOf(q), q).toBe("help");
  });

  it("examples in the reader's language, from what the index holds, cached", async () => {
    const env = await corpus(ROWS, {
      docs: { "royalties.md": "# Royalties\n\nHow do royalties work? A royalty is paid to the artist on every resale.\n\n## How they are paid\n\nRoyalties are paid in PXS." },
    });
    const en = await suggestExamples(env, "en");
    const t = en.examples.map((e) => e.text);
    // the colour that goes with the first subject that has one twice: black cats
    expect(t).toEqual(expect.arrayContaining(["cat", "dragon", "black cat", "who posted the first cat?", "Red Dragon", "how do royalties work?"]));
    expect(en.examples.find((e) => e.text === "how do royalties work?")).toMatchObject({ kind: "help", route: "help" });
    for (const e of en.examples) if (e.kind === "question") expect(await routeOf(e.text), e.text).toBe("ask");
    expect(en.examples.length).toBeLessThanOrEqual(12);
    // never what /search hides by default, never other people's searches
    expect(t).not.toEqual(expect.arrayContaining(["Naughty Dragon"]));
    expect(t.filter((x) => /catwalk|lingerie/i.test(x))).toEqual([]);
    expect(en.examples.filter((e) => e.kind === "popular")).toEqual([]);
    // cached: a new artwork does not show until the cache expires
    await env.DB.prepare("DELETE FROM artwork_concepts").run();
    expect((await suggestExamples(env, "en")).examples).toEqual(en.examples);

    const fr = (await suggestExamples(env, "fr-CH")).examples.map((e) => e.text);
    expect(fr).toEqual(expect.arrayContaining(["qui a posté la première œuvre de chat ?", "quel est l'artiste le plus actif ?"]));
    expect(fr).not.toContain("red dragon");
  });

  it("documentation sections while typing, with their page", async () => {
    const env = await corpus(ROWS, { docs: { "royalties.md": "# Royalties\n\nAbout royalties.\n\n## How they are paid\n\nIn PXS." } });
    const r = await suggestFor(env, "roy", { lang: "en" });
    const help = r.suggestions.filter((s) => s.kind === "help");
    expect(help[0]).toMatchObject({ route: "help", source: { title: "Royalties" } });
    expect(help.map((s) => s.text)).toEqual(expect.arrayContaining(["Royalties › How they are paid"]));
    expect(help.find((s) => s.text === "Royalties › How they are paid")?.source?.url).toBe("https://github.com/pixa/info/blob/main/royalties.md#how-they-are-paid");
  });

  it("over HTTP, cacheable", async () => {
    const env = await corpus(ROWS);
    const exec = new FakeExec();
    const res = await app.fetch(new Request("https://search.test/suggest?q=dra&lang=en"), env, exec as unknown as ExecutionContext);
    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toContain("max-age=60");
    expect((await res.json<any>()).completion).toBe("dragon");
    const ex = await app.fetch(new Request("https://search.test/suggest?lang=de"), env, exec as unknown as ExecutionContext);
    const body = await ex.json<any>();
    expect(body.lang).toBe("de");
    expect(body.examples.map((e: any) => e.text)).toEqual(expect.arrayContaining(["Katze", "wer ist der aktivste Künstler?"]));
  });

  it("/query for the box: results=0 runs no search; a refused answer says so", async () => {
    const env = await corpus(ROWS);
    const exec = new FakeExec();
    const get = async (path: string) => (await app.fetch(new Request(`https://search.test${path}`), env, exec as unknown as ExecutionContext)).json<any>();
    const before = Number((await env.DB.prepare("SELECT COUNT(*) AS n FROM query_log").first<{ n: number }>())?.n);
    const s = await get("/query?q=red%20dragon&results=0");
    expect(s).toMatchObject({ route: "search", results: null });
    const none = await get("/query?q=who%20posted%20the%20first%20unicorn%3F&results=0");
    expect(none.route).toBe("ask");
    expect(none.results).toBeNull();
    await exec.settle();
    expect(Number((await env.DB.prepare("SELECT COUNT(*) AS n FROM query_log").first<{ n: number }>())?.n)).toBe(before);
    (env as any).RL_HEAVY = { limit: async () => ({ success: false }) };
    const spent = await get("/query?q=who%20posted%20the%20first%20cat%3F&results=0");
    expect(spent).toMatchObject({ route: "search", answer_budget: "spent", results: null });
    (env as any).RL_HEAVY = undefined;
    expect((await get("/query?q=who%20posted%20the%20first%20cat%3F&results=0")).answer_budget).toBeUndefined();
  });

  it("a budget of its own, and one answer per location and minute for the same text", async () => {
    const env = await corpus(ROWS);
    const keys: string[] = [];
    (env as any).RL_PUBLIC = { limit: async ({ key }: { key: string }) => (keys.push(key), { success: true }) };
    const store = new Map<string, Response>();
    const puts: string[] = [];
    vi.stubGlobal("caches", {
      default: {
        match: async (r: Request) => store.get(r.url)?.clone(),
        put: async (r: Request, res: Response) => void (puts.push(r.url), store.set(r.url, res)),
      },
    });
    const exec = new FakeExec();
    const get = async (u: string) => {
      const res = await app.fetch(new Request(u, { headers: { "cf-connecting-ip": "203.0.113.9", origin: "https://pixa.pics" } }), env, exec as unknown as ExecutionContext);
      await exec.settle();
      return res;
    };
    const a = await get("https://search.test/suggest?q=dra&lang=en-GB");
    expect(a.headers.get("access-control-allow-origin")).toBe("*");
    expect((await a.json<any>()).completion).toBe("dragon");
    expect(puts).toHaveLength(1);
    // the same text from the same location: the kept answer, even after the corpus changed
    await env.DB.prepare("DELETE FROM artwork_concepts").run();
    const b = await get("https://search.test/suggest?lang=en&q=dra");
    expect((await b.json<any>()).completion).toBe("dragon");
    expect(b.headers.get("access-control-allow-origin")).toBe("*");
    expect(puts).toHaveLength(1);
    await get("https://search.test/search?q=dragon");
    expect(keys).toEqual(["suggest:203.0.113.9", "suggest:203.0.113.9", "public:203.0.113.9"]);
    // with a binding of its own, suggestions use it
    const own: string[] = [];
    (env as any).RL_SUGGEST = { limit: async ({ key }: { key: string }) => (own.push(key), { success: false }) };
    const refused = await get("https://search.test/suggest?q=cat&lang=en");
    expect(refused.status).toBe(429);
    expect(own).toEqual(["suggest:203.0.113.9"]);
  });
});
