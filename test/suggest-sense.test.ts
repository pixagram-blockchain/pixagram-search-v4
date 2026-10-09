// Suggestions that make sense on a corpus like production's: the artists' tags and titles plus
// AI descriptions (Gemma's long captions), which bring words like "catastrophically" and concepts
// like "hair" that nobody posts about. Each case below was nonsense on the live stack (October
// 2026): "dr" → "who posted the first meme?", "ca" → "cabelo", "cat in" → "cat interior",
// "how do" → "how downtown", the examples' "how many hair artworks?".

import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { updateSearchDocAi, upsertPost } from "../src/db/posts";
import { suggestExamples, suggestFor } from "../src/search/suggest";
import { syncDocs } from "../src/docs/sync";

const T0 = Date.UTC(2026, 8, 1) / 1000;

afterEach(() => vi.unstubAllGlobals());

interface Art {
  author: string;
  title: string;
  tags: string[];
  /** [concept, source]: "tag" and "title" are the artist's words, "vlm" the AI description's */
  concepts: Array<[string, "tag" | "title" | "vlm"]>;
  caption?: string;
  color?: string;
}

const ARTS: Art[] = [
  { author: "karo", title: "Black Cat Love", tags: ["cat"], concepts: [["cat", "tag"], ["room", "vlm"]], caption: "A black cat with green eyes sitting in a cozy kitchen.", color: "black" },
  { author: "matus", title: "The King of the Cats", tags: ["cat", "king"], concepts: [["cat", "tag"], ["king", "tag"], ["church", "vlm"]], caption: "A crowned cat on a throne inside a gothic cathedral.", color: "black" },
  { author: "laura", title: "Kitten", tags: ["cat"], concepts: [["cat", "tag"]], caption: "A small kitten playing with a ball of yarn.", color: "orange" },
  { author: "mrdragon", title: "Mr Dragon is here", tags: ["dragon"], concepts: [["dragon", "tag"]], caption: "A green dragon breathing fire, dramatic lighting.", color: "green" },
  { author: "alice", title: "Red Dragon", tags: ["dragon"], concepts: [["dragon", "tag"], ["castle", "vlm"]], caption: "A dramatic red dragon flying over a castle.", color: "red" },
  { author: "bob", title: "Dragon Two", tags: ["dragon"], concepts: [["dragon", "tag"]], caption: "A dragon catastrophically crashing into a tower, dramatic sky.", color: "red" },
  { author: "matus", title: "Blue Church", tags: ["church"], concepts: [["church", "tag"]], caption: "A blue church with an onion dome under a cloudy sky.", color: "blue" },
  { author: "tetiana", title: "Catalonia trip", tags: ["travel"], concepts: [["travel", "tag"]], caption: "A sunny street in Barcelona with palm trees.", color: "yellow" },
  { author: "wang", title: "Pond", tags: ["nature"], concepts: [["insect", "vlm"]], caption: "A dragonfly hovering over a quiet pond.", color: "green" },
  { author: "mathiew", title: "Deep sea diver", tags: ["diver"], concepts: [["diver", "tag"], ["hat", "vlm"]], caption: "A deep sea diver wearing a brass helmet.", color: "blue" },
  // portraits: the AI descriptions all talk about hair, the artists never tag it
  ...["Me", "Myself", "Got my nails done", "Sunday", "Lookin stylish"].map(
    (title, i): Art => ({ author: `p${i}`, title, tags: ["portrait"], concepts: [["portrait", "tag"], ["hair", "vlm"]], caption: "A portrait of a woman with long brown hair, dramatic light.", color: "black" }),
  ),
];

async function corpus(arts: Art[], docs?: Record<string, string>): Promise<TestEnv> {
  const env = makeEnv({ PLANNER_BACKEND: "rules", DOCS_REPO: "pixa/info" });
  let i = 0;
  for (const a of arts) {
    const created = T0 + i++ * 3600;
    const permlink = `${a.title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-${i}`;
    const { id } = await upsertPost(
      env,
      { author: a.author, permlink, type: "artwork", title: a.title, description: "", body: "", bodyLength: 0, category: "pixagram", tags: a.tags, app: "pixagram/3", nsfw: false, aiTraining: null, licenseJson: null, royaltyPct: null, created, updated: created, deleted: false, netVotes: 1, payout: 0, children: 0, image: null },
      null,
    );
    await env.DB.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, primary_color, r2_orig_key, ai_nsfw, updated) VALUES (?, ?, 'image/png', 10, ?, ?, 0.01, ?)")
      .bind(id, `h${id}`, a.color ?? null, `orig/h${id}.png`, created)
      .run();
    for (const [c, source] of a.concepts) await env.DB.prepare("INSERT INTO artwork_concepts (post_id, concept, confidence, source) VALUES (?, ?, 0.9, ?)").bind(id, c, source).run();
    if (a.caption) await updateSearchDocAi(env.DB, id, a.caption, []);
  }
  if (docs) {
    (env as any).VEC_DOCS = new FakeVectorize([]);
    installGitHub({ owner: "pixa", repo: "info", head: "5".repeat(40), files: docs, calls: [] });
    env._ai.handler = (model, input) => (model === "@cf/baai/bge-m3" ? { data: input.text.map(() => [1, 0, 0]) } : { response: {} });
    await syncDocs(env);
  }
  return env;
}

const texts = (r: { suggestions: Array<{ text: string }> }) => r.suggestions.map((s) => s.text);
const kinds = (r: { suggestions: Array<{ kind: string }> }, kind: string) => (r.suggestions as Array<{ kind: string; text: string }>).filter((s) => s.kind === kind).map((s) => s.text);

describe("suggestions that make sense", () => {
  let env: TestEnv;
  beforeAll(async () => {
    env = await corpus(ARTS, { "royalties.md": "# Royalties\n\nHow do royalties work? A royalty is paid to the artist on every resale.\n\n## How they are paid\n\nRoyalties are paid in PXS." });
  });

  it("two letters: the usual names in the reader's language, and no question yet", async () => {
    const dr = await suggestFor(env, "dr", { lang: "en" });
    expect(kinds(dr, "complete")).toContain("dragon");
    expect(kinds(dr, "question")).toEqual([]); // was "who posted the first meme?" ("drôle" is French for funny)
    const ca = await suggestFor(env, "ca", { lang: "en" });
    expect(kinds(ca, "complete")).toContain("cat");
    for (const w of ["cabelo", "cara", "casa", "castillo"]) expect(texts(ca)).not.toContain(w); // other languages' names
    expect(kinds(ca, "question")).toEqual([]); // was "who posted the first portrait?"
  });

  it("a question is about what is typed: never the concept another of its names belongs to", async () => {
    for (const [q, wrong] of [["dra", "insect"], ["drag", "insect"], ["kit", "room"], ["hel", "hat"], ["cath", "church"], ["hello", "cartoon"]] as const) {
      const r = await suggestFor(env, q, { lang: "en" });
      expect(texts(r).filter((t) => t.includes(wrong)), q).toEqual([]);
    }
    // the only subject "dra" can still become: questions about dragons
    expect(texts(await suggestFor(env, "dra", { lang: "en" }))).toEqual(expect.arrayContaining(["dragon", "who posted the first dragon?", "how many dragon artworks?"]));
    // a word that is not a subject's usual name gets no question ("kitten" names cats, "helmet" hats)
    expect(kinds(await suggestFor(env, "kitten", { lang: "en" }), "question")).toEqual([]);
    expect(kinds(await suggestFor(env, "helmet", { lang: "en" }), "question")).toEqual([]);
  });

  it("a word one AI description used once is never proposed; a complete subject is extended only by the artists' words", async () => {
    const cat = await suggestFor(env, "cat", { lang: "en" });
    expect(texts(cat)).not.toContain("catastrophically");
    expect(texts(cat)).not.toContain("cathedral"); // the AI descriptions' word, and "cat" is already a subject
    expect(texts(cat)).toEqual(expect.arrayContaining(["who posted the first cat?", "how many cat artworks?", "Black Cat Love"]));
    expect(kinds(cat, "complete")).toEqual(["catalonia"]); // in a title
    expect(texts(await suggestFor(env, "catas", { lang: "en" }))).not.toContain("catastrophically");
    // a word several descriptions use is a fair completion once it is what the letters lead to
    expect(kinds(await suggestFor(env, "drama", { lang: "en" }), "complete")).toEqual(["dramatic"]);
  });

  it("after other words: only what occurs with them; a function word is not completed", async () => {
    expect(kinds(await suggestFor(env, "cat in", { lang: "en" }), "complete")).toEqual([]); // was "cat interior", "cat inferno"…
    expect(kinds(await suggestFor(env, "red dra", { lang: "en" }), "complete")).toEqual(["red dragon"]);
    expect(kinds(await suggestFor(env, "blue dra", { lang: "en" }), "complete")).toEqual([]); // no post says both
  });

  it("a question begun gets whole questions, never a word completed out of the vocabulary", async () => {
    const how = await suggestFor(env, "how do", { lang: "en" });
    expect(kinds(how, "complete")).toEqual([]); // was "how downtown", "how donna", "how dollar"
    expect(texts(how)).toContain("how do royalties work?"); // the documentation answers it
    expect(texts(how)).not.toContain("how do I mint an artwork?"); // it does not answer that
    const who = await suggestFor(env, "who", { lang: "en" });
    expect(texts(who)).toEqual(expect.arrayContaining(["who posted the first portrait?", "who is the most active artist?"]));
    expect(texts(who).filter((t) => /hair/.test(t))).toEqual([]);
    expect(texts(await suggestFor(env, "what is", { lang: "en" }))).toContain("what is the most liked artwork?");
    // a question word still being typed is a word like any other ("wo" → "woman" in German is "wo")
    expect(kinds(await suggestFor(env, "por", { lang: "en" }), "complete")).toContain("portrait");
  });

  it("examples: subjects the artists post about, colours only for things", async () => {
    const t = (await suggestExamples(env, "en")).examples.map((e) => e.text);
    expect(t.filter((x) => /hair/.test(x))).toEqual([]); // was "hair", "how many hair artworks?"
    expect(t).toEqual(expect.arrayContaining(["portrait", "cat", "who posted the first portrait?"]));
    expect(t).toContain("black cat");
    expect(t).not.toContain("black portrait"); // a dark background, not a black portrait
  });

  it("French: the names as written, with their accents", async () => {
    const r = await suggestFor(env, "cha", { lang: "fr" });
    expect(kinds(r, "complete")).toEqual(expect.arrayContaining(["chat", "château"]));
    expect(kinds(r, "question")).toEqual([]); // "chat" or "château": not clear yet
    expect(texts(await suggestFor(env, "château", { lang: "fr" }))).toContain("qui a posté la première œuvre de château ?");
    expect(texts(await suggestFor(env, "chateau", { lang: "fr" }))).toContain("combien d'œuvres de château ?");
  });
});
