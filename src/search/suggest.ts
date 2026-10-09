// Suggestions for the search box (GET /suggest): what people are probably typing, drawn from what
// the index holds, so that a suggestion always leads somewhere.
//
//   no text     examples for the box's placeholder, in the UI language: the subjects the corpus
//               has the most of (in English with the colour that goes with one), questions /ask
//               answers about them, artwork titles, documentation questions the documentation can
//               answer. Cached per language for half an hour.
//   text        completions of the word being typed: the usual names of concepts in the reader's
//               language, and words the posts carry (in the artists' own words, or often in the AI
//               descriptions), weighted by how many posts carry them, and after other words only
//               those that occur with them; the same inside a question being typed ("who posted
//               the first dra" → "… dragon?"); whole questions while one is begun ("who po",
//               "how do"); questions about the subject when it is clear what it is (English,
//               French); titles; documentation sections; popular searches; a spelling correction
//               when nothing else matches.
//   completion  the first suggestion that extends the text as typed: the box shows the rest as
//               ghost text.
//
// Only D1 (index lookups; the counts and the popular searches are cached) and the concept
// vocabulary, no model call: fast enough for every pause in typing (the client debounces, the
// route caches at the edge). A popular search is shown to others only after three runs with
// results in the last 30 days at two different hours at least, when every word of it is known to
// the index, from searches run in the default safe mode, and never when it looks like an address,
// an e-mail or a number. Nothing a suggestion shows comes from a post /search hides by default
// (NSFW by the author's flag or the AI estimate).

import type { Env } from "../env";
import { now } from "../env";
import { aliasesStartingWith, ancestorsWithDepth, conceptDef, conceptLabel, STRUCTURAL, vocabLang, type VocabLang } from "../concepts";
import { fold, hasCjk, isStopword, tokens } from "../lib/text";
import { COLOR_WORDS } from "./lexicon";
import { lexicalDocs } from "../help/retrieve";
import { blobUrl, repoRef } from "../docs/github";
import { suggest as spellSuggest } from "./spell";
import { DOCS_KNOWS, QUESTION_START, routeQuery } from "./router";
import { planQuery } from "./planner";

export type SuggestionKind = "complete" | "question" | "title" | "help" | "popular" | "correction";

export interface Suggestion {
  /** what the row shows, and what goes into the box when it is picked */
  text: string;
  kind: SuggestionKind;
  /** where the box should send it (search results, an /ask answer, a /help answer) */
  route: "search" | "ask" | "help";
  /** what to run when it is not the text itself (a documentation section) */
  query?: string;
  /** artworks (or documents) behind the subject, when known */
  n?: number;
  /** the documentation section of a help suggestion */
  source?: { title: string; heading: string; url: string };
  /** the post a title suggestion names */
  post?: { id: number; author: string; permlink: string; category: string | null; type: "artwork" | "blog"; image: string | null };
}

export interface SuggestResponse {
  q: string;
  lang: VocabLang;
  completion: string | null;
  suggestions: Suggestion[];
  took_ms: number;
}

export interface ExamplesResponse {
  lang: VocabLang;
  examples: Suggestion[];
  took_ms: number;
}

const MAX_TEXT = 100;
const TOP = String.fromCharCode(0xffff);

/** The typed text with its trailing space kept (it says the last word is finished). */
export function suggestText(raw: string | null | undefined): string {
  return String(raw ?? "")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/^\s+/, "")
    .replace(/\s+$/, (m) => (m ? " " : ""))
    .slice(0, MAX_TEXT);
}

/** Folded text for comparisons: lower case, no accents, "œ" as "oe", apostrophes as spaces, one space between words. */
function norm(s: string): string {
  return fold(s).replace(/œ/g, "oe").replace(/['’]/g, " ").replace(/\s+/g, " ");
}

// ---- counts per concept (live artworks), per database, for ten minutes ----------------------------

/**
 * Live artworks per concept: `all` from every source, `human` where the artists' own words (tags,
 * title, description) name it. The AI descriptions name what an image shows ("hair", "sky"), not
 * what it is about: subjects to propose come from `human`.
 */
interface Counts {
  all: Map<string, number>;
  human: Map<string, number>;
}

const COUNTS = new WeakMap<object, { at: number; counts: Counts }>();
const COUNTS_KEY = "suggest:counts:v2";

/** From this isolate (10 minutes), else KV (10 minutes), else one GROUP BY. */
async function conceptCounts(env: Env): Promise<Counts> {
  const hit = COUNTS.get(env.DB);
  if (hit && Date.now() - hit.at < 10 * 60_000) return hit.counts;
  let rows = (await env.CACHE.get(COUNTS_KEY, "json").catch(() => null)) as Array<[string, number, number]> | null;
  if (!Array.isArray(rows)) {
    const res =
      (
        await env.DB.prepare(
          `SELECT ac.concept AS concept, COUNT(*) AS n, SUM(CASE WHEN ac.source IN ('tag', 'title', 'description') THEN 1 ELSE 0 END) AS human
           FROM artwork_concepts ac JOIN posts p ON p.id = ac.post_id LEFT JOIN artworks a ON a.post_id = p.id
           WHERE p.deleted = 0 AND ${SAFE} GROUP BY ac.concept`,
        )
          .all<{ concept: string; n: number; human: number }>()
          .catch(() => ({ results: [] as Array<{ concept: string; n: number; human: number }> }))
      ).results ?? [];
    rows = res.map((r) => [r.concept, Number(r.n) || 0, Number(r.human) || 0]);
    if (rows.length) await env.CACHE.put(COUNTS_KEY, JSON.stringify(rows), { expirationTtl: 600 }).catch(() => {});
  }
  const counts: Counts = { all: new Map(rows.map(([c, n]) => [c, n])), human: new Map(rows.filter(([, , h]) => h > 0).map(([c, , h]) => [c, h])) };
  COUNTS.set(env.DB, { at: Date.now(), counts });
  return counts;
}

/** Concepts that are a parent of another concept in the index ("animal" over "cat"): too broad for an example. */
function parentsIn(counts: Map<string, number>): Set<string> {
  const out = new Set<string>();
  for (const id of counts.keys()) for (const p of conceptDef(id)?.parents ?? []) out.add(p);
  return out;
}

// ---- completing a word ------------------------------------------------------------------------------

interface WordCompletion {
  /** folded */
  word: string;
  /** posts behind it: the concept's artworks for a concept's usual name, else the posts that carry the word */
  n: number;
  concept?: string;
  /** the languages the word names the concept in */
  langs?: VocabLang[];
  /** the concept's usual name (its first) in the reader's language */
  main?: boolean;
  /** posts whose own words (title, description, body, tags) carry it */
  own?: number;
}

const WORDISH = /^[\p{L}\p{N}][\p{L}\p{N}-]*$/u;

/**
 * The columns where artists name their work: titles and tags. Descriptions and blog bodies are
 * prose, like the AI descriptions ("catastrophically", "seamlessly" come from there).
 */
const OWN_WORDS = "{title tags}";

/** A word the AI descriptions use this often counts even when no artist does. */
const AI_WORD_POSTS = 3;

/**
 * Words that complete `partial` (2+ letters), for a reader of `lang`:
 *   - the usual name of a concept in the reader's language ("dragon", "chat" for French), when
 *     safe artworks show that concept; from memory, so two letters cost no query;
 *   - from three letters, other words, each only when posts carry it: another name of a concept
 *     in the reader's language ("dragonfly" names insects, and counts by the posts that say
 *     dragonfly), a name in another language that an artist used, a word of the corpus that
 *     the artists used or the AI descriptions often do.
 * Best first: the usual names by their concept's artworks, then the artists' words, then the
 * descriptions' words, each by posts. A name in another language nobody wrote ("cabelo" for an
 * English reader) or a word one description used once ("catastrophically") is never proposed.
 */
async function completeWord(env: Env, partial: string, counts: Counts, lang: VocabLang): Promise<WordCompletion[]> {
  const p = fold(partial);
  if (p.length < 2 || !WORDISH.test(p)) return [];
  const out = new Map<string, WordCompletion>();
  for (const a of aliasesStartingWith(p, 240)) {
    const n = counts.all.get(a.concept) ?? 0;
    if (!n || STRUCTURAL.has(a.concept)) continue;
    let c = out.get(a.alias);
    if (c && c.concept !== a.concept) {
      if (n <= c.n) continue; // a name of two concepts: the one with more artworks
      c = undefined;
    }
    if (!c) out.set(a.alias, (c = { word: a.alias, n, concept: a.concept, langs: [] }));
    if (!c.langs!.includes(a.lang)) c.langs!.push(a.lang);
  }
  for (const c of out.values()) c.main = conceptDef(c.concept!)?.aliases[lang]?.[0] === c.word;
  if (p.length < 3) return rank([...out.values()].filter((c) => c.main));

  // The corpus's own words (a two-letter range is a good part of the vocabulary: from three letters).
  const rows =
    (
      await env.DB.prepare("SELECT term, df FROM vocab WHERE term >= ?1 AND term < ?2 AND df > 0 ORDER BY df DESC LIMIT 40")
        .bind(p, p + TOP)
        .all<{ term: string; df: number }>()
        .catch(() => ({ results: [] as Array<{ term: string; df: number }> }))
    ).results ?? [];
  const terms = rows
    .filter((r) => r.term.length >= 3 && !/\d{3,}|^\d+$/.test(r.term) && !(isStopword(r.term) && r.term !== p) && !(r.term !== p && QUESTION_START.test(r.term)))
    .slice(0, 12);
  for (const r of terms) if (!out.has(r.term)) out.set(r.term, { word: r.term, n: 0 });
  // what posts /search shows by default carry each word that is not a concept's usual name
  const ask = [...out.values()]
    .filter((c) => !c.main)
    .sort((a, b) => Number(!!b.langs?.includes(lang)) - Number(!!a.langs?.includes(lang)) || b.n - a.n)
    .slice(0, 24);
  const used = await wordUse(env, ask.map((c) => c.word));
  const kept: WordCompletion[] = [];
  for (const c of out.values()) {
    if (c.main) {
      kept.push(c);
      continue;
    }
    const u = used.get(c.word);
    if (!u) continue;
    // the AI descriptions and the descriptions' prose are in English: their words for English readers
    if (u.own >= 1 || (c.langs?.includes(lang) && u.all >= 1) || (lang === "en" && u.all >= AI_WORD_POSTS)) kept.push({ ...c, n: u.all, own: u.own });
  }
  return rank(kept);
}

const tier = (c: WordCompletion) => (c.main ? 2 : (c.own ?? 0) > 0 ? 1 : 0);
const rank = (list: WordCompletion[]) => list.sort((a, b) => tier(b) - tier(a) || b.n - a.n || a.word.length - b.word.length || (a.word < b.word ? -1 : 1));

/**
 * For each word, how many posts /search shows by default carry it (`all`, up to 100) and how
 * many carry it in their own words (`own`), one statement per word, in one round trip per 40.
 */
async function wordUse(env: Env, words: string[]): Promise<Map<string, { all: number; own: number }>> {
  const out = new Map<string, { all: number; own: number }>();
  const list = [...new Set(words)].filter((w) => WORDISH.test(w));
  const count = (match: string) =>
    `(SELECT COUNT(*) FROM (SELECT 1 FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid LEFT JOIN artworks a ON a.post_id = p.id
      WHERE posts_fts MATCH ${match} AND p.deleted = 0 AND ${SAFE} LIMIT 100))`;
  for (let i = 0; i < list.length; i += 40) {
    const chunk = list.slice(i, i + 40);
    const res = await env.DB.batch<{ n: number; own: number }>(chunk.map((w) => env.DB.prepare(`SELECT ${count("?1")} AS n, ${count("?2")} AS own`).bind(`"${w}"`, `${OWN_WORDS} : "${w}"`))).catch(() => null);
    chunk.forEach((w, k) => {
      const r = res?.[k]?.results?.[0];
      out.set(w, { all: Number(r?.n) || 0, own: Number(r?.own) || 0 });
    });
  }
  return out;
}

/**
 * For each word, how many posts /search shows by default carry it (up to `cap`), in one round trip
 * per 50 words. The vocabulary's own counts include what /search hides; this is what a
 * suggestion may stand on.
 */
async function safeCounts(env: Env, words: string[], cap: number): Promise<Map<string, number>> {
  const out = new Map<string, number>();
  const list = [...new Set(words)].filter((w) => WORDISH.test(w));
  for (let i = 0; i < list.length; i += 50) {
    const chunk = list.slice(i, i + 50);
    const res = await env.DB.batch<{ n: number }>(
      chunk.map((w) =>
        env.DB.prepare(
          `SELECT COUNT(*) AS n FROM (SELECT 1 FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid LEFT JOIN artworks a ON a.post_id = p.id
           WHERE posts_fts MATCH ?1 AND p.deleted = 0 AND ${SAFE} LIMIT ?2)`,
        ).bind(`"${w}"`, cap),
      ),
    ).catch(() => null);
    chunk.forEach((w, k) => out.set(w, Number(res?.[k]?.results?.[0]?.n) || 0));
  }
  return out;
}

/**
 * The completions that some post writes right after the words before ("red dra" → "red dragon",
 * as a title says; never "red dramatic" or "cat interior"), with the other words typed anywhere
 * in the same post: one statement per word, in one round trip.
 */
async function withHead(env: Env, head: string[], list: WordCompletion[]): Promise<WordCompletion[]> {
  const words = head.map((w) => w.replace(/[^\p{L}\p{N}-]+/gu, "")).filter((w) => WORDISH.test(w));
  if (!words.length || !list.length) return list;
  const trailing = words.slice(-3);
  const others = words.slice(0, -3).filter((w) => !isStopword(w)).map((w) => `"${w}"`);
  const res = await env.DB.batch<{ n: number }>(
    list.map((c) =>
      env.DB.prepare(
        `SELECT COUNT(*) AS n FROM (SELECT 1 FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid LEFT JOIN artworks a ON a.post_id = p.id
         WHERE posts_fts MATCH ?1 AND p.deleted = 0 AND ${SAFE} LIMIT 1)`,
      ).bind([`"${[...trailing, c.word].join(" ")}"`, ...others].join(" ")),
    ),
  ).catch(() => null);
  return list.filter((_, k) => Number(res?.[k]?.results?.[0]?.n) > 0);
}

/** How well a word names its concept for this reader: its usual name, in their language, used by artists, English. */
const nameScore = (c: WordCompletion, lang: VocabLang) => (c.main ? 8 : 0) + (c.langs?.includes(lang) ? 4 : 0) + ((c.own ?? 0) > 0 ? 2 : 0) + (c.langs?.includes("en") ? 1 : 0);

/** A good name for the reader: the usual one, one in their language, or the artists' own word. */
const goodName = (c: WordCompletion, lang: VocabLang) => !!c.main || !!c.langs?.includes(lang) || (c.own ?? 0) > 0;

/** Where the word stands among its concept's names (the first is the usual one: "katze" before "kater"). */
function namePosition(c: WordCompletion, lang: VocabLang): number {
  const d = c.concept ? conceptDef(c.concept) : undefined;
  if (!d) return 99;
  for (const l of [lang, "en" as VocabLang, ...(c.langs ?? [])]) {
    const i = d.aliases[l]?.indexOf(c.word) ?? -1;
    if (i >= 0) return i;
  }
  return 99;
}

/**
 * One word per concept, the best name for the reader (then the usual one, then the shorter), in
 * the order given; words of the corpus that name no concept stay as they are. `keep` is the word
 * typed, kept as its concept's name when it is a good one.
 */
function byConcept(list: WordCompletion[], lang: VocabLang, keep?: WordCompletion): WordCompletion[] {
  const best = new Map<string, WordCompletion>();
  const better = (a: WordCompletion, b: WordCompletion) =>
    nameScore(a, lang) - nameScore(b, lang) ||
    namePosition(b, lang) - namePosition(a, lang) ||
    b.word.length - a.word.length ||
    (a.word < b.word ? 1 : a.word > b.word ? -1 : 0);
  for (const c of list) {
    if (!c.concept) continue;
    const cur = best.get(c.concept);
    if (cur && cur === keep && goodName(cur, lang)) continue;
    if (c === keep && goodName(c, lang)) best.set(c.concept, c);
    else if (!cur || better(c, cur) > 0) best.set(c.concept, c);
  }
  return list.filter((c) => !c.concept || best.get(c.concept) === c);
}

/**
 * The completed word as the person writes it: their own letters (accents, capitals), then the
 * rest, as written when folding keeps its length ("for" → "forêt"), else as folded.
 */
function asTyped(partialRaw: string, completed: string): string {
  const p = fold(partialRaw);
  const f = fold(completed);
  if (!f.startsWith(p) || p.length > partialRaw.length) return completed;
  return partialRaw + (f.length === completed.length ? completed.slice(p.length) : f.slice(p.length));
}

// ---- concept names as written --------------------------------------------------------------------

/**
 * The vocabulary is folded ("foret", "bar"): the usual names that need their accents back, in the
 * languages the questions are written in.
 */
const WRITTEN: Partial<Record<VocabLang, Record<string, string>>> = {
  fr: {
    chevre: "chèvre", elephant: "éléphant", meduse: "méduse", lezard: "lézard", creature: "créature", "grand-mere": "grand-mère",
    superheros: "super-héros", samourai: "samouraï", egypte: "Égypte", demon: "démon", fantome: "fantôme", "dessin anime": "dessin animé",
    meme: "mème", batiment: "bâtiment", chateau: "château", eglise: "église", musee: "musée", riviere: "rivière", desert: "désert",
    foret: "forêt", meteo: "météo", ete: "été", velo: "vélo", fusee: "fusée", telephone: "téléphone", isometrique: "isométrique",
    retro: "rétro", noel: "Noël", "etats-unis": "États-Unis", bresil: "Brésil", france: "France", suisse: "Suisse", espagne: "Espagne",
    italie: "Italie", slovaquie: "Slovaquie", inde: "Inde", japon: "Japon", chine: "Chine", singapour: "Singapour", halloween: "Halloween",
  },
  de: {
    bar: "bär", lowe: "löwe", schildkrote: "schildkröte", grossmutter: "großmutter", portrat: "porträt", ausserirdischer: "außerirdischer",
    konig: "könig", agypten: "ägypten", damon: "dämon", strasse: "straße", gebaude: "gebäude", brucke: "brücke", wuste: "wüste",
    fruhling: "frühling", getrank: "getränk", suss: "süß",
  },
};

/** A word of the vocabulary as written in the reader's language. */
const written = (word: string, lang: VocabLang) => WRITTEN[lang]?.[word] ?? word;

/** A concept's usual name in the reader's language, as written. */
const label = (id: string, lang: VocabLang) => written(conceptLabel(id, lang), lang);

/**
 * A completed word for a suggestion row: as written when its accents are back ("cha" → "château",
 * a capital typed kept), else the letters typed and the rest (asTyped).
 */
function shownWord(partialRaw: string, word: string, lang: VocabLang): string {
  const w = written(word, lang);
  if (w === word) return asTyped(partialRaw, word);
  return /^\p{Lu}/u.test(partialRaw) ? capitalize(w) : w;
}

const capitalize = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

// ---- questions --------------------------------------------------------------------------------------

/**
 * Concepts that are not things one posts "a first" of or counts artworks of: styles and moods,
 * parts of a person, the weather, seasons, holidays, countries and topics. Searches, not subjects:
 * "who posted the first hair?", "…the first crypto?", "…the first switzerland?" are nonsense.
 */
const NOT_SUBJECTS = new Set([
  "text", "abstract", "isometric", "retro", "cyberpunk", "fantasy", "horror", "cute", "video game",
  "face", "eyes", "hair", "beard", "tattoo", "shirtless",
  "music", "money", "crypto", "water", "sky", "space", "weather", "night", "underwater",
  "halloween", "christmas", "birthday", "winter", "autumn", "summer", "spring",
  "egypt", "japan", "china", "switzerland", "spain", "france", "italy", "slovakia", "india", "usa", "brazil", "singapore",
  "travel", "love", "work", "video call",
]);

/** Questions about a subject that the box sends to /ask (checked by test/suggest.test.ts). English and French. */
const ASK_TEMPLATES: Partial<Record<VocabLang, Array<(s: string) => string>>> = {
  en: [(s) => `who posted the first ${s}?`, (s) => `how many ${s} artworks?`],
  fr: [(s) => `qui a posté la première œuvre de ${s} ?`, (s) => `combien d'œuvres de ${s} ?`],
};

/** Questions without a subject, per language. */
const GENERIC_QUESTIONS: Partial<Record<VocabLang, string[]>> = {
  en: ["who is the most active artist?", "what's the latest artwork?", "what is the most liked artwork?"],
  fr: ["quel est l'artiste le plus actif ?", "quelle est la dernière œuvre ?", "quelle est l'œuvre la plus aimée ?"],
  de: ["wer ist der aktivste Künstler?", "was ist das neueste Bild?"],
};

/** Platform questions offered as examples when the documentation can answer them. */
const HELP_QUESTIONS: Partial<Record<VocabLang, string[]>> = {
  en: ["how do I mint an artwork?", "what are the fees?", "how do royalties work?", "what is PXS?", "how do I sell my art?"],
  fr: ["comment minter une œuvre ?", "quels sont les frais ?", "comment fonctionnent les royalties ?", "qu'est-ce que PXS ?"],
  de: ["was ist PXS?", "welche Gebühren gibt es?", "wie funktionieren Lizenzgebühren?"],
};

/**
 * Question openers (folded) that a subject completes, with the words after it. `plural`: the
 * subject reads in the plural there ("combien de chats ?", "wie viele Katzen?").
 */
const OPENERS: Array<{ open: string; tail: string; lang: VocabLang; plural?: boolean }> = [
  { open: "who posted the first ", tail: "?", lang: "en" },
  { open: "who posted the latest ", tail: "?", lang: "en" },
  { open: "who posted the last ", tail: "?", lang: "en" },
  { open: "who posted the most ", tail: "?", lang: "en", plural: true },
  { open: "when was the first ", tail: " posted?", lang: "en" },
  { open: "what is the latest ", tail: "?", lang: "en" },
  { open: "what s the latest ", tail: "?", lang: "en" },
  { open: "what is the most liked ", tail: "?", lang: "en" },
  { open: "most liked ", tail: "?", lang: "en" },
  { open: "how many ", tail: " artworks?", lang: "en" },
  { open: "qui a poste le premier ", tail: " ?", lang: "fr" },
  { open: "qui a poste la premiere ", tail: " ?", lang: "fr" },
  { open: "qui a poste la premiere oeuvre de ", tail: " ?", lang: "fr" },
  { open: "combien d oeuvres de ", tail: " ?", lang: "fr" },
  { open: "combien de ", tail: " ?", lang: "fr", plural: true },
  { open: "wer hat das erste bild mit ", tail: " gepostet?", lang: "de" },
  { open: "wer hat die erste ", tail: " gepostet?", lang: "de" },
  { open: "wer hat den ersten ", tail: " gepostet?", lang: "de" },
  { open: "wer hat das erste ", tail: " gepostet?", lang: "de" },
  { open: "wie viele ", tail: "?", lang: "de", plural: true },
];

/** Openers proposed whole while one is being typed ("who po" → "who posted the first …?"), and the template they make. */
const OPENER_TEMPLATES: Array<{ open: string; lang: VocabLang; make: (s: string) => string }> = [
  { open: "who posted the first ", lang: "en", make: (s) => `who posted the first ${s}?` },
  { open: "how many ", lang: "en", make: (s) => `how many ${s} artworks?` },
  { open: "most liked ", lang: "en", make: (s) => `most liked ${s}?` },
  { open: "qui a poste la premiere oeuvre de ", lang: "fr", make: (s) => `qui a posté la première œuvre de ${s} ?` },
  { open: "combien d oeuvres de ", lang: "fr", make: (s) => `combien d'œuvres de ${s} ?` },
];

/** The plural among the concept's names in that language ("chats" for "chat", "katzen" for "katze"), else a regular one. */
function pluralOf(word: string, concept: string | undefined, lang: VocabLang): string {
  const names = (concept && conceptDef(concept)?.aliases[lang]) || [];
  for (const suffix of ["s", "es", "x", "n", "en", "e", "er", "nen"]) {
    if (names.includes(word + suffix)) return word + suffix;
  }
  if (/[sxz]$/.test(word)) return word;
  return lang === "en" || lang === "fr" ? `${word}s` : word;
}

// ---- titles, documentation, popular searches ---------------------------------------------------------

const ftsToken = (w: string) => fold(w).replace(/[^\p{L}\p{N}]+/gu, "");

/** What /search shows by default (sql.ts, nsfw=exclude): suggestions never show more. */
const SAFE = "p.nsfw = 0 AND COALESCE(a.ai_nsfw, 0) < 0.7";

type TitleRow = { id: number; author: string; permlink: string; category: string | null; type: "artwork" | "blog"; title: string; orig: string | null };

/** The post a title suggestion names, with its image's path on this Worker (as /search's images.original). */
const postOf = (r: TitleRow): NonNullable<Suggestion["post"]> => ({
  id: r.id,
  author: r.author,
  permlink: r.permlink,
  category: r.category ?? null,
  type: r.type,
  image: r.orig ? `/img/${r.orig}` : null,
});

/**
 * The column(s) restricted to every finished word and the word being typed, as an FTS5
 * expression. A prefix of one or two letters matches a good part of the index, all of it scored
 * for bm25 (the FTS table has no prefix index): it is left out of the expression and checked on
 * the rows instead (`short`), and with no finished word there is nothing to look up (null).
 */
function titleQuery(words: string[], partial: string, column: string): { expr: string; short: string | null } | null {
  const toks = words.map(ftsToken).filter(Boolean).map((t) => `"${t}"`);
  const p = ftsToken(partial);
  const short = p && p.length < 3 && !hasCjk(p) ? p : null;
  if (p && !short) toks.push(`"${p}"*`);
  if (!toks.length) return null;
  return { expr: `${column} : (${toks.join(" ")})`, short };
}

/** A row's text has a word that starts with the short prefix typed (see titleQuery). */
const hasWordStarting = (text: string, short: string | null) => !short || tokens(text, { keepHyphenated: false }).some((t) => t.startsWith(short));

async function titleMatches(env: Env, words: string[], partial: string, limit: number): Promise<Suggestion[]> {
  const tq = titleQuery(words, partial, "title");
  if (!tq) return [];
  const rows =
    (
      await env.DB.prepare(
        `SELECT p.id AS id, p.author AS author, p.permlink AS permlink, p.category AS category, p.type AS type, p.title AS title, a.r2_orig_key AS orig
         FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid LEFT JOIN artworks a ON a.post_id = p.id
         WHERE posts_fts MATCH ?1 AND p.deleted = 0 AND p.title != '' AND ${SAFE}
         ORDER BY bm25(posts_fts), p.net_votes DESC LIMIT ?2`,
      )
        .bind(tq.expr, tq.short ? 40 : limit * 3)
        .all<TitleRow>()
        .catch(() => ({ results: [] as TitleRow[] }))
    ).results ?? [];
  const seen = new Set<string>();
  const out: Suggestion[] = [];
  for (const r of rows) {
    const title = String(r.title).replace(/\s+/g, " ").trim().slice(0, 80);
    const key = norm(title);
    if (!title || seen.has(key) || !hasWordStarting(title, tq.short)) continue;
    seen.add(key);
    out.push({ text: title, kind: "title", route: "search", post: postOf(r) });
    if (out.length >= limit) break;
  }
  return out;
}

const DOCS_SEEN = new WeakMap<object, { at: number; yes: boolean }>();

/** Whether the documentation is indexed at all (asked once per isolate and ten minutes). */
async function docsExist(env: Env): Promise<boolean> {
  const hit = DOCS_SEEN.get(env.DB);
  if (hit && Date.now() - hit.at < 10 * 60_000) return hit.yes;
  const yes = !!(await env.DB.prepare("SELECT 1 AS x FROM doc_chunks LIMIT 1").first().catch(() => null));
  DOCS_SEEN.set(env.DB, { at: Date.now(), yes });
  return yes;
}

async function docSections(env: Env, words: string[], partial: string, limit: number): Promise<Suggestion[]> {
  const tq = titleQuery(words, partial, "{title heading}");
  if (!tq) return [];
  const rows =
    (
      await env.DB.prepare(
        `SELECT c.path AS path, c.title AS title, c.heading AS heading, c.anchor AS anchor
         FROM doc_chunks_fts f JOIN doc_chunks c ON c.id = f.rowid
         WHERE doc_chunks_fts MATCH ?1 ORDER BY bm25(doc_chunks_fts) LIMIT ?2`,
      )
        .bind(tq.expr, tq.short ? 40 : 12)
        .all<{ path: string; title: string; heading: string; anchor: string }>()
        .catch(() => ({ results: [] as any[] }))
    ).results ?? [];
  const ref = repoRef(env);
  const seen = new Set<string>();
  const out: Suggestion[] = [];
  for (const r of rows) {
    const key = `${r.path}#${r.heading}`;
    if (seen.has(key) || !hasWordStarting(`${r.title} ${r.heading}`, tq.short)) continue;
    seen.add(key);
    const text = r.heading ? `${r.title} › ${r.heading}` : r.title;
    out.push({
      text,
      kind: "help",
      route: "help",
      query: text.replace(/\s*›\s*/g, " "),
      source: { title: r.title, heading: r.heading, url: ref ? blobUrl(ref, r.path, r.anchor) : r.path },
    });
    if (out.length >= limit) break;
  }
  return out;
}

/** Shown to others: words and a few signs only; no address, e-mail, link or long number. */
const SHAREABLE = (q: string) =>
  q.length >= 2 && q.length <= 80 && /^[\p{L}\p{N}\s#@'’?!.,:&-]+$/u.test(q) && !/\d{5,}/.test(q) && !/@[\p{L}\p{N}-]+\./u.test(q) && !/\b(?:https?|www)\b/i.test(q);

/** A word known without asking the database: a function word, a colour, a small number, or the name of a concept safe artworks show. */
function knownWord(t: string, counts: Counts): boolean {
  if (isStopword(t) || !!COLOR_WORDS[t] || /^\d{1,4}$/.test(t)) return true;
  const a = aliasesStartingWith(t, 1)[0];
  return !!a && a.alias === t && (counts.all.get(a.concept) ?? 0) > 0;
}

// ---- popular searches (off unless SUGGEST_POPULAR=on) ---------------------------------------------
//
// A search shown to everybody as "popular" must be what several people looked for, not what one
// person (or one script) repeated. The search log cannot tell people apart (and one search in the
// box is two or three requests), so /search records, for searches that found something in the
// default safe mode, one row per (search, client, day) in query_people. `client` is six hex
// characters of a SHA-256 of the day's salt and the address (an IPv6 address by its /64: one
// household): it tells people apart within a day, never across days, and never who they are —
// about 256 IPv4 addresses share each value, and the salt is random, kept in `settings` for that
// day and the next, then deleted, after which reading the database cannot trace a value back
// (the account holder could, within D1 Time Travel's restore window). A popular search needs
// three people on the same day, on two days at least within 30 days, words the index knows, and
// nothing that looks like an address, an e-mail or a long number. Even so, three addresses can
// plant a phrase of corpus words (a user name and a word, say), which is why the operator turns
// it on.

export const popularEnabled = (env: Env) => String(env.SUGGEST_POPULAR ?? "").toLowerCase() === "on";

const POPULAR_PEOPLE = 3;
const POPULAR_DAYS = 2;
const POPULAR_WINDOW_DAYS = 30;

const SALTS = new WeakMap<object, Map<number, string>>();
export const SALT_PREFIX = "suggest:salt:";

/** The day's random salt: made by the first request of the day (one row, whoever writes first wins). */
async function daySalt(env: Env, day: number): Promise<string> {
  let known = SALTS.get(env.DB);
  if (!known) SALTS.set(env.DB, (known = new Map()));
  const hit = known.get(day);
  if (hit) return hit;
  const k = `${SALT_PREFIX}${day}`;
  const fresh = [...crypto.getRandomValues(new Uint8Array(16))].map((b) => b.toString(16).padStart(2, "0")).join("");
  await env.DB.prepare("INSERT OR IGNORE INTO settings (k, v) VALUES (?, ?)").bind(k, fresh).run();
  const v = (await env.DB.prepare("SELECT v FROM settings WHERE k = ?").bind(k).first<{ v: string }>())?.v ?? fresh;
  if (known.size > 4) known.clear();
  known.set(day, v);
  return v;
}

/** An IPv6 address by its /64 (a household's devices rotate the rest); IPv4 as it is (also mapped: "::ffff:a.b.c.d"). */
export function addressKey(ip: string): string {
  const mapped = ip.trim().match(/^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i);
  if (mapped) return mapped[1];
  if (!ip.includes(":")) return ip.trim();
  const [head, tail] = ip.trim().toLowerCase().split("::");
  const a = head ? head.split(":") : [];
  const b = tail !== undefined && tail ? tail.split(":") : [];
  const groups = tail === undefined ? a : [...a, ...new Array(Math.max(0, 8 - a.length - b.length)).fill("0"), ...b];
  return groups.slice(0, 4).map((g) => (g || "0").replace(/^0+(?=.)/, "")).join(":") + "::/64";
}

async function clientTag(env: Env, ip: string, day: number): Promise<string> {
  const salt = await daySalt(env, day);
  const bytes = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(`${salt}|${addressKey(ip)}`)));
  return [...bytes.slice(0, 3)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Remember who ran a search (see above), for a page-one search with text in the default safe mode
 * that found something. Called by /search after answering (waitUntil); never throws.
 */
export async function recordSearcher(env: Env, q: string, ip: string | null | undefined): Promise<void> {
  if (!popularEnabled(env) || !ip || ip === "unknown") return;
  const shown = String(q ?? "").toLowerCase().replace(/\s+/g, " ").trim();
  if (!SHAREABLE(shown)) return;
  const day = Math.floor(now() / 86400);
  try {
    await env.DB.prepare("INSERT OR IGNORE INTO query_people (qn, q, client, day) VALUES (?, ?, ?, ?)")
      .bind(norm(shown).trim(), shown, await clientTag(env, ip, day), day)
      .run();
  } catch {
    // best effort: a lost row only delays a popular search
  }
}

const POPULAR = new WeakMap<object, { at: number; list: Array<{ q: string; n: number }> }>();
const POPULAR_KEY = "suggest:popular:v2";

/** The popular searches (see above), most people first. Cached half an hour (KV) and five minutes in the isolate. */
async function popularList(env: Env): Promise<Array<{ q: string; n: number }>> {
  if (!popularEnabled(env)) return [];
  const hit = POPULAR.get(env.DB);
  if (hit && Date.now() - hit.at < 5 * 60_000) return hit.list;
  let list = (await env.CACHE.get(POPULAR_KEY, "json").catch(() => null)) as Array<{ q: string; n: number }> | null;
  if (!Array.isArray(list)) {
    list = await buildPopular(env);
    await env.CACHE.put(POPULAR_KEY, JSON.stringify(list), { expirationTtl: 1800 }).catch(() => {});
  }
  POPULAR.set(env.DB, { at: Date.now(), list });
  return list;
}

async function buildPopular(env: Env): Promise<Array<{ q: string; n: number }>> {
  const since = Math.floor(now() / 86400) - POPULAR_WINDOW_DAYS;
  const rows =
    (
      // people are told apart within a day only (the salt changes): three on one day, two days
      await env.DB.prepare(
        `WITH daily AS (SELECT qn, day, MIN(q) AS q, COUNT(DISTINCT client) AS people FROM query_people WHERE day >= ?1 GROUP BY qn, day)
         SELECT qn, MIN(q) AS q, MAX(people) AS people FROM daily
         GROUP BY qn HAVING MAX(people) >= ?2 AND COUNT(*) >= ?3
         ORDER BY people DESC, qn LIMIT 400`,
      )
        .bind(since, POPULAR_PEOPLE, POPULAR_DAYS)
        .all<{ qn: string; q: string; people: number }>()
        .catch(() => ({ results: [] as Array<{ qn: string; q: string; people: number }> }))
    ).results ?? [];
  const counts = await conceptCounts(env);
  const cands = rows.map((r) => ({ q: String(r.q), n: Number(r.people) || 0 })).filter((r) => SHAREABLE(r.q));
  const unknown = new Set<string>();
  for (const r of cands) for (const t of tokens(r.q, { keepHyphenated: false })) if (!knownWord(t, counts)) unknown.add(t);
  // the other words: carried by at least one post /search shows by default
  const safe = await safeCounts(env, [...unknown], 1);
  const out: Array<{ q: string; n: number }> = [];
  for (const r of cands) {
    const toks = tokens(r.q, { keepHyphenated: false });
    if (!toks.length || !toks.every((t) => knownWord(t, counts) || (safe.get(t) ?? 0) > 0)) continue;
    out.push(r);
    if (out.length >= 200) break;
  }
  return out;
}

/** Popular searches that extend the text typed (a trailing space included). */
async function popular(env: Env, raw: string, limit: number): Promise<Array<{ q: string; n: number }>> {
  const prefix = norm(raw);
  const typed = prefix.trim();
  return (await popularList(env)).filter((p) => {
    const k = norm(p.q);
    return k.startsWith(prefix) && k.trim() !== typed;
  }).slice(0, limit);
}

// ---- suggestions for typed text --------------------------------------------------------------------

export async function suggestFor(env: Env, text: string, opts: { lang?: string | null; limit?: number } = {}): Promise<SuggestResponse> {
  const t0 = Date.now();
  const raw = suggestText(text);
  const limit = Math.max(1, Math.min(12, Math.trunc(opts.limit ?? 8)));
  const typed = norm(raw).trim();
  const lang = vocabLang(opts.lang);
  const empty: SuggestResponse = { q: raw, lang, completion: null, suggestions: [], took_ms: 0 };
  if (!typed) return { ...empty, took_ms: Date.now() - t0 };

  // the word being typed (none after a space), and what comes before it, as typed
  const finished = /\s$/.test(raw);
  const rawPartial = finished ? "" : (raw.match(/[^\s]+$/)?.[0] ?? "");
  const rawHead = raw.slice(0, raw.length - rawPartial.length);
  const partialWord = rawPartial.replace(/^[^\p{L}\p{N}]+/u, "");
  const lead = rawPartial.slice(0, rawPartial.length - partialWord.length); // "#", "@", "(" before the word
  const headWords = norm(rawHead).split(" ").filter(Boolean);
  const normText = norm(raw);

  const counts = await conceptCounts(env);
  const out: Suggestion[] = [];
  const seen = new Set<string>([typed]);
  const push = (s: Suggestion) => {
    const key = norm(s.text).trim();
    if (!key || seen.has(key)) return;
    seen.add(key);
    out.push(s);
  };

  // A question whose subject is being typed ("who posted the first dra").
  const opener = OPENERS.filter((o) => normText.startsWith(o.open)).sort((a, b) => b.open.length - a.open.length)[0];
  // Any other question begun ("how do", "what is th"): its words are not completed from the
  // vocabulary ("how do" is not "how downtown"); whole questions are proposed instead (step 2).
  const questionLike = !opener && headWords.length > 0 && QUESTION_START.test(normText);
  // A function word after others ("cat in") is most likely finished: it is not completed.
  const functionWord = headWords.length > 0 && isStopword(fold(partialWord));
  const completing = !!partialWord && !lead && !questionLike && !functionWord;
  const wordLang = opener?.lang ?? lang;

  // What does not depend on the rest, at once: the word's completions, popular searches, titles,
  // documentation sections (one round trip of latency, not four).
  const more = typed.length >= 2;
  const partialForFts = lead ? "" : partialWord;
  const [completed, pops, titles, sections] = await Promise.all([
    completing ? completeWord(env, partialWord, counts, wordLang) : Promise.resolve([] as WordCompletion[]),
    more ? popular(env, raw, 2) : Promise.resolve([] as Array<{ q: string; n: number }>),
    more ? titleMatches(env, headWords, partialForFts, 2) : Promise.resolve([] as Suggestion[]),
    more ? docsExist(env).then((yes) => (yes ? docSections(env, headWords, partialForFts, 2) : [])) : Promise.resolve([] as Suggestion[]),
  ]);
  // after other words, only the words some post carries with them (inside a question, its words
  // are the question's)
  const completions = headWords.length && !opener ? await withHead(env, headWords, completed.slice(0, 8)) : completed;
  const exact = completions.find((c) => c.word === fold(partialWord));
  // one name per concept; the word typed, when it already names its concept well, is not completed
  // into another name of it ("dragon" does not offer "dragons"; "drago" offers "dragon")
  const names = byConcept(completions, wordLang, exact);
  const longer = names.filter((c) => c !== exact && !(exact?.concept && c.concept === exact.concept && goodName(exact, wordLang)));

  // 1. a question being typed: complete its subject ("who posted the first dra" → "…dragon?")
  if (opener && partialWord) {
    const subjects = exact && names.includes(exact) ? [exact, ...names.filter((c) => c !== exact)] : names;
    for (const c of subjects.slice(0, 4)) {
      const w = opener.plural ? pluralOf(c.word, c.concept, opener.lang) : c.word;
      let word = shownWord(partialWord, w, opener.lang);
      if (opener.lang === "de") word = capitalize(word);
      push({ text: `${rawHead}${lead}${word}${opener.tail}`, kind: "question", route: "ask", n: c.n });
    }
  }

  // 2. a question begun: the whole questions that start with what is typed: an opener with the
  //    subjects the artists post most ("who po" → "who posted the first cat?"), the general
  //    questions, and the platform questions the documentation answers ("how do" → "how do I
  //    mint an artwork?")
  if (!opener && typed.length >= 3) {
    const top = topSubjects(counts, 2);
    for (const o of OPENER_TEMPLATES) {
      if (!o.open.startsWith(normText) || o.open === normText) continue;
      for (const id of top) push({ text: o.make(label(id, o.lang)), kind: "question", route: "ask", n: counts.all.get(id) });
      if (out.length >= 3) break;
    }
    for (const q of GENERIC_QUESTIONS[lang] ?? []) if (norm(q).startsWith(normText)) push({ text: q, kind: "question", route: "ask" });
    for (const q of await answerableHelp(env, lang)) if (norm(q).startsWith(normText)) push({ text: q, kind: "help", route: "help" });
  }

  // 3. the word being typed, completed (fewer when the word is already one the posts carry); then
  //    questions about its subject when it is clear what the subject is (English, French)
  if (completing && !opener) {
    // a word that is already a subject or the artists' own: only the artists' words and the usual
    // names extend it ("cat" is not "catastrophically", nor "cathedral" from the AI descriptions)
    const strong = !!exact && (!!exact.main || (exact.own ?? 0) > 0);
    const pool = strong ? longer.filter((c) => tier(c) >= 1) : longer;
    for (const c of pool.slice(0, strong ? 2 : 4)) push({ text: `${rawHead}${shownWord(partialWord, c.word, lang)}`, kind: "complete", route: "search", n: c.n });
    const subject = headWords.length ? null : subjectOf(exact, longer, partialWord, lang);
    const templates = ASK_TEMPLATES[lang];
    if (subject?.concept && templates) {
      const l = label(subject.concept, lang);
      for (const make of templates) push({ text: make(l), kind: "question", route: "ask", n: subject.n });
    }
  }

  // 4. popular searches, titles, documentation sections
  for (const p of pops) push({ text: p.q, kind: "popular", route: "search", n: p.n });
  for (const t of titles) push(t);
  for (const d of sections) push(d);

  // 5. nothing at all: perhaps a typo
  if (out.length < 2 && typed.length >= 4 && !questionLike) {
    const found = await spellSuggest(env.DB, raw).catch(() => []);
    // only into words that posts /search shows by default carry
    const safe = await safeCounts(env, found.map((f) => f.to), 1);
    const fixes = found.filter((f) => (safe.get(f.to) ?? 0) > 0);
    if (fixes.length) {
      const fixed = norm(raw)
        .trim()
        .split(" ")
        .map((w) => fixes.find((f) => f.from === w)?.to ?? w)
        .join(" ");
      push({ text: fixed, kind: "correction", route: "search" });
    }
  }

  const suggestions = out.slice(0, limit);
  // the box shows the rest as ghost text after the letters typed, which stay as typed (accents
  // aside: "cha" is ghosted into "cha|teau" by "château")
  const folded = fold(raw);
  const ahead = suggestions.find((s) => s.text.length > raw.length && fold(s.text.slice(0, raw.length)) === folded);
  const completion = ahead ? raw + ahead.text.slice(raw.length) : null;
  return { q: raw, lang, completion, suggestions, took_ms: Date.now() - t0 };
}

/**
 * The subject of the questions offered for a word, or none: the word typed when it is its
 * concept's usual name in the reader's language (or a plural of it: "dragons"), else, from three
 * letters, the only usual name the word can still become when it is the first completion ("dra"
 * → dragon). Never the concept another of its names belongs to: "helmet" names hats and
 * "dragonfly" insects, and a question about hats or insects is not what was typed.
 */
function subjectOf(exact: WordCompletion | undefined, longer: WordCompletion[], partialWord: string, lang: VocabLang): WordCompletion | null {
  if (exact?.concept && NOT_SUBJECTS.has(exact.concept)) return null;
  if (exact?.concept) {
    const usual = conceptLabel(exact.concept, lang);
    const plural = !!exact.langs?.includes(lang) && exact.word.startsWith(usual) && exact.word.length <= usual.length + 2;
    return exact.main || plural ? exact : null;
  }
  if (fold(partialWord).length < 3) return null;
  const usual = longer.filter((c) => c.concept && c.main);
  return usual.length === 1 && longer[0] === usual[0] && !NOT_SUBJECTS.has(usual[0].concept!) ? usual[0] : null;
}

const HELP_OK = new WeakMap<object, Map<VocabLang, { at: number; list: string[] }>>();

/** The platform questions the documentation answers, in the reader's language (checked once per isolate and ten minutes). */
async function answerableHelp(env: Env, lang: VocabLang): Promise<string[]> {
  if (!(HELP_QUESTIONS[lang] ?? []).length || !(await docsExist(env))) return [];
  let byLang = HELP_OK.get(env.DB);
  if (!byLang) HELP_OK.set(env.DB, (byLang = new Map()));
  const hit = byLang.get(lang);
  if (hit && Date.now() - hit.at < 10 * 60_000) return hit.list;
  const list: string[] = [];
  for (const q of HELP_QUESTIONS[lang] ?? []) {
    const best = (await lexicalDocs(env, q, 5).catch(() => ({ hits: [] as Array<{ score: number }> }))).hits[0]?.score ?? 0;
    if (best >= DOCS_KNOWS) list.push(q);
  }
  byLang.set(lang, { at: Date.now(), list });
  return list;
}

/**
 * The subjects the most artworks are about in the artists' own words (tags, titles,
 * descriptions; the AI descriptions only when nobody tagged anything), leaves only ("cat", not
 * "animal").
 */
function topSubjects(counts: Counts, k: number): string[] {
  const by = counts.human.size ? counts.human : counts.all;
  const parents = parentsIn(by);
  return [...by.entries()]
    .filter(([id, n]) => n >= 1 && !STRUCTURAL.has(id) && !NOT_SUBJECTS.has(id) && !parents.has(id))
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .slice(0, k)
    .map(([id]) => id);
}

// ---- examples for the placeholder ----------------------------------------------------------------

const EXAMPLES_TTL = 30 * 60;

export async function suggestExamples(env: Env, langIn?: string | null): Promise<ExamplesResponse> {
  const t0 = Date.now();
  const lang = vocabLang(langIn);
  const key = `suggest:examples:v2:${lang}`;
  const hit = (await env.CACHE.get(key, "json").catch(() => null)) as Suggestion[] | null;
  if (Array.isArray(hit)) return { lang, examples: hit, took_ms: Date.now() - t0 };
  const examples = await buildExamples(env, lang);
  if (examples.length) await env.CACHE.put(key, JSON.stringify(examples), { expirationTtl: EXAMPLES_TTL }).catch(() => {});
  return { lang, examples, took_ms: Date.now() - t0 };
}

/** Things a colour describes ("black cat"); for a scene it reads as nonsense ("black landscape": a dark one). */
const COLOURED = new Set(["animal", "creature", "vehicle", "object", "accessory", "food", "plant", "flower", "character"]);
const takesColour = (id: string) => COLOURED.has(id) || ancestorsWithDepth(id).some((a) => COLOURED.has(a.id));

async function buildExamples(env: Env, lang: VocabLang): Promise<Suggestion[]> {
  const counts = await conceptCounts(env);
  const by = counts.human.size ? counts.human : counts.all;
  const subjects = topSubjects(counts, 8).filter((id) => (by.get(id) ?? 0) >= 2);
  const name = (id: string) => (lang === "de" ? capitalize(label(id, lang)) : label(id, lang));

  const searches: Suggestion[] = subjects.slice(0, 4).map((id) => ({ text: name(id), kind: "complete", route: "search", n: counts.all.get(id) }));
  // English only: the colour words of other languages agree with the noun
  if (lang === "en" && subjects.length) {
    for (const id of subjects.filter(takesColour).slice(0, 3)) {
      const row = await env.DB.prepare(
        `SELECT a.primary_color AS color, COUNT(*) AS n FROM artwork_concepts ac
         JOIN artworks a ON a.post_id = ac.post_id JOIN posts p ON p.id = ac.post_id
         WHERE p.deleted = 0 AND ${SAFE} AND ac.concept = ?1 AND a.primary_color IS NOT NULL
         GROUP BY a.primary_color ORDER BY n DESC LIMIT 1`,
      )
        .bind(id)
        .first<{ color: string; n: number }>()
        .catch(() => null);
      if (row?.color && row.n >= 2) {
        searches.splice(1, 0, { text: `${row.color} ${name(id)}`, kind: "complete", route: "search", n: row.n });
        break;
      }
    }
  }

  const templates = ASK_TEMPLATES[lang];
  const questions: Suggestion[] = [];
  if (templates) {
    subjects.slice(0, 3).forEach((id, i) => {
      const make = templates[i % templates.length];
      questions.push({ text: make(name(id)), kind: "question", route: "ask", n: counts.all.get(id) });
    });
  }
  for (const q of GENERIC_QUESTIONS[lang] ?? []) questions.push({ text: q, kind: "question", route: "ask" });

  const titleRows =
    (
      await env.DB.prepare(
        `SELECT p.id AS id, p.author AS author, p.permlink AS permlink, p.category AS category, p.type AS type, p.title AS title, a.r2_orig_key AS orig
         FROM posts p LEFT JOIN artworks a ON a.post_id = p.id
         WHERE p.deleted = 0 AND p.type = 'artwork' AND ${SAFE} AND length(p.title) BETWEEN 4 AND 32
         ORDER BY p.net_votes DESC, p.created DESC LIMIT 4`,
      )
        .all<TitleRow>()
        .catch(() => ({ results: [] as any[] }))
    ).results ?? [];
  const titles: Suggestion[] = titleRows.map((r) => ({
    text: String(r.title).trim(),
    kind: "title",
    route: "search",
    post: postOf(r),
  }));

  const help: Suggestion[] = (await answerableHelp(env, lang)).slice(0, 2).map((q) => ({ text: q, kind: "help", route: "help" }));

  // a search, a question, a title, a help question, … in turn
  const lanes = [searches, questions, titles, help];
  const out: Suggestion[] = [];
  const seen = new Set<string>();
  for (let round = 0; out.length < 12 && lanes.some((l) => l.length); round++) {
    for (const lane of lanes) {
      const s = lane.shift();
      if (!s) continue;
      const k = norm(s.text).trim();
      if (!k || seen.has(k)) continue;
      seen.add(k);
      out.push(s);
      if (out.length >= 12) break;
    }
    if (round > 24) break;
  }
  return out;
}

/** Where the box would send a text (the route the server's router picks), for tests and diagnostics. */
export async function routeOf(text: string, authors: Set<string> = new Set()): Promise<string> {
  const plan = planQuery(text, { mode: "ask", authors });
  return (await routeQuery(text, plan, { docsScore: async () => 0, isTitle: async () => false })).route;
}
