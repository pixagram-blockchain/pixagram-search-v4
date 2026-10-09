// Rule-based query planner. Turns free text into structure the engine can execute:
// intent (search, first, last, count, top, compare, similar), the subject (residual text and
// canonical concepts), filters (authors, tags, colours, tones, dates) and the requested output.
//
// /search uses the plan as *hints* (soft ranking features, extra candidate legs): the text legs
// still see the query as typed. /ask executes the plan; there the LLM may refine it (llm-planner),
// but the LLM only plans: the answer always comes from retrieved evidence.

import { matchConcepts, type ConceptMatch } from "../concepts";
import { fold, guessLang, isStopword, type Lang } from "../lib/text";
import {
  COLOR_WORDS,
  INTENT_PATTERNS,
  MONTHS,
  OBJECT_WORDS,
  ORIENTATION_WORDS,
  OUTPUT_PATTERNS,
  QUESTION_FILLER,
  QUESTION_FILLER_V4,
  TIME_FILLER_V4,
  REWRITES_V4,
  TONE_WORDS,
  type Intent,
  type Output,
  type Tone,
} from "./lexicon";

export interface PlanFilters {
  authors?: string[];
  tags?: string[];
  colors?: string[]; // has_color (any bucket with weight >= 0.15)
  background?: string[]; // named colour of the backdrop
  from?: number;
  to?: number;
  orientation?: Array<"portrait" | "landscape" | "square">;
  tones?: Tone[];
}

export interface QueryPlan {
  intent: Intent;
  /** the phrases that set the intent ("first", "how many"); empty when it was inferred ("who posted a cat?") */
  intentWords: string[];
  /** count questions about people ("how many artists…"): distinct authors instead of posts */
  countOf?: "authors";
  object: "artwork" | "blog" | "any";
  /** the query as typed (quotes, @ and # removed) */
  text: string;
  /** the subject once question scaffolding, filters and intent words are removed ("cat") */
  residual: string;
  concepts: string[];
  conceptMatches: ConceptMatch[];
  lexicalTerms: string[];
  phrases: string[];
  filters: PlanFilters;
  /** soft signals for /search ranking */
  hints: {
    colors: string[];
    background: string[];
    tones: Tone[];
    orientation: Array<"portrait" | "landscape" | "square">;
    /** every author the query mentions or names */
    authors: string[];
    /** named as the author in /search ("cat by laura"): full author feature, not a filter */
    namedAuthors: string[];
  };
  temporal?: { operator: "first" | "last" | "before" | "after" | "between"; field: "image" | "created" };
  sort?: "votes" | "payout" | "newest" | "oldest";
  output: Output;
  similarTo?: { id?: number; author?: string; permlink?: string };
  lang: Lang;
  confidence: number;
  source: "rules" | "llm" | "rules+llm";
  notes: string[];
}

export interface PlanContext {
  /** known author names (lowercase), to recognise "by laura" and bare author names */
  authors?: Set<string>;
  /** "search": hints only; "ask": filters and intents */
  mode: "search" | "ask";
  now?: number;
  /**
   * v4's /ask planning: "¿" and "¡" are punctuation, demonstratives are fillers. Off for v3's
   * /ask (mode=v3), /search and the suggestions, which plan exactly as v3 did.
   */
  v4?: boolean;
}

const DAY = 86400;

/** Connective words of questions that are not stopwords in the index sense. */
const ASK_FILLER = new Set(
  "contain,contains,containing,featuring,feature,features,include,includes,including,having,showing,depicting,depict,represent,representing,there,ever,anything,something,someone,somebody,contient,contenant,avec,representant,montrant,enthalt,enthalten,mit,zeigt,gibt,existe,existent,existen,esiste,esistono,contiene,contengono,mostra,raffigura".split(","),
);

const AMBIGUOUS_MONTHS = new Set(["mars", "sept", "may", "mai", "august"]);

/** Explicit "similar to <id>" phrases (folded text); a "#" makes "like" explicit too. */
const SIMILAR_ID =
  /\b(?:(?:similar|similaire|similaires|semblable|semblables|parecido|parecida|parecidos|parecidas|simile|simili) (?:to|a)|ahnlich (?:wie|zu)|looks? like|duplicates? of|copies of|copy of|reposts? of|doublons? de|kopien von|duplicados de|duplicati di) #?(\d{1,9})\b|\b(?:like|comme|wie|como|come) #(\d{1,9})\b/;

/** The noun counted by "how many …" (after an optional "different"). */
const COUNT_NOUN =
  /\b(?:how many|number of|count of|combien(?: de| d)?|nombre d(?:e|es|')?|wie viele|wieviele|anzahl(?: der| an| von)?|cuantos|cuantas|numero de|quanti|quante)\s+(?:(?:different|distinct|unique|differents|differentes|verschiedene[nr]?|unterschiedliche[nr]?|distintos|distintas|diversi|diverse)\s+)?([\p{L}-]+)/u;

/** People, in the languages of the planner (folded). */
const AUTHOR_NOUNS = new Set(
  (
    "artist,artists,author,authors,user,users,account,accounts,creator,creators,member,members,people,person,persons,painter,painters," +
    "artiste,artistes,auteur,auteurs,utilisateur,utilisateurs,membre,membres,createur,createurs,personnes,comptes,compte," +
    "kunstler,kunstlerin,kunstlerinnen,autor,autoren,autorin,nutzer,nutzerin,mitglied,mitglieder,personen,leute,konten," +
    "artista,artistas,autores,usuario,usuarios,miembro,miembros,creadores,personas,cuentas," +
    "artisti,autore,autori,utente,utenti,membri,creatori,persone,account"
  ).split(","),
);

function utcMidnight(t: number): number {
  return Math.floor(t / DAY) * DAY;
}

function monthRange(year: number, month: number): [number, number] {
  return [Date.UTC(year, month, 1) / 1000, Date.UTC(year, month + 1, 1) / 1000];
}

interface DateHit {
  from?: number;
  to?: number;
  operator?: "before" | "after" | "between";
  consumed: string[];
}

/** Dates in folded text: ISO dates, "(in) september [2026]", "15 september", relative words. */
export function parseDates(f: string, now: number, opts: { v4?: boolean } = {}): DateHit | null {
  const consumed: string[] = [];
  let from: number | undefined;
  let to: number | undefined;
  let operator: DateHit["operator"];
  const before = /\b(before|until|avant|jusqu ?a|vor|bis|antes de|prima del|prima di)\b/u.exec(f);
  const after = /\b(after|since|apres|depuis|nach|seit|despues de|desde|dopo|da)\b/u.exec(f);

  const iso = [...f.matchAll(/\b(\d{4})-(\d{2})-(\d{2})\b/g)];
  const dayMonth = /\b(\d{1,2})\.?\s+([a-z]+)\.?(?:\s+(\d{4}))?\b/u.exec(f);
  const monthDay = /\b([a-z]+)\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(\d{4}))?\b/u.exec(f);
  const monthOnly = /\b(?:in|en|im|a|au mois de|nel mese di)?\s*([a-z]+)(?:\s+(\d{4}))?\b/gu;
  const yearOf = (y: string | undefined, m: number): number => {
    if (y) return Number(y);
    const d = new Date(now * 1000);
    // the most recent such month that is not in the future
    return m > d.getUTCMonth() ? d.getUTCFullYear() - 1 : d.getUTCFullYear();
  };

  let point: [number, number] | null = null; // [start, end) of the mentioned day/month
  if (iso.length) {
    const t = Date.UTC(Number(iso[0][1]), Number(iso[0][2]) - 1, Number(iso[0][3])) / 1000;
    point = [t, t + DAY];
    consumed.push(iso[0][0]);
    if (iso.length > 1) {
      const t2 = Date.UTC(Number(iso[1][1]), Number(iso[1][2]) - 1, Number(iso[1][3])) / 1000;
      consumed.push(iso[1][0]);
      return { from: Math.min(t, t2), to: Math.max(t, t2) + DAY, operator: "between", consumed };
    }
  } else if (dayMonth && MONTHS[dayMonth[2]] !== undefined) {
    const m = MONTHS[dayMonth[2]];
    const t = Date.UTC(yearOf(dayMonth[3], m), m, Number(dayMonth[1])) / 1000;
    point = [t, t + DAY];
    consumed.push(dayMonth[0]);
  } else if (monthDay && MONTHS[monthDay[1]] !== undefined && Number(monthDay[2]) <= 31) {
    const m = MONTHS[monthDay[1]];
    const t = Date.UTC(yearOf(monthDay[3], m), m, Number(monthDay[2])) / 1000;
    point = [t, t + DAY];
    consumed.push(monthDay[0]);
  } else {
    // v4: v3's pattern lets the space before "in" start a match, so "in" is read as a word and
    // "posted in may" never sees "in may"; v4 takes the preposition with the month after it
    const months = opts.v4 ? [...f.matchAll(/(?:^|\s)((?:(?:in|en|im|a|au mois de|nel mese di)\s+)?([a-z]+)(?:\s+(\d{4}))?)(?=\s|$)/gu)].map((x) => Object.assign([x[1], x[2], x[3]] as unknown as RegExpMatchArray, { index: x.index })) : [...f.matchAll(monthOnly)];
    for (const mm of months) {
      const m = MONTHS[mm[1]];
      // Short abbreviations ("mar", "jan") and month names that are also ordinary words (the
      // planet Mars, French "sept" = seven, "may", "august") only count with a year or a
      // preposition: "in may", "en mars 2026", "au mois de septembre".
      const weak = mm[1].length <= 3 || AMBIGUOUS_MONTHS.has(mm[1]);
      if (m === undefined || (weak && !mm[2] && !/^(in|en|im|au mois de|nel mese di)\s/.test(mm[0].trim()))) continue;
      point = monthRange(yearOf(mm[2], m), m);
      consumed.push(mm[0].trim());
      break;
    }
  }
  const today = utcMidnight(now);
  const thisYear = new Date(now * 1000).getUTCFullYear();
  const thisMonth = new Date(now * 1000).getUTCMonth();
  const rel: Array<[RegExp, [number, number]]> = [
    [/\b(today|aujourd ?hui|heute|hoy|oggi|今日)\b/u, [today, today + DAY]],
    [/\b(yesterday|hier|gestern|ayer|ieri|昨日)\b/u, [today - DAY, today]],
    [/\b(this week|cette semaine|diese woche|esta semana|questa settimana)\b/u, [today - 6 * DAY, today + DAY]],
    [/\b(last week|la semaine derniere|la semaine passee|letzte woche|vorige woche|la semana pasada|la settimana scorsa)\b/u, [today - 13 * DAY, today - 6 * DAY]],
    [/\b(this month|ce mois(-ci)?|diesen monat|este mes|questo mese)\b/u, monthRange(thisYear, thisMonth)],
    [/\b(last month|le mois dernier|le mois passe|letzten monat|vorigen monat|el mes pasado|il mese scorso)\b/u, monthRange(thisYear, thisMonth - 1)],
    [/\b(this year|cette annee|dieses jahr|este ano|quest ?anno)\b/u, [Date.UTC(thisYear, 0, 1) / 1000, Date.UTC(thisYear + 1, 0, 1) / 1000]],
    [/\b(last year|l annee derniere|l annee passee|l an dernier|l an passe|letztes jahr|voriges jahr|el ano pasado|l anno scorso)\b/u, [Date.UTC(thisYear - 1, 0, 1) / 1000, Date.UTC(thisYear, 0, 1) / 1000]],
  ];
  if (!point) {
    for (const [re, range] of rel) {
      const m = re.exec(f);
      if (m) {
        point = range;
        consumed.push(m[0]);
        break;
      }
    }
  }
  if (!point) {
    const y = /\b(?:in|en|im)?\s*(20\d{2})\b/u.exec(f);
    if (y) {
      point = [Date.UTC(Number(y[1]), 0, 1) / 1000, Date.UTC(Number(y[1]) + 1, 0, 1) / 1000];
      consumed.push(y[0].trim());
    }
  }
  if (!point) return null;
  if (before && before.index < f.indexOf(consumed[0])) {
    to = point[0];
    operator = "before";
    consumed.push(before[0]);
  } else if (after && after.index < f.indexOf(consumed[0])) {
    // "after sept 10" starts on the 11th, "after september" in October; "since" includes the day
    from = /since|depuis|seit|desde/.test(after[0]) ? point[0] : point[1];
    operator = "after";
    consumed.push(after[0]);
  } else {
    from = point[0];
    to = point[1];
    operator = "between";
  }
  return { from, to, operator, consumed };
}

/** Remove whole words and phrases, never parts of longer words ("red" leaves "scared" alone). */
function strip(text: string, parts: string[]): string {
  let s = ` ${text} `;
  for (const p of [...new Set(parts.map((x) => x.trim()).filter(Boolean))].sort((a, b) => b.length - a.length)) {
    s = s.replace(new RegExp(`(?<= )${p.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?= )`, "gu"), " ");
  }
  return s.replace(/\s+/g, " ").trim();
}

/**
 * The query syntax: "@author" and "#tag" are filters, the rest is text ("quotes" are kept as
 * phrases). `text` is empty when the query was nothing but mentions and tags.
 */
export function splitQuerySyntax(raw: string): { mentions: string[]; hashtags: string[]; phrases: string[]; text: string } {
  const phrases = [...raw.matchAll(/"([^"]{2,80})"/g)].map((m) => m[1].trim());
  const mentions = [...raw.matchAll(/(?:^|\s)@([a-z0-9][a-z0-9.-]{1,31})\b/gi)].map((m) => m[1].toLowerCase());
  const hashtags = [...raw.matchAll(/(?:^|\s)#([\p{L}\p{N}_-]{2,64})/gu)].map((m) => m[1].toLowerCase());
  const text = raw
    .replace(/"([^"]*)"/g, "$1")
    .replace(/(?:^|\s)@[a-z0-9][a-z0-9.-]{1,31}\b/gi, " ")
    .replace(/(?:^|\s)#[\p{L}\p{N}_-]{2,64}/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
  return { mentions, hashtags, phrases, text };
}

export function planQuery(raw: string, ctx: PlanContext): QueryPlan {
  const now = ctx.now ?? Math.floor(Date.now() / 1000);
  const notes: string[] = [];
  // "similar to 42", "duplicates of 42", "like #42"; a bare "like 1999" is a title ("Party like 1999")
  const sim = SIMILAR_ID.exec(fold(raw));
  const similarId = sim ? [sim[0], sim[1] ?? sim[2]] : null;
  const syntax = splitQuerySyntax(raw);
  const { mentions, phrases, text } = syntax;
  // "similar to #42" names an artwork, not a tag
  const hashtags = syntax.hashtags.filter((t) => !(similarId && t === similarId[1]));
  const f0 = fold(text).replace(ctx.v4 ? /['\u2019?!.,;:()¿¡]+/g : /['\u2019?!.,;:()]+/g, " ").replace(/\s+/g, " ").trim();
  const f = ctx.v4 ? REWRITES_V4.reduce((t, [re, to]) => t.replace(re, to), f0).replace(TIME_FILLER_V4, " ").replace(/\s+/g, " ").trim() : f0;
  const lang = guessLang(raw);
  // Dates are read first. In "last week", "le mois dernier" or "letzte Woche" the word for "last"
  // names a period, not "the latest one", so intents are matched on the text without its dates.
  const dates = parseDates(f, now, { v4: ctx.v4 });
  const fq = dates ? strip(f, dates.consumed) : f;

  // ---- intent and output (questions) ------------------------------------------------------------
  let intent: Intent = "search";
  let sort: QueryPlan["sort"];
  let output: Output = "results";
  const intentWords: string[] = [];
  if (ctx.mode === "ask") {
    for (const p of INTENT_PATTERNS) {
      const m = p.re.exec(fq);
      if (!m) continue;
      if (intent === "search" || (intent === "find_first" && p.intent === "top")) {
        intent = p.intent ?? intent;
        if (p.output) output = p.output;
        if (p.sort) sort = p.sort;
      }
      intentWords.push(m[0]);
    }
    for (const p of OUTPUT_PATTERNS) {
      const m = p.re.exec(fq);
      if (m) {
        if (output === "results" || (p.output === "author" && intent !== "count")) output = p.output;
        break;
      }
    }
    if (output === "results") output = intent === "count" ? "count" : intent === "search" ? "results" : "post";
    if (intent === "search" && (output === "author" || output === "date")) intent = "find_first"; // "who posted a cat?" → the first one
  }
  // "how many artists…", "combien d'artistes…": people, not posts (the word right after "how many")
  let countOf: QueryPlan["countOf"];
  let countNoun: string | null = null;
  if (intent === "count") {
    const m = COUNT_NOUN.exec(fq);
    if (m && AUTHOR_NOUNS.has(m[1])) {
      countOf = "authors";
      countNoun = m[0]; // consumed: the people are what is counted, not the subject
    }
  }
  if (similarId) {
    if (intent !== "duplicate") intent = "similar";
    notes.push(`similar to #${similarId[1]}`);
  }

  // ---- filters and hints --------------------------------------------------------------------------
  const consumed: string[] = [...intentWords, ...(countNoun ? [countNoun] : [])];
  const colors: string[] = [];
  const tones: Tone[] = [];
  const orientation: Array<"portrait" | "landscape" | "square"> = [];
  // multi-word tone/colour phrases first
  for (const [w, t] of Object.entries(TONE_WORDS).filter(([w]) => w.includes(" ")).sort((a, b) => b[0].length - a[0].length)) {
    if (` ${f} `.includes(` ${w} `)) {
      if (!tones.includes(t)) tones.push(t);
      consumed.push(w);
    }
  }
  for (const [w, names] of Object.entries(COLOR_WORDS).filter(([w]) => w.includes(" "))) {
    if (` ${f} `.includes(` ${w} `) && !consumed.some((c) => c.includes(w))) {
      for (const n of names) if (!colors.includes(n)) colors.push(n);
      consumed.push(w);
    }
  }
  for (const [w, o] of Object.entries(ORIENTATION_WORDS).filter(([w]) => w.includes(" "))) {
    if (` ${f} `.includes(` ${w} `)) {
      if (!orientation.includes(o)) orientation.push(o);
      consumed.push(w);
    }
  }
  const words = strip(f, consumed).split(" ").filter(Boolean);
  for (const w of words) {
    if (COLOR_WORDS[w]) {
      for (const n of COLOR_WORDS[w]) if (!colors.includes(n)) colors.push(n);
      consumed.push(w);
    } else if (TONE_WORDS[w]) {
      if (!tones.includes(TONE_WORDS[w])) tones.push(TONE_WORDS[w]);
      consumed.push(w);
    } else if (ORIENTATION_WORDS[w]) {
      if (!orientation.includes(ORIENTATION_WORDS[w])) orientation.push(ORIENTATION_WORDS[w]);
      consumed.push(w);
    }
  }

  // Authors. "@x" is explicit everywhere: a filter. "by x" / "par x" / "von x" naming a known
  // author is explicit in a question (a filter in /ask) and a strong hint in /search, whose only
  // hard filters are the @ and # syntax. "de/du/di/da/from/por/vom x" are ordinary prepositions
  // ("coucher de soleil" with an author called soleil): they count only when x is not a word the
  // vocabulary knows. Bare known names are weak hints.
  const authors = [...mentions];
  const namedAuthors: string[] = [];
  const hintAuthors: string[] = [];
  if (ctx.authors?.size) {
    for (const m of f.matchAll(/\b(by|from|de|du|par|von|vom|di|da|por)\s+([a-z0-9][a-z0-9.-]{1,31})\b/gu)) {
      const [phrase, prep, name] = m;
      if (!ctx.authors.has(name) || authors.includes(name) || namedAuthors.includes(name) || isStopword(name)) continue;
      const strong = prep === "by" || prep === "par" || prep === "von";
      const wordLike = matchConcepts(name).length > 0 || !!COLOR_WORDS[name] || !!TONE_WORDS[name];
      if (!strong && wordLike) continue;
      if (ctx.mode === "ask") {
        authors.push(name);
        consumed.push(phrase);
      } else namedAuthors.push(name);
    }
    for (const w of words) if (ctx.authors.has(w) && !authors.includes(w) && !namedAuthors.includes(w) && !hintAuthors.includes(w)) hintAuthors.push(w);
  }

  if (dates) consumed.push(...dates.consumed);

  // object type: "blog posts about X" vs "artworks of X"
  let object: QueryPlan["object"] = "any";
  for (const w of words) {
    const o = OBJECT_WORDS[w];
    if (o) {
      object = object === "any" ? o : object;
      consumed.push(w);
    }
  }
  if (ctx.mode === "ask" && object === "any") object = "artwork";

  // background colour: "dark blue background", "fond noir"
  const background: string[] = [];
  if (colors.length && /\b(background|backdrop|fond|hintergrund|fondo|sfondo|plano de fondo|背景)\b/u.test(f)) {
    background.push(...colors.filter((c) => !background.includes(c)));
    consumed.push("background", "backdrop", "fond", "hintergrund", "fondo", "sfondo");
  }

  // "on Pixagram" says where, not what (removed before the fillers, which would take "on" alone)
  if (ctx.mode === "ask") for (const m of f.matchAll(/\b(?:on|in|from|sur|en|de|du|auf|bei|von|sobre|su|di) (?:pixagram|pixa)\b/g)) consumed.push(m[0]);

  // ---- residual subject ------------------------------------------------------------------------------
  // In a question, the platform's name is where, not what ("how many members does pixagram have?"),
  // and a lone letter is what a contraction leaves ("what's" → "what s").
  const fillers = ctx.v4 ? QUESTION_FILLER_V4 : QUESTION_FILLER;
  const askNoise = (w: string) => fillers.has(w) || isStopword(w) || ASK_FILLER.has(w) || w === "pixagram" || w === "pixa" || /^[a-z]$/.test(w);
  let residual = strip(f, consumed)
    .split(" ")
    .filter((w) => w && !(ctx.mode === "ask" && (askNoise(w) || authors.includes(w))))
    .join(" ");
  // A known author named in a question ("how many cats did laura post?") is a filter, unless
  // the word also means something ("retro" is an author and a style, "light" an author and a tone).
  if (ctx.mode === "ask" && ctx.authors?.size) {
    const isName = (w: string) => ctx.authors!.has(w) && !matchConcepts(w).length && !COLOR_WORDS[w] && !TONE_WORDS[w];
    const rw = residual.split(" ").filter(Boolean);
    const named = rw.filter(isName);
    if (named.length) {
      for (const w of named) if (!authors.includes(w)) authors.push(w);
      residual = rw.filter((w) => !isName(w)).join(" ");
    }
  }
  const conceptMatches = matchConcepts(ctx.mode === "ask" ? residual : f);
  const concepts = [...new Set(conceptMatches.map((m) => m.concept))];
  const lexicalTerms = residual.split(" ").filter((w) => w.length > 1 && !isStopword(w));

  const filters: PlanFilters = {};
  if (authors.length) filters.authors = authors;
  if (hashtags.length) filters.tags = hashtags;
  if (dates?.from !== undefined) filters.from = dates.from;
  if (dates?.to !== undefined) filters.to = dates.to;
  if (ctx.mode === "ask") {
    if (background.length) filters.background = background;
    else if (colors.length) filters.colors = colors;
    if (tones.length) filters.tones = tones;
    if (orientation.length) filters.orientation = orientation;
  }

  let temporal: QueryPlan["temporal"];
  if (intent === "find_first") temporal = { operator: "first", field: "image" };
  else if (intent === "find_last") temporal = { operator: "last", field: "image" };
  else if (dates?.operator) temporal = { operator: dates.operator, field: "created" };

  // ---- confidence ----------------------------------------------------------------------------------
  let confidence = 0.9;
  if (ctx.mode === "ask") {
    // (f has lost its punctuation: the question mark is looked for in the raw text)
    const questiony = /[?？]/.test(raw) || /^\s*(who|what|which|when|how|qui|quoi|quel|quand|combien|wer|was|welche|wann|wie|quien|que|cual|cuando|chi|che|quale|quando)\b/u.test(fq);
    if (intent === "search" && questiony) {
      confidence = 0.45;
      notes.push("question without a recognised intent");
    } else if (!residual && !colors.length && !authors.length && !tones.length && intent !== "count" && intent !== "top") {
      confidence = 0.75; // e.g. "who posted the first artwork?" — valid, metadata only
    }
    if (residual && !concepts.length && residual.split(" ").length > 3) {
      confidence = Math.min(confidence, 0.6);
      notes.push("long subject with no known concept");
    }
  }

  return {
    intent,
    intentWords,
    countOf,
    object,
    text,
    residual,
    concepts,
    conceptMatches,
    lexicalTerms,
    phrases,
    filters,
    hints: { colors, background, tones, orientation, authors: [...authors, ...namedAuthors, ...hintAuthors], namedAuthors },
    temporal,
    sort,
    output,
    similarTo: similarId ? { id: Number(similarId[1]) } : undefined,
    lang,
    confidence,
    source: "rules",
    notes,
  };
}
