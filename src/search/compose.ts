// The long-form answer (v4.8): what a reader gets beyond the index's one sentence.
//
//   lead        the index's result sentence, then the model's direct answer when it passed (v4's
//               answer_text, unchanged in meaning)
//   body        the model's full answer in Markdown, verified sentence by sentence against the
//               evidence cards: a sentence whose account, date, number, title or path the cards do
//               not state is removed, one that contradicts a result card is removed, the rest
//               stays (so a long answer is never thrown away whole for one bad sentence, as v4
//               did with its three-sentence explanations); addresses the cards do not contain
//               remove their sentence too
//   thinking    the model's written reasoning trail, each step verified the same way and
//               renumbered — the "thinking" a reader can check, never the model's private
//               chain of thought (spec §21)
//   facts, overview, caveats   the deterministic digest (digest.ts), plus the model's caveats
//               once verified
//   follow_ups  the model's follow-up questions, kept only when they name accounts and titles the
//               evidence holds and the rules planner can plan them, then the digest's templates
//   searches    searches for the box, kept when their words are the index's
//
// assemble() renders the sections as Markdown (citations kept, for a UI that links them to the
// cards) and as plain text (citations and Markdown marks removed) in the five answer languages.

import type { Env } from "../env";
import { int, now as nowSeconds } from "../env";
import { fold, type Lang } from "../lib/text";
import { matchConcepts } from "../concepts";
import { answerLang, type AnswerLang } from "./answer-text";
import { accusations, cardFacts, CITATION, unknownNames, verifyClaims, type ClaimStatus, type VerifiedClaim } from "./claims";
import { OBJECT_WORDS } from "./lexicon";
import { isStopword } from "../lib/text";
import { lexicalDocs } from "../help/retrieve";
import { DOCS_KNOWS } from "./router";
import type { Digest } from "./digest";
import type { EvidenceCard } from "./evidence";
import { COLOR_WORDS } from "./lexicon";
import { planQuery } from "./planner";
import { decomposeRules } from "./query-planner";
import { routeQuestion } from "./query-router";
import { QUESTION_START, routeQuery, routeText, type Route } from "./router";

export type AnswerStyle = "rich" | "brief";
export const isAnswerStyle = (x: unknown): x is AnswerStyle => x === "rich" || x === "brief";
export type TextMode = "short" | "full";
export type LengthRequest = "short" | "medium" | "long";
export const isLengthRequest = (x: unknown): x is LengthRequest => x === "short" || x === "medium" || x === "long";

export interface ThinkingStep {
  n: number;
  text: string;
  evidence: string[];
  status: ClaimStatus;
}

export interface SuggestionItem {
  text: string;
  route: Route;
  source: "model" | "rules";
}

export interface AnswerSections {
  /** the index's sentence, then the model's direct answer (v4's answer_text) */
  lead: string;
  /** the model's body, verified, Markdown with citations */
  body?: string;
  facts: string[];
  overview: string[];
  thinking: ThinkingStep[];
  caveats: string[];
  follow_ups: SuggestionItem[];
  searches: SuggestionItem[];
}

export interface VerifiedText {
  text: string;
  claims: VerifiedClaim[];
  removed: number;
  /** supported sentences / sentences (qualified ones count half) */
  egs: number;
  citation_accuracy: number;
}

// ---- words ---------------------------------------------------------------------------------------------

type L<T> = Record<AnswerLang, T>;

export const TITLES: L<{ facts: string; thinking: string; caveats: string; follow_ups: string; searches: string }> = {
  en: { facts: "From the index", thinking: "How this was worked out", caveats: "Keep in mind", follow_ups: "You may also ask", searches: "Related searches" },
  fr: { facts: "D'après l'index", thinking: "Comment la réponse a été établie", caveats: "À garder en tête", follow_ups: "Vous pouvez aussi demander", searches: "Recherches liées" },
  de: { facts: "Aus dem Index", thinking: "So wurde die Antwort ermittelt", caveats: "Zu beachten", follow_ups: "Sie könnten auch fragen", searches: "Verwandte Suchen" },
  es: { facts: "Según el índice", thinking: "Cómo se obtuvo la respuesta", caveats: "A tener en cuenta", follow_ups: "También puede preguntar", searches: "Búsquedas relacionadas" },
  it: { facts: "Dall'indice", thinking: "Come è stata trovata la risposta", caveats: "Da tenere presente", follow_ups: "Potresti anche chiedere", searches: "Ricerche correlate" },
};

/** Body length targets per mode, in words (SEARCH_ANSWER_WORDS), scaled by the request's length. */
export function wordTarget(env: Env, mode: string, length?: LengthRequest): number {
  const dflt: Record<string, number> = { fast: 120, balanced: 250, deep: 450, expert: 800 };
  let words = dflt[mode] ?? 250;
  for (const part of String(env.SEARCH_ANSWER_WORDS ?? "").split(",")) {
    const [k, v] = part.split(":").map((s) => s.trim());
    if (k === mode && Number.isFinite(Number(v)) && Number(v) > 0) words = Math.floor(Number(v));
  }
  if (length === "short") words = Math.round(words / 2);
  else if (length === "long") words = Math.round(words * 1.8);
  return Math.max(60, Math.min(1500, words));
}

// ---- segments of a Markdown text ---------------------------------------------------------------------------

interface Segment {
  /** a sentence (with its citations), or a whole line that is not one (heading, blank) */
  text: string;
  kind: "sentence" | "heading" | "blank";
  /** the list marker or heading marks before the first sentence of a line */
  prefix: string;
  /** index of the line it belongs to */
  line: number;
}

const LIST_MARKER = /^\s*(?:\d{1,2}[.)]|[-*•])\s+/;
const HEADING = /^\s*#{1,6}\s+/;
/** Abbreviations a sentence does not end at ("e.g. the 5% fee", "z. B.", "p. ex."). */
const ABBREVIATION = /(?:^|[\s(])(?:e\.\s?g|i\.\s?e|etc|vs|cf|approx|incl|min|max|z\.\s?b|d\.\s?h|bzw|usw|vgl|ca|p\.\s?ex|ex|env|p\.\s?ej|ej|ecc|es|ad\s?es|nr|no|n°|st|dr|mr|mrs|ms|prof)\.$/i;
/** A sentence ends at . ! ? … and the citations right after it, before a space and the next sentence. */
const SENTENCE_END = /(?<=[.!?…](?:\s*\[(?:[A-Za-z]{1,2}\d{1,9}|\d{1,2})\])*)\s+(?=[^\s\[])/;

export function segmentMarkdown(text: string): Segment[] {
  const out: Segment[] = [];
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return void out.push({ text: "", kind: "blank", prefix: "", line: i });
    if (HEADING.test(line)) return void out.push({ text: line.trim(), kind: "heading", prefix: "", line: i });
    const marker = LIST_MARKER.exec(line)?.[0] ?? "";
    const raw = line.slice(marker.length).split(SENTENCE_END);
    const parts: string[] = [];
    // "e.g. the fee" continues the sentence; "The answer is no. @alice posted it" does not: an
    // abbreviation only holds before a lower-case letter or a digit
    for (const p of raw) parts.length && ABBREVIATION.test(parts[parts.length - 1].trimEnd()) && /^[\p{Ll}\d]/u.test(p) ? (parts[parts.length - 1] += ` ${p}`) : parts.push(p);
    parts.forEach((p, k) => {
      const t = p.trim();
      if (t) out.push({ text: t, kind: "sentence", prefix: k === 0 ? marker.trimStart() : "", line: i });
    });
  });
  return out;
}

const citedIds = (s: string, valid: Set<string>): string[] => [...new Set([...s.matchAll(/\[([A-Za-z]{1,2}\d{1,9})\]/g)].map((m) => m[1].toUpperCase()).filter((id) => valid.has(id)))];
/**
 * A web address, a domain, an e-mail: a model's text may carry none (the cards hold post paths
 * only, so any address in a reply was made up or planted in a caption). Bare domains are read on
 * the common generic endings only: "fin.de plus" in a French sentence is no address.
 */
export const ADDRESS = /https?:\/\/|www\.|[\w.+-]+@[\w-]+\.[a-z]{2,}|(?<![\w@/])[a-z0-9-]{2,}(?:\.[a-z0-9-]{2,})*\.(?:com|net|org|io|xyz|app|dev|info|site|online|link|store|shop|finance|crypto)(?![\w-])/i;

/**
 * Checks no atom catches: an address, an accusation about an account the evidence does not make.
 * The problems to add to a claim (none when it is clean).
 */
export function contentProblems(text: string, cards: EvidenceCard[]): string[] {
  const out: string[] = [];
  if (ADDRESS.test(text.replace(CITATION, ""))) out.push("contains an address");
  const acc = accusations(text, cards);
  if (acc.length && /@[a-z0-9][a-z0-9.-]{1,31}|\b(?:artist|author|account|artiste|auteur|compte|künstler|autor|konto|artista|autor|cuenta|autore|account)\b/i.test(text)) out.push(`an accusation the evidence does not state (${acc.join(", ")})`);
  return out;
}

/**
 * A Markdown text with every sentence checked against the cards: unsupported and contradicted
 * sentences removed (and sentences with an address the cards do not contain, or an accusation
 * they do not make), headings checked the same way (a heading is a sentence without a full
 * stop) and removed when left without a sentence, the rest joined back as it was.
 */
export function verifyMarkdown(text: string, cards: EvidenceCard[], opts: { strict?: boolean; authors?: Set<string> } = {}): VerifiedText {
  const valid = new Set(cards.map((c) => c.evidence_id));
  const segs = segmentMarkdown(text);
  const claims: Array<{ text: string; evidence: string[]; kind: "fact" | "inference" }> = [];
  const claimOf: Array<number | null> = segs.map((s) => {
    if (s.kind === "blank" || !/[\p{L}\p{N}]/u.test(s.text.replace(CITATION, ""))) return null;
    const plain = s.kind === "heading" ? s.text.replace(HEADING, "") : s.text;
    const ids = citedIds(plain, valid);
    claims.push({ text: plain, evidence: ids, kind: ids.length ? "fact" : "inference" });
    return claims.length - 1;
  });
  const v = verifyClaims({ answer: "", claims, rationale: undefined }, cards, { strict: opts.strict, authors: opts.authors, softTitles: true, citedInterpretations: true });
  const keep = (st: ClaimStatus) => st === "supported" || (!opts.strict && st === "qualified");
  const kept: Segment[] = [];
  const out: VerifiedClaim[] = [];
  let removed = 0;
  segs.forEach((s, i) => {
    const ci = claimOf[i];
    if (ci === null) return void kept.push(s);
    const c = { ...v.claims[ci] };
    const problems = contentProblems(s.text, cards);
    if (problems.length) {
      c.status = "unsupported";
      c.problems = [...c.problems, ...problems];
    }
    out.push(c);
    if (keep(c.status)) kept.push(s);
    else removed++;
  });
  // headings with nothing under them (until the next heading or the end) are removed; a list
  // item whose first sentence went keeps its marker
  const markerOf = new Map<number, string>();
  for (const s of segs) if (s.kind === "sentence" && s.prefix && !markerOf.has(s.line)) markerOf.set(s.line, s.prefix);
  const lines: Array<{ line: number; prefix: string; parts: string[]; kind: Segment["kind"] }> = [];
  for (const s of kept) {
    const last = lines[lines.length - 1];
    if (s.kind === "sentence" && last && last.line === s.line) last.parts.push(s.text);
    else lines.push({ line: s.line, prefix: s.kind === "sentence" ? (markerOf.get(s.line) ?? "") : "", parts: s.kind === "sentence" ? [s.text] : [], kind: s.kind });
  }
  const rendered: string[] = [];
  lines.forEach((l, i) => {
    if (l.kind === "heading") {
      const next = lines.slice(i + 1).find((x) => x.kind !== "blank");
      if (!next || next.kind === "heading") return;
      // the heading text itself: kept as written
      rendered.push(kept.find((s) => s.kind === "heading" && s.line === l.line)!.text);
    } else if (l.kind === "blank") rendered.push("");
    else rendered.push(`${l.prefix}${l.parts.join(" ")}`);
  });
  const joined = rendered.join("\n").replace(/\n{3,}/g, "\n\n").trim();
  const weights: number[] = out.map((c) => (c.status === "supported" ? 1 : c.status === "qualified" ? 0.5 : 0));
  return {
    text: /[\p{L}\p{N}]/u.test(joined.replace(CITATION, "")) ? joined : "",
    claims: out,
    removed,
    egs: weights.length ? Math.round((weights.reduce((a, b) => a + b, 0) / weights.length) * 1000) / 1000 : 0,
    citation_accuracy: v.citation_accuracy,
  };
}

/** Short texts (reasoning steps, caveats) checked one by one; the kept ones in order. */
export function verifyList(items: string[], cards: EvidenceCard[], opts: { strict?: boolean; authors?: Set<string> } = {}): { kept: Array<{ text: string; evidence: string[]; status: ClaimStatus }>; removed: number } {
  const valid = new Set(cards.map((c) => c.evidence_id));
  const claims = items.map((t) => ({ text: t, evidence: citedIds(t, valid), kind: (citedIds(t, valid).length ? "fact" : "inference") as "fact" | "inference" }));
  const v = verifyClaims({ answer: "", claims, rationale: undefined }, cards, { strict: opts.strict, authors: opts.authors, softTitles: true, citedInterpretations: true });
  const kept: Array<{ text: string; evidence: string[]; status: ClaimStatus }> = [];
  let removed = 0;
  v.claims.forEach((c, i) => {
    const ok = (c.status === "supported" || (!opts.strict && c.status === "qualified")) && !contentProblems(items[i], cards).length;
    if (ok) kept.push({ text: items[i], evidence: c.supported_by.length ? c.supported_by : c.evidence, status: c.status });
    else removed++;
  });
  return { kept, removed };
}

// ---- follow-up questions and searches --------------------------------------------------------------------------

export interface SuggestionContext {
  env: Env;
  /** the index's known authors (the planner's) */
  authors: Set<string>;
  lang: Lang;
  now?: number;
  cards: EvidenceCard[];
  question: string;
  /** the cards' words and concepts, when already computed */
  vocabulary?: EvidenceVocabulary;
}

export interface FollowUpVerdict {
  text: string;
  ok: boolean;
  route?: Route;
  why?: string;
}

const QUESTION_END = /[?？]\s*$/;

/** The words of a subject the index knows: the evidence's, its concepts, colours, artwork words, stopwords. */
function subjectKnown(residual: string, v: EvidenceVocabulary, authors: Set<string>): string[] {
  const unknown: string[] = [];
  for (const t of fold(residual).split(/[^\p{L}\p{N}@#-]+/u).filter(Boolean)) {
    const w = t.replace(/^[@#]/, "");
    if (!w || w.length <= 2 || isStopword(w) || v.words.has(w) || COLOR_WORDS[w] !== undefined || OBJECT_WORDS[w] !== undefined || authors.has(w)) continue;
    if (matchConcepts(w).some((m) => v.concepts.has(m.concept))) continue;
    unknown.push(t);
  }
  return unknown;
}

/**
 * Whether a follow-up question can be asked: it names only accounts, titles, posts, numbers and
 * dates the evidence holds, carries no address and no accusation, is not the question itself,
 * and leads somewhere — an /ask the rules plan with confidence whose subject is made of words the
 * evidence carries, a search of such words, or a help question the documentation covers (one
 * lexical lookup).
 */
export async function checkFollowUp(raw: string, c: SuggestionContext): Promise<FollowUpVerdict> {
  let text = raw.replace(/\s+/g, " ").trim().replace(/^["“”«»]+|["“”«»]+$/g, "").trim();
  if (!text || text.length > 160 || !/\p{L}/u.test(text)) return { text, ok: false, why: "empty or too long" };
  // a question written without its mark gets one; a search ("red cats") stays as it is
  if (!QUESTION_END.test(text) && !/[.!]$/.test(text) && QUESTION_START.test(routeText(text))) text = `${text}?`;
  if (fold(text).replace(/[?？!.]/g, "").trim() === fold(c.question).replace(/[?？!.]/g, "").trim()) return { text, ok: false, why: "the question itself" };
  const content = contentProblems(text, c.cards);
  if (content.length) return { text, ok: false, why: content.join("; ") };
  const unknown = unknownNames(text, c.cards, { authors: c.authors, numbers: true });
  if (unknown.length) return { text, ok: false, why: `names ${unknown.join(", ")} the evidence does not hold` };
  const now = c.now ?? nowSeconds();
  const plan = planQuery(text, { authors: c.authors, mode: "ask", now, v4: true });
  const decision = await routeQuery(text, plan, { docsScore: async () => 0, isTitle: async () => false });
  const vocab = c.vocabulary ?? evidenceWords(c.cards);
  if (decision.route === "help") {
    const best = (await lexicalDocs(c.env, text, 3).catch(() => ({ hits: [] as Array<{ score: number }> }))).hits[0]?.score ?? 0;
    return best >= DOCS_KNOWS ? { text, ok: true, route: "help" } : { text, ok: false, route: "help", why: `the documentation does not cover it (${Math.round(best * 100) / 100})` };
  }
  if (decision.route === "search") {
    const subject = !!(plan.residual || plan.concepts.length || plan.filters.authors?.length || plan.filters.colors?.length || plan.filters.tags?.length);
    if (!subject) return { text, ok: false, why: "a search with no subject" };
    const strange = subjectKnown(text, vocab, c.authors);
    return strange.length ? { text, ok: false, route: "search", why: `words the evidence does not carry: ${strange.join(", ")}` } : { text, ok: true, route: "search" };
  }
  const program = decomposeRules(text, plan, { authors: c.authors, lang: plan.lang, now });
  const route = routeQuestion(text, plan, program, { env: c.env });
  if (route.class === "UNKNOWN" || route.class === "AMBIGUOUS") return { text, ok: false, route: "ask", why: `${route.class.toLowerCase()}: ${route.reason}` };
  if (program.pattern === "single" && plan.confidence < 0.5) return { text, ok: false, route: "ask", why: `the rules are unsure of it (${plan.confidence})` };
  // its subject (the steps' subjects, once the rules took the question apart) is made of words
  // the evidence carries: "Why is @bob a scammer who steals from @carol?" is not
  const residuals = (program.pattern === "single" ? [plan.residual] : program.steps.map((st) => st.plan?.residual ?? "")).join(" ");
  const strange = subjectKnown(residuals, vocab, c.authors);
  if (strange.length) return { text, ok: false, route: "ask", why: `words the evidence does not carry: ${strange.join(", ")}` };
  return { text, ok: true, route: "ask" };
}

/**
 * Whether a search for the box leads somewhere: short, and made of words the evidence carries
 * (a word of a title, tag or caption, an account, a concept the cards show, a colour).
 */
export function checkSearch(raw: string, cards: EvidenceCard[], authors: Set<string>, known?: EvidenceVocabulary): { text: string; ok: boolean } {
  const text = raw.replace(/\s+/g, " ").trim().replace(/^["“”«»]+|["“”«»]+$/g, "").replace(/[?？!.]+$/, "").trim();
  if (!text || text.length > 60 || !/\p{L}/u.test(text)) return { text, ok: false };
  const toks = fold(text).split(/[^\p{L}\p{N}@#-]+/u).filter(Boolean);
  if (!toks.length) return { text, ok: false };
  const v = known ?? evidenceWords(cards);
  const ok = toks.every((t) => {
    const w = t.replace(/^[@#]/, "");
    if (!w) return false;
    if (t.startsWith("@")) return authors.has(w) || v.words.has(w);
    if (v.words.has(w) || COLOR_WORDS[w] !== undefined || w.length <= 2) return true;
    return matchConcepts(w).some((m) => v.concepts.has(m.concept));
  });
  return { text, ok };
}

export interface EvidenceVocabulary {
  words: Set<string>;
  concepts: Set<string>;
}

/** Every word the cards carry (titles, tags, captions, accounts, paths), and the concepts they show. */
export function evidenceWords(cards: EvidenceCard[]): EvidenceVocabulary {
  const words = new Set<string>();
  const concepts = new Set<string>();
  for (const c of cards) {
    const f = cardFacts(c);
    for (const w of f.words) words.add(w);
    for (const a of f.accounts) words.add(a);
    for (const id of f.concepts) concepts.add(id);
  }
  return { words, concepts };
}

// ---- assembly ---------------------------------------------------------------------------------------------------

export interface AssembleInput {
  lang: Lang;
  sections: AnswerSections;
}

/** The Markdown of the sections: the lead, the body, then the digest's parts under their headings. */
export function assembleMarkdown(x: AssembleInput): string {
  const l = answerLang(x.lang);
  const T = TITLES[l];
  const s = x.sections;
  const blocks: string[] = [];
  if (s.lead.trim()) blocks.push(s.lead.trim());
  if (s.body?.trim()) blocks.push(s.body.trim());
  if (s.overview.length || s.facts.length) {
    const parts = [s.overview.join(" "), s.facts.map((f) => `- ${f}`).join("\n")].filter(Boolean);
    blocks.push(`### ${T.facts}\n${parts.join("\n\n")}`);
  }
  if (s.thinking.length) blocks.push(`### ${T.thinking}\n${s.thinking.map((t) => `${t.n}. ${t.text}`).join("\n")}`);
  if (s.caveats.length) blocks.push(`### ${T.caveats}\n${s.caveats.map((c) => `- ${c}`).join("\n")}`);
  if (s.follow_ups.length) blocks.push(`### ${T.follow_ups}\n${s.follow_ups.map((f) => `- ${f.text}`).join("\n")}`);
  if (s.searches.length) blocks.push(`${T.searches}: ${s.searches.map((f) => f.text).join(" · ")}`);
  return blocks.join("\n\n");
}

/** The Markdown as plain text: citations and Markdown marks removed, headings as "Title:" lines. */
export function toPlainText(md: string): string {
  return md
    .replace(/[ \t]*\[(?:[A-Za-z]{1,2}\d{1,9}|\d{1,2})\]/g, "")
    .replace(/^[ \t]*#{1,6}[ \t]+(.+?)[ \t]*$/gm, "$1:")
    .replace(/\*\*([^*\n]+)\*\*/g, "$1")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?!\w)/g, "$1$2")
    .replace(/`([^`\n]+)`/g, "$1")
    .replace(/^\s*[*•]\s+/gm, "- ")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/[ \t]+$/gm, "")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export const countWords = (s: string): number => (s.replace(CITATION, "").match(/[\p{L}\p{N}][\p{L}\p{N}'’-]*/gu) ?? []).length;

/** How many follow-up questions an answer shows (SEARCH_FOLLOW_UPS, default 6). */
export const followUpLimit = (env: Env) => Math.max(0, Math.min(12, int(env.SEARCH_FOLLOW_UPS, 6)));

/**
 * The follow-ups of an answer: the model's that pass, then the digest's templates that pass,
 * without repeats, up to the limit. The searches likewise.
 */
export async function suggestions(c: SuggestionContext, model: { followUps?: string[]; searches?: string[] } | null, digest: Digest, opts: { limit: number; notes?: string[] }): Promise<{ follow_ups: SuggestionItem[]; searches: SuggestionItem[]; removed: { follow_ups: number; searches: number } }> {
  const seen = new Set<string>();
  const follow_ups: SuggestionItem[] = [];
  let removedQ = 0;
  const words = c.vocabulary ?? evidenceWords(c.cards);
  const ctx = { ...c, vocabulary: words };
  const consider = async (texts: string[], source: "model" | "rules") => {
    for (const t of texts) {
      if (follow_ups.length >= opts.limit) break;
      const v = await checkFollowUp(t, ctx);
      const key = fold(v.text).replace(/[^\p{L}\p{N}@]+/gu, " ").trim();
      if (!v.ok || seen.has(key)) {
        if (!v.ok) {
          removedQ++;
          opts.notes?.push(`follow-up dropped (${source}): “${t.slice(0, 60)}”: ${v.why}`);
        }
        continue;
      }
      seen.add(key);
      follow_ups.push({ text: v.text, route: v.route ?? "ask", source });
    }
  };
  await consider(model?.followUps ?? [], "model");
  await consider(digest.follow_ups, "rules");
  const searches: SuggestionItem[] = [];
  const seenS = new Set<string>();
  let removedS = 0;
  const considerSearch = (texts: string[], source: "model" | "rules") => {
    for (const t of texts) {
      if (searches.length >= 6) break;
      const v = checkSearch(t, c.cards, c.authors, words);
      const key = fold(v.text);
      if (!v.ok) {
        removedS++;
        continue;
      }
      if (seenS.has(key) || key === fold(c.question)) continue;
      seenS.add(key);
      searches.push({ text: v.text, route: "search", source });
    }
  };
  considerSearch(model?.searches ?? [], "model");
  considerSearch(digest.searches, "rules");
  return { follow_ups, searches, removed: { follow_ups: removedQ, searches: removedS } };
}
