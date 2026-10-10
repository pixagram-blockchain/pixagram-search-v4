// Claim verification (spec §22-23, §34): every claim the reasoning model makes is checked against
// the evidence it was given, deterministically.
//
//   claim text ─► atoms: accounts (@alice, and names without "@": "by Mallory", "Mallory posted"),
//                 dates (2026-09-10, 10.09.2026, "10 September 2026", "the 10th of September",
//                 "10 septembre", "10° settembre"), months, years, numbers with what they count
//                 ("3 votes", "2 œuvres", "7 days", "5 %", "10 MB", "1,000,000 PXS", "three"),
//                 quoted titles (and the numbers inside them), post paths, superlatives ("first",
//                 "latest", "most"), yes / no in five languages
//             ─► each atom looked up in the cited cards' facts, then in all cards. A number is only
//                 looked up among the facts of its kind (votes with votes, counts with counts, a
//                 percentage with percentages), never among ids, scores, hashes or timestamps; an
//                 account only among the accounts the index states (authors, paths, results), not
//                 among words of titles, tags or captions
//             ─► supported      every factual atom is in the evidence (citations fixed when it was
//                               found in another card than the cited ones)
//                 contradicted   it contradicts a result card: another account for "the first …"
//                               (the account of the superlative's own clause), another author for a
//                               named title, another date, count, duration or post than the
//                               result's, "No" where the result says yes
//                 unsupported    a factual atom the evidence does not contain
//                 qualified      no factual atom; an inference the evidence only partly supports
//
// Contradicted and unsupported claims never reach the answer; qualified ones only outside the
// strict (expert) mode, with their confidence capped. EGS = supported claims / claims.
//
// agreesWithResult() is the stricter test of a whole text shown after a deterministic result
// (spec §27: result, then explanation): it must state the result's own value — the same @author
// first, the same date, number, post, and "Yes"/"No" first for a yes/no result.

import { isStopword, singular, tokens, type Lang } from "../lib/text";
import { ancestors, matchConcepts, matchTokens } from "../concepts";
import type { Claim, ReasoningResponse } from "../llm/reasoning";
import type { EvidenceCard, ResultCard } from "./evidence";

export type ClaimStatus = "supported" | "qualified" | "unsupported" | "contradicted";

export type AtomKind = "account" | "date" | "month" | "year" | "number" | "title" | "ref" | "superlative" | "yesno";

/** What a number counts: votes are not artworks, a 5 % fee is not 5 days. "any": a bare number. */
export type NumberUnit = "count" | "votes" | "money" | "percent" | "duration" | "size" | "any";

export interface Atom {
  kind: AtomKind;
  /** normalised: account name, YYYY-MM-DD, YYYY-MM or *-MM-DD, YYYY, number, folded title, path, superlative family, "yes"/"no" */
  value: string;
  raw: string;
  /** other readings of the same text ("14,469": 14.469 or 14469; "01/09/2026": 1 September or 9 January) */
  alts?: string[];
  /** numbers: what they count */
  unit?: NumberUnit;
  /** where it starts in the text (a superlative is checked against the account of its own clause) */
  at?: number;
  /** an account named without "@" ("by Mallory"): not an account if it is only a word of the evidence */
  bare?: boolean;
  /** an evidence author's name met as a plain word ("a light blue sky" when @light is an author) */
  loose?: boolean;
  /** a number inside a quoted title: checked only when the title itself is not in the evidence */
  inTitle?: string;
}

export interface VerifiedClaim extends Claim {
  status: ClaimStatus;
  /** evidence ids where its atoms were found (the citations, corrected) */
  supported_by: string[];
  /** atoms not found, or contradicted, with why */
  problems: string[];
  /** citation was right: every atom is in the cards the claim cited */
  cited_correctly: boolean;
}

export interface ClaimVerification {
  claims: VerifiedClaim[];
  answer: { status: ClaimStatus; problems: string[]; supported_by: string[] };
  rationale: { status: ClaimStatus } | null;
  /** evidence grounding score: supported claims / claims (spec §34) */
  egs: number;
  /** share of supported claims whose citations were right */
  citation_accuracy: number;
  counts: Record<ClaimStatus, number>;
}

// ---- text ----------------------------------------------------------------------------------------

/** Accents off, ligatures spelled out (œuvre → oeuvre, Straße → Strasse); case kept. */
function deaccent(s: string): string {
  return s
    .normalize("NFKC")
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .normalize("NFC")
    .replace(/œ/g, "oe")
    .replace(/Œ/g, "Oe")
    .replace(/æ/g, "ae")
    .replace(/Æ/g, "Ae")
    .replace(/ß/g, "ss");
}

/** fold() with ligatures spelled out: what claims and evidence are compared in. */
export const foldx = (s: string): string => deaccent(s).toLowerCase();

/** Evidence citations ([E12], [R1], [I1], [D2], and /help's [1]…[99]); "[2019]" is a year, not a citation. */
export const CITATION = /\[(?:[A-Z]{1,2}\d{1,9}|\d{1,2})\]/g;

export const titleKey = (s: string): string => foldx(s).replace(/[^\p{L}\p{N}]+/gu, " ").trim();

// ---- vocabularies ----------------------------------------------------------------------------------

const MONTHS: Record<string, number> = (() => {
  const m: Record<string, number> = {};
  const names = [
    "january jan janvier janv januar janner enero gennaio",
    "february feb fevrier fevr februar febrero febbraio",
    "march mar mars marz marzo",
    "april apr avril abril aprile",
    "may mai mayo maggio",
    "june jun juin juni junio giugno",
    "july jul juillet juil juli julio luglio",
    "august aug aout agosto",
    "september sep sept septembre septiembre settembre",
    "october oct octobre oktober okt octubre ottobre",
    "november nov novembre noviembre",
    "december dec decembre dezember dez diciembre dicembre",
  ];
  names.forEach((line, i) => line.split(" ").forEach((w) => (m[w] = i + 1)));
  return m;
})();
const MONTH_RE = Object.keys(MONTHS).sort((a, b) => b.length - a.length).join("|");

/**
 * Numbers written as words, in the five languages. Words that are also other words are left out
 * ("un", "sept", "neuf", "due", "once", "sei", "elf", "tres", "mil"…); English "one" counts only
 * before what it counts ("one block", "one artwork").
 */
const NUMBER_WORDS: Record<string, number> = {
  zero: 0, cero: 0,
  two: 2, deux: 2, zwei: 2, dos: 2,
  three: 3, trois: 3, drei: 3, tre: 3,
  four: 4, quatre: 4, vier: 4, cuatro: 4, quattro: 4,
  five: 5, cinq: 5, funf: 5, cinco: 5, cinque: 5,
  six: 6, sechs: 6, seis: 6,
  seven: 7, sieben: 7, siete: 7, sette: 7,
  eight: 8, huit: 8, acht: 8, ocho: 8,
  nine: 9, neun: 9, nueve: 9,
  ten: 10, dix: 10, zehn: 10, diez: 10, dieci: 10,
  eleven: 11, onze: 11, undici: 11,
  twelve: 12, douze: 12, zwolf: 12, doce: 12, dodici: 12,
  thirteen: 13, treize: 13, dreizehn: 13, trece: 13, tredici: 13,
  fourteen: 14, quatorze: 14, vierzehn: 14, catorce: 14, quattordici: 14,
  fifteen: 15, quinze: 15, funfzehn: 15, quince: 15, quindici: 15,
  sixteen: 16, sechzehn: 16, dieciseis: 16, sedici: 16,
  seventeen: 17, siebzehn: 17, diecisiete: 17, diciassette: 17,
  eighteen: 18, achtzehn: 18, dieciocho: 18, diciotto: 18,
  nineteen: 19, neunzehn: 19, diecinueve: 19, diciannove: 19,
  twenty: 20, vingt: 20, zwanzig: 20, veinte: 20, venti: 20,
  thirty: 30, trente: 30, dreissig: 30, treinta: 30, trenta: 30,
  forty: 40, quarante: 40, vierzig: 40, cuarenta: 40, quaranta: 40,
  fifty: 50, cinquante: 50, funfzig: 50, cincuenta: 50, cinquanta: 50,
  sixty: 60, soixante: 60, sechzig: 60, sesenta: 60, sessanta: 60,
  seventy: 70, siebzig: 70, setenta: 70, settanta: 70,
  eighty: 80, achtzig: 80, ochenta: 80, ottanta: 80,
  ninety: 90, neunzig: 90, noventa: 90, novanta: 90,
  hundred: 100, hundert: 100, cien: 100, ciento: 100, cento: 100,
  thousand: 1000, tausend: 1000, mille: 1000,
  million: 1e6, millions: 1e6, millionen: 1e6, millon: 1e6, millones: 1e6, milione: 1e6, milioni: 1e6,
  billion: 1e9, billions: 1e9, milliard: 1e9, milliards: 1e9, milliarde: 1e9, miliardo: 1e9, miliardi: 1e9,
};
const ENGLISH_UNITS: Record<string, number> = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9 };
const MULTIPLIERS: Record<string, number> = { hundred: 100, hundert: 100, thousand: 1000, tausend: 1000, mille: 1000, mil: 1000, k: 1000, million: 1e6, millions: 1e6, millionen: 1e6, millon: 1e6, millones: 1e6, milione: 1e6, milioni: 1e6, billion: 1e9, billions: 1e9, milliard: 1e9, milliards: 1e9, milliarde: 1e9, miliardo: 1e9, miliardi: 1e9 };

/** The unit words a number may be followed by, folded. */
const UNIT_RES: Array<[NumberUnit, RegExp]> = [
  ["percent", /^(?:%|percent|per ?cent|pour ?cent|prozent|por ?ciento|per ?cento)$/],
  ["votes", /^(?:votes?|voix|stimmen?|votos?|voti|upvotes?|likes?|mentions?)$/],
  ["money", /^(?:pxs|pixa|hive|hbd|usd|eur|chf|dollars?|euros?|francs?|\$|€|£)$/],
  ["duration", /^(?:days?|jours?|journees?|tage?n?|dias?|giorn[oi]|hours?|heures?|stunden?|horas?|or[ae]|h|hrs?|weeks?|semaines?|wochen?|woche|semanas?|settiman[ae]|months?|mois|monate?n?|monat|mes(?:es)?|mes[ei]|years?|ans?|annees?|jahre?n?|anos?|ann[oi]|minutes?|minuten?|minutos?|minuti|mins?|seconds?|secondes?|sekunden?|segundos?|secondi|secs?|blocks?|blocs?|bloques|blocchi|d)$/],
  ["size", /^(?:mb|mo|kb|ko|gb|go|tb|px|pixels?|bytes?|octets?|megabytes?|kilobytes?)$/],
];
/** What artworks are counted in. */
const COUNT_NOUN = /^(?:artworks?|arts?|posts?|images?|pictures?|drawings?|oeuvres?|creations?|dessins?|publications?|kunstwerke?n?|bilder?n?|beitrage?n?|werke?n?|zeichnungen|obras?|imagen(?:es)?|dibujos?|publicacion(?:es)?|opere|opera|immagin[ei]|disegn[oi]|pubblicazion[ei]|results?|resultats?|ergebniss?e?|resultados?|risultat[oi]|matches|times|fois|mal|veces|volte|authors?|artists?|artistes?|kunstler(?:in|innen)?|artistas?|autor(?:es|en)?|autori?|accounts?|comptes?|kont(?:o|en)|cuentas?|conti|edits?|modifications?|bearbeitungen|ediciones|modifiche|reposts?|copies|copias|copie|versions?|versionen|versiones|versioni)$/;

const SUPERLATIVES: Array<[string, RegExp]> = [
  ["first", /\b(first|earliest|oldest|premier|premiere|premiers|premieres|plus ancien\w*|erste\w*|alteste\w*|fruheste\w*|primer\w*|mas antigu\w*|prim[oaie](?!\s+(?:di|del|della|dello|dei|degli|delle|che)\b)|piu vecchi\w*)\b/g],
  ["latest", /\b(latest|newest|most recent|last|dernier\w*|derniere\w*|plus recent\w*|letzte\w*|neueste\w*|ultim[oaie]\w*|mas reciente\w*|piu recent\w*)\b/g],
  ["most", /\b(most(?!\s+recent)|le plus|la plus|les plus|am meisten|die meisten|meiste\w*|la mayor cantidad|piu di tutti|il maggior numero)\b/g],
];

/** Words that look like names after "by" or at the start of a sentence but are not accounts. */
const NOT_NAMES = new Set(
  (
    "pixagram pixa pixachain hive steem blockchain chain index evidence card result results someone somebody anyone anybody everyone everybody nobody nothing " +
    "another other others each both neither either itself himself herself themselves default hand far now then scratch chance design mistake accident date time " +
    "title titles vote votes payout author authors artist artists tag tags month year week day color colour size name number count popularity likes people users user " +
    "quelqu personne quelquun jemand niemand alguien nadie qualcuno nessuno none ninguno nessun keiner aucun aucune " +
    "mit dem den der die das einem einer eines contre exemple ailleurs consequent"
  ).split(" "),
);
const POSTING_VERB = String.raw`(?:posted|published|uploaded|made|created|drew|painted|reposted|shared|minted|a\s+(?:poste|publie|cree|fait|dessine|partage|reposte)|hat|ha\s+(?:pubblicato|postato|creato|disegnato|fatto|condiviso)|publico|subio|creo|dibujo|hizo|compartio|pubblico)`;

const pad = (n: number) => String(n).padStart(2, "0");
const okDate = (y: number, m: number, d: number) => y >= 1990 && y <= 2100 && m >= 1 && m <= 12 && d >= 1 && d <= 31;
const isoOf = (y: number, m: number, d: number) => `${y}-${pad(m)}-${pad(d)}`;

// ---- numbers -------------------------------------------------------------------------------------------

/** A number as written, in every reading: "1,000,000" · "1.234,5" · "1 234" · "1'234" · "14,469" (14469 or 14.469) · "1,5". */
export function numberReadings(raw: string): number[] {
  const g = /^(\d{1,3})((?:[,.'    ]\d{3})+)([.,]\d+)?$/.exec(raw);
  if (g && g[1] !== "0") {
    const seps = new Set(g[2].match(/[,.'    ]/g));
    if (seps.size === 1) {
      const sep = [...seps][0];
      const dec = g[3] && g[3][0] !== sep ? Number(`0.${g[3].slice(1)}`) : 0;
      if (g[3] && g[3][0] === sep) return [];
      const grouped = Number(g[1] + g[2].replace(/\D/g, "")) + dec;
      // one group after a comma or a dot may be decimals: "14,469" (an English thousand or a French
      // decimal), "14.469" (an English decimal or a German thousand); the English reading first
      if (g[2].length === 4 && (sep === "," || sep === ".") && !g[3]) {
        const decimal = Number(`${g[1]}.${g[2].slice(1)}`);
        return sep === "." ? [decimal, grouped] : [grouped, decimal];
      }
      return [grouped];
    }
  }
  const v = Number(raw.replace(/[   ' ]/g, "").replace(",", "."));
  return Number.isFinite(v) ? [v] : [];
}

/** What the words after a number say it counts (up to three words, stopping at punctuation or another number). */
function unitAfter(tail: string, before: string): NumberUnit {
  if (/[$€£]\s?$/.test(before)) return "money";
  const head = /^[^.;:!?()[\]\n]*/.exec(tail)?.[0] ?? "";
  if (/^\s*%/.test(head)) return "percent";
  const words = head.match(/[\p{L}$€£%]+|\d+/gu) ?? [];
  for (let i = 0; i < Math.min(3, words.length); i++) {
    const w = words[i];
    if (/^\d/.test(w)) break;
    if (i === 0 || i === 1) {
      const two = `${w} ${words[i + 1] ?? ""}`.trim();
      for (const [u, re] of UNIT_RES) if (re.test(two)) return u;
    }
    for (const [u, re] of UNIT_RES) if (re.test(w)) return u;
    if (COUNT_NOUN.test(w)) return "count";
  }
  return words.length && !/^\d/.test(words[0]!) ? "count" : "any";
}

// ---- atoms ---------------------------------------------------------------------------------------------

/** The checkable atoms of a text. Citations ([E12], [1]) are not part of what it says. */
export function extractAtoms(text: string, opts: { authors?: Set<string> } = {}): Atom[] {
  const atoms: Atom[] = [];
  // C keeps the case (names), F is folded; both have the same length, consumed spans become spaces
  let C = deaccent(text).replace(CITATION, (m) => " ".repeat(m.length));
  let F = C.toLowerCase();
  if (F.length !== C.length) F = C.replace(/./gsu, (ch) => (ch.toLowerCase().length === ch.length ? ch.toLowerCase() : ch));
  const mask = (i: number, n: number) => {
    const sp = " ".repeat(n);
    C = C.slice(0, i) + sp + C.slice(i + n);
    F = F.slice(0, i) + sp + F.slice(i + n);
  };
  const scan = (re: RegExp, on: "C" | "F", f: (m: RegExpMatchArray) => Atom | Atom[] | null, consume = true) => {
    const spans: Array<[number, number]> = [];
    for (const m of (on === "C" ? C : F).matchAll(re)) {
      const r = f(m);
      if (!r) continue;
      for (const a of Array.isArray(r) ? r : [r]) atoms.push({ ...a, at: a.at ?? m.index });
      spans.push([m.index!, m[0].length]);
    }
    if (consume) for (const [i, n] of spans) mask(i, n);
  };

  // quoted titles first: their words are names, not claims; numbers inside one are still claims
  // ('Click “Claim 500 PXS”' quotes a label the excerpt must contain, with its number)
  scan(/“([^”\n]{1,160})”|"([^"\n]{1,160})"|«\s*([^»\n]{1,160}?)\s*»|„([^“”\n]{1,160})[“”]|‘([^’\n]{1,160})’|(?<![\p{L}\p{N}])'([^'\n]{1,80}?[^\s'])'(?![\p{L}\p{N}])/gu, "C", (m) => {
    const t = (m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5] ?? m[6] ?? "").trim();
    if (!t) return null;
    const key = titleKey(t);
    const inner = extractAtoms(t).filter((a) => a.kind === "number" || a.kind === "year").map((a) => ({ ...a, unit: "any" as NumberUnit, kind: "number" as AtomKind, at: m.index, inTitle: key }));
    return [{ kind: "title", value: key, raw: t }, ...inner];
  });
  // permlinks are lower-case letters, digits and hyphens (a trailing full stop ends the sentence)
  scan(/\/@([a-z0-9][a-z0-9.-]{0,30}[a-z0-9])\/([a-z0-9][a-z0-9-]*)/g, "F", (m) => ({ kind: "ref", value: `/@${m[1]}/${m[2].replace(/-+$/, "")}`, raw: m[0] }));
  scan(/@([a-z0-9][a-z0-9.-]{1,31})/g, "F", (m) => {
    const v = m[1].replace(/[.-]+$/, "");
    return v.length >= 2 ? { kind: "account", value: v, raw: `@${v}` } : null;
  });
  // dates: ISO, numeric (day first, both readings when both are dates), written in five languages
  scan(/\b(\d{4})-(\d{2})-(\d{2})(?:[t ][\d:.]+z?)?\b/g, "F", (m) => (okDate(+m[1], +m[2], +m[3]) ? { kind: "date", value: isoOf(+m[1], +m[2], +m[3]), raw: m[0] } : null));
  scan(/\b(\d{4})\/(\d{1,2})\/(\d{1,2})\b/g, "F", (m) => (okDate(+m[1], +m[2], +m[3]) ? { kind: "date", value: isoOf(+m[1], +m[2], +m[3]), raw: m[0] } : null));
  scan(/(?<![\d.\/,])(\d{1,2})([./])(\d{1,2})\2(\d{4})(?![\d])/g, "F", (m) => {
    const [a, b, y] = [+m[1], +m[3], +m[4]];
    const r = [okDate(y, b, a) ? isoOf(y, b, a) : null, okDate(y, a, b) ? isoOf(y, a, b) : null].filter((x): x is string => !!x);
    const u = [...new Set(r)];
    return u.length ? { kind: "date", value: u[0], raw: m[0], ...(u.length > 1 ? { alts: u.slice(1) } : {}) } : null;
  });
  const ORD = String.raw`(?:st|nd|rd|th|er|re|eme|e|o|a|°)?`;
  scan(new RegExp(String.raw`\b(?:the\s+)?(\d{1,2})${ORD}\.?\s+(?:of\s+|de\s+|del\s+)?(${MONTH_RE})\b\.?(?:,?\s+(?:de\s+|del\s+)?(\d{4}))?`, "g"), "F", (m) => {
    const d = +m[1];
    const mo = MONTHS[m[2]];
    if (!mo || d < 1 || d > 31) return null;
    return m[3] ? { kind: "date", value: isoOf(+m[3], mo, d), raw: m[0].trim() } : { kind: "month", value: `*-${pad(mo)}-${pad(d)}`, raw: m[0].trim() };
  });
  scan(new RegExp(String.raw`\b(${MONTH_RE})\.?\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(\d{4}))?`, "g"), "F", (m) => {
    const d = +m[2];
    const mo = MONTHS[m[1]];
    if (!mo || d < 1 || d > 31) return null;
    return m[3] ? { kind: "date", value: isoOf(+m[3], mo, d), raw: m[0].trim() } : { kind: "month", value: `*-${pad(mo)}-${pad(d)}`, raw: m[0].trim() };
  });
  scan(new RegExp(String.raw`\b(${MONTH_RE})\.?\s+(?:de\s+|del\s+)?(\d{4})\b`, "g"), "F", (m) => ({ kind: "month", value: `${m[2]}-${pad(MONTHS[m[1]])}`, raw: m[0].trim() }));
  scan(/\b(\d{4})-(\d{2})\b(?!-)/g, "F", (m) => (+m[2] >= 1 && +m[2] <= 12 && +m[1] >= 1990 && +m[1] <= 2100 ? { kind: "month", value: `${m[1]}-${m[2]}`, raw: m[0] } : null));

  // numbers: grouped thousands, decimals, attached units (24h, 10MB, 5k), multipliers ("1 million"),
  // what they count; a bare 1990–2100 is a year
  scan(/(?<![\p{L}\p{N}.,'])(\d{1,3}(?:([,.'    ])\d{3})(?:\2\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?![\p{N}])(\s?k\b)?/gu, "F", (m) => {
    let vals = numberReadings(m[1]);
    if (!vals.length) return null;
    const end = m.index! + m[0].length;
    const tail = F.slice(end, end + 80);
    const mult = m[3] ? 1000 : MULTIPLIERS[/^\s*([a-z]+)/.exec(tail)?.[1] ?? ""] ?? 1;
    if (mult !== 1) vals = vals.map((v) => v * mult);
    const unit = unitAfter(mult !== 1 && !m[3] ? tail.replace(/^\s*[a-z]+/, "") : tail, F.slice(Math.max(0, m.index! - 3), m.index!));
    const v = vals[0];
    if (unit === "any" && vals.length === 1 && Number.isInteger(v) && v >= 1990 && v <= 2100 && /^\d{4}$/.test(m[1])) return { kind: "year", value: String(v), raw: m[1] };
    const alts = [...new Set(vals.slice(1).map(String))].filter((x) => x !== String(v));
    return { kind: "number", value: String(v), raw: m[0].trim(), unit, ...(alts.length ? { alts } : {}) };
  });
  // numbers in words: "three", "twenty-five", "two hundred", "one block" (English "one" only before what it counts)
  const unitWord = (x?: string) => !!x && (COUNT_NOUN.test(x) || UNIT_RES.some(([, re]) => re.test(x)));
  const wordNumber = (word: string, at: number): Atom | null => {
    const after = F.slice(at + word.length, at + word.length + 80);
    if (word === "one") {
      const next = /^\s+(?:([a-z]+)\s+)?([a-z]+)/.exec(after);
      return next && (unitWord(next[1]) || unitWord(next[2])) ? { kind: "number", value: "1", raw: word, unit: unitAfter(after, ""), at } : null;
    }
    if (NUMBER_WORDS[word] === undefined) return null;
    return { kind: "number", value: String(NUMBER_WORDS[word]), raw: word, unit: unitAfter(after, ""), at };
  };
  const words = [...F.matchAll(/[a-z]+/g)];
  for (let i = 0; i < words.length; i++) {
    const w = words[i][0];
    const at = words[i].index!;
    const next = words[i + 1];
    const joined = next && /^[- ]$/.test(F.slice(at + w.length, next.index!));
    // "twenty-five", "two hundred": one number
    if (joined && NUMBER_WORDS[w] !== undefined && NUMBER_WORDS[w] >= 20 && NUMBER_WORDS[w] < 100 && ENGLISH_UNITS[next![0]] !== undefined) {
      const end = next!.index! + next![0].length;
      atoms.push({ kind: "number", value: String(NUMBER_WORDS[w] + ENGLISH_UNITS[next![0]]), raw: F.slice(at, end), unit: unitAfter(F.slice(end, end + 80), ""), at });
      i++;
      continue;
    }
    if (joined && (NUMBER_WORDS[w] !== undefined || ENGLISH_UNITS[w] !== undefined) && MULTIPLIERS[next![0]] && MULTIPLIERS[next![0]] >= 100 && !MULTIPLIERS[w]) {
      const end = next!.index! + next![0].length;
      atoms.push({ kind: "number", value: String((NUMBER_WORDS[w] ?? ENGLISH_UNITS[w]) * MULTIPLIERS[next![0]]), raw: F.slice(at, end), unit: unitAfter(F.slice(end, end + 80), ""), at });
      i++;
      continue;
    }
    const a = wordNumber(w, at);
    if (a) atoms.push(a);
  }
  // superlatives, with where they are
  for (const [fam, re] of SUPERLATIVES) for (const m of F.matchAll(re)) atoms.push({ kind: "superlative", value: fam, raw: m[1], at: m.index });
  // yes / no at the start of a sentence or clause ("No one posted…" is not a "no"; "Si l'œuvre…" is not a "yes")
  for (const m of F.matchAll(/(?:^|[.!?;:\n]\s*|[—–]\s*)\s*(yes|yeah|oui|ja|si|no|non|nein)(?=([^\p{L}]|$))(.{0,24})/gu)) {
    const w = m[1];
    const rest = m[3] ?? "";
    const punct = /^\s*[,.:;!—–-]|^\s*$/.test(rest);
    let v: "yes" | "no" | null = null;
    if (w === "yes" || w === "yeah" || w === "oui" || w === "ja") v = "yes";
    else if (w === "nein") v = "no";
    else if (w === "si") v = punct ? "yes" : null;
    else if (w === "no") v = punct || /^\s+(?:se|ha|han|he|fue|es|esta|hay|e|era|lo|la|le|existe|aparece|c'e|sono|risulta|fu|was|is|it|wasn't|isn't|did|does|there)\b/.test(rest) ? "no" : null;
    else if (w === "non") v = punct || /^\s+(?:e|ha|c'e|esiste|risulta|si|sono|fu|era|stato|stata)\b/.test(rest) ? "no" : null;
    if (v) atoms.push({ kind: "yesno", value: v, raw: w, at: m.index! + m[0].length - rest.length - w.length });
  }
  // accounts written without "@": the evidence's authors, names after "by"/"par"/"por"/"da"/"von",
  // and a capitalised name that posted something
  const have = (v: string) => atoms.some((a) => a.kind === "account" && a.value === v);
  if (opts.authors?.size) {
    for (const m of F.matchAll(/(?<![\p{L}\p{N}@.\/-])([a-z0-9][a-z0-9.-]{1,31}[a-z0-9])(?![\p{L}\p{N}])/gu)) {
      const w = m[1];
      if (opts.authors.has(w) && !isStopword(w) && w.length > 2 && !have(w)) atoms.push({ kind: "account", value: w, raw: w, at: m.index, loose: true });
    }
  }
  const bare = (name: string, at: number) => {
    const raw = name.replace(/[.-]+$/, "");
    const v = raw.toLowerCase();
    if (v.length < 3 || isStopword(v) || NOT_NAMES.has(v) || MONTHS[v] || NUMBER_WORDS[v] !== undefined || ENGLISH_UNITS[v] !== undefined || /^\d/.test(v) || have(v)) return;
    atoms.push({ kind: "account", value: v, raw, at, bare: true });
  };
  for (const m of C.matchAll(/\b(by|from|par|por|da|von|vom)\s+([A-Za-z][A-Za-z0-9.-]{1,31})/g)) {
    const [prep, name] = [m[1], m[2]];
    const capital = /^[A-Z]/.test(name);
    // German capitalises every noun ("von Katzen"): only a lower-case handle there; French, Spanish and Italian: only a capitalised name
    if ((prep === "von" || prep === "vom") && capital) continue;
    if ((prep === "par" || prep === "por" || prep === "da") && !capital) continue;
    // "inspired by medieval tapestries" is not a name: a lower-case word after "by"/"from" is one
    // after a posting verb ("posted by mallory"), or when the sentence goes on as after a name
    if ((prep === "by" || prep === "from") && !capital) {
      const before = F.slice(Math.max(0, m.index! - 14), m.index!);
      const after = F.slice(m.index! + m[0].length, m.index! + m[0].length + 12);
      if (!/(?:posted|published|made|created|drawn|painted|uploaded|reposted|shared|minted|submitted|owned)\s*$/.test(before) && !/^\s*(?:[.,;:!?)]|$|on\b|in\b|at\b|and\b|with\b|who\b|as\b)/.test(after)) continue;
    }
    bare(name, m.index! + m[0].length - name.length);
  }
  for (const m of C.matchAll(new RegExp(String.raw`(?:^|[.!?;]\s+)([A-Z][A-Za-z0-9.-]{2,31})\s+${POSTING_VERB}\b`, "g"))) bare(m[1], m.index! + m[0].indexOf(m[1]));
  return atoms.sort((a, b) => (a.at ?? 0) - (b.at ?? 0));
}

// ---- facts of the evidence ------------------------------------------------------------------------------

interface Facts {
  accounts: Set<string>;
  dates: Set<string>;
  months: Set<string>;
  years: Set<string>;
  nums: Map<NumberUnit, Set<number>>;
  titles: Set<string>;
  refs: Set<string>;
  words: Set<string>;
  /** concepts the evidence mentions, in any language, with their ancestors ("chats" is cat, a pet, an animal) */
  concepts: Set<string>;
  text: string;
  /** the text as title keys compare it (folded, punctuation as spaces) */
  textKey: string;
}

const emptyFacts = (): Facts => ({ accounts: new Set(), dates: new Set(), months: new Set(), years: new Set(), nums: new Map(), titles: new Set(), refs: new Set(), words: new Set(), concepts: new Set(), text: "", textKey: "" });

const ISO_DAY = /^(\d{4}-\d{2}-\d{2})(?:T|$)/;

/** Numeric fields that state facts, and what they count. Ids, scores, similarities, timestamps and distances are not facts. */
const NUMERIC_FIELDS: Record<string, NumberUnit> = {
  votes: "votes", net_votes: "votes", upvotes: "votes", likes: "votes",
  payout: "money", pending_payout: "money", total_payout: "money",
  n: "count", count: "count", total: "count", edits: "count", tied_total: "count", artworks: "count", posts: "count", authors: "count", hidden: "count", nw: "count", nl: "count",
  days: "duration", hours: "duration",
  width: "size", height: "size",
};
/** Fields whose text says who did something: the accounts it names are facts. Titles, tags and captions only name things. */
const UNTRUSTED_TEXT = new Set(["title", "ai_caption", "text_in_image", "tags", "concepts", "label", "heading", "caption", "subjects", "description"]);

function addNum(f: Facts, unit: NumberUnit, v: number) {
  if (!Number.isFinite(v)) return;
  if (!f.nums.has(unit)) f.nums.set(unit, new Set());
  const s = f.nums.get(unit)!;
  s.add(v);
  // a payout of 14.469 is also said "14.47", "14.5" or "14"
  if (unit === "money" && !Number.isInteger(v)) [Math.round(v * 100) / 100, Math.round(v * 10) / 10, Math.round(v)].forEach((x) => s.add(x));
}

function addDate(f: Facts, day: string) {
  f.dates.add(day);
  f.months.add(day.slice(0, 7));
  f.months.add(`*-${day.slice(5)}`);
  f.years.add(day.slice(0, 4));
}

/** Names (titles, headings, labels): their numbers are part of a name, not facts ("Page 2", "Problems Plus 3"). */
const NAME_FIELDS = new Set(["title", "heading", "label", "first"]);

/** Facts stated by a text (the same reading as the claims'). */
function textFacts(f: Facts, v: string, trustedAccounts: boolean, numbers = true) {
  for (const a of extractAtoms(v)) {
    if (!numbers && (a.kind === "number" || a.kind === "year")) continue;
    if (a.kind === "date") [a.value, ...(a.alts ?? [])].forEach((d) => addDate(f, d));
    else if (a.kind === "month") f.months.add(a.value);
    else if (a.kind === "year") f.years.add(a.value);
    else if (a.kind === "number") for (const x of [a.value, ...(a.alts ?? [])]) addNum(f, a.unit ?? "any", Number(x));
    else if (a.kind === "title") f.titles.add(a.value);
    else if (a.kind === "ref") (f.refs.add(a.value), trustedAccounts && f.accounts.add(a.value.slice(2, a.value.indexOf("/", 2))));
    else if (a.kind === "account" && trustedAccounts && !a.bare) f.accounts.add(a.value);
  }
}

/** Everything a card states, by kind. */
export function cardFacts(card: EvidenceCard): Facts {
  const f = emptyFacts();
  const texts: string[] = [];
  const result = card.type === "result" ? card : null;
  const metricUnit: NumberUnit = result?.details?.metric === "net_votes" ? "votes" : result?.details?.metric === "payout" ? "money" : "count";
  const visit = (v: unknown, key = "", depth = 0, trusted = true) => {
    if (depth > 6 || v === null || v === undefined) return;
    if (typeof v === "number") {
      const u = key === "value" ? metricUnit : NUMERIC_FIELDS[key];
      if (u) addNum(f, u, v);
      return;
    }
    if (typeof v === "boolean") return;
    if (typeof v === "string") {
      if (key === "image" || key === "sha256" || key === "phash" || key === "dhash" || key === "url" || key === "evidence_id" || key === "source" || key === "type" || key === "op" || key === "step" || key === "time_field" || key === "relation" || key === "kind" || key === "match") return;
      const d = ISO_DAY.exec(v);
      if (d) return addDate(f, d[1]);
      if (/^\d{4}-\d{2}$/.test(v)) return void f.months.add(v);
      if (/^\/@[a-z0-9][a-z0-9.-]{1,31}\/\S+$/.test(v)) {
        f.refs.add(v);
        f.accounts.add(v.slice(2, v.indexOf("/", 2)));
        return;
      }
      if (/^(author|first_author|winner|claimed|actual|w|l)$/.test(key) || (/^[a-z0-9][a-z0-9.-]{1,31}$/.test(v) && /author|tied|^a$|^b$/.test(key))) f.accounts.add(v);
      if (/^(title|label|first)$/.test(key)) f.titles.add(titleKey(v));
      texts.push(v);
      textFacts(f, v, trusted && !UNTRUSTED_TEXT.has(key), !NAME_FIELDS.has(key));
      return;
    }
    if (Array.isArray(v)) {
      for (const x of v) visit(x, key, depth + 1, trusted);
      return;
    }
    if (typeof v === "object") for (const [k, x] of Object.entries(v as Record<string, unknown>)) visit(x, k, depth + 1, trusted && !UNTRUSTED_TEXT.has(key));
  };
  if (card.type === "doc") {
    // a documentation excerpt states what its text, title and heading say (its retrieval score is not a fact)
    visit({ title: card.title, heading: card.heading, text: card.text });
  } else if (result) {
    // the sub-question is the user's words, not evidence ("… posted by Mallory?")
    const { answer, details, n, question: _q, over: _o, answer_type: _t, ...rest } = result;
    visit(rest);
    if (typeof answer === "number") addNum(f, result.answer_type === "duration" ? "duration" : result.answer_type === "value" ? metricUnit : "count", answer);
    else if (typeof answer === "string") visit(answer, result.answer_type === "author" ? "author" : result.answer_type === "post" ? "post" : "answer");
    if (typeof n === "number") addNum(f, "count", n);
    if (details) {
      const d = details as Record<string, unknown>;
      visit(d);
      // a duration in seconds is said in days or hours
      if (typeof d.seconds === "number") [Math.round(d.seconds / 86400), Math.floor(d.seconds / 86400), Math.round(d.seconds / 3600), Math.floor(d.seconds / 3600)].forEach((x) => addNum(f, "duration", x));
      // the groups of a grouping by author are accounts
      if (d.by === "author" && Array.isArray(d.groups)) for (const g of d.groups as Array<{ k?: string; n?: number }>) if (g?.k) f.accounts.add(g.k);
      if (Array.isArray(d.ranking)) for (const g of d.ranking as Array<{ author?: string }>) if (g?.author) f.accounts.add(g.author);
    }
  } else {
    visit(card as unknown as Record<string, unknown>);
  }
  if ("evidence_id" in card) f.refs.add(card.evidence_id);
  f.text = texts.join("\n");
  f.textKey = titleKey(f.text);
  for (const w of tokens(foldx(f.text), { keepHyphenated: false })) f.words.add(w);
  const ids = [...matchConcepts(f.text).map((m) => m.concept), ...((card.type === "artwork" || card.type === "post") && card.concepts ? card.concepts : [])];
  for (const id of ids) {
    f.concepts.add(id);
    for (const p of ancestors(id)) f.concepts.add(p);
  }
  return f;
}

function merge(fs: Facts[]): Facts {
  const out = emptyFacts();
  for (const f of fs) {
    for (const k of ["accounts", "dates", "months", "years", "titles", "refs", "words", "concepts"] as const) for (const x of f[k]) out[k].add(x);
    for (const [u, s] of f.nums) for (const x of s) addNum(out, u, x);
    out.text += `${f.text}\n`;
  }
  out.textKey = titleKey(out.text);
  return out;
}

/**
 * The numbers of the facts a number of this unit may be: votes with votes, days with days, a
 * percentage with percentages; a count also with the evidence's bare numbers ("cat (4)"); a bare
 * number with anything but a percentage.
 */
function numbersFor(f: Facts, unit: NumberUnit): number[] {
  const get = (u: NumberUnit) => [...(f.nums.get(u) ?? [])];
  if (unit === "any") return [...f.nums.entries()].filter(([u]) => u !== "percent").flatMap(([, s]) => [...s]);
  if (unit === "count") return [...get("count"), ...get("any")];
  return get(unit);
}

function hasAtom(f: Facts, a: Atom): boolean {
  switch (a.kind) {
    case "account":
      return f.accounts.has(a.value) || (!!a.bare && f.words.has(a.value));
    case "date":
      return [a.value, ...(a.alts ?? [])].some((d) => f.dates.has(d));
    case "month":
      return f.months.has(a.value);
    case "year":
      return f.years.has(a.value);
    case "number": {
      const ns = numbersFor(f, a.unit ?? "any");
      return [a.value, ...(a.alts ?? [])].map(Number).some((v) => ns.some((n) => Math.abs(n - v) < 1e-6));
    }
    case "title":
      return f.titles.has(a.value) || (a.value.length >= 4 && ` ${f.textKey} `.includes(` ${a.value} `));
    case "ref":
      return f.refs.has(a.value);
    default:
      return true;
  }
}

/** The atoms whose truth the evidence decides; superlatives and yes/no are checked against the results. */
const FACTUAL: AtomKind[] = ["account", "date", "month", "year", "number", "title", "ref"];

const describe = (a: Atom) => (a.kind === "account" ? `account ${a.bare ? a.raw : `@${a.value}`}` : a.kind === "number" && a.unit && a.unit !== "any" && a.unit !== "count" ? `${a.unit} ${a.raw}` : `${a.kind} ${a.raw}`);

// ---- contradictions with the deterministic results --------------------------------------------------------

interface ResultView {
  id: string;
  op: string;
  answer: unknown;
  answerType: string;
  subjectWords: Set<string>;
  accounts: Set<string>;
  /** the post it answers with, and its title */
  post?: string;
  title?: string;
  /** the day of a "first" / "latest" (details.time) */
  day?: string;
}

function resultViews(cards: EvidenceCard[]): ResultView[] {
  const out: ResultView[] = [];
  const titles = new Map<string, string>();
  for (const c of cards) if (c.type === "artwork" || c.type === "post") titles.set(c.path, c.title);
  for (const c of cards) {
    if (c.type !== "result") continue;
    const d = (c.details ?? {}) as Record<string, any>;
    const accounts = new Set<string>();
    if (typeof c.answer === "string" && c.answer_type === "author") accounts.add(c.answer);
    if (typeof c.answer === "string" && c.answer_type === "post") accounts.add(c.answer.slice(2, c.answer.indexOf("/", 2)));
    for (const k of ["first_author", "author", "winner"]) if (typeof d[k] === "string") accounts.add(d[k]);
    for (const t of Array.isArray(d.tied) ? d.tied : []) accounts.add(String(t));
    const post = c.answer_type === "post" && typeof c.answer === "string" ? c.answer : typeof d.post === "string" ? d.post : undefined;
    const title = typeof d.title === "string" ? d.title : post ? titles.get(post) : undefined;
    const when = c.op === "find_first" || c.op === "find_latest" ? d.time : c.op === "resolve" || c.op === "identify" ? d.created : undefined;
    const day = typeof when === "string" ? ISO_DAY.exec(when)?.[1] : undefined;
    out.push({ id: c.evidence_id, op: c.op, answer: c.answer, answerType: c.answer_type, subjectWords: new Set(tokens(foldx(`${c.question ?? ""} ${c.text}`), { keepHyphenated: false }).filter((w) => w.length > 2 && !isStopword(w))), accounts, ...(post ? { post } : {}), ...(title ? { title } : {}), ...(day ? { day } : {}) });
  }
  return out;
}

/** Where the clause that holds position `at` starts and ends (";", ",", "but", "mais", "aber", "pero", "ma", "while"…). */
function clauseOf(text: string, at: number): [number, number] {
  const F = foldx(text);
  const re = /[;,—–]|\b(?:but|while|whereas|although|though|however|mais|alors que|tandis que|aber|wahrend|jedoch|pero|mientras|aunque|ma|mentre|pero|sino)\b|\.\s/g;
  let start = 0;
  let end = F.length;
  for (const m of F.matchAll(re)) {
    if (m.index! + m[0].length <= at) start = m.index! + m[0].length;
    else if (m.index! > at) {
      end = m.index!;
      break;
    }
  }
  return [start, end];
}

/** A reason the claim contradicts a deterministic result, or null. */
function contradiction(text: string, atoms: Atom[], results: ResultView[]): string | null {
  if (!results.length) return null;
  const words = new Set(tokens(foldx(text), { keepHyphenated: false }));
  const related = (r: ResultView) => results.length === 1 || [...r.subjectWords].some((w) => words.has(w));
  // accounts written as accounts first; an author's name met as a plain word only when nothing else names one
  const strong = atoms.filter((a) => a.kind === "account" && !a.loose);
  const accounts = strong.length ? strong : atoms.filter((a) => a.kind === "account");
  const within = (a: Atom, [s, e]: [number, number]) => (a.at ?? -1) >= s && (a.at ?? -1) < e;
  // "the first …" names the account of its own clause ("@bob posted the first cat; @alice reposted it")
  for (const sup of atoms.filter((a) => a.kind === "superlative")) {
    const op = sup.value === "first" ? "find_first" : sup.value === "latest" ? "find_latest" : "count_by_author";
    const rs = results.filter((r) => (r.op === op || (op === "count_by_author" && (r.op === "top" || r.op === "group"))) && related(r));
    if (!rs.length) continue;
    const clause = clauseOf(text, sup.at ?? 0);
    const named = accounts.find((a) => within(a, clause));
    const withAccounts = rs.filter((r) => r.accounts.size);
    if (named && withAccounts.length && !withAccounts.some((r) => r.accounts.has(named.value))) return `"${sup.raw}" names ${named.bare ? named.raw : `@${named.value}`}, but ${withAccounts[0].id} gives ${[...withAccounts[0].accounts].map((a) => "@" + a).join(", ")}`;
    // "the first cat artwork is “Cat nap”": another post than the result's
    const titles = atoms.filter((a) => a.kind === "title" && within(a, clause));
    const refs = atoms.filter((a) => a.kind === "ref" && within(a, clause));
    const r = rs.find((x) => x.title || x.post);
    if (r && (titles.length || refs.length) && !titles.some((t) => r.title && t.value === titleKey(r.title)) && !refs.some((x) => x.value === r.post)) return `"${sup.raw}" names ${titles[0]?.raw ?? refs[0]?.raw}, but ${r.id} gives ${r.title ? `“${r.title}”` : r.post}`;
    // "@alice posted the first cat on 2026-09-03": another day than the result's
    const day = rs.map((x) => x.day).find((x) => !!x);
    const dates = atoms.filter((a) => (a.kind === "date" || (a.kind === "month" && a.value.startsWith("*-"))) && within(a, clause));
    if (day && dates.length && !dates.some((a) => (a.kind === "date" ? [a.value, ...(a.alts ?? [])].includes(day) : a.value === `*-${day.slice(5)}`))) return `"${sup.raw}" is dated ${dates[0].raw}, but ${rs[0].id} gives ${day}`;
  }
  // yes / no against the one boolean result
  const yn = atoms.find((a) => a.kind === "yesno");
  if (yn) {
    const bools = results.filter((r) => r.answerType === "boolean" && typeof r.answer === "boolean");
    if (bools.length === 1 && (bools[0].answer ? "yes" : "no") !== yn.value) return `says "${yn.raw}", but ${bools[0].id} answers ${bools[0].answer ? "yes" : "no"}`;
  }
  const nums = (units: NumberUnit[]) => atoms.filter((a) => a.kind === "number" && units.includes(a.unit ?? "any"));
  const readings = (a: Atom) => [a.value, ...(a.alts ?? [])].map(Number);
  // a count of artworks the result does not give ("3 œuvres" against a count of 2)
  const counts = results.filter((r) => r.answerType === "count" && typeof r.answer === "number" && related(r));
  if (counts.length === 1) {
    const n = counts[0].answer as number;
    const said = nums(["count"]);
    if (said.length && !said.some((a) => readings(a).includes(n))) return `gives ${said[0].raw}, but ${counts[0].id} counts ${n}`;
  }
  // a duration the result does not give
  const durations = results.filter((r) => r.answerType === "duration" && typeof r.answer === "number" && related(r));
  if (durations.length === 1) {
    const said = nums(["duration"]);
    if (said.length && !said.some((a) => readings(a).includes(durations[0].answer as number))) return `gives ${said[0].raw}, but ${durations[0].id} gives ${durations[0].answer}`;
  }
  // another date than the result's ("the first cat artwork was posted on 2026-09-03")
  const dates = results.filter((r) => r.answerType === "date" && typeof r.answer === "string" && related(r));
  if (dates.length === 1) {
    const day = String(dates[0].answer).slice(0, 10);
    const said = atoms.filter((a) => a.kind === "date" || (a.kind === "month" && a.value.startsWith("*-")));
    if (said.length && !said.some((a) => (a.kind === "date" ? [a.value, ...(a.alts ?? [])].includes(day) : a.value === `*-${day.slice(5)}`))) return `gives ${said[0].raw}, but ${dates[0].id} gives ${day}`;
  }
  // another author or day for a named title ("“Lake” was posted by @alice" when @bob posted it)
  for (const r of results.filter((x) => (x.op === "resolve" || x.op === "identify") && x.title)) {
    const t = atoms.find((a) => a.kind === "title" && a.value === titleKey(r.title!));
    if (!t) continue;
    const clause = clauseOf(text, t.at ?? 0);
    const inClause = accounts.filter((a) => within(a, clause));
    if (r.accounts.size && inClause.length && !inClause.some((a) => r.accounts.has(a.value))) return `names ${inClause[0].bare ? inClause[0].raw : `@${inClause[0].value}`} for “${r.title}”, but ${r.id} gives ${[...r.accounts].map((a) => "@" + a).join(", ")}`;
    const dates = atoms.filter((a) => a.kind === "date" && within(a, clause));
    if (r.day && dates.length && !dates.some((a) => [a.value, ...(a.alts ?? [])].includes(r.day!))) return `dates “${r.title}” ${dates[0].raw}, but ${r.id} gives ${r.day}`;
  }
  return null;
}

/**
 * Votes and payouts belong to a post: "@alice's cat has 1 vote" needs a card of @alice's (or of
 * the title it names) with 1 vote, not 1 vote anywhere.
 */
function pairProblem(atoms: Atom[], byId: Map<string, Facts>, cards: Map<string, { author?: string; title?: string }>): string | null {
  const accounts = atoms.filter((a) => a.kind === "account" && !a.loose).map((a) => a.value);
  const titles = atoms.filter((a) => a.kind === "title").map((a) => a.value);
  if (!accounts.length && !titles.length) return null;
  for (const a of atoms.filter((x) => x.kind === "number" && (x.unit === "votes" || x.unit === "money"))) {
    const holders = [...byId.entries()].filter(([, f]) => hasAtom(f, a)).map(([id]) => id);
    if (!holders.length) continue;
    const owns = holders.some((id) => {
      const c = cards.get(id);
      if (!c?.author && !c?.title) return true; // a result or other card: its own subject
      return (c.author && accounts.includes(c.author)) || (c.title && titles.includes(titleKey(c.title)));
    });
    if (!owns) return `${a.unit} ${a.raw} belongs to another post than ${accounts.length ? `@${accounts[0]}'s` : "the one it names"}`;
  }
  return null;
}

// ---- verification ------------------------------------------------------------------------------------

/**
 * The share of a sentence's content words the evidence has: the same word, its singular, or a
 * word naming a concept the evidence mentions in any language ("Elle dessine surtout des chats"
 * over cards tagged "cat").
 */
function overlap(text: string, base: Facts): number {
  const toks = matchTokens(text.replace(CITATION, " "));
  const covered = new Set<number>();
  for (const m of matchConcepts(text.replace(CITATION, " "))) if (base.concepts.has(m.concept)) for (let i = m.start; i < m.end; i++) covered.add(i);
  const seen = new Set<string>();
  let n = 0;
  let hit = 0;
  toks.forEach((raw, i) => {
    const t = foldx(raw);
    if (t.length <= 2 || isStopword(t) || /^\d+$/.test(t) || seen.has(t)) return;
    seen.add(t);
    n++;
    if (covered.has(i) || base.words.has(t) || base.words.has(singular(t))) hit++;
  });
  return n ? hit / n : 1;
}

export interface VerifyOptions {
  /** expert mode: a qualified claim does not reach the answer */
  strict?: boolean;
  /** account names of the evidence, recognised without their @ */
  authors?: Set<string>;
  /**
   * Claims without a factual atom are judged by their words (default). Off for documentation
   * answers: an answer in French over English excerpts shares few words with them, so such a
   * sentence is supported by citing an excerpt, and qualified without one.
   */
  wordOverlap?: boolean;
  /**
   * A quoted label the evidence does not contain makes the claim qualified, not unsupported
   * (documentation answers: « Publier » is the French for the "Publish" the excerpt quotes).
   * The numbers inside it are still checked.
   */
  softTitles?: boolean;
  /**
   * Long-form answers (compose.ts): a sentence without a factual atom that cites a card it was
   * given rests on that card and is supported by it, whatever its words (a French sentence over
   * English captions shares few words with them); one that cites nothing is judged by its words
   * as usual.
   */
  citedInterpretations?: boolean;
}

type CheckContext = { byId: Map<string, Facts>; all: Facts; results: ResultView[]; owners: Map<string, { author?: string; title?: string }>; authors?: Set<string>; wordOverlap: boolean; softTitles: boolean; citedInterpretations: boolean };

function check(text: string, cited: string[], kind: Claim["kind"], ctx: CheckContext): { status: ClaimStatus; supported_by: string[]; problems: string[]; cited_correctly: boolean } {
  const atoms = extractAtoms(text, { authors: ctx.authors });
  const citedFacts = merge(cited.map((id) => ctx.byId.get(id)).filter((x): x is Facts => !!x));
  const problems: string[] = [];
  const soft: string[] = [];
  const contra = contradiction(text, atoms, ctx.results);
  if (contra) return { status: "contradicted", supported_by: [], problems: [contra], cited_correctly: false };
  // a number inside a quoted title is part of the name when the title is the evidence's
  const knownTitles = new Set(atoms.filter((a) => a.kind === "title" && [...ctx.byId.values()].some((f) => hasAtom(f, a))).map((a) => a.value));
  const factual = atoms.filter((a) => FACTUAL.includes(a.kind) && !(a.inTitle && knownTitles.has(a.inTitle)));
  const supportedBy = new Set<string>();
  let citedOk = true;
  for (const a of factual) {
    if (hasAtom(citedFacts, a)) {
      for (const id of cited) if (ctx.byId.get(id) && hasAtom(ctx.byId.get(id)!, a)) supportedBy.add(id);
      continue;
    }
    const where = [...ctx.byId.entries()].filter(([, f]) => hasAtom(f, a)).map(([id]) => id);
    if (where.length) {
      citedOk = false;
      supportedBy.add(where[0]);
      continue;
    }
    (a.kind === "title" && ctx.softTitles ? soft : problems).push(`${describe(a)} is not in the evidence`);
  }
  if (!problems.length) {
    const pair = pairProblem(atoms, ctx.byId, ctx.owners);
    if (pair) problems.push(pair);
  }
  if (problems.length) return { status: "unsupported", supported_by: [...supportedBy], problems, cited_correctly: false };
  if (soft.length) return { status: "qualified", supported_by: [...supportedBy, ...cited.filter((id) => ctx.byId.has(id))].filter((x, i, a) => a.indexOf(x) === i), problems: soft, cited_correctly: false };
  if (factual.length) return { status: "supported", supported_by: [...supportedBy], problems, cited_correctly: citedOk && cited.length > 0 };
  // no factual atom: an interpretation
  const known = cited.filter((id) => ctx.byId.has(id));
  if (ctx.citedInterpretations && known.length) {
    // it rests on the card it cites: supported when at least a quarter of its words are the card's
    // (a French sentence over English captions still shares its concepts), qualified otherwise
    const share = overlap(text, citedFacts);
    if (share >= 0.25) return { status: "supported", supported_by: known, problems, cited_correctly: true };
    return { status: "qualified", supported_by: known, problems: [`only ${Math.round(share * 100)} % of its words are in the evidence it cites`], cited_correctly: true };
  }
  if (!ctx.wordOverlap) {
    // documentation: it rests on the excerpt it cites
    if (known.length) return { status: "supported", supported_by: known, problems, cited_correctly: true };
    return { status: "qualified", supported_by: [], problems: ["cites no excerpt"], cited_correctly: false };
  }
  // supported as far as its words are the evidence's
  const share = overlap(text, cited.length ? citedFacts : ctx.all);
  if (share >= 0.5) return { status: "supported", supported_by: cited, problems, cited_correctly: cited.length > 0 };
  return { status: kind === "inference" ? "qualified" : "unsupported", supported_by: cited, problems: [`only ${Math.round(share * 100)} % of its words are in the evidence`], cited_correctly: false };
}

/** Verify a reasoning reply against the cards it was given. */
export function verifyClaims(reply: Pick<ReasoningResponse, "answer" | "claims" | "rationale">, cards: EvidenceCard[], opts: VerifyOptions = {}): ClaimVerification {
  const byId = new Map(cards.map((c) => [c.evidence_id, cardFacts(c)]));
  const all = merge([...byId.values()]);
  const results = resultViews(cards);
  const owners = new Map<string, { author?: string; title?: string }>();
  for (const c of cards) if (c.type === "artwork" || c.type === "post") owners.set(c.evidence_id, { author: c.author, title: c.title });
  const ctx: CheckContext = { byId, all, results, owners, authors: opts.authors, wordOverlap: opts.wordOverlap !== false, softTitles: !!opts.softTitles, citedInterpretations: !!opts.citedInterpretations };
  const strictly = (s: ClaimStatus): ClaimStatus => (opts.strict && s === "qualified" ? "unsupported" : s);
  const claims: VerifiedClaim[] = reply.claims.map((c) => {
    const r = check(c.text, c.evidence, c.kind, ctx);
    const status = strictly(r.status);
    const confidence = status === "qualified" ? Math.min(0.5, c.confidence ?? 0.5) : c.confidence;
    return { ...c, ...(confidence !== undefined ? { confidence } : {}), status, supported_by: r.supported_by, problems: r.problems, cited_correctly: r.cited_correctly };
  });
  const citedAll = [...new Set(reply.claims.flatMap((c) => c.evidence))];
  const a = check(reply.answer, citedAll.length ? citedAll : [...byId.keys()], "fact", ctx);
  const rationale = reply.rationale ? { status: strictly(check(reply.rationale, citedAll.length ? citedAll : [...byId.keys()], "inference", ctx).status) } : null;
  const counts = { supported: 0, qualified: 0, unsupported: 0, contradicted: 0 } as Record<ClaimStatus, number>;
  for (const c of claims) counts[c.status]++;
  const supported = claims.filter((c) => c.status === "supported");
  return {
    claims,
    answer: { status: strictly(a.status), problems: a.problems, supported_by: a.supported_by },
    rationale,
    egs: claims.length ? Math.round((counts.supported / claims.length) * 1000) / 1000 : a.status === "supported" ? 1 : 0,
    citation_accuracy: supported.length ? Math.round((supported.filter((c) => c.cited_correctly).length / supported.length) * 1000) / 1000 : 0,
    counts,
  };
}

/** The claims that may appear in an answer: supported ones, and qualified ones outside the strict mode. */
export function keptClaims(v: ClaimVerification, strict = false): VerifiedClaim[] {
  return v.claims.filter((c) => c.status === "supported" || (!strict && c.status === "qualified"));
}

/**
 * The names a text uses that the cards do not contain: accounts, quoted titles and post paths. A
 * follow-up question about @nobody, or about a title no card bears, names nothing the reader can
 * ask about (compose.ts drops it).
 */
export function unknownNames(text: string, cards: EvidenceCard[], opts: { authors?: Set<string>; numbers?: boolean } = {}): string[] {
  const all = merge(cards.map(cardFacts));
  const kinds: AtomKind[] = opts.numbers ? ["account", "title", "ref", "number", "date", "month", "year"] : ["account", "title", "ref"];
  return extractAtoms(text, opts)
    .filter((a) => kinds.includes(a.kind))
    .filter((a) => !hasAtom(all, a) && !(a.kind === "account" && opts.authors?.has(a.value)))
    .map(describe);
}

/**
 * Words that accuse: theft, fraud, plagiarism, bans, crime, in the five languages. A sentence of a
 * model that names an account and uses one of them, when no card uses it, states an accusation
 * the evidence does not (compose.ts removes it). "Copy" and "repost" are not here: the index
 * itself says when an image was posted again.
 */
export const ACCUSATION =
  /\b(scam(?:mer|s|med)?|thie(?:f|ves)|st(?:eal|ole|olen)s?|fraud(?:ster|ulent)?|plagiari[sz](?:ed|m|st)?|bann?ed|illegal|criminal|counterfeit|cheat(?:er|ed|s)?|abus(?:e|er|ive)|harass(?:ed|ment)?|voleur|volé|vole|arnaque|arnaqueur|escroc|plagiat|plagié|banni|illégal|criminel|tricheur|betrug|betrüger|dieb|gestohlen|plagiat|gesperrt|verboten|kriminell|betrüg\w*|estafa|estafador|ladr[oó]n|rob[oó]|robado|plagio|baneado|ilegal|delincuente|tramposo|truffa|truffatore|ladro|rubato|plagio|bannato|illegale|criminale|imbroglione)\b/iu;

/** The accusing words of a text that no card uses (folded). */
export function accusations(text: string, cards: EvidenceCard[]): string[] {
  const words = new Set<string>();
  for (const c of cards) for (const w of cardFacts(c).words) words.add(w);
  const out: string[] = [];
  for (const m of foldx(text).matchAll(new RegExp(ACCUSATION.source, "giu"))) if (!words.has(m[1].toLowerCase())) out.push(m[1]);
  return [...new Set(out)];
}

// ---- agreement with the deterministic answer ----------------------------------------------------------------

/** Operators whose answer is a set of posts, not one value to restate. */
const NO_VALUE_OPS = new Set(["search", "similar", "duplicates"]);

/**
 * Does a text shown after deterministic results state their own values (spec §27)? For each
 * result: an author answer names that account (the first account it names); a date answer gives
 * that day; a count, duration or value gives that number; a yes/no answer says "Yes"/"No" (in
 * the question's language) at the start of a sentence or clause, the first one it says; a post
 * answer names the post or its title. A result without a single value (a list of posts, a tie
 * without a winner, nothing found) has nothing to restate.
 */
export function agreesWithResult(text: string, shown: ResultCard[], cards: EvidenceCard[], opts: { authors?: Set<string> } = {}): { ok: boolean; problems: string[] } {
  const atoms = extractAtoms(text, opts);
  const problems: string[] = [];
  const views = new Map(resultViews(cards).map((v) => [v.id, v]));
  const yesno = atoms.filter((a) => a.kind === "yesno");
  let boolIndex = 0;
  for (const r of shown) {
    if (NO_VALUE_OPS.has(r.op) || r.answer === null || r.answer === undefined) continue;
    const d = (r.details ?? {}) as Record<string, unknown>;
    switch (r.answer_type) {
      case "author": {
        const want = new Set([String(r.answer).toLowerCase(), ...(Array.isArray(d.tied) ? (d.tied as unknown[]).map((t) => String(t).toLowerCase()) : [])]);
        const strong = atoms.filter((a) => a.kind === "account" && !a.loose);
        const accs = strong.length ? strong : atoms.filter((a) => a.kind === "account");
        if (!accs.some((a) => want.has(a.value))) problems.push(`does not name @${r.answer} (${r.evidence_id})`);
        else if (!want.has(accs[0].value)) problems.push(`names ${accs[0].bare ? accs[0].raw : `@${accs[0].value}`} first, but ${r.evidence_id} gives @${r.answer}`);
        break;
      }
      case "date": {
        const day = String(r.answer).slice(0, 10);
        const ok = atoms.some((a) => (a.kind === "date" && [a.value, ...(a.alts ?? [])].includes(day)) || (a.kind === "month" && a.value === `*-${day.slice(5)}`));
        if (!ok) problems.push(`does not give ${r.evidence_id}'s date, ${day}`);
        break;
      }
      case "count":
      case "duration":
      case "value": {
        if (typeof r.answer !== "number") break;
        const target = r.answer;
        const close = (v: number) => Math.abs(v - target) < 1e-6 || (!Number.isInteger(target) && [Math.round(target * 100) / 100, Math.round(target * 10) / 10, Math.round(target)].includes(v));
        const ok = atoms.some((a) => a.kind === "number" && a.unit !== "percent" && [a.value, ...(a.alts ?? [])].map(Number).some(close)) || (target === 0 && atoms.some((a) => a.kind === "yesno" && a.value === "no"));
        if (!ok) problems.push(`does not give ${r.evidence_id}'s ${r.answer_type === "duration" ? "duration" : "number"}, ${target}`);
        break;
      }
      case "boolean": {
        const want = r.answer ? "yes" : "no";
        const said = yesno[boolIndex++];
        if (!said) problems.push(`does not say ${want === "yes" ? "Yes" : "No"} (${r.evidence_id})`);
        else if (said.value !== want) problems.push(`says "${said.raw}", but ${r.evidence_id} answers ${want}`);
        break;
      }
      case "post": {
        const path = String(r.answer);
        const title = views.get(r.evidence_id)?.title;
        const key = title ? titleKey(title) : "";
        const ok = atoms.some((a) => a.kind === "ref" && a.value === path) || (!!key && (atoms.some((a) => a.kind === "title" && a.value === key) || (key.length >= 3 && ` ${titleKey(text)} `.includes(` ${key} `))));
        if (!ok) problems.push(`does not name ${r.evidence_id}'s post, ${title ? `“${title}”` : path}`);
        break;
      }
      default:
        break;
    }
  }
  return { ok: !problems.length, problems };
}

export type { Lang };
