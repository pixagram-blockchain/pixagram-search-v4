// The deterministic digest of an answer (v4.8): what the index knows around the answer, written
// from computed values only, in the question's language, with no model call. It is the part of
// a long-form answer that is always there — in fast mode, when the model is unavailable, while a
// deferred elaboration is still running — and the part that can never be wrong about the index.
//
//   facts      one sentence per deciding post: title, author, date, tags, votes, payout, where
//              the image first appeared, edits, deletion, later posts of the same image, the AI
//              caption (labelled as a model's description)
//   overview   the evidence set as a whole: how many posts, by whom, when, the most voted, the
//              frequent tags
//   caveats    lower bounds, ties, inferred histories, deleted posts, hidden posts, conflicts
//   follow_ups questions the reader may ask next, from templates per question family, each of
//              which the rules planner answers (test/v4-rich.test.ts checks every template in the
//              five languages); the ones that repeat the question's own intent are left out
//   searches   searches for the box: the subject, the subject with the evidence's main colour,
//              the accounts, the frequent tags
//
// compose.ts puts the digest after the model's body (or alone), and the UI may show its parts
// separately (sections.facts, sections.overview …).

import type { Lang } from "../lib/text";
import { fold, singular } from "../lib/text";
import { parseList } from "./retrieval";
import { answerLang, quote, subjectLabel, subjectPlural, type AnswerLang } from "./answer-text";
import { artworkCard, type ArtworkCard, type ResultCard } from "./evidence";
import type { HistoryFacts, Verified } from "./operators";
import type { QueryPlan } from "./planner";
import type { QueryProgram } from "./query-planner";
import type { StepOutcome } from "./executor";

export interface DigestStats {
  /** posts the answer is about (an exact count when the index computed one) */
  posts: number;
  /** how many of them the digest could read (the rows at hand) */
  read: number;
  authors: Array<{ author: string; n: number }>;
  from?: string;
  to?: string;
  top_voted?: { path: string; title: string; author: string; votes: number };
  tags: Array<{ tag: string; n: number }>;
  colors: Array<{ color: string; n: number }>;
  /** from, to and top_voted hold for the whole set (read from every row, or exactly by SQL); tags and colours are the rows read */
  exact: boolean;
}

export interface Digest {
  facts: string[];
  overview: string[];
  caveats: string[];
  follow_ups: string[];
  searches: string[];
  stats: DigestStats;
  /** the evidence ids the facts are about, in order */
  about: string[];
}

export interface DigestInput {
  lang: Lang;
  question: string;
  plan: QueryPlan;
  program: QueryProgram;
  outcomes: StepOutcome[];
  final: StepOutcome;
  /** the artwork cards of the answer (verified ones only) */
  cards: ArtworkCard[];
  results: ResultCard[];
  histories: Map<string, HistoryFacts>;
  /** the conflict sentences about the deciding evidence, in the answer's language (verifier.ts conflictSentence) */
  conflicts: string[];
  /** posts the nsfw setting hid from the answer */
  hidden?: number;
  /** rows of the set the answer is about when the operator kept none (an exact count over the filters): its newest */
  rows?: Array<Record<string, any>>;
  /** the set's oldest and most voted posts, read exactly (SQL), when the rows are only its newest */
  extremes?: { first?: Record<string, any> | null; top?: Record<string, any> | null };
}

type L<T> = Record<AnswerLang, T>;
const day = (iso: string | undefined) => (iso ? iso.slice(0, 10) : "");
const r2 = (x: number) => Math.round(x * 100) / 100;

const SET_OPS = new Set(["count", "count_by_author", "search", "top", "group", "aggregate", "similar", "duplicates"]);
const MAX_FACTS = 4;
const MAX_LIST = 3;

// ---- words -----------------------------------------------------------------------------------------

const W: L<{
  by: string; posted: string; postedBlog: string; tagged: string; votes: (n: number) => string; pxs: string;
  firstSeen: (d: string, p: string) => string; edited: (n: number, d: string) => string; deleted: (d: string) => string;
  again: (p: string, d: string) => string; caption: string; textIn: string; and: string;
  artists: (n: number) => string; artworks: (n: number) => string; posts: (n: number) => string; mostBy: string;
  between: (a: string, b: string) => string; on: (d: string) => string; mostVoted: (t: string, a: string, v: number) => string;
  tags: string; colours: string; shown: (k: number) => string; ofNewest: (k: number) => string;
  lowerBound: string; ties: (who: string) => string; inferred: (p: string) => string; deletedPost: (p: string, d: string) => string;
  hiddenPosts: (n: number) => string; captions: string; conflict: (note: string) => string; blogPost: string;
}> = {
  en: {
    by: "by", posted: "posted on", postedBlog: "published on", tagged: "tagged", votes: (n) => `${n} vote${n === 1 ? "" : "s"}`, pxs: "PXS",
    firstSeen: (d, p) => `the image first appeared on ${d} in ${p}`, edited: (n, d) => `edited ${n} time${n === 1 ? "" : "s"}, last on ${d}`, deleted: (d) => `deleted${d ? ` on ${d}` : ""}`,
    again: (p, d) => `the same image was posted again in ${p} (${d})`, caption: "described by the AI as", textIn: "text in the image", and: "and",
    artists: (n) => `${n} artist${n === 1 ? "" : "s"}`, artworks: (n) => `${n} artwork${n === 1 ? "" : "s"}`, posts: (n) => `${n} post${n === 1 ? "" : "s"}`, mostBy: "most by",
    between: (a, b) => `Posted between ${a} and ${b}.`, on: (d) => `Posted on ${d}.`, mostVoted: (t, a, v) => `The most voted is ${t} by @${a} (${v} vote${v === 1 ? "" : "s"}).`,
    tags: "Frequent tags", colours: "Main colours", shown: (k) => `(tags and colours read from the ${k} newest)`, ofNewest: (k) => `Of the ${k} newest:`,
    lowerBound: "The count is a lower bound: the vector search budget was spent, so more matching posts may exist.",
    ties: (who) => `Several accounts tie: ${who}.`, inferred: (p) => `The image history of ${p} is inferred from dates, not read from chain operations.`,
    deletedPost: (p, d) => `${p} was deleted${d ? ` on ${d}` : ""}: it is evidence of what was posted, not a post anyone can see.`,
    hiddenPosts: (n) => `${n} post${n === 1 ? "" : "s"} hidden by the NSFW setting ${n === 1 ? "was" : "were"} left out.`,
    captions: "AI captions describe what a model sees in an image, not what the author wrote.", conflict: (note) => `The evidence disagrees: ${note}`, blogPost: "blog post",
  },
  fr: {
    by: "de", posted: "publié le", postedBlog: "publié le", tagged: "tags :", votes: (n) => `${n} vote${n === 1 ? "" : "s"}`, pxs: "PXS",
    firstSeen: (d, p) => `l'image est apparue pour la première fois le ${d} dans ${p}`, edited: (n, d) => `modifié ${n} fois, la dernière le ${d}`, deleted: (d) => `supprimé${d ? ` le ${d}` : ""}`,
    again: (p, d) => `la même image a été publiée à nouveau dans ${p} (${d})`, caption: "décrit par l'IA comme", textIn: "texte dans l'image", and: "et",
    artists: (n) => `${n} artiste${n === 1 ? "" : "s"}`, artworks: (n) => `${n} œuvre${n === 1 ? "" : "s"}`, posts: (n) => `${n} post${n === 1 ? "" : "s"}`, mostBy: "surtout de",
    between: (a, b) => `Publiées entre le ${a} et le ${b}.`, on: (d) => `Publiées le ${d}.`, mostVoted: (t, a, v) => `La plus votée est ${t} de @${a} (${v} vote${v === 1 ? "" : "s"}).`,
    tags: "Tags fréquents", colours: "Couleurs principales", shown: (k) => `(tags et couleurs lus sur les ${k} plus récentes)`, ofNewest: (k) => `Parmi les ${k} plus récentes :`,
    lowerBound: "Le compte est une borne inférieure : le budget de recherche vectorielle a été épuisé, d'autres posts peuvent correspondre.",
    ties: (who) => `Plusieurs comptes sont à égalité : ${who}.`, inferred: (p) => `L'historique de l'image de ${p} est déduit des dates, pas lu dans les opérations de la chaîne.`,
    deletedPost: (p, d) => `${p} a été supprimé${d ? ` le ${d}` : ""} : c'est une trace de ce qui a été publié, pas un post visible.`,
    hiddenPosts: (n) => `${n} post${n === 1 ? "" : "s"} masqué${n === 1 ? "" : "s"} par le réglage NSFW ${n === 1 ? "a été laissé" : "ont été laissés"} de côté.`,
    captions: "Les descriptions IA disent ce qu'un modèle voit dans l'image, pas ce que l'auteur a écrit.", conflict: (note) => `Les éléments se contredisent : ${note}`, blogPost: "article",
  },
  de: {
    by: "von", posted: "gepostet am", postedBlog: "veröffentlicht am", tagged: "Tags:", votes: (n) => `${n} Stimme${n === 1 ? "" : "n"}`, pxs: "PXS",
    firstSeen: (d, p) => `das Bild erschien zuerst am ${d} in ${p}`, edited: (n, d) => `${n}-mal bearbeitet, zuletzt am ${d}`, deleted: (d) => `gelöscht${d ? ` am ${d}` : ""}`,
    again: (p, d) => `dasselbe Bild wurde erneut in ${p} gepostet (${d})`, caption: "von der KI beschrieben als", textIn: "Text im Bild", and: "und",
    artists: (n) => `${n} Künstler`, artworks: (n) => `${n} Kunstwerk${n === 1 ? "" : "e"}`, posts: (n) => `${n} Beitr${n === 1 ? "ag" : "äge"}`, mostBy: "die meisten von",
    between: (a, b) => `Gepostet zwischen dem ${a} und dem ${b}.`, on: (d) => `Gepostet am ${d}.`, mostVoted: (t, a, v) => `Die meisten Stimmen hat ${t} von @${a} (${v} Stimme${v === 1 ? "" : "n"}).`,
    tags: "Häufige Tags", colours: "Hauptfarben", shown: (k) => `(Tags und Farben aus den ${k} neuesten)`, ofNewest: (k) => `Unter den ${k} neuesten:`,
    lowerBound: "Die Zahl ist eine Untergrenze: das Budget der Vektorsuche war erschöpft, es kann weitere passende Beiträge geben.",
    ties: (who) => `Mehrere Konten liegen gleichauf: ${who}.`, inferred: (p) => `Die Bildgeschichte von ${p} ist aus Daten abgeleitet, nicht aus den Operationen der Kette gelesen.`,
    deletedPost: (p, d) => `${p} wurde${d ? ` am ${d}` : ""} gelöscht: ein Beleg dafür, was gepostet wurde, kein sichtbarer Beitrag.`,
    hiddenPosts: (n) => `${n} durch die NSFW-Einstellung verborgene${n === 1 ? "r Beitrag wurde" : " Beiträge wurden"} ausgelassen.`,
    captions: "KI-Beschreibungen sagen, was ein Modell im Bild sieht, nicht, was der Autor geschrieben hat.", conflict: (note) => `Die Belege widersprechen sich: ${note}`, blogPost: "Blogbeitrag",
  },
  es: {
    by: "de", posted: "publicado el", postedBlog: "publicado el", tagged: "etiquetas:", votes: (n) => `${n} voto${n === 1 ? "" : "s"}`, pxs: "PXS",
    firstSeen: (d, p) => `la imagen apareció por primera vez el ${d} en ${p}`, edited: (n, d) => `editado ${n} ${n === 1 ? "vez" : "veces"}, la última el ${d}`, deleted: (d) => `eliminado${d ? ` el ${d}` : ""}`,
    again: (p, d) => `la misma imagen se volvió a publicar en ${p} (${d})`, caption: "descrito por la IA como", textIn: "texto en la imagen", and: "y",
    artists: (n) => `${n} artista${n === 1 ? "" : "s"}`, artworks: (n) => `${n} obra${n === 1 ? "" : "s"}`, posts: (n) => `${n} post${n === 1 ? "" : "s"}`, mostBy: "sobre todo de",
    between: (a, b) => `Publicadas entre el ${a} y el ${b}.`, on: (d) => `Publicadas el ${d}.`, mostVoted: (t, a, v) => `La más votada es ${t} de @${a} (${v} voto${v === 1 ? "" : "s"}).`,
    tags: "Etiquetas frecuentes", colours: "Colores principales", shown: (k) => `(etiquetas y colores leídos de las ${k} más recientes)`, ofNewest: (k) => `Entre las ${k} más recientes:`,
    lowerBound: "El recuento es un mínimo: se agotó el presupuesto de búsqueda vectorial, puede haber más posts que coincidan.",
    ties: (who) => `Varias cuentas empatan: ${who}.`, inferred: (p) => `El historial de la imagen de ${p} se deduce de las fechas, no se leyó en las operaciones de la cadena.`,
    deletedPost: (p, d) => `${p} fue eliminado${d ? ` el ${d}` : ""}: es prueba de lo que se publicó, no un post visible.`,
    hiddenPosts: (n) => `${n} post${n === 1 ? "" : "s"} oculto${n === 1 ? "" : "s"} por el ajuste NSFW ${n === 1 ? "quedó" : "quedaron"} fuera.`,
    captions: "Las descripciones de la IA dicen lo que un modelo ve en la imagen, no lo que escribió el autor.", conflict: (note) => `Las pruebas se contradicen: ${note}`, blogPost: "entrada de blog",
  },
  it: {
    by: "di", posted: "pubblicato il", postedBlog: "pubblicato il", tagged: "tag:", votes: (n) => `${n} vot${n === 1 ? "o" : "i"}`, pxs: "PXS",
    firstSeen: (d, p) => `l'immagine è apparsa per la prima volta il ${d} in ${p}`, edited: (n, d) => `modificato ${n} volt${n === 1 ? "a" : "e"}, l'ultima il ${d}`, deleted: (d) => `eliminato${d ? ` il ${d}` : ""}`,
    again: (p, d) => `la stessa immagine è stata ripubblicata in ${p} (${d})`, caption: "descritto dall'IA come", textIn: "testo nell'immagine", and: "e",
    artists: (n) => `${n} artist${n === 1 ? "a" : "i"}`, artworks: (n) => `${n} oper${n === 1 ? "a" : "e"}`, posts: (n) => `${n} post`, mostBy: "soprattutto di",
    between: (a, b) => `Pubblicate tra il ${a} e il ${b}.`, on: (d) => `Pubblicate il ${d}.`, mostVoted: (t, a, v) => `La più votata è ${t} di @${a} (${v} vot${v === 1 ? "o" : "i"}).`,
    tags: "Tag frequenti", colours: "Colori principali", shown: (k) => `(tag e colori letti dalle ${k} più recenti)`, ofNewest: (k) => `Tra le ${k} più recenti:`,
    lowerBound: "Il conteggio è un minimo: il budget della ricerca vettoriale è esaurito, altri post potrebbero corrispondere.",
    ties: (who) => `Diversi account sono a pari merito: ${who}.`, inferred: (p) => `La cronologia dell'immagine di ${p} è dedotta dalle date, non letta dalle operazioni della catena.`,
    deletedPost: (p, d) => `${p} è stato eliminato${d ? ` il ${d}` : ""}: è una traccia di ciò che è stato pubblicato, non un post visibile.`,
    hiddenPosts: (n) => `${n} post nascost${n === 1 ? "o" : "i"} dall'impostazione NSFW ${n === 1 ? "è stato lasciato" : "sono stati lasciati"} fuori.`,
    captions: "Le descrizioni dell'IA dicono ciò che un modello vede nell'immagine, non ciò che l'autore ha scritto.", conflict: (note) => `Le prove si contraddicono: ${note}`, blogPost: "articolo",
  },
};

// ---- facts ---------------------------------------------------------------------------------------------

/** One sentence about a post, from its card and its history. */
export function factSentence(card: ArtworkCard, l: AnswerLang, history?: HistoryFacts | null): string {
  const w = W[l];
  const parts: string[] = [];
  const blog = card.type === "post";
  parts.push(`${blog ? w.postedBlog : w.posted} ${day(card.created_at)}`);
  if (card.tags?.length) parts.push(`${w.tagged} ${card.tags.slice(0, 6).join(", ")}`);
  if (typeof card.votes === "number" && card.votes > 0) parts.push(`${w.votes(card.votes)}${card.payout ? `, ${r2(card.payout)} ${w.pxs}` : ""}`);
  else if (card.payout) parts.push(`${r2(card.payout)} ${w.pxs}`);
  if (card.first_seen_in && card.first_seen_in !== card.path && card.first_seen_at) parts.push(w.firstSeen(day(card.first_seen_at), card.first_seen_in));
  const edits = history?.edits ?? card.history?.filter((e) => e.kind === "edit").length ?? 0;
  const lastEdit = history?.last_edit ? new Date(history.last_edit * 1000).toISOString().slice(0, 10) : card.history?.filter((e) => e.kind === "edit").at(-1)?.at;
  if (edits > 0 && lastEdit) parts.push(w.edited(edits, day(lastEdit)));
  if (card.deleted) parts.push(w.deleted(day(card.deleted_at)));
  const later = (history?.appearances ?? []).filter((a) => `/@${a.author}/${a.permlink}` !== card.path && a.from > (history?.created ?? 0)).slice(0, 2);
  for (const a of later) parts.push(w.again(`/@${a.author}/${a.permlink}`, new Date(a.from * 1000).toISOString().slice(0, 10)));
  if (card.ai_caption) parts.push(`${w.caption} “${card.ai_caption.length > 160 ? `${card.ai_caption.slice(0, 157).trimEnd()}…` : card.ai_caption}”`);
  if (card.text_in_image) parts.push(`${w.textIn}: “${card.text_in_image.slice(0, 60)}”`);
  const title = card.title ? quote(card.title, l) : card.path;
  return `${title} ${w.by} @${card.author}${blog ? ` (${w.blogPost})` : ""}: ${parts.join("; ")}.`;
}

/** The posts the facts are about: the deciding evidence of the final step, then of the earlier steps. */
function decidingCards(input: DigestInput): ArtworkCard[] {
  const byId = new Map(input.cards.map((c) => [c.artwork_id, c]));
  const out: ArtworkCard[] = [];
  const add = (x: Verified | undefined) => {
    const c = x && byId.get(x.row.id);
    if (c && !out.includes(c)) out.push(c);
  };
  const setLike = SET_OPS.has(input.final.result.op);
  for (const x of input.final.result.evidence.slice(0, setLike ? 2 : 3)) add(x);
  for (const o of input.outcomes) if (o !== input.final) for (const x of o.result.evidence.slice(0, 1)) add(x);
  // a post the history names (the first sighting, a later post of the same image)
  for (const c of [...out]) {
    if (c.first_seen_in && c.first_seen_in !== c.path) {
      const other = input.cards.find((x) => x.path === c.first_seen_in);
      if (other && !out.includes(other)) out.push(other);
    }
  }
  return out.slice(0, MAX_FACTS);
}

// ---- overview ----------------------------------------------------------------------------------------------

/** Counts over the rows at hand. */
export function digestStats(rows: Array<Record<string, any>>, exactCount?: number, counts?: Array<{ author: string; n: number }>, extremes?: DigestInput["extremes"]): DigestStats {
  const seen = new Set<number>();
  const all = rows.filter((r) => (seen.has(r.id) ? false : (seen.add(r.id), true)));
  const authors = new Map<string, number>();
  const tags = new Map<string, number>();
  const colors = new Map<string, number>();
  let from: number | undefined;
  let to: number | undefined;
  let top: Record<string, any> | undefined;
  for (const r of all) {
    authors.set(r.author, (authors.get(r.author) ?? 0) + 1);
    for (const t of parseList(r.tags_json).slice(0, 8)) tags.set(t, (tags.get(t) ?? 0) + 1);
    const buckets = parseList(r.buckets_json) as unknown as Array<{ name: string; weight: number }>;
    const main = buckets.filter((b) => b && typeof b.weight === "number" && b.weight >= 0.2).sort((a, b) => b.weight - a.weight)[0];
    if (main?.name) colors.set(main.name, (colors.get(main.name) ?? 0) + 1);
    if (typeof r.created === "number") {
      from = from === undefined ? r.created : Math.min(from, r.created);
      to = to === undefined ? r.created : Math.max(to, r.created);
    }
    if ((r.net_votes ?? 0) > 0 && (!top || (r.net_votes ?? 0) > (top.net_votes ?? 0))) top = r;
  }
  // the set's own oldest and most voted, when they were read exactly
  if (extremes?.first && typeof extremes.first.created === "number") from = extremes.first.created;
  if (extremes?.top && (extremes.top.net_votes ?? 0) > 0) top = extremes.top;
  const posts = exactCount ?? all.length;
  const authorList = counts?.length ? counts.map((c) => ({ author: c.author, n: c.n })) : [...authors].map(([author, n]) => ({ author, n })).sort((a, b) => b.n - a.n || a.author.localeCompare(b.author));
  const fmt = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);
  return {
    posts,
    read: all.length,
    authors: authorList.slice(0, 10),
    ...(from !== undefined ? { from: fmt(from) } : {}),
    ...(to !== undefined ? { to: fmt(to) } : {}),
    ...(top ? { top_voted: { path: `/@${top.author}/${top.permlink}`, title: top.title ?? "", author: top.author, votes: top.net_votes ?? 0 } } : {}),
    tags: [...tags].map(([tag, n]) => ({ tag, n })).sort((a, b) => b.n - a.n || a.tag.localeCompare(b.tag)).slice(0, 8),
    colors: [...colors].map(([color, n]) => ({ color, n })).sort((a, b) => b.n - a.n || a.color.localeCompare(b.color)).slice(0, 4),
    exact: all.length >= posts || !!(extremes?.first && extremes?.top !== undefined),
  };
}

function overviewLines(s: DigestStats, l: AnswerLang, plan: QueryPlan, blog: boolean): string[] {
  const w = W[l];
  const out: string[] = [];
  if (s.posts === 0) return out;
  const what = subjectCount(s.posts, plan, l, blog) ?? (blog ? w.posts(s.posts) : w.artworks(s.posts));
  const top = s.authors.slice(0, MAX_LIST).map((a) => `@${a.author} (${a.n})`).join(", ");
  if (s.authors.length === 1) out.push(`${what} ${w.by} @${s.authors[0].author}.`);
  else if (s.authors.length > 1) out.push(`${what}, ${w.artists(s.authors.length)}${s.authors.length > s.read && s.read < s.posts ? "+" : ""}; ${w.mostBy} ${top}.`);
  else out.push(`${what}.`);
  // the range and the most voted: of the whole set when they are its (every row read, or read
  // exactly), else said of the newest rows read
  const partial = s.read < s.posts && !s.exact;
  const range = s.from && s.to ? (s.from === s.to ? w.on(s.from) : w.between(s.from, s.to)) : null;
  const voted = s.top_voted && s.read > 1 ? w.mostVoted(s.top_voted.title ? quote(s.top_voted.title, l) : s.top_voted.path, s.top_voted.author, s.top_voted.votes) : null;
  if (partial && (range || voted)) out.push(`${w.ofNewest(s.read)} ${[range, voted].filter((x): x is string => !!x).map((x) => x.charAt(0).toLowerCase() + x.slice(1, -1)).join("; ")}.`);
  else {
    if (range) out.push(range);
    if (voted) out.push(voted);
  }
  const tags = s.tags.filter((t) => t.n > 1 || s.read <= 3).slice(0, 5);
  if (tags.length > 1) out.push(`${w.tags}: ${tags.map((t) => `${t.tag} (${t.n})`).join(", ")}.`);
  if (s.colors.length > 1 && s.read >= 4) out.push(`${w.colours}: ${s.colors.slice(0, 3).map((c) => `${c.color} (${c.n})`).join(", ")}.`);
  if (s.read > 1 && s.read < s.posts && (tags.length > 1 || (s.colors.length > 1 && s.read >= 4))) out.push(w.shown(s.read));
  return out;
}

// ---- follow-ups and searches ----------------------------------------------------------------------------------

type Family = "author" | "subject" | "title";
type Tpl = (x: { a: string; s: string; sp: string; t: string }) => string;

/**
 * Questions per family, keyed by the intent they ask for (so the question's own intent is left
 * out). Every template is checked against the rules planner in test/v4-rich.test.ts.
 */
const FOLLOW: L<Record<Family, Array<[string, Tpl]>>> = {
  en: {
    author: [["count", (x) => `How many artworks did @${x.a} post?`], ["top", (x) => `What is the most voted artwork of @${x.a}?`], ["find_last", (x) => `What is the latest artwork of @${x.a}?`], ["summary", (x) => `What kind of art does @${x.a} make?`]],
    subject: [["count", (x) => `How many ${x.sp} are there?`], ["find_first", (x) => `Who posted the first ${x.s}?`], ["find_last", (x) => `Who posted the latest ${x.s}?`], ["compare", (x) => `Who posted the most ${x.sp}?`], ["top", (x) => `Which ${x.s} has the most votes?`]],
    title: [["date", (x) => `When was “${x.t}” posted?`], ["edit", (x) => `Was “${x.t}” edited?`], ["repost", (x) => `Was “${x.t}” reposted?`]],
  },
  // the subject is attached to a noun of fixed gender (œuvre, Kunstwerk, obra, opera): a typed
  // subject has no gender or number the templates could know
  fr: {
    author: [["count", (x) => `Combien d'œuvres @${x.a} a-t-il publiées ?`], ["top", (x) => `Quelle œuvre de @${x.a} a le plus de votes ?`], ["find_last", (x) => `Quelle est la dernière œuvre de @${x.a} ?`], ["summary", (x) => `Quel genre d'art fait @${x.a} ?`]],
    subject: [["count", (x) => `Combien d'œuvres de ${x.s} ?`], ["find_first", (x) => `Qui a posté la première œuvre de ${x.s} ?`], ["find_last", (x) => `Qui a posté la dernière œuvre de ${x.s} ?`], ["compare", (x) => `Qui a posté le plus d'œuvres de ${x.s} ?`], ["top", (x) => `Quelle œuvre de ${x.s} a le plus de votes ?`]],
    title: [["date", (x) => `Quand « ${x.t} » a-t-il été publié ?`], ["edit", (x) => `Est-ce que « ${x.t} » a été modifié ?`], ["repost", (x) => `Est-ce que « ${x.t} » a été republié ?`]],
  },
  de: {
    author: [["count", (x) => `Wie viele Kunstwerke hat @${x.a} gepostet?`], ["top", (x) => `Welches Kunstwerk von @${x.a} hat die meisten Stimmen?`], ["find_last", (x) => `Was ist das neueste Kunstwerk von @${x.a}?`], ["summary", (x) => `Welche Art von Kunst macht @${x.a}?`]],
    subject: [["count", (x) => `Wie viele Kunstwerke mit ${x.s} gibt es?`], ["find_first", (x) => `Wer hat das erste Kunstwerk mit ${x.s} gepostet?`], ["find_last", (x) => `Wer hat zuletzt ein Kunstwerk mit ${x.s} gepostet?`], ["compare", (x) => `Wer hat die meisten Kunstwerke mit ${x.s} gepostet?`], ["top", (x) => `Welches Kunstwerk mit ${x.s} hat die meisten Stimmen?`]],
    title: [["date", (x) => `Wann wurde „${x.t}“ gepostet?`], ["edit", (x) => `Wurde „${x.t}“ bearbeitet?`], ["repost", (x) => `Wurde „${x.t}“ erneut gepostet?`]],
  },
  es: {
    author: [["count", (x) => `¿Cuántas obras publicó @${x.a}?`], ["top", (x) => `¿Cuál es la obra más votada de @${x.a}?`], ["find_last", (x) => `¿Cuál es la última obra de @${x.a}?`], ["summary", (x) => `¿Qué tipo de arte hace @${x.a}?`]],
    subject: [["count", (x) => `¿Cuántas obras de ${x.s}?`], ["find_first", (x) => `¿Quién publicó la primera obra de ${x.s}?`], ["find_last", (x) => `¿Quién publicó la última obra de ${x.s}?`], ["compare", (x) => `¿Quién publicó más obras de ${x.s}?`], ["top", (x) => `¿Cuál es la obra de ${x.s} más votada?`]],
    title: [["date", (x) => `¿Cuándo se publicó «${x.t}»?`], ["edit", (x) => `¿Fue editada «${x.t}»?`], ["repost", (x) => `¿Fue republicada «${x.t}»?`]],
  },
  it: {
    author: [["count", (x) => `Quante opere ha pubblicato @${x.a}?`], ["top", (x) => `Quale opera di @${x.a} ha più voti?`], ["find_last", (x) => `Quando ha pubblicato @${x.a} l'ultima opera?`], ["summary", (x) => `Che tipo di arte fa @${x.a}?`]],
    subject: [["count", (x) => `Quante opere di ${x.s}?`], ["find_first", (x) => `Chi ha pubblicato la prima opera di ${x.s}?`], ["find_last", (x) => `Chi ha pubblicato l'ultima opera di ${x.s}?`], ["compare", (x) => `Chi ha pubblicato più opere di ${x.s}?`], ["top", (x) => `Quale opera di ${x.s} ha più voti?`]],
    title: [["date", (x) => `Quando è stato pubblicato «${x.t}»?`], ["edit", (x) => `È stato modificato «${x.t}»?`], ["repost", (x) => `È stato ripubblicato «${x.t}»?`]],
  },
};

/** The intents a question already asks for, so no follow-up repeats it. */
function askedIntents(input: DigestInput): Set<string> {
  const out = new Set<string>();
  const plan = input.final.plan ?? input.plan;
  out.add(plan.intent);
  for (const s of input.program.steps) {
    if (s.op === "history") out.add("repost").add("edit");
    if (s.op === "author_of" || s.op === "resolve_title" || s.op === "identify") out.add("date");
    if (s.op === "compare_counts") out.add("compare");
    if (s.op === "compare_metric") out.add("votes").add("top");
    if (s.plan?.intent) out.add(s.plan.intent);
  }
  if (/\b(votes?|voix|stimmen|votos?|voti)\b/i.test(input.question)) out.add("votes");
  if (plan.intent === "count" || plan.intent === "compare") out.add("count");
  if (/\b(what kind|what type|what sort|quel genre|quel type|welche art|que tipo|che tipo)\b/i.test(fold(input.question))) out.add("summary");
  return out;
}

/**
 * The subject of a question as the templates need it: as typed ("cat", "chat", „Katze“) and, in
 * English, as the plural phrase ("cat artworks"); the other languages attach it to a noun of fixed
 * gender, so they need no plural.
 */
function subjectForms(plan: QueryPlan, l: AnswerLang): { s: string; sp: string } | null {
  if (!plan.residual && !plan.filters.colors?.length && !plan.filters.tones?.length) return null;
  const s = subjectLabel(plan, l);
  return { s, sp: l === "en" ? subjectPlural(plan, l) : s };
}

/** The subject in the overview's count line: "4 cat artworks", « 4 œuvres « chat » », „4 Kunstwerke „Katze““. */
function subjectCount(n: number, plan: QueryPlan, l: AnswerLang, blog: boolean): string | null {
  const forms = subjectForms(plan, l);
  if (!forms) return null;
  if (l === "en") return `${n} ${forms.sp}`;
  const noun = blog ? W[l].posts(n) : W[l].artworks(n);
  return `${noun} ${quote(forms.s, l)}`;
}

/** Operators whose question is about one named post, not a subject. */
const TITLE_OPS = new Set(["resolve", "identify", "history", "sequence", "duration"]);

/** The plan that names the subject: the final step's, or the first step's that has one (a comparison's count steps). */
export function subjectPlanOf(input: Pick<DigestInput, "plan" | "final" | "outcomes" | "program">): QueryPlan {
  const own = input.final.plan;
  if (own?.residual && !/^(more|fewer|less|plus|moins|mehr|weniger|mas|menos|piu|meno)\b/.test(fold(own.residual))) return own;
  const first = input.outcomes.find((o) => o.plan?.residual);
  return first?.plan ?? own ?? input.plan;
}

export function templateFollowUps(input: DigestInput, l: AnswerLang, stats: DigestStats): string[] {
  const plan = subjectPlanOf(input);
  const asked = askedIntents(input);
  const out: string[] = [];
  const push = (fam: Family, x: { a: string; s: string; sp: string; t: string }) => {
    for (const [intent, tpl] of FOLLOW[l][fam]) if (!asked.has(intent)) out.push(tpl(x));
  };
  // a question about a named post ("who posted “Lake”?", "was “Swan” edited?") has no subject to ask about
  const aboutTitle = input.program.titles.length > 0 || TITLE_OPS.has(input.final.result.op) || input.program.steps.some((s) => s.op === "resolve_title" || s.op === "author_of" || s.op === "identify" || s.op === "history");
  const subject = aboutTitle ? null : subjectForms(plan, l);
  const author = plan.filters.authors?.[0] ?? (stats.authors.length === 1 ? stats.authors[0].author : undefined);
  // the post the answer names: its title, for questions about it
  const named = input.cards.find((c) => c.evidence_id === `E${input.final.result.evidence[0]?.row?.id}`) ?? input.cards[0];
  const title = named?.title && named.title.length <= 60 && !/["“”«»„]/.test(named.title) ? named.title : "";
  const x = { a: author ?? named?.author ?? "", s: subject?.s ?? "", sp: subject?.sp ?? "", t: title };
  if (subject) push("subject", x);
  if (title && !SET_OPS.has(input.final.result.op)) push("title", x);
  if (x.a) push("author", x);
  if (title && SET_OPS.has(input.final.result.op)) push("title", x);
  return [...new Set(out)];
}

/** Searches for the box: the subject, with the main colour, the accounts, the tags. */
export function templateSearches(input: DigestInput, stats: DigestStats): string[] {
  const plan = subjectPlanOf(input);
  const out: string[] = [];
  const subject = plan.residual.trim();
  if (subject) {
    out.push(subject);
    const colour = stats.colors[0]?.color;
    if (colour && !plan.filters.colors?.length && !fold(subject).includes(fold(colour))) out.push(`${colour} ${subject}`);
  }
  for (const t of stats.tags.slice(0, 3)) if (t.n > 1 && fold(t.tag) !== fold(subject) && fold(t.tag) !== fold(singular(subject))) out.push(t.tag);
  for (const a of stats.authors.slice(0, 2)) if (!plan.filters.authors?.includes(a.author)) out.push(`@${a.author}`);
  return [...new Set(out)].slice(0, 6);
}

// ---- the digest ------------------------------------------------------------------------------------------

export function buildDigest(input: DigestInput): Digest {
  const l = answerLang(input.lang);
  const w = W[l];
  const plan = subjectPlanOf(input);
  const result = input.final.result;
  const blog = plan.object === "blog";

  // the set the answer is about
  const rows: Array<Record<string, any>> = (() => {
    const scope = input.final.scope;
    if (scope?.verified.length) return scope.verified.map((x) => x.row);
    const xs = (result.items ?? result.evidence) as Verified[];
    if (xs.length) return xs.map((x) => x.row);
    return input.rows ?? [];
  })();

  // facts about the deciding posts; a set with no deciding post (an exact count) shows its most
  // voted and its latest
  const deciding = decidingCards(input);
  if (!deciding.length && rows.length > 1) {
    const top = [...rows].sort((a, b) => (b.net_votes ?? 0) - (a.net_votes ?? 0) || (b.created ?? 0) - (a.created ?? 0))[0];
    const latest = [...rows].sort((a, b) => (b.created ?? 0) - (a.created ?? 0))[0];
    for (const r of [top, latest]) if (r && !deciding.some((c) => c.artwork_id === r.id)) deciding.push(artworkCard(r, { history: input.histories.get(`/@${r.author}/${r.permlink}`) ?? null }));
  }
  const facts = deciding.map((c) => factSentence(c, l, input.histories.get(c.path)));
  const exact = result.answerType === "count" && typeof result.answer === "number" && result.op !== "compare" ? result.answer : undefined;
  const stats = digestStats(rows, exact, result.counts, input.extremes);
  const setLike = SET_OPS.has(result.op) || rows.length > 1;
  const overview = setLike && stats.read > 0 && (stats.posts > 1 || stats.read > 1) ? overviewLines(stats, l, plan, blog) : [];

  // caveats
  const caveats: string[] = [];
  if (result.truncated || input.outcomes.some((o) => o.result.truncated)) caveats.push(w.lowerBound);
  const tied = (result.details as Record<string, unknown> | undefined)?.tied;
  if (Array.isArray(tied) && tied.length > 1) caveats.push(w.ties(tied.slice(0, 6).map((t) => `@${t}`).join(", ")));
  for (const c of deciding) {
    if (c.history_exact === false) caveats.push(w.inferred(c.path));
    if (c.deleted) caveats.push(w.deletedPost(c.path, day(c.deleted_at)));
  }
  if (input.hidden) caveats.push(w.hiddenPosts(input.hidden));
  for (const c of input.conflicts.slice(0, 2)) caveats.push(w.conflict(c));
  if (deciding.some((c) => c.ai_caption)) caveats.push(w.captions);

  const follow_ups = templateFollowUps(input, l, stats);
  const searches = templateSearches(input, stats);
  return { facts, overview, caveats: [...new Set(caveats)], follow_ups, searches, stats, about: deciding.map((c) => c.evidence_id) };
}
