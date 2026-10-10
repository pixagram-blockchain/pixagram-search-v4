// A text overview of a search (v4.8): what a page of results says as a whole, from computed values
// only, in the reader's language — the "answer with text" of a search box that used to show a
// grid alone. Built from the page's items and, when the request asked for facets, from the facet
// counts (which cover every match, not just the page).
//
//   "“red dragon”: 37 artworks match (showing 20). Most by @nova (12) and @pix (8). The most voted
//    is “Fire” by @nova (120 votes, 2026-05-02). Posted between 2026-01-03 and 2026-10-01.
//    Frequent tags: dragon, fire, red. Main colours: red, black."
//
// No model call: fast enough for every search (/query adds it to every search answer; /search on
// overview=1).

import type { Lang } from "../lib/text";
import { answerLang, quote, type AnswerLang } from "./answer-text";
import type { SearchItem, SearchResponse } from "./service";

export interface SearchOverview {
  text: string;
  /** the sentences apart, for a UI that shows them as chips */
  lines: string[];
  stats: {
    shown: number;
    /** every match when known (facets or total_candidates), else the page */
    total: number | null;
    artworks: number;
    posts: number;
    authors: Array<{ author: string; n: number }>;
    tags: Array<{ tag: string; n: number }>;
    colors: Array<{ color: string; n: number }>;
    from: string | null;
    to: string | null;
    top_voted: { path: string; title: string; author: string; votes: number; date: string } | null;
    newest: { path: string; title: string; author: string; date: string } | null;
  };
}

type L<T> = Record<AnswerLang, T>;

const W: L<{
  match: (q: string, n: number, shown: number, blog: number) => string;
  /** the total is known but not by kind: results, not artworks */
  results: (q: string, n: number, shown: number) => string;
  browse: (n: number, shown: number) => string;
  none: (q: string) => string;
  mostBy: (list: string) => string;
  oneBy: (a: string) => string;
  /** sentence fragments, so they can be said of the whole set ("The most voted is …") or of the page ("Of the 20 shown: the most voted is …") */
  mostVoted: (t: string, a: string, v: number, d: string) => string;
  newest: (t: string, a: string, d: string) => string;
  between: (a: string, b: string) => string;
  ofShown: (k: number) => string;
  tags: (list: string) => string;
  colours: (list: string) => string;
  and: string;
}> = {
  en: {
    match: (q, n, shown, blog) => `${q}: ${n} ${n === 1 ? "artwork matches" : "artworks match"}${blog ? ` and ${blog} blog post${blog === 1 ? "" : "s"}` : ""}${shown < n ? ` (showing ${shown})` : ""}.`,
    results: (q, n, shown) => `${q}: ${n} results (showing ${shown}).`,
    browse: (n, shown) => `${n} posts${shown < n ? `, showing ${shown}` : ""}.`,
    none: (q) => `${q}: nothing matches.`,
    mostBy: (list) => `Most by ${list}.`, oneBy: (a) => `All by @${a}.`,
    mostVoted: (t, a, v, d) => `the most voted is ${t} by @${a} (${v} vote${v === 1 ? "" : "s"}, ${d})`,
    newest: (t, a, d) => `the newest is ${t} by @${a} (${d})`,
    between: (a, b) => (a === b ? `posted on ${a}` : `posted between ${a} and ${b}`),
    ofShown: (k) => `Of the ${k} shown:`,
    tags: (list) => `Frequent tags: ${list}.`, colours: (list) => `Main colours: ${list}.`, and: "and",
  },
  fr: {
    match: (q, n, shown, blog) => `${q} : ${n} œuvre${n === 1 ? "" : "s"} correspond${n === 1 ? "" : "ent"}${blog ? ` et ${blog} article${blog === 1 ? "" : "s"}` : ""}${shown < n ? ` (${shown} affichée${shown === 1 ? "" : "s"})` : ""}.`,
    results: (q, n, shown) => `${q} : ${n} résultats (${shown} affichés).`,
    browse: (n, shown) => `${n} posts${shown < n ? `, ${shown} affichés` : ""}.`,
    none: (q) => `${q} : aucun résultat.`,
    mostBy: (list) => `Surtout de ${list}.`, oneBy: (a) => `Toutes de @${a}.`,
    mostVoted: (t, a, v, d) => `la plus votée est ${t} de @${a} (${v} vote${v === 1 ? "" : "s"}, ${d})`,
    newest: (t, a, d) => `la plus récente est ${t} de @${a} (${d})`,
    between: (a, b) => (a === b ? `publiées le ${a}` : `publiées entre le ${a} et le ${b}`),
    ofShown: (k) => `Parmi les ${k} affichées :`,
    tags: (list) => `Tags fréquents : ${list}.`, colours: (list) => `Couleurs principales : ${list}.`, and: "et",
  },
  de: {
    match: (q, n, shown, blog) => `${q}: ${n} Kunstwerk${n === 1 ? " passt" : "e passen"}${blog ? ` und ${blog} Blogbeitr${blog === 1 ? "ag" : "äge"}` : ""}${shown < n ? ` (${shown} angezeigt)` : ""}.`,
    results: (q, n, shown) => `${q}: ${n} Treffer (${shown} angezeigt).`,
    browse: (n, shown) => `${n} Beiträge${shown < n ? `, ${shown} angezeigt` : ""}.`,
    none: (q) => `${q}: nichts gefunden.`,
    mostBy: (list) => `Die meisten von ${list}.`, oneBy: (a) => `Alle von @${a}.`,
    mostVoted: (t, a, v, d) => `die meisten Stimmen hat ${t} von @${a} (${v} Stimme${v === 1 ? "" : "n"}, ${d})`,
    newest: (t, a, d) => `das neueste ist ${t} von @${a} (${d})`,
    between: (a, b) => (a === b ? `gepostet am ${a}` : `gepostet zwischen dem ${a} und dem ${b}`),
    ofShown: (k) => `Unter den ${k} angezeigten:`,
    tags: (list) => `Häufige Tags: ${list}.`, colours: (list) => `Hauptfarben: ${list}.`, and: "und",
  },
  es: {
    match: (q, n, shown, blog) => `${q}: ${n} obra${n === 1 ? " coincide" : "s coinciden"}${blog ? ` y ${blog} entrada${blog === 1 ? "" : "s"} de blog` : ""}${shown < n ? ` (se muestran ${shown})` : ""}.`,
    results: (q, n, shown) => `${q}: ${n} resultados (se muestran ${shown}).`,
    browse: (n, shown) => `${n} posts${shown < n ? `, se muestran ${shown}` : ""}.`,
    none: (q) => `${q}: sin resultados.`,
    mostBy: (list) => `Sobre todo de ${list}.`, oneBy: (a) => `Todas de @${a}.`,
    mostVoted: (t, a, v, d) => `la más votada es ${t} de @${a} (${v} voto${v === 1 ? "" : "s"}, ${d})`,
    newest: (t, a, d) => `la más reciente es ${t} de @${a} (${d})`,
    between: (a, b) => (a === b ? `publicadas el ${a}` : `publicadas entre el ${a} y el ${b}`),
    ofShown: (k) => `Entre las ${k} mostradas:`,
    tags: (list) => `Etiquetas frecuentes: ${list}.`, colours: (list) => `Colores principales: ${list}.`, and: "y",
  },
  it: {
    match: (q, n, shown, blog) => `${q}: ${n} oper${n === 1 ? "a corrisponde" : "e corrispondono"}${blog ? ` e ${blog} articol${blog === 1 ? "o" : "i"}` : ""}${shown < n ? ` (${shown} mostrat${shown === 1 ? "a" : "e"})` : ""}.`,
    results: (q, n, shown) => `${q}: ${n} risultati (${shown} mostrati).`,
    browse: (n, shown) => `${n} post${shown < n ? `, ${shown} mostrati` : ""}.`,
    none: (q) => `${q}: nessun risultato.`,
    mostBy: (list) => `Soprattutto di ${list}.`, oneBy: (a) => `Tutte di @${a}.`,
    mostVoted: (t, a, v, d) => `la più votata è ${t} di @${a} (${v} vot${v === 1 ? "o" : "i"}, ${d})`,
    newest: (t, a, d) => `la più recente è ${t} di @${a} (${d})`,
    between: (a, b) => (a === b ? `pubblicate il ${a}` : `pubblicate tra il ${a} e il ${b}`),
    ofShown: (k) => `Tra le ${k} mostrate:`,
    tags: (list) => `Tag frequenti: ${list}.`, colours: (list) => `Colori principali: ${list}.`, and: "e",
  },
};

const cap = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

const day = (t: number) => new Date(t * 1000).toISOString().slice(0, 10);

function joinList(xs: string[], l: AnswerLang): string {
  if (xs.length <= 1) return xs.join("");
  return `${xs.slice(0, -1).join(", ")} ${W[l].and} ${xs[xs.length - 1]}`;
}

/** The overview of a search response (its first page) in the given language. */
export function searchOverview(res: Pick<SearchResponse, "query" | "items" | "total_candidates" | "facets" | "mode">, lang: Lang | AnswerLang | string): SearchOverview {
  const l = answerLang(lang);
  const w = W[l];
  const items: SearchItem[] = res.items ?? [];
  const facets = res.facets ?? {};
  const authorsF = facets.author?.length ? facets.author : null;
  const tagsF = facets.tag?.length ? facets.tag : null;
  const colorsF = facets.has_color?.length ? facets.has_color : facets.primary_color?.length ? facets.primary_color : null;
  const typeF = facets.type?.length ? facets.type : null;

  const count = (key: (i: SearchItem) => string | null | undefined) => {
    const m = new Map<string, number>();
    for (const i of items) {
      const k = key(i);
      if (k) m.set(k, (m.get(k) ?? 0) + 1);
    }
    return [...m].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  };
  const authors = (authorsF ? authorsF.map((x) => [x.key, x.n] as [string, number]) : count((i) => i.author)).slice(0, 10);
  const tagCounts = new Map<string, number>();
  for (const i of items) for (const t of i.tags.slice(0, 8)) tagCounts.set(t, (tagCounts.get(t) ?? 0) + 1);
  const tags = (tagsF ? tagsF.map((x) => [x.key, x.n] as [string, number]) : [...tagCounts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))).slice(0, 8);
  const colors = (colorsF ? colorsF.map((x) => [x.key, x.n] as [string, number]) : count((i) => i.artwork?.primary_color ?? null)).slice(0, 5);
  const artworksShown = items.filter((i) => i.type === "artwork").length;
  const postsShown = items.length - artworksShown;
  const totalArt = typeF ? (typeF.find((x) => x.key === "artwork")?.n ?? 0) : null;
  const totalBlog = typeF ? (typeF.find((x) => x.key === "blog")?.n ?? 0) : null;
  const total = typeF ? (totalArt ?? 0) + (totalBlog ?? 0) : authorsF ? authorsF.reduce((s, x) => s + x.n, 0) : typeof res.total_candidates === "number" && res.total_candidates >= items.length ? res.total_candidates : null;
  let from: number | null = null;
  let to: number | null = null;
  let top: SearchItem | null = null;
  let newest: SearchItem | null = null;
  for (const i of items) {
    from = from === null ? i.created : Math.min(from, i.created);
    to = to === null ? i.created : Math.max(to, i.created);
    if (i.net_votes > 0 && (!top || i.net_votes > top.net_votes)) top = i;
    if (!newest || i.created > newest.created) newest = i;
  }
  const stats: SearchOverview["stats"] = {
    shown: items.length,
    total,
    artworks: totalArt ?? artworksShown,
    posts: totalBlog ?? postsShown,
    authors: authors.map(([author, n]) => ({ author, n })),
    tags: tags.map(([tag, n]) => ({ tag, n })),
    colors: colors.map(([color, n]) => ({ color, n })),
    from: from === null ? null : day(from),
    to: to === null ? null : day(to),
    top_voted: top ? { path: top.path, title: top.title, author: top.author, votes: top.net_votes, date: day(top.created) } : null,
    newest: newest ? { path: newest.path, title: newest.title, author: newest.author, date: day(newest.created) } : null,
  };

  const lines: string[] = [];
  const q = res.query?.trim() ? quote(res.query.trim(), l) : "";
  if (!items.length) {
    lines.push(q ? w.none(q) : w.browse(0, 0));
    return { text: lines.join(" "), lines, stats };
  }
  // the headline: by kind when the facets count every match, as "results" when only a total is
  // known, from the page otherwise
  if (q && typeF) lines.push(w.match(q, stats.artworks, artworksShown, stats.posts));
  else if (q && total !== null && total > items.length) lines.push(w.results(q, total, items.length));
  else if (q) lines.push(w.match(q, artworksShown, artworksShown, postsShown));
  else lines.push(w.browse(total ?? items.length, items.length));
  if (authors.length === 1 && (total === null || authors[0][1] === total)) lines.push(w.oneBy(authors[0][0]));
  else if (authors.length > 1) lines.push(w.mostBy(joinList(authors.slice(0, 3).map(([a, n]) => `@${a} (${n})`), l)));
  // the page's superlatives and range: said of the whole set only when the page is the set
  const partial = total !== null && total > items.length;
  const facts: string[] = [];
  if (top && items.length > 1) facts.push(w.mostVoted(top.title ? quote(top.title, l) : top.path, top.author, top.net_votes, day(top.created)));
  if (newest && items.length > 1 && newest !== top) facts.push(w.newest(newest.title ? quote(newest.title, l) : newest.path, newest.author, day(newest.created)));
  if (from !== null && to !== null && items.length > 1) facts.push(w.between(day(from), day(to)));
  if (facts.length) {
    if (partial) lines.push(`${w.ofShown(items.length)} ${facts.join("; ")}.`);
    else for (const f of facts) lines.push(`${cap(f)}.`);
  }
  // tags and colours: the facets' when they count every match, else the page's
  const tagList = items.length > 1 ? tags.filter(([, n]) => n > 1 || items.length <= 3).slice(0, 5) : [];
  if (tagList.length > 1 && (tagsF || !partial)) lines.push(w.tags(tagList.map(([t]) => t).join(", ")));
  const colourList = colors.filter(([, n]) => n > 1).slice(0, 3);
  if (colourList.length > 1 && items.length >= 4 && (colorsF || !partial)) lines.push(w.colours(colourList.map(([c]) => c).join(", ")));
  return { text: lines.join(" "), lines, stats };
}
