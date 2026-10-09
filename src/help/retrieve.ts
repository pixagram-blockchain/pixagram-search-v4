// Retrieval over the documentation: full text (FTS5, bm25) and vectors (VEC_DOCS). Each candidate
// is scored by how much of the question it covers (lexical, language-light stemming) and how close
// its vector is (cosine); the better of the two decides whether it counts as relevant.

import type { Env } from "../env";
import { num } from "../env";
import { fold, singular, tokens } from "../lib/text";
import { blobUrl, repoRef } from "../docs/github";
import { docsEmbedModel, embedQuestion } from "../docs/vectors";

export interface DocHit {
  id: number;
  path: string;
  title: string;
  heading: string;
  anchor: string;
  url: string;
  text: string;
  lang: string | null;
  /** weighted share of the question's words found in the chunk */
  lexical: number;
  cosine: number | null;
  /** relevance in [0, 1]; ≥ 0.5 counts as relevant */
  score: number;
}

export interface DocsRetrieval {
  hits: DocHit[];
  relevant: boolean;
  best: number;
  /** chunks in the index (0 = no documentation yet) */
  chunks: number;
  terms: string[];
  /** the vectors could not be used, or some chunks have none yet: a "not found" may be temporary */
  degraded: boolean;
  notes: string[];
}

export const RELEVANT = 0.5;

// Function words of questions (not the artwork stopwords of lib/text, which drop "upload", "post"
// or "artwork": words that matter in a help question).
const DOC_STOP = new Set(
  [
    "a an the and or but of to in on at by for with from into onto over under about as is are was were be been being am do does did doing done have has had having can could should would will shall may might must i me my mine we us our you your he him his she her it its they them their this that these those there here what which who whom whose when where why how please tell explain show give know want need get some any all no not yes if then than so very just also more most much many",
    "le la les l un une des du de d au aux et ou mais en dans sur sous par pour avec sans chez est sont etait etre ete ai as avons avez ont avait faire fait peut peux peuvent puis dois doit faut il elle ils elles on je j me m moi mon ma mes tu te t toi ton ta tes nous vous notre nos votre vos leur leurs son sa ses ce c cet cette ces cela ca y qui que qu quoi quel quelle quels quelles quand comment pourquoi combien est-ce s se si ne n pas plus tres aussi",
    "der die das den dem des ein eine einen einem einer und oder aber zu im am an auf aus bei mit von vom zum zur fur uber unter ist sind war waren sein bin bist hat haben hatte kann kannst konnen muss mussen soll sollte wird werden ich mich mir mein meine du dich dir dein er sie es wir uns unser ihr euch man wer was welche welcher welches wann wo wie warum wieso gibt nicht kein keine auch sehr",
    "el los las lo unos unas del al y pero con por para sin sobre es son era fue ser estar esta hay he ha han puedo puede pueden debo debe yo mi mis tu tus ella nosotros nos su sus quien quienes cual cuales cuando donde como porque cuanto cuantos si muy tambien",
    "il lo gli i della dello dei degli delle alla ai agli alle da dal dalla nel nella sul sulla per tra fra che chi quale quali dove perche quanto quanti quante non io mio mia miei mie ti tuo tua noi ci vostro loro sono essere ho hanno posso puo possono devo deve ce molto anche",
  ]
    .join(" ")
    .split(/\s+/),
);

/** Platform names carry little information in a help question about the platform. */
const LOW_WEIGHT = new Set(["pixagram", "pixa"]);

/** The words of a question that a documentation chunk should contain. */
export function questionTerms(q: string): string[] {
  return [...new Set(tokens(fold(q), { keepHyphenated: false }).filter((t) => t.length > 1 && !DOC_STOP.has(t)))].slice(0, 12);
}

function ftsExpression(terms: string[]): string | null {
  const parts = new Set<string>();
  for (const t of terms) {
    const clean = t.replace(/["*]/g, "");
    if (!clean) continue;
    parts.add(`"${clean}"`);
    const s = singular(clean);
    if (s !== clean) parts.add(`"${s}"`);
    if (clean.length >= 6) parts.add(`"${clean.slice(0, 6)}"*`);
  }
  return parts.size ? [...parts].join(" OR ") : null;
}

/** Weighted share of the terms found in the text (exact, singular, or a shared 6-letter stem). */
export function coverage(terms: string[], text: string): number {
  if (!terms.length) return 0;
  const words = tokens(text, { keepHyphenated: false });
  const set = new Set<string>();
  const stems = new Set<string>();
  for (const w of words) {
    set.add(w);
    set.add(singular(w));
    if (w.length >= 6) stems.add(w.slice(0, 6));
  }
  let got = 0;
  let total = 0;
  for (const t of terms) {
    const w = LOW_WEIGHT.has(t) ? 0.5 : 1;
    total += w;
    if (set.has(t) || set.has(singular(t)) || (t.length >= 6 && stems.has(t.slice(0, 6)))) got += w;
  }
  return total ? got / total : 0;
}

type ChunkRow = { id: number; path: string; title: string; heading: string; anchor: string; text: string; lang: string | null };

const COLS = "c.id, c.path, c.title, c.heading, c.anchor, c.text, c.lang";

export async function docsChunkCount(env: Env): Promise<number> {
  const r = await env.DB.prepare("SELECT COUNT(*) AS n FROM doc_chunks").first<{ n: number }>().catch(() => null);
  return r?.n ?? 0;
}

function toHit(env: Env, r: ChunkRow, terms: string[], cos: number | null): DocHit {
  const ref = repoRef(env);
  const minScore = num(env.DOCS_MIN_SCORE, 0.5);
  const lexBody = coverage(terms, `${r.title}\n${r.heading}\n${r.text}`);
  const inHeading = coverage(terms, `${r.title}\n${r.heading}`) > 0;
  const lexical = Math.min(1, lexBody * 0.9 + (inHeading && lexBody > 0 ? 0.1 : 0));
  const cosNorm = cos === null ? 0 : Math.max(0, Math.min(1, (cos - (minScore - 0.15)) / 0.3));
  const score = Math.min(1, Math.max(lexical, cosNorm) + 0.1 * Math.min(lexical, cosNorm));
  return {
    id: r.id,
    path: r.path,
    title: r.title,
    heading: r.heading,
    anchor: r.anchor,
    url: ref ? blobUrl(ref, r.path, r.anchor) : r.path,
    text: r.text,
    lang: r.lang,
    lexical: Math.round(lexical * 1000) / 1000,
    cosine: cos === null ? null : Math.round(cos * 1000) / 1000,
    score: Math.round(score * 1000) / 1000,
  };
}

async function ftsRows(env: Env, terms: string[], limit: number): Promise<ChunkRow[]> {
  const match = ftsExpression(terms);
  if (!match) return [];
  return (
    (
      await env.DB.prepare(
        `SELECT ${COLS} FROM doc_chunks_fts JOIN doc_chunks c ON c.id = doc_chunks_fts.rowid
         WHERE doc_chunks_fts MATCH ? ORDER BY bm25(doc_chunks_fts, 3.0, 4.0, 1.0) LIMIT ?`,
      )
        .bind(match, limit)
        .all<ChunkRow>()
        .catch(() => ({ results: [] as ChunkRow[] }))
    ).results ?? []
  );
}

const byScore = (rank: Map<number, number>) => (a: DocHit, b: DocHit) => b.score - a.score || (rank.get(a.id) ?? 99) - (rank.get(b.id) ?? 99) || a.id - b.id;

/** Full-text candidates only (no AI call): what the router uses to see whether the docs know a question. */
export async function lexicalDocs(env: Env, question: string, limit = 20): Promise<{ hits: DocHit[]; terms: string[] }> {
  const terms = questionTerms(question);
  const rows = await ftsRows(env, terms, limit);
  const rank = new Map(rows.map((r, i) => [r.id, i]));
  return { hits: rows.map((r) => toHit(env, r, terms, null)).sort(byScore(rank)), terms };
}

export async function retrieveDocs(env: Env, question: string, opts: { k?: number; vectors?: boolean } = {}): Promise<DocsRetrieval> {
  const k = opts.k ?? 5;
  const notes: string[] = [];
  const terms = questionTerms(question);
  const chunks = await docsChunkCount(env);
  if (!chunks) return { hits: [], relevant: false, best: 0, chunks, terms, degraded: false, notes };
  let degraded = false;
  const rows = new Map<number, ChunkRow>();
  const fts = await ftsRows(env, terms, 20);
  for (const r of fts) rows.set(r.id, r);
  const rank = new Map(fts.map((r, i) => [r.id, i]));
  const cos = new Map<number, number>();
  if (opts.vectors !== false && env.VEC_DOCS && env.AI) {
    try {
      // chunks still waiting for their vector (just synced, or an embedding outage)
      degraded = !!(await env.DB.prepare("SELECT 1 AS x FROM doc_chunks WHERE embedded IS NOT ? LIMIT 1").bind(docsEmbedModel(env)).first());
      const qv = await embedQuestion(env, question);
      const res = await env.VEC_DOCS.query(qv, { topK: 20 });
      for (const m of res.matches ?? []) if (/^\d+$/.test(m.id)) cos.set(Number(m.id), m.score);
      const missing = [...cos.keys()].filter((id) => !rows.has(id));
      if (missing.length) {
        const more = (await env.DB.prepare(`SELECT ${COLS} FROM doc_chunks c WHERE c.id IN (${missing.map((x) => Math.trunc(x)).join(",")})`).all<ChunkRow>()).results ?? [];
        // a vector whose chunk is gone (its delete still in flight) finds no row and drops out here
        for (const r of more) rows.set(r.id, r);
      }
    } catch (e) {
      degraded = true;
      notes.push(`documentation vectors unavailable, full text only: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  const hits = [...rows.values()].map((r) => toHit(env, r, terms, cos.get(r.id) ?? null)).sort(byScore(rank));
  const best = hits[0]?.score ?? 0;
  return { hits: hits.slice(0, k), relevant: best >= RELEVANT, best, chunks, terms, degraded, notes };
}
