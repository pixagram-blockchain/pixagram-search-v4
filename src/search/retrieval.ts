// Retrieval for questions (spec §9, §11): the candidate legs of v3's /ask, each candidate keeping
// where it came from, then v3's verification of every candidate against the subject.
//
//   subject ─► concept leg      artworks carrying the subject's concepts            (concept)
//          ─► full-text leg     every alias of the concepts, the subject's words     (fts)
//          ─► image kNN         time-sliced over the whole period, adaptive          (visual)
//          ─► text kNN          title / caption / tags vectors                       (semantic)
//          ─► history leg       deleted posts whose recorded titles match (deep modes only) (history)
//          ─► candidate union by post id (the stable candidate id), provenance per candidate
//          ─► hydrate under the filters ─► verify each against the subject
//
// The legs and their depths are v3's ("normal" retrieval); the deeper modes widen the pools
// (SEARCH_*_K) and add the history leg. Verification is v3's: noisy-OR of lexical/concept
// evidence and z-scored image and text similarity, against a threshold relative to the best match.

import type { Env } from "../env";
import { int } from "../env";
import { aliasesOf } from "../concepts";
import { embedQueriesCached, embeddingEnabled, normalize } from "../enrich/embed";
import { fold, tokens } from "../lib/text";
import { loadBackground, normFor, zScore, type Norm } from "./background";
import type { SearchContext } from "./context";
import { emptyRequest, type NsfwMode, type SearchRequest } from "./params";
import type { QueryPlan } from "./planner";
import { mentionsConcept } from "./ranker";
import { browse, candidateConcepts, conceptCandidates, ftsAnyOf, ftsCandidates, ftsQuery } from "./sql";
import { hydrateRows, runClause } from "./service";
import { adaptiveKnn, knn } from "./vectors";
import type { Provenance, Row, Scope, Verification, Verified } from "./operators/types";

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const Z0 = 3.5; // z of the subject prompt above which an image counts as showing it (measured, see background.ts)
const DELTA = 3;
const SLOPE = 2;

/** Vectorize queries one question may spend on its time slices (each returns at most 100 matches). */
export const ASK_VECTOR_BUDGET = 48;

/** Prompt ensemble for the subject: the mean of a few phrasings is steadier than any one of them. */
export const SUBJECT_TEMPLATES = ["{s}", "a picture of {s}", "an image of {s}", "a drawing of {s}"];

export async function subjectVector(env: Env, subject: string): Promise<number[]> {
  const vs = await embedQueriesCached(env, SUBJECT_TEMPLATES.map((t) => t.replace("{s}", subject)));
  const sum = new Array<number>(vs[0].length).fill(0);
  for (const v of vs) v.forEach((x, i) => (sum[i] += x));
  return normalize(sum);
}

export function requestFromPlan(plan: QueryPlan, a: { type?: "artwork" | "blog"; nsfw?: NsfwMode }): SearchRequest {
  return emptyRequest({
    q: plan.residual,
    type: a.type ?? (plan.object === "any" ? null : plan.object),
    authors: plan.filters.authors ?? [],
    tags: plan.filters.tags ?? [],
    hasColors: plan.filters.colors ?? [],
    minColorWeight: 0.15,
    background: plan.filters.background ?? [],
    orientation: plan.filters.orientation ?? [],
    from: plan.filters.from ?? null,
    to: plan.filters.to ?? null,
    nsfw: a.nsfw ?? "exclude",
    sort: "relevance",
  });
}

const FIELD_WEIGHTS: Array<[string, number]> = [
  ["tags", 0.9],
  ["title", 0.85],
  ["ai", 0.7],
  ["description", 0.6],
];

export function parseList(s: string | null | undefined): string[] {
  try {
    const v = JSON.parse(s ?? "[]");
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

/** Lexical/concept evidence that a row shows the plan's subject, in [0, 1], with reasons. */
export function lexicalEvidence(row: Row, plan: QueryPlan, concepts: Map<string, number>): { score: number; signals: string[] } {
  const fields: Record<string, string> = {
    tags: parseList(row.tags_json).join(" "),
    title: row.title ?? "",
    ai: `${row.ai_caption ?? ""} ${parseList(row.ai_tags_json).join(" ")}`,
    description: row.description ?? "",
  };
  const signals: string[] = [];
  if (plan.concepts.length) {
    let worst = 1;
    for (const k of plan.concepts) {
      let best = concepts.get(k) ?? 0;
      let why = best ? `concept ${k} (${best.toFixed(2)})` : "";
      for (const [f, w] of FIELD_WEIGHTS) {
        if (w > best && mentionsConcept(fields[f], k)) {
          best = w;
          why = `${f} mentions ${k}`;
        }
      }
      if (why) signals.push(why);
      worst = Math.min(worst, best);
    }
    return { score: worst, signals };
  }
  // No known concept: the subject's words found in the fields. A subject of several words is a
  // phrase, often a title ("good vibes"): matching some of its words is weak evidence, so the score
  // is scaled by the share matched once more (half the words → a quarter of the credit).
  const want = [...new Set(tokens(plan.residual, { keepHyphenated: false }).filter((t) => t.length > 1))];
  if (!want.length) return { score: 0, signals };
  let s = 0;
  let matched = 0;
  for (const t of want) {
    let best = 0;
    for (const [f, w] of FIELD_WEIGHTS) if (tokens(fields[f]).includes(t)) best = Math.max(best, w);
    s += best;
    if (best) matched++;
  }
  if (s) signals.push(matched === want.length ? `text matches "${plan.residual}"` : `text matches ${matched} of ${want.length} words of "${plan.residual}"`);
  return { score: (s / want.length) * (matched / want.length), signals };
}

/**
 * z above which an image counts as showing the subject. Absolute floor Z0, raised when the best
 * match stands far out: "sushi" puts the sushi plate at z 7.7 and the kimchi bowl at 3.8 — the
 * kimchi is food, not sushi. Within DELTA of the best match, a candidate is a peer.
 */
export function zThreshold(zMax: number | null): number {
  return Math.max(Z0, (zMax ?? -Infinity) - DELTA);
}

export function verify(
  row: Row,
  plan: QueryPlan,
  concepts: Map<string, number>,
  cosImage: number | null,
  cosText: number | null,
  normI: Norm | null,
  normT: Norm | null,
  zMax: { image: number | null; text: number | null } = { image: null, text: null },
): Verification {
  const lex = lexicalEvidence(row, plan, concepts);
  const semantic = cosImage !== null && normI ? sigmoid(SLOPE * (zScore(cosImage, normI) - zThreshold(zMax.image))) : null;
  const text = cosText !== null && normT ? 0.6 * sigmoid(SLOPE * (zScore(cosText, normT) - zThreshold(zMax.text))) : null;
  const signals = [...lex.signals];
  if (semantic !== null && semantic >= 0.2) signals.push(`image matches "${plan.residual}" (z ${zScore(cosImage!, normI!).toFixed(1)})`);
  if (text !== null && text >= 0.15) signals.push(`caption/title vector matches (z ${zScore(cosText!, normT!).toFixed(1)})`);
  const score = 1 - (1 - lex.score) * (1 - 0.9 * (semantic ?? 0)) * (1 - (text ?? 0));
  return { score: round(score), lexical: round(lex.score), semantic: semantic === null ? null : round(semantic), text: text === null ? null : round(text), signals };
}

const round = (x: number) => Math.round(x * 1000) / 1000;

/**
 * Whether a subject is exactly a post's title, punctuation and case aside. Only subjects with
 * letters or digits count: "🐱" and "❤️" both reduce to nothing.
 */
export function isTitleSubject(subject: string, title: string): boolean {
  const key = (s: string) => fold(s).replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const k = key(subject);
  return k !== "" && k === key(title);
}

/** When the image of this row first appeared: first sighting of the bytes, else this post's own. */
export function timeKey(row: Row): number {
  return row.first_seen ?? row.image_since ?? row.created;
}

// ---- candidate generation -----------------------------------------------------------------------

/** How deep each leg looks (spec §9: pools adapt to the question). */
export interface RetrievalDepth {
  concept: number;
  fts: number;
  /** Vectorize queries for the time-sliced image kNN */
  visualBudget: number;
  semantic: number;
  /** deleted posts whose recorded titles match (0 = off) */
  history: number;
}

/** v3's depths: the "normal" retrieval of the fast and balanced modes. */
export const NORMAL_DEPTH: RetrievalDepth = { concept: 2000, fts: 2000, visualBudget: ASK_VECTOR_BUDGET, semantic: 100, history: 0 };

/** Deeper pools for the multi-stage retrieval of deep and expert, from SEARCH_*_K where set. */
export function depthFor(env: Env, level: "normal" | "expanded" | "multi-stage"): RetrievalDepth {
  if (level === "normal") return NORMAL_DEPTH;
  const k = (v: string | undefined, d: number) => Math.max(1, int(v, d));
  return {
    concept: Math.max(NORMAL_DEPTH.concept, k(env.SEARCH_CONCEPT_K, 2000)),
    fts: Math.max(NORMAL_DEPTH.fts, k(env.SEARCH_FTS_K, 2000)),
    visualBudget: level === "multi-stage" ? Math.max(ASK_VECTOR_BUDGET, Math.min(96, k(env.SEARCH_VISUAL_K, 64))) : ASK_VECTOR_BUDGET,
    semantic: Math.min(100, k(env.SEARCH_SEMANTIC_K, 100)),
    history: level === "multi-stage" ? Math.min(200, k(env.SEARCH_HISTORY_K, 50)) : 0,
  };
}

/** Candidates merged by post id, each with the legs that found it and their scores (spec §11). */
export class CandidatePool {
  readonly prov = new Map<number, Provenance>();
  readonly legs: Record<string, number> = {};
  add(leg: string, hits: Array<{ id: number; score?: number }>): void {
    this.legs[leg] = (this.legs[leg] ?? 0) + hits.length;
    for (const h of hits) {
      let p = this.prov.get(h.id);
      if (!p) this.prov.set(h.id, (p = { sources: [], scores: {} }));
      if (!p.sources.includes(leg)) p.sources.push(leg);
      if (typeof h.score === "number" && Number.isFinite(h.score)) p.scores[leg] = p.scores[leg] === undefined ? h.score : Math.max(p.scores[leg], h.score);
    }
  }
  ids(): number[] {
    return [...this.prov.keys()];
  }
}

export interface SubjectRetrieval {
  all: Verified[];
  truncated: boolean;
  pool: CandidatePool;
  /** deleted posts the history leg found, with the version that matched (never answers by themselves) */
  history: Array<{ author: string; permlink: string; title: string; at: number; content_hash: string | null }>;
  norms: { image: Norm | null; text: Norm | null };
}

/** Every candidate for the plan's subject, verified (v3's collect, with provenance). */
export async function retrieveSubject(env: Env, plan: QueryPlan, req: SearchRequest, ctx: SearchContext, notes: string[], depth: RetrievalDepth = NORMAL_DEPTH): Promise<SubjectRetrieval> {
  const tones = plan.filters.tones;
  const subject = plan.residual;
  const pool = new CandidatePool();
  const cosI = new Map<number, number>();
  const cosT = new Map<number, number>();
  let truncated = false;

  if (plan.concepts.length && req.type !== "blog") {
    pool.add("concept", await runClause<{ id: number; score: number }>(env.DB, conceptCandidates(req, plan.concepts, depth.concept, { tones })));
  }
  // query expansion (SEARCH_EXPANSION): the concepts' aliases in every language (rules, v3's), or the subject's own words only (off)
  const expand = String(env.SEARCH_EXPANSION ?? "rules").toLowerCase() !== "off";
  const words = [...new Set([...(expand ? plan.concepts.flatMap((k) => aliasesOf(k)) : []), ...tokens(subject, { keepHyphenated: false }).filter((t) => t.length > 2)])];
  const lexExpr = ftsAnyOf(words.filter((w) => !w.includes(" ")).slice(0, 24)) ?? ftsQuery(subject, "or");
  // bm25 is "lower is better": stored negated so that every leg's score grows with relevance
  if (lexExpr) pool.add("fts", (await runClause<{ id: number; score: number }>(env.DB, ftsCandidates(req, lexExpr, depth.fts, { tones }))).map((r) => ({ id: r.id, score: -r.score })));

  let normI: Norm | null = null;
  let normT: Norm | null = null;
  if (req.semantic && embeddingEnabled(env)) {
    try {
      const vec = await subjectVector(env, subject);
      // Every period must be searched for "first/last/how many": slice time so each slice holds
      // fewer artworks than Vectorize's top-100, and split any slice that still comes back full.
      if (req.type !== "blog") {
        const nSlices = Math.min(16, Math.max(1, Math.ceil(ctx.stats.artworks / 80)));
        const from = req.from ?? ctx.stats.minCreated;
        const to = req.to ?? ctx.stats.maxCreated + 1;
        const r = await adaptiveKnn(env, "image", vec, req, from, to, { slices: nSlices, budget: depth.visualBudget });
        for (const h of r.hits) cosI.set(h.id, h.score);
        pool.add("visual", r.hits);
        truncated = r.truncated;
      }
      const th = env.VEC_TEXT ? await knn(env, "text", vec, req, depth.semantic).catch(() => []) : [];
      for (const h of th) cosT.set(h.id, h.score);
      pool.add("semantic", th);
      const [bi, bt] = await Promise.all([loadBackground(env, "image"), env.VEC_TEXT ? loadBackground(env, "text") : Promise.resolve(null)]);
      normI = bi ? normFor(vec, bi) : null;
      normT = bt ? normFor(vec, bt) : null;
      if (!normI) notes.push("no background sample yet: image evidence not used (it is built by the cron or on first use)");
    } catch (e) {
      notes.push(`semantic evidence unavailable: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  const rows = await hydrateRows(env.DB, pool.ids(), req, { tones });
  const conceptRows = new Map<number, Map<string, number>>();
  if (plan.concepts.length && rows.size) {
    const all = [...rows.keys()];
    for (let i = 0; i < all.length; i += 400) {
      for (const k of await runClause<{ post_id: number; concept: string; confidence: number }>(env.DB, candidateConcepts(all.slice(i, i + 400), plan.concepts))) {
        const m = conceptRows.get(k.post_id) ?? new Map<string, number>();
        m.set(k.concept, Math.max(m.get(k.concept) ?? 0, k.confidence));
        conceptRows.set(k.post_id, m);
      }
    }
  }
  // Rows the vector legs did not return have a cosine at most the lowest returned one.
  const floorI = cosI.size ? Math.min(...cosI.values()) : null;
  const zMax = {
    image: normI && cosI.size ? zScore(Math.max(...[...cosI.entries()].filter(([id]) => rows.has(id)).map(([, c]) => c), -1), normI) : null,
    text: normT && cosT.size ? zScore(Math.max(...[...cosT.entries()].filter(([id]) => rows.has(id)).map(([, c]) => c), -1), normT) : null,
  };
  const all = [...rows.values()].map((row) => ({
    row,
    v: verify(row, plan, conceptRows.get(row.id) ?? new Map(), cosI.get(row.id) ?? (row.type === "artwork" ? floorI : null), cosT.get(row.id) ?? null, normI, normT, zMax),
  }));

  const history = depth.history > 0 && req.type !== "blog" ? await historyLeg(env, plan, words, depth.history).catch(() => []) : [];
  if (depth.history > 0) pool.legs.history = history.length;
  return { all, truncated, pool, history, norms: { image: normI, text: normT } };
}

/**
 * Deleted posts whose recorded titles name the subject (post_versions keeps every version of a
 * deleted post, with the title and the image hash it had). They have no tags, captions or vectors
 * any more, so they are context for the deeper modes, never an answer by themselves.
 */
async function historyLeg(env: Env, plan: QueryPlan, words: string[], limit: number): Promise<SubjectRetrieval["history"]> {
  const terms = [...new Set(words.filter((w) => w.length > 2 && !w.includes(" ")))].slice(0, 12);
  if (!terms.length) return [];
  const like = terms.map(() => "LOWER(v.title) LIKE ?").join(" OR ");
  const rows = await env.DB
    .prepare(
      `SELECT v.author, v.permlink, v.title, MIN(v.at) AS at, MAX(v.content_hash) AS content_hash FROM post_versions v JOIN posts p ON p.author = v.author AND p.permlink = v.permlink
       WHERE p.deleted = 1 AND v.source != 'snapshot' AND v.body_kind = 'image' AND (${like}) GROUP BY v.author, v.permlink ORDER BY at LIMIT ?`,
    )
    .bind(...terms.map((t) => `%${t}%`), limit)
    .all<{ author: string; permlink: string; title: string; at: number; content_hash: string | null }>();
  // whole words only ("cat" must not match "catalogue")
  return (rows.results ?? []).filter((r) => {
    const t = tokens(r.title ?? "");
    return terms.some((w) => t.includes(w)) || plan.concepts.some((k) => mentionsConcept(r.title ?? "", k));
  });
}

/** Metadata-only questions ("who posted the first artwork?"): exact SQL, no verification needed. */
export async function metadataRows(env: Env, plan: QueryPlan, req: SearchRequest, limit: number): Promise<Row[]> {
  const sort = plan.intent === "find_first" ? "oldest" : plan.intent === "find_last" ? "newest" : plan.intent === "top" ? (plan.sort === "payout" ? "payout" : "votes") : "newest";
  const ids = (await runClause<{ id: number }>(env.DB, browse({ ...req, sort }, null, limit, { tones: plan.filters.tones }))).map((r) => r.id);
  const rows = await hydrateRows(env.DB, ids, null);
  return ids.filter((id) => rows.has(id)).map((id) => rows.get(id)!);
}

/**
 * The posts a question is about. Without a subject, the filters alone decide (exact, SQL, rows
 * loaded by the operator that needs them). With one, every candidate is verified and those above
 * the threshold count; a subject that is exactly a post's title keeps only the posts carrying all
 * its words (v3: "Good vibes" is a name, not a mood).
 */
export async function resolveScope(
  env: Env,
  plan: QueryPlan,
  req: SearchRequest,
  ctx: SearchContext,
  opts: { threshold?: number; depth?: RetrievalDepth; notes: string[] },
): Promise<Scope & { history: SubjectRetrieval["history"]; norms: SubjectRetrieval["norms"] | null }> {
  if (!plan.residual) {
    return { kind: "metadata", plan, req, all: [], verified: [], truncated: false, provenance: new Map(), legs: {}, notes: opts.notes, history: [], norms: null };
  }
  const r = await retrieveSubject(env, plan, req, ctx, opts.notes, opts.depth ?? NORMAL_DEPTH);
  let verified = r.all.filter((x) => x.v.score >= (opts.threshold ?? 0.5));
  if (!plan.concepts.length && verified.some((x) => isTitleSubject(plan.residual, x.row.title ?? ""))) {
    verified = verified.filter((x) => x.v.lexical >= 0.5);
    opts.notes.push(`"${plan.residual}" is the title of a post: only posts with all its words count`);
  }
  return { kind: "subject", plan, req, all: r.all, verified, truncated: r.truncated, provenance: r.pool.prov, legs: r.pool.legs, notes: opts.notes, history: r.history, norms: r.norms };
}
