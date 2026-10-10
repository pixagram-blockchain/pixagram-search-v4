// Search orchestration: plan → retrieve (several legs, adaptive depth) → hydrate (filters) →
// rank (features) → paginate. Browse mode (no text) is a keyset listing as in v2.

import type { Env } from "../env";
import { num } from "../env";
import { embedQueriesCached, embeddingEnabled } from "../enrich/embed";
import { fold, randomId } from "../lib/text";
import { sha256Hex } from "../lib/bytes";
import { emptyRequest, hasRestrictiveFilters, requestKey, type SearchRequest } from "./params";
import { planQuery, splitQuerySyntax, type QueryPlan } from "./planner";
import { loadContext, type SearchContext } from "./context";
import { isColorLed, rankCandidates, type Features, type QueryVariant, type RankInput, type Ranked } from "./ranker";
import { boost, reciprocalRankFusion } from "./rrf";
import { suggest, type Correction } from "./spell";
import {
  browse,
  candidateConcepts,
  colorCandidates,
  conceptCandidates,
  decodeCursor,
  encodeCursor,
  facetQueries,
  ftsAnyOf,
  ftsCandidates,
  ftsQuery,
  hydrate,
  POST_SELECT,
  predicateCandidates,
  type Clause,
} from "./sql";
import { knn, slicedKnn, timeSlices, type Hit } from "./vectors";
import { queryNorm } from "./background";
import type { Tone } from "./lexicon";
import { blendOrder, rerankBlend, rerankDepth, rerankRows } from "./reranker";

export interface ArtworkView {
  hash: string;
  mime: string;
  bytes: number;
  lossy: boolean;
  width: number | null;
  height: number | null;
  size_class: string | null;
  color_count: number | null;
  has_transparency: boolean | null;
  transparent_share: number | null;
  primary_color: string | null;
  background_hex: string | null;
  background_name: string | null;
  palette: Array<{ hex: string; share: number }>;
  buckets: Array<{ name: string; weight: number }>;
  phash: string | null;
  dhash: string | null;
  features: {
    brightness: number | null;
    contrast: number | null;
    saturation: number | null;
    colorfulness: number | null;
    monochrome: boolean | null;
    orientation: string | null;
    aspect: number | null;
  } | null;
  images: { original: string | null; upscaled: string | null };
  upscaled_size: { width: number; height: number; factor: number } | null;
  ai: { caption: string; subjects: string[]; tags: string[]; style: string | null; mood: string | null; text: string | null; nsfw: number | null; status: string | null } | null;
  history: { image_since: number | null; first_seen: number | null; first_seen_post: string | null; first_seen_match: string | null; exact: boolean | null } | null;
  stages: { stats: boolean; embed: boolean; describe: boolean };
}

export interface SearchItem {
  id: number;
  author: string;
  permlink: string;
  path: string; // /@author/permlink
  type: "artwork" | "blog";
  title: string;
  description: string;
  category: string | null;
  tags: string[];
  app: string | null;
  created: number;
  updated: number;
  net_votes: number;
  payout: number;
  children: number;
  nsfw: boolean;
  ai_training: boolean | null;
  listed: boolean;
  price: number | null;
  price_symbol: string | null;
  artwork: ArtworkView | null;
  score?: { fused: number; relevance?: number; ranks: Record<string, number>; duplicate_of?: number; rerank?: number };
  features?: Features & { quality: number; freshness: number };
}

export interface SearchResponse {
  query: string;
  mode: "browse" | "text" | "hybrid";
  items: SearchItem[];
  next_cursor: string | null;
  total_candidates?: number;
  facets?: Record<string, Array<{ key: string; n: number }>>;
  took_ms: number;
  notes?: string[];
  query_id?: string;
  did_you_mean?: string;
  rank?: "v3" | "rrf";
  plan?: Partial<QueryPlan>;
  legs?: Record<string, number>;
  /** v4 (spec §29): candidates per retrieval family, after the filters */
  retrieval?: Record<string, number>;
  /** v4: whether the cross-encoder reordered the top of the ranking, and which */
  reranked?: boolean;
  reranker?: string;
  /** v4.8: a text overview of the page (/search?overview=1, and every search of /query in the rich style) */
  overview?: import("./overview").SearchOverview;
}

type Row = Record<string, any>;

const parseArr = (s: string | null): any[] => {
  if (!s) return [];
  try {
    const v = JSON.parse(s);
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
};

export function rowToItem(r: Row): SearchItem {
  const art: ArtworkView | null =
    r.type === "artwork" && r.content_hash
      ? {
          hash: r.content_hash,
          mime: r.mime,
          bytes: r.bytes,
          lossy: r.lossy === 1,
          width: r.width,
          height: r.height,
          size_class: r.size_class,
          color_count: r.color_count,
          has_transparency: r.has_transparency === null ? null : r.has_transparency === 1,
          transparent_share: r.transparent_share,
          primary_color: r.primary_color,
          background_hex: r.background_hex,
          background_name: r.background_name ?? null,
          palette: parseArr(r.palette_json).slice(0, 12).map((p: any) => ({ hex: p.hex, share: p.share })),
          buckets: parseArr(r.buckets_json),
          phash: r.phash,
          dhash: r.dhash ?? null,
          features:
            r.brightness === null || r.brightness === undefined
              ? null
              : {
                  brightness: r.brightness,
                  contrast: r.contrast,
                  saturation: r.saturation,
                  colorfulness: r.colorfulness,
                  monochrome: r.monochrome === null ? null : r.monochrome === 1,
                  orientation: r.orientation,
                  aspect: r.aspect,
                },
          images: {
            original: r.r2_orig_key ? `/img/${r.r2_orig_key}` : null,
            upscaled: r.r2_up_key ? `/img/${r.r2_up_key}` : null,
          },
          upscaled_size: r.up_width ? { width: r.up_width, height: r.up_height, factor: r.up_factor } : null,
          ai: r.ai_caption
            ? { caption: r.ai_caption, subjects: parseArr(r.ai_subjects_json), tags: parseArr(r.ai_tags_json), style: r.ai_style, mood: r.ai_mood, text: r.ai_text, nsfw: r.ai_nsfw, status: r.ai_status ?? null }
            : null,
          history:
            r.image_since === undefined
              ? null
              : {
                  image_since: r.image_since,
                  first_seen: r.first_seen,
                  first_seen_post: r.first_seen_author ? `/@${r.first_seen_author}/${r.first_seen_permlink}` : null,
                  first_seen_match: r.first_seen_match ?? null,
                  exact: r.history_exact === null ? null : r.history_exact === 1,
                },
          stages: { stats: !!r.stats_hash, embed: !!r.embed_hash && r.embed_hash === r.content_hash, describe: !!r.describe_hash && r.describe_hash === r.content_hash && !!r.ai_caption },
        }
      : null;
  return {
    id: r.id,
    author: r.author,
    permlink: r.permlink,
    path: `/@${r.author}/${r.permlink}`,
    type: r.type,
    title: r.title,
    description: r.description,
    category: r.category,
    tags: parseArr(r.tags_json),
    app: r.app,
    created: r.created,
    updated: r.updated,
    net_votes: r.net_votes,
    payout: r.payout,
    children: r.children,
    nsfw: r.nsfw === 1,
    ai_training: r.ai_training === null ? null : r.ai_training === 1,
    listed: r.listed === 1,
    price: r.price,
    price_symbol: r.price_symbol,
    artwork: art,
  };
}

export async function runClause<T = Row>(db: D1Database, c: Clause): Promise<T[]> {
  const r = await db.prepare(c.sql).bind(...c.params).all<T>();
  return r.results ?? [];
}

/** Rows for ids, filters re-applied (Vectorize cannot express all of them). Order is not preserved. */
export async function hydrateRows(db: D1Database, ids: number[], r: SearchRequest | null, extra: { tones?: Tone[] } = {}): Promise<Map<number, Row>> {
  const out = new Map<number, Row>();
  const uniq = [...new Set(ids)];
  for (let i = 0; i < uniq.length; i += 300) {
    for (const row of await runClause(db, hydrate(uniq.slice(i, i + 300), r, extra))) out.set(row.id as number, row);
  }
  return out;
}

/** Fetch and order rows for ids (order preserved). Rows failing the filters are dropped. */
export async function hydrateOrdered(db: D1Database, ids: number[], r: SearchRequest | null): Promise<SearchItem[]> {
  const rows = await hydrateRows(db, ids, r);
  return ids.filter((id) => rows.has(id)).map((id) => rowToItem(rows.get(id)!));
}

// ---- retrieval ------------------------------------------------------------------------------------

export interface Candidate {
  id: number;
  ranks: Record<string, number>; // 1-based rank per leg
  bm25: number | null;
  bm25Leg: "and" | "or" | "spell" | null;
  /** cosine per query variant (Retrieval.variants), null where that variant's leg did not return it */
  cosImage: Array<number | null>;
  cosText: Array<number | null>;
}

/** The query as typed, plus its spelling-corrected form when some word is unknown to the corpus. */
export interface RetrievalVariant {
  text: string;
  factor: number;
  vector: number[] | null;
  cosImageFloor: number | null;
  cosTextFloor: number | null;
}

export interface Retrieval {
  plan: QueryPlan;
  req: SearchRequest;
  cands: Map<number, Candidate>;
  rows: Map<number, Row>;
  legs: Record<string, number>;
  corrections: Correction[];
  variants: RetrievalVariant[];
  semantic: "on" | "off" | "failed";
  notes: string[];
  matchExpr: string | null;
}

const VEC_TOPK = 100;

function addHits(cands: Map<number, Candidate>, leg: string, hits: Array<{ id: number; score: number }>, apply?: (c: Candidate, h: { id: number; score: number }) => void): void {
  hits.forEach((h, i) => {
    let c = cands.get(h.id);
    if (!c) {
      c = { id: h.id, ranks: {}, bm25: null, bm25Leg: null, cosImage: [], cosText: [] };
      cands.set(h.id, c);
    }
    if (c.ranks[leg] === undefined) c.ranks[leg] = i + 1;
    apply?.(c, h);
  });
}

/** Which full-text leg says the most about a candidate: every word > a corrected word > any word. */
const LEG_PRECEDENCE = { and: 3, spell: 2, or: 1 } as const;

/**
 * Record a full-text hit. A candidate keeps the bm25 of its most specific leg, not the lowest
 * score across legs: scores of different MATCH expressions are not comparable (the OR query often
 * scores an exact all-words match better than the AND query does), and taking the minimum labelled
 * exact matches "or" while prefix-only matches kept "and".
 */
function setBm25(leg: "and" | "or" | "spell") {
  return (c: Candidate, h: { score: number }) => {
    const cur = c.bm25Leg ? LEG_PRECEDENCE[c.bm25Leg] : 0;
    if (LEG_PRECEDENCE[leg] > cur) {
      c.bm25 = h.score;
      c.bm25Leg = leg;
    } else if (LEG_PRECEDENCE[leg] === cur && (c.bm25 === null || h.score < c.bm25)) c.bm25 = h.score;
  };
}

/** Apply spelling corrections to the query text ("elodrado capital" → "eldorado capital"). */
export function correctedText(text: string, corrections: Correction[]): string {
  let s = fold(text);
  for (const c of corrections) s = s.replace(new RegExp(`(^|[^\\p{L}])${c.from}(?=$|[^\\p{L}])`, "gu"), `$1${c.to}`);
  return s;
}

/**
 * First stage. Legs run in parallel; depth adapts to the query (longer queries get deeper text
 * legs) and to the filters (restrictive filters trigger time-sliced kNN when too few vector
 * candidates survive hydration). A query word no document contains is spell-checked first; the
 * corrected query then runs as a second variant through the text and vector legs.
 */
export async function retrieve(env: Env, r: SearchRequest, ctx: SearchContext, opts: { need: number; tones?: Tone[]; plan?: QueryPlan; embedText?: string } = { need: 24 }): Promise<Retrieval> {
  const plan = opts.plan ?? planQuery(r.q, { authors: ctx.authors, mode: "search", now: ctx.now });
  const req: SearchRequest = {
    ...r,
    authors: [...new Set([...r.authors, ...(plan.filters.authors ?? [])])],
    tags: [...new Set([...r.tags, ...(plan.filters.tags ?? [])])],
  };
  const notes: string[] = [];
  const cands = new Map<number, Candidate>();
  const legs: Record<string, number> = {};
  const words = plan.text.split(/\s+/).filter(Boolean).length;
  const ftsLimit = words >= 4 ? 400 : 200;
  const extra = { tones: opts.tones };
  const andExpr = ftsQuery(plan.text, "and");
  const orExpr = words > 1 ? ftsQuery(plan.text, "or") : null;
  const embedText = opts.embedText ?? plan.text;
  const wantVec = req.semantic && embeddingEnabled(env) && !!embedText;

  let corrections: Correction[] = [];
  if (req.expand) corrections = await suggest(env.DB, plan.text).catch((e) => (notes.push(`spelling unavailable: ${errMsg(e)}`), []));
  const variants: RetrievalVariant[] = [{ text: embedText, factor: 1, vector: null, cosImageFloor: null, cosTextFloor: null }];
  if (corrections.length) variants.push({ text: correctedText(embedText, corrections), factor: ctx.weights.correctedFactor, vector: null, cosImageFloor: null, cosTextFloor: null });

  const tasks: Array<Promise<void>> = [];
  if (andExpr) {
    tasks.push(
      runClause<{ id: number; score: number }>(env.DB, ftsCandidates(req, andExpr, ftsLimit, extra)).then((hits) => {
        legs.fts = hits.length;
        addHits(cands, "fts", hits, setBm25("and"));
      }),
    );
  }
  if (orExpr && orExpr !== andExpr) {
    tasks.push(
      runClause<{ id: number; score: number }>(env.DB, ftsCandidates(req, orExpr, ftsLimit, extra)).then((hits) => {
        legs.fts_or = hits.length;
        addHits(cands, "fts_or", hits, setBm25("or"));
      }),
    );
  }
  const spellExpr = ftsAnyOf(corrections.map((c) => c.to));
  if (spellExpr) {
    tasks.push(
      runClause<{ id: number; score: number }>(env.DB, ftsCandidates(req, spellExpr, 100, extra)).then((hits) => {
        legs.spell = hits.length;
        addHits(cands, "spell", hits, setBm25("spell"));
      }),
    );
  }
  if (req.expand) {
    if (plan.concepts.length && req.type !== "blog") {
      tasks.push(
        runClause<{ id: number; score: number }>(env.DB, conceptCandidates(req, plan.concepts, 300, extra)).then((hits) => {
          legs.concept = hits.length;
          addHits(cands, "concept", hits);
        }),
      );
    }
    if (plan.hints.colors.length && req.type !== "blog") {
      tasks.push(
        runClause<{ id: number; score: number }>(env.DB, colorCandidates(req, plan.hints.colors, 150)).then((hits) => {
          legs.color = hits.length;
          addHits(cands, "color", hits);
        }),
      );
    }
    if ((plan.hints.tones.length || plan.hints.background.length) && req.type !== "blog") {
      const bgReq = plan.hints.background.length ? { ...req, background: plan.hints.background } : req;
      tasks.push(
        runClause<{ id: number; score: number }>(env.DB, predicateCandidates(bgReq, { tones: plan.hints.tones }, 150)).then((hits) => {
          legs.tone = hits.length;
          addHits(cands, "tone", hits);
        }),
      );
    }
  }

  let semantic: Retrieval["semantic"] = wantVec ? "on" : "off";
  let imageLegFull = false;
  if (wantVec) {
    tasks.push(
      (async () => {
        try {
          const vectors = await embedQueriesCached(env, variants.map((v) => v.text));
          await Promise.all(
            variants.map(async (v, k) => {
              v.vector = vectors[k];
              const [img, txt] = await Promise.all([
                req.type !== "blog" ? knn(env, "image", v.vector, req, VEC_TOPK) : Promise.resolve([] as Hit[]),
                env.VEC_TEXT ? knn(env, "text", v.vector, req, VEC_TOPK).catch(() => [] as Hit[]) : Promise.resolve([] as Hit[]),
              ]);
              const tag = k === 0 ? "" : `_${k}`;
              legs[`vec${tag}`] = img.length;
              legs[`txt${tag}`] = txt.length;
              addHits(cands, `vec${tag}`, img, (c, h) => (c.cosImage[k] = h.score));
              addHits(cands, `txt${tag}`, txt, (c, h) => (c.cosText[k] = h.score));
              if (img.length >= VEC_TOPK) v.cosImageFloor = img[img.length - 1].score;
              if (txt.length >= VEC_TOPK) v.cosTextFloor = txt[txt.length - 1].score;
              if (k === 0) imageLegFull = img.length >= VEC_TOPK;
            }),
          );
        } catch (e) {
          semantic = "failed";
          notes.push(`semantic search unavailable: ${errMsg(e)}`);
        }
      })(),
    );
  }
  await Promise.all(tasks);

  let rows = await hydrateRows(env.DB, [...cands.keys()], req, extra);

  // Adaptive depth: under restrictive filters, Vectorize's top-100 may hold few survivors. Widen
  // with time-sliced kNN (each slice returns its own top-100) until enough candidates survive.
  const qv = variants[0].vector;
  if (qv && imageLegFull && rows.size < opts.need + 10 && hasRestrictiveFilters(req) && req.type !== "blog") {
    for (const n of [4, 12]) {
      const slices = timeSlices(req.from ?? ctx.stats.minCreated, req.to ?? ctx.stats.maxCreated + 1, n);
      const more = await slicedKnn(env, "image", qv, req, slices, VEC_TOPK);
      const fresh = more.filter((h) => !cands.has(h.id));
      addHits(cands, "vec_wide", fresh, (c, h) => (c.cosImage[0] = h.score));
      legs.vec_wide = (legs.vec_wide ?? 0) + fresh.length;
      const extraRows = await hydrateRows(env.DB, fresh.map((h) => h.id), req, extra);
      for (const [id, row] of extraRows) rows.set(id, row);
      notes.push(`widened vector recall over ${n} time slices (+${extraRows.size})`);
      if (rows.size >= opts.need + 10 || fresh.length === 0) break;
    }
  }
  // Candidates that failed the filters are gone from rows; drop them from the map too.
  for (const id of [...cands.keys()]) if (!rows.has(id)) cands.delete(id);
  rows = new Map([...rows].filter(([id]) => cands.has(id)));

  return { plan, req, cands, rows, legs, corrections, variants, semantic, notes, matchExpr: andExpr };
}

// ---- ranking ----------------------------------------------------------------------------------------

export async function rankRetrieval(env: Env, ret: Retrieval, ctx: SearchContext, exec?: ExecutionContext): Promise<Ranked[]> {
  const ids = [...ret.cands.keys()];
  const concepts = new Map<number, Map<string, number>>();
  if (ret.plan.concepts.length && ids.length) {
    for (let i = 0; i < ids.length; i += 400) {
      for (const k of await runClause<{ post_id: number; concept: string; confidence: number }>(env.DB, candidateConcepts(ids.slice(i, i + 400), ret.plan.concepts))) {
        const m = concepts.get(k.post_id) ?? new Map<string, number>();
        m.set(k.concept, Math.max(m.get(k.concept) ?? 0, k.confidence));
        concepts.set(k.post_id, m);
      }
    }
  }
  const inputs: RankInput[] = ids.map((id) => toRankInput(ret.rows.get(id)!, ret.cands.get(id)!, concepts.get(id) ?? new Map()));
  const variants: QueryVariant[] = await Promise.all(
    ret.variants.map(async (v) => ({
      text: v.text,
      factor: v.factor,
      imageNorm: await queryNorm(env, "image", v.vector, exec),
      textNorm: env.VEC_TEXT ? await queryNorm(env, "text", v.vector, exec) : null,
      cosImageFloor: v.cosImageFloor,
      cosTextFloor: v.cosTextFloor,
    })),
  );
  return rankCandidates(inputs, { plan: ret.plan, weights: ctx.weights, calibration: ctx.calibration, now: ctx.now, variants, voteRate: ctx.stats.voteRate });
}

export function toRankInput(row: Row, c: Candidate, concepts: Map<string, number>): RankInput {
  return {
    id: row.id,
    type: row.type,
    author: row.author,
    title: row.title ?? "",
    description: row.description ?? "",
    tags: parseArr(row.tags_json),
    aiCaption: row.ai_caption ?? null,
    aiTags: parseArr(row.ai_tags_json),
    created: row.created,
    netVotes: row.net_votes ?? 0,
    phash: row.phash ?? null,
    buckets: parseArr(row.buckets_json),
    brightness: row.brightness ?? null,
    contrast: row.contrast ?? null,
    saturation: row.saturation ?? null,
    colorfulness: row.colorfulness ?? null,
    monochrome: row.monochrome === null || row.monochrome === undefined ? null : row.monochrome === 1,
    colorCount: row.color_count ?? null,
    paletteEntropy: row.palette_entropy ?? null,
    orientation: row.orientation ?? null,
    backgroundName: row.background_name ?? null,
    bm25: c.bm25,
    bm25Leg: c.bm25Leg,
    cosImage: c.cosImage,
    cosText: c.cosText,
    concepts,
  };
}

/** v2's ordering (RRF over the AND-then-OR full-text list and the image list, then the boost). */
export function rrfOrder(ret: Retrieval, ctx: SearchContext): Ranked[] {
  const byRank = (leg: string) => [...ret.cands.values()].filter((c) => c.ranks[leg] !== undefined).sort((a, b) => a.ranks[leg] - b.ranks[leg]).map((c) => c.id);
  const fts = byRank("fts");
  const ftsList = fts.length < 5 && (ret.legs.fts_or ?? 0) > 0 ? [...fts, ...byRank("fts_or").filter((id) => !fts.includes(id))] : fts;
  const fused = reciprocalRankFusion([
    { name: "fts", ids: ftsList },
    { name: "vec", ids: byRank("vec") },
  ]);
  const zero = { title: 0, lexical: 0, coverage: 0, tag: 0, author: 0, concept: 0, sem: 0, sem_txt: 0, color: 0, tone: 0, orientation: 0 };
  return fused
    .filter((f) => ret.rows.has(f.id))
    .map((f) => {
      const row = ret.rows.get(f.id)!;
      return { id: f.id, final: boost(f.score, row.net_votes ?? 0, row.created, ctx.now), relevance: f.score, features: zero, quality: 0, freshness: 0 };
    })
    .sort((a, b) => b.final - a.final || a.id - b.id);
}

// ---- /search ----------------------------------------------------------------------------------------

interface CachedRanking {
  /** the request this ranking answers (requestKey), verified on every cache read */
  key: string;
  /** x: what the ranker saw (features, quality, freshness, and flags the trainer needs); ce: the cross-encoder's score */
  ranked: Array<{ id: number; f: number; r: number; d?: number; ce?: number; ranks: Record<string, number>; x?: Features & { quality: number; freshness: number; color_led: number; dup: number } }>;
  retrieval?: Record<string, number>;
  reranker?: string | null;
  mode: "text" | "hybrid";
  notes: string[];
  did_you_mean?: string;
  plan: Partial<QueryPlan>;
  legs: Record<string, number>;
  matchExpr: string | null;
}

export async function search(env: Env, r: SearchRequest, exec?: ExecutionContext): Promise<SearchResponse> {
  const t0 = Date.now();

  // "@laura" or "#dragon" alone: nothing left to rank by, so browse with those filters.
  if (r.q) {
    const syntax = splitQuerySyntax(r.q);
    if (!syntax.text && (syntax.mentions.length || syntax.hashtags.length)) {
      r = {
        ...r,
        q: "",
        authors: [...new Set([...r.authors, ...syntax.mentions])],
        tags: [...new Set([...r.tags, ...syntax.hashtags])],
        sort: r.sort === "relevance" ? "newest" : r.sort,
      };
    }
  }

  // ---- browse mode ----------------------------------------------------------------------------------
  if (!r.q) {
    const cursor = decodeCursor(r.cursor);
    const rows = await runClause<{ id: number; sort_value: number }>(env.DB, browse(r, cursor, r.limit));
    const page = rows.slice(0, r.limit);
    const items = await hydrateOrdered(env.DB, page.map((x) => x.id), null);
    const last = page[page.length - 1];
    const next = rows.length > r.limit && last ? encodeCursor({ v: last.sort_value, id: last.id }) : null;
    const resp: SearchResponse = { query: "", mode: "browse", items, next_cursor: next, took_ms: Date.now() - t0 };
    if (r.facets) resp.facets = await facets(env, r, null);
    return resp;
  }

  // ---- text / hybrid mode -----------------------------------------------------------------------------
  const offset = Math.max(0, parseInt(r.cursor ?? "0", 10) || 0);
  // 60 s result cache. The key is a SHA-256 of the full request (filters included) and the entry
  // carries the request itself, checked on read: a cached ranking can only ever be served to the
  // exact request that produced it (a 32-bit hash could be collided on purpose to serve an
  // nsfw=only ranking to an nsfw=exclude visitor).
  const reqKey = requestKey(r);
  const cacheKey = `s3:${(await sha256Hex(new TextEncoder().encode(reqKey))).slice(0, 40)}`;
  let cached = (await env.CACHE.get(cacheKey, "json").catch(() => null)) as CachedRanking | null;
  if (cached && cached.key !== reqKey) cached = null;
  const ctx = await loadContext(env);
  if (!cached) {
    const ret = await retrieve(env, r, ctx, { need: offset + r.limit });
    const ranked: Array<Ranked & { rerank?: number }> = r.rank === "rrf" ? rrfOrder(ret, ctx) : await rankRetrieval(env, ret, ctx, exec);
    const notes = [...ret.notes];
    // v4: the cross-encoder reorders the top of a relevance ranking when asked (rerank=1, spec §13, §29)
    let order = ranked;
    let reranker: string | null = null;
    if (r.rerank && r.rank === "v3" && r.sort === "relevance" && ranked.length > 1) {
      try {
        const k = rerankDepth(env);
        const head = ranked.slice(0, k).map((x) => ret.rows.get(x.id)).filter((x): x is Row => !!x);
        const rr = await rerankRows(env, r.q, head);
        if (rr) {
          order = blendOrder(ranked, rr.scores, rerankBlend(env), k);
          reranker = rr.model;
        }
      } catch (e) {
        notes.push(`reranker unavailable: ${errMsg(e)}`);
      }
    }
    const reordered = applySort(order, ret.rows, r.sort);
    let didYouMean: string | undefined;
    if (ret.corrections.length) didYouMean = correctedText(ret.plan.text, ret.corrections);
    const colorLed = isColorLed(ret.plan) ? 1 : 0;
    cached = {
      key: reqKey,
      ranked: reordered.map((x) => ({
        id: x.id, f: x.final, r: x.relevance, d: x.duplicateOf, ranks: ret.cands.get(x.id)?.ranks ?? {},
        ...(typeof (x as { rerank?: number }).rerank === "number" ? { ce: (x as { rerank?: number }).rerank } : {}),
        x: { ...x.features, quality: x.quality, freshness: x.freshness, color_led: colorLed, dup: x.duplicateOf ? 1 : 0 },
      })),
      retrieval: retrievalCounts(ret),
      reranker,
      mode: ret.semantic === "on" ? "hybrid" : "text",
      notes,
      did_you_mean: didYouMean,
      plan: compactPlan(ret.plan),
      legs: ret.legs,
      matchExpr: ret.matchExpr,
    };
    // Degraded answers (vector leg down) are not cached, so the next request retries it.
    if (cached.ranked.length && ret.semantic !== "failed") exec?.waitUntil(env.CACHE.put(cacheKey, JSON.stringify(cached), { expirationTtl: 60 }).catch(() => {}));
  }

  const pageIds = cached.ranked.slice(offset, offset + r.limit);
  // Filters are applied once more on the page itself (cheap, and a post deleted or flagged since
  // the ranking was cached drops out).
  const items = await hydrateOrdered(env.DB, pageIds.map((x) => x.id), r);
  const meta = new Map(pageIds.map((x) => [x.id, x]));
  for (const it of items) {
    const m = meta.get(it.id)!;
    it.score = { fused: round(m.f), relevance: round(m.r), ranks: m.ranks, ...(m.d ? { duplicate_of: m.d } : {}), ...(m.ce !== undefined ? { rerank: round(m.ce) } : {}) };
    if (r.explain && m.x) it.features = Object.fromEntries(Object.entries(m.x).map(([k, v]) => [k, round(v as number)])) as any;
  }
  const qid = randomId();
  const resp: SearchResponse = {
    query: r.q,
    mode: cached.mode,
    items,
    next_cursor: offset + r.limit < cached.ranked.length ? String(offset + r.limit) : null,
    total_candidates: cached.ranked.length,
    took_ms: Date.now() - t0,
    query_id: qid,
    rank: r.rank,
    retrieval: cached.retrieval,
    reranked: !!cached.reranker,
    ...(cached.reranker ? { reranker: cached.reranker } : {}),
  };
  if (cached.notes.length) resp.notes = cached.notes;
  if (cached.did_you_mean) resp.did_you_mean = cached.did_you_mean;
  if (r.explain) {
    resp.plan = cached.plan;
    resp.legs = cached.legs;
  }
  if (r.facets) resp.facets = await facets(env, r, { ids: cached.ranked.slice(0, 2000).map((x) => x.id) });

  exec?.waitUntil(logQuery(env, qid, r, resp, cached, offset).catch(() => {}));
  return resp;
}

/** Candidates per retrieval family, after the filters (spec §29: fts, semantic = text vectors, visual = image vectors…). */
export function retrievalCounts(ret: Retrieval): Record<string, number> {
  const fam: Record<string, RegExp> = { fts: /^(fts|fts_or|spell)$/, semantic: /^txt/, visual: /^vec/, concept: /^concept$/, color: /^color$/, tone: /^tone$/ };
  const out: Record<string, number> = {};
  for (const [name, re] of Object.entries(fam)) {
    let n = 0;
    for (const c of ret.cands.values()) if (Object.keys(c.ranks).some((k) => re.test(k))) n++;
    if (n || name === "fts" || name === "semantic" || name === "visual") out[name] = n;
  }
  return out;
}

function applySort<T extends Ranked>(ranked: T[], rows: Map<number, Row>, sort: SearchRequest["sort"]): T[] {
  if (sort === "relevance") return ranked;
  const v = (id: number, k: string) => (rows.get(id)?.[k] as number) ?? 0;
  const out = [...ranked];
  if (sort === "newest") out.sort((a, b) => v(b.id, "created") - v(a.id, "created"));
  else if (sort === "oldest") out.sort((a, b) => v(a.id, "created") - v(b.id, "created"));
  else if (sort === "votes") out.sort((a, b) => v(b.id, "net_votes") - v(a.id, "net_votes"));
  else if (sort === "payout") out.sort((a, b) => v(b.id, "payout") - v(a.id, "payout"));
  return out;
}

export function compactPlan(p: QueryPlan): Partial<QueryPlan> {
  const { conceptMatches: _m, ...rest } = p;
  return rest;
}

const round = (x: number) => Math.round(x * 10000) / 10000;
const roundAll = (o: Record<string, number>) => Object.fromEntries(Object.entries(o).map(([k, v]) => [k, round(v)]));

async function logQuery(env: Env, qid: string, r: SearchRequest, resp: SearchResponse, c: CachedRanking, offset: number): Promise<void> {
  const stmts = [
    env.DB.prepare("INSERT INTO query_log (q, filters, results, ms, at, qid, mode, plan_json, top_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)").bind(
      r.q,
      requestKey(r),
      resp.total_candidates ?? 0,
      resp.took_ms,
      Math.floor(Date.now() / 1000),
      qid,
      `${resp.mode}/${r.rank}`,
      JSON.stringify({ concepts: c.plan.concepts, hints: c.plan.hints, filters: c.plan.filters }),
      JSON.stringify(resp.items.map((x) => x.id)), // what this page showed (feedback is checked against it)
    ),
  ];
  // Features of what was shown, for learning to rank (joined with /feedback later).
  // Only relevance-ordered v3 rankings (a page sorted by date says nothing about relevance), a
  // sample of them, and at most the first 24 results shown: anonymous traffic must not be able to
  // write unbounded rows.
  if (Math.random() < num(env.RANK_LOG_SAMPLE, 0.25) && r.rank === "v3" && r.sort === "relevance") {
    const shown = c.ranked.slice(offset, offset + Math.min(24, resp.items.length));
    const now = Math.floor(Date.now() / 1000);
    shown.forEach((x, i) => {
      if (!x.x) return;
      stmts.push(env.DB.prepare("INSERT OR IGNORE INTO rank_log (qid, post_id, rank, features, at) VALUES (?, ?, ?, ?, ?)").bind(qid, x.id, offset + i + 1, JSON.stringify(roundAll(x.x)), now));
    });
  }
  await env.DB.batch(stmts);
}

export async function facets(env: Env, r: SearchRequest, scope: string | { ids: number[] } | null): Promise<Record<string, Array<{ key: string; n: number }>>> {
  const q = facetQueries(r, scope);
  const names = Object.keys(q) as Array<keyof typeof q>;
  const results = await env.DB.batch(names.map((n) => env.DB.prepare(q[n].sql).bind(...q[n].params)));
  const out: Record<string, Array<{ key: string; n: number }>> = {};
  names.forEach((n, i) => {
    out[n] = ((results[i].results ?? []) as Array<{ key: string; n: number }>).filter((x) => x.key !== null);
  });
  return out;
}

/** One live post (deleted posts are not served; their versions stay readable through /history). */
export async function getItem(env: Env, idOrRef: { id: number } | { author: string; permlink: string }): Promise<SearchItem | null> {
  const row =
    "id" in idOrRef
      ? await env.DB.prepare(`SELECT ${POST_SELECT} FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.id = ? AND p.deleted = 0`).bind(idOrRef.id).first<Row>()
      : await env.DB.prepare(`SELECT ${POST_SELECT} FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ? AND p.deleted = 0`).bind(idOrRef.author, idOrRef.permlink).first<Row>();
  return row ? rowToItem(row) : null;
}

export { emptyRequest };

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
