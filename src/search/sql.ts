// SQL building for D1. Filters and relevance are kept apart: filters become WHERE clauses over
// posts (p) LEFT JOIN artworks (a); relevance comes from FTS5 bm25, concepts and Vectorize, and is
// combined by the ranker in the Worker.

import type { Tone } from "./lexicon";
import type { SearchRequest, SortKey } from "./params";

export interface Clause {
  sql: string;
  params: unknown[];
}

const placeholders = (n: number) => Array.from({ length: n }, () => "?").join(", ");

/**
 * Tone predicates on the native-image features (thresholds from the live corpus, Oct 2026:
 * brightness median 0.43, colourfulness median ~66 on the 0..255 scale).
 */
export const TONE_SQL: Record<Tone, string> = {
  dark: "a.brightness < 0.30",
  light: "a.brightness > 0.55",
  greyscale: "a.colorfulness < 10",
  monochrome: "a.monochrome = 1",
  colorful: "a.colorfulness > 85",
  pastel: "(a.saturation < 0.40 AND a.brightness > 0.55)",
  high_contrast: "a.contrast > 0.30",
  minimal: "(a.color_count <= 16 OR a.palette_entropy < 2.2)",
};

/** WHERE fragment (without the WHERE keyword) for everything except free text. */
export function buildFilter(r: SearchRequest, extra: { tones?: Tone[]; minConcept?: number } = {}): Clause {
  const w: string[] = ["p.deleted = 0"];
  const params: unknown[] = [];

  if (r.type) (w.push("p.type = ?"), params.push(r.type));
  if (r.authors.length) (w.push(`p.author IN (${placeholders(r.authors.length)})`), params.push(...r.authors));
  for (const tag of r.tags) (w.push("EXISTS (SELECT 1 FROM post_tags t WHERE t.post_id = p.id AND t.tag = ?)"), params.push(tag));
  if (r.colors.length) (w.push(`a.primary_color IN (${placeholders(r.colors.length)})`), params.push(...r.colors));
  if (r.hasColors.length) {
    w.push(`EXISTS (SELECT 1 FROM artwork_colors c WHERE c.post_id = p.id AND c.bucket IN (${placeholders(r.hasColors.length)}) AND c.weight >= ?)`);
    params.push(...r.hasColors, r.minColorWeight);
  }
  if (r.sizes.length) (w.push(`a.size_class IN (${placeholders(r.sizes.length)})`), params.push(...r.sizes));
  if (r.minColors !== null) (w.push("a.color_count >= ?"), params.push(r.minColors));
  if (r.maxColors !== null) (w.push("a.color_count <= ?"), params.push(r.maxColors));
  if (r.minWidth !== null) (w.push("a.width >= ?"), params.push(r.minWidth));
  if (r.maxWidth !== null) (w.push("a.width <= ?"), params.push(r.maxWidth));
  if (r.minHeight !== null) (w.push("a.height >= ?"), params.push(r.minHeight));
  if (r.maxHeight !== null) (w.push("a.height <= ?"), params.push(r.maxHeight));
  if (r.from !== null) (w.push("p.created >= ?"), params.push(r.from));
  if (r.to !== null) (w.push("p.created < ?"), params.push(r.to));
  if (r.transparent !== null) (w.push("a.has_transparency = ?"), params.push(r.transparent ? 1 : 0));
  if (r.nsfw === "exclude") w.push("p.nsfw = 0 AND COALESCE(a.ai_nsfw, 0) < 0.7");
  else if (r.nsfw === "only") w.push("(p.nsfw = 1 OR COALESCE(a.ai_nsfw, 0) >= 0.7)");
  if (r.listed !== null) (w.push("p.listed = ?"), params.push(r.listed ? 1 : 0));
  if (r.aiTraining !== null) w.push(r.aiTraining ? "COALESCE(p.ai_training, 1) = 1" : "p.ai_training = 0");
  if (r.orientation.length) (w.push(`a.orientation IN (${placeholders(r.orientation.length)})`), params.push(...r.orientation));
  if (r.monochrome !== null) w.push(r.monochrome ? "a.monochrome = 1" : "COALESCE(a.monochrome, 0) = 0");
  if (r.background.length) (w.push(`a.background_name IN (${placeholders(r.background.length)})`), params.push(...r.background));
  for (const c of r.concepts) {
    w.push("EXISTS (SELECT 1 FROM artwork_concepts k WHERE k.post_id = p.id AND k.concept = ? AND k.confidence >= ?)");
    params.push(c, extra.minConcept ?? 0.5);
  }
  for (const t of extra.tones ?? []) w.push(TONE_SQL[t]);

  return { sql: w.join(" AND "), params };
}

// ---- FTS -----------------------------------------------------------------------------

const FTS_STOP = new Set(
  "a an the and or of to in on at by for with from is are was it its this that le la les un une des du de et ou en au aux der die das ein eine und oder zu im am mit von el los las y o del il lo gli e di".split(" "),
);

/**
 * Turn free text into a safe FTS5 MATCH expression. "and": every token (prefix on the last one).
 * "or": any non-stopword token (the relaxed fallback; v2 also OR-ed "the" and "in", which then
 * matched every long blog post).
 */
export function ftsQuery(q: string, mode: "and" | "or" = "and"): string | null {
  let tokens = q
    .normalize("NFKC")
    .split(/\s+/)
    .map((t) => t.replace(/["*():^{}[\]\\<>~|&!?,;.@#]/g, "").trim())
    .filter((t) => t.length > 0)
    .slice(0, 12);
  if (mode === "or") {
    const content = tokens.filter((t) => !FTS_STOP.has(t.toLowerCase()));
    if (content.length) tokens = content;
  }
  if (!tokens.length) return null;
  const quoted = tokens.map((t, i) => `"${t.replace(/"/g, '""')}"${mode === "and" && i === tokens.length - 1 && t.length >= 2 ? "*" : ""}`);
  return quoted.join(mode === "and" ? " " : " OR ");
}

/** OR of whole terms (spelling corrections, synonyms). */
export function ftsAnyOf(terms: string[]): string | null {
  const t = [...new Set(terms.map((x) => x.replace(/["*():^{}[\]\\<>~|&!?,;.@#]/g, "").trim()).filter(Boolean))].slice(0, 24);
  return t.length ? t.map((x) => `"${x.replace(/"/g, '""')}"`).join(" OR ") : null;
}

// column weights: title, description, body, tags, ai_caption, ai_tags, author
export const BM25 = "bm25(posts_fts, 6.0, 3.0, 1.0, 4.0, 2.5, 3.0, 0.5)";

/** Top-N full-text candidates that also satisfy the filters. Lower score = better. */
export function ftsCandidates(r: SearchRequest, match: string, limit: number, extra: { tones?: Tone[] } = {}): Clause {
  const f = buildFilter(r, extra);
  return {
    sql: `SELECT posts_fts.rowid AS id, ${BM25} AS score
          FROM posts_fts
          JOIN posts p ON p.id = posts_fts.rowid
          LEFT JOIN artworks a ON a.post_id = p.id
          WHERE posts_fts MATCH ? AND ${f.sql}
          ORDER BY score LIMIT ?`,
    params: [match, ...f.params, limit],
  };
}

/** Artworks carrying any of these concepts, strongest first (sum of confidences). */
export function conceptCandidates(r: SearchRequest, concepts: string[], limit: number, extra: { tones?: Tone[] } = {}): Clause {
  const f = buildFilter(r, extra);
  return {
    sql: `SELECT k.post_id AS id, SUM(k.confidence) AS score, MAX(k.confidence) AS best
          FROM artwork_concepts k
          JOIN posts p ON p.id = k.post_id
          LEFT JOIN artworks a ON a.post_id = p.id
          WHERE k.concept IN (${placeholders(concepts.length)}) AND ${f.sql}
          GROUP BY k.post_id ORDER BY score DESC, best DESC LIMIT ?`,
    params: [...concepts, ...f.params, limit],
  };
}

/** Artworks whose palette carries these colours (sum of bucket weights), for colour-led queries. */
export function colorCandidates(r: SearchRequest, colors: string[], limit: number): Clause {
  const f = buildFilter(r);
  return {
    sql: `SELECT c.post_id AS id, SUM(c.weight) AS score
          FROM artwork_colors c
          JOIN posts p ON p.id = c.post_id
          LEFT JOIN artworks a ON a.post_id = p.id
          WHERE c.bucket IN (${placeholders(colors.length)}) AND c.weight >= 0.08 AND ${f.sql}
          GROUP BY c.post_id ORDER BY score DESC LIMIT ?`,
    params: [...colors, ...f.params, limit],
  };
}

/** Candidates by a feature predicate only (tones, background), most recent first. */
export function predicateCandidates(r: SearchRequest, extra: { tones?: Tone[] }, limit: number): Clause {
  const f = buildFilter(r, extra);
  return {
    sql: `SELECT p.id AS id, 0 AS score FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql} ORDER BY p.created DESC LIMIT ?`,
    params: [...f.params, limit],
  };
}

// ---- browse (no text) -------------------------------------------------------------------

export interface KeysetCursor {
  v: number; // sort value
  id: number;
}

export function encodeCursor(c: KeysetCursor): string {
  return btoa(JSON.stringify(c)).replace(/=+$/, "");
}

export function decodeCursor(s: string | null): KeysetCursor | null {
  if (!s) return null;
  try {
    const o = JSON.parse(atob(s));
    return typeof o?.v === "number" && typeof o?.id === "number" ? { v: o.v, id: o.id } : null;
  } catch {
    return null;
  }
}

export function sortColumn(sort: SortKey): { col: string; dir: "ASC" | "DESC" } {
  switch (sort) {
    case "oldest":
      return { col: "p.created", dir: "ASC" };
    case "votes":
      return { col: "p.net_votes", dir: "DESC" };
    case "payout":
      return { col: "p.payout", dir: "DESC" };
    default:
      return { col: "p.created", dir: "DESC" };
  }
}

/** Keyset-paginated listing under the filters. Returns limit+1 rows so the caller can tell if there is more. */
export function browse(r: SearchRequest, cursor: KeysetCursor | null, limit: number, extra: { tones?: Tone[] } = {}): Clause {
  const f = buildFilter(r, extra);
  const { col, dir } = sortColumn(r.sort);
  const cmp = dir === "DESC" ? "<" : ">";
  const params: unknown[] = [...f.params];
  let where = f.sql;
  if (cursor) {
    where += ` AND (${col} ${cmp} ? OR (${col} = ? AND p.id ${cmp} ?))`;
    params.push(cursor.v, cursor.v, cursor.id);
  }
  params.push(limit + 1);
  return {
    sql: `SELECT p.id, ${col} AS sort_value FROM posts p LEFT JOIN artworks a ON a.post_id = p.id
          WHERE ${where} ORDER BY ${col} ${dir}, p.id ${dir} LIMIT ?`,
    params,
  };
}

// ---- hydration ---------------------------------------------------------------------------

export const POST_SELECT = `
  p.id, p.author, p.permlink, p.type, p.title, p.description, p.category, p.tags_json, p.app, p.nsfw, p.ai_training,
  p.royalty_pct, p.created, p.updated, p.deleted, p.net_votes, p.payout, p.children, p.listed, p.price, p.price_symbol,
  a.content_hash, a.mime, a.bytes, a.lossy, a.width, a.height, a.size_class, a.color_count, a.has_transparency,
  a.transparent_share, a.primary_color, a.background_hex, a.palette_json, a.buckets_json, a.phash, a.dhash,
  a.ai_caption, a.ai_subjects_json, a.ai_tags_json, a.ai_style, a.ai_mood, a.ai_text, a.ai_nsfw, a.ai_status,
  a.r2_orig_key, a.r2_up_key, a.up_width, a.up_height, a.up_factor, a.embed_hash, a.describe_hash, a.stats_hash,
  a.brightness, a.contrast, a.saturation, a.colorfulness, a.monochrome, a.orientation, a.aspect, a.background_name,
  a.background_share, a.foreground_share, a.palette_entropy, a.edge_density, a.symmetry_x,
  a.image_since, a.first_seen, a.first_seen_author, a.first_seen_permlink, a.first_seen_match, a.history_exact`;

/** Fetch rows for a set of ids, re-applying the filters (Vectorize cannot express all of them). */
export function hydrate(ids: number[], r: SearchRequest | null, extra: { tones?: Tone[] } = {}): Clause {
  const f = r ? buildFilter(r, extra) : { sql: "1 = 1", params: [] };
  return {
    sql: `SELECT ${POST_SELECT} FROM posts p LEFT JOIN artworks a ON a.post_id = p.id
          WHERE p.id IN (${ids.map((i) => Number(i) | 0).join(", ")}) AND ${f.sql}`,
    params: [...f.params],
  };
}

/** Concept rows of candidates for the requested concepts (ids inlined: integers only). */
export function candidateConcepts(ids: number[], concepts: string[]): Clause {
  return {
    sql: `SELECT post_id, concept, confidence, source FROM artwork_concepts
          WHERE post_id IN (${ids.map((i) => Number(i) | 0).join(", ")}) AND concept IN (${placeholders(concepts.length)})`,
    params: [...concepts],
  };
}

// ---- facets ----------------------------------------------------------------------------

export interface FacetQueries {
  primary_color: Clause;
  has_color: Clause;
  size_class: Clause;
  type: Clause;
  author: Clause;
  tag: Clause;
  month: Clause;
  color_count: Clause;
  transparency: Clause;
  orientation: Clause;
}

/**
 * Facet counts over a scope: everything matching the filters (browse), a full-text match, or a
 * set of result ids (text and hybrid search: the ranked candidates, whichever legs found them).
 */
export function facetQueries(r: SearchRequest, scope: string | { ids: number[] } | null): FacetQueries {
  const f = buildFilter(r);
  let base = f.sql;
  let params: unknown[] = f.params;
  if (typeof scope === "string") {
    base = `${f.sql} AND p.id IN (SELECT rowid FROM posts_fts WHERE posts_fts MATCH ?)`;
    params = [...f.params, scope];
  } else if (scope) {
    // integers we produced: inlined (thousands of ids are far above D1's 100 bound parameters)
    base = `${f.sql} AND p.id IN (${scope.ids.length ? scope.ids.map((i) => Number(i) | 0).join(",") : "NULL"})`;
  }
  const from = "FROM posts p LEFT JOIN artworks a ON a.post_id = p.id";
  const group = (keyExpr: string, extraFrom = "", extraWhere = "", extraParams: unknown[] = [], limit = 50): Clause => ({
    sql: `SELECT ${keyExpr} AS key, COUNT(*) AS n ${from} ${extraFrom} WHERE ${base} ${extraWhere} AND key IS NOT NULL GROUP BY key ORDER BY n DESC, key LIMIT ${limit}`,
    params: [...params, ...extraParams],
  });
  return {
    primary_color: group("a.primary_color"),
    has_color: {
      sql: `SELECT c.bucket AS key, COUNT(*) AS n ${from} JOIN artwork_colors c ON c.post_id = p.id WHERE ${base} AND c.weight >= ? GROUP BY key ORDER BY n DESC LIMIT 50`,
      params: [...params, r.minColorWeight],
    },
    size_class: group("a.size_class"),
    type: group("p.type"),
    author: group("p.author", "", "", [], 20),
    tag: {
      sql: `SELECT t.tag AS key, COUNT(*) AS n ${from} JOIN post_tags t ON t.post_id = p.id WHERE ${base} GROUP BY key ORDER BY n DESC LIMIT 40`,
      params,
    },
    month: {
      sql: `SELECT strftime('%Y-%m', p.created, 'unixepoch') AS key, COUNT(*) AS n ${from} WHERE ${base} GROUP BY key ORDER BY key DESC LIMIT 60`,
      params,
    },
    color_count: group(
      `CASE WHEN a.color_count IS NULL THEN NULL WHEN a.color_count <= 4 THEN '1-4' WHEN a.color_count <= 8 THEN '5-8' WHEN a.color_count <= 16 THEN '9-16'
            WHEN a.color_count <= 32 THEN '17-32' WHEN a.color_count <= 64 THEN '33-64' WHEN a.color_count <= 256 THEN '65-256' ELSE '257+' END`,
    ),
    transparency: group("CASE WHEN a.has_transparency IS NULL THEN NULL WHEN a.has_transparency = 1 THEN 'transparent' ELSE 'opaque' END"),
    orientation: group("a.orientation"),
  };
}
