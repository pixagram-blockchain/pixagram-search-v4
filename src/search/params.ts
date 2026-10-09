// Query-string parsing for the search API. Every filter is optional; unknown values are dropped.

import { COLOR_NAMES } from "../enrich/color";
import { SIZE_CLASSES, type SizeClass } from "../enrich/stats";

export type SortKey = "relevance" | "newest" | "oldest" | "votes" | "payout";
export type NsfwMode = "exclude" | "include" | "only";
export type RankMode = "v3" | "rrf";

export interface SearchRequest {
  q: string;
  type: "artwork" | "blog" | null;
  authors: string[];
  tags: string[];
  /** primary colour in (...) */
  colors: string[];
  /** any palette bucket in (...) with weight >= minColorWeight */
  hasColors: string[];
  minColorWeight: number;
  sizes: SizeClass[];
  minColors: number | null;
  maxColors: number | null;
  minWidth: number | null;
  maxWidth: number | null;
  minHeight: number | null;
  maxHeight: number | null;
  /** unix seconds, inclusive / exclusive */
  from: number | null;
  to: number | null;
  transparent: boolean | null;
  nsfw: NsfwMode;
  listed: boolean | null;
  aiTraining: boolean | null;
  /** v3 filters */
  orientation: Array<"portrait" | "landscape" | "square">;
  monochrome: boolean | null;
  background: string[]; // named colour of the backdrop, or "transparent"
  concepts: string[]; // canonical concept ids that must be present (artwork_concepts)
  sort: SortKey;
  limit: number;
  /** numeric offset within the ranked candidates, or a keyset cursor when browsing */
  cursor: string | null;
  facets: boolean;
  /** use the vector indexes when a query is present (default true) */
  semantic: boolean;
  /** v3: "v3" (feature ranker, default) or "rrf" (v2's fusion, for comparisons) */
  rank: RankMode;
  /** v3: planner hints from the query text (colour words, @author, synonyms, spelling); default on */
  expand: boolean;
  /** v3: include per-item features and the query plan */
  explain: boolean;
  /** v4: reorder the top of the ranking with the cross-encoder (search/reranker.ts); default off */
  rerank: boolean;
}

const COLOR_SET = new Set(COLOR_NAMES);
const SIZE_SET = new Set<string>(SIZE_CLASSES);
const SORTS = new Set<string>(["relevance", "newest", "oldest", "votes", "payout"]);
const ORIENT = new Set(["portrait", "landscape", "square"]);

function multi(sp: URLSearchParams, key: string): string[] {
  const out: string[] = [];
  for (const v of sp.getAll(key)) for (const part of v.split(",")) {
    const s = part.trim().toLowerCase();
    if (s) out.push(s);
  }
  return [...new Set(out)];
}

function num(sp: URLSearchParams, key: string): number | null {
  const v = sp.get(key);
  if (v === null || v.trim() === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function boolOrNull(sp: URLSearchParams, key: string): boolean | null {
  const v = sp.get(key);
  if (v === null || v === "") return null;
  if (/^(1|true|yes)$/i.test(v)) return true;
  if (/^(0|false|no)$/i.test(v)) return false;
  return null;
}

/** Accepts unix seconds, unix millis, or anything Date.parse understands (ISO dates). */
export function parseDate(v: string | null): number | null {
  if (!v) return null;
  const s = v.trim();
  if (/^\d+$/.test(s)) {
    const n = Number(s);
    return n > 1e11 ? Math.floor(n / 1000) : n; // millis vs seconds
  }
  const t = Date.parse(s.length === 10 ? `${s}T00:00:00Z` : s);
  return Number.isFinite(t) ? Math.floor(t / 1000) : null;
}

/** User text without control characters (a NUL breaks full-text queries), trimmed. */
export function cleanText(s: string): string {
  return s.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, " ").trim();
}

export function parseSearchRequest(sp: URLSearchParams): SearchRequest {
  const typeRaw = (sp.get("type") ?? "").toLowerCase();
  const type = typeRaw === "artwork" || typeRaw === "art" ? "artwork" : typeRaw === "blog" || typeRaw === "post" ? "blog" : null;
  const sortRaw = (sp.get("sort") ?? "").toLowerCase();
  const q = cleanText(sp.get("q") ?? "").slice(0, 200);
  const nsfwRaw = (sp.get("nsfw") ?? "exclude").toLowerCase();
  const limitRaw = num(sp, "limit");
  const sizes = multi(sp, "size").filter((s) => SIZE_SET.has(s)) as SizeClass[];
  return {
    q,
    type,
    authors: multi(sp, "author").map((a) => a.replace(/^@/, "")).slice(0, 10),
    tags: multi(sp, "tag").slice(0, 10),
    colors: multi(sp, "color").filter((c) => COLOR_SET.has(c)).slice(0, 10),
    hasColors: multi(sp, "has_color").filter((c) => COLOR_SET.has(c)).slice(0, 10),
    minColorWeight: Math.min(1, Math.max(0, num(sp, "min_color_weight") ?? 0.08)),
    sizes,
    minColors: num(sp, "min_colors"),
    maxColors: num(sp, "max_colors"),
    minWidth: num(sp, "min_width"),
    maxWidth: num(sp, "max_width"),
    minHeight: num(sp, "min_height"),
    maxHeight: num(sp, "max_height"),
    from: parseDate(sp.get("from")),
    to: parseDate(sp.get("to")),
    transparent: boolOrNull(sp, "transparent"),
    nsfw: nsfwRaw === "include" || nsfwRaw === "only" ? (nsfwRaw as NsfwMode) : "exclude",
    listed: boolOrNull(sp, "listed"),
    aiTraining: boolOrNull(sp, "ai_training"),
    orientation: multi(sp, "orientation").filter((o) => ORIENT.has(o)) as SearchRequest["orientation"],
    monochrome: boolOrNull(sp, "monochrome"),
    background: multi(sp, "background").filter((c) => c === "transparent" || COLOR_SET.has(c)).slice(0, 10),
    concepts: multi(sp, "concept").slice(0, 10),
    sort: SORTS.has(sortRaw) ? (sortRaw as SortKey) : q ? "relevance" : "newest",
    limit: Math.min(50, Math.max(1, Math.floor(limitRaw ?? 24))),
    cursor: sp.get("cursor"),
    facets: /^(1|true|yes)$/i.test(sp.get("facets") ?? ""),
    semantic: !/^(0|false|no)$/i.test(sp.get("semantic") ?? ""),
    rank: (sp.get("rank") ?? "").toLowerCase() === "rrf" ? "rrf" : "v3",
    expand: !/^(0|false|no)$/i.test(sp.get("expand") ?? ""),
    explain: /^(1|true|yes)$/i.test(sp.get("explain") ?? ""),
    rerank: /^(1|true|yes)$/i.test(sp.get("rerank") ?? ""),
  };
}

/** The keys a JSON search body may carry (POST /search), as query-string parameters. */
const BODY_KEYS = [
  "q", "query", "type", "author", "tag", "color", "has_color", "min_color_weight", "size", "min_colors", "max_colors", "min_width", "max_width", "min_height", "max_height",
  "from", "to", "transparent", "nsfw", "listed", "ai_training", "orientation", "monochrome", "background", "concept", "sort", "limit", "cursor", "facets", "semantic", "rank", "expand", "explain", "rerank",
];

/** A POST /search body ({"query": "red pixel cat", "limit": 20, "rerank": true}) as the same request a GET would make. */
export function searchParamsFromBody(body: Record<string, unknown>): URLSearchParams {
  const sp = new URLSearchParams();
  for (const k of BODY_KEYS) {
    const v = body[k];
    if (v === undefined || v === null) continue;
    const key = k === "query" ? "q" : k;
    if (Array.isArray(v)) for (const x of v.slice(0, 20)) sp.append(key, String(x));
    else if (typeof v === "boolean") sp.set(key, v ? "1" : "0");
    else if (typeof v === "string" || typeof v === "number") sp.set(key, String(v));
  }
  return sp;
}

/** A request with every filter at its default (for internal callers such as /ask). */
export function emptyRequest(over: Partial<SearchRequest> = {}): SearchRequest {
  return { ...parseSearchRequest(new URLSearchParams()), ...over };
}

/** Is a post shown under the nsfw setting? The same test as the SQL filter (sql.ts). */
export function visibleUnder(row: { nsfw?: unknown; ai_nsfw?: unknown }, nsfw: NsfwMode): boolean {
  const flagged = Number(row.nsfw ?? 0) === 1 || Number(row.ai_nsfw ?? 0) >= 0.7;
  return nsfw === "include" ? true : nsfw === "only" ? flagged : !flagged;
}

/** True when the request narrows the corpus beyond type/nsfw (used to size candidate retrieval). */
export function hasRestrictiveFilters(r: SearchRequest): boolean {
  return !!(
    r.authors.length || r.tags.length || r.colors.length || r.hasColors.length || r.sizes.length || r.minColors !== null || r.maxColors !== null ||
    r.minWidth !== null || r.maxWidth !== null || r.minHeight !== null || r.maxHeight !== null || r.from !== null || r.to !== null ||
    r.transparent !== null || r.listed !== null || r.aiTraining !== null || r.orientation.length || r.monochrome !== null || r.background.length ||
    r.concepts.length || r.nsfw === "only"
  );
}

/** Stable key for caching a request (excludes cursor and presentation-only flags). */
export function requestKey(r: SearchRequest): string {
  const { cursor: _c, explain: _e, facets: _f, limit: _l, ...rest } = r;
  return JSON.stringify(rest);
}
