// Deterministic operators (spec §26): first, latest, count, count by author, top, search, similar —
// v3's /ask answers, unchanged — and v4's history, comparison, sequence, duration, aggregate, group,
// title resolution and identification. Each works on a Scope (search/retrieval.ts) or on another
// operator's result and returns an exact OperatorResult. No model is ever involved.

import type { Env } from "../../env";
import { fold, type Lang } from "../../lib/text";
import { emptyRequest, visibleUnder, type NsfwMode } from "../params";
import type { SearchContext } from "../context";
import { qualityScore } from "../ranker";
import { buildFilter } from "../sql";
import { hydrateRows } from "../service";
import { duplicates, similar } from "../visual";
import { metadataRows, timeKey } from "../retrieval";
import {
  answerLang,
  capitalize,
  compareAnswer,
  firstLastText,
  durationIn,
  fmtDate,
  fmtDuration,
  metricUnit,
  metricValue,
  metricWord,
  quote,
  say,
  yesNo,
} from "../answer-text";
import type { OperatorResult, Row, Scope, Verified } from "./types";

export * from "./types";

/** What every operator needs besides its input. */
export interface OpContext {
  env: Env;
  ctx: SearchContext;
  lang: Lang;
  /** v3 wording: English, French and German only, "votes" in every language */
  v3: boolean;
  limit: number;
  /** what the answer calls the subject (answer-text.ts subjectLabel) */
  subject: string;
}

const round = (x: number) => Math.round(x * 1000) / 1000;
/** A sentence continued after "Yes: ": its first letter in lower case, unless it opens with a name or a quote. */
export const afterColon = (s: string) => (/^[“„«"@\d]/.test(s) ? s : s.charAt(0).toLowerCase() + s.slice(1));
const path = (r: Record<string, any>) => `/@${r.author}/${r.permlink}`;

/** v3's confidence of an answer resting on one row: its verification, the plan's confidence, inferred history. */
export function rowConfidence(x: Verified, planConfidence: number): number {
  return round(Math.min(1, x.v.score * Math.min(1, planConfidence + 0.1) * (x.row.history_exact === 0 ? 0.9 : 1)));
}

const metaVerified = (rows: Row[]): Verified[] => rows.map((row) => ({ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["matches the filters"] } }));

// ---- v3: first / latest -----------------------------------------------------------------------------

/**
 * "The first X" follows an image to its first appearance anywhere (a repost counts from the
 * original, a deleted original included). A question that names an author or a period is about
 * those posts themselves ("when did @bob first post a cat?"): their own dates.
 */
export function findFirstLast(scope: Scope, verified: Verified[], oc: OpContext, first: boolean): OperatorResult {
  const plan = scope.plan;
  const constrained = !!(plan.filters.authors?.length || plan.filters.from !== undefined || plan.filters.to !== undefined);
  const own = (x: Verified) => x.row.image_since ?? x.row.created;
  const key = (x: Verified) => (first && !constrained ? timeKey(x.row) : own(x));
  const sorted = [...verified].sort((x, y) => (first ? key(x) - key(y) : key(y) - key(x)) || x.row.id - y.row.id);
  const top = sorted[0];
  const firstAuthor = first && !constrained && top.row.first_seen_author ? top.row.first_seen_author : top.row.author;
  const x = { subject: oc.subject, author: firstAuthor, date: fmtDate(key(top)), title: top.row.title };
  let text: string;
  let answer: string;
  let answerType: OperatorResult["answerType"];
  if (plan.output === "date") {
    answer = fmtDate(key(top));
    answerType = "date";
    text = firstLastText("date", first, oc.lang, x, oc.v3);
  } else if (plan.output === "author") {
    answer = firstAuthor;
    answerType = "author";
    text = firstLastText("author", first, oc.lang, x, oc.v3);
  } else {
    // "what is the first …": the artwork is the answer, the author is said about it — the one
    // the question named ("… from @retro") or the one who posted it
    answer = path(top.row);
    answerType = "post";
    text = firstLastText(plan.filters.authors?.length ? "post_by" : "post", first, oc.lang, x, oc.v3);
  }
  const reposted = first && top.row.first_seen_author && (top.row.first_seen_author !== top.row.author || top.row.first_seen_permlink !== top.row.permlink);
  if (reposted) {
    text += say(top.row.first_seen_match === "near" ? "repost_near" : "repost", oc.lang, { date: fmtDate(top.row.first_seen), post: `/@${top.row.first_seen_author}/${top.row.first_seen_permlink}` }, oc.v3);
  }
  return {
    op: first ? "find_first" : "find_latest",
    answer,
    answerType,
    text,
    confidence: rowConfidence(top, plan.confidence),
    evidence: [top],
    alternatives: sorted.slice(1, 4),
    items: sorted.slice(0, 4),
    verified: verified.length,
    truncated: scope.truncated,
    details: {
      time: key(top),
      time_field: first && !constrained ? "image_first_seen" : "image_since",
      post: path(top.row),
      author: top.row.author,
      first_author: firstAuthor,
      first_post: reposted ? `/@${top.row.first_seen_author}/${top.row.first_seen_permlink}` : path(top.row),
      title: top.row.title,
      constrained,
    },
    notes: [],
  };
}

// ---- v3: counts ------------------------------------------------------------------------------------

/** Exact counts over the filters (no subject): SQL. */
export async function countMetadata(scope: Scope, oc: OpContext): Promise<OperatorResult> {
  const plan = scope.plan;
  const f = buildFilter(scope.req, { tones: plan.filters.tones });
  const what = plan.countOf === "authors" ? "COUNT(DISTINCT p.author)" : "COUNT(*)";
  const n = (await oc.env.DB.prepare(`SELECT ${what} AS n FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql}`).bind(...f.params).first<{ n: number }>())?.n ?? 0;
  return {
    op: "count",
    answer: n,
    answerType: "count",
    text: say(plan.countOf === "authors" ? "count_authors" : "count", oc.lang, { n, subject: oc.subject }, oc.v3),
    confidence: plan.confidence,
    evidence: [],
    verified: n,
    exact: true,
    details: { n, of: plan.countOf === "authors" ? "authors" : "posts", exact: true },
    notes: [],
  };
}

/** Counts over the verified posts (subject questions). */
export function countVerified(scope: Scope, verified: Verified[], oc: OpContext): OperatorResult {
  const plan = scope.plan;
  const byAuthors = plan.countOf === "authors";
  const n = byAuthors ? new Set(verified.map((x) => x.row.author)).size : verified.length;
  const notes: string[] = [];
  if (scope.truncated) notes.push("vector retrieval hit its cap in some periods: the count is a lower bound");
  const sorted = [...verified].sort((x, y) => y.v.score - x.v.score);
  const mean = sorted.reduce((s, x) => s + x.v.score, 0) / sorted.length;
  return {
    op: "count",
    answer: n,
    answerType: "count",
    text: say(byAuthors ? "count_authors" : "count", oc.lang, { n, subject: oc.subject, atLeast: scope.truncated ? 1 : "" }, oc.v3),
    confidence: round(mean * Math.min(1, plan.confidence + 0.1)),
    evidence: sorted.slice(0, oc.limit),
    items: sorted.slice(0, oc.limit),
    verified: verified.length,
    truncated: scope.truncated,
    details: { n, of: byAuthors ? "authors" : "posts", lower_bound: scope.truncated, ids: sorted.map((x) => x.row.id) },
    notes,
  };
}

/** "Who posted the most …" over the filters (no subject): SQL, ties counted in full. */
export async function countByAuthorMetadata(scope: Scope, oc: OpContext): Promise<OperatorResult> {
  const plan = scope.plan;
  const f = buildFilter(scope.req, { tones: plan.filters.tones });
  const rows =
    (await oc.env.DB.prepare(`SELECT p.author AS author, COUNT(*) AS n FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql} GROUP BY p.author ORDER BY n DESC, p.author LIMIT 50`).bind(...f.params).all<{ author: string; n: number }>()).results ?? [];
  const top = rows[0];
  if (!top) return { op: "count_by_author", answer: null, answerType: "none", text: say("none", oc.lang, { subject: oc.subject }, oc.v3), confidence: 0, evidence: [], counts: [], exact: true, notes: [] };
  // all 50 rows tied: count everyone who shares the first place
  let tiedTotal: number | undefined;
  if (rows.length === 50 && rows[49].n === top.n) {
    tiedTotal = Number(
      (await oc.env.DB.prepare(`SELECT COUNT(*) AS k FROM (SELECT p.author FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql} GROUP BY p.author HAVING COUNT(*) = ?)`).bind(...f.params, top.n).first<{ k: number }>())?.k ?? 50,
    );
  }
  const c = compareAnswer(rows, oc.lang, oc.subject, plan.confidence, tiedTotal, oc.v3);
  const tied = rows.filter((r) => r.n === top.n).map((r) => r.author);
  return { op: "count_by_author", answer: top.author, answerType: "author", text: c.answer_text, confidence: c.confidence, evidence: [], counts: rows.slice(0, 10), exact: true, details: { ranking: rows.slice(0, 10), tied: tied.length > 1 || (tiedTotal ?? 0) > 1 ? tied : [], tied_total: Math.max(tied.length, tiedTotal ?? 0) }, notes: [] };
}

/** "Who posted the most …" over the verified posts. */
export function countByAuthorVerified(scope: Scope, verified: Verified[], oc: OpContext): OperatorResult {
  const plan = scope.plan;
  const counts = new Map<string, number>();
  for (const x of verified) counts.set(x.row.author, (counts.get(x.row.author) ?? 0) + 1);
  const ranked = [...counts.entries()].map(([author, n]) => ({ author, n })).sort((p, q) => q.n - p.n || p.author.localeCompare(q.author));
  const top = ranked[0];
  const notes: string[] = [];
  const tied = ranked.filter((r) => r.n === top.n).map((r) => r.author);
  if (ranked[1] && ranked[1].n === top.n) notes.push(`tie: ${tied.map((a) => "@" + a).join(", ")}`);
  const theirs = verified.filter((x) => x.row.author === top.author).sort((x, y) => y.v.score - x.v.score);
  const c = compareAnswer(ranked, oc.lang, oc.subject, Math.min(1, plan.confidence + 0.1), undefined, oc.v3);
  return {
    op: "count_by_author",
    answer: top.author,
    answerType: "author",
    text: c.answer_text,
    confidence: c.confidence,
    evidence: theirs.slice(0, oc.limit),
    items: theirs.slice(0, oc.limit),
    counts: ranked.slice(0, 10),
    verified: verified.length,
    truncated: scope.truncated,
    details: { ranking: ranked.slice(0, 10), tied: tied.length > 1 ? tied : [], tied_total: tied.length },
    notes,
  };
}

// ---- v3: top, search, nothing found -----------------------------------------------------------------

export function topOp(scope: Scope, verified: Verified[], oc: OpContext): OperatorResult {
  const plan = scope.plan;
  const metric = plan.sort === "payout" ? "payout" : "net_votes";
  const sorted = [...verified].sort((x, y) => (y.row[metric] ?? 0) - (x.row[metric] ?? 0) || y.v.score - x.v.score);
  const top = sorted[0];
  const value = metricValue(metric, metric === "payout" ? Number(top.row.payout) : top.row.net_votes, oc.lang, oc.v3);
  return {
    op: "top",
    answer: path(top.row),
    answerType: "post",
    // v4 names the top artwork of all as such (v3's sentence quotes the word for artwork as a subject)
    text: say(!oc.v3 && !plan.residual && !plan.filters.colors?.length && !plan.filters.tones?.length && plan.object !== "blog" ? "top_any" : "top", oc.lang, { subject: oc.subject, title: top.row.title, author: top.row.author, value, metric: metricWord(metric, oc.lang, oc.v3) }, oc.v3),
    confidence: rowConfidence(top, plan.confidence),
    evidence: [top],
    alternatives: sorted.slice(1, 4),
    items: sorted.slice(0, 4),
    verified: verified.length,
    details: { metric, value: metric === "payout" ? Number(top.row.payout) : top.row.net_votes, post: path(top.row), author: top.row.author, title: top.row.title },
    notes: [],
  };
}

/** Verified items, strongest first, popularity as a tie-breaker. */
export function searchOp(scope: Scope, verified: Verified[], oc: OpContext): OperatorResult {
  const plan = scope.plan;
  const q = (x: Verified) => x.v.score * (0.9 + 0.2 * qualityScore(x.row.net_votes, x.row.created, oc.ctx.now, oc.ctx.stats.voteRate));
  const sorted = [...verified].sort((x, y) => q(y) - q(x));
  const top = sorted[0];
  return {
    op: "search",
    answer: verified.length,
    answerType: "count",
    text: say("post", oc.lang, { title: top.row.title, author: top.row.author, date: fmtDate(top.row.created) }, oc.v3),
    confidence: rowConfidence(top, plan.confidence),
    evidence: sorted.slice(0, oc.limit),
    items: sorted.slice(0, oc.limit),
    verified: verified.length,
    details: { n: verified.length, ids: sorted.slice(0, oc.limit).map((x) => x.row.id) },
    notes: [],
  };
}

/** Nothing verified: "No cat found" (a count says 0), with the closest candidates as alternatives. */
export function noneOp(scope: Scope, oc: OpContext): OperatorResult {
  const plan = scope.plan;
  const closest = [...scope.all].sort((x, y) => y.v.score - x.v.score).slice(0, 3);
  const count = plan.intent === "count";
  return {
    op: count ? "count" : plan.intent === "find_last" ? "find_latest" : plan.intent === "find_first" ? "find_first" : plan.intent === "top" ? "top" : plan.intent === "compare" ? "count_by_author" : "search",
    answer: count ? 0 : null,
    answerType: count ? "count" : "none",
    text: count ? say(plan.countOf === "authors" ? "count_authors" : "count", oc.lang, { n: 0, subject: oc.subject }, oc.v3) : say("none", oc.lang, { subject: oc.subject }, oc.v3),
    confidence: round(1 - (closest[0]?.v.score ?? 0)),
    evidence: [],
    alternatives: closest,
    verified: 0,
    details: { n: 0 },
    notes: [],
  };
}

/** "Similar to 42", "duplicates of 42". */
export async function similarOp(env: Env, id: number | undefined, dup: boolean, oc: OpContext, nsfw: NsfwMode): Promise<OperatorResult & { searchItems?: unknown[] }> {
  if (!id) return { op: dup ? "duplicates" : "similar", answer: null, answerType: "none", text: oc.v3 ? "Name the artwork: “similar to 123” (its id)." : say("name_artwork", oc.lang), confidence: 0, evidence: [], notes: [] };
  const vreq = emptyRequest({ nsfw, sort: "relevance" });
  const r = dup ? await duplicates(env, id, 8, oc.limit, vreq) : await similar(env, id, oc.limit, vreq);
  const text = oc.v3 ? `${r.items.length} ${dup ? "near-duplicate" : "similar"} artworks of #${id}.` : say("similar", oc.lang, { n: r.items.length, id, dup: dup ? 1 : "" });
  return { op: dup ? "duplicates" : "similar", answer: r.items.length, answerType: "count", text, confidence: 1, evidence: [], searchItems: r.items, details: { id, n: r.items.length, items: r.items.map((i: any) => ({ id: i.id, path: i.path, similarity: i.similarity ?? null, distance: i.distance ?? null })) }, notes: [] };
}

/**
 * v3's answer to a one-step question: the intent applied to the scope. Metadata scopes count and
 * rank in SQL; subject scopes work on the verified candidates.
 */
export async function runIntent(scope: Scope, oc: OpContext): Promise<OperatorResult> {
  const plan = scope.plan;
  let verified: Verified[];
  if (scope.kind === "metadata") {
    if (plan.intent === "count") return countMetadata(scope, oc);
    if (plan.intent === "compare") return countByAuthorMetadata(scope, oc);
    const rows = await metadataRows(oc.env, plan, scope.req, oc.limit);
    // nothing matches the filters ("the first #unicorn", "latest @nobody")
    if (!rows.length) return { op: plan.intent === "find_first" ? "find_first" : plan.intent === "find_last" ? "find_latest" : plan.intent === "top" ? "top" : "search", answer: null, answerType: "none", text: say("none", oc.lang, { subject: oc.subject }, oc.v3), confidence: 0, evidence: [], verified: 0, exact: true, notes: [] };
    verified = metaVerified(rows);
  } else {
    verified = scope.verified;
    if (!verified.length) return noneOp(scope, oc);
  }
  let r: OperatorResult;
  switch (plan.intent) {
    case "find_first":
      r = findFirstLast(scope, verified, oc, true);
      break;
    case "find_last":
      r = findFirstLast(scope, verified, oc, false);
      break;
    case "count":
      r = countVerified(scope, verified, oc);
      break;
    case "top":
      r = topOp(scope, verified, oc);
      break;
    case "compare":
      r = countByAuthorVerified(scope, verified, oc);
      break;
    default:
      r = searchOp(scope, verified, oc);
  }
  if (scope.kind === "metadata") r.exact = true;
  return r;
}

// ---- v4: history -------------------------------------------------------------------------------------

export interface HistoryEvent {
  kind: "create" | "edit" | "delete";
  at: number;
  title: string | null;
  /** what the version showed: this artwork's image, another image, or none (text, patch, deleted) */
  image: "same" | "other" | "none";
  source: string;
}

export interface Appearance {
  author: string;
  permlink: string;
  /** when this post first showed the image, and until when (null: it still does) */
  from: number;
  to: number | null;
  match: "exact" | "near";
  deleted: boolean;
}

export interface HistoryFacts {
  post: string;
  author: string;
  title: string;
  created: number;
  deleted: boolean;
  deleted_at: number | null;
  events: HistoryEvent[];
  edits: number;
  last_edit: number | null;
  /** every post that showed the same image (exact bytes) or a near-identical re-upload by the same author */
  appearances: Appearance[];
  first_seen: number | null;
  first_seen_in: string | null;
  /** last time any post showed the image (null while one still does) */
  last_seen: number | null;
  still_shown_by: string | null;
  exact: boolean;
  /** posts with this image the nsfw filter hides: left out of everything above, as /search leaves them out */
  hidden: number;
}

/**
 * Edit history, deletion and every appearance of a post's image (post_versions, deleted posts
 * included). Posts the nsfw setting hides are left out, as /search leaves them out, and counted.
 */
export async function historyFacts(env: Env, target: { author: string; permlink: string }, opts: { nsfw?: NsfwMode } = {}): Promise<HistoryFacts | null> {
  const nsfw: NsfwMode = opts.nsfw ?? "include";
  const post = await env.DB
    .prepare("SELECT p.id, p.author, p.permlink, p.title, p.created, p.deleted, a.content_hash, a.phash, a.buckets_json, a.first_seen, a.first_seen_author, a.first_seen_permlink, a.first_seen_match, a.history_exact FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ?")
    .bind(target.author, target.permlink)
    .first<Row>();
  if (!post) return null;
  const versions =
    (await env.DB.prepare("SELECT at, kind, body_kind, content_hash, title, source FROM post_versions WHERE author = ? AND permlink = ? ORDER BY at, block_num, op_in_trx").bind(post.author, post.permlink).all<Row>()).results ?? [];
  // the image this post shows (or showed last, when it was deleted)
  const hash: string | null = post.content_hash ?? [...versions].reverse().find((v) => v.content_hash)?.content_hash ?? null;
  const exactRows = versions.filter((v) => v.source !== "snapshot");
  const events: HistoryEvent[] = (exactRows.length ? exactRows : versions).map((v) => ({
    kind: v.kind,
    at: v.at,
    title: v.title ?? null,
    image: v.body_kind === "image" ? (hash && v.content_hash === hash ? "same" : "other") : "none",
    source: v.source,
  }));
  const deletedAt = post.deleted ? (events.filter((e) => e.kind === "delete").at(-1)?.at ?? null) : null;
  const edits = events.filter((e) => e.kind === "edit");

  // every post whose versions showed the same bytes, with the span it showed them; for a near
  // re-upload, the bytes of the post it came from as well (others may have copied those)
  const appearances: Appearance[] = [];
  const hiddenPosts = new Set<string>();
  const shownUnder = async (author: string, permlink: string) => {
    if (nsfw === "include") return true;
    const r = await env.DB.prepare("SELECT p.nsfw, a.ai_nsfw FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ?").bind(author, permlink).first<Row>().catch(() => null);
    const ok = !r || visibleUnder(r, nsfw);
    if (!ok) hiddenPosts.add(`${author}/${permlink}`);
    return ok;
  };
  const hashes = new Set<string>(hash ? [hash] : []);
  if (post.first_seen_match === "near" && post.first_seen_author) {
    const o = await env.DB.prepare("SELECT content_hash FROM post_versions WHERE author = ? AND permlink = ? AND content_hash IS NOT NULL AND source != 'snapshot' ORDER BY at LIMIT 1").bind(post.first_seen_author, post.first_seen_permlink).first<Row>();
    if (o?.content_hash) hashes.add(o.content_hash);
  }
  if (hashes.size) {
    const hs = [...hashes];
    const refs = (await env.DB.prepare(`SELECT DISTINCT author, permlink FROM post_versions WHERE content_hash IN (${hs.map(() => "?").join(",")}) LIMIT 50`).bind(...hs).all<Row>()).results ?? [];
    const byPost = new Map<string, Row[]>();
    const deletedPosts = new Set<string>();
    if (refs.length) {
      const res = await env.DB.batch(
        refs.flatMap((r) => [
          env.DB.prepare("SELECT author, permlink, at, content_hash, body_kind, source FROM post_versions WHERE author = ? AND permlink = ? ORDER BY at, block_num, op_in_trx").bind(r.author, r.permlink),
          env.DB.prepare("SELECT p.deleted, p.nsfw, a.ai_nsfw FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ?").bind(r.author, r.permlink),
        ]),
      );
      refs.forEach((r, i) => {
        const state = res[2 * i + 1].results?.[0] as Row | undefined;
        const ref = `${r.author}/${r.permlink}`;
        if (state && !visibleUnder(state, nsfw) && ref !== `${post.author}/${post.permlink}`) return void hiddenPosts.add(ref);
        byPost.set(ref, (res[2 * i].results ?? []) as Row[]);
        if (state?.deleted === 1) deletedPosts.add(ref);
      });
    }
    for (const [ref, vs] of byPost) {
      const [author, permlink] = [ref.slice(0, ref.indexOf("/")), ref.slice(ref.indexOf("/") + 1)];
      const exact = vs.filter((v) => v.source !== "snapshot");
      const rows = exact.length ? exact : vs;
      let from: number | null = null;
      let to: number | null = null;
      let shown: string | null = null;
      for (const v of rows) {
        if (v.content_hash && hashes.has(v.content_hash) && from === null) (from = v.at), (shown = v.content_hash);
        else if (from !== null && v.content_hash !== shown && to === null) to = v.at;
      }
      if (from !== null) appearances.push({ author, permlink, from, to: to ?? (deletedPosts.has(ref) ? (rows.at(-1)?.at ?? null) : null), match: shown === hash || !hash ? "exact" : "near", deleted: deletedPosts.has(ref) });
    }
    // a near-identical re-upload by the same author (artworks.first_seen_match = near links them)
    const near = (await env.DB.prepare("SELECT p.author, p.permlink, p.deleted, p.nsfw, a.ai_nsfw, COALESCE(a.image_since, p.created) AS since FROM artworks a JOIN posts p ON p.id = a.post_id WHERE a.first_seen_author = ? AND a.first_seen_permlink = ? AND a.first_seen_match = 'near'").bind(post.first_seen_author ?? post.author, post.first_seen_permlink ?? post.permlink).all<Row>()).results ?? [];
    for (const n of near) {
      if (appearances.some((a) => a.author === n.author && a.permlink === n.permlink)) continue;
      if (!visibleUnder(n, nsfw)) {
        hiddenPosts.add(`${n.author}/${n.permlink}`);
        continue;
      }
      appearances.push({ author: n.author, permlink: n.permlink, from: n.since, to: null, match: "near", deleted: n.deleted === 1 });
    }
    // the post a near re-upload came from
    if (post.first_seen_author && !appearances.some((a) => a.author === post.first_seen_author && a.permlink === post.first_seen_permlink) && !hiddenPosts.has(`${post.first_seen_author}/${post.first_seen_permlink}`) && (await shownUnder(post.first_seen_author, post.first_seen_permlink))) {
      const origin = await env.DB.prepare("SELECT MIN(at) AS at, MAX(at) AS last FROM post_versions WHERE author = ? AND permlink = ?").bind(post.first_seen_author, post.first_seen_permlink).first<Row>();
      const gone = await env.DB.prepare("SELECT deleted FROM posts WHERE author = ? AND permlink = ?").bind(post.first_seen_author, post.first_seen_permlink).first<Row>();
      if (origin?.at) appearances.push({ author: post.first_seen_author, permlink: post.first_seen_permlink, from: origin.at, to: gone?.deleted ? origin.last : null, match: "near", deleted: gone?.deleted === 1 });
    }
  }
  appearances.sort((a, b) => a.from - b.from || a.author.localeCompare(b.author));
  const live = appearances.filter((a) => a.to === null && !a.deleted);
  const ended = appearances.filter((a) => a.to !== null);
  // the first sighting the index recorded, unless the filter hides the post it was in: then the first visible one
  const firstHidden = !!post.first_seen_author && (hiddenPosts.has(`${post.first_seen_author}/${post.first_seen_permlink}`) || (nsfw !== "include" && !(await shownUnder(post.first_seen_author, post.first_seen_permlink))));
  return {
    post: path(post as any),
    author: post.author,
    title: post.title,
    created: post.created,
    deleted: post.deleted === 1,
    deleted_at: deletedAt,
    events,
    edits: edits.length,
    last_edit: edits.at(-1)?.at ?? null,
    appearances,
    first_seen: firstHidden ? (appearances[0]?.from ?? null) : (post.first_seen ?? appearances[0]?.from ?? null),
    first_seen_in: post.first_seen_author && !firstHidden ? `/@${post.first_seen_author}/${post.first_seen_permlink}` : appearances[0] ? path(appearances[0]) : null,
    last_seen: live.length ? null : ended.length ? Math.max(...ended.map((a) => a.to!)) : null,
    still_shown_by: live.length ? path(live.at(-1)!) : null,
    exact: post.history_exact === null || post.history_exact === undefined ? exactRows.length > 0 : post.history_exact === 1,
    hidden: hiddenPosts.size,
  };
}

export type HistoryRelation = "repost" | "edit" | "delete" | "first_seen" | "last_seen" | "all";

/**
 * A history question about one post. "repost": was its image posted again later — by the same
 * author (the question's "they"), else by anyone; "edit": how many edits; "delete": was it deleted;
 * "first_seen" / "last_seen": when its image appeared first / was shown last.
 */
export function historyOp(h: HistoryFacts, relation: HistoryRelation, oc: OpContext, opts: { byAuthor?: string; when?: boolean } = {}): OperatorResult {
  const r = historyResult(h, relation, oc, opts);
  // posts the nsfw filter hides are left out of the answer; it says so where they would count
  if (h.hidden > 0 && relation !== "edit" && relation !== "delete") {
    const l = answerLang(oc.lang);
    return { ...r, text: `${r.text.replace(/\s+$/, "")}${say("hidden_posts", l, { n: h.hidden })}`, notes: [...r.notes, `${h.hidden} post(s) with this image are hidden by the nsfw filter`] };
  }
  return r;
}

function historyResult(h: HistoryFacts, relation: HistoryRelation, oc: OpContext, opts: { byAuthor?: string; when?: boolean }): OperatorResult {
  const l = answerLang(oc.lang);
  const titleQ = quote(h.title || h.post, l);
  const base = { op: "history" as const, evidence: [] as Verified[], confidence: h.exact ? 0.95 : 0.8, notes: h.exact ? [] : ["the history of this post is inferred from its dates, not read from chain operations"], details: { ...h } as Record<string, unknown> };
  switch (relation) {
    case "repost": {
      // posts that showed the image after this one did (this post itself, and what came before it, are not reposts of it)
      const own = h.appearances.find((a) => path(a) === h.post)?.from ?? h.first_seen ?? h.created;
      const after = h.appearances.filter((a) => a.from > own + 30 && path(a) !== h.post && path(a) !== (h.first_seen_in ?? h.post));
      // "did they post it again": by that author; "was it reposted": by anyone
      const by = opts.byAuthor;
      const mine = by ? after.filter((a) => a.author === by) : after;
      const pick = mine[0] ?? null;
      if (pick) return { ...base, answer: true, answerType: "boolean", text: capitalize(yesNo(true, l) + say("repost_yes", l, { date: fmtDate(pick.from), post: path(pick), near: pick.match === "near" ? 1 : "" })), details: { ...base.details, relation, reposts: after } };
      if (by && after.length) {
        const o = after[0];
        return { ...base, answer: false, answerType: "boolean", text: yesNo(false, l) + say("repost_by_other", l, { author: by, other: o.author, date: fmtDate(o.from), post: path(o) }), details: { ...base.details, relation, reposts: after } };
      }
      return { ...base, answer: false, answerType: "boolean", text: capitalize(yesNo(false, l) + say("repost_no", l)), details: { ...base.details, relation, reposts: [] } };
    }
    case "edit":
      // "was it edited by @x?" when @x is not its author: only the author edits a post on the chain
      if (opts.byAuthor && opts.byAuthor !== h.author) return { ...base, answer: 0, answerType: "count", text: yesNo(false, l) + say("edit_by_other", l, { title: titleQ, author: h.author, actor: opts.byAuthor }), details: { ...base.details, relation, actor: opts.byAuthor } };
      // "when was it edited?": the date of the last edit; "was it edited?": how many times
      return h.edits
        ? { ...base, answer: opts.when && h.last_edit ? fmtDate(h.last_edit) : h.edits, answerType: opts.when && h.last_edit ? "date" : "count", text: say("edited", l, { title: titleQ, n: h.edits, date: fmtDate(h.last_edit!) }), details: { ...base.details, relation } }
        : { ...base, answer: 0, answerType: "count", text: say("not_edited", l, { title: titleQ, date: fmtDate(h.created) }), details: { ...base.details, relation } };
    case "delete":
      if (opts.byAuthor && opts.byAuthor !== h.author) return { ...base, answer: false, answerType: "boolean", text: yesNo(false, l) + say("delete_by_other", l, { title: titleQ, author: h.author, actor: opts.byAuthor }), details: { ...base.details, relation, actor: opts.byAuthor } };
      // "when was it deleted?": the date; "was it deleted?": yes or no
      return h.deleted
        ? { ...base, answer: opts.when && h.deleted_at ? fmtDate(h.deleted_at) : true, answerType: opts.when && h.deleted_at ? "date" : "boolean", text: say("deleted", l, { title: titleQ, author: h.author, date: h.deleted_at ? fmtDate(h.deleted_at) : "?" }), details: { ...base.details, relation } }
        : { ...base, answer: false, answerType: "boolean", text: say("not_deleted", l, { title: titleQ }), details: { ...base.details, relation } };
    case "first_seen":
      return { ...base, answer: h.first_seen ? fmtDate(h.first_seen) : null, answerType: h.first_seen ? "date" : "none", text: h.first_seen ? say("first_seen", l, { date: fmtDate(h.first_seen), post: h.first_seen_in ?? h.post }) : say("insufficient", l), details: { ...base.details, relation } };
    case "last_seen":
      return h.still_shown_by
        ? { ...base, answer: null, answerType: "none", text: say("still_shown", l, { post: h.still_shown_by }), details: { ...base.details, relation } }
        : { ...base, answer: h.last_seen ? fmtDate(h.last_seen) : null, answerType: h.last_seen ? "date" : "none", text: h.last_seen ? say("last_seen", l, { date: fmtDate(h.last_seen), post: path(h.appearances.at(-1)!) }) : say("insufficient", l), details: { ...base.details, relation } };
    default: {
      const parts = [h.edits ? say("edited", l, { title: titleQ, n: h.edits, date: fmtDate(h.last_edit!) }) : say("not_edited", l, { title: titleQ, date: fmtDate(h.created) })];
      if (h.deleted) parts.push(say("deleted", l, { title: titleQ, author: h.author, date: h.deleted_at ? fmtDate(h.deleted_at) : "?" }));
      if (h.first_seen_in && h.first_seen_in !== h.post && h.first_seen) parts.push(say("first_seen", l, { date: fmtDate(h.first_seen), post: h.first_seen_in }));
      return { ...base, answer: h.edits, answerType: "count", text: parts.join(" "), details: { ...base.details, relation } };
    }
  }
}

// ---- v4: comparisons, sequence, duration ------------------------------------------------------------------

/** Two counts side by side: "Did @a post more cats than @b?", "Who has more artworks, @a or @b?". */
export function compareCounts(a: { author: string; n: number }, b: { author: string; n: number }, oc: OpContext, opts: { subject: string; want?: "more" | "less"; yesNo?: boolean; claimed?: string }): OperatorResult {
  const l = answerLang(oc.lang);
  const tie = a.n === b.n;
  const [w, lo] = a.n >= b.n ? [a, b] : [b, a];
  let text = tie ? say("same_count", l, { a: a.author, b: b.author, s: opts.subject, n: a.n }) : say("more", l, { w: w.author, l: lo.author, s: opts.subject, nw: w.n, nl: lo.n });
  let answer: string | boolean | null = tie ? null : opts.want === "less" ? lo.author : w.author;
  let answerType: OperatorResult["answerType"] = tie ? "authors" : "author";
  if (opts.yesNo) {
    // "Did @a post more than @b?": the claimed one is a
    const claimed = opts.claimed ?? a.author;
    const yes = !tie && (opts.want === "less" ? lo.author === claimed : w.author === claimed);
    answer = yes;
    answerType = "boolean";
    text = yesNo(yes, l) + text;
  }
  return { op: "compare", answer, answerType, text, confidence: 0.95, evidence: [], details: { a, b, tie, winner: tie ? null : w.author }, notes: [] };
}

/** Two authors' totals of a metric over their artworks: "Does @a have more votes than @b?". */
export function compareTotals(a: { author: string; v: number }, b: { author: string; v: number }, metric: "net_votes" | "payout", oc: OpContext, opts: { want?: "more" | "less"; yesNo?: boolean; claimed?: string }): OperatorResult {
  const l = answerLang(oc.lang);
  const unit = metricUnit(metric, l);
  const fmt = (v: number) => (metric === "payout" ? v.toFixed(3) : String(v));
  const tie = a.v === b.v;
  const [w, lo] = a.v >= b.v ? [a, b] : [b, a];
  let text = tie ? say("same_total", l, { a: a.author, b: b.author, unit, v: fmt(a.v) }) : say("more_total", l, { w: w.author, l: lo.author, unit, vw: fmt(w.v), vl: fmt(lo.v) });
  let answer: string | boolean | null = tie ? null : opts.want === "less" ? lo.author : w.author;
  let answerType: OperatorResult["answerType"] = tie ? "authors" : "author";
  if (opts.yesNo) {
    const claimed = opts.claimed ?? a.author;
    const yes = !tie && (opts.want === "less" ? lo.author === claimed : w.author === claimed);
    answer = yes;
    answerType = "boolean";
    text = yesNo(yes, l) + text;
  }
  return { op: "compare", answer, answerType, text, confidence: 0.95, evidence: [], details: { a, b, metric, tie, winner: tie ? null : w.author }, notes: [] };
}

/** Two posts by a metric: "Which has more votes, Swan or Sushi?". */
export function compareMetric(a: { label: string; value: number; post: string }, b: { label: string; value: number; post: string }, metric: "net_votes" | "payout", oc: OpContext, opts: { yesNo?: boolean } = {}): OperatorResult {
  const l = answerLang(oc.lang);
  const unit = metricUnit(metric, l);
  const fmt = (v: number) => (metric === "payout" ? v.toFixed(3) : String(v));
  const tie = a.value === b.value;
  const [w, lo] = a.value >= b.value ? [a, b] : [b, a];
  let text = tie ? say("same_metric", l, { a: a.label, b: b.label, unit, v: fmt(a.value) }) : say("more_metric", l, { w: w.label, l: lo.label, unit, vw: fmt(w.value), vl: fmt(lo.value) });
  let answer: string | boolean | null = tie ? null : w.post;
  if (opts.yesNo) {
    answer = !tie && w.post === a.post;
    text = yesNo(answer, l) + text;
  }
  return { op: "compare", answer, answerType: opts.yesNo ? "boolean" : tie ? "none" : "post", text, confidence: 0.95, evidence: [], details: { a, b, metric, tie }, notes: [] };
}

export interface TimedEvent {
  label: string;
  at: number;
  post?: string;
}

/** Which came first; "was A before B?" answers yes or no. */
export function sequenceOp(a: TimedEvent, b: TimedEvent, oc: OpContext, opts: { yesNo?: "before" | "after" } = {}): OperatorResult {
  const l = answerLang(oc.lang);
  const same = Math.abs(a.at - b.at) < 60;
  const [e1, e2] = a.at <= b.at ? [a, b] : [b, a];
  let text = same ? say("same_time", l, { a: a.label, b: b.label, d: fmtDate(a.at) }) : say("before", l, { a: e1.label, b: e2.label, da: fmtDate(e1.at), db: fmtDate(e2.at) });
  text = capitalize(text);
  let answer: string | boolean | null = same ? null : (e1.post ?? e1.label);
  let answerType: OperatorResult["answerType"] = same ? "none" : e1.post ? "post" : "value";
  if (opts.yesNo) {
    const yes = !same && (opts.yesNo === "before" ? a.at < b.at : a.at > b.at);
    answer = yes;
    answerType = "boolean";
    text = yesNo(yes, l) + afterColon(text);
  }
  return { op: "sequence", answer, answerType, text, confidence: 0.95, evidence: [], details: { a, b, first: same ? null : e1.label }, notes: [] };
}

/** Time between two events. */
export function durationOp(a: TimedEvent, b: TimedEvent, oc: OpContext, opts: { unit?: "hours" | "days" | "weeks" } = {}): OperatorResult {
  const l = answerLang(oc.lang);
  const seconds = Math.abs(b.at - a.at);
  const [e1, e2] = a.at <= b.at ? [a, b] : [b, a];
  // in the unit the question asks for ("how many hours …"), else days from two days on, hours below
  const auto = Math.round(seconds / 86400) >= 2 ? "days" : "hours";
  const asked = opts.unit ? durationIn(seconds, opts.unit, l) : null;
  const dur = asked ? asked.text : fmtDuration(seconds, l);
  return {
    op: "duration",
    answer: asked ? asked.n : auto === "days" ? Math.round(seconds / 86400) : Math.round(seconds / 3600),
    answerType: "duration",
    text: capitalize(say("duration", l, { dur, a: e1.label, b: e2.label, da: fmtDate(e1.at), db: fmtDate(e2.at) })),
    confidence: 0.95,
    evidence: [],
    details: { seconds, unit: opts.unit ?? auto, a, b },
    notes: [],
  };
}

// ---- v4: aggregates and groups ------------------------------------------------------------------------------

export type Agg = "sum" | "avg" | "max" | "min";

/** A metric over a set of posts: total votes of @laura's artworks, average payout of cat artworks. */
export function aggregateOp(rows: Verified[], metric: "net_votes" | "payout", agg: Agg, oc: OpContext, scopeLabel: string, totals?: { value: number; n: number }): OperatorResult {
  const l = answerLang(oc.lang);
  const vals = rows.map((x) => Number(x.row[metric] ?? 0));
  // totals computed in SQL (questions without a subject), else over the verified posts
  const n = totals ? totals.n : vals.length;
  const v = totals ? totals.value : !n ? 0 : agg === "sum" ? vals.reduce((s, x) => s + x, 0) : agg === "avg" ? vals.reduce((s, x) => s + x, 0) / n : agg === "max" ? Math.max(...vals) : Math.min(...vals);
  const shown = metric === "payout" ? v.toFixed(3) : agg === "avg" ? (Math.round(v * 10) / 10).toString() : String(v);
  const aggWord: Record<string, Record<Agg, string>> = {
    en: { sum: "in total", avg: "on average", max: "at most", min: "at least" },
    fr: { sum: "au total", avg: "en moyenne", max: "au plus", min: "au moins" },
    de: { sum: "insgesamt", avg: "im Durchschnitt", max: "höchstens", min: "mindestens" },
    es: { sum: "en total", avg: "de media", max: "como máximo", min: "como mínimo" },
    it: { sum: "in totale", avg: "in media", max: "al massimo", min: "almeno" },
  };
  return {
    op: "aggregate",
    answer: metric === "payout" ? Number(v.toFixed(3)) : agg === "avg" ? Math.round(v * 10) / 10 : v,
    answerType: "value",
    text: say("aggregate", l, { scope: scopeLabel, v: shown, unit: metricUnit(metric, l), agg: aggWord[l][agg], n }),
    confidence: n ? (totals ? 0.98 : 0.95) : 0.5,
    evidence: [...rows].sort((x, y) => Number(y.row[metric] ?? 0) - Number(x.row[metric] ?? 0)).slice(0, oc.limit),
    ...(totals ? { exact: true } : {}),
    details: { metric, agg, value: v, n },
    notes: [],
  };
}

export type GroupBy = "month" | "author" | "tag";

/** The sentence of a grouping: the month with the most posts, the author with the most, or what a set of posts is mostly about. */
export function groupResult(rows: Array<{ k: string; n: number }>, by: GroupBy, oc: OpContext, scopeLabel: string, total: number, exact: boolean, metric?: "net_votes" | "payout"): OperatorResult {
  const l = answerLang(oc.lang);
  if (!rows.length) return { op: "group", answer: null, answerType: "none", text: say("none", l, { subject: oc.subject }), confidence: 0, evidence: [], exact, notes: [] };
  const top = rows[0];
  const tie = rows[1]?.n === top.n;
  const text =
    by === "tag"
      ? say("group_tags", l, { scope: scopeLabel, list: rows.slice(0, 4).map((r) => `${r.k} (${r.n})`).join(", ") })
      : by === "author" && metric
        ? say("group_author_total", l, { key: top.k, unit: metricUnit(metric, l), n: metric === "payout" ? Number(top.n).toFixed(3) : top.n })
        : by === "author"
          ? say("group_author", l, { key: `@${top.k}`, scope: scopeLabel, n: top.n })
          : say("group_month", l, { key: top.k, scope: scopeLabel, n: top.n, total });
  return {
    op: "group",
    answer: by === "tag" ? rows.slice(0, 4).map((r) => r.k).join(", ") : top.k,
    answerType: by === "author" ? "author" : by === "tag" ? "list" : "value",
    text,
    confidence: tie && by !== "tag" ? 0.5 : 0.95,
    evidence: [],
    exact,
    details: { by, groups: rows.slice(0, 24), total, ...(metric ? { metric } : {}) },
    notes: tie && by !== "tag" ? [`tie between ${rows.filter((r) => r.n === top.n).map((r) => r.k).join(", ")}`] : [],
  };
}

/** Posts per month, per author or per tag: "Which month had the most artworks?", "What kind of art does @a make?". */
export async function groupMetadata(scope: Scope, by: GroupBy, oc: OpContext, scopeLabel = "", metric?: "net_votes" | "payout"): Promise<OperatorResult> {
  const f = buildFilter(scope.req, { tones: scope.plan.filters.tones });
  // per author, a number of posts, or (v4) the total of a metric over them
  const value = metric ? `SUM(${metric === "payout" ? "p.payout" : "p.net_votes"})` : "COUNT(*)";
  const sql =
    by === "tag"
      ? `SELECT t.tag AS k, COUNT(DISTINCT p.id) AS n FROM post_tags t JOIN posts p ON p.id = t.post_id LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql} GROUP BY t.tag ORDER BY n DESC, k LIMIT 60`
      : `SELECT ${by === "month" ? "strftime('%Y-%m', p.created, 'unixepoch')" : "p.author"} AS k, ${value} AS n FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql} GROUP BY k ORDER BY n DESC, k LIMIT 60`;
  const [rows, total] = await Promise.all([
    oc.env.DB.prepare(sql).bind(...f.params).all<{ k: string; n: number }>(),
    oc.env.DB.prepare(`SELECT COUNT(*) AS n FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql}`).bind(...f.params).first<{ n: number }>(),
  ]);
  return groupResult(rows.results ?? [], by, oc, scopeLabel, total?.n ?? 0, true, metric);
}

// ---- v4: titles and identity --------------------------------------------------------------------------------

export interface TitleMatch {
  rows: Row[];
  status: "found" | "ambiguous" | "not_found";
}

/**
 * Live posts whose title is exactly the given one (case, accents and punctuation aside). With
 * `includeDeleted` (history questions: "when was “T” deleted?"), deleted posts are looked for too
 * when no live post has the title (they have left the full-text index: matched on posts.title).
 */
export async function resolveTitle(env: Env, title: string, nsfw: NsfwMode = "exclude", opts: { includeDeleted?: boolean } = {}): Promise<TitleMatch> {
  const key = (s: string) => fold(s).replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const k = key(title);
  if (!k) return { rows: [], status: "not_found" };
  const words = k.split(" ").filter(Boolean);
  const phrase = words.map((w) => `"${w.replace(/"/g, "")}"`).join(" ");
  const ids = ((await env.DB.prepare("SELECT p.id FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid WHERE posts_fts MATCH ? AND p.deleted = 0 LIMIT 50").bind(`title : ${phrase}`).all<{ id: number }>().catch(() => ({ results: [] as Array<{ id: number }> }))).results ?? []).map((r) => r.id);
  let rows = [...(await hydrateRows(env.DB, ids, emptyRequest({ nsfw }))).values()].filter((r) => key(r.title ?? "") === k).sort((a, b) => a.created - b.created || a.id - b.id);
  if (!rows.length && opts.includeDeleted) {
    const longest = [...words].sort((a, b) => b.length - a.length)[0].replace(/[%_]/g, "");
    const del = ((await env.DB.prepare("SELECT id FROM posts WHERE deleted = 1 AND title LIKE ? LIMIT 200").bind(`%${longest}%`).all<{ id: number }>().catch(() => ({ results: [] as Array<{ id: number }> }))).results ?? []).map((r) => r.id);
    const flagged = (r: Row) => r.nsfw === 1 || Number(r.ai_nsfw ?? 0) >= 0.7;
    rows = [...(await hydrateRows(env.DB, del, null)).values()]
      .filter((r) => key(r.title ?? "") === k && (nsfw === "include" || (nsfw === "only" ? flagged(r) : !flagged(r))))
      .sort((a, b) => a.created - b.created || a.id - b.id);
  }
  return { rows, status: rows.length === 0 ? "not_found" : rows.length === 1 ? "found" : "ambiguous" };
}

export function identifyOp(row: Row, oc: OpContext): OperatorResult {
  const l = answerLang(oc.lang);
  return {
    op: "identify",
    answer: path(row as any),
    answerType: "post",
    text: say("identify", l, { title: row.title, author: row.author, id: row.id, path: path(row as any) }),
    confidence: 1,
    evidence: [{ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["the title names this post"] } }],
    exact: true,
    details: { id: row.id, path: path(row as any), title: row.title, author: row.author, created: row.created },
    notes: [],
  };
}

export function authorOfOp(row: Row, oc: OpContext): OperatorResult {
  const l = answerLang(oc.lang);
  return {
    op: "resolve",
    answer: row.author,
    answerType: "author",
    text: say("author_of", l, { title: row.title, author: row.author, date: fmtDate(row.created) }),
    confidence: 1,
    evidence: [{ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["the title names this post"] } }],
    exact: true,
    details: { id: row.id, path: path(row as any), title: row.title, author: row.author, created: row.created },
    notes: [],
  };
}
