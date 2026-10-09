// Evidence cards and the evidence graph (spec §14-15, §43, §47).
//
// A card is a compact, deterministic, provenance-preserving description of one piece of evidence:
//   E<post id>   an artwork or blog post (deleted posts too, flagged), from the index
//   R<n>         the result of a deterministic operator (step n of the program)
//   D<n>         a documentation excerpt (/help)
//   C<n>         a conflict between pieces of evidence (verifier.ts)
//   I1           the image a visual question was asked about
// Ids are stable: the same index state gives the same cards, in the same order, so answers can be
// reproduced and models compared on exactly the same context (spec §31).
//
// The graph links them — artwork created_by author, artwork shows_image image, image first_seen_in
// post, artwork same_image_as / near_identical_to artwork, result answers entity — so that several
// steps of a question can be followed without the model rebuilding database relationships.

import type { Env } from "../env";
import { parseList } from "./retrieval";
import { hydrateRows } from "./service";
import type { HistoryFacts, OperatorResult, Provenance, Row, Verified } from "./operators";

export const iso = (t: number | null | undefined): string | undefined => (typeof t === "number" && Number.isFinite(t) ? new Date(t * 1000).toISOString().replace(/\.\d{3}Z$/, "Z") : undefined);

export interface ArtworkCard {
  evidence_id: string;
  type: "artwork" | "post";
  source: "pixagram-index";
  artwork_id: number;
  path: string;
  author: string;
  title: string;
  created_at: string;
  /** when this post first showed its current image */
  image_since_at?: string;
  /** when the same image first appeared on chain, in any post (deleted ones included) */
  first_seen_at?: string;
  first_seen_in?: string;
  /** exact: the same bytes; near: a near-identical re-upload by the same author; self: this post */
  first_seen_match?: string;
  /** false: the image's history is inferred from dates, not read from chain operations */
  history_exact?: boolean;
  deleted?: boolean;
  deleted_at?: string;
  tags?: string[];
  concepts?: string[];
  /** the AI description of the image (a model's, not the author's) */
  ai_caption?: string;
  text_in_image?: string;
  colors?: string[];
  tones?: string[];
  votes?: number;
  payout?: number;
  /** how well it matches the question's subject (v3 verification) */
  match?: { score: number; lexical: number; signals: string[] };
  visual_similarity?: number | null;
  text_similarity?: number | null;
  /** cross-encoder score, when the reranker ran */
  rerank_score?: number;
  retrieval?: string[];
  history?: Array<{ kind: string; at: string; image: string; title?: string }>;
  identity?: ImageIdentity;
  /** set by the evidence verifier */
  valid?: boolean;
  issues?: string[];
  /** the image hash, to link cards that show the same image (not shown to the model) */
  image?: string;
}

/** How an artwork relates to a reference image (spec §47): similarity is not identity. */
export interface ImageIdentity {
  /** the same image bytes */
  exact: boolean;
  /** same shapes and colours: pHash within NEAR, colours agreeing */
  perceptual: { phash_distance: number | null; dhash_distance: number | null; near_identical: boolean } | null;
  /** SigLIP image similarity, 0..1 */
  visual: number | null;
  /** both trace back to the same first post */
  historical: "same_origin" | "different_origin" | null;
}

export interface ResultCard {
  evidence_id: string;
  type: "result";
  source: "operator";
  step: string;
  op: string;
  /** the sub-question in words */
  question?: string;
  answer: string | number | boolean | null;
  answer_type: string;
  text: string;
  /** cards it was computed over (the first ones) */
  over?: string[];
  n?: number;
  /** false: a lower bound (the vector budget was spent) */
  complete: boolean;
  /** computed by SQL over the filters, no subject matching involved */
  exact: boolean;
  details?: Record<string, unknown>;
}

export interface DocCard {
  evidence_id: string;
  type: "doc";
  source: "pixagram-docs";
  path: string;
  title: string;
  heading: string;
  url: string;
  text: string;
  score: number;
}

export interface ConflictCard {
  evidence_id: string;
  type: "conflict";
  source: "verifier";
  status: "conflict";
  field: string;
  evidence: string[];
  values: string[];
  note: string;
}

export interface QueryImageCard {
  evidence_id: "I1";
  type: "query_image";
  source: "query-image";
  width: number;
  height: number;
  phash: string;
  sha256: string;
  colors?: string[];
  /** the strongest identity any indexed artwork has with it (image-question.ts) */
  identity?: "exact_identity" | "perceptual_identity" | "visual_similarity" | "none";
  /** what a vision model sees in it (deep modes): a model's description, not the index's */
  description?: { caption: string; subjects?: string[]; tags?: string[]; text_in_image?: string; by: string };
  concepts?: string[];
}

export type EvidenceCard = ArtworkCard | ResultCard | DocCard | ConflictCard | QueryImageCard;

const TONES: Array<[string, (r: Row) => boolean]> = [
  ["greyscale", (r) => r.colorfulness !== null && r.colorfulness !== undefined && r.colorfulness < 10],
  ["monochrome", (r) => r.monochrome === 1],
  ["dark", (r) => typeof r.brightness === "number" && r.brightness < 0.3],
  ["light", (r) => typeof r.brightness === "number" && r.brightness > 0.55],
  ["colorful", (r) => typeof r.colorfulness === "number" && r.colorfulness > 85],
];

const r3 = (x: number) => Math.round(x * 1000) / 1000;

/** The card of a hydrated post row. */
export function artworkCard(row: Row, opts: { v?: Verified["v"]; provenance?: Provenance; concepts?: string[]; rerank?: number; history?: HistoryFacts | null } = {}): ArtworkCard {
  const tags = parseList(row.tags_json).slice(0, 12);
  const buckets = parseList(row.buckets_json) as unknown as Array<{ name: string; weight: number }>;
  const card: ArtworkCard = {
    evidence_id: `E${row.id}`,
    type: row.type === "blog" ? "post" : "artwork",
    source: "pixagram-index",
    artwork_id: row.id,
    path: `/@${row.author}/${row.permlink}`,
    author: row.author,
    title: row.title ?? "",
    created_at: iso(row.created)!,
  };
  if (row.type === "artwork") {
    if (row.image_since !== null && row.image_since !== undefined && Math.abs(row.image_since - row.created) > 30) card.image_since_at = iso(row.image_since);
    if (row.first_seen !== null && row.first_seen !== undefined) {
      card.first_seen_at = iso(row.first_seen);
      if (row.first_seen_author) card.first_seen_in = `/@${row.first_seen_author}/${row.first_seen_permlink}`;
      if (row.first_seen_match) card.first_seen_match = row.first_seen_match;
    }
    if (row.history_exact === 0) card.history_exact = false;
    if (buckets.length) card.colors = buckets.filter((b) => b.weight >= 0.08).sort((a, b) => b.weight - a.weight).slice(0, 4).map((b) => b.name);
    const tones = TONES.filter(([, f]) => f(row)).map(([t]) => t);
    if (tones.length) card.tones = tones;
    if (row.ai_caption) card.ai_caption = String(row.ai_caption).slice(0, 300);
    if (row.ai_text) card.text_in_image = String(row.ai_text).slice(0, 120);
    if (row.content_hash) card.image = String(row.content_hash);
  }
  if (row.deleted === 1) card.deleted = true;
  if (tags.length) card.tags = tags;
  if (opts.concepts?.length) card.concepts = opts.concepts;
  card.votes = row.net_votes ?? 0;
  card.payout = r3(Number(row.payout ?? 0));
  if (opts.v && !opts.v.signals.includes("matches the filters")) {
    card.match = { score: opts.v.score, lexical: opts.v.lexical, signals: opts.v.signals.slice(0, 4) };
    if (opts.v.semantic !== null) card.visual_similarity = opts.v.semantic;
    if (opts.v.text !== null) card.text_similarity = opts.v.text;
  }
  if (opts.provenance?.sources.length) card.retrieval = [...opts.provenance.sources];
  if (typeof opts.rerank === "number") card.rerank_score = r3(opts.rerank);
  if (opts.history) {
    const h = opts.history;
    card.history = h.events.slice(0, 12).map((e) => ({ kind: e.kind, at: iso(e.at)!, image: e.image, ...(e.title && e.title !== row.title ? { title: e.title } : {}) }));
    if (h.deleted_at) card.deleted_at = iso(h.deleted_at);
  }
  return card;
}

/** The card of an operator's result. */
export function resultCard(n: number, step: string, r: OperatorResult, opts: { question?: string; over?: string[] } = {}): ResultCard {
  const d = r.details ?? {};
  const keep: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(d)) {
    if (k === "ids" || k === "items") continue;
    if (typeof v === "number" && (k === "time" || k.endsWith("_at") || k === "first_seen" || k === "last_seen" || k === "created" || k === "deleted_at" || k === "last_edit")) keep[k] = iso(v);
    else if (k === "events" && Array.isArray(v)) keep[k] = v.slice(0, 12).map((e: any) => ({ kind: e.kind, at: iso(e.at), image: e.image }));
    else if ((k === "appearances" || k === "reposts") && Array.isArray(v)) keep[k] = v.slice(0, 8).map((a: any) => ({ post: `/@${a.author}/${a.permlink}`, from: iso(a.from), to: a.to ? iso(a.to) : null, match: a.match, deleted: a.deleted }));
    else if (k === "a" || k === "b") keep[k] = v && typeof v === "object" && "at" in (v as any) ? { ...(v as any), at: iso((v as any).at) } : v;
    else keep[k] = v;
  }
  return {
    evidence_id: `R${n}`,
    type: "result",
    source: "operator",
    step,
    op: r.op,
    ...(opts.question ? { question: opts.question } : {}),
    answer: r.answer,
    answer_type: r.answerType,
    text: r.text,
    ...(opts.over?.length ? { over: opts.over.slice(0, 12) } : {}),
    ...(typeof (d as any).n === "number" ? { n: (d as any).n } : r.verified !== undefined ? { n: r.verified } : {}),
    complete: !r.truncated,
    exact: !!r.exact,
    ...(Object.keys(keep).length ? { details: keep } : {}),
  };
}

export function docCard(n: number, h: { path: string; title: string; heading: string; url: string; text: string; score: number }): DocCard {
  return { evidence_id: `D${n}`, type: "doc", source: "pixagram-docs", path: h.path, title: h.title, heading: h.heading, url: h.url, text: h.text.length > 1500 ? `${h.text.slice(0, 1497)}…` : h.text, score: h.score };
}

/** Cards for the model (spec §43): only what it may use, without internal fields. */
export function modelView(c: EvidenceCard): Record<string, unknown> {
  if (c.type === "artwork" || c.type === "post") {
    const { image: _i, valid: _v, issues, source: _s, ...rest } = c;
    return issues?.length ? { ...rest, issues } : rest;
  }
  if (c.type === "result") {
    const { source: _s, ...rest } = c;
    return rest;
  }
  if (c.type === "doc") {
    const { source: _s, score: _sc, path: _p, ...rest } = c;
    return rest;
  }
  const { source: _s, ...rest } = c as any;
  return rest;
}

// ---- the evidence graph -------------------------------------------------------------------------

export interface GraphEntity {
  id: string;
  type: "artwork" | "post" | "author" | "image" | "concept" | "result" | "query_image";
  label?: string;
  evidence?: string;
}
export interface GraphEvent {
  type: "post_created" | "image_first_seen" | "image_last_seen" | "edited" | "deleted" | "reposted";
  subject: string;
  timestamp: number;
  at: string;
  evidence?: string;
  where?: string;
}
export interface GraphRelation {
  type: "created_by" | "shows_image" | "same_image_as" | "near_identical_to" | "visually_similar_to" | "first_seen_in" | "depicts" | "answers";
  subject: string;
  object: string;
  evidence?: string;
}
export interface EvidenceGraph {
  entities: GraphEntity[];
  events: GraphEvent[];
  relations: GraphRelation[];
}

const ts = (s?: string) => (s ? Math.floor(Date.parse(s) / 1000) : NaN);

/** The graph of a set of cards and the history facts behind them. */
export function buildGraph(cards: EvidenceCard[], history: Map<string, HistoryFacts>): EvidenceGraph {
  const entities = new Map<string, GraphEntity>();
  const events: GraphEvent[] = [];
  const relations: GraphRelation[] = [];
  const rel = new Set<string>();
  const ent = (e: GraphEntity) => (entities.has(e.id) ? entities.get(e.id)! : (entities.set(e.id, e), e));
  const link = (r: GraphRelation) => {
    const k = `${r.type}|${r.subject}|${r.object}`;
    if (!rel.has(k)) (rel.add(k), relations.push(r));
  };
  const byPath = new Map<string, string>();
  for (const c of cards) if (c.type === "artwork" || c.type === "post") byPath.set(c.path, `artwork:${c.artwork_id}`);
  const nodeOf = (path: string) => byPath.get(path) ?? `post:${path.slice(2)}`;
  for (const c of cards) {
    if (c.type === "artwork" || c.type === "post") {
      const a = ent({ id: `artwork:${c.artwork_id}`, type: c.type, label: c.title, evidence: c.evidence_id });
      ent({ id: `user:${c.author}`, type: "author", label: `@${c.author}` });
      link({ type: "created_by", subject: a.id, object: `user:${c.author}`, evidence: c.evidence_id });
      events.push({ type: "post_created", subject: a.id, timestamp: ts(c.created_at), at: c.created_at, evidence: c.evidence_id });
      if (c.image) {
        const img = ent({ id: `image:${c.image.slice(0, 16)}`, type: "image" });
        link({ type: "shows_image", subject: a.id, object: img.id, evidence: c.evidence_id });
        if (c.first_seen_at && c.first_seen_in) {
          const where = nodeOf(c.first_seen_in);
          if (!entities.has(where)) ent({ id: where, type: "post", label: c.first_seen_in });
          events.push({ type: "image_first_seen", subject: img.id, timestamp: ts(c.first_seen_at), at: c.first_seen_at, evidence: c.evidence_id, where: c.first_seen_in });
          if (where !== a.id) link({ type: c.first_seen_match === "near" ? "near_identical_to" : "first_seen_in", subject: c.first_seen_match === "near" ? a.id : img.id, object: where, evidence: c.evidence_id });
        }
      }
      for (const k of c.concepts ?? []) {
        ent({ id: `concept:${k}`, type: "concept", label: k });
        link({ type: "depicts", subject: a.id, object: `concept:${k}`, evidence: c.evidence_id });
      }
      if (c.deleted && c.deleted_at) events.push({ type: "deleted", subject: a.id, timestamp: ts(c.deleted_at), at: c.deleted_at, evidence: c.evidence_id });
    } else if (c.type === "query_image") {
      ent({ id: "query_image:I1", type: "query_image", label: "the uploaded image", evidence: "I1" });
    } else if (c.type === "result") {
      const r = ent({ id: `result:${c.evidence_id}`, type: "result", label: c.text, evidence: c.evidence_id });
      if (c.answer_type === "author" && typeof c.answer === "string") link({ type: "answers", subject: r.id, object: `user:${c.answer}`, evidence: c.evidence_id });
      if (c.answer_type === "post" && typeof c.answer === "string") link({ type: "answers", subject: r.id, object: nodeOf(c.answer), evidence: c.evidence_id });
    }
  }
  // what the history operator knows: edits, deletions, every appearance of the image
  for (const [post, h] of history) {
    const node = nodeOf(post);
    if (!entities.has(node)) ent({ id: node, type: "post", label: post });
    for (const e of h.events) if (e.kind !== "create") events.push({ type: e.kind === "edit" ? "edited" : "deleted", subject: node, timestamp: e.at, at: iso(e.at)! });
    const origin = h.appearances[0];
    for (const a of h.appearances) {
      const other = nodeOf(`/@${a.author}/${a.permlink}`);
      if (!entities.has(other)) ent({ id: other, type: "post", label: `/@${a.author}/${a.permlink}` });
      if (other !== node) link({ type: a.match === "near" ? "near_identical_to" : "same_image_as", subject: node, object: other });
      if (origin && a !== origin) events.push({ type: "reposted", subject: other, timestamp: a.from, at: iso(a.from)!, where: `/@${a.author}/${a.permlink}` });
    }
    if (h.last_seen) events.push({ type: "image_last_seen", subject: node, timestamp: h.last_seen, at: iso(h.last_seen)! });
  }
  // the uploaded image (a visual question): identical, near-identical, or only similar
  if (cards.some((c) => c.type === "query_image")) {
    for (const c of cards) {
      if ((c.type !== "artwork" && c.type !== "post") || !c.identity) continue;
      const id = c.identity;
      const type = id.exact ? "same_image_as" : id.perceptual?.near_identical ? "near_identical_to" : id.visual !== null ? "visually_similar_to" : null;
      if (type) link({ type, subject: "query_image:I1", object: `artwork:${c.artwork_id}`, evidence: c.evidence_id });
    }
  }
  // same bytes shown by several cards
  const byImage = new Map<string, string[]>();
  for (const c of cards) if ((c.type === "artwork" || c.type === "post") && c.image) byImage.set(c.image, [...(byImage.get(c.image) ?? []), `artwork:${c.artwork_id}`]);
  for (const ids of byImage.values()) for (let i = 1; i < ids.length; i++) link({ type: "same_image_as", subject: ids[i], object: ids[0] });
  events.sort((a, b) => a.timestamp - b.timestamp || a.subject.localeCompare(b.subject));
  return { entities: [...entities.values()], events: dedupeEvents(events), relations };
}

function dedupeEvents(ev: GraphEvent[]): GraphEvent[] {
  const seen = new Set<string>();
  return ev.filter((e) => {
    const k = `${e.type}|${e.subject}|${e.timestamp}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

/** Rows for posts named by path (a first sighting in a deleted post, a history appearance). */
export async function rowsByPath(env: Env, paths: string[]): Promise<Map<string, Row>> {
  const out = new Map<string, Row>();
  const refs = [...new Set(paths)].map((p) => /^\/@([a-z0-9][a-z0-9.-]{1,31})\/([^/\s]+)$/.exec(p)).filter((m): m is RegExpExecArray => !!m).slice(0, 40);
  if (!refs.length) return out;
  const res = await env.DB.batch(refs.map((m) => env.DB.prepare("SELECT id FROM posts WHERE author = ? AND permlink = ?").bind(m[1], m[2])));
  const ids = res.map((r) => (r.results?.[0] as { id?: number } | undefined)?.id).filter((x): x is number => typeof x === "number");
  const rows = await hydrateRows(env.DB, ids, null);
  for (const row of rows.values()) out.set(`/@${row.author}/${row.permlink}`, row);
  return out;
}
