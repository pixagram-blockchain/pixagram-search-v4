// D1 access for posts, artworks, search docs, vocabulary and jobs. All writes are idempotent.

import type { Env, Stage } from "../env";
import { now } from "../env";
import type { ParsedPost } from "../chain/parse";
import { tokens, trigrams } from "../lib/text";
import { base64Decode, sha256Hex } from "../lib/bytes";
import { homeShard } from "../paph/shards";

export interface PostRow {
  id: number;
  author: string;
  permlink: string;
  type: "artwork" | "blog";
  title: string;
  description: string;
  body: string;
  body_length: number;
  category: string | null;
  tags_json: string;
  app: string | null;
  nsfw: number;
  ai_training: number | null;
  license_json: string | null;
  royalty_pct: number | null;
  created: number;
  updated: number;
  block_num: number | null;
  deleted: number;
  net_votes: number;
  payout: number;
  children: number;
  listed: number;
  price: number | null;
  price_symbol: string | null;
  indexed_at: number;
  text_hash: string | null;
  text_model: string | null;
}

export interface ArtworkRow {
  post_id: number;
  content_hash: string;
  mime: string;
  bytes: number;
  lossy: number;
  width: number | null;
  height: number | null;
  pixels: number | null;
  size_class: string | null;
  color_count: number | null;
  has_transparency: number | null;
  transparent_share: number | null;
  primary_color: string | null;
  background_hex: string | null;
  palette_json: string | null;
  buckets_json: string | null;
  phash: string | null;
  dhash: string | null;
  stats_hash: string | null;
  features_hash: string | null;
  embed_hash: string | null;
  embed_model: string | null;
  embed_views: string | null;
  describe_hash: string | null;
  vlm_model: string | null;
  ai_caption: string | null;
  ai_subjects_json: string | null;
  ai_objects_json: string | null;
  ai_tags_json: string | null;
  ai_style: string | null;
  ai_mood: string | null;
  ai_text: string | null;
  ai_nsfw: number | null;
  ai_status: string | null;
  concepts_hash: string | null;
  orientation: string | null;
  monochrome: number | null;
  r2_orig_key: string | null;
  r2_up_key: string | null;
  up_width: number | null;
  up_height: number | null;
  up_factor: number | null;
  image_since: number | null;
  first_seen: number | null;
  /** content_hash the copy-detection stage last completed for, and under which identity (migration 0006) */
  paph_hash?: string | null;
  paph_engine?: string | null;
  updated: number;
}

export interface UpsertResult {
  id: number;
  inserted: boolean;
  /** title/description/tags/body/deleted changed vs the stored row */
  textChanged: boolean;
  /** the post is an artwork whose image payload should be (re)processed */
  needsEnrich: boolean;
  /** nsfw / ai-training changed: the vector metadata must follow */
  flagsChanged: boolean;
}

const BODY_FTS_LIMIT = 8000;

/** Vector ids are the post id in both indexes (VEC: image, VEC_TEXT: text), as in v2. */
export const vectorId = (postId: number) => String(postId);

export async function getPostByRef(db: D1Database, author: string, permlink: string): Promise<PostRow | null> {
  return db.prepare("SELECT * FROM posts WHERE author = ? AND permlink = ?").bind(author, permlink).first<PostRow>();
}

export async function getPostById(db: D1Database, id: number): Promise<PostRow | null> {
  return db.prepare("SELECT * FROM posts WHERE id = ?").bind(id).first<PostRow>();
}

export async function getArtwork(db: D1Database, postId: number): Promise<ArtworkRow | null> {
  return db.prepare("SELECT * FROM artworks WHERE post_id = ?").bind(postId).first<ArtworkRow>();
}

/**
 * Insert or update a post from chain state. Returns whether the image pipeline should run.
 * Enrichment of the image is keyed on the payload; a title edit alone re-runs only the cheap
 * text stages (concepts, text vector), decided by the caller from `textChanged`.
 */
export async function upsertPost(env: Env, p: ParsedPost, blockNum: number | null): Promise<UpsertResult> {
  const db = env.DB;
  const existing = await getPostByRef(db, p.author, p.permlink);
  const t = now();
  const tagsJson = JSON.stringify(p.tags);

  if (!existing) {
    const r = await db
      .prepare(
        `INSERT INTO posts (author, permlink, type, title, description, body, body_length, category, tags_json, app,
           nsfw, ai_training, license_json, royalty_pct, created, updated, block_num, deleted, net_votes, payout, children, indexed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         RETURNING id`,
      )
      .bind(
        p.author, p.permlink, p.type, p.title, p.description, p.body, p.bodyLength, p.category, tagsJson, p.app,
        p.nsfw ? 1 : 0, p.aiTraining === null ? null : p.aiTraining ? 1 : 0, p.licenseJson, p.royaltyPct,
        p.created, p.updated, blockNum, p.deleted ? 1 : 0, p.netVotes, p.payout, p.children, t,
      )
      .first<{ id: number }>();
    const id = r!.id;
    if (!p.deleted) {
      await replaceTags(db, id, p.tags);
      await writeSearchDoc(db, id, { author: p.author, title: p.title, description: p.description, body: p.body, tags: p.tags });
    }
    return { id, inserted: true, textChanged: true, needsEnrich: p.type === "artwork" && !p.deleted && !!p.image, flagsChanged: false };
  }

  // A snapshot older than what is stored (a backfill page fetched before an edit or a
  // delete-by-edit that the tail has already applied) must not roll the post back.
  if (p.updated < existing.updated) return { id: existing.id, inserted: false, textChanged: false, needsEnrich: false, flagsChanged: false };

  const textChanged =
    existing.title !== p.title ||
    existing.description !== p.description ||
    existing.body !== p.body ||
    existing.tags_json !== tagsJson ||
    existing.deleted !== (p.deleted ? 1 : 0) ||
    existing.type !== p.type;
  const aiTraining = p.aiTraining === null ? null : p.aiTraining ? 1 : 0;
  // flags that live in the vector metadata as well
  const flagsChanged = existing.nsfw !== (p.nsfw ? 1 : 0) || existing.ai_training !== aiTraining;

  await db
    .prepare(
      `UPDATE posts SET type = ?, title = ?, description = ?, body = ?, body_length = ?, category = ?, tags_json = ?, app = ?,
         nsfw = ?, ai_training = ?, license_json = ?, royalty_pct = ?, updated = ?, block_num = COALESCE(?, block_num),
         deleted = ?, net_votes = ?, payout = ?, children = ?, indexed_at = ?
       WHERE id = ?`,
    )
    .bind(
      p.type, p.title, p.description, p.body, p.bodyLength, p.category, tagsJson, p.app,
      p.nsfw ? 1 : 0, aiTraining, p.licenseJson, p.royaltyPct,
      Math.max(p.updated, existing.updated), blockNum, p.deleted ? 1 : 0, p.netVotes, p.payout, p.children, t,
      existing.id,
    )
    .run();

  if (p.deleted) {
    await removeFromIndexes(env, existing.id);
    return { id: existing.id, inserted: false, textChanged, needsEnrich: false, flagsChanged: false };
  }
  // An artwork edited into a text post leaves the image indexes (the row stays for its history).
  if (existing.type === "artwork" && p.type === "blog") await removeImageIndexes(env, existing.id);

  if (textChanged) {
    await replaceTags(db, existing.id, p.tags);
    const art = await getArtwork(db, existing.id);
    await writeSearchDoc(db, existing.id, {
      author: p.author,
      title: p.title,
      description: p.description,
      body: p.body,
      tags: p.tags,
      aiCaption: art?.ai_caption ?? "",
      aiTags: art?.ai_tags_json ? (JSON.parse(art.ai_tags_json) as string[]) : [],
    });
  }

  // Image changed? The payload's SHA-256 against the stored content hash (comparing lengths
  // missed edits that kept the byte length, common on small lossless sprites).
  let needsEnrich = false;
  if (p.type === "artwork" && p.image) {
    const art = await getArtwork(db, existing.id);
    needsEnrich = !art || art.stats_hash === null || existing.deleted === 1 || art.content_hash !== (await sha256Hex(base64Decode(p.image.base64)));
  }
  return { id: existing.id, inserted: false, textChanged, needsEnrich, flagsChanged };
}

export async function replaceTags(db: D1Database, postId: number, tags: string[]): Promise<void> {
  const stmts = [db.prepare("DELETE FROM post_tags WHERE post_id = ?").bind(postId)];
  for (const tag of tags) stmts.push(db.prepare("INSERT OR IGNORE INTO post_tags (post_id, tag) VALUES (?, ?)").bind(postId, tag));
  await db.batch(stmts);
}

export interface SearchDocInput {
  author: string;
  title: string;
  description: string;
  body: string;
  tags: string[];
  aiCaption?: string;
  aiTags?: string[];
}

interface SearchDocRow {
  title: string;
  description: string;
  tags: string;
  ai_caption: string;
  ai_tags: string;
  author: string;
}

/** Vocabulary terms of a search doc (blog bodies excluded: long, and not what people misspell). */
export function docTerms(d: { title: string; description: string; tags: string; ai_caption: string; ai_tags: string; author: string }): Set<string> {
  const out = new Set<string>();
  for (const f of [d.title, d.description, d.tags, d.ai_caption, d.ai_tags, d.author]) {
    for (const t of tokens(f ?? "", { keepHyphenated: true })) if (t.length >= 3 && t.length <= 40 && !/^\d+$/.test(t)) out.add(t);
  }
  return out;
}

/** Apply document-frequency deltas to the vocabulary (and index trigrams of new terms). */
export async function updateVocab(db: D1Database, before: Set<string>, after: Set<string>): Promise<void> {
  const added = [...after].filter((t) => !before.has(t));
  const removed = [...before].filter((t) => !after.has(t));
  if (!added.length && !removed.length) return;
  const known = new Set<string>();
  for (let i = 0; i < added.length; i += 90) {
    const chunk = added.slice(i, i + 90);
    const r = await db.prepare(`SELECT term FROM vocab WHERE term IN (${chunk.map(() => "?").join(",")})`).bind(...chunk).all<{ term: string }>();
    for (const x of r.results ?? []) known.add(x.term);
  }
  const stmts: D1PreparedStatement[] = [];
  for (const t of added) {
    stmts.push(db.prepare("INSERT INTO vocab (term, df) VALUES (?, 1) ON CONFLICT(term) DO UPDATE SET df = df + 1").bind(t));
    if (!known.has(t)) for (const g of trigrams(t)) stmts.push(db.prepare("INSERT OR IGNORE INTO vocab_grams (gram, term) VALUES (?, ?)").bind(g, t));
  }
  for (const t of removed) stmts.push(db.prepare("UPDATE vocab SET df = MAX(0, df - 1) WHERE term = ?").bind(t));
  for (let i = 0; i < stmts.length; i += 100) await db.batch(stmts.slice(i, i + 100));
}

/**
 * Recompute the spelling vocabulary from every search doc. Needed once after applying 0002 in
 * place on a v2 database (whose documents predate the vocabulary), and usable as a repair.
 */
export async function rebuildVocab(db: D1Database): Promise<{ docs: number; terms: number }> {
  const df = new Map<string, number>();
  let last = 0;
  let docs = 0;
  for (;;) {
    const page = await db
      .prepare("SELECT post_id, title, description, tags, ai_caption, ai_tags, author FROM search_docs WHERE post_id > ? ORDER BY post_id LIMIT 500")
      .bind(last)
      .all<SearchDocRow & { post_id: number }>();
    const rows = page.results ?? [];
    if (!rows.length) break;
    for (const r of rows) for (const t of docTerms(r)) df.set(t, (df.get(t) ?? 0) + 1);
    docs += rows.length;
    last = rows[rows.length - 1].post_id;
  }
  await db.batch([db.prepare("DELETE FROM vocab"), db.prepare("DELETE FROM vocab_grams")]);
  const stmts: D1PreparedStatement[] = [];
  for (const [t, n] of df) {
    stmts.push(db.prepare("INSERT INTO vocab (term, df) VALUES (?, ?)").bind(t, n));
    for (const g of trigrams(t)) stmts.push(db.prepare("INSERT OR IGNORE INTO vocab_grams (gram, term) VALUES (?, ?)").bind(g, t));
  }
  for (let i = 0; i < stmts.length; i += 100) await db.batch(stmts.slice(i, i + 100));
  return { docs, terms: df.size };
}

async function readDoc(db: D1Database, postId: number): Promise<SearchDocRow | null> {
  return db.prepare("SELECT title, description, tags, ai_caption, ai_tags, author FROM search_docs WHERE post_id = ?").bind(postId).first<SearchDocRow>();
}

const EMPTY = new Set<string>();

/** Upsert the row that feeds posts_fts (triggers keep the FTS index in sync) and the vocabulary. */
export async function writeSearchDoc(db: D1Database, postId: number, d: SearchDocInput): Promise<void> {
  const body = stripMarkdown(d.body).slice(0, BODY_FTS_LIMIT);
  const before = await readDoc(db, postId);
  const row: SearchDocRow = { title: d.title, description: d.description, tags: d.tags.join(" "), ai_caption: d.aiCaption ?? "", ai_tags: (d.aiTags ?? []).join(" "), author: d.author };
  await db
    .prepare(
      `INSERT INTO search_docs (post_id, title, description, body, tags, ai_caption, ai_tags, author)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
       ON CONFLICT(post_id) DO UPDATE SET title = excluded.title, description = excluded.description, body = excluded.body,
         tags = excluded.tags, ai_caption = excluded.ai_caption, ai_tags = excluded.ai_tags, author = excluded.author`,
    )
    .bind(postId, row.title, row.description, body, row.tags, row.ai_caption, row.ai_tags, row.author)
    .run();
  await updateVocab(db, before ? docTerms(before) : EMPTY, docTerms(row));
}

/** Only the AI-derived columns of the search doc (called by the describe stage). */
export async function updateSearchDocAi(db: D1Database, postId: number, aiCaption: string, aiTags: string[]): Promise<void> {
  const before = await readDoc(db, postId);
  await db.prepare("UPDATE search_docs SET ai_caption = ?, ai_tags = ? WHERE post_id = ?").bind(aiCaption, aiTags.join(" "), postId).run();
  if (before) await updateVocab(db, docTerms(before), docTerms({ ...before, ai_caption: aiCaption, ai_tags: aiTags.join(" ") }));
}

export function stripMarkdown(md: string): string {
  if (!md) return "";
  return md
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ")
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/<[^>]+>/g, " ")
    .replace(/data:[a-z]+\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g, " ")
    .replace(/https?:\/\/\S+/g, " ")
    .replace(/[#*_>`~|-]{1,}/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Soft-deleted or unindexable posts leave FTS, colours, bands, concepts and Vectorize; the rows stay. */
export async function removeFromIndexes(env: Env, postId: number): Promise<void> {
  const before = await readDoc(env.DB, postId);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM search_docs WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM artwork_colors WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM phash_bands WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM artwork_concepts WHERE post_id = ?").bind(postId),
    env.DB.prepare("UPDATE posts SET text_hash = NULL WHERE id = ?").bind(postId),
    // What was just removed must be rebuilt if the post comes back (edited again after the
    // "deleted" edit): clear the hashes that would otherwise mark those stages as done. The AI
    // description stays valid for the same image and is kept.
    env.DB.prepare("UPDATE artworks SET stats_hash = NULL, features_hash = NULL, embed_hash = NULL, concepts_hash = NULL, paph_hash = NULL, paph_engine = NULL WHERE post_id = ?").bind(postId),
    // copy detection: its verdicts go (the fingerprint leaves the index below)
    env.DB.prepare("DELETE FROM paph_matches WHERE a = ?1 OR b = ?1").bind(postId),
    env.DB.prepare("DELETE FROM paph_progress WHERE post_id = ?").bind(postId),
  ]);
  await removeFromCopyIndex(env, postId);
  if (before) await updateVocab(env.DB, docTerms(before), EMPTY);
  for (const index of [env.VEC, env.VEC_TEXT]) {
    if (!index) continue;
    try {
      await index.deleteByIds([vectorId(postId)]);
    } catch (e) {
      console.warn("vectorize delete failed", postId, e);
    }
  }
}

/** A post that is no longer an artwork (edited into text) leaves the image indexes only. */
export async function removeImageIndexes(env: Env, postId: number): Promise<void> {
  await env.DB.batch([
    env.DB.prepare("DELETE FROM artwork_colors WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM phash_bands WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM artwork_concepts WHERE post_id = ?").bind(postId),
    env.DB.prepare("UPDATE artworks SET stats_hash = NULL, features_hash = NULL, embed_hash = NULL, concepts_hash = NULL, paph_hash = NULL, paph_engine = NULL WHERE post_id = ?").bind(postId),
    env.DB.prepare("DELETE FROM paph_matches WHERE a = ?1 OR b = ?1").bind(postId),
    env.DB.prepare("DELETE FROM paph_progress WHERE post_id = ?").bind(postId),
  ]);
  await removeFromCopyIndex(env, postId);
  try {
    await env.VEC.deleteByIds([vectorId(postId)]);
  } catch (e) {
    console.warn("vectorize delete failed", postId, e);
  }
}

/** The work's fingerprint leaves its copy-detection shard (best effort: a stale one is never listed). */
async function removeFromCopyIndex(env: Env, postId: number): Promise<void> {
  if (!env.PAPH) return;
  try {
    await homeShard(env, postId).remove(postId);
  } catch (e) {
    console.warn("paph index delete failed", postId, e);
  }
}

export async function updateCounters(db: D1Database, postId: number, netVotes: number, payout: number, children: number): Promise<void> {
  await db.prepare("UPDATE posts SET net_votes = ?, payout = ?, children = ? WHERE id = ?").bind(netVotes, payout, children, postId).run();
}

type JobStatus = "queued" | "done" | "failed" | "skipped";

/** `attempts` counts consecutive failures (the sweeper gives up at 40); a success resets it. */
function jobStatement(db: D1Database, postId: number, stage: Stage, status: JobStatus, error?: string): D1PreparedStatement {
  return db
    .prepare(
      `INSERT INTO jobs (post_id, stage, status, attempts, error, updated) VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(post_id, stage) DO UPDATE SET status = excluded.status,
         attempts = CASE excluded.status WHEN 'done' THEN 0 WHEN 'failed' THEN jobs.attempts + 1 ELSE jobs.attempts END,
         error = excluded.error, updated = excluded.updated`,
    )
    .bind(postId, stage, status, status === "failed" ? 1 : 0, error ?? null, now());
}

export async function setJob(db: D1Database, postId: number, stage: Stage, status: JobStatus, error?: string): Promise<void> {
  await jobStatement(db, postId, stage, status, error).run();
}

/** Record a stage as done unless it already is (one statement; nothing written when it is). */
export async function settleJob(db: D1Database, postId: number, stage: Stage): Promise<void> {
  await db.prepare("UPDATE jobs SET status = 'done', attempts = 0, error = NULL, updated = ? WHERE post_id = ? AND stage = ? AND status <> 'done'").bind(now(), postId, stage).run();
}

export async function setJobs(db: D1Database, postId: number, stages: Stage[], status: JobStatus, error?: string): Promise<void> {
  if (stages.length) await db.batch(stages.map((s) => jobStatement(db, postId, s, status, error)));
}

/** Many posts at once, 100 statements per batch (one subrequest per batch, not per job). */
export async function setJobsMany(db: D1Database, items: Array<{ postId: number; stages: Stage[] }>, status: JobStatus): Promise<void> {
  const stmts = items.flatMap((x) => x.stages.map((s) => jobStatement(db, x.postId, s, status)));
  for (let i = 0; i < stmts.length; i += 100) await db.batch(stmts.slice(i, i + 100));
}

export async function getSetting(db: D1Database, k: string): Promise<string | null> {
  const r = await db.prepare("SELECT v FROM settings WHERE k = ?").bind(k).first<{ v: string }>();
  return r?.v ?? null;
}

export async function setSetting(db: D1Database, k: string, v: string): Promise<void> {
  await db.prepare("INSERT INTO settings (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v").bind(k, v).run();
}
