// Post history: one row per top-level comment operation (post_versions), and the derived
// "since when does this image exist" fields on artworks (image_since, first_seen).
//
// Sources, from most to least exact:
//   tail      ops read from irreversible blocks by the ChainIndexer (block, trx id, op index)
//   history   ops read from account_history_api during the backfill (same precision)
//   snapshot  nothing better known: the post's created/updated times from get_content
//
// Pixagram deletes by editing the body to "deleted", so a deleted post keeps its earlier
// versions here, with the content hash of the image it showed. That is what lets /ask say who
// *first* posted an image even when that post was deleted and the image re-posted later.

import type { Env } from "../env";
import { base64Decode, sha256Hex } from "../lib/bytes";
import { chainTime, normalizeOp, type AccountHistoryEntry, type ChainRpc, type CommentOp } from "./rpc";
import { extractImage, isDeletedBody } from "./parse";
import { decodeImage, sniff } from "../enrich/decode";
import { hamming, phash } from "../enrich/phash";
import { computeStats } from "../enrich/stats";

export type BodyKind = "image" | "text" | "patch" | "deleted";

export interface VersionRow {
  author: string;
  permlink: string;
  block_num: number;
  trx_id: string;
  op_in_trx: number;
  at: number;
  kind: "create" | "edit" | "delete";
  body_kind: BodyKind;
  content_hash: string | null;
  phash: string | null;
  /** named-colour shares of the image ([{name, weight}]), JSON */
  buckets_json: string | null;
  mime: string | null;
  title: string | null;
  source: "tail" | "history" | "snapshot";
}

export function bodyKind(body: string): BodyKind {
  if (isDeletedBody(body)) return "deleted";
  if (body.startsWith("@@ ")) return "patch"; // diff-match-patch edit: the image cannot be hashed without the previous body
  return extractImage(body) ? "image" : "text";
}

/** A version row from a comment operation (tail or account history). `created` decides create vs edit. */
export async function versionFromOp(
  op: CommentOp,
  where: { block_num: number; trx_id: string; op_in_trx: number; at: number; source: "tail" | "history" },
  created: number | null,
): Promise<VersionRow> {
  const body = typeof op.body === "string" ? op.body : "";
  const kind = bodyKind(body);
  let hash: string | null = null;
  let ph: string | null = null;
  let buckets: string | null = null;
  let mime: string | null = null;
  if (kind === "image") {
    const img = extractImage(body)!;
    mime = img.mime;
    try {
      const bytes = base64Decode(img.base64);
      hash = await sha256Hex(bytes);
      // pHash links a re-encoded re-upload of the same artwork (e.g. deleted, then posted again);
      // the colours keep a recolour (same shapes, same pHash) from counting as the same artwork.
      const c = sniff(bytes);
      if (img.supported && c.format !== "unknown") {
        const decoded = await decodeImage(bytes, c);
        ph = phash(decoded);
        buckets = JSON.stringify(computeStats(decoded, { lossy: c.lossy }).buckets);
      }
    } catch {
      // keep what we have: an undecodable image still has its content hash
    }
  }
  return {
    author: op.author,
    permlink: op.permlink,
    block_num: where.block_num,
    trx_id: where.trx_id,
    op_in_trx: where.op_in_trx,
    at: where.at,
    kind: kind === "deleted" ? "delete" : created !== null && where.at > created ? "edit" : "create",
    body_kind: kind,
    content_hash: hash,
    phash: ph,
    buckets_json: buckets,
    mime,
    title: typeof op.title === "string" ? op.title.slice(0, 512) : null,
    source: where.source,
  };
}

export function snapshotVersion(
  p: { author: string; permlink: string; created: number; title: string },
  contentHash: string | null,
  mime: string | null,
  ph: string | null = null,
  buckets: Array<{ name: string; weight: number }> | null = null,
): VersionRow {
  return {
    author: p.author,
    permlink: p.permlink,
    block_num: 0,
    trx_id: "snapshot",
    op_in_trx: 0,
    at: p.created,
    kind: "create",
    body_kind: contentHash ? "image" : "text",
    content_hash: contentHash,
    phash: ph,
    buckets_json: buckets ? JSON.stringify(buckets) : null,
    mime,
    title: p.title,
    source: "snapshot",
  };
}

export function versionStatements(db: D1Database, rows: VersionRow[]): D1PreparedStatement[] {
  return rows.map((v) =>
    db
      .prepare(
        `INSERT INTO post_versions (author, permlink, block_num, trx_id, op_in_trx, at, kind, body_kind, content_hash, phash, buckets_json, mime, title, source)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(author, permlink, block_num, trx_id, op_in_trx) DO UPDATE SET
           content_hash = COALESCE(excluded.content_hash, post_versions.content_hash), phash = COALESCE(excluded.phash, post_versions.phash),
           buckets_json = COALESCE(excluded.buckets_json, post_versions.buckets_json),
           mime = COALESCE(excluded.mime, post_versions.mime), title = excluded.title, kind = excluded.kind, body_kind = excluded.body_kind, at = excluded.at`,
      )
      .bind(v.author, v.permlink, v.block_num, v.trx_id, v.op_in_trx, v.at, v.kind, v.body_kind, v.content_hash, v.phash, v.buckets_json, v.mime, v.title, v.source),
  );
}

export async function writeVersions(db: D1Database, rows: VersionRow[]): Promise<void> {
  for (let i = 0; i < rows.length; i += 50) await db.batch(versionStatements(db, rows.slice(i, i + 50)));
}

/**
 * Recompute image_since / first_seen for one artwork from post_versions.
 *   image_since  earliest exact version of this post showing the current image; without exact
 *                versions, `created` (exact when the post was never edited, else a lower bound)
 *   first_seen   earliest appearance of the same image: exact bytes in any post (deleted ones
 *                included), or a near-identical upload by the same author (pHash <= NEAR)
 */
export async function refreshImageHistory(db: D1Database, postId: number): Promise<void> {
  const row = await db
    .prepare("SELECT p.author, p.permlink, p.created, p.updated, a.content_hash, a.phash, a.buckets_json FROM posts p JOIN artworks a ON a.post_id = p.id WHERE p.id = ?")
    .bind(postId)
    .first<{ author: string; permlink: string; created: number; updated: number; content_hash: string; phash: string | null; buckets_json: string | null }>();
  if (!row?.content_hash) return;
  const own = await db
    .prepare("SELECT MIN(at) AS at FROM post_versions WHERE author = ? AND permlink = ? AND content_hash = ? AND source != 'snapshot'")
    .bind(row.author, row.permlink, row.content_hash)
    .first<{ at: number | null }>();
  let imageSince: number;
  let exact: number;
  if (own?.at) {
    // The creating op's block time trails `created` by a block: report `created` for it.
    imageSince = own.at <= row.created + CREATE_SKEW ? row.created : own.at;
    exact = 1;
  } else {
    imageSince = row.created;
    exact = row.updated <= row.created ? 1 : 0;
  }
  await db.prepare("UPDATE artworks SET image_since = ?, history_exact = ? WHERE post_id = ?").bind(imageSince, exact, postId).run();

  // Earliest sighting of the same image: the same bytes in any post (deleted ones included) or
  // another artwork's image_since; or a near-identical image (pHash <= NEAR) by the same author,
  // which is how a deleted-then-reposted artwork shows up (the re-upload is re-encoded).
  const candidates: Array<{ author: string; permlink: string; at: number; match: "exact" | "near" | "self" }> = [{ author: row.author, permlink: row.permlink, at: imageSince, match: "self" }];
  const v = await db
    .prepare("SELECT author, permlink, at FROM post_versions WHERE content_hash = ? AND source != 'snapshot' ORDER BY at ASC, id ASC LIMIT 1")
    .bind(row.content_hash)
    .first<{ author: string; permlink: string; at: number }>();
  if (v) candidates.push({ ...v, match: v.author === row.author && v.permlink === row.permlink ? "self" : "exact" });
  const a = await db
    .prepare(
      `SELECT p.author, p.permlink, COALESCE(a.image_since, p.created) AS at FROM artworks a JOIN posts p ON p.id = a.post_id
       WHERE a.content_hash = ? ORDER BY at ASC, p.id ASC LIMIT 1`,
    )
    .bind(row.content_hash)
    .first<{ author: string; permlink: string; at: number }>();
  if (a) candidates.push({ ...a, match: a.author === row.author && a.permlink === row.permlink ? "self" : "exact" });
  if (row.phash && row.buckets_json) {
    const near = await db
      .prepare("SELECT author, permlink, at, phash, buckets_json FROM post_versions WHERE author = ? AND phash IS NOT NULL AND buckets_json IS NOT NULL AND at < ? AND source != 'snapshot'")
      .bind(row.author, imageSince)
      .all<{ author: string; permlink: string; at: number; phash: string; buckets_json: string }>();
    // pHash reads brightness only: a recoloured variant (green slime, then blue slime) has the
    // same pHash. Near needs the same shapes *and* the same colours.
    for (const n of near.results ?? []) {
      if (hamming(n.phash, row.phash) <= NEAR && colourAgreement(n.buckets_json, row.buckets_json) >= NEAR_COLOURS) candidates.push({ author: n.author, permlink: n.permlink, at: n.at, match: "near" });
    }
  }
  const best = candidates.sort((x, y) => x.at - y.at)[0];
  await db
    .prepare("UPDATE artworks SET first_seen = ?, first_seen_author = ?, first_seen_permlink = ?, first_seen_match = ? WHERE post_id = ?")
    .bind(best.at, best.author, best.permlink, best.match, postId)
    .run();
}

/** pHash distance under which two uploads by the same author count as the same artwork... */
export const NEAR = 4;
/** ...provided their named-colour shares also overlap this much (histogram intersection). */
export const NEAR_COLOURS = 0.8;

export function colourAgreement(a: string | null, b: string | null): number {
  const parse = (s: string | null): Array<{ name: string; weight: number }> => {
    try {
      const v = JSON.parse(s ?? "[]");
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  };
  const m = new Map(parse(a).map((x) => [x.name, x.weight]));
  let s = 0;
  for (const x of parse(b)) s += Math.min(m.get(x.name) ?? 0, x.weight);
  return s;
}

/** Every artwork showing one of these images (to refresh after new versions were written). */
export async function artworksWithHashes(db: D1Database, hashes: string[]): Promise<number[]> {
  const out: number[] = [];
  const uniq = [...new Set(hashes.filter(Boolean))];
  for (let i = 0; i < uniq.length; i += 90) {
    const chunk = uniq.slice(i, i + 90);
    const r = await db.prepare(`SELECT post_id FROM artworks WHERE content_hash IN (${chunk.map(() => "?").join(",")})`).bind(...chunk).all<{ post_id: number }>();
    for (const x of r.results ?? []) out.push(x.post_id);
  }
  return out;
}

// ---- account history walk -----------------------------------------------------------------------

export interface HistoryWalkResult {
  calls: number;
  ops: number;
  versions: number;
  complete: boolean; // reached the start of the account's history
}

/**
 * Walk an account's comment operations from newest to oldest (account_history_api with the
 * comment filter) and collect the top-level ones (posts, not replies) as versions. Witness
 * accounts have tens of thousands of producer rewards between posts; each call scans at most 2000
 * operations, so the walk is bounded by `maxCalls` and reports whether it reached the start.
 */
export async function walkCommentHistory(
  rpc: Pick<ChainRpc, "getAccountHistory">,
  account: string,
  opts: { maxCalls: number; pageSize?: number },
): Promise<{ rows: VersionRow[]; result: HistoryWalkResult }> {
  const rows: VersionRow[] = [];
  let start = -1;
  let calls = 0;
  let ops = 0;
  let complete = false;
  // Small pages: every matching op carries its full body (an artwork is a data URI of up to a few
  // hundred KB), and each op is hashed and dropped as soon as it arrives.
  const pageSize = opts.pageSize ?? 100;
  while (calls < opts.maxCalls) {
    calls++;
    const page = await rpc.getAccountHistory(account, start, pageSize, 2);
    for (const [, e] of page.history) {
      const op = normalizeOp(e.op);
      if (op.type !== "comment") continue;
      const c = op.value as CommentOp;
      ops++;
      if (c.parent_author !== "" || c.author !== account) continue;
      rows.push(await versionFromOp(c, { block_num: e.block, trx_id: e.trx_id, op_in_trx: e.op_in_trx, at: chainTime(e.timestamp), source: "history" }, null));
    }
    if (page.next === null) {
      complete = true;
      break;
    }
    start = page.next;
  }
  // Oldest first; when the walk reached the start of the account, the first op of a permlink is
  // its creation. Otherwise relabelKinds() fixes the labels from posts.created afterwards.
  rows.sort((a, b) => a.at - b.at || a.block_num - b.block_num || a.op_in_trx - b.op_in_trx);
  const seen = new Set<string>();
  for (const r of rows) {
    if (r.body_kind === "deleted") r.kind = "delete";
    else r.kind = complete && !seen.has(r.permlink) ? "create" : "edit";
    seen.add(r.permlink);
  }
  return { rows, result: { calls, ops, versions: rows.length, complete } };
}

/** Top-level comment ops of a block range, as versions (tail). */
export async function versionsFromBlocks(
  blocks: Array<{ timestamp: string; transactions?: Array<{ operations?: unknown[] }>; transaction_ids?: string[] }>,
  firstBlock: number,
  allowed: (author: string, permlink: string) => boolean,
): Promise<VersionRow[]> {
  const out: VersionRow[] = [];
  for (let b = 0; b < blocks.length; b++) {
    const block = blocks[b];
    const at = chainTime(block.timestamp);
    const txs = block.transactions ?? [];
    for (let t = 0; t < txs.length; t++) {
      const ops = txs[t].operations ?? [];
      for (let k = 0; k < ops.length; k++) {
        const op = normalizeOp(ops[k] as any);
        if (op.type !== "comment") continue;
        const c = op.value as CommentOp;
        if (c.parent_author !== "" || !allowed(c.author, c.permlink)) continue;
        out.push(await versionFromOp(c, { block_num: firstBlock + b, trx_id: block.transaction_ids?.[t] ?? `b${firstBlock + b}t${t}`, op_in_trx: k, at, source: "tail" }, null));
      }
    }
  }
  return out;
}

/**
 * After writing versions: label create / edit / delete. The creating operation is the earliest
 * exact version of a permlink; its block time is a block (≈3 s) later than the post's `created`
 * (head block time when the op was applied), so `created` is matched with a margin.
 */
export async function relabelKinds(env: Env, author: string): Promise<void> {
  await env.DB
    .prepare(
      `UPDATE post_versions SET kind = CASE
         WHEN body_kind = 'deleted' THEN 'delete'
         WHEN source = 'snapshot' THEN 'create'
         WHEN id = (SELECT v2.id FROM post_versions v2 WHERE v2.author = post_versions.author AND v2.permlink = post_versions.permlink AND v2.source != 'snapshot'
                    ORDER BY v2.at, v2.block_num, v2.op_in_trx LIMIT 1)
              AND at <= COALESCE((SELECT p.created FROM posts p WHERE p.author = post_versions.author AND p.permlink = post_versions.permlink), at) + ?
           THEN 'create'
         ELSE 'edit' END
       WHERE author = ?`,
    )
    .bind(CREATE_SKEW, author)
    .run();
}

/** Seconds between a post's `created` and the block time of its creating operation (one or two blocks). */
export const CREATE_SKEW = 30;

export type { AccountHistoryEntry };
