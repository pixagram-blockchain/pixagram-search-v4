// One account's share of the backfill. Kept free of `cloudflare:workers` so it can be unit tested.

import { RpcError, type BridgePost, type ChainRpc } from "./rpc";

export interface BackfillCounters {
  authors: number;
  posts: number;
  inserted: number;
  updated: number;
  enqueued: number;
  skipped: number;
  /** Accounts that Hivemind does not know (see backfillAuthor). */
  missing: number;
}

export function emptyCounters(authors = 0): BackfillCounters {
  return { authors, posts: 0, inserted: 0, updated: 0, enqueued: 0, skipped: 0, missing: 0 };
}

export function addCounters(into: BackfillCounters, c: BackfillCounters): void {
  for (const k of Object.keys(into) as (keyof BackfillCounters)[]) if (k !== "authors") into[k] += c[k];
}

export type IngestFn = (post: BridgePost) => Promise<{ action: string; enqueued: boolean }>;

/**
 * Walk one account's top-level posts (Hivemind bridge.get_account_posts, 20 per page, newest
 * first) and ingest each one.
 *
 * Some chain accounts are unknown to Hivemind on api.pixagram.com. condenser_api.lookup_accounts
 * lists them, but bridge.get_account_posts answers "Assert Exception:Account <name> does not
 * exist". Seen 2026-09-30: pixa.omnibus, pixa.rex, pixa.team, steem, steem.dao, all created at
 * the TGE (2026-09-04 12:00) and all with 0 posts. That answer is deterministic, so the account
 * is counted as `missing` and skipped. Before this, the error failed the Workflow step, and so
 * the whole backfill after its retries. Any other error still propagates, so Workflows retries
 * the step and real failures stay visible.
 */
export async function backfillAuthor(
  rpc: Pick<ChainRpc, "getAccountPosts">,
  author: string,
  ingest: IngestFn,
  c: BackfillCounters,
): Promise<void> {
  let start: { author: string; permlink: string } | undefined;
  for (let page = 0; page < 5000; page++) {
    let posts: BridgePost[];
    try {
      posts = await rpc.getAccountPosts(author, start, 20);
    } catch (e) {
      if (page === 0 && e instanceof RpcError && /does not exist/i.test(e.message)) {
        c.missing++;
        console.warn("backfill: account unknown to Hivemind, skipped", author);
        return;
      }
      throw e;
    }
    if (!posts?.length) break;
    for (const p of posts) {
      if (p.author !== author) continue; // reblogs never appear with sort=posts, but be safe
      c.posts++;
      const r = await ingest(p);
      if (r.action === "inserted") c.inserted++;
      else if (r.action === "updated" || r.action === "deleted") c.updated++;
      else c.skipped++;
      if (r.enqueued) c.enqueued++;
    }
    if (posts.length < 20) break;
    const last = posts[posts.length - 1];
    start = { author: last.author, permlink: last.permlink };
  }
}
