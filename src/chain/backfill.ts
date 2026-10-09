// BackfillWorkflow: (re)ingest every top-level post of every account, and record each account's
// post history (post_versions) from account_history_api.
//
// database_api.list_comments is not enabled on the public Pixa nodes, so current posts are walked
// account by account through Hivemind's bridge.get_account_posts (20 posts per page). History
// comes from account_history_api filtered to comment operations, which also reaches posts that
// were deleted since (Pixagram deletes by editing the body to "deleted").
// Each step is retried by Workflows on failure and returns only counters, so state stays tiny.
// Accounts that Hivemind does not know are skipped (see backfill-author.ts).

import { WorkflowEntrypoint, type WorkflowEvent, type WorkflowStep } from "cloudflare:workers";
import type { Env } from "../env";
import { bool, int } from "../env";
import { ingestPost, rpcFor } from "./ingest";
import { setSetting } from "../db/posts";
import { addCounters, backfillAuthor, emptyCounters, type BackfillCounters } from "./backfill-author";
import { artworksWithHashes, refreshImageHistory, relabelKinds, walkCommentHistory, writeVersions } from "./versions";

export interface BackfillParams {
  /** Restrict to these accounts; default = every account on the chain. */
  authors?: string[];
  /** Free-text label for logs. */
  reason?: string;
  /** Only (re)build post_versions; do not re-ingest posts. */
  historyOnly?: boolean;
  /** Skip the history walk (default: HISTORY_BACKFILL, on). */
  noHistory?: boolean;
}

type Counters = BackfillCounters & { history_calls: number; versions: number; history_incomplete: number };

const AUTHORS_PER_STEP = 5;

export class BackfillWorkflow extends WorkflowEntrypoint<Env, BackfillParams> {
  async run(event: WorkflowEvent<BackfillParams>, step: WorkflowStep): Promise<Counters> {
    const params = event.payload ?? {};
    const withHistory = !params.noHistory && bool(this.env.HISTORY_BACKFILL, true);

    const authors = await step.do("list accounts", async () => {
      if (params.authors?.length) return params.authors;
      const rpc = rpcFor(this.env);
      const all: string[] = [];
      let start = "";
      for (;;) {
        const page = await rpc.lookupAccounts(start, 1000);
        for (const a of page) if (a !== start) all.push(a);
        if (page.length < 1000) break;
        start = page[page.length - 1];
        all.push(start);
      }
      return Array.from(new Set(all));
    });

    const total: Counters = { ...emptyCounters(authors.length), history_calls: 0, versions: 0, history_incomplete: 0 };
    const reason = `backfill:${params.reason ?? event.instanceId}`;

    for (let i = 0; i < authors.length; i += AUTHORS_PER_STEP) {
      const chunk = authors.slice(i, i + AUTHORS_PER_STEP);
      if (!params.historyOnly) {
        const c = await step.do(
          `authors ${i + 1}-${i + chunk.length}`,
          { retries: { limit: 5, delay: "10 seconds", backoff: "exponential" }, timeout: "10 minutes" },
          async () => {
            const rpc = rpcFor(this.env);
            const c = emptyCounters(chunk.length);
            for (const author of chunk) await backfillAuthor(rpc, author, (p) => ingestPost(this.env, p, null, reason), c);
            return c;
          },
        );
        addCounters(total, c);
      }
      if (withHistory) {
        // Only accounts with at least one post (deleted ones included), one step each: a witness
        // account has tens of thousands of producer rewards to scan past.
        const posting = await step.do(`posting authors ${i + 1}-${i + chunk.length}`, async () => {
          const r = await this.env.DB.prepare(`SELECT DISTINCT author FROM posts WHERE author IN (${chunk.map(() => "?").join(",")})`).bind(...chunk).all<{ author: string }>();
          return (r.results ?? []).map((x) => x.author);
        });
        for (const author of posting) {
          const h = await step.do(
            `history ${author}`,
            { retries: { limit: 4, delay: "15 seconds", backoff: "exponential" }, timeout: "15 minutes" },
            async () => {
              const { rows, result } = await walkCommentHistory(rpcFor(this.env), author, { maxCalls: int(this.env.HISTORY_MAX_CALLS, 300) });
              if (rows.length) {
                await writeVersions(this.env.DB, rows);
                await relabelKinds(this.env, author);
                for (const id of await artworksWithHashes(this.env.DB, rows.map((r) => r.content_hash ?? ""))) await refreshImageHistory(this.env.DB, id);
              }
              return result;
            },
          );
          total.history_calls += h.calls;
          total.versions += h.versions;
          if (!h.complete) total.history_incomplete++;
        }
      }
    }

    await step.do("record", async () => {
      await setSetting(this.env.DB, "backfill:last", JSON.stringify({ at: Date.now(), instance: event.instanceId, ...total }));
    });
    return total;
  }
}
