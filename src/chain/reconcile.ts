// Reconciliation: the tail's safety net. Hivemind lists the newest posts chain-wide
// (condenser_api.get_discussions_by_created); every post it lists from the last RECONCILE_HOURS
// that the index lacks, or holds in an older state, is ingested now, as the tail would have.
//
// Until v4.8.3 the tail dropped any post Hivemind had not processed yet when the tail asked
// (src/chain/indexer-do.ts), and the only cure was a backfill run by hand every few days. The tail
// now waits for Hivemind instead; this pass catches whatever else slips through (a Hivemind
// outage longer than the tail waits, a parked or deferred post dropped, a replay that skipped
// blocks): every ten minutes over the last RECONCILE_HOURS (3), nightly over RECONCILE_DEEP_HOURS
// (48). A page is 20 posts with their bodies truncated (about 45 KB with the votes); three hours
// are one or two pages.

import type { Env } from "../env";
import { list, now, num } from "../env";
import { chainTime, type CondenserPost } from "./rpc";
import { appAllowed, parseJsonMetadata } from "./parse";
import { ingestPostRef, rpcFor } from "./ingest";
import { getSetting, setSetting } from "../db/posts";

export interface ReconcileResult {
  /** when it ran (unix seconds) and the window it looked at */
  at: number;
  hours: number;
  pages: number;
  /** top-level posts Hivemind listed inside the window (the last few minutes are the tail's) */
  listed: number;
  /** of those, from apps APP_PREFIXES leaves out */
  skipped_app: number;
  /** listed but not in the index */
  missing: number;
  /** in the index, but in a state older than the one on chain (an edit the index did not get) */
  stale: number;
  ingested: number;
  failed: number;
  /** the first posts brought in (author/permlink) */
  recovered: string[];
  /** stopped before the end of the window (page, ingest or time budget): the next run goes on */
  truncated: boolean;
  error?: string;
}

/** What /healthz and /admin/stats show: the last run, and the posts recovered since v4.8.3. */
export interface ReconcileRecord {
  last: ReconcileResult;
  /** posts the reconciliation brought in, in all */
  recovered_total: number;
  /** the last run that brought any in */
  last_recovery: { at: number; missing: number; stale: number; recovered: string[] } | null;
}

const PAGE = 20;
/** Posts younger than this are the tail's: it may be asking Hivemind about them right now. */
const MIN_AGE_S = 300;
const SETTING = "reconcile:last";
const KEEP_RECOVERED = 20;

const key = (p: { author: string; permlink: string }) => `${p.author}/${p.permlink}`;

export async function reconcileRecent(
  env: Env,
  o: { hours?: number; maxPages?: number; maxIngest?: number; budgetMs?: number; record?: boolean } = {},
): Promise<ReconcileResult> {
  const hours = o.hours ?? num(env.RECONCILE_HOURS, 3);
  const res: ReconcileResult = { at: now(), hours, pages: 0, listed: 0, skipped_app: 0, missing: 0, stale: 0, ingested: 0, failed: 0, recovered: [], truncated: false };
  if (!(hours > 0)) return res;
  const since = res.at - hours * 3600;
  const newest = res.at - MIN_AGE_S;
  const maxPages = Math.max(1, o.maxPages ?? 40);
  const maxIngest = Math.max(0, o.maxIngest ?? 100);
  const deadline = Date.now() + (o.budgetMs ?? 120_000);
  const prefixes = list(env.APP_PREFIXES);
  const rpc = rpcFor(env);
  const seen = new Set<string>();
  let start: { author: string; permlink: string } | undefined;

  try {
    pages: for (;;) {
      if (res.pages >= maxPages || Date.now() > deadline) {
        res.truncated = true;
        break;
      }
      const page = (await rpc.getDiscussionsByCreated(start, PAGE, 1)) ?? [];
      res.pages++;
      // Pixa's Hivemind starts the next page after `start`; a node that repeats it is handled too
      const fresh = page.filter((p) => p && !seen.has(key(p)));
      for (const p of fresh) seen.add(key(p));
      const inWindow = fresh.filter((p) => !p.parent_author && chainTime(p.created) >= since && chainTime(p.created) <= newest);
      res.listed += inWindow.length;

      const wanted: CondenserPost[] = [];
      for (const p of inWindow) {
        if (appAllowed(parseJsonMetadata(p.json_metadata).app ?? null, prefixes)) wanted.push(p);
        else res.skipped_app++;
      }
      if (wanted.length) {
        const rows = await env.DB
          .prepare(`SELECT author, permlink, updated FROM posts WHERE ${wanted.map(() => "(author = ? AND permlink = ?)").join(" OR ")}`)
          .bind(...wanted.flatMap((p) => [p.author, p.permlink]))
          .all<{ author: string; permlink: string; updated: number }>();
        const stored = new Map((rows.results ?? []).map((r) => [key(r), r.updated]));
        for (const p of wanted) {
          const have = stored.get(key(p));
          const onChain = chainTime(p.last_update ?? p.updated ?? p.created);
          if (have === undefined) res.missing++;
          else if (onChain > have) res.stale++;
          else continue;
          if (res.ingested + res.failed >= maxIngest || Date.now() > deadline) {
            res.truncated = true;
            break pages;
          }
          try {
            const r = await ingestPostRef(env, p.author, p.permlink, null, "reconcile");
            if (r.action === "missing") res.failed++;
            else {
              res.ingested++;
              if (res.recovered.length < KEEP_RECOVERED) res.recovered.push(key(p));
            }
          } catch (e) {
            res.failed++;
            console.warn("reconcile: could not ingest", key(p), e instanceof Error ? e.message : e);
          }
        }
      }
      // the end of the list, or of the window (the page is sorted newest first)
      if (page.length < PAGE || !fresh.length || fresh.some((p) => chainTime(p.created) < since)) break;
      const last = page[page.length - 1];
      start = { author: last.author, permlink: last.permlink };
    }
  } catch (e) {
    res.error = (e instanceof Error ? e.message : String(e)).slice(0, 240);
  }

  if (res.missing || res.stale) console.warn("reconcile: the index had missed posts the chain has", JSON.stringify(res));
  if (o.record ?? true) await recordRun(env, res).catch((e) => console.warn("reconcile: could not record the run", e));
  return res;
}

async function recordRun(env: Env, res: ReconcileResult): Promise<void> {
  const prev = await reconcileRecord(env);
  const next: ReconcileRecord = {
    last: res,
    recovered_total: (prev?.recovered_total ?? 0) + res.ingested,
    last_recovery: res.ingested ? { at: res.at, missing: res.missing, stale: res.stale, recovered: res.recovered } : (prev?.last_recovery ?? null),
  };
  await setSetting(env.DB, SETTING, JSON.stringify(next));
}

/** The recorded state of the reconciliation, or null before its first run. */
export async function reconcileRecord(env: Env): Promise<ReconcileRecord | null> {
  const v = await getSetting(env.DB, SETTING);
  if (!v) return null;
  try {
    return JSON.parse(v) as ReconcileRecord;
  } catch {
    return null;
  }
}
