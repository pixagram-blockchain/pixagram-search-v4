// Reconciliation: the tail's safety net. Hivemind lists the newest posts chain-wide
// (condenser_api.get_discussions_by_created); every post it lists that the index lacks, holds in an
// older state, or holds as deleted while the chain shows it, is ingested now, as the tail would have.
//
// Until v4.8.3 the tail dropped any post Hivemind had not processed yet when the tail asked
// (src/chain/indexer-do.ts), and the only cure was a backfill run by hand every few days. The tail
// now waits for Hivemind instead; this pass catches whatever else slips through (a Hivemind
// outage longer than the tail waits, a parked or deferred post dropped, a replay that skipped
// blocks): every ten minutes over the last RECONCILE_HOURS (3), nightly over RECONCILE_DEEP_HOURS
// (48). A page is 20 posts with their bodies truncated (about 40 KB with the votes); three hours
// are one or two pages.
//
// v4.8.4: and a rolling pass over the whole listing (reconcileFull): RECONCILE_FULL_PAGES pages
// per ten-minute run, newest to oldest, then again from the newest. Whatever the index lacks, of
// any age, comes back within one pass (hours), so no gap waits for someone to notice it and start
// a backfill: the v4.8.3 fixes were lost in a merge on 16 November 2026, and the posts the tail
// dropped from then on, with every Pixtopia post, stayed out of the index until a pass like this.
// It also brings back posts the index marked deleted while the chain still shows them (the old
// tail did that when a lagging Hivemind answered "does not exist" about an edited post).

import type { Env } from "../env";
import { int, list, now, num } from "../env";
import { chainTime, missingReason, type CondenserPost } from "./rpc";
import { appAllowed, isDeletedBody, parseJsonMetadata } from "./parse";
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
  /** v4.8.4: in the index as deleted, but shown on chain (not deleted by an edit) */
  revived: number;
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
  last_recovery: { at: number; missing: number; stale: number; revived?: number; recovered: string[] } | null;
}

const PAGE = 20;
/**
 * Characters of each body the listing returns: enough to tell a post deleted by an edit (its body
 * is the word "deleted") from one that is shown, and to keep an artwork's data URI out of the page.
 */
export const LIST_TRUNCATE = 16;
/** Posts younger than this are the tail's: it may be asking Hivemind about them right now. */
const MIN_AGE_S = 300;
const SETTING = "reconcile:last";
const KEEP_RECOVERED = 20;

const key = (p: { author: string; permlink: string }) => `${p.author}/${p.permlink}`;

export type IngestReason = "missing" | "stale" | "revived";

/**
 * Of the posts a listing shows, those the index needs: not in it (unless the chain shows them
 * deleted by an edit: nothing to show), in an older state, or marked deleted while the chain shows
 * them. One query for up to 20 posts (40 bound values).
 */
export async function postsToIngest(env: Env, posts: CondenserPost[]): Promise<Array<{ post: CondenserPost; why: IngestReason }>> {
  if (!posts.length) return [];
  const rows = await env.DB
    .prepare(`SELECT author, permlink, updated, deleted FROM posts WHERE ${posts.map(() => "(author = ? AND permlink = ?)").join(" OR ")}`)
    .bind(...posts.flatMap((p) => [p.author, p.permlink]))
    .all<{ author: string; permlink: string; updated: number; deleted: number }>();
  const stored = new Map((rows.results ?? []).map((r) => [key(r), r]));
  const out: Array<{ post: CondenserPost; why: IngestReason }> = [];
  for (const p of posts) {
    const have = stored.get(key(p));
    const shownDeleted = typeof p.body === "string" && isDeletedBody(p.body);
    const onChain = chainTime(p.last_update ?? p.updated ?? p.created);
    if (!have) {
      if (!shownDeleted) out.push({ post: p, why: "missing" });
    } else if (onChain > have.updated) out.push({ post: p, why: "stale" });
    else if (have.deleted === 1 && !shownDeleted) out.push({ post: p, why: "revived" });
  }
  return out;
}

export async function reconcileRecent(
  env: Env,
  o: { hours?: number; maxPages?: number; maxIngest?: number; budgetMs?: number; record?: boolean } = {},
): Promise<ReconcileResult> {
  const hours = o.hours ?? num(env.RECONCILE_HOURS, 3);
  const res: ReconcileResult = { at: now(), hours, pages: 0, listed: 0, skipped_app: 0, missing: 0, stale: 0, revived: 0, ingested: 0, failed: 0, recovered: [], truncated: false };
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
      const page = (await rpc.getDiscussionsByCreated(start, PAGE, LIST_TRUNCATE)) ?? [];
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
      for (const { post: p, why } of await postsToIngest(env, wanted)) {
        if (res.ingested + res.failed >= maxIngest || Date.now() > deadline) {
          res.truncated = true;
          break pages;
        }
        res[why]++;
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
      // the end of the list, or of the window (the page is sorted newest first)
      if (page.length < PAGE || !fresh.length || fresh.some((p) => chainTime(p.created) < since)) break;
      const last = page[page.length - 1];
      start = { author: last.author, permlink: last.permlink };
    }
  } catch (e) {
    res.error = (e instanceof Error ? e.message : String(e)).slice(0, 240);
  }

  if (res.missing || res.stale || res.revived) console.warn("reconcile: the index had missed posts the chain has", JSON.stringify(res));
  if (o.record ?? true) await recordRun(env, res).catch((e) => console.warn("reconcile: could not record the run", e));
  return res;
}

async function recordRun(env: Env, res: ReconcileResult): Promise<void> {
  const prev = await reconcileRecord(env);
  const next: ReconcileRecord = {
    last: res,
    recovered_total: (prev?.recovered_total ?? 0) + res.ingested,
    last_recovery: res.ingested ? { at: res.at, missing: res.missing, stale: res.stale, revived: res.revived, recovered: res.recovered } : (prev?.last_recovery ?? null),
  };
  await setSetting(env.DB, SETTING, JSON.stringify(next));
}

/** The recorded state of the reconciliation, or null before its first run. */
export async function reconcileRecord(env: Env): Promise<ReconcileRecord | null> {
  return readJson<ReconcileRecord>(env, SETTING);
}

async function readJson<T>(env: Env, k: string): Promise<T | null> {
  const v = await getSetting(env.DB, k);
  if (!v) return null;
  try {
    return JSON.parse(v) as T;
  } catch {
    return null;
  }
}

// ---- v4.8.4: the rolling pass over the whole listing -------------------------------------------

const FULL_SETTING = "reconcile:full";

/** A pass in progress: where it goes on from, and what it found so far. */
export interface FullPass {
  /** the last post of the last page read (the next page starts after it); null: from the newest */
  start: { author: string; permlink: string; created: string } | null;
  started_at: number;
  pages: number;
  /** top-level posts examined (the newest five minutes are the tail's) */
  listed: number;
  skipped_app: number;
  missing: number;
  stale: number;
  revived: number;
  ingested: number;
  failed: number;
  /** the first posts brought in */
  recovered: string[];
}

export interface FullRecord {
  current: FullPass;
  /** the last pass that reached the oldest post Hivemind lists */
  last_pass: (FullPass & { finished_at: number }) | null;
  passes: number;
  /** posts the rolling pass brought in, in all */
  recovered_total: number;
  last_run: { at: number; pages: number; ingested: number; error: string | null };
  /** runs in a row that failed on the page the pass goes on from; at FULL_RESTART_AFTER the pass starts over */
  failed_runs?: number;
}

/** Runs in a row failing on the same page before the pass starts over from the newest (whatever the node says). */
const FULL_RESTART_AFTER = 6;

export interface FullRunResult {
  at: number;
  pages: number;
  listed: number;
  missing: number;
  stale: number;
  revived: number;
  ingested: number;
  failed: number;
  recovered: string[];
  /** this run reached the oldest post: the next run starts a new pass from the newest */
  pass_done: boolean;
  /** the post the pass went on from is gone from the listing (deleted): the pass started over */
  restarted?: boolean;
  error?: string;
}

const newPass = (): FullPass => ({ start: null, started_at: now(), pages: 0, listed: 0, skipped_app: 0, missing: 0, stale: 0, revived: 0, ingested: 0, failed: 0, recovered: [] });

/** The rolling pass's record, or null before its first run. */
export async function fullRecord(env: Env): Promise<FullRecord | null> {
  return readJson<FullRecord>(env, FULL_SETTING);
}

/**
 * One run of the rolling pass: up to `pages` pages from where the last run stopped. The budgets
 * (`maxIngest` posts, `budgetMs`) are checked between pages, so a page is always finished and the
 * cursor only ever moves past pages done with; a run that fails on a page leaves the cursor where
 * it was, and the next run reads that page again (what was ingested is then current, and skipped).
 * Returns null when RECONCILE_FULL_PAGES is 0.
 */
export async function reconcileFull(env: Env, o: { pages?: number; maxIngest?: number; budgetMs?: number } = {}): Promise<FullRunResult | null> {
  const pagesPerRun = o.pages ?? int(env.RECONCILE_FULL_PAGES, 10);
  if (!(pagesPerRun > 0)) return null;
  const maxIngest = Math.max(0, o.maxIngest ?? int(env.RECONCILE_FULL_MAX_INGEST, 100));
  const deadline = Date.now() + (o.budgetMs ?? 150_000);
  const prefixes = list(env.APP_PREFIXES);
  const rpc = rpcFor(env);
  const rec: FullRecord = (await fullRecord(env)) ?? { current: newPass(), last_pass: null, passes: 0, recovered_total: 0, last_run: { at: 0, pages: 0, ingested: 0, error: null } };
  const run: FullRunResult = { at: now(), pages: 0, listed: 0, missing: 0, stale: 0, revived: 0, ingested: 0, failed: 0, recovered: [], pass_done: false };
  const startedFrom = JSON.stringify(rec.current.start);

  try {
    while (run.pages < pagesPerRun && Date.now() < deadline && run.ingested + run.failed < maxIngest) {
      const cur = rec.current;
      const from = cur.start ? { author: cur.start.author, permlink: cur.start.permlink } : undefined;
      let page: CondenserPost[];
      try {
        page = (await rpc.getDiscussionsByCreated(from, PAGE, LIST_TRUNCATE)) ?? [];
      } catch (e) {
        // no page starts after a post the listing no longer holds: start the pass over
        if (from && missingReason(e, from.author, from.permlink)) {
          console.warn("reconcile (full pass): the post it went on from is gone; starting over from the newest", key(from));
          rec.current = newPass();
          run.restarted = true;
          break;
        }
        throw e;
      }
      run.pages++;
      // a node that repeats the start post at the top of the page (Hive's condenser does)
      const fresh = page.filter((p) => p && !(from && key(p) === key(from)));
      // the listing is newest first: a page after `start` holds nothing newer than it. A node that
      // ignored the start would have the pass read its first page for ever.
      if (cur.start && fresh.some((p) => p.created > cur.start!.created)) {
        throw new Error(`the node listed posts newer than ${key(cur.start)} after it (start_author ignored?)`);
      }
      const youngest = now() - MIN_AGE_S;
      const examined = fresh.filter((p) => !p.parent_author && chainTime(p.created) <= youngest);
      const wanted: CondenserPost[] = [];
      let skippedApp = 0;
      for (const p of examined) {
        if (appAllowed(parseJsonMetadata(p.json_metadata).app ?? null, prefixes)) wanted.push(p);
        else skippedApp++;
      }
      const todo = await postsToIngest(env, wanted);
      // the page counts once it is read and compared (a run that fails here reads it again)
      cur.pages++;
      run.listed += examined.length;
      cur.listed += examined.length;
      cur.skipped_app += skippedApp;
      for (const { post: p, why } of todo) {
        run[why]++;
        cur[why]++;
        try {
          const r = await ingestPostRef(env, p.author, p.permlink, null, "reconcile-full");
          if (r.action === "missing") {
            run.failed++;
            cur.failed++;
          } else {
            run.ingested++;
            cur.ingested++;
            if (run.recovered.length < KEEP_RECOVERED) run.recovered.push(key(p));
            if (cur.recovered.length < KEEP_RECOVERED) cur.recovered.push(key(p));
          }
        } catch (e) {
          run.failed++;
          cur.failed++;
          console.warn("reconcile (full pass): could not ingest", key(p), e instanceof Error ? e.message : e);
        }
      }
      if (page.length < PAGE || !fresh.length) {
        // the oldest post: this pass is complete, the next run starts another from the newest
        rec.last_pass = { ...cur, finished_at: now() };
        rec.passes++;
        rec.current = newPass();
        run.pass_done = true;
        break;
      }
      const last = page[page.length - 1];
      cur.start = { author: last.author, permlink: last.permlink, created: last.created };
    }
  } catch (e) {
    run.error = (e instanceof Error ? e.message : String(e)).slice(0, 240);
  }

  // A page that fails run after run (a node refusing the cursor in some way of its own) must not
  // hold the pass there: after FULL_RESTART_AFTER runs (an hour) it starts over from the newest.
  const moved = run.pass_done || !!run.restarted || JSON.stringify(rec.current.start) !== startedFrom;
  if (run.error && !moved) {
    rec.failed_runs = (rec.failed_runs ?? 0) + 1;
    if (rec.failed_runs >= FULL_RESTART_AFTER && rec.current.start) {
      console.error("reconcile (full pass): the page after", key(rec.current.start), `failed ${rec.failed_runs} runs in a row; starting over from the newest`, run.error);
      rec.current = newPass();
      rec.failed_runs = 0;
      run.restarted = true;
    }
  } else rec.failed_runs = 0;
  rec.recovered_total += run.ingested;
  rec.last_run = { at: run.at, pages: run.pages, ingested: run.ingested, error: run.error ?? null };
  await setSetting(env.DB, FULL_SETTING, JSON.stringify(rec)).catch((e) => console.warn("reconcile (full pass): could not record the run", e));
  if (run.missing || run.stale || run.revived) console.warn("reconcile (full pass): the index had missed posts the chain has", JSON.stringify(run));
  return run;
}
