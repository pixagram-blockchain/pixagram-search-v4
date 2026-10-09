// Keeps the documentation index (docs, doc_chunks + FTS, VEC_DOCS) in step with the repository.
//
//   cron every 10 min / GitHub webhook / POST /admin/docs/sync
//     ─► head commit of DOCS_BRANCH (git's ref advertisement: a few hundred bytes)
//     ─► unchanged since the last sync: nothing to do
//     ─► the commit's archive, streamed: its Markdown files ─► files whose content changed are
//        parsed and chunked; unchanged chunks keep their row and vector ─► files gone lose theirs
//     ─► chunks without a vector from DOCS_EMBED_MODEL are embedded (also after a model change)
//
// The commit is recorded only once every file of it is done. Until then the commit being indexed
// is kept in `docs:target`, so a sync cut short by its budget or an error continues on the next
// run, even if the branch moved back to the last recorded commit meanwhile. A forced re-index
// clears every file's content hash, so it too continues over as many runs as it takes. A lock in
// `settings` keeps two runs (cron, webhook) apart.

import type { Env } from "../env";
import { now } from "../env";
import { getSetting, setSetting } from "../db/posts";
import { blobUrl, fetchDocFiles, headCommit, repoName, repoRef, type ArchiveFile, type RepoRef } from "./github";
import { chunkDoc, parseDoc, type DocChunk, type ParsedDoc } from "./markdown";
import { embedPendingChunks } from "./vectors";

export interface SyncReport {
  repo: string | null;
  branch: string | null;
  status: "disabled" | "locked" | "unchanged" | "synced" | "partial" | "error";
  reason?: string;
  commit: string | null;
  previous: string | null;
  files: number;
  indexed: string[];
  removed: string[];
  skipped: Array<{ path: string; reason: string }>;
  failed: Array<{ path: string; error: string }>;
  /** files left for the next run (time or file budget) */
  pending: number;
  /** chunks that got a vector in this run */
  embedded: number;
  notes: string[];
  error?: string;
  took_ms: number;
}

export const DOCS_SETTINGS = { commit: "docs:commit", target: "docs:target", lock: "docs:lock", last: "docs:last" } as const;



const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e)).slice(0, 300);

async function acquireLock(env: Env, ttlSeconds: number): Promise<string | null> {
  const t = now();
  const token = `${t + ttlSeconds}.${Math.random().toString(36).slice(2, 10)}`;
  // Takes the lock when there is none or the holder's expired (CAST reads the leading number).
  const r = await env.DB.prepare("INSERT INTO settings (k, v) VALUES (?1, ?2) ON CONFLICT(k) DO UPDATE SET v = excluded.v WHERE CAST(settings.v AS INTEGER) < ?3")
    .bind(DOCS_SETTINGS.lock, token, t)
    .run();
  return r.meta?.changes ? token : null;
}

async function releaseLock(env: Env, token: string): Promise<void> {
  await env.DB.prepare("DELETE FROM settings WHERE k = ? AND v = ?").bind(DOCS_SETTINGS.lock, token).run().catch(() => {});
}

async function sha256Hex(s: string): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)));
  return [...d.slice(0, 12)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export const chunkHash = (title: string, c: Pick<DocChunk, "heading" | "anchor" | "text">) => sha256Hex(`${title}\u0001${c.heading}\u0001${c.anchor}\u0001${c.text}`);

/** Remove a file from the index (rows, FTS through the triggers, vectors). */
export async function removeDoc(env: Env, path: string): Promise<void> {
  const ids = ((await env.DB.prepare("SELECT id FROM doc_chunks WHERE path = ?").bind(path).all<{ id: number }>()).results ?? []).map((r) => r.id);
  await env.DB.batch([env.DB.prepare("DELETE FROM doc_chunks WHERE path = ?").bind(path), env.DB.prepare("DELETE FROM docs WHERE path = ?").bind(path)]);
  if (ids.length && env.VEC_DOCS) await env.VEC_DOCS.deleteByIds(ids.map(String)).catch(() => {});
}

/**
 * Replace a file's chunks. A chunk whose text (and title, heading, anchor) is unchanged keeps its
 * row, so its vector stays valid; the others are inserted without a vector (embedded = NULL).
 */
export async function replaceChunks(env: Env, path: string, doc: ParsedDoc, chunks: DocChunk[]): Promise<{ kept: number; inserted: number; removed: number }> {
  const rows = (await env.DB.prepare("SELECT id, hash FROM doc_chunks WHERE path = ? ORDER BY ord").bind(path).all<{ id: number; hash: string }>()).results ?? [];
  const byHash = new Map<string, number[]>();
  for (const r of rows) byHash.set(r.hash, [...(byHash.get(r.hash) ?? []), r.id]);
  const stmts: D1PreparedStatement[] = [];
  const kept = new Set<number>();
  let inserted = 0;
  for (const c of chunks) {
    const hash = await chunkHash(doc.title, c);
    const reuse = byHash.get(hash)?.shift();
    if (reuse !== undefined) {
      kept.add(reuse);
      stmts.push(env.DB.prepare("UPDATE doc_chunks SET ord = ?, lang = ? WHERE id = ?").bind(c.ord, doc.lang, reuse));
    } else {
      inserted++;
      stmts.push(
        env.DB.prepare("INSERT INTO doc_chunks (path, ord, title, heading, anchor, text, lang, hash, embedded) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)").bind(
          path,
          c.ord,
          doc.title,
          c.heading,
          c.anchor,
          c.text,
          doc.lang,
          hash,
        ),
      );
    }
  }
  const obsolete = rows.map((r) => r.id).filter((id) => !kept.has(id));
  if (obsolete.length) stmts.unshift(env.DB.prepare(`DELETE FROM doc_chunks WHERE id IN (${obsolete.map((id) => Math.trunc(id)).join(",")})`));
  for (let i = 0; i < stmts.length; i += 100) await env.DB.batch(stmts.slice(i, i + 100));
  // Ids are never reused (AUTOINCREMENT), so a vector whose delete is still in flight can only
  // point at a missing row, and the reader drops those.
  if (obsolete.length && env.VEC_DOCS) await env.VEC_DOCS.deleteByIds(obsolete.map(String)).catch(() => {});
  return { kept: kept.size, inserted, removed: obsolete.length };
}

async function upsertDoc(
  env: Env,
  d: { path: string; sha: string; title: string; lang: string | null; url: string; chunks: number; status: "indexed" | "skipped" | "failed"; error: string | null },
): Promise<void> {
  await env.DB.prepare(
    `INSERT INTO docs (path, sha, title, lang, url, chunks, status, error, attempts, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(path) DO UPDATE SET sha = excluded.sha, title = excluded.title, lang = excluded.lang, url = excluded.url, chunks = excluded.chunks,
       status = excluded.status, error = excluded.error, attempts = CASE WHEN excluded.status = 'failed' THEN docs.attempts + 1 ELSE 0 END, updated = excluded.updated`,
  )
    .bind(d.path, d.sha, d.title, d.lang, d.url, d.chunks, d.status, d.error, d.status === "failed" ? 1 : 0, now())
    .run();
}

/** Index one documentation file of the archive. */
export async function indexArchiveFile(env: Env, ref: RepoRef, f: ArchiveFile): Promise<{ chunks: number; skipped?: string }> {
  if (f.text === null) {
    const reason = f.size > 512 * 1024 ? "larger than 512 KB" : "not UTF-8 text";
    const title = f.path;
    await replaceChunks(env, f.path, { title, lang: "en", meta: { title: null, lang: null, description: null, keywords: [], skip: true }, sections: [] }, []);
    await upsertDoc(env, { path: f.path, sha: f.hash, title, lang: null, url: blobUrl(ref, f.path), chunks: 0, status: "skipped", error: reason });
    return { chunks: 0, skipped: reason };
  }
  return indexDocText(env, ref, { path: f.path, sha: f.hash }, f.text);
}

/** Index a file's text (also used by tests and by a manual upload of a single page). */
export async function indexDocText(env: Env, ref: RepoRef, f: { path: string; sha: string }, text: string): Promise<{ chunks: number; skipped?: string }> {
  const doc = parseDoc(text, f.path);
  const url = blobUrl(ref, f.path);
  if (doc.meta.skip) {
    await replaceChunks(env, f.path, doc, []);
    await upsertDoc(env, { path: f.path, sha: f.sha, title: doc.title, lang: doc.lang, url, chunks: 0, status: "skipped", error: "draft or noindex" });
    return { chunks: 0, skipped: "draft or noindex" };
  }
  const chunks = chunkDoc(doc);
  await replaceChunks(env, f.path, doc, chunks);
  await upsertDoc(env, { path: f.path, sha: f.sha, title: doc.title, lang: doc.lang, url, chunks: chunks.length, status: "indexed", error: null });
  return { chunks: chunks.length };
}

export interface SyncOptions {
  /** re-index every file even when its content is unchanged (after a chunking change); vectors of unchanged chunks are kept */
  force?: boolean;
  /** the head commit when the caller knows it (a push webhook's "after") */
  commit?: string;
  reason?: string;
  /** stop starting new files after this long (default 25 s; the cron passes more, a webhook less) */
  budgetMs?: number;
  maxFiles?: number;
}

export async function syncDocs(env: Env, opts: SyncOptions = {}): Promise<SyncReport> {
  const t0 = Date.now();
  const ref = repoRef(env);
  const report: SyncReport = {
    repo: ref ? repoName(ref) : null,
    branch: ref?.branch ?? null,
    status: "disabled",
    reason: opts.reason,
    commit: null,
    previous: null,
    files: 0,
    indexed: [],
    removed: [],
    skipped: [],
    failed: [],
    pending: 0,
    embedded: 0,
    notes: [],
    took_ms: 0,
  };
  if (!ref) return report;
  const budget = opts.budgetMs ?? 25_000;
  const lock = await acquireLock(env, Math.ceil(budget / 1000) + 120);
  if (!lock) return { ...report, status: "locked", took_ms: Date.now() - t0 };
  try {
    const previous = await getSetting(env.DB, DOCS_SETTINGS.commit);
    report.previous = previous;
    const commit = opts.commit && /^[0-9a-f]{40}$/.test(opts.commit) ? opts.commit : await headCommit(env, ref);
    report.commit = commit;
    // unfinished work (of this commit or an earlier one) keeps the sync going
    const inProgress = await getSetting(env.DB, DOCS_SETTINGS.target);
    if (!opts.force && !inProgress && commit === previous) {
      report.status = "unchanged";
    } else {
      await setSetting(env.DB, DOCS_SETTINGS.target, commit);
      // forced: every file counts as changed until it is redone (its chunks stay searchable meanwhile)
      if (opts.force) await env.DB.prepare("UPDATE docs SET sha = ''").run();
      const files = await fetchDocFiles(env, ref, commit);
      report.files = files.length;
      const existing = new Map(
        ((await env.DB.prepare("SELECT path, sha, status FROM docs").all<{ path: string; sha: string; status: string }>()).results ?? []).map((r) => [r.path, r]),
      );
      const inRepo = new Set(files.map((f) => f.path));
      for (const path of existing.keys()) {
        if (inRepo.has(path)) continue;
        await removeDoc(env, path);
        report.removed.push(path);
      }
      const todo = files.filter((f) => {
        const e = existing.get(f.path);
        return !e || e.sha !== f.hash || e.status === "failed";
      });
      const maxFiles = opts.maxFiles ?? 200;
      let started = 0;
      for (const f of todo) {
        if (started >= maxFiles || Date.now() - t0 > budget) break;
        started++;
        try {
          const r = await indexArchiveFile(env, ref, f);
          if (r.skipped) report.skipped.push({ path: f.path, reason: r.skipped });
          else report.indexed.push(f.path);
        } catch (e) {
          report.failed.push({ path: f.path, error: errMsg(e) });
          // Keeps its previous chunks (if any) and is retried on the next run.
          await env.DB.prepare(
            `INSERT INTO docs (path, sha, title, lang, url, chunks, status, error, attempts, updated) VALUES (?, '', ?, NULL, ?, 0, 'failed', ?, 1, ?)
             ON CONFLICT(path) DO UPDATE SET status = 'failed', error = excluded.error, attempts = docs.attempts + 1, updated = excluded.updated`,
          )
            .bind(f.path, f.path, blobUrl(ref, f.path), errMsg(e), now())
            .run()
            .catch(() => {});
        }
      }
      report.pending = todo.length - started;
      const complete = report.pending === 0 && report.failed.length === 0;
      if (complete) {
        await setSetting(env.DB, DOCS_SETTINGS.commit, commit);
        await env.DB.prepare("DELETE FROM settings WHERE k = ?").bind(DOCS_SETTINGS.target).run();
      }
      report.status = complete ? "synced" : "partial";
    }
    // Vectors: new chunks, chunks whose embedding failed earlier, everything after a model change.
    // (Half the budget again: a webhook-triggered run must end within waitUntil's 30 s.)
    try {
      report.embedded = await embedPendingChunks(env, 400, t0 + Math.round(budget * 1.5));
    } catch (e) {
      report.notes.push(`embedding deferred: ${errMsg(e)}`);
    }
  } catch (e) {
    report.status = "error";
    report.error = errMsg(e);
  } finally {
    await releaseLock(env, lock);
    report.took_ms = Date.now() - t0;
    const { indexed, removed, skipped, failed, ...rest } = report;
    await setSetting(
      env.DB,
      DOCS_SETTINGS.last,
      JSON.stringify({ ...rest, indexed: indexed.length, removed: removed.length, skipped: skipped.length, failed: failed.slice(0, 20), at: now() }),
    ).catch(() => {});
  }
  return report;
}

/** The index as it stands: what /admin/docs shows and what /help and the router consult. */
export async function docsStatus(env: Env): Promise<{
  repo: string | null;
  branch: string | null;
  commit: string | null;
  /** the commit a sync is still working on (cut short, failed files, a forced re-index) */
  in_progress: string | null;
  docs: Array<{ status: string; n: number }>;
  chunks: number;
  embedded: number;
  failed: Array<{ path: string; error: string | null; attempts: number }>;
  last_sync: unknown;
}> {
  const ref = repoRef(env);
  const [byStatus, chunks, failed, commit, last, inProgress] = await Promise.all([
    env.DB.prepare("SELECT status, COUNT(*) AS n FROM docs GROUP BY status").all<{ status: string; n: number }>(),
    env.DB.prepare("SELECT COUNT(*) AS n, SUM(embedded IS NOT NULL) AS e FROM doc_chunks").first<{ n: number; e: number | null }>(),
    env.DB.prepare("SELECT path, error, attempts FROM docs WHERE status = 'failed' ORDER BY updated DESC LIMIT 50").all<{ path: string; error: string | null; attempts: number }>(),
    getSetting(env.DB, DOCS_SETTINGS.commit),
    getSetting(env.DB, DOCS_SETTINGS.last),
    getSetting(env.DB, DOCS_SETTINGS.target),
  ]);
  return {
    repo: ref ? repoName(ref) : null,
    branch: ref?.branch ?? null,
    commit,
    in_progress: inProgress,
    docs: byStatus.results ?? [],
    chunks: chunks?.n ?? 0,
    embedded: chunks?.e ?? 0,
    failed: failed.results ?? [],
    last_sync: last ? JSON.parse(last) : null,
  };
}
