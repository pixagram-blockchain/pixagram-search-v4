// Copy detection, Worker side.
//
//   enrich  (stage "paph", after stats)
//           decode ─► PAPH hash (5–60 ms of WASM; or the stored wires of the same image) ─►
//           home shard: store + index ─► every shard at once: nominate (keys, PAPH-SI, pHash and
//           embedding neighbours, previous partners) ─► XRank ─► verdicts ─► D1 paph_matches
//   API     /copies/:id           stored verdicts (D1 only: milliseconds)
//           /copies/:id?live=1    re-checked now (admin)
//           /copies-by-image      an upload: hashed here, checked in every shard under the
//                                 interactive budget (sub-second), the answer kept for a day
//           /copies/:a/report/:b  the full PAPH-X report with comparator 42's beside it (audit),
//                                 optionally with both wires, kept for a day
//
// What a verdict means is fixed by `identity` (engine.ts): it is stored with each verdict, with
// the stage's mark of each artwork (artworks.paph_engine) and in every cache key, so a new release,
// profile or policy never passes for the one that reached a result — and the stage re-checks
// every work whose mark names another.

import type { Env, Stage } from "../env";
import { int, now } from "../env";
import type { RgbaImage } from "../enrich/decode";
import { embeddingEnabled } from "../enrich/embed";
import { phash as phashOf } from "../enrich/phash";
import { getSetting, setJobsMany, setSetting, vectorId } from "../db/posts";
import { phashNeighbours } from "../search/visual";
import { knn } from "../search/vectors";
import { hydrateOrdered, type SearchItem } from "../search/service";
import { emptyRequest, type SearchRequest } from "../search/params";
import { base64Encode, sha256Hex } from "../lib/bytes";
import { budgetFor, budgetLabel } from "./budget";
import { DEFAULT_MAX_PIXELS, STATE, STATES, WIRE_3, WIRE_VERSION, fingerprint, onWire, paphRuntime, stateOf, verdictIdentity, wireFormat, type Policy } from "./engine";
import { fanOut, findEverywhere, homeShard, shardCount, shardOf, shardSize, shardStub, type MergedFind, type ShardStub } from "./shards";
import type { Candidate, Checked, Match, PutResult, WireInput, WorkInfo } from "./shard-store";

// ---- settings ---------------------------------------------------------------------------------

export function policyOf(env: Pick<Env, "PAPH_POLICY">): Policy {
  const p = (env.PAPH_POLICY ?? "safe").trim().toLowerCase();
  return p === "exact" || p === "fast" ? p : "safe";
}
const maxPixels = (env: Env): number => Math.max(64 * 64, int(env.PAPH_MAX_PIXELS, DEFAULT_MAX_PIXELS));
const channelK = (env: Env): number => Math.min(64, Math.max(0, int(env.PAPH_CHANNEL_K, 24)));
const phashDistance = (env: Env): number => Math.min(15, Math.max(0, int(env.PAPH_PHASH_DISTANCE, 10)));
const storedMin = (env: Env): number => Math.max(STATE.Suspected, stateOf(env.PAPH_MIN_VERDICT, STATE.Suspected));
const queryMs = (env: Env): number => Math.min(30_000, Math.max(300, int(env.PAPH_QUERY_MS, 900)));
const cacheTtl = (env: Env): number => Math.max(60, int(env.PAPH_CACHE_TTL, 86_400));
/** An answer some shard (or channel) did not complete is kept for ten minutes, not a day. */
const PARTIAL_TTL = 600;

/** A public threshold: Suspected at the lowest (nothing below is stored, nor worth listing). */
export function minStateOf(v: string | undefined | null): number {
  return Math.max(STATE.Suspected, stateOf(v, STATE.Copy));
}

/** The identity verdicts reached now carry (engine, profiles, policy). */
export async function currentEngine(env: Env): Promise<string> {
  return verdictIdentity(await paphRuntime(), policyOf(env));
}

// ---- wire formats while the store moves to wire 4 (PAPH-X 1.2.0, SPEC-W4 §9) -------------------

/**
 * Whether some shard still holds works hashed before 1.2.0 (wire 3, waiting for their re-hash),
 * as the last check that heard from every shard found; null until one has (a new isolate). While
 * it may, a query is hashed in both formats — its wire-3 twin is compared with those works, which
 * would otherwise go unread — at the cost of a second hash (5–60 ms); once every shard has said it
 * holds none, in the current format only. Nothing writes wire 3 any more, so that is final.
 */
let legacyStore: boolean | null = null;

/** A query should bring its wire-3 twin (see legacyStore). */
export function storeMayHoldLegacy(): boolean {
  return legacyStore !== false;
}

/** What a check's answers say about the shards' formats: one that did not hear from every shard proves no absence. */
function noteLegacy(res: MergedFind, everyShard: boolean): void {
  if (res.stats.legacy > 0) legacyStore = true;
  else if (everyShard && res.shards.answered === res.shards.asked) legacyStore = false;
}

/** Forget what the checks found (tests). */
export function resetLegacyStore(): void {
  legacyStore = null;
}

/** An image's wire-3 twin, or null when it cannot be hashed so (the works on wire 3 then wait for their own re-check). */
async function twinOf(env: Env, img: RgbaImage): Promise<WireInput | null> {
  try {
    const fp = await fingerprint(img, maxPixels(env), WIRE_3);
    return { t1: fp.t1, t2: fp.t2 };
  } catch (e) {
    console.warn("paph: no wire-3 twin", e instanceof Error ? e.message : e);
    return null;
  }
}

// ---- candidates from outside the index --------------------------------------------------------

/** Candidates from one channel; `failed`: the channel was unavailable (not merely empty). */
export interface Channel {
  list: Candidate[];
  failed: boolean;
}

const none: Channel = { list: [], failed: false };

/** Works whose pHash is within PAPH_PHASH_DISTANCE (the band index of /duplicates), nearest first. */
export async function phashCandidates(env: Env, hash: string | null, excludeId?: number): Promise<Channel> {
  const d = phashDistance(env), k = channelK(env);
  if (!hash || !/^[0-9a-f]{16}$/.test(hash) || d <= 0 || k <= 0) return none;
  try {
    const { hits } = await phashNeighbours(env, hash, d, k, excludeId);
    return { list: hits.map((h, i) => ({ id: h.id, via: "phash", rank: i + 1 })), failed: false };
  } catch (e) {
    console.warn("paph: pHash neighbours unavailable", e instanceof Error ? e.message : e);
    return { list: [], failed: true };
  }
}

/** Nearest neighbours of an image embedding (Vectorize). */
export async function vectorCandidates(env: Env, vector: number[] | null, excludeId?: number): Promise<Channel> {
  const k = channelK(env);
  if (!vector || !vector.length || k <= 0) return none;
  try {
    const hits = await knn(env, "image", vector, null, k + 1);
    return { list: hits.filter((h) => h.id !== excludeId).slice(0, k).map((h, i) => ({ id: h.id, via: "vector", rank: i + 1 })), failed: false };
  } catch (e) {
    console.warn("paph: vector neighbours unavailable", e instanceof Error ? e.message : e);
    return { list: [], failed: true };
  }
}

/** The neighbours of a stored work's own image vector, when it has one. */
export async function storedVectorCandidates(env: Env, postId: number): Promise<Channel> {
  if (!embeddingEnabled(env)) return none;
  let values: number[] | null;
  try {
    const v = await env.VEC.getByIds([vectorId(postId)]);
    values = v[0]?.values ? Array.from(v[0].values as ArrayLike<number>) : null;
  } catch (e) {
    console.warn("paph: stored vector unavailable", postId, e instanceof Error ? e.message : e);
    return { list: [], failed: true };
  }
  return vectorCandidates(env, values, postId);
}

/**
 * Partners a work already has verdicts with, strongest first: re-checked first when the work is
 * checked again for the same image, so their verdicts are refreshed. A few hundred at most.
 */
export async function previousPartners(db: D1Database, postId: number, limit = 256): Promise<Candidate[]> {
  const r = await db
    .prepare("SELECT CASE WHEN a = ?1 THEN b ELSE a END AS id FROM paph_matches WHERE a = ?1 OR b = ?1 ORDER BY state DESC, structural_hi DESC LIMIT ?2")
    .bind(postId, limit)
    .all<{ id: number }>();
  return (r.results ?? []).map((x) => ({ id: x.id, via: "previous", rank: 0 }));
}

// ---- verdicts in D1 -----------------------------------------------------------------------------

interface PostMeta {
  id: number;
  created: number;
  author: string;
}

/** x was published before y: chain time, then id. */
export function earlier(x: PostMeta, y: PostMeta): boolean {
  return x.created < y.created || (x.created === y.created && x.id < y.id);
}

async function postMeta(db: D1Database, ids: number[]): Promise<Map<number, PostMeta>> {
  if (!ids.length) return new Map();
  const r = await db.prepare("SELECT id, created, author FROM posts WHERE id IN (SELECT value FROM json_each(?))").bind(JSON.stringify([...new Set(ids)])).all<PostMeta>();
  return new Map((r.results ?? []).map((x) => [x.id, x]));
}

/** The artwork `?N` is live and shows the image `?M` (an SQL condition over two parameters). */
const LIVE_IMAGE = (id: string, hash: string) =>
  `EXISTS (SELECT 1 FROM artworks w JOIN posts p ON p.id = w.post_id
           WHERE w.post_id = ${id} AND w.content_hash = ${hash} AND p.deleted = 0 AND p.type = 'artwork')`;

/**
 * The stage's bookkeeping, written in the same batch as the verdicts. `basis`: the token of the
 * progress row the run continued from (it skipped the shards that row lists), or null for a run
 * that asked every shard. `progress`: what a retry may skip — the shards whose verdicts are now
 * stored — or null when the check completed (the row goes).
 */
export interface StageWrite {
  basis: string | null;
  progress: { token: string; engine: string; shardSize: number; done: number[] } | null;
}

/**
 * Write what one check of `postId` (showing `contentHash`) concluded, atomically:
 *   - the verdicts of the pairs it compared are replaced — a pair that no longer holds disappears —
 *     but only while this work still shows the image it was checked for, and only for partners
 *     still showing the image it compared: a check that raced an edit can neither erase the
 *     verdicts a newer check established nor write any;
 *   - pairs it did not compare are left alone (another work's check established them, or the
 *     screen merely rejected them);
 *   - a verdict is written only while both posts are live artworks showing those images;
 *   - with `mark`, artworks.paph_hash / paph_engine record the stage as done for this image under
 *     that identity (only while the work is live and still shows it);
 *   - with `stage`, the progress row is replaced under the same condition, or removed. A run
 *     that continued from a progress row writes the mark and the row only while that row is still
 *     there: deleting the post (or a run that started over) removed it, with the verdicts it
 *     vouched for, and the work is then checked in full again.
 *   - a verdict reached on wire 3 (a stored work not re-hashed since 1.2.0, through the query's
 *     twin) never replaces or erases one reached on the current wire;
 * One batch of at most six statements, whatever the number of verdicts. Returns how many
 * verdicts were written, and whether the mark and the progress row were.
 */
export async function writeCheck(
  db: D1Database,
  postId: number,
  contentHash: string,
  res: { matches: Match[]; checked: Checked[]; identity: string | null },
  mark: string | null,
  stage?: StageWrite,
): Promise<{ stored: number; marked: boolean; recorded: boolean }> {
  const meta = await postMeta(db, [postId, ...res.matches.map((m) => m.id)]);
  const me = meta.get(postId);
  // a pair compared on wire 3 (through the query's twin: the stored side is not re-hashed yet)
  // is provisional: its verdict says so, and it never replaces or erases one reached on the
  // current wire (two checks of the pair in flight at once: the stored side's own re-check may
  // have written that one meanwhile). The stored side's re-check replaces it on the current wire,
  // or the clean-up does (paphHealWire3) when that re-check ran before it was written.
  const legacyWire = (w: number | undefined) => w !== undefined && w !== WIRE_VERSION;
  const rows: Array<Record<string, unknown>> = [];
  const legacyRows: Array<Record<string, unknown>> = [];
  for (const m of res.matches) {
    const other = meta.get(m.id);
    if (!me || !other || m.id === postId) continue;
    const meFirst = earlier(me, other);
    (legacyWire(m.wire) ? legacyRows : rows).push({
      a: meFirst ? me.id : other.id, b: meFirst ? other.id : me.id,
      a_hash: meFirst ? contentHash : m.contentHash, b_hash: meFirst ? m.contentHash : contentHash,
      other: other.id, other_hash: m.contentHash, verdict: m.verdict, state: m.state,
      certifiable: m.certifiable ? 1 : 0, certificate: m.certificate ? 1 : 0, lo: m.structuralLo, hi: m.structuralHi,
      geometry: m.geometry, inliers: m.inliers, mirrored: m.mirrored ? 1 : 0, execution: m.execution,
      same_author: me.author === other.author ? 1 : 0, via: m.via.join(","),
      engine: res.identity ? onWire(res.identity, m.wire ?? WIRE_VERSION) : null,
    });
  }
  const stmts: D1PreparedStatement[] = [];
  // only a pair with a stored verdict can be replaced: bind those, not every work compared (a check
  // compares hundreds per shard; D1 binds at most 2 MB)
  let replaced: Checked[] = [];
  if (res.checked.length) {
    const stored = await db.prepare("SELECT CASE WHEN a = ?1 THEN b ELSE a END AS id FROM paph_matches WHERE a = ?1 OR b = ?1").bind(postId).all<{ id: number }>();
    const have = new Set((stored.results ?? []).map((r) => r.id));
    replaced = res.checked.filter((c) => have.has(c.id));
  }
  const partners = `SELECT json_extract(c.value, '$.id') FROM json_each(?2) AS c
                    JOIN artworks w2 ON w2.post_id = json_extract(c.value, '$.id') AND w2.content_hash = json_extract(c.value, '$.contentHash')`;
  const DELETE = (guard: string) =>
    `DELETE FROM paph_matches
     WHERE ${LIVE_IMAGE("?1", "?3")}
       AND ((a = ?1 AND b IN (${partners})) OR (b = ?1 AND a IN (${partners})))${guard}`;
  const current = replaced.filter((c) => !legacyWire(c.wire));
  const provisional = replaced.filter((c) => legacyWire(c.wire));
  if (current.length) stmts.push(db.prepare(DELETE("")).bind(postId, JSON.stringify(current), contentHash));
  if (provisional.length) stmts.push(db.prepare(DELETE(" AND engine IS NOT ?4")).bind(postId, JSON.stringify(provisional), contentHash, res.identity));
  const INSERT = (guard: string) =>
    `INSERT INTO paph_matches (a, b, a_hash, b_hash, verdict, state, certifiable, certificate, structural_lo, structural_hi,
       geometry, inliers, mirrored, execution, rescued, same_author, via, engine, computed)
     SELECT json_extract(j.value, '$.a'), json_extract(j.value, '$.b'), json_extract(j.value, '$.a_hash'), json_extract(j.value, '$.b_hash'),
            json_extract(j.value, '$.verdict'), json_extract(j.value, '$.state'), json_extract(j.value, '$.certifiable'),
            json_extract(j.value, '$.certificate'), json_extract(j.value, '$.lo'), json_extract(j.value, '$.hi'),
            json_extract(j.value, '$.geometry'), json_extract(j.value, '$.inliers'), json_extract(j.value, '$.mirrored'),
            json_extract(j.value, '$.execution'), 0, json_extract(j.value, '$.same_author'),
            json_extract(j.value, '$.via'), json_extract(j.value, '$.engine'), ?2
     FROM json_each(?1) AS j
     WHERE ${LIVE_IMAGE("json_extract(j.value, '$.other')", "json_extract(j.value, '$.other_hash')")}
       AND ${LIVE_IMAGE("?3", "?4")}
     ON CONFLICT(a, b) DO UPDATE SET a_hash = excluded.a_hash, b_hash = excluded.b_hash, verdict = excluded.verdict,
       state = excluded.state, certifiable = excluded.certifiable, certificate = excluded.certificate,
       structural_lo = excluded.structural_lo, structural_hi = excluded.structural_hi, geometry = excluded.geometry,
       inliers = excluded.inliers, mirrored = excluded.mirrored, execution = excluded.execution,
       same_author = excluded.same_author, via = excluded.via, engine = excluded.engine, computed = excluded.computed${guard}`;
  const inserts: number[] = [];
  if (rows.length && res.identity) {
    inserts.push(stmts.length);
    stmts.push(db.prepare(INSERT("")).bind(JSON.stringify(rows), now(), postId, contentHash));
  }
  if (legacyRows.length && res.identity) {
    inserts.push(stmts.length);
    stmts.push(db.prepare(INSERT(" WHERE paph_matches.engine IS NOT ?5")).bind(JSON.stringify(legacyRows), now(), postId, contentHash, res.identity));
  }
  // the progress row the run continued from is still there (or the run continued from none)
  const BASIS = (id: string, token: string) => `(${token} IS NULL OR EXISTS (SELECT 1 FROM paph_progress WHERE post_id = ${id} AND token = ${token}))`;
  const markAt = mark ? stmts.length : -1;
  if (mark) {
    stmts.push(
      db
        .prepare(`UPDATE artworks SET paph_hash = ?1, paph_engine = ?3 WHERE post_id = ?2 AND ${LIVE_IMAGE("?2", "?1")} AND ${BASIS("?2", "?4")}`)
        .bind(contentHash, postId, mark, stage?.basis ?? null),
    );
  }
  const progressAt = stage?.progress ? stmts.length : -1;
  if (stage && !stage.progress) {
    stmts.push(db.prepare("DELETE FROM paph_progress WHERE post_id = ?").bind(postId));
  } else if (stage?.progress) {
    const p = stage.progress;
    stmts.push(
      db
        .prepare(
          `INSERT INTO paph_progress (post_id, content_hash, engine, shard_size, token, done, updated)
           SELECT ?1, ?2, ?3, ?4, ?5, ?6, ?7
           WHERE ${LIVE_IMAGE("?1", "?2")} AND ${BASIS("?1", "?8")}
           ON CONFLICT(post_id) DO UPDATE SET content_hash = excluded.content_hash, engine = excluded.engine,
             shard_size = excluded.shard_size, token = excluded.token, done = excluded.done, updated = excluded.updated`,
        )
        .bind(postId, contentHash, p.engine, p.shardSize, p.token, JSON.stringify(p.done), now(), stage.basis),
    );
  }
  const out = stmts.length ? await db.batch(stmts) : [];
  const changes = (i: number) => (i < 0 ? 0 : Number((out[i]?.meta as { changes?: number } | undefined)?.changes ?? 0));
  return { stored: inserts.reduce((n, i) => n + changes(i), 0), marked: changes(markAt) > 0, recorded: changes(progressAt) > 0 };
}

/** writeCheck without the stage's bookkeeping: the number of verdicts written. */
export async function replaceMatches(
  db: D1Database,
  postId: number,
  contentHash: string,
  res: { matches: Match[]; checked: Checked[]; identity: string | null },
  mark: string | null,
): Promise<number> {
  return (await writeCheck(db, postId, contentHash, res, mark)).stored;
}

// ---- the enrichment stage -----------------------------------------------------------------------

export interface StageInput {
  postId: number;
  /** sha256 of the image bytes */
  hash: string;
  /** the work's pHash, when the stats stage has computed it */
  phash: string | null;
  /** artworks.paph_hash / paph_engine: the image and the identity the stage last completed for */
  paphHash: string | null;
  paphEngine: string | null;
  force: boolean;
}

export type StageOutcome =
  | { status: "unchanged" }
  | { status: "superseded" }
  | { status: "gone" }
  /** too little of the consumer run is left for a check: none was made (the stage is not marked), the message is retried */
  | { status: "deferred" }
  | {
      status: "done";
      /** every shard and channel answered in full, under this Worker's identity: the stage is marked done */
      complete: boolean;
      /** why not complete */
      incomplete: string[];
      stored: number;
      /** the stored wires of the same image were reused (nothing hashed) */
      reused: boolean;
      /** the image was hashed in wire 3 too, for the works not re-hashed since 1.2.0 */
      twin: boolean;
      keys: { codes: number; bands: number; si: number };
      fit: { divided: number; boxed: number } | null;
      stats: MergedFind["stats"];
      shards: MergedFind["shards"];
    };

/** How long the stage waits for a shard beyond its deadline (the last slice, RPC, a cold start). */
export const STAGE_GRACE_MS = 10_000;
/** A check that would get less time than this is not started (the message is retried instead). */
export const STAGE_MIN_MS = 5_000;
/** How long the stage waits for one call to the work's home shard (info, wires, put, remove). */
export const STAGE_HOME_MS = 20_000;

/** A call to the work's home shard did not answer in time; `run`: the consumer's run was the limit, not the shard. */
export class HomeTimeout extends Error {
  constructor(
    readonly run: boolean,
    message: string,
  ) {
    super(message);
  }
}

/**
 * A call to the work's home shard, within STAGE_HOME_MS and within `leftMs` (what the run can
 * spare): a shard that does not answer must not hold the stage, nor the consumer past its run.
 */
async function homeCall<T>(call: Promise<T>, leftMs: number, what: string): Promise<T> {
  const run = leftMs < STAGE_HOME_MS;
  const ms = Math.max(0, Math.min(STAGE_HOME_MS, leftMs));
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      call,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new HomeTimeout(run, `paph: the work's shard did not answer ${what} within ${Math.round(ms)} ms`)), ms);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

interface ProgressRow {
  content_hash: string;
  engine: string;
  shard_size: number;
  token: string;
  done: string;
}

function doneShards(row: ProgressRow): number[] {
  try {
    const v = JSON.parse(row.done) as unknown;
    return Array.isArray(v) ? v.filter((x): x is number => Number.isSafeInteger(x) && x >= 0) : [];
  } catch {
    return [];
  }
}

/**
 * Index the work and check it against every shard; store the verdicts. A no-op for an image the
 * stage already completed under the current engine, profiles and policy, unless forced: a new
 * release or policy re-checks every work (and the sweeper finds those not re-checked yet). When the
 * image changed, its old verdicts go first — they were about another image — and if the run then
 * fails, the old fingerprint goes too (unless a newer run has replaced it, or the post moved on).
 * For the same image the stored wires are reused (a re-check
 * after a release re-derives keys and signatures without fetching or hashing the image), and
 * previous partners are compared first so their verdicts are refreshed. The checks run as
 * background checks: uploads overtake them in every shard. The work's own embedding nominates
 * later, in `paphVectorPass`, once this run has one.
 *
 * `endsAt`: when the caller must be done (the queue consumer's run). A check that would get less
 * than STAGE_MIN_MS is not made ("deferred"); otherwise its deadline is cut to end by then, and
 * every call to the work's home shard is bounded too.
 */
export async function paphStage(env: Env, s: StageInput, image: () => Promise<RgbaImage>, o: { endsAt?: number } = {}): Promise<StageOutcome> {
  // what is left for the check (its grace kept back), and what the steps before it may spend
  const room = () => (o.endsAt ?? Infinity) - Date.now() - STAGE_GRACE_MS;
  const spare = () => room() - STAGE_MIN_MS;
  const engine = await currentEngine(env);
  if (room() < STAGE_MIN_MS) {
    // no time to ask the shard: a check the mark says is complete for this image under this
    // identity is taken as it is (the nightly clean-up reconciles the index with the marks)
    return !s.force && s.paphHash === s.hash && s.paphEngine === engine ? { status: "unchanged" } : { status: "deferred" };
  }
  const home = homeShard(env, s.postId);
  let info: WorkInfo | null;
  try {
    info = await homeCall(home.info(s.postId), spare(), "info");
  } catch (e) {
    if (e instanceof HomeTimeout && e.run) return { status: "deferred" };
    throw e;
  }
  const sameImage = !!info && info.contentHash === s.hash;
  if (!s.force && s.paphHash === s.hash && s.paphEngine === engine && sameImage && info!.current) return { status: "unchanged" };
  const changed = !!info && !sameImage;
  const size = shardSize(env);
  // a check some shard did not finish, for this image under this identity and shard size: its
  // retry asks only the shards that did not answer in full (anything else starts over)
  const row = changed || s.force
    ? null
    : await env.DB.prepare("SELECT content_hash, engine, shard_size, token, done FROM paph_progress WHERE post_id = ?").bind(s.postId).first<ProgressRow>();
  const usable = !!row && row.content_hash === s.hash && row.engine === engine && Number(row.shard_size) === size;
  const done = new Set<number>(usable ? doneShards(row!) : []);
  const basis = usable && done.size ? row!.token : null;
  // incomplete until the verdicts are written; a changed image's verdicts were about the old one
  await env.DB.batch([
    env.DB.prepare("UPDATE artworks SET paph_hash = NULL, paph_engine = NULL WHERE post_id = ?").bind(s.postId),
    ...(changed ? [env.DB.prepare("DELETE FROM paph_matches WHERE a = ?1 OR b = ?1").bind(s.postId)] : []),
    ...(basis ? [] : [env.DB.prepare("DELETE FROM paph_progress WHERE post_id = ?").bind(s.postId)]),
  ]);
  try {
    let wires: WireInput | null = null;
    let twin: WireInput | null = null;
    let fit: { divided: number; boxed: number } | null = null;
    if (sameImage && info!.wire === WIRE_VERSION && info!.derivation !== "rehash") {
      const w = await homeCall(home.wires(s.postId), spare(), "wires");
      if (w && w.contentHash === s.hash && wireFormat(w.t1) === WIRE_VERSION) wires = { t1: w.t1, t2: w.t2 ?? null };
    }
    const reused = !!wires;
    let img: RgbaImage | null = null;
    if (!wires) {
      img = await image();
      const fp = await fingerprint(img, maxPixels(env));
      wires = { t1: fp.t1, t2: fp.t2 };
      fit = fp.fit;
    }
    // works hashed before 1.2.0 may still wait for their re-hash: they are compared with this
    // image's wire-3 twin (SPEC-W4 §9) — for reused wires too (a retry, a forced re-check), the
    // image then decoded for it alone, best effort: without a twin those works go unread, and
    // their pair with this one waits for their own re-check
    if (storeMayHoldLegacy()) {
      const im = img ?? (await image().catch((e) => (console.warn("paph: no image for a wire-3 twin", s.postId, e instanceof Error ? e.message : e), null)));
      if (im) twin = await twinOf(env, im);
    }
    // a newer image of this post may have overtaken this run (two messages in flight): the stats
    // stage has written its hash, and indexing this one would put an older image back
    const before = await liveImage(env, s.postId);
    if (before !== s.hash) return { status: before === null ? "gone" : "superseded" };
    // only over the entry this run saw: a newer image's run may have put its wires since
    const work = { postId: s.postId, contentHash: s.hash, t1: wires.t1, t2: wires.t2 ?? null };
    const putIf = (expect: string | null) => homeCall(home.putIf(work, expect) as unknown as Promise<PutResult>, spare(), "put");
    let put = await putIf(info?.contentHash ?? null);
    if (put.refused) {
      const shows = await liveImage(env, s.postId);
      if (shows !== s.hash) return { status: shows === null ? "gone" : "superseded" };
      // this run's image is the post's: what the entry shows is older (or was removed)
      put = await putIf(put.current);
      if (put.refused) throw new Error("paph: the work's shard entry changed twice during this run");
    }
    // deleted (or edited) while it was being indexed: the deletion may have missed this entry
    const after = await liveImage(env, s.postId);
    if (after !== s.hash) {
      if (after === null) await homeCall(home.removeIf(s.postId, s.hash), spare(), "remove").catch(() => false);
      return { status: after === null ? "gone" : "superseded" };
    }
    const [previous, near, similar, n] = await Promise.all([
      changed ? Promise.resolve([]) : previousPartners(env.DB, s.postId),
      phashCandidates(env, s.phash, s.postId),
      storedVectorCandidates(env, s.postId),
      shardCount(env),
    ]);
    const all = Array.from({ length: n }, (_, i) => i);
    // the check ends by the end of the consumer's run (hashing and indexing took some of it)
    const left = room();
    if (left < STAGE_MIN_MS) return { status: "deferred" };
    const budget = budgetFor(env, "stage");
    const asked = all.filter((i) => !done.has(i));
    const res = await findEverywhere(env, wires, {
      budget: { ...budget, deadlineMs: Math.min(budget.deadlineMs, left) },
      policy: policyOf(env),
      identity: engine,
      exclude: [s.postId],
      extra: [...previous, ...near.list, ...similar.list],
      minState: storedMin(env),
      priority: "background",
      graceMs: STAGE_GRACE_MS,
      shards: asked,
      twin,
    });
    noteLegacy(res, asked.length === all.length);
    const incomplete: string[] = [];
    const channelsOk = !near.failed && !similar.failed;
    // a shard counts as done only when every channel's candidates reached it (and it answered in
    // full under this identity: findEverywhere counts any other answer as failed)
    if (channelsOk) for (const sh of res.shards.complete) done.add(sh);
    // (a shard that answered in full while a channel failed is not done, but the channel is the reason)
    const missing = all.filter((i) => !done.has(i) && !(res.shards.complete.includes(i) && !channelsOk));
    if (missing.length) {
      const failed = new Map(res.shards.failed.map((f) => [f.shard, f]));
      const why = (i: number) => {
        const f = failed.get(i);
        if (f) return f.engine ? ` (runs ${f.engine})` : f.timedOut ? " (timeout)" : " (failed)";
        return res.shards.complete.includes(i) ? "" : " (deadline)";
      };
      incomplete.push(`shards ${missing.map((i) => `${i}${why(i)}`).join(", ")}`);
    }
    if (near.failed) incomplete.push("pHash neighbours unavailable");
    if (similar.failed) incomplete.push("embedding neighbours unavailable");
    const complete = incomplete.length === 0 && all.every((i) => done.has(i));
    // verdicts, mark and progress in one batch, each only while the post is live and shows this
    // image (and, for a run that continued from a progress row, while that row is still there)
    const recordable = !complete && done.size > 0;
    const w = await writeCheck(env.DB, s.postId, s.hash, res, complete ? engine : null, {
      basis,
      progress: recordable ? { token: basis ?? crypto.randomUUID(), engine, shardSize: size, done: [...done].sort((a, b) => a - b) } : null,
    });
    const out = { status: "done" as const, complete, incomplete, stored: w.stored, reused, twin: !!twin, keys: put.keys, fit, stats: res.stats, shards: res.shards };
    if ((complete && !w.marked) || (recordable && !w.recorded)) {
      const st = await standing(env, s.postId);
      if (st.image !== s.hash) return { status: st.image === null ? "gone" : "superseded" };
      // another run of the same check (a retry delivered twice) completed it first
      if (st.mark === s.hash && st.engine === engine) return { ...out, complete: true, incomplete: [] };
      // the progress row this run continued from went (the post was deleted and restored, or
      // another run started over): what the skipped shards vouched for may be gone
      return { ...out, complete: false, incomplete: [...incomplete, "the progress this run continued from was reset: every shard is asked again"] };
    }
    return out;
  } catch (e) {
    // a home shard too slow for what is left of the run: nothing to undo, the retry starts again
    if (e instanceof HomeTimeout && e.run) return { status: "deferred" };
    // the old image's fingerprint goes too — only while the post still shows this run's image and
    // the entry still shows the old one: a newer run (the reason this one may have failed) keeps
    // its entry, and the verdicts it may have written meanwhile
    if (changed) {
      const wait = Math.min(STAGE_HOME_MS, Math.max(0, (o.endsAt ?? Infinity) - Date.now()));
      if (wait > 0 && (await liveImage(env, s.postId).catch(() => null)) === s.hash) {
        await homeCall(home.removeIf(s.postId, info!.contentHash), wait, "remove").catch(() => false);
      }
    }
    throw e;
  }
}

/** The image a post shows now (null: no live artwork) and the stage's mark on it. */
async function standing(env: Env, postId: number): Promise<{ image: string | null; mark: string | null; engine: string | null }> {
  const r = await env.DB
    .prepare("SELECT a.content_hash, a.paph_hash, a.paph_engine FROM artworks a JOIN posts p ON p.id = a.post_id WHERE a.post_id = ? AND p.deleted = 0 AND p.type = 'artwork'")
    .bind(postId)
    .first<{ content_hash: string; paph_hash: string | null; paph_engine: string | null }>();
  return { image: r?.content_hash ?? null, mark: r?.paph_hash ?? null, engine: r?.paph_engine ?? null };
}

/** The image a post shows now, or null when it is no live artwork. */
async function liveImage(env: Env, postId: number): Promise<string | null> {
  const r = await env.DB
    .prepare("SELECT a.content_hash FROM artworks a JOIN posts p ON p.id = a.post_id WHERE a.post_id = ? AND p.deleted = 0 AND p.type = 'artwork'")
    .bind(postId)
    .first<{ content_hash: string }>();
  return r?.content_hash ?? null;
}

/**
 * Check a work against the neighbours of the image embedding this run just computed (Vectorize
 * does not show a vector it was just given, so the stage could not use it). Only those
 * candidates are compared, in their own shards; their verdicts are replaced, nothing else. Best
 * effort: skipped when too little of the consumer run is left (`endsAt`) — the work's next check,
 * and its neighbours' own checks, see the vector.
 */
export async function paphVectorPass(env: Env, postId: number, contentHash: string, vector: number[], o: { endsAt?: number } = {}): Promise<number> {
  const room = () => (o.endsAt ?? Infinity) - Date.now() - STAGE_GRACE_MS;
  if (room() < STAGE_MIN_MS) return 0;
  const extra = await vectorCandidates(env, vector, postId);
  if (!extra.list.length) return 0;
  const w = await homeCall(homeShard(env, postId).wires(postId), room() - STAGE_MIN_MS, "wires").catch((e) => {
    if (e instanceof HomeTimeout) return null;
    throw e;
  });
  if (!w || w.contentHash !== contentHash) return 0;
  const left = room();
  if (left < STAGE_MIN_MS) return 0;
  const budget = budgetFor(env, "stage");
  const res = await findEverywhere(env, { t1: w.t1, t2: w.t2 ?? null }, {
    budget: { ...budget, deadlineMs: Math.min(budget.deadlineMs, left) },
    policy: policyOf(env),
    identity: await currentEngine(env),
    exclude: [postId],
    extra: extra.list,
    nominate: false,
    extraShardsOnly: true,
    minState: storedMin(env),
    priority: "background",
    graceMs: STAGE_GRACE_MS,
  });
  return replaceMatches(env.DB, postId, contentHash, res, null);
}

/**
 * Errors the paph stage should not retry: the engine refuses this image or its wires. (A broken
 * profile or a missing module fails every work alike: those stay retryable, so the sweeper picks
 * the works up again once it is fixed.)
 */
export function paphPermanent(e: unknown): boolean {
  return /^limit:|tier 1 refused/i.test(e instanceof Error ? e.message : String(e));
}

// ---- reading verdicts -----------------------------------------------------------------------------

export interface CopyInfo {
  verdict: string;
  state: number;
  /** the verifier stands behind the verdict (enough evidence either way) */
  certifiable: boolean;
  /** the sparse geometry alone certified it */
  certificate: boolean;
  /** structural agreement interval and geometric evidence, 0..10000 */
  structural: [number, number];
  geometry: number;
  inliers: number;
  /** the copy is a reflection */
  mirrored: boolean;
  /** how PAPH-X reached it: FAST, DEFERRED, FALLBACK (comparator 42 ran), AUDIT */
  execution: string;
  /** when the listed work was published relative to the one asked about (chain time) */
  relation: "earlier" | "later" | null;
  same_author: boolean | null;
  /** the channels that nominated the pair: codes, bands, si, phash, vector, previous */
  via: string[];
  /** identity of the engine, profiles and policy behind the verdict */
  engine: string;
  computed?: number;
}

export type CopyItem = SearchItem & { copy: CopyInfo };

export interface MatchRow {
  a: number;
  b: number;
  a_hash: string;
  b_hash: string;
  verdict: string;
  state: number;
  certifiable: number;
  certificate: number;
  structural_lo: number;
  structural_hi: number;
  geometry: number;
  inliers: number;
  mirrored: number;
  execution: string;
  same_author: number;
  via: string;
  engine: string;
  computed: number;
}

function copyFromRow(r: MatchRow, about: number): CopyInfo {
  return {
    verdict: r.verdict,
    state: r.state,
    certifiable: r.certifiable === 1,
    certificate: r.certificate === 1,
    structural: [r.structural_lo, r.structural_hi],
    geometry: r.geometry,
    inliers: r.inliers,
    mirrored: r.mirrored === 1,
    execution: r.execution,
    relation: r.a === about ? "later" : "earlier",
    same_author: r.same_author === 1,
    via: r.via ? r.via.split(",") : [],
    engine: r.engine,
    computed: r.computed,
  };
}

/** The part of a Match an answer keeps (and the cache stores). */
interface KeptMatch {
  id: number;
  contentHash: string;
  verdict: string;
  state: number;
  certifiable: boolean;
  certificate: boolean;
  structuralLo: number;
  structuralHi: number;
  geometry: number;
  inliers: number;
  mirrored: boolean;
  execution: string;
  via: string[];
  /** the wire format the pair was compared in (absent: the current one) */
  wire?: number;
}

const keep = (m: Match): KeptMatch => ({
  id: m.id, contentHash: m.contentHash, verdict: m.verdict, state: m.state, certifiable: m.certifiable, certificate: m.certificate,
  structuralLo: m.structuralLo, structuralHi: m.structuralHi, geometry: m.geometry, inliers: m.inliers, mirrored: m.mirrored,
  execution: m.execution, via: m.via, ...(m.wire !== undefined && m.wire !== WIRE_VERSION ? { wire: m.wire } : {}),
});

function copyFromMatch(m: KeptMatch, about: PostMeta | null, item: SearchItem, engine: string): CopyInfo {
  return {
    verdict: m.verdict,
    state: m.state,
    certifiable: m.certifiable,
    certificate: m.certificate,
    structural: [m.structuralLo, m.structuralHi],
    geometry: m.geometry,
    inliers: m.inliers,
    mirrored: m.mirrored,
    execution: m.execution,
    relation: about ? (earlier({ id: item.id, created: item.created, author: item.author }, about) ? "earlier" : "later") : null,
    same_author: about ? about.author === item.author : null,
    via: m.via,
    engine: onWire(engine, m.wire ?? WIRE_VERSION),
  };
}

/**
 * Hydrate ids in order, a page at a time, until `limit` pass the request's filters and `keepIt`
 * (current image…). `hidden` counts the ids passed over before the listing filled up.
 */
async function hydrateUntil(env: Env, ids: number[], req: SearchRequest, limit: number, keepIt: (it: SearchItem) => boolean): Promise<{ items: SearchItem[]; hidden: number }> {
  const out: SearchItem[] = [];
  let hidden = 0;
  for (let at = 0; at < ids.length && out.length < limit; at += 100) {
    const page = ids.slice(at, at + 100);
    const passed = new Map((await hydrateOrdered(env.DB, page, req)).map((it) => [it.id, it]));
    for (const id of page) {
      if (out.length >= limit) break;
      const it = passed.get(id);
      if (it && keepIt(it)) out.push(it);
      else hidden++;
    }
  }
  return { items: out, hidden };
}

/**
 * Matches as listed items: live posts that pass the request's filters (nsfw…), whose image is
 * still the one the verdict was reached on. `hidden` counts the others passed over.
 */
async function hydrateMatches(env: Env, matches: KeptMatch[], about: PostMeta | null, limit: number, req: SearchRequest, engine: string): Promise<{ items: CopyItem[]; hidden: number }> {
  const byId = new Map(matches.map((m) => [m.id, m]));
  const { items, hidden } = await hydrateUntil(env, matches.map((m) => m.id), req, limit, (it) => it.artwork?.hash === byId.get(it.id)?.contentHash);
  return { items: items.map((it) => ({ ...it, copy: copyFromMatch(byId.get(it.id)!, about, it, engine) })), hidden };
}

export interface CopiesResponse {
  id: number;
  min: string;
  method: "stored" | "live";
  /** the stage has completed for the work's current image, under the current engine */
  indexed: boolean;
  items: CopyItem[];
  /** verdicts not listed: the post is filtered out (nsfw…) or deleted, or an image changed since */
  hidden: number;
  /** live only */
  partial?: boolean;
  stats?: MergedFind["stats"];
  shards?: { asked: number; answered: number; failed: number; wall_ms: number };
  took_ms?: number;
  note?: string;
}

/** Copies of a stored work: its stored verdicts about the two works' current images (D1 only). */
export async function copiesOf(env: Env, id: number, o: { min?: string; limit?: number; req?: SearchRequest | null }): Promise<CopiesResponse> {
  const minState = minStateOf(o.min);
  const limit = Math.min(100, Math.max(1, o.limit ?? 24));
  const art = await env.DB.prepare("SELECT content_hash, paph_hash, paph_engine FROM artworks WHERE post_id = ?").bind(id).first<{ content_hash: string; paph_hash: string | null; paph_engine: string | null }>();
  // a listing is a D1 read: it does not fail because the engine cannot start in this isolate
  const engine = await currentEngine(env).catch(() => null);
  const indexed = !!art?.paph_hash && art.paph_hash === art.content_hash && !!engine && art.paph_engine === engine;
  const r = await env.DB
    .prepare(
      `SELECT m.*, (wa.content_hash IS m.a_hash AND wb.content_hash IS m.b_hash) AS current
       FROM paph_matches m LEFT JOIN artworks wa ON wa.post_id = m.a LEFT JOIN artworks wb ON wb.post_id = m.b
       WHERE (m.a = ?1 OR m.b = ?1) AND m.state >= ?2
       ORDER BY m.state DESC, m.certifiable DESC, m.structural_hi DESC, m.a, m.b LIMIT ?3`,
    )
    .bind(id, minState, Math.min(500, limit * 4))
    .all<MatchRow & { current: number }>();
  const rows = r.results ?? [];
  const other = (x: MatchRow) => (x.a === id ? x.b : x.a);
  const byOther = new Map(rows.map((x) => [other(x), x]));
  // a verdict about an image either work no longer shows is passed over (and counted)
  const { items, hidden } = await hydrateUntil(env, rows.map(other), o.req ?? emptyRequest(), limit, (it) => byOther.get(it.id)?.current === 1);
  return {
    id,
    min: STATES[minState],
    method: "stored",
    indexed,
    items: items.map((it) => ({ ...it, copy: copyFromRow(byOther.get(it.id)!, id) })),
    hidden,
  };
}

/** Copies of a stored work, checked now in every shard under the interactive budget (admin). */
export async function liveCopiesOf(env: Env, id: number, o: { min?: string; limit?: number; req?: SearchRequest | null }): Promise<CopiesResponse> {
  const t0 = Date.now();
  const minState = minStateOf(o.min);
  const limit = Math.min(100, Math.max(1, o.limit ?? 24));
  const art = await env.DB.prepare("SELECT content_hash, paph_hash, paph_engine, phash FROM artworks WHERE post_id = ?").bind(id).first<{ content_hash: string; paph_hash: string | null; paph_engine: string | null; phash: string | null }>();
  if (!art) return { id, min: STATES[minState], method: "live", indexed: false, items: [], hidden: 0, note: "no such artwork" };
  const engine = await currentEngine(env);
  const indexed = !!art.paph_hash && art.paph_hash === art.content_hash && art.paph_engine === engine;
  const w = await homeShard(env, id).wires(id);
  if (!w || w.contentHash !== art.content_hash) return { id, min: STATES[minState], method: "live", indexed, items: [], hidden: 0, note: "not in the copy index for its current image yet" };
  const [previous, near, similar, about] = await Promise.all([
    previousPartners(env.DB, id),
    phashCandidates(env, art.phash, id),
    storedVectorCandidates(env, id),
    postMeta(env.DB, [id]).then((m) => m.get(id) ?? null),
  ]);
  const budget = budgetFor(env, "query");
  const res = await findEverywhere(env, { t1: w.t1, t2: w.t2 ?? null }, {
    budget: { ...budget, deadlineMs: Math.min(budget.deadlineMs, Math.max(100, queryMs(env) - (Date.now() - t0) - 400)) },
    policy: policyOf(env),
    identity: engine,
    exclude: [id],
    extra: [...previous, ...near.list, ...similar.list],
    minState,
    countAgeMs: 60_000,
    graceMs: 250,
  });
  noteLegacy(res, true);
  const { items, hidden } = await hydrateMatches(env, res.matches.map(keep), about, limit, o.req ?? emptyRequest(), engine);
  // a live check runs on the stored wires: no image, so no twin. Works stored on another format
  // than this one's (one side not re-hashed since 1.2.0) are not compared, and the answer says so
  const unread = res.stats.unreadable;
  const wire = wireFormat(w.t1);
  return {
    id, min: STATES[minState], method: "live", indexed, items, hidden, partial: res.partial || near.failed || similar.failed || unread > 0, stats: res.stats,
    shards: { asked: res.shards.asked, answered: res.shards.answered, failed: res.shards.failed.length, wall_ms: res.shards.wallMs },
    took_ms: Date.now() - t0,
    ...(unread > 0
      ? {
          note:
            `${unread} nominated ${unread === 1 ? "work was" : "works were"} not compared: stored on another wire format than this work's (wire ${wire})` +
            `${wire === WIRE_VERSION ? "" : ", which the paph stage re-hashes to wire " + WIRE_VERSION}, or unreadable`,
        }
      : {}),
  };
}

// ---- uploads: sub-second, kept for a day ----------------------------------------------------------

export interface ImageCopiesResponse {
  min: string;
  /** when the comparisons ran (unix seconds): a cached answer does not know of works indexed since */
  as_of: number;
  cached: boolean;
  /** some shard or channel did not finish: kept ten minutes instead of a day */
  partial: boolean;
  engine: string;
  width: number;
  height: number;
  kp: number;
  /** how the upload was reduced before hashing, when it was over the pixel budget */
  fit: { divided: number; boxed: number };
  phash: string;
  items: CopyItem[];
  hidden: number;
  stats: MergedFind["stats"];
  shards: { asked: number; answered: number; failed: number; wall_ms: number };
  notes?: string[];
  took_ms: number;
}

/** What the cache keeps: the comparisons, not the listing (hydrated per request: deletions, filters). */
interface CachedImageCopies {
  as_of: number;
  partial: boolean;
  engine: string;
  width: number;
  height: number;
  kp: number;
  fit: { divided: number; boxed: number };
  phash: string;
  matches: KeptMatch[];
  stats: MergedFind["stats"];
  shards: ImageCopiesResponse["shards"];
  notes: string[];
}

async function shortHash(s: string): Promise<string> {
  return (await sha256Hex(new TextEncoder().encode(s))).slice(0, 16);
}

/**
 * Copies of an uploaded image (not stored): hashed here, nominated by its keys, its PAPH-SI
 * signature and its pHash neighbours (with `semantic`, its embedding's neighbours too), compared
 * in every shard at once under the interactive budget — the whole request within PAPH_QUERY_MS
 * (900 ms; with `semantic`, counted from when the embedding is in). The comparisons are kept for a
 * day (PAPH_CACHE_TTL) under the fingerprint, the engine's identity and the budget, and looked up
 * before anything else is computed (the embedding included): the same pixels, however encoded,
 * are answered from the cache without touching a shard. The listing is rebuilt on every request,
 * so a post deleted or edited since drops out.
 */
export async function copiesOfImage(
  env: Env,
  img: RgbaImage,
  o: { min?: string; limit?: number; semantic?: boolean; embed?: () => Promise<number[]>; req?: SearchRequest | null; startedAt?: number },
): Promise<ImageCopiesResponse> {
  const t0 = o.startedAt ?? Date.now();
  const minState = minStateOf(o.min);
  const limit = Math.min(100, Math.max(1, o.limit ?? 24));
  const semantic = !!o.semantic && !!o.embed;
  const engine = await currentEngine(env);
  const budget = budgetFor(env, "query");
  const fp = await fingerprint(img, maxPixels(env));
  const wire = new Uint8Array(fp.t1.length + fp.t2.length);
  wire.set(fp.t1, 0);
  wire.set(fp.t2, fp.t1.length);
  const key = `paph:img:${await shortHash(`${engine}|${budgetLabel(budget)}`)}:${await sha256Hex(wire)}:${minState}:${semantic ? "v" : "-"}`;
  let kept = (await env.CACHE.get(key, "json").catch(() => null)) as CachedImageCopies | null;
  const cached = !!kept;
  if (!kept) {
    const notes: string[] = [];
    let clock = t0;
    let vector: number[] | null = null;
    let degraded = false;
    if (semantic) {
      try {
        vector = await o.embed!();
      } catch (e) {
        notes.push(`semantic candidates unavailable: ${e instanceof Error ? e.message : String(e)}`);
        degraded = true;
      }
      clock = Date.now();
    }
    const ph = phashOf(img);
    // works hashed before 1.2.0 may still wait for their re-hash: they are compared with the
    // upload's wire-3 twin (SPEC-W4 §9), hashed now that the cache has missed
    const [near, similar, twin] = await Promise.all([
      phashCandidates(env, ph),
      vectorCandidates(env, vector),
      storeMayHoldLegacy() ? twinOf(env, img) : Promise.resolve(null),
    ]);
    if (near.failed) notes.push("pHash neighbours unavailable");
    if (similar.failed) notes.push("embedding neighbours unavailable");
    // the shards get what is left of the request's budget, less the hydration's share and the RPC margin
    const remaining = queryMs(env) - (Date.now() - clock) - 150;
    const res = await findEverywhere(env, { t1: fp.t1, t2: fp.t2 }, {
      budget: { ...budget, deadlineMs: Math.min(budget.deadlineMs, Math.max(100, remaining - 250)) },
      policy: policyOf(env),
      identity: engine,
      extra: [...near.list, ...similar.list],
      minState,
      countAgeMs: 60_000,
      graceMs: 250,
      twin,
    });
    noteLegacy(res, true);
    // nominated works that could not be compared (another wire format and no twin of it, a corrupt
    // row): the answer is incomplete, kept ten minutes
    const unread = res.stats.unreadable;
    if (unread > 0) notes.push(`${unread} nominated ${unread === 1 ? "work was" : "works were"} not compared (stored on another wire format, or unreadable)`);
    kept = {
      as_of: now(),
      partial: res.partial || near.failed || similar.failed || degraded || unread > 0,
      engine,
      width: fp.width,
      height: fp.height,
      kp: fp.kp,
      fit: fp.fit,
      phash: ph,
      matches: res.matches.map(keep),
      stats: res.stats,
      shards: { asked: res.shards.asked, answered: res.shards.answered, failed: res.shards.failed.length, wall_ms: res.shards.wallMs },
      notes,
    };
    await env.CACHE.put(key, JSON.stringify(kept), { expirationTtl: kept.partial ? PARTIAL_TTL : cacheTtl(env) }).catch(() => {});
  }
  const { items, hidden } = await hydrateMatches(env, kept.matches, null, limit, o.req ?? emptyRequest(), kept.engine);
  return {
    min: STATES[minState],
    as_of: kept.as_of,
    cached,
    partial: kept.partial,
    engine: kept.engine,
    width: kept.width,
    height: kept.height,
    kp: kept.kp,
    fit: kept.fit,
    phash: kept.phash,
    items,
    hidden,
    stats: kept.stats,
    shards: kept.shards,
    ...(kept.notes?.length ? { notes: kept.notes } : {}),
    took_ms: Date.now() - t0,
  };
}

// ---- reports ------------------------------------------------------------------------------------------

/** Live artworks among `ids`, with the image each shows. */
async function liveImages(env: Env, ids: number[]): Promise<Map<number, string>> {
  const r = await env.DB
    .prepare("SELECT w.post_id AS id, w.content_hash AS hash FROM artworks w JOIN posts p ON p.id = w.post_id WHERE w.post_id IN (SELECT value FROM json_each(?)) AND p.deleted = 0 AND p.type = 'artwork'")
    .bind(JSON.stringify(ids))
    .all<{ id: number; hash: string }>();
  return new Map((r.results ?? []).map((x) => [x.id, x.hash]));
}

/**
 * The PAPH-X report for two stored works (policy as configured, full scope, with comparator 42's
 * own report beside it), and optionally both wires so anyone can re-run it. Only live artworks'
 * current images (checked in D1 before any shard is asked); kept for a day under both content
 * hashes and the identity: an edit is a new key, never a stale answer.
 */
export async function pairReport(env: Env, a: number, b: number, withWires: boolean): Promise<Record<string, unknown> | null> {
  const live = await liveImages(env, [a, b]);
  if (!live.has(a) || !live.has(b)) return null;
  const [wa, wb] = await Promise.all([homeShard(env, a).wires(a), homeShard(env, b).wires(b)]);
  if (!wa || !wb || live.get(a) !== wa.contentHash || live.get(b) !== wb.contentHash) return null;
  const rt = await paphRuntime();
  const policy = policyOf(env);
  const engine = verdictIdentity(rt, policy);
  // the formats are in the key: a work re-hashed since (same image, new wires) is a new report
  const formats = { a: wireFormat(wa.t1), b: wireFormat(wb.t1) };
  const key = `paph:report:${await shortHash(engine)}:${a}:${wa.contentHash}:${b}:${wb.contentHash}:w${formats.a}${formats.b}:${withWires ? 1 : 0}`;
  const hit = (await env.CACHE.get(key, "json").catch(() => null)) as Record<string, unknown> | null;
  if (hit) return { ...hit, cached: true };
  const text = rt.engine.xcompare({ t1: wa.t1, t2: wa.t2 ?? null }, { t1: wb.t1, t2: wb.t2 ?? null }, { profile: rt.x, policy, scope: "full", audit: true, json: true });
  const out: Record<string, unknown> = {
    a, b, engine: onWire(engine, formats.a === formats.b ? formats.a : WIRE_VERSION), content_hashes: { a: wa.contentHash, b: wb.contentHash }, wire_formats: formats, report: JSON.parse(text),
    // a wire-3 side and a wire-4 side are refused as a pair (Indeterminate, WIRE_MISMATCH): one of
    // the two works has not been re-hashed since 1.2.0 yet
    ...(formats.a !== formats.b ? { note: `the stored wires are of two formats (${formats.a} and ${formats.b}): the pair is compared once both are re-hashed` } : {}),
  };
  if (withWires) {
    out.wires = {
      a: { t1: base64Encode(wa.t1), t2: wa.t2 ? base64Encode(wa.t2) : null },
      b: { t1: base64Encode(wb.t1), t2: wb.t2 ? base64Encode(wb.t2) : null },
    };
  }
  await env.CACHE.put(key, JSON.stringify(out), { expirationTtl: cacheTtl(env) }).catch(() => {});
  return { ...out, cached: false };
}

/** Index facts about one artwork (only an artwork D1 knows: no shard is asked about anything else). */
export async function paphInfo(env: Env, id: number): Promise<Record<string, unknown> | null> {
  const art = await env.DB
    .prepare("SELECT a.content_hash, a.paph_hash, a.paph_engine FROM artworks a JOIN posts p ON p.id = a.post_id WHERE a.post_id = ? AND p.deleted = 0")
    .bind(id)
    .first<{ content_hash: string; paph_hash: string | null; paph_engine: string | null }>();
  if (!art) return null;
  const [info, engine] = await Promise.all([homeShard(env, id).info(id), currentEngine(env)]);
  return {
    id,
    shard: shardOf(env, id),
    indexed: !!info,
    complete: art.paph_hash === art.content_hash && art.paph_engine === engine,
    engine: art.paph_engine,
    content_hash: info?.contentHash ?? null,
    image_current: !!info && info.contentHash === art.content_hash,
    kp: info?.kp ?? null,
    width: info?.width ?? null,
    height: info?.height ?? null,
    derivation: info?.derivation ?? null,
    derivation_current: info?.current ?? null,
  };
}

// ---- operations -------------------------------------------------------------------------------------------

/** Every shard's size and state, the verdicts, the stage's progress, the identity and budgets. */
export async function paphStatus(env: Env): Promise<Record<string, unknown>> {
  const rt = await paphRuntime();
  const engine = verdictIdentity(rt, policyOf(env));
  const n = await shardCount(env);
  const calls = await fanOut(env, Array.from({ length: n }, (_, i) => i), (stub) => stub.stats(), 25_000);
  const [verdicts, stage, engines] = await env.DB.batch([
    env.DB.prepare("SELECT verdict, COUNT(*) AS n, SUM(same_author = 0) AS cross_author FROM paph_matches GROUP BY verdict"),
    env.DB.prepare(
      `SELECT COUNT(*) AS artworks, SUM(a.paph_hash = a.content_hash AND a.paph_engine = ?1) AS complete,
              SUM(a.paph_hash = a.content_hash AND a.paph_engine IS NOT ?1) AS other_engine
       FROM artworks a JOIN posts p ON p.id = a.post_id WHERE p.deleted = 0 AND p.type = 'artwork'`,
    ).bind(engine),
    env.DB.prepare("SELECT engine, COUNT(*) AS n FROM paph_matches GROUP BY engine"),
  ]);
  // works still on wire 3 (hashed before 1.2.0): the stage re-hashes them; meanwhile queries bring a twin
  let legacyWorks = 0, legacyShards = 0;
  for (const c of calls) if (c.ok && c.value.legacy > 0) (legacyWorks += c.value.legacy, legacyShards++);
  return {
    identity: rt.identity,
    engine,
    wire: { format: WIRE_VERSION, legacy_works: legacyWorks, legacy_shards: legacyShards, twin: storeMayHoldLegacy() },
    si: rt.si ? "on" : `off (${rt.siOff})`,
    budgets: { query: budgetFor(env, "query"), stage: budgetFor(env, "stage") },
    query_ms: queryMs(env),
    cache_ttl: cacheTtl(env),
    shard_size: shardSize(env),
    shards: calls.map((c) => (c.ok ? { shard: c.shard, ...c.value } : { shard: c.shard, error: c.error })),
    verdicts: verdicts.results,
    engines: engines.results,
    stage: stage.results?.[0],
  };
}

/** Re-derive keys and signatures in every shard (after a release or profile change), a batch each. */
export async function paphRederive(env: Env, limit: number): Promise<{ rederived: number; rehash: number; remaining: number; failed: Array<{ shard: number; error: string }> }> {
  const n = await shardCount(env);
  const calls = await fanOut(env, Array.from({ length: n }, (_, i) => i), (stub) => stub.rederive(limit), 25_000);
  const out = { rederived: 0, rehash: 0, remaining: 0, failed: [] as Array<{ shard: number; error: string }> };
  for (const c of calls) {
    if (!c.ok) out.failed.push({ shard: c.shard, error: c.error });
    else (out.rederived += c.value.rederived, (out.rehash += c.value.rehash), (out.remaining += c.value.remaining));
  }
  return out;
}

/** Cross-author copies found in the last `days`, about the works' current images, newest first: the earlier work is the candidate original. */
export async function paphAlerts(env: Env, days: number, limit: number): Promise<Array<Record<string, unknown>>> {
  const r = await env.DB
    .prepare(
      `SELECT m.a, m.b, m.verdict, m.state, m.certifiable, m.structural_lo, m.structural_hi, m.mirrored, m.via, m.engine, m.computed,
              pa.author AS a_author, pa.permlink AS a_permlink, pa.created AS a_created,
              pb.author AS b_author, pb.permlink AS b_permlink, pb.created AS b_created
       FROM paph_matches m JOIN posts pa ON pa.id = m.a JOIN posts pb ON pb.id = m.b
            JOIN artworks wa ON wa.post_id = m.a AND wa.content_hash = m.a_hash
            JOIN artworks wb ON wb.post_id = m.b AND wb.content_hash = m.b_hash
       WHERE m.computed >= ?1 AND m.same_author = 0 AND m.state >= ?2 AND pa.deleted = 0 AND pb.deleted = 0
       ORDER BY m.computed DESC LIMIT ?3`,
    )
    .bind(now() - Math.max(1, days) * 86400, STATE.Copy, Math.min(500, Math.max(1, limit)))
    .all();
  return r.results ?? [];
}

/**
 * Verdicts reached by another engine or policy than the current one: what the re-check after a
 * release has not replaced (yet). `purge` deletes them — refused while live artworks still wait for
 * their re-check (their verdicts may yet be confirmed), unless `force`. Verdicts this engine reached
 * on wire 3 (`wire3`: a stored work not re-hashed since 1.2.0, compared through the query's twin)
 * are counted apart and never purged: they are replaced on the current wire when that work is
 * re-checked, or by paphHealWire3, and stay when it never can be (they are then the only
 * comparison there is).
 */
export async function paphStaleVerdicts(env: Env, o: { purge?: boolean; force?: boolean } = {}): Promise<{ engine: string; stale: number; wire3: number; awaiting_recheck: number; purged: number; refused?: string }> {
  const engine = await currentEngine(env);
  const w3 = onWire(engine, WIRE_3);
  const [stale, waiting] = await env.DB.batch([
    env.DB.prepare("SELECT SUM(engine <> ?1 AND engine <> ?2) AS n, SUM(engine = ?2) AS w3 FROM paph_matches").bind(engine, w3),
    env.DB.prepare(
      `SELECT COUNT(*) AS n FROM artworks a JOIN posts p ON p.id = a.post_id
       WHERE p.deleted = 0 AND p.type = 'artwork' AND a.stats_hash = a.content_hash AND a.paph_engine IS NOT ?`,
    ).bind(engine),
  ]);
  const st = (stale.results?.[0] ?? {}) as { n?: number | null; w3?: number | null };
  const out = { engine, stale: Number(st.n ?? 0), wire3: Number(st.w3 ?? 0), awaiting_recheck: Number((waiting.results?.[0] as { n?: number })?.n ?? 0), purged: 0 };
  if (!o.purge) return out;
  if (out.awaiting_recheck > 0 && !o.force) return { ...out, refused: `${out.awaiting_recheck} artworks have not been re-checked under the current engine: run the paph stage on them first (or force)` };
  out.purged = (await env.DB.prepare("DELETE FROM paph_matches WHERE engine <> ?1 AND engine <> ?2").bind(engine, w3).run()).meta.changes ?? 0;
  return out;
}

/**
 * Verdicts reached on wire 3 whose two works have both completed their check under the current
 * identity since. The stored side's re-check, which replaces such a verdict on the current wire,
 * ran before it was written (two checks of the pair in flight at once): each is compared again
 * here, from the earlier work's stored wires against the later work alone, and replaced or
 * withdrawn. At most `limit` per call, oldest first; with the nightly clean-up, and
 * `scripts/admin.sh paph-heal`. `unresolved`: still not comparable on the current wire (an entry
 * missing or not re-hashed, the screen rejecting the pair: its verdict stays as it is).
 */
export async function paphHealWire3(env: Env, limit = 100): Promise<{ found: number; healed: number; unresolved: number; failed: number }> {
  const out = { found: 0, healed: 0, unresolved: 0, failed: 0 };
  const engine = await currentEngine(env);
  const w3 = onWire(engine, WIRE_3);
  if (w3 === engine) return out;
  const r = await env.DB
    .prepare(
      `SELECT m.a, m.b, m.a_hash FROM paph_matches m
       JOIN artworks wa ON wa.post_id = m.a AND wa.content_hash = m.a_hash AND wa.paph_hash = m.a_hash AND wa.paph_engine = ?1
       JOIN artworks wb ON wb.post_id = m.b AND wb.content_hash = m.b_hash AND wb.paph_hash = m.b_hash AND wb.paph_engine = ?1
       WHERE m.engine = ?2 ORDER BY m.computed LIMIT ?3`,
    )
    .bind(engine, w3, Math.max(1, limit))
    .all<{ a: number; b: number; a_hash: string }>();
  const pairs = r.results ?? [];
  out.found = pairs.length;
  const budget = budgetFor(env, "stage");
  for (const p of pairs) {
    try {
      const w = await homeShard(env, p.a).wires(p.a);
      if (!w || w.contentHash !== p.a_hash || wireFormat(w.t1) !== WIRE_VERSION) {
        out.unresolved++;
        continue;
      }
      const res = await findEverywhere(env, { t1: w.t1, t2: w.t2 ?? null }, {
        budget,
        policy: policyOf(env),
        identity: engine,
        exclude: [p.a],
        extra: [{ id: p.b, via: "previous", rank: 0 }],
        nominate: false,
        extraShardsOnly: true,
        minState: storedMin(env),
        priority: "background",
        graceMs: STAGE_GRACE_MS,
      });
      if (!res.checked.some((c) => c.id === p.b && (c.wire === undefined || c.wire === WIRE_VERSION))) {
        out.unresolved++;
        continue;
      }
      await replaceMatches(env.DB, p.a, p.a_hash, res, null);
      out.healed++;
    } catch (e) {
      console.warn("paph heal: pair failed", p.a, p.b, e instanceof Error ? e.message : e);
      out.failed++;
    }
  }
  return out;
}

/**
 * One page of the clean-up in one shard, after the shard's cursor: works whose posts are no live
 * artworks leave it, and the page's id range is reconciled with the marks (`reconcile`).
 */
async function gcPage(env: Env, shard: number, limit: number): Promise<{ looked: number; removed: number; reset: number; wrapped: boolean }> {
  const stub = shardStub(env, shard);
  const size = shardSize(env);
  const keyName = `paph:gc:${size}:${shard}`;
  const after = Number((await getSetting(env.DB, keyName)) ?? 0) || 0;
  const works = await stub.list(after, limit);
  const live = await liveImages(env, works.map((w) => w.postId));
  const gone = works.filter((w) => !live.has(w.postId)).map((w) => w.postId);
  const removed = gone.length ? await stub.removeMany(gone) : 0;
  const listedAll = works.length < limit;
  // the ids this page covers: after the cursor, up to the last listed (or the shard's end)
  const rec = await reconcile(env, stub, works, Math.max(after, shard * size - 1), listedAll ? (shard + 1) * size - 1 : works[works.length - 1].postId);
  if (rec.upTo !== null) {
    console.warn(`paph gc: shard ${shard}: ${RECONCILE_MAX} works marked complete without their entry in one page; scripts/admin.sh reindex-all paph re-indexes faster`);
  }
  // stopped early: the next page resumes after the last work reconciled (listing again what it must)
  const wrapped = listedAll && rec.upTo === null;
  await setSetting(env.DB, keyName, String(rec.upTo ?? (wrapped ? 0 : works[works.length - 1].postId)));
  return { looked: works.length, removed, reset: rec.reset, wrapped };
}

/** Works one clean-up page reconciles at most: one D1 read, one shard call, one queue batch. */
const RECONCILE_MAX = 100;

/**
 * The index must hold what a completed check vouches for. Live works with ids in (lo, hi] whose
 * check is marked complete for their image while the shard's entry (as listed in `works`) shows
 * another image, or is missing — an edit and its revert in flight at once, a removal that raced a
 * restore — are looked at again; those still out of step lose the mark and go back to the stage.
 * Rare: one D1 read per page finds them, at most RECONCILE_MAX at a time (a shard that lost its
 * entries wholesale — a new shard size — heals a page at a time; a re-index is faster). `upTo`:
 * the last id handled when the cap stopped the page early.
 */
async function reconcile(
  env: Env,
  stub: ShardStub,
  works: Array<{ postId: number; contentHash: string }>,
  lo: number,
  hi: number,
): Promise<{ reset: number; upTo: number | null }> {
  if (hi <= lo) return { reset: 0, upTo: null };
  const listed = JSON.stringify(works.map((w) => `${w.postId}:${w.contentHash.slice(0, 16)}`));
  const r = await env.DB
    .prepare(
      `SELECT a.post_id AS id, a.content_hash AS hash FROM artworks a JOIN posts p ON p.id = a.post_id
       WHERE a.post_id > ?1 AND a.post_id <= ?2 AND p.deleted = 0 AND p.type = 'artwork' AND a.paph_hash = a.content_hash
         AND (a.post_id || ':' || substr(a.content_hash, 1, 16)) NOT IN (SELECT value FROM json_each(?3))
       ORDER BY a.post_id LIMIT ?4`,
    )
    .bind(lo, hi, listed, RECONCILE_MAX)
    .all<{ id: number; hash: string }>();
  const suspects = r.results ?? [];
  const upTo = suspects.length >= RECONCILE_MAX ? Number(suspects[suspects.length - 1].id) : null;
  if (!suspects.length) return { reset: 0, upTo };
  // a stage may have put the entry since the page was listed: look again, in one call
  const entries = new Map((await stub.hashes(suspects.map((x) => x.id))) as unknown as Array<[number, string]>);
  const stale = suspects.filter((x) => entries.get(Number(x.id)) !== x.hash);
  if (!stale.length) return { reset: 0, upTo };
  const out = await env.DB
    .prepare(
      `UPDATE artworks SET paph_hash = NULL, paph_engine = NULL
       WHERE post_id IN (SELECT json_extract(value, '$.id') FROM json_each(?1)) AND paph_hash = content_hash
         AND content_hash = (SELECT json_extract(j.value, '$.hash') FROM json_each(?1) AS j WHERE json_extract(j.value, '$.id') = artworks.post_id)
       RETURNING post_id`,
    )
    .bind(JSON.stringify(stale.map((x) => ({ id: x.id, hash: x.hash }))))
    .all<{ post_id: number }>();
  const ids = (out.results ?? []).map((x) => Number(x.post_id));
  if (!ids.length) return { reset: 0, upTo };
  // back to the stage now (should the send fail, the sweeper finds them: they are not marked)
  try {
    const posts = await env.DB.prepare("SELECT id, author, permlink FROM posts WHERE id IN (SELECT value FROM json_each(?))").bind(JSON.stringify(ids)).all<{ id: number; author: string; permlink: string }>();
    const msgs = (posts.results ?? []).map((p) => ({ body: { postId: p.id, author: p.author, permlink: p.permlink, stages: ["paph"] as Stage[], reason: "paph-gc" } }));
    for (let i = 0; i < msgs.length; i += 100) await env.ENRICH_QUEUE.sendBatch(msgs.slice(i, i + 100)); // a batch holds 100 at most
    await setJobsMany(env.DB, msgs.map((m) => ({ postId: m.body.postId, stages: m.body.stages })), "queued");
  } catch (e) {
    console.warn("paph gc: could not send works back to the stage", ids, e instanceof Error ? e.message : e);
  }
  return { reset: ids.length, upTo };
}

/**
 * Clean the shards up, a page per shard per call: an indexed work whose post was deleted or is no
 * artwork any more leaves the index (its removal at deletion is best effort). A work showing
 * another image than the one indexed is left to its stage, which replaces it — unless its check
 * is marked complete for the post's image although its entry shows another or is missing (an
 * edit and its revert in flight at once, a removal that raced a restore): that mark goes
 * (`reset`) and the work is sent to the stage again. Resumes where the previous call stopped
 * (settings `paph:gc:<size>:<shard>`) and wraps around at the end.
 */
export async function paphGc(env: Env, limit = 500): Promise<{ looked: number; removed: number; reset: number; failed: number; wrapped: number[]; shards: number }> {
  const n = await shardCount(env);
  const out = { looked: 0, removed: 0, reset: 0, failed: 0, wrapped: [] as number[], shards: n };
  const calls = await fanOut(env, Array.from({ length: n }, (_, i) => i), (_stub, shard) => gcPage(env, shard, limit), 60_000);
  for (const c of calls) {
    if (!c.ok) out.failed++;
    else {
      out.looked += c.value.looked;
      out.removed += c.value.removed;
      out.reset += c.value.reset;
      if (c.value.wrapped) out.wrapped.push(c.shard);
    }
  }
  return out;
}

/**
 * The nightly clean-up: every shard, in parallel, pages until it has been gone through once (or
 * `maxPages` pages); a shard that fails stops alone, the others go on.
 */
export async function paphGcPass(env: Env, pageSize = 4000, maxPages = 30): Promise<{ looked: number; removed: number; reset: number; failed: number[]; unfinished: number[] }> {
  const n = await shardCount(env);
  const out = { looked: 0, removed: 0, reset: 0, failed: [] as number[], unfinished: [] as number[] };
  await Promise.all(
    Array.from({ length: n }, async (_, shard) => {
      try {
        for (let page = 0; page < maxPages; page++) {
          const r = await gcPage(env, shard, pageSize);
          out.looked += r.looked;
          out.removed += r.removed;
          out.reset += r.reset;
          if (r.wrapped) return;
        }
        out.unfinished.push(shard);
      } catch (e) {
        console.warn("paph gc: shard failed", shard, e instanceof Error ? e.message : e);
        out.failed.push(shard);
      }
    }),
  );
  return out;
}
