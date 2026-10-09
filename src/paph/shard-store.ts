// ShardStore — one shard of the copy-detection index: PAPH wires, the two nominators over them
// (exact index keys and PAPH-SI), and the verifier (PAPH-X XRank), on SQLite.
//
// It runs inside a PaphShard Durable Object (ctx.storage.sql), so the verifier reads candidates'
// wires from local storage instead of shipping ~10–25 KB per candidate over the network, and in
// tests on node:sqlite. Nothing here touches the network or decodes images.
//
//   works     one row per artwork: what derived its index entries (`derivation`), then its keys
//             (packed), its PAPH-SI signature and its Tier 1 + Tier 2 wires — small columns first,
//             so scans that read them never walk a row's overflow pages
//   postings  inverted index (kind, key) → post   kind 1: Tier-1 local code (53-bit)
//                                                 kind 2: descriptor band (j·2^24 + value)
//                                                 kind 3: PAPH-SI posting key
//   df        how many works hold each key: 1/df weights, the rarest-first order, the df cap
//   meta      running counts (postings per kind), so a status read scans nothing
//
// A check (find) is bounded by a Budget (budget.ts) in every step that grows with the shard, so
// its work does not depend on how many works the shard holds. Keys and SI only nominate; XRank
// (policy safe, copy scope) decides.
//
// Wire formats (PAPH-X 1.2.0, SPEC-W4 §9): each work's row records the format its wires are in
// (Tier 1's byte 4). A wire-3 side and a wire-4 side are never compared: while works hashed before
// 1.2.0 wait for their re-hash, a check brings the query's twin in the other format, and each
// stored side is compared with the query of its own format. The index keys are the same in both
// formats, so they nominate across them; PAPH-SI holds the current format's signatures only.
//
// One check runs at a time in a shard (Turns): interactive ones (an upload, a live re-check) in
// arrival order, then background ones (the enrichment stage, re-derivation, clean-up). A check runs
// in slices — the key reads in chunks, XRank in batches — and yields to the runtime between them:
// that is what makes the deadline work on Cloudflare, where the clock only moves between turns of
// the event loop (never inside a synchronous call), and it is where a background check steps
// aside for a waiting interactive one. What a check does is bounded by the budget's counts
// (postings read, candidates compared); time bounds it only through its deadline, measured from
// its arrival, waiting included — when its caller stops waiting, it stops too (partial).

import type { Budget } from "./budget";
import { STATE, STATES, WIRE_3, WIRE_VERSION, verdictIdentity, wireFormat, type PaphRuntime, type Policy, type RouteReading, type XRankRecord, type XSide } from "./engine";

export interface SqlCursor {
  toArray(): Record<string, unknown>[];
}
/** The subset of Durable Object SqlStorage the store uses (node:sqlite in tests). */
export interface Sql {
  exec(query: string, ...bindings: unknown[]): SqlCursor;
}
export type Transact = <T>(fn: () => T) => T;

export const KIND_CODE = 1;
export const KIND_BAND = 2;
export const KIND_SI = 3;

export interface WireInput {
  t1: Uint8Array;
  t2?: Uint8Array | null;
}

export interface WorkInput extends WireInput {
  postId: number;
  /** sha256 of the image bytes the wires were computed from */
  contentHash: string;
}

/** What `putIf` did: stored and indexed the work, or refused (the entry shows `current`). */
export type PutResult =
  | { refused: false; keys: { codes: number; bands: number; si: number }; replaced: boolean }
  | { refused: true; current: string | null };

export interface WorkInfo {
  postId: number;
  contentHash: string;
  kp: number;
  width: number;
  height: number;
  /** wire version of the stored wires */
  wire: number;
  /** what derived the stored keys and signature; "rehash" when the wires must be computed again */
  derivation: string;
  /** derived by this engine and these profiles */
  current: boolean;
  updated: number;
}

/** A candidate nominated outside the shard's own index (previous partners, pHash, embeddings). */
export interface Candidate {
  id: number;
  via: string;
  /** 1 = this channel's best; previous partners always go first */
  rank?: number;
}

export type Priority = "interactive" | "background";

export interface FindOptions {
  budget: Budget;
  /** the cascade's policy (default safe: every Copy is comparator 42's or certified) */
  policy?: Policy;
  exclude?: number[];
  extra?: Candidate[];
  /** nominate with the shard's keys and SI too (default true); false checks only `extra` */
  nominate?: boolean;
  /** keep verdicts at or above this state (default Suspected); Indeterminate and NotCopy never */
  minState?: number;
  /** interactive (default): served first; background (the stage): steps aside for interactive checks */
  priority?: Priority;
  /**
   * The query's image hashed in another wire format (SPEC-W4 §9): stored works of that format —
   * hashed before 1.2.0, not re-hashed yet — are compared with it instead of being skipped.
   */
  twin?: WireInput | null;
}

export interface Match {
  id: number;
  /** content hash of the stored wires the verdict was reached against */
  contentHash: string;
  verdict: string;
  state: number;
  execution: string;
  certifiable: boolean;
  certificate: boolean;
  structuralLo: number;
  structuralHi: number;
  geometry: number;
  inliers: number;
  mirrored: boolean;
  swapped: boolean;
  route: RouteReading;
  /** channels that nominated it: codes, bands, si, phash, vector, previous */
  via: string[];
  /** the wire format both sides were compared in (absent: the current one) */
  wire?: number;
}

/** A work a check compared, and the image (content hash) it compared. */
export interface Checked {
  id: number;
  contentHash: string;
  /** the wire format the pair was compared in, when not the current one (a stored work not re-hashed yet) */
  wire?: number;
}

export interface ShardStats {
  /** candidates per channel (a candidate may come from several), and in all */
  nominated: Record<string, number> & { total: number };
  /** XRank records produced, compared (state ≥ 0), rejected by the screen */
  verified: number;
  compared: number;
  rejected: number;
  /** nominated but not verified: past the verify cap (by design) / past the deadline */
  capped: number;
  pending: number;
  /**
   * nominated works not compared because their stored wires could not be: another wire format
   * than the query's and its twin's, or a corrupt row (re-hash them)
   */
  unreadable: number;
  /** the shard holds works on another wire format than the current one (1, else 0; merged: shards that do) */
  legacy: number;
  /** posting rows read by each nominator, and query keys left unread (df cap or budget) */
  postings: { keys: number; si: number };
  skipped: { keys: number; si: number };
  /** time the check ran, and time it waited for its turn */
  ms: number;
  waitedMs: number;
  /** stopped by the deadline or the caller's give-up: some candidates (or keys) never looked at */
  partial: boolean;
}

export interface FindResult {
  matches: Match[];
  /**
   * The works this check compared (state ≥ 0, whatever the verdict), with the image it compared:
   * the pairs whose stored verdicts it may replace. A candidate the screen rejected is not in it —
   * PAPH-X's gate is not a verdict — nor one never reached.
   */
  checked: Checked[];
  stats: ShardStats;
  /** the identity of these verdicts (engine, profiles, policy) */
  identity: string;
}

/** Comparisons per batch, and posting rows per nomination statement: the size of a slice. */
const BATCH = 32;
const CHUNK_POSTINGS = 25_000;

const now = (): number => Math.floor(Date.now() / 1000);

/** Hand the turn back to the runtime: the clock moves, other requests to the shard get in. */
export function yieldToRuntime(): Promise<void> {
  const s = (globalThis as { scheduler?: { wait?: (ms: number) => Promise<void> } }).scheduler;
  return s?.wait ? s.wait(0) : new Promise((resolve) => setTimeout(resolve, 0));
}

/** The shard's queue is full: the caller should come back later (an upload answers partial). */
export class ShardBusy extends Error {}

type Lane = Priority;

interface Ticket {
  lane: Lane;
  /** when it was queued (for aging) and when its holder last showed it is alive (for the lease) */
  since: number;
  touched: number;
  /** slices it may keep the turn although interactive work waits (an aged grant's share) */
  share: number;
  grant: () => void;
}

/**
 * Whose turn it is in a shard: one holder at a time; interactive work in arrival order first,
 * then background work in arrival order. A background holder steps aside at its next slice when
 * interactive work waits — except a background ticket that has waited `ageMs`: it goes ahead at the
 * next change of turn and keeps the turn for `agedSlices` slices (at most one such grant per
 * `ageMs`), so background work progresses and completes under any interactive load. Each queue
 * holds at most `maxQueued` tickets. A holder that has not shown it is alive for `leaseMs` (its
 * request was dropped before it could hand the turn back) loses the turn to the next in line: one
 * timer, armed only while a holder has tickets waiting behind it (a pending timer keeps a Durable
 * Object awake, and an idle shard should hibernate).
 */
export class Turns {
  private holder: Ticket | null = null;
  private readonly queues: Record<Lane, Ticket[]> = { interactive: [], background: [] };
  private agedAt = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;

  constructor(
    private readonly ageMs = 1_000,
    private readonly maxQueued = 64,
    private readonly agedSlices = 4,
    private readonly leaseMs = 30_000,
  ) {}

  get queued(): Record<Lane, number> {
    return { interactive: this.queues.interactive.length, background: this.queues.background.length };
  }

  /** Whether the lease timer is armed (tests). */
  get watching(): boolean {
    return this.timer !== undefined;
  }

  /** Wait for the turn (`front`: ahead of the lane's queue, for a holder that stepped aside). */
  acquire(lane: Lane, front = false): Promise<Ticket> {
    this.reap();
    if (!this.holder && !this.queues.interactive.length && !this.queues.background.length) {
      const now = Date.now();
      const t: Ticket = { lane, since: now, touched: now, share: 0, grant: () => {} };
      this.holder = t;
      return Promise.resolve(t);
    }
    if (!front && this.queues[lane].length >= this.maxQueued) return Promise.reject(new ShardBusy(`shard busy: ${this.queues[lane].length} ${lane} checks queued`));
    return new Promise((resolve) => {
      const t: Ticket = {
        lane,
        since: Date.now(),
        touched: 0,
        share: 0,
        grant: () => {
          t.touched = Date.now();
          this.holder = t;
          resolve(t);
        },
      };
      if (front) this.queues[lane].unshift(t);
      else this.queues[lane].push(t);
      this.watch();
    });
  }

  release(t: Ticket): void {
    if (this.holder !== t) return;
    this.holder = null;
    this.next();
  }

  private next(): void {
    // one aged background grant per `ageMs` at most goes ahead of waiting interactive work
    const now = Date.now();
    const bg = this.queues.background[0];
    const aged = !!bg && !!this.queues.interactive.length && now - bg.since >= this.ageMs && now - this.agedAt >= this.ageMs;
    if (aged) this.agedAt = now;
    const t = aged ? this.queues.background.shift() : this.queues.interactive.shift() ?? this.queues.background.shift();
    if (t) {
      t.share = aged ? this.agedSlices : 0;
      t.grant();
    }
    this.watch();
  }

  /** A holder that has not shown it is alive for `leaseMs` gives the turn up. */
  private reap(): void {
    if (this.holder && Date.now() - this.holder.touched > this.leaseMs) {
      this.holder = null;
      this.next();
    }
  }

  /**
   * Should the holder's request have been dropped, the turn must still come round: while a holder
   * has tickets waiting behind it, one timer looks again when its lease would run out (a holder
   * that is alive has touched it since: the timer looks again later). Cleared when no one waits.
   */
  private watch(): void {
    const waiting = this.queues.interactive.length > 0 || this.queues.background.length > 0;
    if (!this.holder || !waiting) {
      if (this.timer !== undefined) clearTimeout(this.timer);
      this.timer = undefined;
      return;
    }
    if (this.timer !== undefined) return;
    const due = Math.max(10, this.holder.touched + this.leaseMs + 10 - Date.now());
    this.timer = setTimeout(() => {
      this.timer = undefined;
      this.reap();
      this.watch();
    }, due);
  }

  /** At a slice boundary: the holder shows it is alive; a background holder steps aside while interactive work waits. */
  async handover(t: Ticket): Promise<Ticket> {
    t.touched = Date.now();
    if (t.lane !== "background" || !this.queues.interactive.length) return t;
    if (t.share > 0) {
      t.share--;
      return t;
    }
    this.release(t);
    return this.acquire("background", true);
  }
}

/** BLOBs come back as ArrayBuffer (Durable Objects), Uint8Array (node:sqlite) or number[] (D1). */
export function asBytes(v: unknown): Uint8Array | null {
  if (v === null || v === undefined) return null;
  if (v instanceof Uint8Array) return v;
  if (v instanceof ArrayBuffer) return new Uint8Array(v);
  if (ArrayBuffer.isView(v)) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
  if (Array.isArray(v)) return Uint8Array.from(v as number[]);
  return null;
}

/** An exact ArrayBuffer for a BLOB binding (Durable Object SQL binds ArrayBuffers). */
export function blob(u: Uint8Array | null | undefined): ArrayBuffer | null {
  if (!u) return null;
  return (u.byteOffset === 0 && u.byteLength === u.buffer.byteLength ? u.buffer : u.slice().buffer) as ArrayBuffer;
}

/** The engine refused stored wires (another wire version, a corrupt row): re-hash, not retry. */
function wireRefused(e: unknown): boolean {
  return /tier 1 refused|wire version/i.test(e instanceof Error ? e.message : String(e));
}

// ---- keys, packed ---------------------------------------------------------------------------
// A work's keys are stored with it so it can be taken out of the postings exactly (the index's
// key derivation may have moved on since). Packed: codes as float64 (integers below 2^53),
// bands and SI keys as uint32 — about 3.8 KB a work instead of ~7.5 KB of JSON.

export interface KeySet {
  c: number[];
  b: number[];
  s: number[];
}

export function packKeys(k: KeySet): Uint8Array {
  const out = new Uint8Array(13 + 8 * k.c.length + 4 * (k.b.length + k.s.length));
  const dv = new DataView(out.buffer);
  out[0] = 1;
  dv.setUint32(1, k.c.length, true);
  dv.setUint32(5, k.b.length, true);
  dv.setUint32(9, k.s.length, true);
  let o = 13;
  for (const c of k.c) (dv.setFloat64(o, c, true), (o += 8));
  for (const b of k.b) (dv.setUint32(o, b, true), (o += 4));
  for (const s of k.s) (dv.setUint32(o, s, true), (o += 4));
  return out;
}

export function unpackKeys(u: Uint8Array): KeySet {
  const dv = new DataView(u.buffer, u.byteOffset, u.byteLength);
  if (u[0] !== 1) throw new Error(`paph: unknown key packing ${u[0]}`);
  const nc = dv.getUint32(1, true), nb = dv.getUint32(5, true), ns = dv.getUint32(9, true);
  if (13 + 8 * nc + 4 * (nb + ns) !== u.byteLength) throw new Error("paph: truncated key packing");
  const k: KeySet = { c: new Array(nc), b: new Array(nb), s: new Array(ns) };
  let o = 13;
  for (let i = 0; i < nc; i++, o += 8) k.c[i] = dv.getFloat64(o, true);
  for (let i = 0; i < nb; i++, o += 4) k.b[i] = dv.getUint32(o, true);
  for (let i = 0; i < ns; i++, o += 4) k.s[i] = dv.getUint32(o, true);
  return k;
}

/** SELECT kind, key FROM the JSON key set {c, b, s} bound as ?1 */
const KEYSET = `SELECT ${KIND_CODE} AS kind, value AS key FROM json_each(?1, '$.c')
                UNION ALL SELECT ${KIND_BAND}, value FROM json_each(?1, '$.b')
                UNION ALL SELECT ${KIND_SI}, value FROM json_each(?1, '$.s')`;

/** Every statement a check or an index write runs, so a test can hold their plans to the primary keys. */
export const SQL = {
  /** ?1 key set, ?2 post id */
  insertPostings: `INSERT OR IGNORE INTO postings (kind, key, post_id) SELECT kind, key, ?2 FROM (${KEYSET})`,
  /** ?1 key set */
  incDf: `INSERT INTO df (kind, key, n) SELECT kind, key, 1 FROM (${KEYSET}) WHERE true
          ON CONFLICT(kind, key) DO UPDATE SET n = n + 1`,
  /** ?1 key set, ?2 post id — the whole primary key in the IN, or SQLite scans the table */
  deletePostings: `DELETE FROM postings WHERE (kind, key, post_id) IN (SELECT kind, key, ?2 FROM (${KEYSET}))`,
  /** ?1 key set (a compound key set goes through a FROM subquery: written bare, SQLite scans) */
  decDf: `UPDATE df SET n = n - 1 WHERE (kind, key) IN (SELECT kind, key FROM (${KEYSET}))`,
  /** ?1 key set */
  dropDf: `DELETE FROM df WHERE n <= 0 AND (kind, key) IN (SELECT kind, key FROM (${KEYSET}))`,
  /** ?1..?3 postings added (or removed: negative) per kind */
  countPostings: `INSERT INTO meta (k, n) VALUES ('postings:1', ?1), ('postings:2', ?2), ('postings:3', ?3)
                  ON CONFLICT(k) DO UPDATE SET n = n + excluded.n`,
  /** ?1 key set: the document frequency of each of its keys that is indexed */
  dfOf: `SELECT q.kind AS kind, q.key AS key, d.n AS n FROM (${KEYSET}) AS q CROSS JOIN df d ON d.kind = q.kind AND d.key = q.key`,
  /** ?1 a chunk of the chosen keys [[kind, key, df]…], ?2 excluded post ids: Σ 1/df per family per work */
  sumKeys: `WITH k AS (SELECT value->>0 AS kind, value->>1 AS key, 1.0 / (value->>2) AS w FROM json_each(?1))
     SELECT p.post_id AS id,
            SUM(CASE WHEN k.kind = ${KIND_CODE} THEN k.w ELSE 0 END) AS codes,
            SUM(CASE WHEN k.kind = ${KIND_BAND} THEN k.w ELSE 0 END) AS bands
     FROM k CROSS JOIN postings p ON p.kind = k.kind AND p.key = k.key
     WHERE p.post_id NOT IN (SELECT value FROM json_each(?2))
     GROUP BY p.post_id`,
  /** ?1 a chunk of the chosen SI probe keys [[key, df]…], ?2 excluded post ids: Σ 1/df per work */
  reachSi: `WITH k AS (SELECT value->>0 AS key, 1.0 / (value->>1) AS w FROM json_each(?1))
     SELECT p.post_id AS id, SUM(k.w) AS reach
     FROM k CROSS JOIN postings p ON p.kind = ${KIND_SI} AND p.key = k.key
     WHERE p.post_id NOT IN (SELECT value FROM json_each(?2))
     GROUP BY p.post_id`,
  /** ?1 post ids, ?2 the current derivation (a signature of another SI profile does not score) */
  loadSigs: "SELECT post_id, si_sig FROM works WHERE post_id IN (SELECT value FROM json_each(?1)) AND derivation = ?2 AND si_sig IS NOT NULL",
  /** ?1 post ids */
  loadWires: "SELECT post_id, content_hash, wire, t1, t2 FROM works WHERE post_id IN (SELECT value FROM json_each(?1))",
  /** ?1 the current wire format: does the shard hold a work on another (two index probes, whatever its size) */
  legacy: "SELECT EXISTS (SELECT 1 FROM works WHERE wire < ?1) OR EXISTS (SELECT 1 FROM works WHERE wire > ?1) AS legacy",
};

export const SCHEMA = [
  `CREATE TABLE IF NOT EXISTS works (
     post_id      INTEGER PRIMARY KEY,
     content_hash TEXT    NOT NULL,
     wire         INTEGER NOT NULL,
     kp           INTEGER NOT NULL,
     width        INTEGER NOT NULL,
     height       INTEGER NOT NULL,
     derivation   TEXT    NOT NULL,
     updated      INTEGER NOT NULL,
     keys         BLOB    NOT NULL,
     si_sig       BLOB,
     t1           BLOB    NOT NULL,
     t2           BLOB
   )`,
  `CREATE TABLE IF NOT EXISTS postings (
     kind    INTEGER NOT NULL,
     key     INTEGER NOT NULL,
     post_id INTEGER NOT NULL,
     PRIMARY KEY (kind, key, post_id)
   ) WITHOUT ROWID`,
  `CREATE TABLE IF NOT EXISTS df (
     kind INTEGER NOT NULL,
     key  INTEGER NOT NULL,
     n    INTEGER NOT NULL,
     PRIMARY KEY (kind, key)
   ) WITHOUT ROWID`,
  `CREATE TABLE IF NOT EXISTS meta (
     k TEXT    PRIMARY KEY,
     n INTEGER NOT NULL
   )`,
  // which works are on which wire format (1.2.0: wire 4; works hashed before, wire 3 until re-hashed)
  `CREATE INDEX IF NOT EXISTS works_wire ON works (wire)`,
];

interface Nom {
  id: number;
  via: string[];
  /** best rank over the channels that nominated it (previous partners: 0) */
  rank: number;
}

interface Verified {
  id: number;
  contentHash: string;
  rec: XRankRecord;
  /** the format both sides were in */
  wire: number;
}

/** A query side, prepared for XRank, and the format it is in. */
interface QuerySide {
  side: XSide;
  wire: number;
}

/** The `k` entries with the largest scores (ties: smaller id first), best first. */
export function topK<T extends { id: number }>(items: Iterable<T>, k: number, score: (t: T) => number): T[] {
  const out: T[] = [];
  if (k <= 0) return out;
  const better = (a: T, b: T) => score(a) > score(b) || (score(a) === score(b) && a.id < b.id);
  // a few best of many: one pass keeping them in order; many: a sort
  if (k > 64) return [...items].sort((a, b) => (better(a, b) ? -1 : better(b, a) ? 1 : 0)).slice(0, k);
  for (const it of items) {
    if (out.length === k && !better(it, out[k - 1])) continue;
    let i = out.length === k ? k - 1 : out.length;
    while (i > 0 && better(it, out[i - 1])) i--;
    out.splice(i, 0, it);
    if (out.length > k) out.pop();
  }
  return out;
}

/** Keys (rarest first) split into chunks of at most `limit` posting rows each (at least one key). */
function chunked<K extends { n: number }>(keys: K[], limit: number): K[][] {
  const out: K[][] = [];
  let cur: K[] = [];
  let rows = 0;
  for (const k of keys) {
    if (cur.length && rows + k.n > limit) (out.push(cur), (cur = []), (rows = 0));
    cur.push(k);
    rows += k.n;
  }
  if (cur.length) out.push(cur);
  return out;
}

export interface ShardStoreOptions {
  /** how a slice hands the turn back (default: the runtime's scheduler; tests may slow it down) */
  pause?: () => Promise<void>;
  /** posting rows per nomination statement (default 25,000) */
  chunkPostings?: number;
  /** after how long a waiting background check gets a slice ahead of interactive ones (default 1 s) */
  ageMs?: number;
  /** checks queued per lane at most (default 64) */
  maxQueued?: number;
  /** slices an aged background grant keeps the turn (default 4) */
  agedSlices?: number;
}

export class ShardStore {
  readonly turns: Turns;
  private readonly pause: () => Promise<void>;
  private readonly chunk: number;

  constructor(
    private readonly sql: Sql,
    private readonly transact: Transact,
    private readonly rt: PaphRuntime,
    opts: ShardStoreOptions = {},
  ) {
    this.pause = opts.pause ?? yieldToRuntime;
    this.chunk = Math.max(1, opts.chunkPostings ?? CHUNK_POSTINGS);
    this.turns = new Turns(opts.ageMs ?? 1_000, opts.maxQueued ?? 64, opts.agedSlices ?? 4);
  }

  /** Run background work in turns; `slice()` ends a slice (yield, and step aside for interactive work). */
  private async background<T>(work: (slice: () => Promise<void>) => Promise<T>): Promise<T> {
    const held = { ticket: await this.turns.acquire("background") };
    try {
      return await work(async () => {
        await this.pause();
        held.ticket = await this.turns.handover(held.ticket);
      });
    } finally {
      this.turns.release(held.ticket);
    }
  }

  migrate(): void {
    for (const s of SCHEMA) this.sql.exec(s).toArray();
  }

  // ---- writes ------------------------------------------------------------------------------

  /**
   * Store (or replace) a work's wires and index it under its keys and its PAPH-SI signature,
   * derived here from the wires, so the index always matches the engine that will compare. The
   * row records the wires' format, as their Tier 1 says; PAPH-SI holds the current format's
   * signatures only (another format's would not score against current queries). Throws when the
   * engine refuses the wires.
   */
  put(w: WorkInput): { keys: { codes: number; bands: number; si: number }; replaced: boolean } {
    const { engine, x, si } = this.rt;
    const xs = engine.xprepare(w.t1, w.t2 ?? null, { strict: true, profile: x });
    try {
      const format = wireFormat(w.t1);
      const k = engine.indexKeys({ t1: w.t1, t2: w.t2 ?? null });
      const sig = si && format === WIRE_VERSION ? engine.sisig(xs, { profile: si }) : null;
      const set: KeySet = { c: k.codes, b: k.bands, s: sig?.keys ?? [] };
      let replaced = false;
      this.transact(() => {
        const old = this.sql.exec("SELECT keys FROM works WHERE post_id = ?", w.postId).toArray()[0];
        if (old) {
          this.unindex(w.postId, unpackKeys(asBytes(old.keys)!));
          replaced = true;
        }
        this.sql
          .exec(
            `INSERT INTO works (post_id, content_hash, wire, kp, width, height, derivation, updated, keys, si_sig, t1, t2)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
             ON CONFLICT(post_id) DO UPDATE SET content_hash = excluded.content_hash, wire = excluded.wire, kp = excluded.kp,
               width = excluded.width, height = excluded.height, derivation = excluded.derivation, updated = excluded.updated,
               keys = excluded.keys, si_sig = excluded.si_sig, t1 = excluded.t1, t2 = excluded.t2`,
            // wires of another format than the hasher's are indexed by their keys, and wait to be re-hashed
            w.postId, w.contentHash, format, xs.kp, xs.width, xs.height, format === WIRE_VERSION ? this.rt.derivation : "rehash", now(),
            blob(packKeys(set)), blob(sig?.bytes ?? null), blob(w.t1), blob(w.t2 ?? null),
          )
          .toArray();
        this.index(w.postId, set);
      });
      return { keys: { codes: set.c.length, bands: set.b.length, si: set.s.length }, replaced };
    } finally {
      xs.free();
    }
  }

  /**
   * `put`, compare-and-set: only while the stored entry shows the image `expect` (null: no entry)
   * or already the one being written. The stage reads the entry before it hashes; if a run for
   * a newer image of the same post has put its wires since (two messages in flight), this older
   * run cannot put its image back over them. Checked and written in one synchronous step: no other
   * request to the shard runs in between.
   */
  putIf(w: WorkInput, expect: string | null): PutResult {
    const row = this.sql.exec("SELECT content_hash FROM works WHERE post_id = ?", w.postId).toArray()[0];
    const current = row ? String(row.content_hash) : null;
    if (current !== expect && current !== w.contentHash) return { refused: true, current };
    return { refused: false, ...this.put(w) };
  }

  /** Drop several works (the clean-up of works whose posts are gone), in background slices; how many were there. */
  removeMany(postIds: number[]): Promise<number> {
    return this.background(async (slice) => {
      let n = 0;
      for (let i = 0; i < postIds.length; i++) {
        if (i && i % 25 === 0) await slice();
        if (this.remove(postIds[i])) n++;
      }
      return n;
    });
  }

  /** `remove`, only while the entry shows the image `expect` (a run's clean-up must not take a newer run's entry). */
  removeIf(postId: number, expect: string): boolean {
    const row = this.sql.exec("SELECT content_hash FROM works WHERE post_id = ?", postId).toArray()[0];
    return !!row && String(row.content_hash) === expect && this.remove(postId);
  }

  /** Drop a work and its postings. */
  remove(postId: number): boolean {
    let found = false;
    this.transact(() => {
      const old = this.sql.exec("SELECT keys FROM works WHERE post_id = ?", postId).toArray()[0];
      if (!old) return;
      this.unindex(postId, unpackKeys(asBytes(old.keys)!));
      this.sql.exec("DELETE FROM works WHERE post_id = ?", postId).toArray();
      found = true;
    });
    return found;
  }

  /**
   * Re-derive keys and signatures of works indexed by another engine or profile (a new
   * KEYS_VERSION, another X or SI profile), from their stored wires — no image fetched, nothing
   * re-hashed. A wire-3 entry (hashed before 1.2.0) is put again as it is: indexed by its keys,
   * which are the same in both formats, its earlier PAPH-SI signature and postings dropped, and
   * marked "rehash", as are wires the engine refuses (another format, a corrupt row, a row whose
   * column and bytes disagree): the enrichment stage computes them again from the image. Any other
   * error stops the batch (and is reported), so a transient failure never sends a work to be
   * re-hashed.
   */
  rederive(limit = 200): Promise<{ rederived: number; rehash: number; remaining: number }> {
    return this.background(async (slice) => {
      const rows = this.sql
        .exec("SELECT post_id, content_hash, wire FROM works WHERE derivation <> ? AND derivation <> 'rehash' LIMIT ?", this.rt.derivation, limit)
        .toArray();
      let rederived = 0, rehash = 0;
      for (let i = 0; i < rows.length; i++) {
        if (i && i % 10 === 0) await slice();
        const r = rows[i];
        const id = Number(r.post_id);
        const w = this.wires(id);
        if (!w || String(w.contentHash) !== String(r.content_hash)) continue; // replaced or removed meanwhile
        try {
          const format = wireFormat(w.t1);
          if (format !== Number(r.wire) || (format !== WIRE_VERSION && format !== WIRE_3)) throw new Error("wire version");
          this.put({ postId: id, contentHash: w.contentHash, t1: w.t1, t2: w.t2 });
          if (format === WIRE_VERSION) rederived++;
          else rehash++;
        } catch (e) {
          if (!wireRefused(e)) throw e;
          this.sql.exec("UPDATE works SET derivation = 'rehash' WHERE post_id = ?", id).toArray();
          rehash++;
        }
      }
      const left = this.sql.exec("SELECT COUNT(*) AS n FROM works WHERE derivation <> ? AND derivation <> 'rehash'", this.rt.derivation).toArray()[0];
      return { rederived, rehash, remaining: Number(left?.n ?? 0) };
    });
  }

  private index(postId: number, k: KeySet): void {
    const json = JSON.stringify(k);
    this.sql.exec(SQL.insertPostings, json, postId).toArray();
    this.sql.exec(SQL.incDf, json).toArray();
    this.sql.exec(SQL.countPostings, k.c.length, k.b.length, k.s.length).toArray();
  }

  private unindex(postId: number, k: KeySet): void {
    const json = JSON.stringify(k);
    this.sql.exec(SQL.deletePostings, json, postId).toArray();
    this.sql.exec(SQL.decDf, json).toArray();
    this.sql.exec(SQL.dropDf, json).toArray();
    this.sql.exec(SQL.countPostings, -k.c.length, -k.b.length, -k.s.length).toArray();
  }

  // ---- reads -------------------------------------------------------------------------------

  info(postId: number): WorkInfo | null {
    const r = this.sql
      .exec("SELECT post_id, content_hash, kp, width, height, wire, derivation, updated FROM works WHERE post_id = ?", postId)
      .toArray()[0];
    if (!r) return null;
    return {
      postId: Number(r.post_id),
      contentHash: String(r.content_hash),
      kp: Number(r.kp),
      width: Number(r.width),
      height: Number(r.height),
      wire: Number(r.wire),
      derivation: String(r.derivation),
      current: String(r.derivation) === this.rt.derivation,
      updated: Number(r.updated),
    };
  }

  wires(postId: number): (WireInput & { contentHash: string; wire: number }) | null {
    const r = this.sql.exec("SELECT content_hash, wire, t1, t2 FROM works WHERE post_id = ?", postId).toArray()[0];
    if (!r) return null;
    return { contentHash: String(r.content_hash), t1: asBytes(r.t1)!, t2: asBytes(r.t2), wire: Number(r.wire) };
  }

  /** The image each of these works' entries shows (absent: no entry), in one read. */
  hashes(postIds: number[]): Array<[number, string]> {
    if (!postIds.length) return [];
    return this.sql
      .exec("SELECT post_id, content_hash FROM works WHERE post_id IN (SELECT value FROM json_each(?))", JSON.stringify(postIds))
      .toArray()
      .map((r) => [Number(r.post_id), String(r.content_hash)] as [number, string]);
  }

  /** Works after `after`, in id order (for the clean-up of works whose posts are gone). */
  list(after: number, limit: number): Array<{ postId: number; contentHash: string }> {
    return this.sql
      .exec("SELECT post_id, content_hash FROM works WHERE post_id > ? ORDER BY post_id LIMIT ?", after, limit)
      .toArray()
      .map((r) => ({ postId: Number(r.post_id), contentHash: String(r.content_hash) }));
  }

  /**
   * The works of this shard a query copies or is copied by. `query`: wires (an upload, or a
   * stored work's wires fetched from its own shard — exclude its id). Bounded by `o.budget` in
   * every step; see budget.ts. Waits for its turn (Turns); throws ShardBusy when the queue is full.
   */
  async find(query: WireInput, o: FindOptions): Promise<FindResult> {
    const arrived = Date.now();
    // the check's whole life, waiting included, ends when its caller stops waiting
    const stopAt = arrived + Math.max(0, o.budget.deadlineMs);
    const held = { ticket: await this.turns.acquire(o.priority === "background" ? "background" : "interactive") };
    try {
      // a turn granted by another check's release starts inside that check's turn, where the
      // clock still reads the time it began: let it move before reading it
      await this.pause();
      return await this.check(query, o, held, stopAt, Date.now() - arrived);
    } finally {
      this.turns.release(held.ticket);
    }
  }

  private async check(query: WireInput, o: FindOptions, held: { ticket: Ticket }, stopAt: number, waitedBefore: number): Promise<FindResult> {
    const b = o.budget;
    // the work is bounded by the budget's counts; time bounds it only through `stopAt` — never
    // through a running time, which a shard sharing its isolate with others cannot measure
    let ran = 0;
    let waited = waitedBefore;
    let segment = Date.now();
    let cut = segment >= stopAt;
    const step = async (): Promise<boolean> => {
      await this.pause();
      ran += Date.now() - segment;
      const before = held.ticket;
      const t0 = Date.now();
      held.ticket = await this.turns.handover(held.ticket);
      if (held.ticket !== before) {
        await this.pause(); // granted inside another check's turn: let the clock move
        waited += Date.now() - t0;
      }
      segment = Date.now();
      if (segment >= stopAt) cut = true;
      return !cut;
    };
    const policy: Policy = o.policy ?? "safe";
    const minState = o.minState ?? STATE.Suspected;
    const exclude = [...new Set((o.exclude ?? []).filter(Number.isSafeInteger))];
    const excluded = new Set(exclude);
    const stats: ShardStats = {
      nominated: { total: 0 },
      verified: 0, compared: 0, rejected: 0, capped: 0, pending: 0, unreadable: 0,
      legacy: this.holdsLegacy() ? 1 : 0,
      postings: { keys: 0, si: 0 },
      skipped: { keys: 0, si: 0 },
      ms: 0,
      waitedMs: 0,
      partial: false,
    };
    const { engine, x } = this.rt;
    const main: QuerySide = { side: engine.xprepare(query.t1, query.t2 ?? null, { strict: true, profile: x }), wire: wireFormat(query.t1) };
    const qs = main.side;
    // the twin is prepared when a stored side of its format is first met: a shard whose works are
    // all on the query's format never parses it (and one it cannot parse leaves those works unread)
    const twinWire = o.twin ? wireFormat(o.twin.t1) : -1;
    const twin = o.twin && twinWire !== main.wire ? { wires: o.twin, side: null as XSide | null, failed: false } : null;
    const sideFor = (wire: number): XSide | null => {
      if (wire === main.wire) return qs;
      if (!twin || wire !== twinWire) return null;
      if (!twin.side && !twin.failed) {
        try {
          twin.side = engine.xprepare(twin.wires.t1, twin.wires.t2 ?? null, { strict: true, profile: x });
        } catch {
          twin.failed = true;
        }
      }
      return twin.side;
    };
    try {
      const noms = new Map<number, Nom>();
      const add = (id: number, via: string, rank: number) => {
        if (!Number.isSafeInteger(id) || excluded.has(id)) return;
        const n = noms.get(id);
        if (!n) noms.set(id, { id, via: [via], rank });
        else if (!n.via.includes(via)) (n.via.push(via), (n.rank = Math.min(n.rank, rank)));
        else return;
        stats.nominated[via] = (stats.nominated[via] ?? 0) + 1;
      };
      (o.extra ?? []).forEach((c, i) => add(c.id, c.via, c.via === "previous" ? 0 : Math.max(1, c.rank ?? i + 1)));
      if (o.nominate !== false && !cut) {
        // the keys are the same in both formats: they reach stored works of either
        await this.nominateKeys(query, b, exclude, stats, add, step);
        // PAPH-SI holds the current format's signatures: a query in another (a work not re-hashed
        // yet, re-checked live from its stored wires) does not probe it
        if (!cut && main.wire === WIRE_VERSION) await this.nominateSi(qs, b, exclude, stats, add, step);
      }
      stats.nominated.total = noms.size;

      // best first: previous partners, then by the best rank any channel gave, then by how many
      // channels agree — so the verify cap and the deadline cut the weakest candidates
      const order = [...noms.values()].sort((p, q) => p.rank - q.rank || q.via.length - p.via.length || p.id - q.id);
      const list = order.slice(0, b.verify);
      stats.capped = order.length - list.length;
      const results = new Map<number, Verified>();
      let at = 0;
      for (; at < list.length && !cut; at += BATCH) {
        if (!(await step())) break;
        this.rank(sideFor, list.slice(at, at + BATCH), policy, results, stats);
      }
      stats.pending = Math.max(0, list.length - at);

      const via = new Map(list.map((n) => [n.id, n.via]));
      const matches: Match[] = [];
      const checked: Checked[] = [];
      for (const v of results.values()) {
        const r = v.rec;
        if (r.state < 0) {
          stats.rejected++;
          continue;
        }
        stats.compared++;
        checked.push({ id: v.id, contentHash: v.contentHash, ...(v.wire !== WIRE_VERSION ? { wire: v.wire } : {}) });
        if (r.state < minState || r.state > STATE.Identical) continue;
        matches.push({
          id: v.id,
          contentHash: v.contentHash,
          verdict: STATES[r.state],
          state: r.state,
          execution: r.execution,
          certifiable: r.certifiable,
          certificate: r.certificate,
          structuralLo: r.structuralLo,
          structuralHi: r.structuralHi,
          geometry: r.geometryEvidence,
          inliers: r.inliers,
          mirrored: r.mirrored,
          swapped: r.swapped,
          route: r.route,
          via: via.get(v.id) ?? [],
          wire: v.wire,
        });
      }
      matches.sort(byStrength);
      stats.partial = cut;
      stats.ms = ran + (Date.now() - segment);
      stats.waitedMs = waited;
      return { matches, checked, stats, identity: verdictIdentity(this.rt, policy) };
    } finally {
      qs.free();
      twin?.side?.free();
    }
  }

  /** Whether the shard holds a work on another wire format than the hasher's (two index probes). */
  holdsLegacy(): boolean {
    const r = this.sql.exec(SQL.legacy, WIRE_VERSION).toArray()[0];
    return Number(r?.legacy ?? 0) === 1;
  }

  /**
   * Exact keys: the query's codes and bands (every keypoint, and the mirrored descriptors), the
   * rarest first until the posting budget is spent, read in chunks; the top K per family by Σ 1/df.
   */
  private async nominateKeys(
    query: WireInput,
    b: Budget,
    exclude: number[],
    stats: ShardStats,
    add: (id: number, via: string, rank: number) => void,
    step: () => Promise<boolean>,
  ): Promise<void> {
    if (b.perFamily <= 0 || b.keyPostings <= 0) return;
    const qk = this.rt.engine.indexKeys({ t1: query.t1, t2: query.t2 ?? null }, { query: true });
    const indexed = this.sql.exec(SQL.dfOf, JSON.stringify({ c: qk.codes, b: qk.bands, s: [] })).toArray();
    const chosen = rarestFirst(
      indexed.map((r) => ({ kind: Number(r.kind), key: Number(r.key), n: Number(r.n) })),
      b.dfCap,
      b.keyPostings,
    );
    stats.postings.keys += chosen.read;
    stats.skipped.keys += indexed.length - chosen.keys.length;
    if (!chosen.keys.length) return;
    const sums = new Map<number, { id: number; codes: number; bands: number }>();
    for (const chunk of chunked(chosen.keys, this.chunk)) {
      if (!(await step())) return; // past the deadline: what was read is not nominated (the check is partial)
      for (const r of this.sql.exec(SQL.sumKeys, JSON.stringify(chunk.map((k) => [k.kind, k.key, k.n])), JSON.stringify(exclude)).toArray()) {
        const id = Number(r.id);
        const s = sums.get(id);
        if (s) (s.codes += Number(r.codes)), (s.bands += Number(r.bands));
        else sums.set(id, { id, codes: Number(r.codes), bands: Number(r.bands) });
      }
    }
    const all = [...sums.values()];
    topK(all.filter((s) => s.codes > 0), b.perFamily, (s) => s.codes).forEach((s, i) => add(s.id, "codes", i + 1));
    topK(all.filter((s) => s.bands > 0), b.perFamily, (s) => s.bands).forEach((s, i) => add(s.id, "bands", i + 1));
  }

  /**
   * PAPH-SI: the query's probes, the rarest first until the posting budget is spent; the
   * candidates they reach most strongly scored exactly against their stored signatures; the best
   * at or above the threshold.
   */
  private async nominateSi(
    qs: XSide,
    b: Budget,
    exclude: number[],
    stats: ShardStats,
    add: (id: number, via: string, rank: number) => void,
    step: () => Promise<boolean>,
  ): Promise<void> {
    const { engine, si } = this.rt;
    if (!si || b.siTop <= 0 || b.siPostings <= 0 || b.siReach <= 0) return;
    const q = engine.siquery(qs, { profile: si });
    try {
      const plan = q.plan();
      const probes = [...new Set(plan.probes.map((p) => p[0]))];
      const indexed = this.sql.exec(SQL.dfOf, JSON.stringify({ c: [], b: [], s: probes })).toArray();
      const chosen = rarestFirst(
        indexed.map((r) => ({ kind: KIND_SI, key: Number(r.key), n: Number(r.n) })),
        Number.MAX_SAFE_INTEGER,
        b.siPostings,
      );
      stats.postings.si += chosen.read;
      stats.skipped.si += indexed.length - chosen.keys.length;
      if (!chosen.keys.length) return;
      const reach = new Map<number, { id: number; reach: number }>();
      for (const chunk of chunked(chosen.keys, this.chunk)) {
        if (!(await step())) return;
        for (const r of this.sql.exec(SQL.reachSi, JSON.stringify(chunk.map((k) => [k.key, k.n])), JSON.stringify(exclude)).toArray()) {
          const id = Number(r.id);
          const s = reach.get(id);
          if (s) s.reach += Number(r.reach);
          else reach.set(id, { id, reach: Number(r.reach) });
        }
      }
      const reached = topK(reach.values(), b.siReach, (s) => s.reach).map((s) => s.id);
      if (!reached.length || !(await step())) return;
      const threshold = b.siMinScore ?? plan.threshold;
      const scored: Array<{ id: number; score: number }> = [];
      for (const r of this.sql.exec(SQL.loadSigs, JSON.stringify(reached), this.rt.derivation).toArray()) {
        const sig = asBytes(r.si_sig);
        if (!sig || sig.length !== 104) continue;
        const score = q.score(sig);
        if (score !== null && score >= threshold) scored.push({ id: Number(r.post_id), score });
      }
      topK(scored, b.siTop, (s) => s.score).forEach((s, i) => add(s.id, "si", i + 1));
    } finally {
      q.free();
    }
  }

  /**
   * XRank a batch: load the wires, prepare, rank, free. Works this shard does not hold are skipped.
   * Each stored side is ranked against the query side of its own format (`sideFor`; SPEC-W4 §9: a
   * wire-3 side and a wire-4 side are never compared); a side of a format no query side is in, or
   * one whose row and bytes disagree, is not read (unreadable). Gated: under X2 and X3 the
   * screen's exits ask the structural channels before dropping a pair, and XRank reads Copy on
   * every copy comparator 42 does (PAPH-X 1.1.1–1.2.0, synthetic and chain corpora); a rejected
   * candidate (state −1) is not a verdict.
   */
  private rank(sideFor: (wire: number) => XSide | null, batch: Nom[], policy: Policy, out: Map<number, Verified>, stats: ShardStats): void {
    if (!batch.length) return;
    const { engine, x } = this.rt;
    const groups = new Map<number, { query: XSide; sides: XSide[]; order: Array<{ id: number; hash: string }> }>();
    try {
      for (const r of this.sql.exec(SQL.loadWires, JSON.stringify(batch.map((n) => n.id))).toArray()) {
        // stored wires came from this engine's hasher; one it can no longer read is skipped, not fatal
        try {
          const t1 = asBytes(r.t1)!;
          const wire = Number(r.wire);
          if (wireFormat(t1) !== wire) throw new Error("wire version");
          const query = sideFor(wire);
          if (!query) throw new Error("wire version");
          const side = engine.xprepare(t1, asBytes(r.t2), { strict: true, profile: x });
          let g = groups.get(wire);
          if (!g) groups.set(wire, (g = { query, sides: [], order: [] }));
          g.sides.push(side);
          g.order.push({ id: Number(r.post_id), hash: String(r.content_hash) });
        } catch {
          stats.unreadable++;
        }
      }
      for (const [wire, g] of groups) {
        const recs = engine.xrank(g.query, g.sides, { profile: x, policy, gate: true, scope: "copy" });
        stats.verified += recs.length;
        recs.forEach((rec, i) => out.set(g.order[i].id, { id: g.order[i].id, contentHash: g.order[i].hash, rec, wire }));
      }
    } finally {
      for (const g of groups.values()) for (const s of g.sides) s.free();
    }
  }

  /** The shard's state from running counts and one pass over the small columns of `works` — none over the postings. */
  stats(): {
    works: number;
    current: number;
    stale: number;
    rehash: number;
    /** works per wire format, and how many are on another than the hasher's (waiting to be re-hashed) */
    wires: Record<string, number>;
    legacy: number;
    derivation: string;
    identity: string;
    si: string | null;
    postings: Record<string, number>;
    queued: Record<Lane, number>;
  } {
    const w = this.sql
      .exec("SELECT COUNT(*) AS n, SUM(derivation = ?1) AS cur, SUM(derivation = 'rehash') AS bad FROM works", this.rt.derivation)
      .toArray()[0];
    const names: Record<string, string> = { "1": "codes", "2": "bands", "3": "si" };
    const postings: Record<string, number> = {};
    for (const r of this.sql.exec("SELECT k, n FROM meta WHERE k LIKE 'postings:%'").toArray()) {
      postings[names[String(r.k).slice(9)] ?? String(r.k)] = Number(r.n);
    }
    const wires: Record<string, number> = {};
    for (const r of this.sql.exec("SELECT wire, COUNT(*) AS n FROM works GROUP BY wire").toArray()) wires[String(r.wire)] = Number(r.n);
    const works = Number(w?.n ?? 0), current = Number(w?.cur ?? 0), rehash = Number(w?.bad ?? 0);
    return {
      works,
      current,
      stale: works - current - rehash,
      rehash,
      wires,
      legacy: works - (wires[String(WIRE_VERSION)] ?? 0),
      derivation: this.rt.derivation,
      identity: this.rt.identity.id,
      si: this.rt.si ? this.rt.identity.siprofileId : null,
      postings,
      queued: this.turns.queued,
    };
  }
}

/** Strongest verdict first, then the more certain, then more structure, then id. */
export function byStrength(a: Pick<Match, "state" | "certifiable" | "structuralHi" | "id">, b: Pick<Match, "state" | "certifiable" | "structuralHi" | "id">): number {
  return b.state - a.state || Number(b.certifiable) - Number(a.certifiable) || b.structuralHi - a.structuralHi || a.id - b.id;
}

/**
 * The keys a nominator reads: those held by at most `dfCap` works, rarest first, while the
 * posting rows they cost stay within `budget`. The rarest keys carry the most weight (1/df), so
 * a budget cuts the least informative reads.
 */
export function rarestFirst<K extends { kind: number; key: number; n: number }>(keys: K[], dfCap: number, budget: number): { keys: K[]; read: number } {
  const sorted = keys.filter((k) => k.n > 0 && k.n <= dfCap).sort((a, b) => a.n - b.n || a.kind - b.kind || a.key - b.key);
  const out: K[] = [];
  let read = 0;
  for (const k of sorted) {
    if (read + k.n > budget) break;
    out.push(k);
    read += k.n;
  }
  return { keys: out, read };
}
