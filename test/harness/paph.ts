// Copy detection in Node: a Durable Object SqlStorage over node:sqlite, and a PAPH namespace whose
// stubs run a real ShardStore per shard (the real engine), with RPC's structured cloning, call
// counting, and injectable failures and delays.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { PaphRuntime, SIProfile, XProfile } from "../../src/paph/engine";
import { SQL, ShardStore, packKeys, type Sql, type Transact } from "../../src/paph/shard-store";

const require = createRequire(import.meta.url);

/** What @pixagram/paph-x 1.1.2's Worker derived its entries with. */
export const DERIVATION_112 = "paph-x/1.1.2 k1 x:27993afaaca76d11 si:dfe99f33b6a8dfc0";

let profiles112: { x: XProfile; si: SIProfile } | null = null;

/**
 * Write a shard entry exactly as the Worker on @pixagram/paph-x 1.1.2 did: wire-3 wires, that
 * release's derivation, and beside the keys its PAPH-SI signature and postings — SI3 under X2 and
 * CAL-004, which 1.2.0 ships in docs/calibration.
 */
export function putAs112(db: DatabaseSync, rt: PaphRuntime, postId: number, contentHash: string, w: { t1: Uint8Array; t2: Uint8Array }): void {
  if (!profiles112) {
    const dir = join(dirname(require.resolve("@pixagram/paph-x/wasm/paph.wasm")), "..", "docs", "calibration");
    const read = (f: string) => new Uint8Array(readFileSync(join(dir, f)));
    const x = rt.engine.xprofile({ base: read("CAL-004-PROPOSED.pcal"), x: read("X2-PROVISIONAL.pxcl") });
    if (x.status() !== "ok") throw new Error(`X2 under CAL-004: ${x.status()}`);
    profiles112 = { x, si: rt.engine.siprofile(read("SI3-PROVISIONAL.psi")) };
  }
  const xs = rt.engine.xprepare(w.t1, w.t2, { strict: true, profile: profiles112.x });
  try {
    const k = rt.engine.indexKeys({ t1: w.t1, t2: w.t2 });
    const sig = rt.engine.sisig(xs, { profile: profiles112.si });
    const set = { c: k.codes, b: k.bands, s: sig.keys };
    db.prepare(
      "INSERT INTO works (post_id, content_hash, wire, kp, width, height, derivation, updated, keys, si_sig, t1, t2) VALUES (?, ?, 3, ?, ?, ?, ?, 1, ?, ?, ?, ?)",
    ).run(postId, contentHash, xs.kp, xs.width, xs.height, DERIVATION_112, packKeys(set), sig.bytes, w.t1, w.t2);
    const json = JSON.stringify(set);
    db.prepare(SQL.insertPostings).run(json, postId);
    db.prepare(SQL.incDf).run(json);
    db.prepare(SQL.countPostings).run(set.c.length, set.b.length, set.s.length);
  } finally {
    xs.free();
  }
}

/** ctx.storage.sql and transactionSync, over an in-memory SQLite. BLOBs bind as ArrayBuffer there; node:sqlite wants bytes. */
export function sqliteStorage(): { db: DatabaseSync; sql: Sql; transact: Transact } {
  const db = new DatabaseSync(":memory:");
  const conv = (v: unknown) => (v instanceof ArrayBuffer ? new Uint8Array(v) : v);
  const sql: Sql = {
    exec(query: string, ...bindings: unknown[]) {
      const st = db.prepare(query);
      const args = bindings.map(conv) as any[];
      const reads = /^\s*(select|with|explain|pragma)\b/i.test(query) || /\breturning\b/i.test(query);
      const rows = reads ? (st.all(...args) as Record<string, unknown>[]) : (st.run(...args), []);
      return { toArray: () => rows.map((r) => ({ ...r })) };
    },
  };
  let depth = 0;
  const transact: Transact = (fn) => {
    const name = `sp${depth}`;
    db.exec(depth === 0 ? "BEGIN" : `SAVEPOINT ${name}`);
    depth++;
    try {
      const out = fn();
      depth--;
      db.exec(depth === 0 ? "COMMIT" : `RELEASE ${name}`);
      return out;
    } catch (e) {
      depth--;
      db.exec(depth === 0 ? "ROLLBACK" : `ROLLBACK TO ${name}`);
      throw e;
    }
  };
  return { db, sql, transact };
}

export interface FakeShard {
  store: ShardStore;
  db: DatabaseSync;
}

export interface FakePaph {
  /** the PAPH binding */
  ns: DurableObjectNamespace;
  shards: Map<string, FakeShard>;
  /** "<shard name>.<method>" per call */
  calls: string[];
  /** shard names whose calls throw / are delayed by ms */
  failing: Set<string>;
  delays: Map<string, number>;
  /** called after each method returns (to interleave events with a check) */
  after: Array<(shard: string, method: string) => void | Promise<void>>;
  /** shard names whose checks answer under another identity (a deploy under way) */
  engines: Map<string, string>;
  /** the arguments of each call, cloned, in order */
  inputs: Array<{ shard: string; method: string; args: unknown[] }>;
  shard(name: string): FakeShard;
}

export function fakePaph(rt: PaphRuntime): FakePaph {
  const shards = new Map<string, FakeShard>();
  const calls: string[] = [];
  const failing = new Set<string>();
  const delays = new Map<string, number>();
  const after: FakePaph["after"] = [];
  const engines = new Map<string, string>();
  const inputs: FakePaph["inputs"] = [];
  const shard = (name: string): FakeShard => {
    let s = shards.get(name);
    if (!s) {
      const st = sqliteStorage();
      const store = new ShardStore(st.sql, st.transact, rt);
      store.migrate();
      s = { store, db: st.db };
      shards.set(name, s);
    }
    return s;
  };
  const stub = (name: string) =>
    new Proxy(
      {},
      {
        get(_t, method) {
          if (method === "then" || typeof method !== "string") return undefined;
          return async (...args: unknown[]) => {
            calls.push(`${name}.${method}`);
            const delay = delays.get(name);
            if (delay) await new Promise((r) => setTimeout(r, delay));
            if (failing.has(name)) throw new Error(`shard ${name} unavailable`);
            const s = shard(name);
            const input = structuredClone(args);
            inputs.push({ shard: name, method, args: structuredClone(args) });
            const out = method === "stats" ? { ...s.store.stats(), bytes: 0 } : await (s.store as any)[method](...input);
            if (method === "find" && engines.has(name)) out.identity = engines.get(name);
            for (const f of after) await f(name, method);
            return structuredClone(out);
          };
        },
      },
    );
  const ns = { idFromName: (n: string) => n, get: (id: string) => stub(id) } as unknown as DurableObjectNamespace;
  return { ns, shards, calls, failing, delays, after, engines, inputs, shard };
}
