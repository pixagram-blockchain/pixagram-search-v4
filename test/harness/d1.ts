// A D1Database over node:sqlite, for tests and the offline evaluation. It implements the subset
// the Worker uses (prepare/bind/all/first/run/raw, batch, exec) with D1's result shapes, and is
// strict where D1 is strict (undefined bind values throw).

import { DatabaseSync, type StatementSync } from "node:sqlite";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

type Value = null | number | bigint | string | Uint8Array;

function toValue(v: unknown, sql: string): Value {
  if (v === undefined) throw new Error(`D1_TYPE_ERROR: undefined bound in: ${sql.slice(0, 120)}`);
  if (v === null) return null;
  if (typeof v === "boolean") return v ? 1 : 0; // D1 coerces booleans the same way
  if (typeof v === "number" || typeof v === "string" || typeof v === "bigint") return v;
  if (v instanceof Uint8Array) return v;
  if (v instanceof ArrayBuffer) return new Uint8Array(v);
  throw new Error(`D1_TYPE_ERROR: unsupported bind type ${typeof v}`);
}

class Statement {
  constructor(private db: FakeD1, readonly sql: string, readonly params: Value[] = []) {}
  bind(...values: unknown[]): Statement {
    return new Statement(this.db, this.sql, values.map((v) => toValue(v, this.sql)));
  }
  private stmt(): StatementSync {
    return this.db.raw.prepare(this.sql);
  }
  async all<T = Record<string, unknown>>(): Promise<{ results: T[]; success: true; meta: Record<string, unknown> }> {
    this.db.count(this.sql);
    const rows = this.stmt().all(...this.params) as T[];
    return { results: rows.map((r) => ({ ...(r as object) }) as T), success: true, meta: { rows_read: rows.length } };
  }
  async first<T = Record<string, unknown>>(col?: string): Promise<T | null> {
    this.db.count(this.sql);
    const row = this.stmt().get(...this.params) as Record<string, unknown> | undefined;
    if (!row) return null;
    return (col ? (row[col] as T) : ({ ...row } as T)) ?? null;
  }
  async run(): Promise<{ success: true; meta: { changes: number; last_row_id: number } }> {
    this.db.count(this.sql);
    const r = this.stmt().run(...this.params);
    return { success: true, meta: { changes: Number(r.changes), last_row_id: Number(r.lastInsertRowid) } };
  }
  async raw<T = unknown[]>(): Promise<T[]> {
    const s = this.stmt();
    s.setReturnArrays?.(true);
    return s.all(...this.params) as T[];
  }
  /** synchronous execution inside batch() */
  runSync(): { results: unknown[]; success: true; meta: Record<string, unknown> } {
    this.db.count(this.sql);
    const s = this.stmt();
    if (/^\s*(select|with|pragma)\b/i.test(this.sql) || /\breturning\b/i.test(this.sql)) {
      const rows = s.all(...this.params);
      return { results: rows.map((r) => ({ ...(r as object) })), success: true, meta: {} };
    }
    const r = s.run(...this.params);
    return { results: [], success: true, meta: { changes: Number(r.changes) } };
  }
}

export class FakeD1 {
  readonly raw: DatabaseSync;
  readonly queries: string[] = [];
  constructor(path = ":memory:") {
    this.raw = new DatabaseSync(path);
    this.raw.exec("PRAGMA foreign_keys = OFF");
  }
  count(sql: string): void {
    this.queries.push(sql);
  }
  prepare(sql: string): Statement {
    return new Statement(this, sql);
  }
  async batch(stmts: Statement[]): Promise<Array<{ results: unknown[]; success: true; meta: Record<string, unknown> }>> {
    this.raw.exec("BEGIN");
    try {
      const out = stmts.map((s) => s.runSync());
      this.raw.exec("COMMIT");
      return out;
    } catch (e) {
      this.raw.exec("ROLLBACK");
      throw e;
    }
  }
  async exec(sql: string): Promise<{ count: number; duration: number }> {
    this.raw.exec(sql);
    return { count: 1, duration: 0 };
  }
  /** Apply every migrations/*.sql in order (what `wrangler d1 migrations apply` does). */
  migrate(dir: string): this {
    for (const f of readdirSync(dir).filter((x) => x.endsWith(".sql")).sort()) this.raw.exec(readFileSync(join(dir, f), "utf8"));
    return this;
  }
}

export function asD1(db: FakeD1): D1Database {
  return db as unknown as D1Database;
}
