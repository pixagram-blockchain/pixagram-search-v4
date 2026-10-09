// In-memory stand-ins for the Cloudflare bindings (KV, R2, Queues, Vectorize, Workers AI) and an
// ExecutionContext, enough to run the Worker's code paths in Node against a real SQLite (d1.ts).

import type { Env, EnrichMessage } from "../../src/env";
import { FakeD1, asD1 } from "./d1";

export class FakeKV {
  readonly m = new Map<string, string>();
  /** when a key expires (Date.now() milliseconds), from put's expirationTtl */
  readonly expires = new Map<string, number>();
  async get(key: string, type?: "json" | "text"): Promise<any> {
    const at = this.expires.get(key);
    if (at !== undefined && Date.now() >= at) await this.delete(key);
    const v = this.m.get(key);
    if (v === undefined) return null;
    return type === "json" ? JSON.parse(v) : v;
  }
  async put(key: string, value: string, opts?: { expirationTtl?: number }): Promise<void> {
    this.m.set(key, value);
    if (opts?.expirationTtl) this.expires.set(key, Date.now() + opts.expirationTtl * 1000);
    else this.expires.delete(key);
  }
  async delete(key: string): Promise<void> {
    this.m.delete(key);
    this.expires.delete(key);
  }
}

export class FakeR2 {
  readonly m = new Map<string, { body: Uint8Array; httpMetadata?: Record<string, string> }>();
  async head(key: string) {
    return this.m.has(key) ? { key } : null;
  }
  async put(key: string, body: Uint8Array | ArrayBuffer, opts?: { httpMetadata?: Record<string, string> }) {
    this.m.set(key, { body: body instanceof Uint8Array ? body : new Uint8Array(body), httpMetadata: opts?.httpMetadata });
  }
  async get(key: string) {
    const o = this.m.get(key);
    if (!o) return null;
    return {
      body: o.body,
      httpEtag: `"${key}"`,
      writeHttpMetadata: (h: Headers) => {
        if (o.httpMetadata?.contentType) h.set("content-type", o.httpMetadata.contentType);
      },
    };
  }
}

export class FakeQueue {
  readonly sent: EnrichMessage[] = [];
  /** the delay each sent message asked for (send's options) */
  readonly delays: Array<number | undefined> = [];
  async send(body: EnrichMessage, options?: { delaySeconds?: number }) {
    this.sent.push(body);
    this.delays.push(options?.delaySeconds);
  }
  async sendBatch(msgs: Array<{ body: EnrichMessage }>) {
    for (const m of msgs) this.sent.push(m.body);
  }
  drain(): EnrichMessage[] {
    return this.sent.splice(0, this.sent.length);
  }
}

type Meta = Record<string, string | number | boolean>;

/** Brute-force cosine kNN with Vectorize's metadata filter semantics ($eq, $ne, $in, $nin, $lt, $lte, $gt, $gte). */
export class FakeVectorize {
  readonly v = new Map<string, { values: number[]; metadata: Meta }>();
  queries = 0;
  constructor(readonly indexed: string[] = []) {}
  async upsert(vs: Array<{ id: string; values: number[] | Float32Array; metadata?: Meta }>) {
    for (const x of vs) this.v.set(x.id, { values: Array.from(x.values), metadata: x.metadata ?? {} });
    return { mutationId: "m", count: vs.length };
  }
  async getByIds(ids: string[]) {
    return ids.filter((id) => this.v.has(id)).map((id) => ({ id, values: this.v.get(id)!.values, metadata: this.v.get(id)!.metadata }));
  }
  async deleteByIds(ids: string[]) {
    for (const id of ids) this.v.delete(id);
    return { mutationId: "m", count: ids.length };
  }
  static match(meta: Meta, filter: Record<string, any> | undefined): boolean {
    if (!filter) return true;
    for (const [k, cond] of Object.entries(filter)) {
      const val = meta[k];
      if (cond !== null && typeof cond === "object" && !Array.isArray(cond)) {
        for (const [op, arg] of Object.entries(cond)) {
          if (op === "$eq" && val !== arg) return false;
          if (op === "$ne" && val === arg) return false;
          if (op === "$in" && !(arg as unknown[]).includes(val)) return false;
          if (op === "$nin" && (arg as unknown[]).includes(val)) return false;
          if (op === "$lt" && !(typeof val === "number" && val < (arg as number))) return false;
          if (op === "$lte" && !(typeof val === "number" && val <= (arg as number))) return false;
          if (op === "$gt" && !(typeof val === "number" && val > (arg as number))) return false;
          if (op === "$gte" && !(typeof val === "number" && val >= (arg as number))) return false;
        }
      } else if (val !== cond) return false;
    }
    return true;
  }
  async query(vector: number[], opts: { topK?: number; filter?: Record<string, any> }) {
    this.queries++;
    const topK = opts.topK ?? 5;
    if (topK > 100) throw new Error("VECTOR_QUERY_ERROR: topK > 100");
    if (opts.filter && this.indexed.length) {
      for (const k of Object.keys(opts.filter)) if (!this.indexed.includes(k)) throw new Error(`filter on non-indexed property ${k}`);
    }
    const qn = Math.hypot(...vector) || 1;
    const matches = [...this.v.entries()]
      .filter(([, x]) => FakeVectorize.match(x.metadata, opts.filter))
      .map(([id, x]) => {
        let d = 0;
        for (let i = 0; i < vector.length; i++) d += vector[i] * x.values[i];
        return { id, score: d / qn / (Math.hypot(...x.values) || 1) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
    return { matches, count: matches.length };
  }
}

export type AiHandler = (model: string, input: any) => any | Promise<any>;

export class FakeAI {
  readonly calls: Array<{ model: string; input: any }> = [];
  constructor(public handler: AiHandler = () => ({})) {}
  async run(model: string, input: any) {
    this.calls.push({ model, input });
    return this.handler(model, input);
  }
}

export class FakeExec {
  readonly waits: Promise<unknown>[] = [];
  waitUntil(p: Promise<unknown>) {
    this.waits.push(p.catch(() => {}));
  }
  passThroughOnException() {}
  async settle() {
    while (this.waits.length) await Promise.all(this.waits.splice(0));
  }
}

export const IMAGE_INDEXED = ["author", "primary_color", "size_class", "created", "color_count", "nsfw", "listed", "ai_training", "orientation", "transparent"];
export const TEXT_INDEXED = ["author", "primary_color", "size_class", "created", "type", "nsfw", "listed", "ai_training", "orientation", "transparent"];

export interface TestEnv extends Env {
  _db: FakeD1;
  _kv: FakeKV;
  _r2: FakeR2;
  _queue: FakeQueue;
  _vec: FakeVectorize;
  _vecText: FakeVectorize;
  _ai: FakeAI;
}

export function makeEnv(over: Partial<Env> & { migrationsDir?: string; db?: FakeD1 } = {}): TestEnv {
  const db = over.db ?? new FakeD1().migrate(over.migrationsDir ?? new URL("../../migrations", import.meta.url).pathname);
  const kv = new FakeKV();
  const r2 = new FakeR2();
  const queue = new FakeQueue();
  const vec = new FakeVectorize(IMAGE_INDEXED);
  const vecText = new FakeVectorize(TEXT_INDEXED);
  const ai = new FakeAI();
  const { migrationsDir: _m, db: _d, ...vars } = over;
  return {
    DB: asD1(db),
    ART: r2 as unknown as R2Bucket,
    CACHE: kv as unknown as KVNamespace,
    VEC: vec as unknown as VectorizeIndex,
    VEC_TEXT: vecText as unknown as VectorizeIndex,
    AI: ai as unknown as Ai,
    ENRICH_QUEUE: queue as unknown as Queue<EnrichMessage>,
    INDEXER: {} as DurableObjectNamespace,
    BACKFILL: {} as Workflow,
    RPC_URL: "https://rpc.test",
    APP_PREFIXES: "pixagram",
    EMBED_MODEL: "stub",
    EMBED_DIM: "16",
    VLM_BACKEND: "off",
    STORE_IN_R2: "true",
    ...vars,
    _db: db,
    _kv: kv,
    _r2: r2,
    _queue: queue,
    _vec: vec,
    _vecText: vecText,
    _ai: ai,
  } as TestEnv;
}
