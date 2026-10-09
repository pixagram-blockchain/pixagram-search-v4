// PaphShard: one shard of the copy-detection index, as a SQLite-backed Durable Object.
//
// A shard holds the PAPH wires of the artworks whose post ids fall in its range (shards.ts), the
// index keys and PAPH-SI postings over them, and runs the verifier next to that data: a check
// ships one fingerprint in and a few verdicts out, never the candidates' wires. Hashing — the
// step that needs pixels — stays in the queue consumer and the API Worker. Every check runs in
// every shard at once, each bounded by the same budget (budget.ts): a shard's latency does not
// grow with the corpus. One check runs at a time per shard, interactive ones (uploads) before the
// enrichment stage's background ones, which step aside at their next slice (shard-store.ts).

import { DurableObject } from "cloudflare:workers";
import type { Env } from "../env";
import { paphRuntime } from "./engine";
import { ShardStore, type FindOptions, type FindResult, type PutResult, type WireInput, type WorkInfo, type WorkInput } from "./shard-store";

export class PaphShard extends DurableObject<Env> {
  private store!: ShardStore;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
    ctx.blockConcurrencyWhile(async () => {
      const rt = await paphRuntime();
      this.store = new ShardStore(ctx.storage.sql, (fn) => ctx.storage.transactionSync(fn), rt);
      this.store.migrate();
    });
  }

  /** Store a work's wires and index them (re-derived here, by this engine and these profiles). */
  async put(work: WorkInput): Promise<ReturnType<ShardStore["put"]>> {
    return this.store.put(work);
  }

  /** `put`, only while the stored entry shows the image `expect` (null: none) or this one. */
  async putIf(work: WorkInput, expect: string | null): Promise<PutResult> {
    return this.store.putIf(work, expect);
  }

  async remove(postId: number): Promise<boolean> {
    return this.store.remove(postId);
  }

  async removeIf(postId: number, expect: string): Promise<boolean> {
    return this.store.removeIf(postId, expect);
  }

  async removeMany(postIds: number[]): Promise<number> {
    return this.store.removeMany(postIds);
  }

  /** The works of this shard a fingerprint copies or is copied by. */
  async find(query: WireInput, opts: FindOptions): Promise<FindResult> {
    return this.store.find(query, opts);
  }

  /** Works after a post id, in order (the clean-up pages through them). */
  async list(after: number, limit: number): Promise<Array<{ postId: number; contentHash: string }>> {
    return this.store.list(after, limit);
  }

  async hashes(postIds: number[]): Promise<Array<[number, string]>> {
    return this.store.hashes(postIds);
  }

  async info(postId: number): Promise<WorkInfo | null> {
    return this.store.info(postId);
  }

  async wires(postId: number): Promise<ReturnType<ShardStore["wires"]>> {
    return this.store.wires(postId);
  }

  async rederive(limit = 200): Promise<Awaited<ReturnType<ShardStore["rederive"]>>> {
    return this.store.rederive(limit);
  }

  async stats(): Promise<ReturnType<ShardStore["stats"]> & { bytes: number }> {
    return { ...this.store.stats(), bytes: this.ctx.storage.sql.databaseSize };
  }
}
