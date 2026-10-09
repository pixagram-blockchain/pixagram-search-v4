// Offline evaluation of the cross-encoder (spec §13: the reranker must be evaluated): the judged
// queries of eval/queries.jsonl on the snapshot index, ranked by v3's feature ranker alone, then
// with BAAI/bge-reranker-base (the model behind @cf/baai/bge-reranker-base, served locally by
// rerank_server.py) blended in at several weights and depths. The cross-encoder reads text only
// (title, tags, the author's description, the AI caption, author); the feature ranker carries the
// image similarity, which is why /search blends rather than replaces.
//
//   python3 eval/offline/rerank_server.py --port 7862 &
//   SNAPSHOT_DIR=… STATE=…/s1005.sqlite npx vitest run -c vitest.eval.config.ts eval/offline/rerank.test.ts
//
// Env: SNAPSHOT_DIR, STATE (as run.test.ts), RERANK_URL (default http://127.0.0.1:7862/rerank),
// BLENDS ("0.1,0.2,0.3,0.5,1"), DEPTHS ("20,50"), OUT (results JSON).

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { readFileSync, writeFileSync } from "node:fs";
import { FakeExec, type TestEnv } from "../../test/harness/fakes";
import { parseSearchRequest } from "../../src/search/params";
import { search } from "../../src/search/service";
import { meanOf, rankMetrics, snapshotEnv, type RankMetrics, type Rel } from "./env";

const SNAP = process.env.SNAPSHOT_DIR ?? "";
const run = SNAP ? describe : describe.skip;
const BLENDS = (process.env.BLENDS ?? "0.1,0.2,0.3,0.5,1").split(",").map(Number);
const DEPTHS = (process.env.DEPTHS ?? "20,50").split(",").map(Number);

interface Q {
  cat: string;
  q: string;
  type?: string;
  rel: Rel;
}

let env: TestEnv;

run("the cross-encoder on the judged queries", () => {
  beforeAll(async () => {
    env = await snapshotEnv({
      snapshot: SNAP,
      state: process.env.STATE ?? "",
      vars: { SEARCH_RERANKER_MODEL: "http", SEARCH_RERANK_URL: process.env.RERANK_URL ?? "http://127.0.0.1:7862/rerank" },
    });
  }, 3_600_000);
  afterAll(() => vi.unstubAllGlobals());

  it("feature ranker alone vs blended with the cross-encoder", async () => {
    const queries: Q[] = readFileSync(new URL("../queries.jsonl", import.meta.url), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const configs: Array<{ name: string; rerank: boolean; blend?: number; k?: number }> = [{ name: "v3", rerank: false }];
    for (const k of DEPTHS) for (const b of BLENDS) configs.push({ name: `rr k${k} b${b}`, rerank: true, blend: b, k });
    const out: Record<string, { per: Record<string, RankMetrics & { n: number }>; all: RankMetrics; worse: string[]; better: string[]; ms: number }> = {};
    const baseline = new Map<string, RankMetrics>();
    for (const c of configs) {
      env.SEARCH_RERANK_BLEND = String(c.blend ?? 0.3);
      env.SEARCH_RERANK_K = String(c.k ?? 50);
      const per: Record<string, RankMetrics[]> = {};
      const worse: string[] = [];
      const better: string[] = [];
      const t0 = Date.now();
      for (const x of queries) {
        const exec = new FakeExec();
        const sp = new URLSearchParams({ q: x.q, limit: "50", nsfw: "include", ...(x.type ? { type: x.type } : {}), ...(c.rerank ? { rerank: "1" } : {}) });
        const r = await search(env, parseSearchRequest(sp), exec as unknown as ExecutionContext);
        await exec.settle();
        if (c.rerank && !r.reranked && r.items.length > 1) throw new Error(`reranker did not run for "${x.q}": ${(r.notes ?? []).join("; ")}`);
        const m = rankMetrics(r.items.map((i) => `${i.author}/${i.permlink}`), x.rel);
        (per[x.cat] ??= []).push(m);
        if (!c.rerank) baseline.set(x.q, m);
        else {
          const b = baseline.get(x.q)!;
          if (m.ndcg10 < b.ndcg10 - 1e-9) worse.push(`${x.q} (${b.ndcg10.toFixed(2)} → ${m.ndcg10.toFixed(2)})`);
          if (m.ndcg10 > b.ndcg10 + 1e-9) better.push(`${x.q} (${b.ndcg10.toFixed(2)} → ${m.ndcg10.toFixed(2)})`);
        }
        // the 60 s result cache would answer the next configuration with this one's order
        for (const k of [...env._kv.m.keys()]) if (k.startsWith("s3:")) env._kv.m.delete(k);
      }
      out[c.name] = { per: Object.fromEntries(Object.entries(per).map(([k, rows]) => [k, { n: rows.length, ...meanOf(rows) }])), all: meanOf(Object.values(per).flat()), worse, better, ms: Date.now() - t0 };
    }
    const cats = Object.keys(out.v3.per);
    const names = Object.keys(out);
    const lines = [`${"config".padEnd(14)}${["ALL", ...cats].map((c) => c.slice(0, 9).padStart(10)).join("")}   MRR   R@50  better worse`];
    for (const n of names) {
      const o = out[n];
      lines.push(`${n.padEnd(14)}${[o.all, ...cats.map((c) => o.per[c])].map((m) => m.ndcg10.toFixed(3).padStart(10)).join("")}  ${o.all.mrr.toFixed(3)} ${o.all.r50.toFixed(3)}  ${String(o.better.length).padStart(5)} ${String(o.worse.length).padStart(5)}`);
    }
    console.log(`nDCG@10 by category (${queries.length} judged queries)\n${lines.join("\n")}`);
    if (process.env.OUT) writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
    expect(out.v3.all.ndcg10).toBeGreaterThan(0);
  }, 3_600_000);
});
