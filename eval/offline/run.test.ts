// Offline evaluation: the real code (ingestion, history, enrichment, search, v3's /ask) on a
// snapshot of the chain, with real SigLIP 2 vectors from a local copy of the Space
// (eval/offline/embed_server.py) and SQLite in place of D1. The same judged queries as
// scripts/eval.py; "rrf" reruns them with v2's ranking on the same index.
//
//   MODEL_ID=google/siglip2-base-patch16-naflex MAX_NUM_PATCHES=576 python3 eval/offline/embed_server.py --port 7861 &
//   SNAPSHOT_DIR=/path/with/chain_posts.json+chain_history.json npx vitest run -c vitest.eval.config.ts
//
// Env: SNAPSHOT_DIR (required), EMBED_URL (default http://127.0.0.1:7861/embed), EMBED_MODEL, EMBED_PATCHES
// and EMBED_DIM (default google/siglip2-base-patch16-naflex, 576, 768: start the server with the same;
// EMBED_PATCHES= for a model without a patch budget), STATE (sqlite file
// to reuse between runs; delete it to rebuild), OUT (results JSON), CAPTIONS (optional JSON
// {"author/permlink": {"caption", "tags"}} to simulate the describe stage), WEIGHTS (JSON ranker weights).

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { FakeExec, FakeVectorize, type TestEnv } from "../../test/harness/fakes";
import { parseSearchRequest } from "../../src/search/params";
import { search } from "../../src/search/service";
import { askV3 } from "../../src/search/ask";
import { rankMetrics as metrics, type Rel } from "./env";
import { snapshotEnv } from "./env";

const SNAP = process.env.SNAPSHOT_DIR ?? "";
const EMBED_URL = process.env.EMBED_URL ?? "http://127.0.0.1:7861/embed";
const STATE = process.env.STATE ?? "";
const run = SNAP ? describe : describe.skip;

interface Q {
  cat: string;
  q: string;
  type?: string;
  rel: Rel;
}

let env: TestEnv;

run("offline evaluation on the chain snapshot", () => {
  beforeAll(async () => {
    env = await snapshotEnv({ snapshot: SNAP, state: STATE, embedUrl: EMBED_URL, captions: process.env.CAPTIONS, weights: process.env.WEIGHTS });
  }, 3_600_000);

  afterAll(() => vi.unstubAllGlobals());

  it("indexed the snapshot", async () => {
    const s = await env.DB.prepare("SELECT COUNT(*) AS n, SUM(embed_hash = content_hash) AS e, SUM(features_hash = content_hash) AS f FROM artworks").first<any>();
    const v = await env.DB.prepare("SELECT source, COUNT(*) AS n FROM post_versions GROUP BY source").all<any>();
    console.log("artworks", s, "versions", v.results, "image vectors", env._vec.v.size, "text vectors", env._vecText.v.size);
    expect(s.n).toBeGreaterThan(100);
  });

  it("search quality: v3 ranking vs v2's RRF on the same index", async () => {
    const queries: Q[] = readFileSync(new URL("../queries.jsonl", import.meta.url), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    const modes: string[] = (process.env.MODES ?? "v3,rrf").split(",");
    const out: Record<string, any> = {};
    for (const mode of modes) {
      const per: Record<string, ReturnType<typeof metrics>[]> = {};
      const detail: any[] = [];
      for (const x of queries) {
        const exec = new FakeExec();
        const sp = new URLSearchParams({ q: x.q, limit: "50", nsfw: "include", ...(x.type ? { type: x.type } : {}), rank: mode === "rrf" ? "rrf" : "v3" });
        if (mode === "v3-noexpand") sp.set("expand", "0");
        if (mode === "v3-text") sp.set("semantic", "0");
        const r = await search(env, parseSearchRequest(sp), exec as unknown as ExecutionContext);
        await exec.settle();
        const refs = r.items.map((i) => `${i.author}/${i.permlink}`);
        const m = metrics(refs, x.rel);
        (per[x.cat] ??= []).push(m);
        detail.push({ q: x.q, cat: x.cat, m, top: refs.slice(0, 10), notes: r.notes });
      }
      // clear the 60 s result cache between modes
      for (const k of [...env._kv.m.keys()]) if (k.startsWith("s3:")) env._kv.m.delete(k);
      const mean = (rows: ReturnType<typeof metrics>[]) => Object.fromEntries((["ndcg10", "p10", "r10", "r50", "mrr", "ap"] as const).map((k) => [k, rows.reduce((s, r) => s + r[k], 0) / rows.length]));
      out[mode] = { per: Object.fromEntries(Object.entries(per).map(([c, rows]) => [c, { n: rows.length, ...mean(rows) }])), all: mean(Object.values(per).flat()), detail };
    }
    const cats = Object.keys(out[modes[0]].per);
    const lines = [`${"category".padEnd(13)}${"n".padStart(4)}  ${modes.map((m) => `${m.padStart(12)} nDCG  R@50`).join("  ")}`];
    for (const c of [...cats, "ALL"]) {
      const row = modes.map((m) => {
        const s = c === "ALL" ? out[m].all : out[m].per[c];
        return `${s.ndcg10.toFixed(3).padStart(17)} ${s.r50.toFixed(2).padStart(5)}`;
      });
      lines.push(`${c.padEnd(13)}${String(c === "ALL" ? queries.length : out[modes[0]].per[c].n).padStart(4)}  ${row.join("  ")}`);
    }
    console.log(lines.join("\n"));
    if (process.env.OUT) writeFileSync(process.env.OUT, JSON.stringify(out, null, 1));
    expect(out[modes[0]].all.ndcg10).toBeGreaterThan(0);
  }, 3_600_000);

  it("/ask questions", async () => {
    const file = new URL("../ask.jsonl", import.meta.url);
    if (!existsSync(file)) return;
    const rows = readFileSync(file, "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
    let ok = 0;
    const log: string[] = [];
    for (const x of rows) {
      const r = await askV3(env, { question: x.question, planner: "rules" });
      const ev = r.evidence[0];
      let hit = false;
      if (x.expect.ref) hit = ev ? (Array.isArray(x.expect.ref) ? x.expect.ref : [x.expect.ref]).includes(`${ev.author}/${ev.permlink}`) : false;
      else if (x.expect.author) hit = r.answer === x.expect.author;
      else if (x.expect.count !== undefined) hit = typeof r.answer === "number" && Math.abs(r.answer - x.expect.count) <= (x.expect.tolerance ?? 0);
      else if (x.expect.date) hit = r.answer === x.expect.date;
      ok += hit ? 1 : 0;
      log.push(`${hit ? "ok  " : "MISS"} ${x.question}\n      → ${JSON.stringify(r.answer)} | ${r.answer_text} | conf ${r.confidence} | verified ${r.verified ?? "-"}`);
    }
    console.log(log.join("\n") + `\n${ok}/${rows.length} answered as expected`);
    if (process.env.OUT) writeFileSync(process.env.OUT.replace(/\.json$/, "") + ".ask.txt", log.join("\n"));
  }, 600_000);
});

void FakeVectorize;
