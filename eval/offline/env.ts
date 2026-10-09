// The offline evaluation's index: the real code (ingestion, history, enrichment) on a snapshot of
// the chain, with real SigLIP 2 vectors from a local copy of the Space (embed_server.py) and SQLite
// in place of D1. Built once into STATE (an SQLite file, with its vectors in STATE.vectors.json)
// and reused by every evaluation (run.test.ts: v3 search and /ask; rerank.test.ts: the
// cross-encoder; v4.test.ts: the v4 question set).

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { vi } from "vitest";
import { codecsReady } from "../../test/helpers";
import { FakeD1 } from "../../test/harness/d1";
import { makeEnv, type TestEnv } from "../../test/harness/fakes";
import { ingestPost, rpcFor } from "../../src/chain/ingest";
import { artworksWithHashes, refreshImageHistory, relabelKinds, walkCommentHistory, writeVersions } from "../../src/chain/versions";
import { enrichOne } from "../../src/enrich/consumer";
import { refreshBackground } from "../../src/search/background";
import { setSetting } from "../../src/db/posts";

export interface SnapshotOptions {
  /** directory with chain_posts.json and chain_history.json */
  snapshot: string;
  /** SQLite file to build once and reuse ("" = in memory) */
  state?: string;
  embedUrl?: string;
  /** more variables for the Worker's environment */
  vars?: Record<string, string>;
  /** {"author/permlink": {"caption", "tags"}}: a simulated describe stage */
  captions?: string;
  /** ranker weights JSON */
  weights?: string;
}

const realFetch = globalThis.fetch;

/** Apply the migrations a reused state predates (it was built by an older version of this code). */
function upgrade(db: FakeD1, dir: string): void {
  const has = (table: string) => !!db.raw.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?").get(table);
  if (!has("ask_log")) db.raw.exec(readFileSync(`${dir}/0005_v4.sql`, "utf8"));
}

export async function snapshotEnv(o: SnapshotOptions): Promise<TestEnv> {
  await codecsReady();
  const posts = JSON.parse(readFileSync(`${o.snapshot}/chain_posts.json`, "utf8"));
  const history: Record<string, Array<[number, any]>> = JSON.parse(readFileSync(`${o.snapshot}/chain_history.json`, "utf8"));
  const byRef = new Map<string, any>(posts.map((p: any) => [`${p.author}/${p.permlink}`, p]));
  const migrations = new URL("../../migrations", import.meta.url).pathname;
  const reuse = !!o.state && existsSync(o.state);
  const db = new FakeD1(o.state || ":memory:");
  if (!reuse) db.migrate(migrations);
  else upgrade(db, migrations);
  const env = makeEnv({
    db,
    RPC_URL: "https://rpc.snapshot",
    HF_EMBED_URL: o.embedUrl ?? "http://127.0.0.1:7861/embed",
    EMBED_MODEL: process.env.EMBED_MODEL ?? "google/siglip2-base-patch16-naflex",
    EMBED_PATCHES: process.env.EMBED_PATCHES ?? "576",
    EMBED_DIM: process.env.EMBED_DIM ?? "768",
    VLM_BACKEND: "off",
    TEXT_VECTORS: "true",
    EMBED_VIEWS: process.env.EMBED_VIEWS ?? "xbrz",
    ...(o.vars ?? {}),
  } as any);
  // vectors persist next to the SQLite state
  const vecFile = o.state ? `${o.state}.vectors.json` : "";
  if (reuse && vecFile && existsSync(vecFile)) {
    const saved = JSON.parse(readFileSync(vecFile, "utf8"));
    for (const [k, v] of Object.entries(saved.image)) env._vec.v.set(k, v as any);
    for (const [k, v] of Object.entries(saved.text)) env._vecText.v.set(k, v as any);
    for (const [k, v] of Object.entries(saved.kv ?? {})) env._kv.m.set(k, v as string);
  }
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    if (url.startsWith("https://rpc.snapshot")) {
      const { method, params, id } = JSON.parse(String(init!.body));
      const ok = (result: unknown) => new Response(JSON.stringify({ jsonrpc: "2.0", id, result }));
      if (method === "condenser_api.get_content") return ok(byRef.get(`${params[0]}/${params[1]}`) ?? { author: "" });
      if (method === "account_history_api.get_account_history") {
        const h = (history[params.account] ?? []).filter(([seq]) => params.start < 0 || seq <= params.start);
        return ok({ history: h.slice(-params.limit) });
      }
      throw new Error(`rpc ${method}`);
    }
    return realFetch(input as any, init);
  });

  if (!reuse) {
    const t0 = Date.now();
    for (const p of posts) await ingestPost(env, p, null, "eval");
    for (const author of Object.keys(history)) {
      const { rows } = await walkCommentHistory(rpcFor(env), author, { maxCalls: 50 });
      await writeVersions(env.DB, rows);
      await relabelKinds(env, author);
    }
    // optional simulated describe stage (captions from a local VLM), written like the real one
    const captions = o.captions ? JSON.parse(readFileSync(o.captions, "utf8")) : null;
    let n = 0;
    for (const m of env._queue.drain()) {
      try {
        await enrichOne(env, m);
      } catch (e) {
        console.warn("enrich", m.postId, e instanceof Error ? e.message : e);
      }
      if (++n % 25 === 0) console.log(`enriched ${n}`);
    }
    if (captions) {
      const { updateSearchDocAi } = await import("../../src/db/posts");
      for (const [ref, c] of Object.entries<any>(captions)) {
        const [author, permlink] = ref.split("/");
        const row = await env.DB.prepare("SELECT p.id, a.content_hash FROM posts p JOIN artworks a ON a.post_id = p.id WHERE p.author = ? AND p.permlink = ?").bind(author, permlink).first<any>();
        if (!row) continue;
        await env.DB.prepare("UPDATE artworks SET ai_caption = ?, ai_tags_json = ?, ai_subjects_json = '[]', ai_objects_json = '[]', describe_hash = content_hash, ai_status = 'ok' WHERE post_id = ?").bind(c.caption, JSON.stringify(c.tags ?? []), row.id).run();
        await updateSearchDocAi(env.DB, row.id, c.caption, c.tags ?? []);
        await enrichOne(env, { postId: row.id, author, permlink, stages: ["concepts", "text"], force: true });
      }
      env._queue.drain();
    }
    for (const id of await artworksWithHashes(env.DB, (await env.DB.prepare("SELECT content_hash FROM artworks").all<any>()).results.map((r: any) => r.content_hash))) await refreshImageHistory(env.DB, id);
    console.log(`built in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
    if (vecFile) {
      const kv = Object.fromEntries([...env._kv.m.entries()].filter(([k]) => k.startsWith("qemb:") || k.startsWith("calib:")));
      writeFileSync(vecFile, JSON.stringify({ image: Object.fromEntries(env._vec.v), text: Object.fromEntries(env._vecText.v), kv }));
    }
  }
  await refreshBackground(env, "image");
  await refreshBackground(env, "text");
  if (o.weights) await setSetting(env.DB, "ranker:weights", readFileSync(o.weights, "utf8"));
  return env;
}

// ---- metrics (src/evaluation/retrieval.ts) ------------------------------------------------------------

export { rankMetrics, meanMetrics as meanOf, type RankMetrics, type Judgments as Rel } from "../../src/evaluation/retrieval";
