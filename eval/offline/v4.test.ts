// Offline evaluation of /ask v4 on its question set (src/evaluation/datasets/questions.jsonl,
// eval/v4/generate.py): every question through the real engine on the snapshot index (real
// SigLIP 2 vectors from embed_server.py, SQLite for D1), scored against the oracle's answers —
// correctness, evidence, abstention, retrieval metrics for the semantic questions, latency — per
// category, language and mode. No reasoning model runs offline (Workers AI is not reachable from
// here): this measures the deterministic engine, which answers everything the operators can.
//
//   SNAPSHOT_DIR=…/snapshot-1005 STATE=…/s1005-v4.sqlite MODES=auto,fast,v3 OUT=… \
//     npx vitest run -c vitest.eval.config.ts eval/offline/v4.test.ts
//
// Env: MODES (auto,fast — "v3" runs v3's /ask on the same questions), LIMIT, ONLY (a category), OUT,
// DATASET (another question file: eval/v4/paraphrases.jsonl rewords 48 of them, a probe of phrasings
// the rules were not written from).

import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
import { readFileSync, writeFileSync } from "node:fs";
import { type TestEnv } from "../../test/harness/fakes";
import { ask } from "../../src/search/ask";
import { base64Decode } from "../../src/lib/bytes";
import { decodeImage, encodePng, sniff, type RgbaImage } from "../../src/enrich/decode";
import { queryImageOf } from "../../src/api/image";
import { loadItems, scoreAnswer, summarizeBy, type AnswerLike, type EvalItem, type ItemScore } from "../../src/evaluation/answer";
import { snapshotEnv } from "./env";

const SNAP = process.env.SNAPSHOT_DIR ?? "";
const run = SNAP ? describe : describe.skip;
const MODES: string[] = String(process.env.MODES ?? "auto,fast").split(",");

let env: TestEnv;
let chainPosts: Map<string, any>;

/** The image of a dataset item: a post's own bytes, the same picture at twice its size, or a novel one. */
async function imageOf(item: EvalItem): Promise<Uint8Array | null> {
  const im = item.image!;
  if (im.transform.startsWith("novel:")) {
    // seeded pixel art no artwork shows: blobs of four colours on a 24x24 grid
    let s = Number(im.transform.slice(6)) * 2654435761 + 12345;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) / 2 ** 32);
    const palette = [0, 1, 2, 3].map(() => [Math.floor(rnd() * 256), Math.floor(rnd() * 256), Math.floor(rnd() * 256)]);
    const w = 24;
    const data = new Uint8Array(w * w * 4);
    for (let y = 0; y < w; y++) for (let x = 0; x < w; x++) data.set([...palette[Math.floor(rnd() * 4) & (x % 3 ? 3 : 1)], 255], (y * w + x) * 4);
    return encodePng({ width: w, height: w, data });
  }
  const p = chainPosts.get(im.ref!);
  const m = /data:image\/(?:png|webp);base64,([A-Za-z0-9+/=\s]+)/.exec(p?.body ?? "");
  if (!m) return null;
  const bytes = base64Decode(m[1].replace(/\s+/g, ""));
  if (im.transform === "exact") return bytes;
  // scale2: nearest-neighbour at twice the size (a rescaled copy: other bytes, same picture); an
  // artwork too large to double within the 1024x1024 upload limit is re-encoded at its own size
  const img = await decodeImage(bytes, sniff(bytes));
  const f = img.width * img.height * 4 <= 1024 * 1024 ? 2 : 1;
  const out: RgbaImage = { width: img.width * f, height: img.height * f, data: new Uint8Array(img.width * img.height * 4 * f * f) };
  for (let y = 0; y < out.height; y++) for (let x = 0; x < out.width; x++) {
    const o = (Math.floor(y / f) * img.width + Math.floor(x / f)) * 4;
    out.data.set(img.data.subarray(o, o + 4), (y * out.width + x) * 4);
  }
  return encodePng(out);
}

run("/ask v4 on its question set", () => {
  beforeAll(async () => {
    env = await snapshotEnv({ snapshot: SNAP, state: process.env.STATE ?? "", vars: { SEARCH_TRACE_SAMPLE: "0" } });
    chainPosts = new Map(JSON.parse(readFileSync(`${SNAP}/chain_posts.json`, "utf8")).map((p: any) => [`/@${p.author}/${p.permlink}`, p]));
  }, 3_600_000);
  afterAll(() => vi.unstubAllGlobals());

  it("answers scored against the oracle", async () => {
    // DATASET: another question file of the same format (eval/v4/paraphrases.jsonl: the same questions reworded)
    let items = loadItems(readFileSync(process.env.DATASET ? process.env.DATASET : new URL("../../src/evaluation/datasets/questions.jsonl", import.meta.url), "utf8"));
    if (process.env.ONLY) items = items.filter((x) => x.query_type === process.env.ONLY);
    if (process.env.LIMIT) items = items.slice(0, Number(process.env.LIMIT));
    const report: Record<string, unknown> = {};
    const misses: Record<string, string[]> = {};
    for (const mode of MODES) {
      const scores: ItemScore[] = [];
      misses[mode] = [];
      for (const item of items) {
        if (mode === "v3" && item.image) continue; // v3 has no image questions
        const notes: string[] = [];
        let image;
        const t0 = Date.now();
        let r: AnswerLike;
        try {
          if (item.image) {
            const bytes = await imageOf(item);
            if (!bytes) continue;
            image = await queryImageOf(env, bytes, sniff(bytes), notes);
          }
          r = (await ask(env, { question: item.question, mode: mode === "auto" ? undefined : (mode as "fast" | "v3"), ...(image ? { image } : {}), noCache: true })) as unknown as AnswerLike;
          if (mode === "v3") r = { ...r, status: r.answer === null ? "no_match" : "answered", mode: "v3" };
        } catch (e) {
          r = { status: "error", answer: null, answer_text: e instanceof Error ? e.message : String(e), took_ms: Date.now() - t0 };
        }
        const s = scoreAnswer(item, r);
        scores.push(s);
        if (!s.correct) misses[mode].push(`${item.id} [${item.query_type}/${item.lang}] ${item.question}\n      want ${JSON.stringify(item.expected_status ?? item.expected_answer)} got ${JSON.stringify(r.status)} ${JSON.stringify(r.answer)} | ${String(r.answer_text).slice(0, 160)}`);
      }
      report[mode] = { summary: summarizeBy(scores), scores };
      for (const k of [...env._kv.m.keys()]) if (k.startsWith("s3:") || k.startsWith("llm4:")) env._kv.m.delete(k);
    }
    // the table
    const cats = ["factual", "semantic", "visual", "temporal", "comparative", "multi_hop", "ambiguous", "multilingual", "adversarial"];
    const lines = [`${"category".padEnd(13)}${"n".padStart(5)}  ${MODES.map((m: string) => `${m.padStart(6)} acc  rel  abst`).join("   ")}`];
    for (const c of [...cats, "ALL"]) {
      const cell = MODES.map((m: string) => {
        const sum = (report[m] as any).summary;
        const g = c === "ALL" ? sum.all : sum.by_type[c];
        return g ? `${g.accuracy.toFixed(3).padStart(10)} ${g.reliability.toFixed(2)} ${g.abstention_accuracy.toFixed(2)}` : `${"-".padStart(10)}            `;
      });
      const n = c === "ALL" ? items.length : items.filter((x) => x.query_type === c).length;
      lines.push(`${c.padEnd(13)}${String(n).padStart(5)}  ${cell.join("   ")}`);
    }
    for (const m of MODES) {
      const a = (report[m] as any).summary.all;
      lines.push(`${m}: engine P50 ${a.performance.engine_ms.p50} ms, P95 ${a.performance.engine_ms.p95} ms, P99 ${a.performance.engine_ms.p99} ms; confident errors ${a.confident_errors}; false answers ${a.false_answers}, false abstentions ${a.false_abstentions}; evidence accuracy ${a.evidence_accuracy?.toFixed(3)}${a.retrieval ? `; semantic nDCG@10 ${a.retrieval.ndcg10.toFixed(3)} MRR ${a.retrieval.mrr.toFixed(3)} MAP ${a.retrieval.ap.toFixed(3)} R@10 ${a.retrieval.r10.toFixed(3)}` : ""}`);
      const byLang = (report[m] as any).summary.by_lang;
      lines.push(`   by language: ${Object.entries(byLang).map(([k, g]: any) => `${k} ${g.accuracy.toFixed(3)} (${g.n})`).join(", ")}`);
    }
    console.log(lines.join("\n"));
    for (const m of MODES) console.log(`\n${m}: ${misses[m].length} misses\n${misses[m].slice(0, Number(process.env.SHOW_MISSES ?? 60)).join("\n")}`);
    if (process.env.OUT) {
      writeFileSync(process.env.OUT, JSON.stringify({ items: items.length, modes: MODES, report: Object.fromEntries(MODES.map((m: string) => [m, (report[m] as any).summary])), misses }, null, 1));
      writeFileSync(process.env.OUT.replace(/\.json$/, "") + ".scores.json", JSON.stringify(Object.fromEntries(MODES.map((m: string) => [m, (report[m] as any).scores]))));
    }
    expect(items.length).toBeGreaterThan(0);
  }, 6 * 3_600_000);
});
