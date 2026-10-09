// Dump every candidate of every judged query with its ranking features (for weight fitting:
// eval/offline/fit_weights.py). Reuses the index built by run.test.ts (STATE).
import { it } from "vitest";
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { codecsReady } from "../../test/helpers";
import { FakeD1 } from "../../test/harness/d1";
import { FakeExec, makeEnv } from "../../test/harness/fakes";
import { refreshBackground } from "../../src/search/background";
import { parseSearchRequest } from "../../src/search/params";
import { loadContext } from "../../src/search/context";
import { rankRetrieval, retrieve } from "../../src/search/service";
import { isColorLed } from "../../src/search/ranker";

const STATE = process.env.STATE ?? "";

it.skipIf(!STATE || !existsSync(STATE))("dump candidate features", async () => {
  await codecsReady();
  const env = makeEnv({ db: new FakeD1(STATE), HF_EMBED_URL: process.env.EMBED_URL ?? "http://127.0.0.1:7861/embed", EMBED_MODEL: process.env.EMBED_MODEL ?? "google/siglip2-base-patch16-naflex", EMBED_PATCHES: process.env.EMBED_PATCHES ?? "576", EMBED_DIM: "768" });
  const saved = JSON.parse(readFileSync(`${STATE}.vectors.json`, "utf8"));
  for (const [k, v] of Object.entries(saved.image)) env._vec.v.set(k, v as any);
  for (const [k, v] of Object.entries(saved.text)) env._vecText.v.set(k, v as any);
  for (const [k, v] of Object.entries(saved.kv ?? {})) env._kv.m.set(k, v as string);
  await refreshBackground(env, "image");
  await refreshBackground(env, "text");
  const ctx = await loadContext(env);
  const queries = readFileSync(new URL("../queries.jsonl", import.meta.url), "utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
  const out: string[] = [];
  for (const [qi, x] of queries.entries()) {
    const r = parseSearchRequest(new URLSearchParams({ q: x.q, limit: "50", nsfw: "include", type: x.type ?? "artwork" }));
    const ret = await retrieve(env, r, ctx, { need: 50 });
    const ranked = await rankRetrieval(env, ret, ctx, new FakeExec() as any);
    const colorLed = isColorLed(ret.plan);
    for (const c of ranked) {
      const row = ret.rows.get(c.id)!;
      const ref = `${row.author}/${row.permlink}`;
      out.push(JSON.stringify({ qi, q: x.q, cat: x.cat, ref, rel: x.rel[ref] ?? 0, colorLed, f: c.features, z: c.raw, quality: c.quality, freshness: c.freshness, dup: c.duplicateOf ?? null }));
    }
  }
  writeFileSync(process.env.DUMP ?? "candidates.jsonl", out.join("\n"));
  console.log(`dumped ${out.length} candidates for ${queries.length} queries`);
}, 600_000);
