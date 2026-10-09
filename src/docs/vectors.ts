// Embeddings of the documentation (Workers AI, multilingual: a French question finds an English
// page) and the vectors' upkeep in VEC_DOCS.

import type { Env } from "../env";
import { int } from "../env";
import { normQuery } from "../enrich/embed";
import { aiReply } from "../lib/ai";

export const DEFAULT_DOCS_EMBED_MODEL = "@cf/baai/bge-m3"; // 1024 dimensions, 100+ languages
const BATCH = 16;
const MAX_CHARS = 6000;

export const docsEmbedModel = (env: Env) => (env.DOCS_EMBED_MODEL || DEFAULT_DOCS_EMBED_MODEL).trim();

/** What a chunk's vector is computed from: its document title and heading path give it context. */
export function chunkEmbedText(c: { title: string; heading: string; text: string }): string {
  return [c.title, c.heading, c.text].filter(Boolean).join("\n").slice(0, MAX_CHARS);
}

export async function embedDocTexts(env: Env, texts: string[]): Promise<number[][]> {
  const ai = env.AI as unknown as { run: (model: string, input: unknown) => Promise<any> };
  const out: number[][] = [];
  for (let i = 0; i < texts.length; i += BATCH) {
    const batch = texts.slice(i, i + BATCH).map((t) => t.slice(0, MAX_CHARS) || " ");
    const r = await aiReply(await ai.run(docsEmbedModel(env), { text: batch }));
    const data: unknown = r?.data ?? r?.result?.data;
    if (!Array.isArray(data) || data.length !== batch.length || !data.every((v) => Array.isArray(v) && v.length > 0)) {
      throw new Error(`docs embedding: unexpected reply from ${docsEmbedModel(env)}: ${JSON.stringify(r).slice(0, 160)}`);
    }
    const dim = int(env.DOCS_EMBED_DIM, 0);
    if (dim && (data as number[][])[0].length !== dim) {
      throw new Error(`docs embedding: ${docsEmbedModel(env)} gives ${(data as number[][])[0].length} dimensions, DOCS_EMBED_DIM (and VEC_DOCS) ${dim}`);
    }
    out.push(...(data as number[][]));
  }
  return out;
}

/** A question's vector, cached in KV like the SigLIP query vectors. */
export async function embedQuestion(env: Env, question: string): Promise<number[]> {
  const key = `demb:${docsEmbedModel(env)}:${normQuery(question)}`;
  const hit = await env.CACHE.get(key, "json").catch(() => null);
  if (Array.isArray(hit)) return hit as number[];
  const [v] = await embedDocTexts(env, [normQuery(question)]);
  await env.CACHE.put(key, JSON.stringify(v), { expirationTtl: 7 * 24 * 3600 }).catch(() => {});
  return v;
}

/**
 * Embed chunks that have no vector from the current model (new chunks, or all of them after
 * DOCS_EMBED_MODEL changed). Bounded per call; the rest follows on the next sync.
 */
export async function embedPendingChunks(env: Env, maxChunks = 200, deadline = Date.now() + 20_000): Promise<number> {
  if (!env.VEC_DOCS || !env.AI) return 0;
  const model = docsEmbedModel(env);
  let done = 0;
  while (done < maxChunks && Date.now() < deadline) {
    const rows =
      (
        await env.DB.prepare("SELECT id, path, title, heading, text, lang FROM doc_chunks WHERE embedded IS NOT ? ORDER BY id LIMIT ?")
          .bind(model, Math.min(BATCH * 2, maxChunks - done))
          .all<{ id: number; path: string; title: string; heading: string; text: string; lang: string | null }>()
      ).results ?? [];
    if (!rows.length) break;
    const vectors = await embedDocTexts(env, rows.map(chunkEmbedText));
    await env.VEC_DOCS.upsert(rows.map((r, i) => ({ id: String(r.id), values: vectors[i], metadata: { path: r.path, lang: r.lang ?? "" } })));
    await env.DB.prepare(`UPDATE doc_chunks SET embedded = ? WHERE id IN (${rows.map((r) => Math.trunc(r.id)).join(",")})`).bind(model).run();
    done += rows.length;
  }
  return done;
}
