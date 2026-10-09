// Admin routes (Authorization: Bearer ADMIN_TOKEN): the indexer, backfills, re-enrichment, ranker
// weights, the documentation index — and, v4, the /ask log and traces, the frozen-context model
// benchmark, the configured models, and the learning-to-rank and fine-tuning exports.

import { Hono } from "hono";
import { ALL_STAGES, BLOG_STAGES, bool, int, type Stage } from "../env";
import { exportPairs, exportSft, exportTraining, getWeights, setWeights } from "../search/feedback";
import { refreshBackground } from "../search/background";
import { planQuery } from "../search/planner";
import { indexerStub } from "../chain/indexer-do";
import { ingestPostRef, rpcFor } from "../chain/ingest";
import { parsePost } from "../chain/parse";
import { setJobsMany, getPostById, rebuildVocab } from "../db/posts";
import { base64Decode, base64Encode } from "../lib/bytes";
import { guessLang, type Lang } from "../lib/text";
import { decodeImage, encodePng, sniff } from "../enrich/decode";
import { factorFor, upscale } from "../enrich/upscale";
import { embedLabel, embeddingEnabled, getCalibration } from "../enrich/embed";
import { describeImage, isVlmBackend, VLM_BACKENDS } from "../enrich/describe";
import { sweep } from "../enrich/sweeper";
import { knownAuthors } from "../search/context";
import { answerHelp } from "../help/answer";
import { docsStatus, syncDocs } from "../docs/sync";
import { embedPendingChunks } from "../docs/vectors";
import { isReasoningLevel } from "../llm/provider";
import { endpoints, isModelId, KNOWN_MODELS, KNOWN_PRICES, modelSpec } from "../llm/model";
import { configuredReasoningModels, modelFor } from "../llm/router";
import { PROMPT_VERSION } from "../llm/prompts";
import type { EvidenceCard } from "../search/evidence";
import { freezeContext, runOnContext } from "../evaluation/benchmark";
import { paphAlerts, paphGc, paphHealWire3, paphRederive, paphStaleVerdicts, paphStatus } from "../paph/copies";
import { paphEnabled } from "../paph/shards";
import { isAdmin, type Bindings } from "./common";

export const admin = new Hono<Bindings>();

admin.use("*", async (c, next) => {
  if (!c.env.ADMIN_TOKEN) return c.json({ error: "ADMIN_TOKEN secret not set" }, 503);
  if (!isAdmin(c)) return c.json({ error: "unauthorized" }, 401);
  await next();
});

admin.get("/indexer", async (c) => c.json(await indexerStub(c.env).status()));
admin.post("/indexer/start", async (c) => {
  const from = Number(new URL(c.req.url).searchParams.get("from") ?? "");
  return c.json(await indexerStub(c.env).start(Number.isFinite(from) && from > 0 ? from : undefined));
});
admin.post("/indexer/stop", async (c) => c.json(await indexerStub(c.env).stop()));

/** {authors?, reason?, historyOnly?, noHistory?} */
admin.post("/backfill", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as { authors?: string[]; reason?: string; historyOnly?: boolean; noHistory?: boolean };
  const instance = await c.env.BACKFILL.create({ params: { authors: body.authors, reason: body.reason ?? "admin", historyOnly: !!body.historyOnly, noHistory: !!body.noHistory } });
  return c.json({ id: instance.id, status: await instance.status() });
});
admin.get("/backfill/:id", async (c) => {
  const instance = await c.env.BACKFILL.get(c.req.param("id"));
  return c.json({ id: instance.id, status: await instance.status() });
});

/** Re-ingest one post from the chain (and enqueue enrichment if its image changed). */
admin.post("/ingest/:author/:permlink", async (c) => {
  const r = await ingestPostRef(c.env, c.req.param("author").replace(/^@/, ""), c.req.param("permlink"), null, "admin");
  return c.json(r);
});

/**
 * Re-run enrichment. Body: { "post_id": 1 } | { "author": "x" } | { "all": true }, plus optional
 * "stages": ["stats","paph","embed","describe","concepts","text"] and "force": true. Blog posts only get
 * the text stage. Use after changing a model, the VLM, the concept vocabulary or EMBED_VIEWS.
 */
admin.post("/reindex", async (c) => {
  const body = (await c.req.json().catch(() => ({}))) as { post_id?: number; author?: string; all?: boolean; stages?: Stage[]; force?: boolean };
  const stages = (body.stages ?? ALL_STAGES).filter((s): s is Stage => ALL_STAGES.includes(s));
  let rows: Array<{ id: number; author: string; permlink: string; type: string }>;
  const sel = "SELECT id, author, permlink, type FROM posts WHERE deleted = 0";
  if (body.post_id) rows = (await c.env.DB.prepare(`${sel} AND id = ?`).bind(body.post_id).all<any>()).results ?? [];
  else if (body.author) rows = (await c.env.DB.prepare(`${sel} AND author = ?`).bind(body.author).all<any>()).results ?? [];
  else if (body.all) rows = (await c.env.DB.prepare(`${sel} ORDER BY net_votes DESC, created DESC`).all<any>()).results ?? [];
  else return c.json({ error: "give post_id, author or all:true" }, 400);
  const msgs = rows
    .map((r) => ({ r, s: r.type === "blog" ? stages.filter((s) => BLOG_STAGES.includes(s)) : stages }))
    .filter((x) => x.s.length)
    .map((x) => ({ body: { postId: x.r.id, author: x.r.author, permlink: x.r.permlink, stages: x.s, force: !!body.force, reason: "reindex" } }));
  for (let i = 0; i < msgs.length; i += 100) await c.env.ENRICH_QUEUE.sendBatch(msgs.slice(i, i + 100));
  await setJobsMany(c.env.DB, msgs.map((m) => ({ postId: m.body.postId, stages: m.body.stages })), "queued");
  return c.json({ enqueued: msgs.length, stages, force: !!body.force });
});

// ---- copy detection (src/paph) ------------------------------------------------------------------
// After a new @pixagram/paph-x release, profile or policy: rederive until remaining is 0 (keys and
// SI signatures from the stored wires, no image fetched), then re-check every artwork (reindex with
// stages ["paph"]: the stage sees the new identity, re-checks and reuses the stored wires; the
// sweeper also finds what is left), then purge the verdicts no re-check confirmed (refused while
// some artwork still waits for its re-check).

const paphOff = { error: "copy detection is not configured (no PAPH binding, or PAPH_ENABLED=false)" };

/** Shards (size, works, derivation), verdicts by kind and engine, the stage's progress, identity and budgets. */
admin.get("/paph", async (c) => (paphEnabled(c.env) ? c.json(await paphStatus(c.env)) : c.json(paphOff, 503)));

/** Cross-author copies (Copy and up) found in the last `days`: the earlier work is the candidate original. */
admin.get("/paph/alerts", async (c) => {
  if (!paphEnabled(c.env)) return c.json(paphOff, 503);
  const sp = new URL(c.req.url).searchParams;
  return c.json({ items: await paphAlerts(c.env, int(sp.get("days") ?? undefined, 30), int(sp.get("limit") ?? undefined, 100)) });
});

/** Re-derive keys and signatures in every shard, `limit` works per shard per call: repeat until remaining is 0. */
admin.post("/paph/rederive", async (c) => {
  if (!paphEnabled(c.env)) return c.json(paphOff, 503);
  return c.json(await paphRederive(c.env, Math.min(2000, Math.max(1, int(new URL(c.req.url).searchParams.get("limit") ?? undefined, 200)))));
});

/** Verdicts reached by another engine or policy than the current one; POST purge=1 deletes them (force=1: even before every re-check). */
admin.on(["GET", "POST"], "/paph/stale", async (c) => {
  if (!paphEnabled(c.env)) return c.json(paphOff, 503);
  const sp = new URL(c.req.url).searchParams;
  return c.json(await paphStaleVerdicts(c.env, { purge: c.req.method === "POST" && sp.get("purge") === "1", force: sp.get("force") === "1" }));
});

/** Compare again on the current wire the verdicts reached on wire 3 that no re-check will replace (the nightly cron does it too). */
admin.post("/paph/heal", async (c) => {
  if (!paphEnabled(c.env)) return c.json(paphOff, 503);
  return c.json(await paphHealWire3(c.env, Math.min(1000, Math.max(1, int(new URL(c.req.url).searchParams.get("limit") ?? undefined, 100)))));
});

/** Remove indexed works whose posts were deleted, a page per shard (the nightly cron does it too). */
admin.post("/paph/gc", async (c) => {
  if (!paphEnabled(c.env)) return c.json(paphOff, 503);
  return c.json(await paphGc(c.env, Math.min(5000, Math.max(1, int(new URL(c.req.url).searchParams.get("limit") ?? undefined, 500)))));
});

/** Enqueue whatever is missing or stale (the cron does this every 10 minutes too). */
admin.post("/sweep", async (c) => c.json(await sweep(c.env, int(new URL(c.req.url).searchParams.get("max") ?? undefined, 200))));

/** Rebuild the spelling vocabulary from the search docs (after an in-place upgrade from v2). */
admin.post("/vocab/rebuild", async (c) => c.json(await rebuildVocab(c.env.DB)));

/** Re-sample the background vectors used to normalise semantic scores. */
admin.post("/background", async (c) => c.json({ image: await refreshBackground(c.env, "image"), text: await refreshBackground(c.env, "text") }));

admin.get("/stats", async (c) => {
  const [posts, jobs, art, last, ver, conc] = await c.env.DB.batch([
    c.env.DB.prepare("SELECT type, deleted, COUNT(*) AS n FROM posts GROUP BY type, deleted"),
    c.env.DB.prepare("SELECT stage, status, COUNT(*) AS n FROM jobs GROUP BY stage, status"),
    // live artworks only (a deleted post keeps its artworks row, without vectors)
    c.env.DB.prepare(
      `SELECT COUNT(*) AS n, SUM(a.embed_hash = a.content_hash) AS embedded, SUM(a.describe_hash = a.content_hash AND COALESCE(a.ai_caption, '') != '') AS described,
              SUM(a.ai_status = 'caption_only') AS caption_only, SUM(a.features_hash = a.content_hash) AS featured, SUM(a.concepts_hash IS NOT NULL) AS with_concepts,
              SUM(a.history_exact = 1) AS history_exact, SUM(a.lossy) AS lossy
       FROM artworks a JOIN posts p ON p.id = a.post_id WHERE p.deleted = 0`,
    ),
    c.env.DB.prepare("SELECT v FROM settings WHERE k = 'backfill:last'"),
    c.env.DB.prepare("SELECT source, COUNT(*) AS n FROM post_versions GROUP BY source"),
    c.env.DB.prepare("SELECT COUNT(DISTINCT concept) AS concepts, COUNT(*) AS rows FROM artwork_concepts"),
  ]);
  return c.json({
    posts: posts.results,
    jobs: jobs.results,
    artworks: art.results?.[0],
    versions: ver.results,
    concepts: conc.results?.[0],
    backfill_last: last.results?.[0] ? JSON.parse((last.results[0] as any).v) : null,
    indexer: await (async () => indexerStub(c.env).status())().catch((e: unknown) => ({ error: e instanceof Error ? e.message : String(e) })),
    config: {
      semantic: embeddingEnabled(c.env), embed_model: c.env.EMBED_MODEL, embed_dim: int(c.env.EMBED_DIM, 768), embed_views: embedLabel(c.env),
      text_vectors: bool(c.env.TEXT_VECTORS, true) && !!c.env.VEC_TEXT, calibration: await getCalibration(c.env),
      vlm: c.env.VLM_BACKEND, planner: c.env.PLANNER_BACKEND ?? "auto", scaler: c.env.SCALER, store_in_r2: bool(c.env.STORE_IN_R2, true),
      ai_training_false_blocks: c.env.AI_TRAINING_FALSE_BLOCKS ?? (bool(c.env.RESPECT_AI_TRAINING_FLAG, false) ? "describe" : ""),
      models: { planner: modelFor(c.env, "planner"), help: modelFor(c.env, "help"), reranker: modelFor(c.env, "reranker"), reasoning: configuredReasoningModels(c.env) },
    },
  });
});

/** Zero-result queries in the last N days (missing synonyms and concepts), plus the most frequent ones. */
admin.get("/queries", async (c) => {
  const days = int(new URL(c.req.url).searchParams.get("days") ?? undefined, 7);
  const since = Math.floor(Date.now() / 1000) - days * 86400;
  const zero = await c.env.DB.prepare("SELECT q, COUNT(*) AS n FROM query_log WHERE at >= ? AND results = 0 GROUP BY q ORDER BY n DESC LIMIT 100").bind(since).all();
  const top = await c.env.DB.prepare("SELECT q, COUNT(*) AS n, AVG(ms) AS avg_ms FROM query_log WHERE at >= ? GROUP BY q ORDER BY n DESC LIMIT 100").bind(since).all();
  // Query words that map to no concept: candidates for the vocabulary.
  const unknown = new Map<string, number>();
  for (const r of (top.results ?? []) as Array<{ q: string; n: number }>) {
    const plan = planQuery(r.q ?? "", { mode: "search", authors: await knownAuthors(c.env) });
    if (!plan.concepts.length) for (const t of plan.lexicalTerms) unknown.set(t, (unknown.get(t) ?? 0) + r.n);
  }
  return c.json({ days, zero_results: zero.results, top: top.results, words_without_concept: [...unknown.entries()].sort((a, b) => b[1] - a[1]).slice(0, 50) });
});

admin.get("/jobs/failed", async (c) => {
  const r = await c.env.DB.prepare("SELECT j.post_id, p.author, p.permlink, j.stage, j.attempts, j.error, j.updated FROM jobs j JOIN posts p ON p.id = j.post_id WHERE j.status = 'failed' ORDER BY j.updated DESC LIMIT 200").all();
  return c.json(r.results);
});

admin.get("/ranker/weights", async (c) => c.json(await getWeights(c.env)));
admin.post("/ranker/weights", async (c) => {
  const r = await setWeights(c.env, await c.req.json().catch(() => null));
  return r.ok ? c.json(r) : c.json(r, 400);
});

/** Training data for scripts/train-ranker.py: shown results with features and engagement labels. */
admin.get("/ltr/export", async (c) => {
  const days = int(new URL(c.req.url).searchParams.get("days") ?? undefined, 30);
  return c.json(await exportTraining(c.env, days));
});

/** Pairwise preferences (a result acted on over the ones skipped above it), with both feature vectors. */
admin.get("/ltr/pairs", async (c) => {
  const days = int(new URL(c.req.url).searchParams.get("days") ?? undefined, 30);
  return c.json(await exportPairs(c.env, days));
});

/**
 * Run the VLM on one artwork and show the raw reply — the quickest way to see what the model
 * returns. ?write=1 also stores the description (as the describe stage would).
 */
admin.post("/debug/describe/:id", async (c) => {
  const id = Number(c.req.param("id"));
  const asked = new URL(c.req.url).searchParams.get("backend")?.toLowerCase();
  if (asked && !isVlmBackend(asked)) return c.json({ error: `backend must be one of ${VLM_BACKENDS.join(", ")}` }, 400);
  // without ?backend: the stack's own, or Gemma when descriptions are off
  const configured = (c.env.VLM_BACKEND ?? "gemma").toLowerCase();
  const backend = asked && isVlmBackend(asked) ? asked : isVlmBackend(configured) ? configured : "gemma";
  const p = await getPostById(c.env.DB, id);
  if (!p || p.type !== "artwork") return c.json({ error: "not an artwork" }, 404);
  const chain = await rpcFor(c.env).getContent(p.author, p.permlink);
  const parsed = chain ? parsePost(chain) : null;
  if (!parsed?.image?.supported) return c.json({ error: "no supported image on chain" }, 422);
  const bytes = base64Decode(parsed.image.base64);
  const img = await decodeImage(bytes, sniff(bytes));
  const png = await encodePng(upscale(img, factorFor(img.width, img.height, int(c.env.VLM_TARGET, 512)), (c.env.SCALER ?? "xbrz") === "nearest" ? "nearest" : "xbrz"));
  try {
    const r = await describeImage(c.env, backend, `data:image/png;base64,${base64Encode(png)}`, { title: p.title, tags: JSON.parse(p.tags_json || "[]"), description: p.description });
    if (new URL(c.req.url).searchParams.get("write") === "1") {
      await c.env.ENRICH_QUEUE.send({ postId: id, author: p.author, permlink: p.permlink, stages: ["describe", "concepts", "text"], force: true, reason: "debug" });
    }
    return c.json({ ok: true, backend, png_bytes: png.length, ...r });
  } catch (e) {
    return c.json({ ok: false, backend, png_bytes: png.length, error: e instanceof Error ? e.message : String(e), raw: (e as any)?.raw ?? null }, 502);
  }
});

// ---- admin: documentation ----------------------------------------------------------------------

/** The documentation index: commit, files per status, chunks and vectors, failures, last sync. */
admin.get("/docs", async (c) => c.json(await docsStatus(c.env)));

/** Sync now. ?force=1 re-indexes every file (vectors of unchanged chunks are kept). */
admin.post("/docs/sync", async (c) => c.json(await syncDocs(c.env, { force: new URL(c.req.url).searchParams.get("force") === "1", reason: "admin" })));

/** Re-embed every chunk (after changing DOCS_EMBED_MODEL to one with the same dimensions). */
admin.post("/docs/reembed", async (c) => {
  await c.env.DB.prepare("UPDATE doc_chunks SET embedded = NULL").run();
  return c.json({ embedded: await embedPendingChunks(c.env, 2000, Date.now() + 25_000), note: "the rest is embedded by the next syncs" });
});

/**
 * /help answered by another model, to compare models on the same documentation before changing
 * HELP_MODEL: ?q=…&model=@cf/nvidia/nemotron-3-120b-a12b (&mode=…&reasoning=…). Not recorded in the help log.
 */
admin.get("/debug/help", async (c) => {
  const u = new URL(c.req.url);
  const q = (u.searchParams.get("q") ?? "").trim();
  const model = (u.searchParams.get("model") ?? "").trim();
  if (!q) return c.json({ error: "q is required" }, 400);
  if (model && !/^@(cf|hf)\/[\w.-]+\/[\w.-]+$/.test(model) && !isModelId(model, c.env)) return c.json({ error: "model must be a Workers AI model id, e.g. @cf/nvidia/nemotron-3-120b-a12b, or one of SEARCH_LLM_ENDPOINTS" }, 400);
  const reasoning = u.searchParams.get("reasoning");
  return c.json(await answerHelp(c.env, q, { model: model || undefined, log: false, mode: u.searchParams.get("mode"), reasoning: isReasoningLevel(reasoning) ? reasoning : undefined }));
});

/** Help questions the documentation could not answer (the repository's to-do list), most asked first. */
admin.get("/docs/gaps", async (c) => {
  const days = int(new URL(c.req.url).searchParams.get("days") ?? undefined, 30);
  const since = Math.floor(Date.now() / 1000) - days * 86400;
  const [gaps, byStatus] = await c.env.DB.batch([
    c.env.DB.prepare(
      `SELECT LOWER(TRIM(q)) AS q, COUNT(*) AS n, MAX(at) AS last, MAX(status) AS status, MAX(top_path) AS closest FROM help_log
       WHERE at >= ? AND status IN ('not_found', 'no_docs', 'excerpts') GROUP BY LOWER(TRIM(q)) ORDER BY n DESC, last DESC LIMIT 200`,
    ).bind(since),
    c.env.DB.prepare("SELECT status, COUNT(*) AS n FROM help_log WHERE at >= ? GROUP BY status").bind(since),
  ]);
  return c.json({ days, by_status: byStatus.results, unanswered: gaps.results });
});

// ---- admin: v4 answers, traces, models, benchmarks ---------------------------------------------------

/** The /ask log: ?days=7&status=…&mode=…&limit=100, newest first, with totals (cost, tokens, grounding). */
admin.get("/ask/log", async (c) => {
  const sp = new URL(c.req.url).searchParams;
  const days = int(sp.get("days") ?? undefined, 7);
  const limit = Math.min(500, Math.max(1, int(sp.get("limit") ?? undefined, 100)));
  const since = Math.floor(Date.now() / 1000) - days * 86400;
  const where = ["at >= ?"];
  const params: unknown[] = [since];
  for (const k of ["status", "mode", "class", "model"] as const) {
    const v = sp.get(k);
    if (v) (where.push(`${k} = ?`), params.push(v));
  }
  const w = where.join(" AND ");
  const [rows, totals, byStatus] = await c.env.DB.batch([
    c.env.DB.prepare(`SELECT qid, q, lang, class, complexity, mode, reasoning, model, status, answer, confidence, egs, claims, claims_supported, input_tokens, output_tokens, cost_usd, model_ms, took_ms, versions, trace IS NOT NULL AS traced, at FROM ask_log WHERE ${w} ORDER BY at DESC LIMIT ?`).bind(...params, limit),
    c.env.DB.prepare(`SELECT COUNT(*) AS n, SUM(model IS NOT NULL) AS reasoned, SUM(cost_usd) AS cost_usd, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, AVG(egs) AS egs, AVG(took_ms) AS avg_ms FROM ask_log WHERE ${w}`).bind(...params),
    c.env.DB.prepare(`SELECT status, mode, COUNT(*) AS n FROM ask_log WHERE ${w} GROUP BY status, mode ORDER BY n DESC`).bind(...params),
  ]);
  const list = ((rows.results ?? []) as Array<Record<string, unknown>>).map((r) => ({ ...r, versions: typeof r.versions === "string" ? safeJson(r.versions) : null, traced: !!r.traced }));
  return c.json({ days, totals: totals.results?.[0] ?? {}, by_status: byStatus.results, rows: list });
});

/** One logged question with its stored trace (a SEARCH_TRACE_SAMPLE share of questions keep one), and the votes on it. */
admin.get("/ask/trace/:qid", async (c) => {
  const qid = c.req.param("qid").slice(0, 32);
  const row = await c.env.DB.prepare("SELECT * FROM ask_log WHERE qid = ? ORDER BY id DESC LIMIT 1").bind(qid).first<Record<string, unknown>>();
  if (!row) return c.json({ error: "not found" }, 404);
  const votes = await c.env.DB.prepare("SELECT rating, reason, at FROM answer_feedback WHERE qid = ? ORDER BY at").bind(qid).all();
  return c.json({ ...row, versions: typeof row.versions === "string" ? safeJson(row.versions) : null, trace: typeof row.trace === "string" ? safeJson(row.trace) : null, feedback: votes.results ?? [] });
});

/**
 * The frozen context of a question (spec §31): retrieval, operators and evidence verification run,
 * the reasoning model does not. POST {question, mode?, nsfw?, type?} → the exact cards, context
 * notes and language a model would be given, plus the deterministic answer. Give the same context to
 * several models with POST /admin/ask/reason (scripts/benchmark.py does both).
 */
admin.post("/ask/context", async (c) => {
  const b = ((await c.req.json().catch(() => ({}))) ?? {}) as Record<string, unknown>;
  const question = typeof b.question === "string" ? b.question.trim().slice(0, 300) : "";
  if (!question) return c.json({ error: "question is required" }, 400);
  const mode = b.mode === "fast" || b.mode === "balanced" || b.mode === "expert" ? b.mode : "deep";
  return c.json(await freezeContext(c.env, question, { mode, nsfw: b.nsfw === "include" ? "include" : "exclude", type: b.type === "blog" ? "blog" : b.type === "artwork" ? "artwork" : undefined }));
});

/**
 * One model on a frozen context: POST {question, lang, cards, context, shown?, model, reasoning?, max_output_tokens?, strict?}
 * → the reply, its claims verified against the same cards, grounding, whether it states the
 * index's answer (agreement), tokens, cost and latency.
 * Never cached, never logged: for comparing models on exactly the same input.
 */
admin.post("/ask/reason", async (c) => {
  const b = ((await c.req.json().catch(() => ({}))) ?? {}) as Record<string, unknown>;
  const question = typeof b.question === "string" ? b.question.trim().slice(0, 300) : "";
  const cards = Array.isArray(b.cards) ? (b.cards as EvidenceCard[]).filter((x) => x && typeof x === "object" && typeof (x as any).evidence_id === "string").slice(0, 60) : [];
  const model = typeof b.model === "string" ? b.model.trim() : modelFor(c.env, "reasoning");
  if (!question) return c.json({ error: "question is required" }, 400);
  if (!isModelId(model, c.env)) return c.json({ error: `unknown model ${model}` }, 400);
  const lang = (typeof b.lang === "string" ? b.lang : guessLang(question)) as Lang;
  const context = Array.isArray(b.context) ? (b.context as unknown[]).filter((x): x is string => typeof x === "string").slice(0, 20) : [];
  const shown = Array.isArray(b.shown) ? (b.shown as unknown[]).filter((x): x is string => typeof x === "string").slice(0, 12) : [];
  const run = await runOnContext(c.env, { question, lang, cards, context, shown }, model, { reasoning: isReasoningLevel(b.reasoning) ? b.reasoning : "medium", maxTokens: Number(b.max_output_tokens) || 2000, strict: !!b.strict });
  return run.error ? c.json(run, 502) : c.json(run);
});

/** The models this stack is configured with, the model table (style, reasoning control, context, prices), endpoints (names only). */
admin.get("/models", (c) =>
  c.json({
    roles: { planner: modelFor(c.env, "planner"), help: modelFor(c.env, "help"), reranker: modelFor(c.env, "reranker"), reasoning: configuredReasoningModels(c.env) },
    public_models: (c.env.SEARCH_PUBLIC_MODELS ?? "").split(",").map((s) => s.trim()).filter(Boolean),
    known: Object.keys(KNOWN_MODELS).map((id) => {
      const s = modelSpec(id, c.env);
      return { id, style: s.style, reasoning: s.reasoning, can_disable_reasoning: s.canDisableReasoning, context_tokens: s.contextTokens, vision: s.vision, price: s.price ?? null };
    }),
    other_prices: KNOWN_PRICES,
    endpoints: Object.keys(endpoints(c.env)),
    prompt_version: PROMPT_VERSION,
  }),
);

/** Fine-tuning examples: verified, sampled answers nobody voted down (?days=30&min_egs=1). */
admin.get("/ask/export-sft", async (c) => {
  const sp = new URL(c.req.url).searchParams;
  const minEgs = Number(sp.get("min_egs") ?? 1);
  return c.json(await exportSft(c.env, int(sp.get("days") ?? undefined, 30), { minEgs: Number.isFinite(minEgs) ? minEgs : 1, limit: int(sp.get("limit") ?? undefined, 1000) }));
});

function safeJson(s: string): unknown {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

export { isAdmin };
