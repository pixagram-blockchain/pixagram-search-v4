// Queue consumer: per-post enrichment. Stages are idempotent and keyed on content hashes, so a
// retried message (or a re-run after a model change) only redoes what is missing.
//
//   artwork:
//   decode ─► native stats, features, pHash/dHash/bands, colours ─► D1      (stats)
//      ├────► PAPH fingerprint ─► copy index, checked in every shard ─► D1   (paph)
//      ├────► views (xBRZ by default) ─► SigLIP image vector ─► VEC          (embed)
//      └────► xBRZ preview ─► R2, and ─► VLM description ─► D1 + FTS         (describe)
//   tags/title/description/AI ─► concepts ─► artwork_concepts               (concepts)
//   title + caption + tags     ─► SigLIP text vector ─► VEC_TEXT            (text)
//   blog: title + body ─► SigLIP text vector ─► VEC_TEXT                    (text)
//
// The image path no longer depends on the VLM path: an embedding failure does not block the
// description and vice versa; each stage records its own job status.

import type { Env, EnrichMessage, Stage } from "../env";
import { ALL_STAGES, aiTrainingBlocks, bool, int, now } from "../env";
import { base64Decode, base64Encode, sha256Hex } from "../lib/bytes";
import { fnv1a } from "../lib/text";
import { rpcFor } from "../chain/ingest";
import { parsePost } from "../chain/parse";
import { refreshImageHistory, snapshotVersion, writeVersions } from "../chain/versions";
import { getArtwork, getPostById, setJob, settleJob, stripMarkdown, updateSearchDocAi, vectorId, type ArtworkRow, type PostRow } from "../db/posts";
import { refreshVectorMetadata, textVectorMetadata, vectorMetadata } from "./vector-meta";
import { decodeImage, encodePng, sniff, type ContainerInfo, type RgbaImage } from "./decode";
import { computeStats } from "./stats";
import { computeFeatures } from "./features";
import { dhash, halves, phash, phashBands } from "./phash";
import { factorFor, upscale, upscaleNearest, type Scaler } from "./upscale";
import { EmbedUnavailable, embedAndRemember, embedLabel, embeddingEnabled, mixVectors, parseViews, spaceSlots, type ViewName } from "./embed";
import { describeImage, EmptyDescription, isVlmBackend, vlmConfigError } from "./describe";
import { extractArtworkConcepts } from "../concepts";
import { paphPermanent, paphStage, paphVectorPass } from "../paph/copies";
import { paphEnabled } from "../paph/shards";

type StageResult = "done" | "skipped" | "failed" | "unchanged" | "retry";

export interface EnrichReport {
  postId: number;
  type?: "artwork" | "blog";
  hash?: string;
  stats: StageResult;
  paph: StageResult;
  embed: StageResult;
  describe: StageResult;
  concepts: StageResult;
  text: StageResult;
  error?: string;
}

class Retry extends Error {
  constructor(msg: string, public readonly delaySeconds: number) {
    super(msg);
  }
}

/**
 * Cloudflare stops a queue consumer run after 15 minutes of wall time. A run plans to be done with
 * its waits by RUN_MS: no message starts later (it is retried), and the paph stage — the one stage
 * that waits on other objects for long (a busy or hung shard) — cuts its check to end by then, or
 * defers it. What follows the paph stage in a message (embedding, description) has the rest.
 */
export const RUN_MS = 12 * 60_000;
/** A message the run had no time to start (or a check it had no time for) comes back after this. */
const NOT_STARTED_DELAY_S = 30;

/** When the current run must be done waiting (paphStage's `endsAt`). */
export interface RunOptions {
  endsAt?: number;
}

/**
 * The messages of a batch, as many at once as the Space computes embeddings at once (one per CPU:
 * an upgraded Space indexes faster, without a deploy). The Space queues what it cannot start yet,
 * and the other consumer invocation sends as many, so it does not sit idle while a description
 * (Workers AI) or a database write is under way.
 */
export async function handleEnrichBatch(batch: MessageBatch<EnrichMessage>, env: Env): Promise<void> {
  const endsAt = Date.now() + RUN_MS;
  const slots = embeddingEnabled(env) ? await spaceSlots(env).catch(() => 2) : 1;
  let next = 0;
  const worker = async () => {
    while (next < batch.messages.length) {
      const msg = batch.messages[next++];
      if (Date.now() >= endsAt) await notStarted(msg, env);
      else await enrichMessage(msg, env, { endsAt });
    }
  };
  await Promise.all(Array.from({ length: Math.max(1, Math.min(slots, batch.messages.length)) }, worker));
}

/**
 * A message the run had no time for comes back later as a new message: it was not tried, so it
 * does not use up one of its deliveries (the queue sends a message to the dead-letter queue after
 * 8). Should the send fail, the queue retries it as usual.
 */
async function notStarted(msg: Message<EnrichMessage>, env: Env): Promise<void> {
  try {
    await env.ENRICH_QUEUE.send(msg.body, { delaySeconds: NOT_STARTED_DELAY_S });
    msg.ack();
  } catch (e) {
    console.warn("enrich: could not send back a message the run had no time for", msg.body.postId, errMsg(e));
    msg.retry({ delaySeconds: NOT_STARTED_DELAY_S });
  }
}

async function enrichMessage(msg: Message<EnrichMessage>, env: Env, o: RunOptions): Promise<void> {
  try {
    const report = await enrichOne(env, msg.body, o);
    console.log("enrich", JSON.stringify(report));
    msg.ack();
  } catch (e) {
    if (e instanceof Retry) {
      console.warn("enrich retry", msg.body.postId, e.message);
      msg.retry({ delaySeconds: Math.min(e.delaySeconds * Math.max(1, msg.attempts), 3600) });
    } else if (isPermanent(e)) {
      // Undecodable / unsupported image: recorded in `jobs`, no point retrying.
      console.error("enrich permanent failure", msg.body.postId, errMsg(e));
      msg.ack();
    } else {
      console.error("enrich failed", msg.body.postId, e instanceof Error ? e.stack ?? e.message : e);
      msg.retry({ delaySeconds: Math.min(60 * Math.max(1, msg.attempts), 3600) });
    }
  }
}

const emptyReport = (postId: number): EnrichReport => ({ postId, stats: "skipped", paph: "skipped", embed: "skipped", describe: "skipped", concepts: "skipped", text: "skipped" });

export async function enrichOne(env: Env, m: EnrichMessage, o: RunOptions = {}): Promise<EnrichReport> {
  const stages = new Set<Stage>(m.stages?.length ? m.stages : ALL_STAGES);
  const report = emptyReport(m.postId);
  const post = await getPostById(env.DB, m.postId);
  if (!post || post.deleted) {
    for (const s of stages) await setJob(env.DB, m.postId, s, "skipped", "not an active post");
    return report;
  }
  report.type = post.type;
  if (post.type === "blog") {
    if (stages.has("text")) report.text = await textStage(env, post, null, !!m.force);
    return report;
  }

  let pendingRetry: Retry | null = null;
  let art = await getArtwork(env.DB, m.postId);
  const blocked = post.ai_training === 0 ? aiTrainingBlocks(env) : new Set<Stage>();
  const imageStages = (["stats", "embed", "describe"] as Stage[]).filter((s) => stages.has(s));
  // the paph stage runs inside the image pipeline (on the pixels stats decoded) when that runs;
  // alone (a re-check after a release) it reuses the stored wires and fetches the image only if needed
  let paphDone = false;

  if (imageStages.length) {
    // The image lives on chain: re-read the post (json_metadata is tiny, body is the data URI).
    const chainPost = await rpcFor(env).getContent(post.author, post.permlink);
    if (!chainPost) throw new Error(`post ${post.author}/${post.permlink} not found on chain`);
    const parsed = parsePost(chainPost);
    // an image no stage can read is no image for copy detection either
    const unusable = (why: string) => Promise.all([...imageStages, ...(stages.has("paph") ? (["paph"] as Stage[]) : [])].map((s) => setJob(env.DB, m.postId, s, "skipped", why)));
    if (!parsed.image || !parsed.image.supported) {
      await unusable(parsed.image ? `unsupported mime ${parsed.image.mime}` : "no image payload");
      paphDone = true;
    } else {
      const bytes = base64Decode(parsed.image.base64);
      const hash = await sha256Hex(bytes);
      report.hash = hash;
      const container = sniff(bytes);
      if (container.format === "unknown") {
        await unusable("unrecognised container");
        paphDone = true;
      } else {
        const r = await imagePipeline(env, m, post, art, bytes, hash, container, stages, blocked, report, o);
        pendingRetry = r.retry;
        paphDone = true;
        art = await getArtwork(env.DB, m.postId);
        // A new description feeds the concepts and the text vector: refresh them in the same run
        // even when the message only asked for "describe" (sweeper, debug). Both are hash-keyed,
        // so they are no-ops when nothing they read changed.
        if (report.describe === "done") (stages.add("concepts"), stages.add("text"));
      }
    }
  }

  if (stages.has("paph") && !paphDone) {
    if (!art?.content_hash) {
      // not yet: the stats stage first (no job record — "skipped" would keep the sweeper away for good)
    } else {
      const known = art;
      let img: RgbaImage | null = null;
      const fetchImage = async (): Promise<RgbaImage> => {
        if (img) return img;
        const chainPost = await rpcFor(env).getContent(post.author, post.permlink);
        if (!chainPost) throw new Error(`post ${post.author}/${post.permlink} not found on chain`);
        const parsed = parsePost(chainPost);
        if (!parsed.image?.supported) throw new Error("unsupported image container: no supported image payload");
        const bytes = base64Decode(parsed.image.base64);
        // the image on chain moved on since stats ran: the ingest that saw it enqueued every stage
        if ((await sha256Hex(bytes)) !== known.content_hash) throw new Error("superseded: the image changed since the stats stage");
        img = await decodeImage(bytes, sniff(bytes));
        return img;
      };
      const r = await runPaph(env, m, known, known.content_hash, fetchImage, report, o);
      pendingRetry = pendingRetry ?? r;
    }
  }

  if (stages.has("concepts")) report.concepts = await conceptsStage(env, post, art, !!m.force);
  if (stages.has("text")) {
    if (blocked.has("text")) {
      await setJob(env.DB, m.postId, "text", "skipped", "ai-training=false blocks text vectors");
    } else {
      try {
        report.text = await textStage(env, post, art, !!m.force);
      } catch (e) {
        if (e instanceof Retry) pendingRetry = pendingRetry ?? e;
        else throw e;
        report.text = "retry";
      }
    }
  }
  if (pendingRetry) throw pendingRetry;
  return report;
}

// ---- image stages ---------------------------------------------------------------------------------

async function imagePipeline(
  env: Env,
  m: EnrichMessage,
  post: PostRow,
  art: ArtworkRow | null,
  bytes: Uint8Array,
  hash: string,
  container: ContainerInfo,
  stages: Set<Stage>,
  blocked: Set<Stage>,
  report: EnrichReport,
  o: RunOptions,
): Promise<{ retry: Retry | null }> {
  const t = now();
  let img: RgbaImage | null = null;
  const decoded = async () => (img ??= await decodeImage(bytes, container));

  // ---- stats (+ features, hashes, history) -----------------------------------------------------
  if (stages.has("stats")) {
    if (!m.force && art && art.stats_hash === hash && art.features_hash === hash) {
      report.stats = "unchanged";
    } else {
      try {
        const im = await decoded();
        const st = computeStats(im, { lossy: container.lossy });
        const ft = computeFeatures(im, { backgroundHex: st.backgroundHex, transparentShare: st.transparentShare, clusters: st.clusters });
        const ph = phash(im);
        const dh = dhash(im);
        const [hi, lo] = halves(ph);
        const origKey = bool(env.STORE_IN_R2, true) ? `orig/${hash}.${container.format}` : null;
        // Side effects first, the "done" hashes (stats_hash, features_hash) last: a failure in
        // between leaves them NULL, so the retry (or the sweeper) redoes the stage instead of
        // reporting it unchanged with the R2 original or the history missing.
        if (origKey) {
          const head = await env.ART.head(origKey);
          if (!head) await env.ART.put(origKey, bytes, { httpMetadata: { contentType: container.mime, cacheControl: "public, max-age=31536000, immutable" } });
        }
        await env.DB.batch([
          env.DB
            .prepare(
              `INSERT INTO artworks (post_id, content_hash, mime, bytes, lossy, width, height, pixels, size_class, color_count,
                 has_transparency, transparent_share, primary_color, background_hex, palette_json, buckets_json, phash, stats_hash, r2_orig_key, updated,
                 dhash, phash_hi, phash_lo, brightness, contrast, saturation, colorfulness, monochrome, edge_density, symmetry_x, symmetry_y,
                 foreground_share, center_x, center_y, aspect, orientation, palette_entropy, background_name, background_share, lab_l, lab_a, lab_b,
                 features_json, features_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(post_id) DO UPDATE SET content_hash = excluded.content_hash, mime = excluded.mime, bytes = excluded.bytes,
                 lossy = excluded.lossy, width = excluded.width, height = excluded.height, pixels = excluded.pixels, size_class = excluded.size_class,
                 color_count = excluded.color_count, has_transparency = excluded.has_transparency, transparent_share = excluded.transparent_share,
                 primary_color = excluded.primary_color, background_hex = excluded.background_hex, palette_json = excluded.palette_json,
                 buckets_json = excluded.buckets_json, phash = excluded.phash, stats_hash = excluded.stats_hash,
                 r2_orig_key = COALESCE(excluded.r2_orig_key, artworks.r2_orig_key), updated = excluded.updated,
                 dhash = excluded.dhash, phash_hi = excluded.phash_hi, phash_lo = excluded.phash_lo, brightness = excluded.brightness,
                 contrast = excluded.contrast, saturation = excluded.saturation, colorfulness = excluded.colorfulness, monochrome = excluded.monochrome,
                 edge_density = excluded.edge_density, symmetry_x = excluded.symmetry_x, symmetry_y = excluded.symmetry_y,
                 foreground_share = excluded.foreground_share, center_x = excluded.center_x, center_y = excluded.center_y, aspect = excluded.aspect,
                 orientation = excluded.orientation, palette_entropy = excluded.palette_entropy, background_name = excluded.background_name,
                 background_share = excluded.background_share, lab_l = excluded.lab_l, lab_a = excluded.lab_a, lab_b = excluded.lab_b,
                 features_json = excluded.features_json, features_hash = excluded.features_hash`,
            )
            .bind(
              m.postId, hash, container.mime, bytes.length, container.lossy ? 1 : 0, st.width, st.height, st.pixels, st.sizeClass, st.colorCount,
              st.hasTransparency ? 1 : 0, st.transparentShare, st.primaryColor, st.backgroundHex, JSON.stringify(st.palette), JSON.stringify(st.buckets),
              ph, null, origKey, t,
              dh, hi, lo, ft.brightness, ft.contrast, ft.saturation, ft.colorfulness, ft.monochrome ? 1 : 0, ft.edgeDensity, ft.symmetryX, ft.symmetryY,
              ft.foregroundShare, ft.centerX, ft.centerY, ft.aspect, ft.orientation, ft.paletteEntropy, ft.backgroundName, ft.backgroundShare,
              ft.lab.L, ft.lab.a, ft.lab.b, JSON.stringify(ft), null,
            ),
          env.DB.prepare("DELETE FROM artwork_colors WHERE post_id = ?").bind(m.postId),
          ...st.buckets.map((b) => env.DB.prepare("INSERT INTO artwork_colors (post_id, bucket, weight) VALUES (?, ?, ?)").bind(m.postId, b.name, b.weight)),
          env.DB.prepare("DELETE FROM phash_bands WHERE post_id = ?").bind(m.postId),
          ...phashBands(ph).map((v, i) => env.DB.prepare("INSERT INTO phash_bands (post_id, band, val) VALUES (?, ?, ?)").bind(m.postId, i, v)),
        ]);
        // History: a snapshot row (exact rows come from the tail and the history backfill), then
        // image_since / first_seen for this artwork.
        await writeVersions(env.DB, [snapshotVersion(post, hash, container.mime, ph, st.buckets)]);
        await refreshImageHistory(env.DB, m.postId);
        await env.DB.prepare("UPDATE artworks SET stats_hash = ?, features_hash = ? WHERE post_id = ?").bind(hash, hash, m.postId).run();
        await setJob(env.DB, m.postId, "stats", "done");
        report.stats = "done";
      } catch (e) {
        // An image that will never decode (too large, corrupt) is skipped for good, so the sweeper
        // stops re-driving it; anything else may be transient.
        await setJob(env.DB, m.postId, "stats", isPermanent(e) ? "skipped" : "failed", errMsg(e));
        report.stats = "failed";
        throw e; // nothing downstream makes sense without a decodable image
      }
    }
    art = await getArtwork(env.DB, m.postId);
  }

  // ---- paph (copy detection) --------------------------------------------------------------------------
  let paphRetry: Retry | null = null;
  if (stages.has("paph")) {
    // (not yet indexed for this image: the stats stage first; no job record, see runPaph)
    if (art && art.content_hash === hash) paphRetry = await runPaph(env, m, art, hash, decoded, report, o);
  }

  // ---- what else is needed ------------------------------------------------------------------------
  const label = embedLabel(env);
  const staleModel = !!env.EMBED_MODEL && !!art && art.embed_model !== env.EMBED_MODEL;
  const staleViews = !!art?.embed_hash && (art.embed_views ?? "xbrz") !== label;
  const embedOn = stages.has("embed") && embeddingEnabled(env) && !blocked.has("embed");
  const needEmbed = embedOn && (!!m.force || !art || art.embed_hash !== hash || staleModel || staleViews);
  const vlm = (env.VLM_BACKEND ?? "gemma").toLowerCase();
  const vlmError = isVlmBackend(vlm) ? vlmConfigError(env, vlm) : null;
  const describeOn = stages.has("describe") && isVlmBackend(vlm) && !vlmError && !blocked.has("describe");
  // v2 marked descriptions done even when they came back empty: redo those.
  const describedOk = !!art && art.describe_hash === hash && (!!art.ai_caption || art.ai_status === "ok");
  const needDescribe = describeOn && (!!m.force || !describedOk);

  if (stages.has("embed") && !embeddingEnabled(env)) await setJob(env.DB, m.postId, "embed", "skipped", "HF_EMBED_URL not configured");
  else if (stages.has("embed") && blocked.has("embed")) await setJob(env.DB, m.postId, "embed", "skipped", "ai-training=false blocks embeddings");
  else if (stages.has("embed") && !needEmbed) report.embed = "unchanged";
  if (stages.has("describe") && blocked.has("describe")) await setJob(env.DB, m.postId, "describe", "skipped", "ai-training=false blocks descriptions");
  else if (stages.has("describe") && !describeOn) await setJob(env.DB, m.postId, "describe", "skipped", vlmError ?? "VLM_BACKEND off");
  else if (stages.has("describe") && !needDescribe) report.describe = "unchanged";
  if (!needEmbed && !needDescribe) return { retry: paphRetry };

  // ---- images for the models ------------------------------------------------------------------------
  const im = await decoded();
  const scaler: Scaler = (env.SCALER ?? "xbrz") === "nearest" ? "nearest" : "xbrz";
  const previewFactor = factorFor(im.width, im.height, int(env.UPSCALE_TARGET, 800));
  let previewPng: Uint8Array | null = null;
  const preview = async () => (previewPng ??= await encodePng(upscale(im, previewFactor, scaler)));
  const upKey = bool(env.STORE_IN_R2, true) ? `up/${hash}.png` : null;
  if (upKey && !(await env.ART.head(upKey))) {
    await env.ART.put(upKey, await preview(), { httpMetadata: { contentType: "image/png", cacheControl: "public, max-age=31536000, immutable" } });
  }
  if (!art?.r2_up_key || art.up_factor !== previewFactor) {
    await env.DB
      .prepare("UPDATE artworks SET r2_up_key = COALESCE(?, r2_up_key), up_width = ?, up_height = ?, up_factor = ? WHERE post_id = ?")
      .bind(upKey, im.width * previewFactor, im.height * previewFactor, previewFactor, m.postId)
      .run();
  }

  let retry: Retry | null = paphRetry;
  let freshVector: number[] | null = null;

  // ---- embed -------------------------------------------------------------------------------------
  if (needEmbed) {
    try {
      const r = await embedImageViews(env, im, preview);
      freshVector = r.vector;
      await env.VEC.upsert([{ id: vectorId(m.postId), values: r.vector, metadata: vectorMetadata(post, art, hash) }]);
      await env.DB.prepare("UPDATE artworks SET embed_hash = ?, embed_model = ?, embed_views = ?, updated = ? WHERE post_id = ?").bind(hash, r.model, r.label, now(), m.postId).run();
      await setJob(env.DB, m.postId, "embed", "done");
      report.embed = "done";
      // copy detection: the fresh vector's neighbours, which the paph stage could not see yet
      if (paphEnabled(env)) {
        const a = await getArtwork(env.DB, m.postId);
        if (a?.paph_hash === hash) await paphVectorPass(env, m.postId, hash, r.vector, o).catch((e) => console.warn("paph vector pass failed", m.postId, errMsg(e)));
      }
    } catch (e) {
      await setJob(env.DB, m.postId, "embed", "failed", errMsg(e));
      if (e instanceof EmbedUnavailable && e.retryable) {
        report.embed = "retry";
        retry = new Retry(`embedding endpoint unavailable: ${e.message}`, 60);
      } else {
        report.embed = "failed";
        report.error = errMsg(e);
      }
    }
  }

  // ---- describe ------------------------------------------------------------------------------------
  if (needDescribe) {
    try {
      const vlmFactor = factorFor(im.width, im.height, int(env.VLM_TARGET, 512));
      const vlmPng = vlmFactor === previewFactor ? await preview() : await encodePng(upscale(im, vlmFactor, scaler));
      const { model, description: d, status, raw } = await describeImage(env, vlm, `data:image/png;base64,${base64Encode(vlmPng)}`, {
        title: post.title,
        tags: JSON.parse(post.tags_json || "[]"),
        description: post.description,
      });
      const aiTags = [...new Set([...d.tags, ...d.subjects, ...d.objects])].slice(0, 32);
      // Full-text first, describe_hash (the "done" mark) last: a failure in between is redone.
      await updateSearchDocAi(env.DB, m.postId, [d.caption, d.style, d.mood, d.text_in_image].filter(Boolean).join(" ").trim(), aiTags);
      await env.DB
        .prepare(
          `UPDATE artworks SET describe_hash = ?, vlm_model = ?, ai_caption = ?, ai_subjects_json = ?, ai_objects_json = ?, ai_tags_json = ?, ai_style = ?,
             ai_mood = ?, ai_text = ?, ai_nsfw = ?, ai_status = ?, ai_raw = ?, updated = ? WHERE post_id = ?`,
        )
        .bind(hash, model, d.caption, JSON.stringify(d.subjects), JSON.stringify(d.objects), JSON.stringify(aiTags), d.style, d.mood, d.text_in_image, d.nsfw, status, raw.slice(0, 2000), now(), m.postId)
        .run();
      // The NSFW estimate is part of the vector filter: keep the image vector's metadata in step
      // (with the vector computed above when there is one: Vectorize reads lag behind writes).
      // The text vector follows in the text stage, whose hash covers its metadata.
      const fresh = await getArtwork(env.DB, m.postId);
      if (fresh?.embed_hash === hash) await refreshVectorMetadata(env, post, fresh, { image: freshVector, text: false });
      await setJob(env.DB, m.postId, "describe", "done", status === "caption_only" ? "JSON reply unusable; caption task used" : undefined);
      report.describe = "done";
    } catch (e) {
      const msg = errMsg(e);
      await env.DB.prepare("UPDATE artworks SET ai_status = ?, ai_raw = ? WHERE post_id = ?").bind(e instanceof EmptyDescription ? "empty" : "error", e instanceof EmptyDescription ? e.raw.slice(0, 2000) : msg.slice(0, 2000), m.postId).run();
      await setJob(env.DB, m.postId, "describe", "failed", msg);
      // Transient infrastructure errors are retried through the queue. An empty description is
      // not: it is usually systematic (every Moondream reply in production once was), and each
      // queue retry is a paid call; the sweeper tries the stage again in 10 minutes, up to 40 times.
      if (/429|capacity|timeout|temporar|overload|503|502|504/i.test(msg) && !(e instanceof EmptyDescription)) {
        report.describe = "retry";
        retry = retry ?? new Retry(`vlm: ${msg}`, 120);
      } else {
        report.describe = "failed";
        report.error = msg;
      }
    }
  }
  return { retry };
}

/**
 * The paph stage (src/paph/copies.ts) with its job record. Its failures are its own: the other
 * stages go on, and a transient one (or a check some shard did not finish, or one the run had no
 * time left for) retries the message, whose other stages are then no-ops (hash-keyed).
 */
async function runPaph(env: Env, m: EnrichMessage, art: ArtworkRow, hash: string, image: () => Promise<RgbaImage>, report: EnrichReport, o: RunOptions): Promise<Retry | null> {
  if (!paphEnabled(env)) {
    await setJob(env.DB, m.postId, "paph", "skipped", "copy detection off (no PAPH binding, or PAPH_ENABLED=false)");
    return null;
  }
  try {
    const out = await paphStage(env, { postId: m.postId, hash, phash: art.phash, paphHash: art.paph_hash ?? null, paphEngine: art.paph_engine ?? null, force: !!m.force }, image, { endsAt: o.endsAt });
    if (out.status === "unchanged") {
      // a delivery that raced another one of the same check may have recorded it as failed
      await settleJob(env.DB, m.postId, "paph");
      report.paph = "unchanged";
      return null;
    }
    if (out.status === "deferred") {
      // no check was made (the job keeps its last record): the next delivery, in a fresh run, makes it
      report.paph = "retry";
      return new Retry("paph: deferred, too little of the consumer run left for a check", NOT_STARTED_DELAY_S);
    }
    if (out.status === "superseded" || out.status === "gone") {
      // the post was deleted meanwhile, or a newer image's run is under way: no job record, as for
      // anything that is not this image's permanent outcome ("skipped" keeps the sweeper away)
      return null;
    }
    console.log("paph", JSON.stringify({ postId: m.postId, stored: out.stored, reused: out.reused, twin: out.twin, keys: out.keys, fit: out.fit, shards: out.shards, compared: out.stats.compared, rejected: out.stats.rejected, unreadable: out.stats.unreadable, legacy_shards: out.stats.legacy, waited_ms: out.stats.waitedMs, incomplete: out.incomplete }));
    if (!out.complete) {
      await setJob(env.DB, m.postId, "paph", "failed", `incomplete: ${out.incomplete.join("; ")}`);
      report.paph = "retry";
      return new Retry(`paph: ${out.incomplete.join("; ")}`, 120);
    }
    await setJob(env.DB, m.postId, "paph", "done");
    report.paph = "done";
    return null;
  } catch (e) {
    if (/^superseded:/.test(errMsg(e))) return null; // the image moved on: its own run checks it
    const permanent = paphPermanent(e) || isPermanent(e);
    await setJob(env.DB, m.postId, "paph", permanent ? "skipped" : "failed", errMsg(e));
    report.paph = "failed";
    return permanent ? null : new Retry(`paph: ${errMsg(e)}`, 60);
  }
}

/**
 * The PNG SigLIP sees for one view. Shared by the embed stage and /search-by-image so a query
 * image goes through exactly the same pipeline as the indexed artworks. `preview` is the cached
 * R2 preview (xBRZ at UPSCALE_TARGET when SCALER is xbrz), reused instead of upscaling twice.
 */
export async function viewPng(view: ViewName, im: RgbaImage, env: Pick<Env, "SCALER" | "UPSCALE_TARGET">, preview?: () => Promise<Uint8Array>): Promise<Uint8Array> {
  if (view === "xbrz") {
    if (preview && (env.SCALER ?? "xbrz") !== "nearest") return preview();
    return encodePng(upscale(im, factorFor(im.width, im.height, int(env.UPSCALE_TARGET, 800)), "xbrz"));
  }
  if (view === "nearest") return encodePng(upscaleNearest(im, factorFor(im.width, im.height, 512)));
  return encodePng(im);
}

/** SigLIP image vector of a decoded image per EMBED_VIEWS (one view, or the weighted mix of several). */
export async function embedImageViews(env: Env, im: RgbaImage, preview?: () => Promise<Uint8Array>): Promise<{ vector: number[]; model: string; label: string }> {
  const views = parseViews(env.EMBED_VIEWS);
  const images: string[] = [];
  for (const v of views) images.push(base64Encode(await viewPng(v.view, im, env, preview)));
  const r = await embedAndRemember(env, "images", images);
  const vector = r.embeddings.length === 1 ? r.embeddings[0] : mixVectors(r.embeddings, views.map((v) => v.weight));
  return { vector, model: r.model, label: embedLabel(env) };
}

// ---- concepts --------------------------------------------------------------------------------------

function conceptInputs(post: PostRow, art: ArtworkRow | null) {
  const arr = (s: string | null | undefined): string[] => {
    try {
      const v = JSON.parse(s ?? "[]");
      return Array.isArray(v) ? v : [];
    } catch {
      return [];
    }
  };
  return {
    tags: arr(post.tags_json),
    title: post.title,
    description: post.description,
    ai: art?.ai_caption || art?.ai_tags_json
      ? { subjects: arr(art.ai_subjects_json), objects: arr(art.ai_objects_json), tags: arr(art.ai_tags_json), caption: art.ai_caption ?? "", style: art.ai_style }
      : null,
  };
}

async function conceptsStage(env: Env, post: PostRow, art: ArtworkRow | null, force: boolean): Promise<StageResult> {
  const input = conceptInputs(post, art);
  const fp = fnv1a(JSON.stringify(input));
  if (!force && art?.concepts_hash === fp) return "unchanged";
  const concepts = extractArtworkConcepts(input);
  await env.DB.batch([
    env.DB.prepare("DELETE FROM artwork_concepts WHERE post_id = ?").bind(post.id),
    ...concepts.map((c) => env.DB.prepare("INSERT INTO artwork_concepts (post_id, concept, confidence, source) VALUES (?, ?, ?, ?)").bind(post.id, c.concept, c.confidence, c.source)),
    env.DB.prepare("UPDATE artworks SET concepts_hash = ? WHERE post_id = ?").bind(fp, post.id),
  ]);
  await setJob(env.DB, post.id, "concepts", "done");
  return "done";
}

// ---- text vectors ----------------------------------------------------------------------------------

/** The text the SigLIP text tower sees (it keeps ~64 tokens, so the most descriptive part first). */
export function textForVector(post: Pick<PostRow, "type" | "title" | "description" | "tags_json" | "body">, art: Pick<ArtworkRow, "ai_caption"> | null): string {
  const tags = (() => {
    try {
      return (JSON.parse(post.tags_json || "[]") as string[]).filter((t) => !/^portal-\d+$/.test(t));
    } catch {
      return [];
    }
  })();
  const parts =
    post.type === "blog"
      ? [post.title, stripMarkdown(post.body).slice(0, 400)]
      : [post.title, art?.ai_caption ?? "", tags.join(", "), post.description.length > 2 ? post.description : ""];
  return parts.map((p) => p.trim()).filter(Boolean).join(". ").replace(/\s+/g, " ").slice(0, 480);
}

async function textStage(env: Env, post: PostRow, art: ArtworkRow | null, force: boolean): Promise<StageResult> {
  if (!bool(env.TEXT_VECTORS, true) || !env.VEC_TEXT) {
    await setJob(env.DB, post.id, "text", "skipped", "TEXT_VECTORS off or no VEC_TEXT binding");
    return "skipped";
  }
  if (!embeddingEnabled(env)) {
    await setJob(env.DB, post.id, "text", "skipped", "HF_EMBED_URL not configured");
    return "skipped";
  }
  const text = textForVector(post, art);
  if (!text) {
    await setJob(env.DB, post.id, "text", "skipped", "no text");
    return "skipped";
  }
  // The hash covers the vector's metadata too (flags, colours of a changed image, the AI NSFW
  // estimate), so the text vector is rewritten whenever its filters would otherwise go stale.
  const metadata = textVectorMetadata(post, art, art?.content_hash ?? "");
  const h = fnv1a(`${env.EMBED_MODEL ?? ""}|${text}|${JSON.stringify(metadata)}`);
  if (!force && post.text_hash === h) return "unchanged";
  try {
    const r = await embedAndRemember(env, "texts", [text]);
    await env.VEC_TEXT.upsert([{ id: vectorId(post.id), values: r.embeddings[0], metadata }]);
    await env.DB.prepare("UPDATE posts SET text_hash = ?, text_model = ? WHERE id = ?").bind(h, r.model, post.id).run();
    await setJob(env.DB, post.id, "text", "done");
    return "done";
  } catch (e) {
    await setJob(env.DB, post.id, "text", "failed", errMsg(e));
    if (e instanceof EmbedUnavailable && e.retryable) throw new Retry(`embedding endpoint unavailable: ${e.message}`, 60);
    return "failed";
  }
}

function isPermanent(e: unknown): boolean {
  return /unsupported image container|image too large|Decoding error|Encoding error|unrecognised container/i.test(errMsg(e));
}

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
