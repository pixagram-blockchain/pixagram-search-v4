// The image vector of an artwork that has none yet, computed when a request needs it.
//
// /similar (and /ask "similar to N") rank an artwork's neighbours by its SigLIP vector. An artwork
// whose embed stage has not succeeded yet has none: a post seconds old, or one posted while the
// Space was down (298 of them from 29 October to 3 November 2026). /similar then returned nothing.
// Now the request embeds the artwork itself, from the same image, views, model and patch budget
// as the embed stage (views.ts), so its neighbours are the ones the stored vector will give.
//
// The request writes no index. The vector is cached for a day by content hash, and the artwork's
// embed stage is queued: the pipeline stores the cached vector (without asking the Space again)
// with its metadata and runs copy detection's vector pass, as for any artwork.
//
// What it costs is bounded: only artworks without a stored vector get here; the computation is
// charged to the caller's heavy rate limit (`allow`); an image the Space refused is not tried again
// for six hours; the image comes from R2 (the chain only with a short budget); and the Space gets
// 20 s. While the last check says the Space is not ready, nothing is tried at all.

import type { Env, Stage } from "../env";
import { aiTrainingBlocks, bool, list, now } from "../env";
import { getArtwork, getPostById, setJobs, type ArtworkRow, type PostRow } from "../db/posts";
import { base64Decode, sha256Hex } from "../lib/bytes";
import { ChainRpc } from "../chain/rpc";
import { parsePost } from "../chain/parse";
import { decodeImage, sniff } from "./decode";
import { cachedSpaceStatus, embedLabel, embeddingEnabled, endpointFault, noteEndpointFault } from "./embed";
import { cachedImageVector, embedImageViews, rememberImageVector } from "./views";

/** How long a request waits for the Space (indexing waits two minutes). */
export const ON_DEMAND_TIMEOUT_MS = 20_000;
/** How long an image that cannot be embedded (the Space refused it, it does not decode) is left alone. */
const REFUSED_TTL_S = 6 * 3600;

export type VectorNow =
  | { vector: number[]; source: "cache" | "computed"; queued: boolean }
  | { vector: null; reason: string };

const refusedKey = (env: Env, hash: string) => `ivec-no:${env.EMBED_MODEL ?? "m"}:${embedLabel(env)}:${hash}`;

/** The image cannot be embedded, whatever the Space's state: remembered so that it is not tried again soon. */
class Unembeddable extends Error {}

/**
 * The image vector of artwork `postId` computed now, for an artwork whose stored vector is missing.
 * Null, with the reason, when it cannot be: not an indexed artwork, embedding blocked or skipped for
 * good, the image refused or unreadable, the Space not ready, or `allow` (the caller's rate limit,
 * asked only when the Space is about to be called) says no.
 */
export async function imageVectorNow(env: Env, postId: number, o: { allow?: () => Promise<boolean> } = {}): Promise<VectorNow> {
  if (!embeddingEnabled(env)) return { vector: null, reason: "HF_EMBED_URL not configured" };
  const [post, art] = await Promise.all([getPostById(env.DB, postId), getArtwork(env.DB, postId)]);
  if (!post || post.deleted || post.type !== "artwork") return { vector: null, reason: "not an active artwork" };
  if (!art?.content_hash || art.stats_hash !== art.content_hash) return { vector: null, reason: "image not indexed yet" };
  if (post.ai_training === 0 && aiTrainingBlocks(env).has("embed")) return { vector: null, reason: "ai-training=false blocks embeddings" };
  const skipped = await env.DB.prepare("SELECT error FROM jobs WHERE post_id = ? AND stage = 'embed' AND status = 'skipped'").bind(postId).first<{ error: string | null }>();
  if (skipped) return { vector: null, reason: `embedding skipped: ${skipped.error ?? "permanently"}` };

  const hash = art.content_hash;
  const space = await cachedSpaceStatus(env);
  const hit = await cachedImageVector(env, hash);
  if (hit) return { vector: hit, source: "cache", queued: await queueEmbed(env, post, art, space?.ready !== false).catch(() => false) };
  const refused = await env.CACHE.get(refusedKey(env, hash)).catch(() => null);
  if (refused) return { vector: null, reason: refused };
  // the last check of the Space (no new one: a request does not wait for /health)
  if (space && !space.ready) return { vector: null, reason: `the Space is not ready: ${space.problem}` };
  if (o.allow && !(await o.allow())) return { vector: null, reason: "rate limited (computing a vector is a heavy request)" };
  try {
    const bytes = await originalBytes(env, post, art);
    if ((await sha256Hex(bytes)) !== hash) throw new Unembeddable("the image changed since it was indexed");
    const im = await decodeImage(bytes, sniff(bytes)).catch((e: unknown) => {
      throw new Unembeddable(`the image does not decode: ${e instanceof Error ? e.message : String(e)}`);
    });
    const r = await embedImageViews(env, im, undefined, { timeoutMs: ON_DEMAND_TIMEOUT_MS });
    await rememberImageVector(env, hash, r.vector);
    return { vector: r.vector, source: "computed", queued: await queueEmbed(env, post, art, true).catch(() => false) };
  } catch (e) {
    const reason = e instanceof Error ? e.message : String(e);
    // the Space down: the requests behind this one fall back at once (noteEndpointFault); the image
    // refused or unreadable: left alone for a while
    if (endpointFault(e)) noteEndpointFault(env, e);
    else if (e instanceof Unembeddable || (e as { blame?: string })?.blame === "input") {
      await env.CACHE.put(refusedKey(env, hash), reason.slice(0, 300), { expirationTtl: REFUSED_TTL_S }).catch(() => {});
    }
    return { vector: null, reason };
  }
}

/** The artwork's original bytes: R2 (the stats stage stores them), else the chain within 5 s, without retries. */
async function originalBytes(env: Env, post: PostRow, art: ArtworkRow): Promise<Uint8Array> {
  if (art.r2_orig_key) {
    const obj = await env.ART.get(art.r2_orig_key).catch(() => null);
    if (obj) return new Uint8Array(await new Response(obj.body).arrayBuffer());
  }
  const rpc = new ChainRpc({ url: env.RPC_URL, fallbacks: list(env.RPC_FALLBACK_URLS), timeoutMs: 5_000, retries: 0 });
  const chainPost = await rpc.getContent(post.author, post.permlink);
  const parsed = chainPost ? parsePost(chainPost) : null;
  if (!parsed?.image?.supported) throw new Unembeddable("no supported image in the post");
  return base64Decode(parsed.image.base64);
}

/**
 * Queue the embed stage so the pipeline stores the vector computed here (it reuses the cached one),
 * and the text stage when the post has no current text vector and the Space is not known to be
 * down (the text needs the Space; the embed stage, with the cached vector, does not). `force` when
 * the database says the vector is current although the index has none (the stage would otherwise
 * find nothing to do). Not when it was queued in the last half hour (the sweeper's rule): a burst
 * of requests for one artwork queues it once.
 */
async function queueEmbed(env: Env, post: PostRow, art: ArtworkRow, spaceUp: boolean): Promise<boolean> {
  const busy = await env.DB.prepare("SELECT 1 AS x FROM jobs WHERE post_id = ? AND stage = 'embed' AND status = 'queued' AND updated >= ?").bind(post.id, now() - 30 * 60).first();
  if (busy) return false;
  const stages: Stage[] = ["embed"];
  const textOn = bool(env.TEXT_VECTORS, true) && !!env.VEC_TEXT;
  if (spaceUp && textOn && (!post.text_hash || (!!env.EMBED_MODEL && post.text_model !== env.EMBED_MODEL))) stages.push("text");
  const current = art.embed_hash === art.content_hash && (art.embed_views ?? "xbrz") === embedLabel(env) && (!env.EMBED_MODEL || art.embed_model === env.EMBED_MODEL);
  await env.ENRICH_QUEUE.send({ postId: post.id, author: post.author, permlink: post.permlink, stages, ...(current ? { force: true } : {}), reason: "similar" });
  await setJobs(env.DB, post.id, stages, "queued");
  return true;
}
