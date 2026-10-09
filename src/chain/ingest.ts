// Shared ingestion path used by the live tail, the backfill workflow and the admin API.

import type { Env, Stage } from "../env";
import { ALL_STAGES, BLOG_STAGES, bool, list } from "../env";
import { ChainRpc, type CondenserPost } from "./rpc";
import { appAllowed, parsePost } from "./parse";
import { setJobs, upsertPost } from "../db/posts";
import { refreshVectorMetadataById } from "../enrich/vector-meta";

export function rpcFor(env: Env): ChainRpc {
  return new ChainRpc({ url: env.RPC_URL, fallbacks: list(env.RPC_FALLBACK_URLS) });
}

export interface IngestOutcome {
  postId: number | null;
  action: "inserted" | "updated" | "skipped-app" | "skipped-reply" | "missing" | "deleted";
  enqueued: boolean;
  stages?: Stage[];
}

/** Which enrichment stages an upsert calls for. */
export function stagesFor(type: "artwork" | "blog", r: { needsEnrich: boolean; textChanged: boolean; inserted: boolean }, deleted: boolean, env: Env): Stage[] {
  if (deleted) return [];
  const textOn = bool(env.TEXT_VECTORS, true);
  if (type === "artwork") {
    if (r.needsEnrich) return textOn ? ALL_STAGES : ALL_STAGES.filter((s) => s !== "text");
    // Title, tags or description changed but not the image: only the cheap derived stages.
    if (r.textChanged && !r.inserted) return textOn ? ["concepts", "text"] : ["concepts"];
    return [];
  }
  return textOn && (r.textChanged || r.inserted) ? BLOG_STAGES : [];
}

/** Ingest a post given its current state (from get_content or a bridge listing). */
export async function ingestPost(env: Env, post: CondenserPost, blockNum: number | null, reason: string): Promise<IngestOutcome> {
  if (post.parent_author) return { postId: null, action: "skipped-reply", enqueued: false };
  const parsed = parsePost(post);
  if (!appAllowed(parsed.app, list(env.APP_PREFIXES))) return { postId: null, action: "skipped-app", enqueued: false };

  const r = await upsertPost(env, parsed, blockNum);
  const stages = stagesFor(parsed.type, r, parsed.deleted, env);
  if (stages.length) {
    await env.ENRICH_QUEUE.send({ postId: r.id, author: parsed.author, permlink: parsed.permlink, stages, reason });
    await setJobs(env.DB, r.id, stages, "queued");
  }
  // nsfw / ai-training changed without a new image: the vectors' filter metadata follows now
  // (the enrichment stages would rewrite it anyway when they run).
  if (r.flagsChanged && !stages.includes("embed")) await refreshVectorMetadataById(env, r.id);
  return { postId: r.id, action: parsed.deleted ? "deleted" : r.inserted ? "inserted" : "updated", enqueued: stages.length > 0, stages };
}

/** Fetch the current state of a post from the chain and ingest it. */
export async function ingestPostRef(env: Env, author: string, permlink: string, blockNum: number | null, reason: string): Promise<IngestOutcome> {
  const rpc = rpcFor(env);
  const post = await rpc.getContent(author, permlink);
  if (!post) return { postId: null, action: "missing", enqueued: false };
  return ingestPost(env, post, blockNum, reason);
}
