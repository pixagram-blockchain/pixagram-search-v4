// Metadata mirrored into Vectorize, so that filters apply before kNN, and keeping it in step with
// D1 when a flag or a derived field changes.

import type { Env } from "../env";
import { getArtwork, getPostById, vectorId, type ArtworkRow, type PostRow } from "../db/posts";

/**
 * Indexed properties (10, the maximum): author, primary_color, size_class, created, color_count,
 * nsfw, listed, ai_training, orientation, transparent — VEC_TEXT swaps color_count for type.
 * `nsfw` is the same predicate as the SQL filter: the author's flag or an AI estimate >= 0.7
 * (v2 only mirrored the flag).
 */
export function vectorMetadata(
  post: Pick<PostRow, "author" | "permlink" | "created" | "nsfw" | "ai_training" | "listed">,
  art: Pick<ArtworkRow, "primary_color" | "size_class" | "color_count" | "ai_nsfw" | "orientation" | "has_transparency"> | null,
  hash: string,
): Record<string, string | number | boolean> {
  return {
    author: post.author,
    permlink: post.permlink,
    created: post.created,
    nsfw: post.nsfw === 1 || (art?.ai_nsfw ?? 0) >= 0.7,
    ai_training: post.ai_training === null ? true : post.ai_training === 1,
    listed: post.listed === 1,
    primary_color: art?.primary_color ?? "",
    size_class: art?.size_class ?? "",
    color_count: art?.color_count ?? 0,
    orientation: art?.orientation ?? "",
    transparent: art?.has_transparency === 1,
    hash,
  };
}

/** Metadata of the text vector: the same, with the post type. */
export function textVectorMetadata(post: Parameters<typeof vectorMetadata>[0] & Pick<PostRow, "type">, art: Parameters<typeof vectorMetadata>[1], hash: string) {
  return { ...vectorMetadata(post, art, hash), type: post.type };
}

/**
 * Re-write the metadata of a post's vectors in both indexes. `image` is the image vector when
 * the caller just computed it: Vectorize applies writes asynchronously, so reading back a vector
 * upserted moments ago can find nothing (and the refresh would silently do nothing).
 */
export async function refreshVectorMetadata(env: Env, post: PostRow, art: ArtworkRow | null, opts: { image?: number[] | null; text?: boolean } = {}): Promise<void> {
  const id = vectorId(post.id);
  const hash = art?.content_hash ?? "";
  try {
    if (art && art.embed_hash && art.embed_hash === art.content_hash) {
      const values = opts.image ?? (await env.VEC.getByIds([id]))[0]?.values;
      if (values) await env.VEC.upsert([{ id, values: Array.from(values as ArrayLike<number>), metadata: vectorMetadata(post, art, hash) }]);
    }
    if (opts.text !== false && env.VEC_TEXT && post.text_hash) {
      const got = await env.VEC_TEXT.getByIds([id]);
      if (got[0]?.values) await env.VEC_TEXT.upsert([{ id, values: Array.from(got[0].values as ArrayLike<number>), metadata: textVectorMetadata(post, art, hash) }]);
    }
  } catch (e) {
    console.warn("vector metadata refresh failed", post.id, e instanceof Error ? e.message : String(e));
  }
}

/** Load the post and its artwork, then refresh (after a flag change from the chain or the market). */
export async function refreshVectorMetadataById(env: Env, postId: number): Promise<void> {
  const post = await getPostById(env.DB, postId);
  if (!post || post.deleted) return;
  await refreshVectorMetadata(env, post, post.type === "artwork" ? await getArtwork(env.DB, postId) : null);
}
