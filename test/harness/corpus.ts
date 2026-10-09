// A small index built directly in the database (posts, artworks, chain versions, image history),
// for the /ask v4 tests: no chain, no images, no embeddings — the deterministic operators and the
// evidence pipeline only.

import { createHash } from "node:crypto";
import { makeEnv, type TestEnv } from "./fakes";
import { upsertPost } from "../../src/db/posts";
import { refreshImageHistory, writeVersions, type VersionRow } from "../../src/chain/versions";

export const DAY = 86400;
export const at = (d: string) => Date.parse(`${d}T12:00:00Z`) / 1000;
export const hashOf = (s: string) => createHash("sha256").update(s).digest("hex");

export interface Art {
  author: string;
  permlink: string;
  title: string;
  tags?: string[];
  /** YYYY-MM-DD (noon UTC) */
  created: string;
  votes?: number;
  payout?: number;
  /** the image: a name, hashed (posts with the same name show the same bytes) */
  image?: string;
  deleted?: string;
  /** later versions: an edit (a new title, or a new image) */
  edits?: Array<{ on: string; title?: string; image?: string }>;
  nsfw?: boolean;
  blog?: boolean;
}

export async function artCorpus(arts: Art[], over: Record<string, string> = {}): Promise<TestEnv & { ids: Record<string, number> }> {
  const env = makeEnv({ PLANNER_BACKEND: "rules", HF_EMBED_URL: "", ...over } as any) as TestEnv & { ids: Record<string, number> };
  const ids: Record<string, number> = {};
  let block = 1000;
  const versions: VersionRow[] = [];
  for (const a of arts) {
    const created = at(a.created);
    const lastEdit = a.edits?.length ? at(a.edits[a.edits.length - 1].on) : created;
    const updated = a.deleted ? at(a.deleted) : lastEdit;
    const title = a.edits?.filter((e) => e.title).at(-1)?.title ?? a.title;
    const { id } = await upsertPost(
      env,
      {
        author: a.author, permlink: a.permlink, type: a.blog ? "blog" : "artwork", title, description: "", body: a.blog ? "A blog post." : "", bodyLength: 0, category: "pixagram",
        tags: a.tags ?? [], app: "pixagram/3", nsfw: !!a.nsfw, aiTraining: null, licenseJson: null, royaltyPct: null, created, updated, deleted: false,
        netVotes: a.votes ?? 0, payout: a.payout ?? 0, children: 0, image: null,
      },
      null,
    );
    ids[`${a.author}/${a.permlink}`] = id;
    const image = a.edits?.filter((e) => e.image).at(-1)?.image ?? a.image ?? `${a.author}/${a.permlink}`;
    if (!a.blog) await env.DB.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, updated) VALUES (?, ?, 'image/png', 10, ?)").bind(id, hashOf(image), created).run();
    const v = (on: number, kind: VersionRow["kind"], img: string | null, t: string): VersionRow => ({
      author: a.author, permlink: a.permlink, block_num: ++block, trx_id: `t${block}`, op_in_trx: 0, at: on, kind, body_kind: img ? "image" : kind === "delete" ? "deleted" : "text",
      content_hash: img ? hashOf(img) : null, phash: null, buckets_json: null, mime: img ? "image/png" : null, title: t, source: "history",
    });
    let img = a.blog ? null : (a.image ?? `${a.author}/${a.permlink}`);
    let t = a.title;
    versions.push(v(created, "create", img, t));
    for (const e of a.edits ?? []) {
      if (e.image) img = e.image;
      if (e.title) t = e.title;
      versions.push(v(at(e.on), "edit", img, t));
    }
    if (a.deleted) versions.push(v(at(a.deleted), "delete", null, t));
  }
  await writeVersions(env.DB, versions);
  for (const a of arts) {
    const id = ids[`${a.author}/${a.permlink}`];
    if (a.deleted) await env.DB.prepare("UPDATE posts SET deleted = 1 WHERE id = ?").bind(id).run();
  }
  for (const id of Object.values(ids)) await refreshImageHistory(env.DB, id);
  env.ids = ids;
  return env;
}

/** The test corpus most v4 tests use. */
export const GALLERY: Art[] = [
  { author: "alice", permlink: "black-cat", title: "Black cat", tags: ["cat", "black-and-white"], created: "2026-09-01", votes: 4, payout: 1.5 },
  { author: "bob", permlink: "cat-in-hat", title: "Cat in a hat", tags: ["cat"], created: "2026-09-03", votes: 2, payout: 0.5 },
  { author: "alice", permlink: "cat-nap", title: "Cat nap", tags: ["cat"], created: "2026-09-05", votes: 6, payout: 2.25 },
  { author: "carol", permlink: "sunset", title: "Sunset", tags: ["sunset"], created: "2026-09-07", votes: 3 },
  { author: "alice", permlink: "swan", title: "Swan", tags: ["swan", "lake"], created: "2026-09-10", votes: 5, edits: [{ on: "2026-09-11", title: "Swan" }] },
  { author: "bob", permlink: "lake", title: "Lake", tags: ["lake"], created: "2026-09-12", votes: 9, payout: 3 },
  { author: "bob", permlink: "red-dragon", title: "Red dragon", tags: ["dragon", "red"], created: "2026-09-15", votes: 1 },
  { author: "alice", permlink: "dragon-egg", title: "Dragon egg", tags: ["dragon"], created: "2026-10-02", votes: 7 },
  // carol posts alice's black cat again: the same image bytes
  { author: "carol", permlink: "found-cat", title: "Found this cat", tags: ["cat"], created: "2026-09-25", image: "alice/black-cat", votes: 0 },
  // a post that was deleted
  { author: "dave", permlink: "old-dragon", title: "Old dragon", tags: ["dragon"], created: "2026-09-08", deleted: "2026-09-09", votes: 0 },
];
