// Per-request context that changes slowly: ranker weights, the embedding calibration, the known
// authors (for the planner) and corpus statistics (for priors and time slicing). Cached in KV.

import type { Env } from "../env";
import { now } from "../env";
import { getCalibration, type Calibration } from "../enrich/embed";
import { getSetting } from "../db/posts";
import { mergeWeights, type RankerWeights } from "./ranker";

export interface CorpusStats {
  posts: number;
  artworks: number;
  minCreated: number;
  maxCreated: number;
  /** mean net votes per day of exposure (≤ 7 days), the Bayesian prior of the quality score */
  voteRate: number;
}

export interface SearchContext {
  weights: RankerWeights;
  calibration: Calibration | null;
  authors: Set<string>;
  stats: CorpusStats;
  now: number;
}

const TTL = 300;

async function cached<T>(env: Env, key: string, ttl: number, load: () => Promise<T>): Promise<T> {
  const hit = (await env.CACHE.get(key, "json").catch(() => null)) as T | null;
  if (hit !== null && hit !== undefined) return hit;
  const v = await load();
  await env.CACHE.put(key, JSON.stringify(v), { expirationTtl: ttl }).catch(() => {});
  return v;
}

export async function corpusStats(env: Env): Promise<CorpusStats> {
  return cached(env, "ctx:stats", TTL, async () => {
    const t = now();
    const r = await env.DB
      .prepare(
        `SELECT COUNT(*) AS posts, SUM(type = 'artwork') AS artworks, MIN(created) AS minc, MAX(created) AS maxc,
                AVG(MAX(net_votes, 0) * 1.0 / MIN(7.0, MAX(0.25, (? - created) / 86400.0))) AS rate
         FROM posts WHERE deleted = 0`,
      )
      .bind(t)
      .first<{ posts: number; artworks: number; minc: number | null; maxc: number | null; rate: number | null }>();
    return {
      posts: r?.posts ?? 0,
      artworks: r?.artworks ?? 0,
      minCreated: r?.minc ?? t - 86400,
      maxCreated: r?.maxc ?? t,
      voteRate: r?.rate ?? 0.5,
    };
  });
}

export async function knownAuthors(env: Env): Promise<Set<string>> {
  const list = await cached(env, "ctx:authors", 600, async () => {
    const r = await env.DB.prepare("SELECT DISTINCT author FROM posts WHERE deleted = 0").all<{ author: string }>();
    return (r.results ?? []).map((x) => x.author.toLowerCase());
  });
  return new Set(list);
}

export async function rankerWeights(env: Env): Promise<RankerWeights> {
  const raw = await cached(env, "ctx:weights", 60, async () => {
    const s = await getSetting(env.DB, "ranker:weights");
    return s ? JSON.parse(s) : {};
  });
  return mergeWeights(raw);
}

export async function loadContext(env: Env): Promise<SearchContext> {
  const [weights, calibration, authors, stats] = await Promise.all([rankerWeights(env), getCalibration(env), knownAuthors(env), corpusStats(env)]);
  return { weights, calibration, authors, stats, now: now() };
}
