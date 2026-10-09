// How much work one copy check may do in one shard. Every step that grows with the corpus is
// bounded here, so a check costs the same whether a shard holds a thousand works or a hundred
// thousand — that is what keeps an interactive query under a second at any size:
//
//   key nomination   the query's local codes and descriptor bands, rarest first, until
//                    `keyPostings` posting rows are read; the top `perFamily` of each family by
//                    Σ 1/df (keys held by more than `dfCap` works are never read)
//   SI nomination    the PAPH-SI probes, rarest first, until `siPostings` rows are read; the
//                    `siReach` candidates the probes reached most strongly are scored exactly
//                    against their stored signatures, the best `siTop` at or above the threshold
//                    are kept
//   verification     XRank on at most `verify` candidates, best first, in batches (gated: under
//                    PAPH-X's X2 and X3 profiles the gate drops no copy comparator 42 certifies)
//   deadline         `deadlineMs` from the check's arrival in the shard, waiting for its turn
//                    included: the caller waits that long (plus a grace) and the shard stops
//                    there, answering what it has as partial. The counts above bound the work;
//                    the deadline only stops what a busy shard could not finish in time.
//
// Two presets: `query` for interactive checks (an upload, a live re-check; sub-second, and the
// answer is cached for a day) and `stage` for the background check of every new artwork (no one
// waits on it: it reads and compares far more, and its verdicts are what /copies/:id lists).

import type { Env } from "../env";

export interface Budget {
  perFamily: number;
  dfCap: number;
  keyPostings: number;
  siPostings: number;
  siReach: number;
  siTop: number;
  /** null: the SI profile's own threshold */
  siMinScore: number | null;
  verify: number;
  deadlineMs: number;
}

export const QUERY_BUDGET: Readonly<Budget> = Object.freeze({
  perFamily: 24,
  dfCap: 256,
  keyPostings: 40_000,
  siPostings: 30_000,
  siReach: 256,
  siTop: 64,
  siMinScore: null,
  verify: 96,
  deadlineMs: 600,
});

export const STAGE_BUDGET: Readonly<Budget> = Object.freeze({
  perFamily: 32,
  dfCap: 256,
  keyPostings: 400_000,
  siPostings: 400_000,
  siReach: 2_000,
  siTop: 256,
  siMinScore: null,
  verify: 512,
  // waiting for its turn included: a background check yields to every upload. With the stage's
  // grace (copies.ts) the stage waits at most 70 s for the shards; the queue consumer cuts that
  // to what is left of its run (consumer.ts RUN_MS), and bounds the calls to the work's own shard
  deadlineMs: 60_000,
});

const FIELDS: Record<string, keyof Budget> = {
  per_family: "perFamily",
  df_cap: "dfCap",
  key_postings: "keyPostings",
  si_postings: "siPostings",
  si_reach: "siReach",
  si_top: "siTop",
  si_min_score: "siMinScore",
  verify: "verify",
  deadline_ms: "deadlineMs",
};

const LIMITS: Record<keyof Budget, [number, number]> = {
  perFamily: [1, 256],
  dfCap: [2, 1_000_000],
  keyPostings: [0, 50_000_000],
  siPostings: [0, 50_000_000],
  siReach: [0, 100_000],
  siTop: [0, 10_000],
  siMinScore: [-1_000_000, 1_000_000],
  verify: [0, 10_000],
  deadlineMs: [50, 120_000],
};

/**
 * A preset with the overrides of PAPH_BUDGET_QUERY / PAPH_BUDGET_STAGE, e.g.
 * "verify:128,deadline_ms:700,si_top:96". Unknown names and unreadable values are ignored;
 * values are clamped to sane ranges.
 */
export function parseBudget(base: Readonly<Budget>, spec: string | undefined): Budget {
  const out: Budget = { ...base };
  for (const part of (spec ?? "").split(",")) {
    const [k, v] = part.split(":").map((s) => s.trim());
    const field = FIELDS[k ?? ""];
    if (!field) continue;
    if (field === "siMinScore" && (v === "" || v === "profile" || v === undefined)) {
      out.siMinScore = null;
      continue;
    }
    const n = Number(v);
    if (!Number.isFinite(n)) continue;
    const [lo, hi] = LIMITS[field];
    (out as unknown as Record<string, number>)[field] = Math.min(hi, Math.max(lo, Math.round(n)));
  }
  return out;
}

export function budgetFor(env: Pick<Env, "PAPH_BUDGET_QUERY" | "PAPH_BUDGET_STAGE">, kind: "query" | "stage"): Budget {
  return kind === "query" ? parseBudget(QUERY_BUDGET, env.PAPH_BUDGET_QUERY) : parseBudget(STAGE_BUDGET, env.PAPH_BUDGET_STAGE);
}

/** A short stable label of a budget, for cache keys (a different budget may find other copies). */
export function budgetLabel(b: Budget): string {
  return [b.perFamily, b.dfCap, b.keyPostings, b.siPostings, b.siReach, b.siTop, b.siMinScore ?? "p", b.verify].join(".");
}
