// Spelling suggestions from the corpus vocabulary (vocab + vocab_grams, kept up to date by
// writeSearchDoc). A query token that no document contains is matched against vocabulary terms
// sharing character trigrams, accepted within a small Damerau–Levenshtein distance, and the most
// frequent close term wins. Corrections feed an extra full-text leg; they never replace the query.

import { editDistance, fold, isStopword, trigrams } from "../lib/text";
import { matchConcepts } from "../concepts";
import { COLOR_WORDS, TONE_WORDS } from "./lexicon";

export interface Correction {
  from: string;
  to: string;
  distance: number;
  df: number;
}

export function maxEdits(len: number): number {
  return len <= 4 ? 1 : len <= 8 ? 2 : 3;
}

/**
 * Query tokens worth checking: letters only, 4+ chars, not CJK, and not a word we know means
 * something even though no document contains it yet — a concept alias in any language ("beach",
 * "chien"), a colour or tone word, a function word. Otherwise "beach" became "peach" and "chien"
 * became "chain" on a corpus where nobody had written them.
 */
export function spellTokens(q: string): string[] {
  return [
    ...new Set(
      fold(q)
        .split(/[^\p{L}]+/u)
        .filter((t) => t.length >= 4 && /^[a-z]+$/.test(t) && !isStopword(t) && !COLOR_WORDS[t] && !TONE_WORDS[t] && !matchConcepts(t).length),
    ),
  ].slice(0, 8);
}

export async function suggest(db: D1Database, q: string): Promise<Correction[]> {
  const toks = spellTokens(q);
  if (!toks.length) return [];
  const known = await db
    .prepare(`SELECT term, df FROM vocab WHERE term IN (${toks.map(() => "?").join(",")}) AND df > 0`)
    .bind(...toks)
    .all<{ term: string; df: number }>();
  const present = new Set((known.results ?? []).map((r) => r.term));
  const out: Correction[] = [];
  for (const t of toks) {
    if (present.has(t)) continue;
    const grams = trigrams(t);
    const rows = await db
      .prepare(
        `SELECT g.term AS term, COUNT(*) AS shared, v.df AS df FROM vocab_grams g JOIN vocab v ON v.term = g.term
         WHERE g.gram IN (${grams.map(() => "?").join(",")}) AND v.df > 0 AND length(g.term) BETWEEN ? AND ?
         GROUP BY g.term ORDER BY shared DESC, v.df DESC LIMIT 40`,
      )
      .bind(...grams, t.length - maxEdits(t.length), t.length + maxEdits(t.length))
      .all<{ term: string; shared: number; df: number }>();
    const best = pickCorrection(t, rows.results ?? []);
    if (best) out.push(best);
  }
  return out;
}

/** Closest candidate within maxEdits(len); ties go to the more frequent term. */
export function pickCorrection(token: string, candidates: Array<{ term: string; df: number }>): Correction | null {
  let best: Correction | null = null;
  const limit = maxEdits(token.length);
  for (const c of candidates) {
    if (c.term === token) return null;
    const d = editDistance(token, c.term, limit);
    if (d > limit) continue;
    if (!best || d < best.distance || (d === best.distance && c.df > best.df)) best = { from: token, to: c.term, distance: d, df: c.df };
  }
  return best;
}
