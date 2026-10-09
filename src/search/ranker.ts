// Second-stage ranker. The first stage (service.ts: retrieve) maximises recall by unioning several
// legs (full text, spelling, concepts, colours, image kNN, text kNN). Here every candidate gets a
// feature vector and a score:
//
//   relevance = Σ w_f · f            features in [0, 1], weights below (or learned, see /admin/ranker)
//   final     = relevance × quality × freshness × diversity
//   quality   ∈ [0.90, 1.10]          Bayesian vote rate, bounded so popularity cannot beat relevance
//   freshness ∈ [0.95, 1.05]
//
// v2 multiplied reciprocal-rank-fusion scores by up to ~1.5 for votes and recency while adjacent
// RRF ranks differ by ~1.6 %: popularity decided most orderings. Here relevance features carry
// absolute meaning (exact title/tag matches, concept presence, how far an image stands out from
// the corpus for this query) and quality only breaks near-ties.

import { aliasesOf } from "../concepts";
import { fold, singular, tokens } from "../lib/text";
import { COLOR_WORDS, TONE_WORDS, type Tone } from "./lexicon";
import type { QueryPlan } from "./planner";

export const FEATURES = ["title", "lexical", "coverage", "tag", "author", "concept", "sem", "sem_txt", "color", "tone", "orientation"] as const;
export type FeatureName = (typeof FEATURES)[number];
export type Features = Record<FeatureName, number>;

export interface RankerWeights {
  w: Record<FeatureName, number>;
  /** bounds of the multiplicative factors */
  quality: [number, number];
  freshness: [number, number];
  freshnessDays: number;
  /** semantic feature = sigmoid(slope · (z − z0)), z standardised against the background sample */
  semZ0: number;
  semSlope: number;
  txtZ0: number;
  txtSlope: number;
  /** without a background sample: SigLIP's own sigmoid(scale·cos + bias + shift) */
  semShift: number;
  /** cosine range mapped to 0..1 for text-vector similarity without a background sample */
  txtLo: number;
  txtHi: number;
  /** colour-led queries ("blue", "black and white"): lexical features × this, colour/tone × colorBoost */
  colorLedLexical: number;
  colorBoost: number;
  /** a spelling-corrected variant of the query counts this much of the original */
  correctedFactor: number;
  /** pHash distance under which a lower-ranked item counts as a near-duplicate of a higher one */
  dupDistance: number;
  dupFactor: number;
  version: string;
}

export const DEFAULT_WEIGHTS: RankerWeights = {
  w: { title: 0.4, lexical: 0.1, coverage: 0.08, tag: 0.1, author: 0.25, concept: 0.15, sem: 0.45, sem_txt: 0.1, color: 0.2, tone: 0.15, orientation: 0.05 },
  quality: [0.9, 1.1],
  freshness: [0.95, 1.05],
  freshnessDays: 180,
  semZ0: 2.5,
  semSlope: 1.5,
  txtZ0: 2.5,
  txtSlope: 1.5,
  semShift: 0,
  txtLo: 0.55,
  txtHi: 0.9,
  colorLedLexical: 0.3,
  colorBoost: 5,
  correctedFactor: 0.9,
  dupDistance: 4,
  dupFactor: 0.92,
  version: "default-1",
};

/** Merge a partial weights object (from settings) over the defaults; unknown keys are ignored. */
export function mergeWeights(over: unknown): RankerWeights {
  const o = (over && typeof over === "object" ? over : {}) as Partial<RankerWeights> & { w?: Partial<Record<FeatureName, number>> };
  const w = { ...DEFAULT_WEIGHTS.w };
  for (const k of FEATURES) if (typeof o.w?.[k] === "number" && Number.isFinite(o.w[k])) w[k] = o.w[k]!;
  const pair = (v: unknown, d: [number, number]): [number, number] =>
    Array.isArray(v) && v.length === 2 && v.every((x) => typeof x === "number") && v[0] <= v[1] ? [v[0], v[1]] : d;
  const n = (v: unknown, d: number) => (typeof v === "number" && Number.isFinite(v) ? v : d);
  const out: RankerWeights = { ...DEFAULT_WEIGHTS, w, quality: pair(o.quality, DEFAULT_WEIGHTS.quality), freshness: pair(o.freshness, DEFAULT_WEIGHTS.freshness) };
  for (const k of ["freshnessDays", "semZ0", "semSlope", "txtZ0", "txtSlope", "semShift", "txtLo", "txtHi", "colorLedLexical", "colorBoost", "correctedFactor", "dupDistance", "dupFactor"] as const) {
    (out as any)[k] = n((o as any)[k], DEFAULT_WEIGHTS[k] as number);
  }
  out.version = typeof o.version === "string" ? o.version : DEFAULT_WEIGHTS.version;
  return out;
}

/** What the ranker needs to know about a candidate (a hydrated row plus its retrieval legs). */
export interface RankInput {
  id: number;
  type: "artwork" | "blog";
  author: string;
  title: string;
  description: string;
  tags: string[];
  aiCaption: string | null;
  aiTags: string[];
  created: number;
  netVotes: number;
  phash: string | null;
  buckets: Array<{ name: string; weight: number }>;
  brightness: number | null;
  contrast: number | null;
  saturation: number | null;
  colorfulness: number | null;
  monochrome: boolean | null;
  colorCount: number | null;
  paletteEntropy: number | null;
  orientation: string | null;
  backgroundName: string | null;
  /** retrieval evidence */
  bm25: number | null; // best (most negative) bm25 over the full-text legs; null = not matched
  bm25Leg: "and" | "or" | "spell" | null;
  /** query ↔ image / text-vector cosine per query variant (ctx.variants); null = not retrieved */
  cosImage: Array<number | null>;
  cosText: Array<number | null>;
  concepts: Map<string, number>; // concept → confidence, for the plan's concepts
}

export interface Norm {
  median: number;
  scale: number;
}

/** One phrasing of the query: as typed, or spelling-corrected. Each has its own vector statistics. */
export interface QueryVariant {
  text: string;
  factor: number; // 1 for the query as typed, correctedFactor for corrections
  imageNorm: Norm | null; // background statistics of this variant's cosines (background.ts)
  textNorm: Norm | null;
  cosImageFloor: number | null; // imputation for candidates the leg did not return
  cosTextFloor: number | null;
}

export interface RankContext {
  plan: QueryPlan;
  weights: RankerWeights;
  calibration: { logit_scale: number; logit_bias: number } | null;
  now: number;
  variants: QueryVariant[];
  /** corpus mean votes/day for the Bayesian prior */
  voteRate: number;
}

const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** Words of a field, folded, singular-aware. */
function wordSet(...fields: Array<string | null | undefined>): Set<string> {
  const out = new Set<string>();
  for (const f of fields) for (const t of tokens(f ?? "")) (out.add(t), out.add(singular(t)));
  return out;
}

const norm = (s: string) => tokens(s, { keepHyphenated: false }).join(" ");

/** SigLIP text↔image probability, sigmoid(scale·cos + bias + shift) (fallback before a background sample exists). */
export function semProbability(cos: number, cal: RankContext["calibration"], shift: number): number {
  if (!cal) return clamp01((cos - 0.05) / 0.15);
  return sigmoid(cal.logit_scale * cos + cal.logit_bias + shift);
}

/** Tone satisfaction in [0, 1] (continuous versions of TONE_SQL). */
export function toneScore(t: Tone, x: Pick<RankInput, "brightness" | "contrast" | "saturation" | "colorfulness" | "monochrome" | "colorCount" | "paletteEntropy">): number {
  const b = x.brightness ?? 0.45;
  switch (t) {
    case "dark":
      return clamp01((0.42 - b) / 0.2);
    case "light":
      return clamp01((b - 0.42) / 0.2);
    case "greyscale":
      return clamp01((16 - (x.colorfulness ?? 60)) / 10);
    case "monochrome":
      return x.monochrome ? 1 : clamp01((25 - (x.colorfulness ?? 60)) / 25);
    case "colorful":
      return clamp01(((x.colorfulness ?? 0) - 55) / 50);
    case "pastel":
      return clamp01(((0.45 - (x.saturation ?? 0.5)) / 0.25) * clamp01((b - 0.4) / 0.2));
    case "high_contrast":
      return clamp01(((x.contrast ?? 0.25) - 0.2) / 0.12);
    case "minimal":
      return Math.max(clamp01((32 - (x.colorCount ?? 256)) / 24), clamp01((3 - (x.paletteEntropy ?? 4)) / 1.5));
  }
}

const QSTOP = /^(a|an|the|of|in|on|and|or|with|le|la|les|de|du|des|et|der|die|das|und|el|los|las|y|il|lo|e)$/;

export interface QueryTerms {
  all: string[];
  content: string[];
  folded: string;
  glued: string;
  factor: number;
}

export function queryTerms(text: string, factor = 1): QueryTerms {
  const all = tokens(text, { keepHyphenated: false });
  const content = all.filter((t) => !QSTOP.test(t));
  const folded = norm(text);
  return { all, content: content.length ? content : all, folded, glued: folded.replace(/\s+/g, ""), factor };
}

/** "blue", "dark blue", "black and white", "pastel pink": every word is a colour or tone word. */
export function isColorLed(plan: QueryPlan): boolean {
  if (!plan.hints.colors.length && !plan.hints.tones.length && !plan.hints.background.length) return false;
  let rest = ` ${fold(plan.text).replace(/[^\p{L}\p{N}&\s-]+/gu, " ")} `;
  for (const w of [...Object.keys(TONE_WORDS), ...Object.keys(COLOR_WORDS)].sort((a, b) => b.length - a.length)) rest = rest.split(` ${w} `).join(" ");
  return rest.split(/\s+/).filter((t) => t && !QSTOP.test(t) && !/^(background|backdrop|colou?r|colou?rs|image|images|art|artwork|artworks|pixel)$/.test(t)).length === 0;
}

function titleScore(c: RankInput, q: QueryTerms): number {
  const titleNorm = norm(c.title);
  const titleWords = wordSet(c.title);
  if (q.folded && titleNorm === q.folded) return 1;
  if (q.glued.length >= 4 && titleNorm.replace(/\s+/g, "") === q.glued) return 1;
  if (q.folded.length >= 4 && (titleNorm.startsWith(`${q.folded} `) || titleNorm.endsWith(` ${q.folded}`))) return 0.85;
  if (q.folded.length >= 4 && q.all.length >= 2 && titleNorm.includes(q.folded)) return 0.8;
  if (!q.content.length) return 0;
  const hit = q.content.filter((t) => titleWords.has(t) || titleWords.has(singular(t))).length;
  let s = (0.7 * hit) / q.content.length;
  // the whole title is inside the query ("ride into the sunset bike" vs "Ride Into the Sunset")
  const tw = tokens(c.title, { keepHyphenated: false });
  if (tw.length >= 2 && tw.every((t) => q.all.includes(t))) s = Math.max(s, 0.75);
  return s;
}

function tagScore(c: RankInput, q: QueryTerms): number {
  const glued = c.tags.map((t) => fold(t).replace(/[-_\s]+/g, ""));
  if (q.glued && glued.includes(q.glued)) return 1;
  if (!q.content.length) return 0;
  // whole tags and their parts ("video-game" counts for "video" and "game")
  const set = new Set(c.tags.flatMap((t) => [fold(t), ...fold(t).split(/[-_\s]+/)]).flatMap((t) => [t, singular(t)]));
  return q.content.filter((t) => set.has(t) || set.has(singular(t))).length / q.content.length;
}

function coverageScore(c: RankInput, q: QueryTerms): number {
  if (!q.content.length) return 0;
  const words = wordSet(c.title, c.description, c.tags.join(" "), c.aiCaption, c.aiTags.join(" "), c.author);
  const covered = q.content.filter((t) => words.has(t) || words.has(singular(t)) || (t.length >= 5 && [...words].some((w) => w.length > t.length && w.startsWith(t)))).length;
  return covered / q.content.length;
}

export interface Raw {
  zImage: number | null;
  zText: number | null;
}

/** Best (lowest) bm25 per full-text leg among the candidates: each leg is normalised on its own. */
export type Bm25Best = Partial<Record<"and" | "or" | "spell", number>>;

export function bm25BestByLeg(cands: Array<Pick<RankInput, "bm25" | "bm25Leg">>): Bm25Best {
  const best: Bm25Best = {};
  for (const c of cands) if (c.bm25 !== null && c.bm25Leg && (best[c.bm25Leg] === undefined || c.bm25 < best[c.bm25Leg]!)) best[c.bm25Leg] = c.bm25;
  return best;
}

export function computeFeatures(c: RankInput, ctx: RankContext, qs: QueryTerms[], bm25Best: Bm25Best): { f: Features; raw: Raw } {
  const { plan, weights: w } = ctx;
  // Lexical features: the best of the query as typed and its spelling-corrected variants.
  let title = 0;
  let tag = 0;
  let coverage = 0;
  for (const q of qs) {
    title = Math.max(title, q.factor * titleScore(c, q));
    tag = Math.max(tag, q.factor * tagScore(c, q));
    coverage = Math.max(coverage, q.factor * coverageScore(c, q));
  }
  let lexical = 0;
  const best = c.bm25Leg ? bm25Best[c.bm25Leg] : undefined;
  if (c.bm25 !== null && best !== undefined && best < 0) {
    lexical = clamp01(c.bm25 / best) * (c.bm25Leg === "and" ? 1 : c.bm25Leg === "spell" ? w.correctedFactor : 0.6) * (0.5 + 0.5 * coverage);
  }

  // Author: named explicitly (@x, "by x") or the whole query; a bare name among other words is a hint.
  const a = c.author.toLowerCase();
  let author = 0;
  if ((plan.filters.authors ?? []).includes(a) || (plan.hints.namedAuthors ?? []).includes(a) || qs[0].folded === a) author = 1;
  else if (plan.hints.authors.includes(a)) author = 0.3;

  let concept = 0;
  if (plan.concepts.length) {
    let s = 0;
    for (const k of plan.concepts) s += c.concepts.get(k) ?? 0;
    concept = s / plan.concepts.length;
  }

  // Semantic: how far this candidate stands out from the corpus for the query (z against the
  // background sample), best over the variants; candidates a leg did not return get its floor.
  let sem = 0;
  let semTxt = 0;
  let zImage: number | null = null;
  let zText: number | null = null;
  ctx.variants.forEach((v, i) => {
    const ci = c.cosImage[i] ?? (c.type === "artwork" ? v.cosImageFloor : null);
    if (ci !== null && ci !== undefined) {
      if (v.imageNorm) {
        const z = (ci - v.imageNorm.median) / v.imageNorm.scale;
        if (zImage === null || z > zImage) zImage = z;
        sem = Math.max(sem, v.factor * sigmoid(w.semSlope * (z - w.semZ0)));
      } else sem = Math.max(sem, v.factor * semProbability(ci, ctx.calibration, w.semShift));
    }
    const ct = c.cosText[i] ?? v.cosTextFloor;
    if (ct !== null && ct !== undefined) {
      if (v.textNorm) {
        const z = (ct - v.textNorm.median) / v.textNorm.scale;
        if (zText === null || z > zText) zText = z;
        semTxt = Math.max(semTxt, v.factor * sigmoid(w.txtSlope * (z - w.txtZ0)));
      } else semTxt = Math.max(semTxt, v.factor * clamp01((ct - w.txtLo) / Math.max(1e-6, w.txtHi - w.txtLo)));
    }
  });

  // colour words: share of the palette in those named buckets (sqrt: 25 % of the image already counts)
  let color = 0;
  if (plan.hints.colors.length && c.type === "artwork") {
    const want = new Set(plan.hints.colors);
    const share = c.buckets.reduce((s, b) => s + (want.has(b.name) ? b.weight : 0), 0);
    color = Math.sqrt(clamp01(share));
  }
  if (plan.hints.background.length && c.type === "artwork") color = Math.max(color, plan.hints.background.includes(c.backgroundName ?? "") ? 1 : 0);

  let tone = 0;
  if (plan.hints.tones.length && c.type === "artwork") tone = plan.hints.tones.reduce((s, t) => s + toneScore(t, c), 0) / plan.hints.tones.length;

  const orientation = plan.hints.orientation.length && c.orientation && plan.hints.orientation.includes(c.orientation as any) ? 1 : 0;

  // Title squared: an exact title (1.0) counts fully, half the words (0.35) barely.
  return { f: { title: title * title, lexical, coverage, tag, author, concept, sem, sem_txt: semTxt, color, tone, orientation }, raw: { zImage, zText } };
}

export function relevance(f: Features, w: RankerWeights, colorLed = false): number {
  let s = 0;
  for (const k of FEATURES) {
    let wk = w.w[k];
    if (colorLed) {
      if (k === "title" || k === "lexical" || k === "coverage" || k === "tag") wk *= w.colorLedLexical;
      else if (k === "color" || k === "tone") wk *= w.colorBoost;
    }
    s += wk * f[k];
  }
  return s;
}

/**
 * Bayesian vote rate: votes per day of exposure (a post collects votes during its 7-day payout
 * window), shrunk towards the corpus rate with a prior worth two days. Two votes in the first hour
 * do not look better than forty over a week. Returns 0..1 (0.5 = corpus average).
 */
export function qualityScore(netVotes: number, created: number, now: number, corpusRate: number): number {
  const exposure = Math.min(7, Math.max(0.25, (now - created) / 86400));
  const prior = 2;
  const mu = Math.max(0.05, corpusRate);
  const rate = (Math.max(0, netVotes) + mu * prior) / (exposure + prior);
  return rate / (rate + mu);
}

export function freshnessScore(created: number, now: number, days: number): number {
  return Math.exp(-Math.max(0, now - created) / 86400 / days);
}

export interface Ranked {
  id: number;
  final: number;
  relevance: number;
  features: Features;
  quality: number;
  freshness: number;
  raw?: Raw;
  duplicateOf?: number;
}

export function rankCandidates(cands: RankInput[], ctx: RankContext): Ranked[] {
  const qs = ctx.variants.map((v) => queryTerms(v.text, v.factor));
  if (!qs.length) qs.push(queryTerms(ctx.plan.text, 1));
  const colorLed = isColorLed(ctx.plan);
  const bm25Best = bm25BestByLeg(cands);
  const { weights: w } = ctx;
  const out: Ranked[] = cands.map((c) => {
    const { f: features, raw } = computeFeatures(c, ctx, qs, bm25Best);
    const rel = relevance(features, w, colorLed);
    const quality = qualityScore(c.netVotes, c.created, ctx.now, ctx.voteRate);
    const fresh = freshnessScore(c.created, ctx.now, w.freshnessDays);
    const qf = w.quality[0] + (w.quality[1] - w.quality[0]) * quality;
    const ff = w.freshness[0] + (w.freshness[1] - w.freshness[0]) * fresh;
    return { id: c.id, final: rel * qf * ff, relevance: rel, features, quality, freshness: fresh, raw };
  });
  out.sort((a, b) => b.final - a.final || a.id - b.id);
  // Diversity: a near-identical image (same pHash neighbourhood) below a better copy steps down.
  if (w.dupFactor < 1) {
    const byId = new Map(cands.map((c) => [c.id, c]));
    const seen: Array<{ id: number; phash: string }> = [];
    for (const r of out) {
      const p = byId.get(r.id)?.phash;
      if (!p) continue;
      const dup = seen.find((s) => hammingHex(s.phash, p) <= w.dupDistance);
      if (dup) {
        r.final *= w.dupFactor;
        r.duplicateOf = dup.id;
      } else seen.push({ id: r.id, phash: p });
    }
    out.sort((a, b) => b.final - a.final || a.id - b.id);
  }
  return out;
}

function hammingHex(a: string, b: string): number {
  let d = 0;
  for (let i = 0; i < 16; i += 8) {
    let x = (parseInt(a.slice(i, i + 8), 16) ^ parseInt(b.slice(i, i + 8), 16)) >>> 0;
    while (x) {
      x &= x - 1;
      d++;
    }
  }
  return d;
}

/** Does a text mention a concept in any language (title/tags/description/AI fields)? */
export function mentionsConcept(text: string, concept: string): boolean {
  const words = ` ${tokens(text).join(" ")} `;
  return aliasesOf(concept).some((a) => words.includes(` ${a} `));
}
