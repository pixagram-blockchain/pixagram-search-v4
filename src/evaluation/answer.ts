// Answer quality (spec §33-35): each answer of the question set scored against its expected answer
// — exact correctness, evidence correctness, claim grounding (EGS), citation accuracy, abstention
// correctness — and the composite reliability score, per category, language and mode.

import { meanMetrics, rankMetrics, type Judgments, type RankMetrics } from "./retrieval";
import { summarizePerformance, type PerformanceSummary, type RunCost } from "./models";

/** One question of the set (src/evaluation/datasets/questions.jsonl, written by eval/v4/generate.py). */
export interface EvalItem {
  id: string;
  question: string;
  query_type: string;
  lang: string;
  answer_type: "author" | "date" | "count" | "boolean" | "post" | "duration" | "value" | "status" | "retrieval";
  expected_answer: unknown;
  acceptable_answers: unknown[];
  required_evidence: string[];
  expected_status?: string;
  rel?: Judgments;
  image?: { ref?: string; transform: string };
  forbidden?: string[];
  note?: string;
}

/** What scoring reads of an /ask response (v4's shape; v3's fields suffice for most). */
export interface AnswerLike {
  status?: string;
  answer: unknown;
  answer_text: string;
  confidence?: number;
  evidence?: Array<{ path?: string; author?: string; permlink?: string }>;
  cards?: Array<{ type: string; path?: string; evidence_id?: string }>;
  items?: Array<{ author: string; permlink: string }>;
  claims?: Array<{ status: string; cited_correctly?: boolean }>;
  grounding?: { egs: number; citation_accuracy: number };
  model?: string | null;
  mode?: string;
  usage?: { input_tokens: number; output_tokens: number; cost_usd: number | null; model_ms: number };
  took_ms: number;
}

export interface ItemScore {
  id: string;
  query_type: string;
  lang: string;
  answer_type: string;
  correct: boolean;
  should_abstain: boolean;
  abstained: boolean;
  abstention_correct: boolean;
  /** the required evidence is among the answer's evidence (null: none required) */
  evidence_ok: boolean | null;
  /** the model's claims: supported share (null: no model answered) */
  egs: number | null;
  citation: number | null;
  forbidden_hit: boolean;
  retrieval?: RankMetrics;
  confidence: number | null;
  mode: string | null;
  model: string | null;
  cost: RunCost;
  got: unknown;
  status: string | null;
}

const ABSTAIN = new Set(["clarify", "not_found", "no_match", "insufficient_evidence"]);

const norm = (v: unknown): string => (typeof v === "string" ? v.trim().replace(/^@/, "").toLowerCase() : JSON.stringify(v));
const asPath = (v: unknown): string | null => {
  if (typeof v !== "string") return null;
  const s = v.trim();
  if (/^\/@[^/]+\/.+/.test(s)) return s;
  const m = /^@?([a-z0-9][a-z0-9.-]+)\/(.+)$/.exec(s);
  return m ? `/@${m[1]}/${m[2]}` : null;
};

function matches(type: EvalItem["answer_type"], got: unknown, want: unknown): boolean {
  if (want === null || want === undefined) return got === null || got === undefined;
  switch (type) {
    case "author":
      return typeof got === "string" && norm(got) === norm(want);
    case "post":
      return asPath(got) !== null && asPath(got) === asPath(want);
    case "date":
      return typeof got === "string" && got.slice(0, 10) === String(want);
    case "count":
    case "duration":
      return typeof got === "number" && typeof want === "number" && Math.abs(got - want) < 1e-9;
    case "boolean":
      return typeof got === "boolean" ? got === want : typeof want === "string" && typeof got === "string" && norm(got) === norm(want);
    case "value":
      return norm(got) === norm(want);
    default:
      return false;
  }
}

function paths(r: AnswerLike): Set<string> {
  const out = new Set<string>();
  for (const e of r.evidence ?? []) {
    const p = e.path ?? (e.author && e.permlink ? `/@${e.author}/${e.permlink}` : null);
    if (p) out.add(p);
  }
  for (const c of r.cards ?? []) if (c.path) out.add(c.path);
  for (const i of r.items ?? []) out.add(`/@${i.author}/${i.permlink}`);
  return out;
}

export function scoreAnswer(item: EvalItem, r: AnswerLike): ItemScore {
  const status = r.status ?? null;
  const shouldAbstain = !!item.expected_status && ABSTAIN.has(item.expected_status);
  // "no match" with a value ("0 artworks", "no") answers; without one it abstains
  const valued = r.answer !== null && r.answer !== undefined;
  const abstained = !valued || (status !== null && ABSTAIN.has(status) && status !== "no_match");
  let correct: boolean;
  let retrieval: RankMetrics | undefined;
  if (item.answer_type === "status") correct = status === item.expected_status;
  else if (item.answer_type === "retrieval") {
    const ranked = [...(r.items ?? []).map((i) => `/@${i.author}/${i.permlink}`)];
    const ev = (r.evidence ?? []).map((e) => e.path ?? `/@${e.author}/${e.permlink}`);
    const order = [...new Set([...ev, ...ranked])];
    retrieval = rankMetrics(order, item.rel ?? {});
    correct = (item.rel?.[order[0]] ?? 0) > 0;
  } else {
    const wants = [item.expected_answer, ...(item.acceptable_answers ?? [])];
    correct = wants.some((w) => matches(item.answer_type, r.answer, w) || (item.answer_type === "boolean" && matches("author", r.answer, w)));
    if (item.expected_status && status !== item.expected_status) correct = false;
  }
  const text = (r.answer_text ?? "").toLowerCase();
  const forbiddenHit = (item.forbidden ?? []).some((f) => text.includes(f.toLowerCase()));
  if (forbiddenHit) correct = false;
  const have = paths(r);
  const evidenceOk = item.required_evidence?.length && item.answer_type !== "status" ? item.required_evidence.every((p) => have.has(p)) : null;
  const claims = r.claims ?? [];
  return {
    id: item.id,
    query_type: item.query_type,
    lang: item.lang,
    answer_type: item.answer_type,
    correct,
    should_abstain: shouldAbstain,
    abstained,
    abstention_correct: shouldAbstain === abstained,
    evidence_ok: evidenceOk,
    egs: r.grounding ? r.grounding.egs : claims.length ? claims.filter((c) => c.status === "supported").length / claims.length : null,
    citation: r.grounding ? r.grounding.citation_accuracy : null,
    forbidden_hit: forbiddenHit,
    ...(retrieval ? { retrieval } : {}),
    confidence: typeof r.confidence === "number" ? r.confidence : null,
    mode: r.mode ?? null,
    model: r.model ?? null,
    cost: { took_ms: r.took_ms ?? 0, model_ms: r.usage?.model_ms ?? 0, input_tokens: r.usage?.input_tokens ?? 0, output_tokens: r.usage?.output_tokens ?? 0, cost_usd: r.usage?.cost_usd ?? 0 },
    got: r.answer ?? null,
    status,
  };
}

/** Reliability weights (spec §35): correctness, grounding, evidence accuracy, abstention. */
export const RELIABILITY_WEIGHTS = { correctness: 0.4, grounding: 0.3, evidence: 0.2, abstention: 0.1 };

export interface GroupSummary {
  n: number;
  accuracy: number;
  abstention_accuracy: number;
  /** answered when it should not have, or the reverse */
  false_answers: number;
  false_abstentions: number;
  evidence_accuracy: number | null;
  egs: number | null;
  citation_accuracy: number | null;
  reliability: number;
  /** answers given with confidence ≥ 0.8 that were wrong */
  confident_errors: number;
  retrieval?: RankMetrics;
  performance: PerformanceSummary;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : null);

export function summarizeScores(scores: ItemScore[], w = RELIABILITY_WEIGHTS): GroupSummary {
  const n = scores.length;
  const acc = n ? scores.filter((s) => s.correct).length / n : 0;
  const abst = n ? scores.filter((s) => s.abstention_correct).length / n : 0;
  const ev = mean(scores.filter((s) => s.evidence_ok !== null).map((s) => (s.evidence_ok ? 1 : 0)));
  const egs = mean(scores.filter((s) => s.egs !== null).map((s) => s.egs!));
  const cit = mean(scores.filter((s) => s.citation !== null).map((s) => s.citation!));
  // grounding: the model's claims where a model answered; an operator's answer is grounded by construction when correct
  const grounding = mean(scores.map((s) => (s.egs !== null ? s.egs : s.correct ? 1 : 0))) ?? 0;
  const reliability = w.correctness * acc + w.grounding * grounding + w.evidence * (ev ?? acc) + w.abstention * abst;
  const ret = scores.filter((s) => s.retrieval).map((s) => s.retrieval!);
  return {
    n,
    accuracy: acc,
    abstention_accuracy: abst,
    false_answers: scores.filter((s) => s.should_abstain && !s.abstained).length,
    false_abstentions: scores.filter((s) => !s.should_abstain && s.abstained).length,
    evidence_accuracy: ev,
    egs,
    citation_accuracy: cit,
    reliability,
    confident_errors: scores.filter((s) => !s.correct && !s.abstained && (s.confidence ?? 0) >= 0.8).length,
    ...(ret.length ? { retrieval: meanMetrics(ret) } : {}),
    performance: summarizePerformance(scores.map((s) => s.cost)),
  };
}

/** Summaries overall, per category, per language and per mode. */
export function summarizeBy(scores: ItemScore[]): { all: GroupSummary; by_type: Record<string, GroupSummary>; by_lang: Record<string, GroupSummary>; by_mode: Record<string, GroupSummary> } {
  const group = (f: (s: ItemScore) => string) => {
    const m = new Map<string, ItemScore[]>();
    for (const s of scores) m.set(f(s), [...(m.get(f(s)) ?? []), s]);
    return Object.fromEntries([...m.entries()].map(([k, v]) => [k, summarizeScores(v)]));
  };
  return { all: summarizeScores(scores), by_type: group((s) => s.query_type), by_lang: group((s) => s.lang), by_mode: group((s) => s.mode ?? "?") };
}

export function loadItems(jsonl: string): EvalItem[] {
  return jsonl
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as EvalItem);
}
