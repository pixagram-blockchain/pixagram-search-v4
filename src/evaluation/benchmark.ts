// Model benchmarking (spec §31): every model receives exactly the same question, retrieval
// results, evidence and instructions. A question's context is frozen once (retrieval, operators and
// evidence verification run; no model), then given to each model; each reply is verified against
// the same cards, so models are compared on grounding, correctness, tokens, cost and latency only.
//
// The admin routes POST /admin/ask/context and POST /admin/ask/reason expose the two halves;
// scripts/benchmark.py drives them over a question set (src/evaluation/datasets/questions.jsonl).

import type { Env } from "../env";
import { guessLang, type Lang } from "../lib/text";
import { ask, type AskResponseV4 } from "../search/ask";
import type { EvidenceCard } from "../search/evidence";
import { agreesWithResult, verifyClaims, type ClaimVerification } from "../search/claims";
import { reasoningModel, UnusableReply } from "../llm/reasoning";
import type { ReasoningLevel } from "../llm/provider";
import { PROMPT_VERSION } from "../llm/prompts";

export interface FrozenContext {
  question: string;
  lang: Lang;
  /** the cards as the model sees them (modelView), in order */
  cards: EvidenceCard[];
  context: string[];
  /** the result cards the deterministic answer states (a model's answer must state their values) */
  shown: string[];
  deterministic: { status: string; answer: unknown; answer_type: string; text: string; complete: boolean };
  route: { class: string; complexity: number; band: string; mode: string };
  versions: Record<string, string | null>;
  prompt_version: string;
}

/** Retrieval, operators and verification for a question, stopped before the reasoning model. */
export async function freezeContext(env: Env, question: string, opts: { mode?: "fast" | "balanced" | "deep" | "expert"; nsfw?: "exclude" | "include"; type?: "artwork" | "blog" } = {}): Promise<FrozenContext> {
  const r = (await ask(env, { question, mode: opts.mode ?? "deep", nsfw: opts.nsfw ?? "exclude", type: opts.type, admin: true, contextOnly: true, noCache: true })) as AskResponseV4;
  const t = (r.trace ?? {}) as Record<string, any>;
  return {
    question,
    lang: (t.lang ?? r.plan.lang ?? guessLang(question)) as Lang,
    cards: (t.model_cards ?? []) as EvidenceCard[],
    context: (t.context ?? []) as string[],
    shown: (t.shown ?? []) as string[],
    deterministic: { status: r.status, answer: r.answer, answer_type: r.answer_type, text: r.result_text ?? r.answer_text, complete: !!t.has_deterministic },
    route: { class: r.class, complexity: r.complexity, band: r.band, mode: r.mode },
    versions: r.versions,
    prompt_version: PROMPT_VERSION,
  };
}

export interface ModelRun {
  model: string;
  reasoning: ReasoningLevel;
  status: string | null;
  answer: string | null;
  rationale: string | null;
  claims: ClaimVerification["claims"];
  grounding: { egs: number; citation_accuracy: number; answer: string; answer_problems: string[]; counts: Record<string, number> } | null;
  /** does the answer state the index's own answer (when the index has one): what /ask requires before showing it */
  agreement: { ok: boolean; problems: string[] } | null;
  confidence: number | null;
  usage: { inputTokens: number; outputTokens: number; reasoningTokens?: number } | null;
  cost_usd: number | null;
  model_ms: number;
  took_ms: number;
  finish_reason: string | null;
  prompt_version: string;
  error?: string;
  unusable?: boolean;
  notes: string[];
}

/** One model on a frozen context: never cached, never logged. */
export async function runOnContext(env: Env, ctx: Pick<FrozenContext, "question" | "lang" | "cards" | "context"> & { shown?: string[] }, model: string, opts: { reasoning?: ReasoningLevel; maxTokens?: number; strict?: boolean } = {}): Promise<ModelRun> {
  const reasoning = opts.reasoning ?? "medium";
  const t0 = Date.now();
  try {
    const reply = await reasoningModel(env, model).generate({
      question: ctx.question,
      evidence: ctx.cards as unknown as Array<{ evidence_id: string; type: string }>,
      reasoning,
      maxTokens: Math.min(8000, Math.max(64, opts.maxTokens ?? 2000)),
      lang: ctx.lang,
      context: ctx.context,
      task: "answer",
    });
    const accounts = new Set(ctx.cards.filter((c) => c.type === "artwork" || c.type === "post").map((c) => (c as { author: string }).author));
    const cv = verifyClaims(reply, ctx.cards, { strict: !!opts.strict, authors: accounts });
    const shown = ctx.cards.filter((c) => c.type === "result" && (ctx.shown?.length ? ctx.shown.includes(c.evidence_id) : true));
    const stated = ctx.shown?.length ? shown : shown.slice(-1);
    const agreement = stated.length ? agreesWithResult(reply.answer, stated as any, ctx.cards, { authors: accounts }) : null;
    return {
      model: reply.model,
      reasoning: reply.reasoning,
      status: reply.status,
      answer: reply.answer,
      rationale: reply.rationale ?? null,
      claims: cv.claims,
      grounding: { egs: cv.egs, citation_accuracy: cv.citation_accuracy, answer: cv.answer.status, answer_problems: cv.answer.problems, counts: cv.counts },
      agreement,
      confidence: reply.confidence ?? null,
      usage: reply.usage ?? null,
      cost_usd: reply.costUsd,
      model_ms: reply.latencyMs,
      took_ms: Date.now() - t0,
      finish_reason: reply.finishReason,
      prompt_version: reply.promptVersion,
      notes: reply.notes,
    };
  } catch (e) {
    return {
      model,
      reasoning,
      status: null,
      answer: null,
      rationale: null,
      claims: [],
      grounding: null,
      agreement: null,
      confidence: null,
      usage: null,
      cost_usd: null,
      model_ms: 0,
      took_ms: Date.now() - t0,
      finish_reason: null,
      prompt_version: PROMPT_VERSION,
      error: e instanceof Error ? e.message : String(e),
      unusable: e instanceof UnusableReply,
      notes: [],
    };
  }
}

/** Every model on every question's frozen context; the contexts are built once and shared. */
export async function benchmark(env: Env, questions: string[], models: string[], opts: { reasoning?: ReasoningLevel; maxTokens?: number; mode?: "balanced" | "deep" | "expert" } = {}): Promise<{ contexts: FrozenContext[]; runs: Record<string, ModelRun[]> }> {
  const contexts: FrozenContext[] = [];
  for (const q of questions) contexts.push(await freezeContext(env, q, { mode: opts.mode ?? "deep" }));
  const runs: Record<string, ModelRun[]> = {};
  for (const m of models) {
    runs[m] = [];
    for (const ctx of contexts) runs[m].push(await runOnContext(env, ctx, m, opts));
  }
  return { contexts, runs };
}
