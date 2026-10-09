// Which model does which job (spec §19-20). Every choice is configuration:
//
//   role       variable (first set wins)                                           default
//   planner    SEARCH_PLANNER_MODEL, PLANNER_MODEL (v3)                             Llama 3.3 70B
//   reasoning  SEARCH_REASONING_MODEL_<BAND>, SEARCH_REASONING_MODEL                 gpt-oss-120b
//   help       SEARCH_HELP_MODEL, HELP_MODEL (v3)                                   Llama 3.3 70B
//   reranker   SEARCH_RERANKER_MODEL                                                bge-reranker-base
//
// <BAND> is the complexity band of the question (search/query-router.ts): SIMPLE, NORMAL,
// COMPLEX or DEEP, so cheap questions can go to a small model and hard ones to a large one
// ("trivial" questions are answered deterministically and use no model at all).

import type { Env } from "../env";

export type ModelRole = "planner" | "reasoning" | "help" | "reranker";
export type ComplexityBand = "trivial" | "simple" | "normal" | "complex" | "deep";

export const DEFAULT_PLANNER_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
export const DEFAULT_REASONING_MODEL = "@cf/openai/gpt-oss-120b";
export const DEFAULT_HELP_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
export const DEFAULT_RERANKER_MODEL = "@cf/baai/bge-reranker-base";

const set = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);

/** The model for a role (and, for reasoning, a complexity band). */
export function modelFor(env: Env, role: ModelRole, band?: ComplexityBand): string {
  const e = env as unknown as Record<string, unknown>;
  switch (role) {
    case "planner":
      return set(e.SEARCH_PLANNER_MODEL) ?? set(e.PLANNER_MODEL) ?? DEFAULT_PLANNER_MODEL;
    case "help":
      return set(e.SEARCH_HELP_MODEL) ?? set(e.HELP_MODEL) ?? DEFAULT_HELP_MODEL;
    case "reranker":
      return set(e.SEARCH_RERANKER_MODEL) ?? DEFAULT_RERANKER_MODEL;
    case "reasoning": {
      const byBand = band && band !== "trivial" ? set(e[`SEARCH_REASONING_MODEL_${band.toUpperCase()}`]) : null;
      return byBand ?? set(e.SEARCH_REASONING_MODEL) ?? DEFAULT_REASONING_MODEL;
    }
  }
}

/** Every reasoning model the configuration can route to (for /admin and the benchmark). */
export function configuredReasoningModels(env: Env): Record<string, string> {
  const out: Record<string, string> = {};
  for (const b of ["simple", "normal", "complex", "deep"] as ComplexityBand[]) out[b] = modelFor(env, "reasoning", b);
  return out;
}
