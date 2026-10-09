// Confidence from the evidence (spec §25), not from the model's say-so:
//
//   confidence = Σ w_k · component_k / Σ w_k      over the components that apply
//   retrieval     how well the deciding evidence matches (v3's verification score of the rows the
//                 answer rests on, 1 for exact SQL answers), lowered for counts that are lower
//                 bounds and histories inferred from dates
//   reranking     the cross-encoder's sigmoid score of the deciding evidence, when it ran
//   agreement     1 − what the evidence disagrees on: conflicts, ties for the first place
//   verification  the share of the model's claims that are supported (EGS), and whether its answer
//                 is; 1 when the answer is deterministic (an operator's result is exact by construction)
//   model         the model's own confidence, when it gave one
//   × plan        min(1, plan confidence + 0.1): an unsure plan makes every answer less sure (v3)
//
// Weights: SEARCH_CONFIDENCE_WEIGHTS ("retrieval:0.25,reranking:0.2,agreement:0.2,verification:0.2,model:0.15").

import type { Env } from "../env";

export type ConfidenceComponent = "retrieval" | "reranking" | "agreement" | "verification" | "model";

export const DEFAULT_CONFIDENCE_WEIGHTS: Record<ConfidenceComponent, number> = { retrieval: 0.25, reranking: 0.2, agreement: 0.2, verification: 0.2, model: 0.15 };

export function confidenceWeights(env?: Env): Record<ConfidenceComponent, number> {
  const out = { ...DEFAULT_CONFIDENCE_WEIGHTS };
  for (const part of String(env?.SEARCH_CONFIDENCE_WEIGHTS ?? "").split(",")) {
    const [k, v] = part.split(":").map((s) => s.trim());
    const n = Number(v);
    if (k in out && Number.isFinite(n) && n >= 0 && n <= 1) out[k as ConfidenceComponent] = n;
  }
  return out;
}

export interface ConfidenceInput {
  retrieval: number | null;
  reranking: number | null;
  agreement: number | null;
  verification: number | null;
  model: number | null;
  planConfidence: number;
}

export interface Confidence {
  value: number;
  parts: Partial<Record<ConfidenceComponent | "plan", number>>;
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;
const clamp = (x: number) => Math.min(1, Math.max(0, x));

export function combineConfidence(input: ConfidenceInput, weights: Record<ConfidenceComponent, number> = DEFAULT_CONFIDENCE_WEIGHTS): Confidence {
  let num = 0;
  let den = 0;
  const parts: Confidence["parts"] = {};
  for (const k of ["retrieval", "reranking", "agreement", "verification", "model"] as ConfidenceComponent[]) {
    const v = input[k];
    if (v === null || !Number.isFinite(v)) continue;
    parts[k] = r3(clamp(v));
    num += weights[k] * clamp(v);
    den += weights[k];
  }
  const plan = Math.min(1, input.planConfidence + 0.1);
  parts.plan = r3(plan);
  return { value: r3(den ? (num / den) * plan : 0), parts };
}
