// Optional LLM query planner for /ask (Workers AI, JSON mode). The model only *plans*: it returns
// intent, subject, filters and output, which are validated against fixed vocabularies and merged
// with the rule-based plan. It never sees artworks and never produces the answer, which always
// comes from retrieved evidence.
//
// PLANNER_BACKEND: "rules" (never call it), "auto" (default: only when the rules are unsure),
// "llm" (always). The model: SEARCH_PLANNER_MODEL, else v3's PLANNER_MODEL, else Llama 3.3 70B
// (llm/router.ts), in JSON mode with reasoning off, through the model layer (llm/provider.ts).

import type { Env } from "../env";
import { COLOR_NAMES } from "../enrich/color";
import { matchConcepts } from "../concepts";
import { complete } from "../llm/provider";
import { DEFAULT_PLANNER_MODEL, modelFor } from "../llm/router";
import { fold } from "../lib/text";
import type { Intent, Output, Tone } from "./lexicon";
import type { QueryPlan } from "./planner";

export { DEFAULT_PLANNER_MODEL };

const INTENTS: Intent[] = ["search", "find_first", "find_last", "count", "top", "compare", "similar", "duplicate"];
const OUTPUTS: Output[] = ["results", "author", "date", "count", "post"];
const TONES: Tone[] = ["dark", "light", "greyscale", "monochrome", "colorful", "pastel", "high_contrast", "minimal"];

const SCHEMA = {
  type: "object",
  properties: {
    intent: { type: "string", enum: INTENTS },
    output: { type: "string", enum: OUTPUTS },
    subject_en: { type: "string", description: "What the artworks must show, in English, a few words; empty if none" },
    object: { type: "string", enum: ["artwork", "blog", "any"] },
    authors: { type: "array", items: { type: "string" } },
    colors: { type: "array", items: { type: "string", enum: COLOR_NAMES } },
    tones: { type: "array", items: { type: "string", enum: TONES } },
    date_from: { type: "string", description: "ISO date or empty" },
    date_to: { type: "string", description: "ISO date (exclusive) or empty" },
    sort: { type: "string", enum: ["none", "votes", "payout", "newest", "oldest"] },
  },
  required: ["intent", "output", "subject_en", "object", "authors", "colors", "tones", "date_from", "date_to", "sort"],
};

export function plannerPrompt(question: string, today: string): string {
  return [
    "You turn questions about artworks on Pixagram (a pixel-art social network on its own blockchain) into a search plan.",
    "intent: search (find matching artworks), find_first (earliest one), find_last (most recent one), count (how many),",
    "top (most liked/best/most paid), compare (which author has the most), similar, duplicate.",
    "output: author (question asks who), date (asks when), count (asks how many), post (asks which/what), results (a list).",
    "subject_en: what must be depicted, translated to English, without words like image/artwork/first/posted (e.g. 'cat', 'samurai in japan').",
    "authors: account names only if the question names them. colors: only from the enum. tones: dark, light, greyscale (black and white), monochrome (one hue, sepia), colorful, pastel, high_contrast, minimal.",
    `Dates relative to today (${today}). Answer with JSON only.`,
    `Question: ${question}`,
  ].join("\n");
}

export interface LlmPlanFields {
  intent: Intent;
  output: Output;
  subject: string;
  object: "artwork" | "blog" | "any";
  authors: string[];
  colors: string[];
  tones: Tone[];
  from?: number;
  to?: number;
  sort?: "votes" | "payout" | "newest" | "oldest";
}

/** Validate a model reply against the vocabularies; null when unusable. */
export function validateLlmPlan(raw: unknown, knownAuthors: Set<string>): LlmPlanFields | null {
  let o: any = raw;
  if (typeof raw === "string") {
    try {
      const s = raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1);
      o = JSON.parse(s);
    } catch {
      return null;
    }
  }
  if (!o || typeof o !== "object") return null;
  if (!INTENTS.includes(o.intent) || !OUTPUTS.includes(o.output)) return null;
  const date = (v: unknown) => {
    if (typeof v !== "string" || !/^\d{4}-\d{2}-\d{2}/.test(v)) return undefined;
    const t = Date.parse(v.slice(0, 10) + "T00:00:00Z");
    return Number.isFinite(t) ? t / 1000 : undefined;
  };
  return {
    intent: o.intent,
    output: o.output,
    subject: typeof o.subject_en === "string" ? fold(o.subject_en).replace(/[^\p{L}\p{N}\s'-]/gu, " ").replace(/\s+/g, " ").trim().slice(0, 80) : "",
    object: ["artwork", "blog", "any"].includes(o.object) ? o.object : "artwork",
    authors: (Array.isArray(o.authors) ? o.authors : []).map((a: unknown) => String(a).toLowerCase().replace(/^@/, "")).filter((a: string) => knownAuthors.has(a)),
    colors: (Array.isArray(o.colors) ? o.colors : []).filter((c: unknown) => COLOR_NAMES.includes(String(c))),
    tones: (Array.isArray(o.tones) ? o.tones : []).filter((t: unknown) => TONES.includes(t as Tone)),
    from: date(o.date_from),
    to: date(o.date_to),
    sort: ["votes", "payout", "newest", "oldest"].includes(o.sort) ? o.sort : undefined,
  };
}

/** Rules first; the LLM fills in what the rules could not settle. */
export function mergePlans(rules: QueryPlan, llm: LlmPlanFields): QueryPlan {
  const subject = llm.subject || rules.residual;
  const conceptMatches = matchConcepts(subject);
  const filters = { ...rules.filters };
  if (!filters.authors?.length && llm.authors.length) filters.authors = llm.authors;
  if (!filters.colors?.length && !filters.background?.length && llm.colors.length) filters.colors = llm.colors;
  if (!filters.tones?.length && llm.tones.length) filters.tones = llm.tones;
  if (filters.from === undefined && llm.from !== undefined) filters.from = llm.from;
  if (filters.to === undefined && llm.to !== undefined) filters.to = llm.to;
  const intent = rules.intent !== "search" && rules.confidence >= 0.6 ? rules.intent : llm.intent;
  return {
    ...rules,
    intent,
    output: rules.confidence >= 0.6 && rules.output !== "results" ? rules.output : llm.output,
    residual: subject,
    concepts: [...new Set(conceptMatches.map((m) => m.concept))],
    conceptMatches,
    lexicalTerms: subject.split(" ").filter((w) => w.length > 1),
    filters,
    object: rules.object !== "any" ? rules.object : llm.object,
    sort: rules.sort ?? (llm.sort === "votes" || llm.sort === "payout" ? llm.sort : undefined),
    temporal: intent === "find_first" ? { operator: "first", field: "image" } : intent === "find_last" ? { operator: "last", field: "image" } : rules.temporal,
    source: "rules+llm",
    confidence: Math.max(rules.confidence, 0.7),
    notes: [...rules.notes, "plan refined by the LLM planner"],
  };
}

export async function llmPlan(env: Env, question: string, rules: QueryPlan, knownAuthors: Set<string>): Promise<QueryPlan> {
  const today = new Date().toISOString().slice(0, 10);
  const model = modelFor(env, "planner");
  const r = await complete(env, {
    model,
    messages: [
      { role: "system", content: "You output only JSON that matches the schema." },
      { role: "user", content: plannerPrompt(question, today) },
    ],
    json: { name: "search_plan", schema: SCHEMA },
    reasoning: "none",
    maxOutputTokens: 300,
    temperature: 0,
  });
  const reply = r.json ?? r.text;
  const fields = validateLlmPlan(reply, knownAuthors);
  if (!fields) throw new Error(`planner reply unusable: ${JSON.stringify(reply).slice(0, 160)}`);
  return mergePlans(rules, fields);
}
