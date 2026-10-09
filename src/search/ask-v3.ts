// /ask exactly as v3 answered it (mode=v3, spec §56): plan → candidates → verification per
// candidate → one deterministic operator → answer + evidence. v4's /ask (search/ask.ts) runs the
// same retrieval and operators, and adds routing, decomposition, evidence cards, verification and
// reasoning around them.

import type { Env } from "../env";
import { now } from "../env";
import { loadContext } from "./context";
import { llmPlan } from "./llm-planner";
import type { NsfwMode } from "./params";
import { planQuery, type QueryPlan } from "./planner";
import { compactPlan, rowToItem, type SearchItem } from "./service";
import { requestFromPlan, resolveScope } from "./retrieval";
import { runIntent, similarOp, type OpContext, type OperatorResult, type Row, type Verification } from "./operators";
import { subjectLabel } from "./answer-text";

export interface AskRequest {
  question: string;
  type?: "artwork" | "blog";
  limit?: number;
  nsfw?: NsfwMode;
  /** minimum verification score (default 0.5) */
  threshold?: number;
  /** "rules" | "llm" | "auto": overrides PLANNER_BACKEND for this request */
  planner?: string;
  /** the rule plan, when the caller already made it (the search box router) */
  plan?: QueryPlan;
}

export interface Evidence {
  post_id: number;
  author: string;
  permlink: string;
  path: string;
  title: string;
  created: number;
  image_since: number | null;
  first_seen: number | null;
  first_seen_post: string | null;
  /** how the first sighting matches: exact (same bytes), near (same author, same shapes and colours), self */
  first_seen_match: string | null;
  history_exact: boolean | null;
  net_votes: number;
  payout: number;
  verification: Verification;
}

export interface AskResponse {
  question: string;
  answer: string | number | null;
  answer_text: string;
  confidence: number;
  intent: QueryPlan["intent"];
  output: QueryPlan["output"];
  plan: Partial<QueryPlan>;
  evidence: Evidence[];
  alternatives?: Evidence[];
  counts?: Array<{ author: string; n: number }>;
  items?: SearchItem[];
  verified?: number;
  notes: string[];
  took_ms: number;
}

export function evidenceOf(row: Row, v: Verification): Evidence {
  return {
    post_id: row.id,
    author: row.author,
    permlink: row.permlink,
    path: `/@${row.author}/${row.permlink}`,
    title: row.title,
    created: row.created,
    image_since: row.image_since ?? null,
    first_seen: row.first_seen ?? null,
    first_seen_post: row.first_seen_author ? `/@${row.first_seen_author}/${row.first_seen_permlink}` : null,
    first_seen_match: row.first_seen_match ?? null,
    history_exact: row.history_exact === null || row.history_exact === undefined ? null : row.history_exact === 1,
    net_votes: row.net_votes ?? 0,
    payout: row.payout ?? 0,
    verification: v,
  };
}

/** The plan for a question: rules, and the LLM planner when they are unsure (or when asked). */
export async function planQuestion(env: Env, question: string, a: Pick<AskRequest, "plan" | "planner">, authors: Set<string>, notes: string[], opts: { v4?: boolean } = {}): Promise<QueryPlan> {
  let plan = a.plan ?? planQuery(question, { authors, mode: "ask", now: now(), ...(opts.v4 ? { v4: true } : {}) });
  const backend = (a.planner ?? env.PLANNER_BACKEND ?? "auto").toLowerCase();
  if (backend === "llm" || (backend === "auto" && plan.confidence < 0.6)) {
    try {
      plan = await llmPlan(env, question, plan, authors);
    } catch (e) {
      notes.push(`LLM planner unavailable, rules used: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  return plan;
}

/** An operator's result in v3's response shape. */
export function v3Response(base: Pick<AskResponse, "question" | "intent" | "output" | "plan" | "notes">, r: OperatorResult & { searchItems?: unknown[] }, t0: number): AskResponse {
  const out: AskResponse = { ...base, answer: r.answer as string | number | null, answer_text: r.text, confidence: r.confidence, evidence: r.evidence.map((x) => evidenceOf(x.row, x.v)), took_ms: Date.now() - t0 };
  if (r.alternatives) out.alternatives = r.alternatives.map((x) => evidenceOf(x.row, x.v));
  if (r.counts) out.counts = r.counts;
  if (r.searchItems) out.items = r.searchItems as SearchItem[];
  else if (r.items) out.items = r.items.map((x) => rowToItem(x.row));
  if (r.verified !== undefined) out.verified = r.verified;
  base.notes.push(...r.notes);
  return out;
}

export async function askV3(env: Env, a: AskRequest): Promise<AskResponse> {
  const t0 = Date.now();
  const notes: string[] = [];
  const ctx = await loadContext(env);
  const question = a.question.slice(0, 300);
  const plan = await planQuestion(env, question, a, ctx.authors, notes);
  const limit = Math.min(50, Math.max(1, a.limit ?? 10));
  const lang = plan.lang;
  const oc: OpContext = { env, ctx, lang, v3: true, limit, subject: subjectLabel(plan, lang, true) };
  const base = { question, intent: plan.intent, output: plan.output, plan: compactPlan(plan), notes };

  // similar / duplicate: delegate
  if (plan.intent === "similar" || plan.intent === "duplicate") {
    return v3Response(base, await similarOp(env, plan.similarTo?.id, plan.intent === "duplicate", oc, a.nsfw ?? "exclude"), t0);
  }
  const req = requestFromPlan(plan, a);
  const scope = await resolveScope(env, plan, req, ctx, { threshold: a.threshold ?? 0.5, notes });
  return v3Response(base, await runIntent(scope, oc), t0);
}
