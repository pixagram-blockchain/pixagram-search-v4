// Shared types of the deterministic operators (spec §26). An operator takes a set of verified posts
// (or, for questions without a subject, the filters themselves) and computes an exact answer from
// the index: no model is involved, and the same index state always gives the same result.

import type { QueryPlan } from "../planner";
import type { SearchRequest } from "../params";

export type Row = Record<string, any>;

/** Why a candidate counts as showing the subject (v3's verification, per candidate). */
export interface Verification {
  score: number;
  lexical: number;
  semantic: number | null;
  text: number | null;
  signals: string[];
}

export interface Verified {
  row: Row;
  v: Verification;
}

export type OperatorName =
  | "find_first"
  | "find_latest"
  | "count"
  | "count_by_author"
  | "top"
  | "search"
  | "similar"
  | "duplicates"
  | "aggregate"
  | "group"
  | "history"
  | "compare"
  | "sequence"
  | "duration"
  | "resolve"
  | "identify";

/** What an answer value is (a client shows "alice" and "2026-09-10" differently). */
export type AnswerType = "author" | "authors" | "date" | "post" | "count" | "boolean" | "duration" | "value" | "list" | "none";

/**
 * The posts an operator works on. "subject": candidates retrieved and verified against the subject
 * (most questions); "metadata": no subject, the filters alone decide, exactly (SQL).
 */
export interface Scope {
  kind: "subject" | "metadata";
  plan: QueryPlan;
  req: SearchRequest;
  /** every candidate looked at, with its verification (subject scopes) */
  all: Verified[];
  /** those that passed */
  verified: Verified[];
  /** some period's vector search came back full: counts over this scope are lower bounds */
  truncated: boolean;
  /** where each candidate came from (spec §11), by post id */
  provenance: Map<number, Provenance>;
  /** candidates per retrieval leg */
  legs: Record<string, number>;
  notes: string[];
}

export interface Provenance {
  sources: string[];
  scores: Record<string, number>;
}

export interface OperatorResult {
  op: OperatorName;
  /** the machine answer: an account, an ISO date, a post path, a number, a boolean, or null */
  answer: string | number | boolean | null;
  answerType: AnswerType;
  /** the deterministic sentence, in the question's language */
  text: string;
  confidence: number;
  /** what the answer rests on, strongest first */
  evidence: Verified[];
  alternatives?: Verified[];
  /** posts behind the answer, for display */
  items?: Verified[];
  counts?: Array<{ author: string; n: number }>;
  /** candidates that passed verification (v3's "verified") */
  verified?: number;
  /** a count or a "first" that may miss matches (vector budget spent) */
  truncated?: boolean;
  /** an exact answer computed by SQL over the filters (no subject verification needed) */
  exact?: boolean;
  /** structured facts of the result (tie lists, history events, durations…), for evidence cards */
  details?: Record<string, unknown>;
  notes: string[];
}
