// /ask v4: a retrieval, verification and reasoning engine over the Pixagram index (spec §1, §60).
//
//   question ─► plan (rules; planner model when unsure)            query-planner.ts, planner.ts
//            ─► program: one step, or several (multi-hop, comparisons, sequences, durations, history)
//            ─► route: class, complexity, band ─► mode (fast / balanced / deep / expert)   query-router.ts
//            ─► execute every step: hybrid retrieval → verification → deterministic operator   executor.ts
//            ─► evidence cards + graph ─► evidence verification (validity, conflicts)    evidence.ts, verifier.ts
//            ─► deterministic answer (always, when the operators answer)
//            ─► reasoning model, when the mode allows and it is worth its cost           llm/reasoning.ts
//            ─► claim verification: unsupported claims removed, contradictions rejected   claims.ts
//            ─► confidence from the evidence, versions, timings, trace                  confidence.ts
//            ─► v4.8: the long-form answer — the deterministic digest (digest.ts), the model's
//               body, reasoning trail, caveats and follow-ups verified sentence by sentence,
//               assembled as Markdown and plain text (compose.ts)
//
// The index decides what happened; the model only explains it (spec §61). mode=v3 is v3's /ask,
// unchanged (ask-v3.ts). Every v3 response field keeps its meaning; v4's are added beside them,
// and v4.8's beside those: answer_text still carries the direct answer (unless text=full is
// asked), answer_full and answer_markdown the whole.
//
// Rich answers (SEARCH_ANSWER_STYLE=rich, the default) call the model for every answered question
// in balanced and above, and auto never runs below SEARCH_RICH_MIN_MODE. With defer=1 the answer
// comes back at once with the digest, and the model's part is fetched afterwards from
// GET /ask/elaboration/:query_id (the frozen context waits in KV): the search box shows the
// index's answer in milliseconds and fills in the rest when the model is done.

import type { Env } from "../env";
import { bool, int, list, num, now } from "../env";
import type { Lang } from "../lib/text";
import { fold, guessLang, randomId } from "../lib/text";
import { isReasoningLevel, LlmError, type ReasoningLevel } from "../llm/provider";
import { generateCached, reasoningModel, UnusableReply, type ReasoningResponse } from "../llm/reasoning";
import { isModelId } from "../llm/model";
import { modelFor } from "../llm/router";
import { PROMPT_VERSION } from "../llm/prompts";
import { loadContext, type SearchContext } from "./context";
import { visibleUnder, type NsfwMode } from "./params";
import type { QueryPlan } from "./planner";
import { compactPlan, rowToItem, type SearchItem } from "./service";
import { askV3, evidenceOf, planQuestion, type AskRequest, type AskResponse, type Evidence } from "./ask-v3";
import { decompose, decomposeRules, looksMultiStep, subqueries, withoutInstructions, withoutQuoted, type QueryProgram, type Subquery } from "./query-planner";
import { chooseMode, COMPOSE_TOKENS, PROFILES, richMinMode, routeQuestion, type ExecutionProfile, type Mode, type ModeRequest, type QueryClass, type QueryRoute } from "./query-router";
import { depthFor, metadataRows } from "./retrieval";
import { composeDeterministic, executeProgram, type StepOutcome } from "./executor";
import { applyVerification, conflictSentence, verifyEvidence } from "./verifier";
import { artworkCard, buildGraph, modelView, resultCard, rowsByPath, type ArtworkCard, type ConflictCard, type EvidenceCard, type EvidenceGraph } from "./evidence";
import { agreesWithResult, keptClaims, verifyClaims, type ClaimVerification, type VerifiedClaim } from "./claims";
import { combineConfidence, confidenceWeights } from "./confidence";
import { rerankBlend, rerankRows, rerankerVersion } from "./reranker";
import { answerLang, say, type AnswerLang } from "./answer-text";
import type { HistoryFacts, Row, Verified } from "./operators";
import { imageOutcome, imageTask, queryImageCard, type ImageFindings, type QueryImage } from "./image-question";
import type { ImageIdentity, QueryImageCard, ResultCard } from "./evidence";
import { buildDigest, type Digest, type DigestStats } from "./digest";
import { answerSuggestionsKey, type AnswerSuggestions } from "./suggest";
import {
  assembleMarkdown,
  countWords,
  followUpLimit,
  isAnswerStyle,
  suggestions,
  toPlainText,
  verifyList,
  verifyMarkdown,
  wordTarget,
  type AnswerSections,
  type AnswerStyle,
  type LengthRequest,
  type SuggestionItem,
  type TextMode,
  type ThinkingStep,
} from "./compose";

export { askV3, evidenceOf, type AskRequest, type AskResponse, type Evidence } from "./ask-v3";
export { lexicalEvidence, verify, zThreshold, isTitleSubject, timeKey, SUBJECT_TEMPLATES } from "./retrieval";

export const RETRIEVAL_VERSION = "4.0.0";
/** The long-form answer layer (compose.ts, digest.ts): part of every answer's versions. */
export const ANSWER_VERSION = "4.8.0";

export interface AskRequestV4 extends AskRequest {
  mode?: ModeRequest;
  reasoning?: ReasoningLevel | "auto";
  /** a reasoning model for this request (admin, or one of SEARCH_PUBLIC_MODELS) */
  model?: string;
  max_output_tokens?: number;
  /** return the internal trace (SEARCH_TRACE=true, or the admin token) */
  trace?: boolean;
  /** include the evidence graph (always in deep and expert) */
  graph?: boolean;
  /** the caller holds the admin token: models, the expert mode and traces are allowed */
  admin?: boolean;
  /** the deepest mode auto may choose for this caller (the search box: SEARCH_QUERY_MAX_MODE); an explicit mode is not capped by it */
  ceiling?: Mode;
  /** benchmarks: no cached model answers */
  noCache?: boolean;
  /** benchmarks: stop before the reasoning model and return the frozen context */
  contextOnly?: boolean;
  /** a question about an uploaded image (spec §46-47): its hashes, colours and vector */
  image?: QueryImage & {
    /** a vision model's description of it, made on demand (the deeper modes, "what is it?") */
    describe?: () => Promise<QueryImage["description"] | null>;
  };
  // ---- v4.8 ----
  /** rich (default: SEARCH_ANSWER_STYLE): the long-form answer; brief: v4's */
  style?: AnswerStyle;
  /** full: answer_text carries the whole long-form answer; short (default): the direct answer, as v4 */
  text?: TextMode;
  /** how long the model's body should be, against the mode's target */
  length?: LengthRequest;
  /** rich: answer now with the digest, and leave the model's elaboration to GET /ask/elaboration/:query_id */
  defer?: boolean;
}

export type AskStatus = "answered" | "no_match" | "insufficient_evidence" | "conflict" | "clarify" | "not_found";

/** inline: the model's part is in this answer; pending: fetch it from url; ready: this is that part (GET /ask/elaboration); none: there is none */
export type ElaborationStatus = "inline" | "pending" | "ready" | "none";

export interface AskResponseV4 extends AskResponse {
  status: AskStatus;
  answer_type: string;
  /** the deterministic answer, when the operators gave one (spec §27: result, then explanation) */
  result_text?: string;
  /**
   * The reasoning model's explanation of that result, when it passed claim verification and
   * states the result's own values; answer_text is result_text followed by it.
   */
  explanation?: string;
  mode: Mode | "v3";
  reasoning: ReasoningLevel;
  model: string | null;
  class: QueryClass;
  complexity: number;
  band: string;
  subqueries: Subquery[];
  steps: Array<{ id: string; op: string; status: string; answer: unknown; answer_type: string; text: string; evidence: string[]; ms: number }>;
  claims: VerifiedClaim[];
  rationale?: string;
  contradictions: ConflictCard[];
  cards: EvidenceCard[];
  graph?: EvidenceGraph;
  /** the components of the confidence (confidence.ts), and the operator's own when the answer is deterministic */
  confidence_parts: Record<string, number>;
  grounding?: { egs: number; citation_accuracy: number; answer: string; counts: Record<string, number> };
  versions: Record<string, string | null>;
  usage?: { input_tokens: number; output_tokens: number; reasoning_tokens?: number; cost_usd: number | null; model_ms: number; cached?: boolean };
  timings: Record<string, number>;
  query_id: string;
  /** a question about an uploaded image: what it was asked, and how the index knows the image (spec §47) */
  image?: {
    task: string;
    identity: "exact_identity" | "perceptual_identity" | "visual_similarity" | "none";
    first_seen: { author: string; post: string; at: number; match: "exact" | "near"; deleted: boolean } | null;
    matches: Array<{ post_id: number; path: string; identity: ImageIdentity }>;
    hidden: number;
  };
  trace?: Record<string, unknown>;
  // ---- v4.8: the long-form answer (README "Rich answers") ----
  style: AnswerStyle;
  /** the direct answer: result_text then the explanation (what answer_text is unless text=full) */
  answer_short: string;
  /** the whole answer as plain text: the direct answer, the body, the index's facts, the reasoning trail, caveats, follow-ups */
  answer_full: string;
  /** the same as Markdown, citations kept ([E12], [R1]) for a UI that links them to cards */
  answer_markdown: string;
  sections: AnswerSections;
  /** the model's reasoning trail, verified step by step (sections.thinking) */
  thinking: ThinkingStep[];
  suggestions: { follow_ups: SuggestionItem[]; searches: SuggestionItem[] };
  digest: { stats: DigestStats; about: string[] };
  /** words of the body, and the target the model was given */
  length: { words: number; target: number | null };
  /** where the model's part of a rich answer is: inline, pending at url (defer=1), or none (fast mode, brief style, nothing to explain) */
  elaboration: { status: ElaborationStatus; url?: string; reason?: string };
}

/**
 * What GET /ask/elaboration/:query_id returns: pending while another call runs the model on this
 * answer, ready with the fields the model's part changed (merge them over the first answer: its
 * notes already include the first answer's), failed when the model did not answer (a transient
 * failure is tried again by the next call, up to ELABORATION_ATTEMPTS; a reply the engine could
 * not use is final), unknown for an id that expired or never was.
 */
export interface ElaborationResponse {
  query_id: string;
  status: "pending" | "ready" | "failed" | "unknown";
  /** the fields of the answer that the elaboration changed (merge them over the first answer) */
  answer?: Pick<AskResponseV4, "status" | "answer_text" | "answer_short" | "answer_full" | "answer_markdown" | "explanation" | "sections" | "thinking" | "suggestions" | "claims" | "rationale" | "grounding" | "usage" | "model" | "reasoning" | "confidence" | "confidence_parts" | "length" | "elaboration"> & { versions: Record<string, string | null>; timings: Record<string, number>; notes: string[] };
  error?: string;
  /** failed: whether the next call will try the model again */
  retry?: boolean;
  took_ms: number;
}

const SUMMARY = /\b(what kind|what type|what sort|describe|summari[sz]e|tell me about|what does @?[a-z0-9.-]+ (draw|post|make|paint|create)|quel genre|quel type|quels types|decri|resume|parle moi|was fur|welche art|beschreib|zusammenfass|que tipo|que clase|describe|resume|che tipo|che genere|descrivi|riassum)\b/;

const ORDER: Mode[] = ["fast", "balanced", "deep", "expert"];

/** Words only one of the five languages writes (v3's guess reads "più" as French: its ù). */
const LANG_HINTS: Array<[Lang, RegExp]> = [
  ["it", /(?<!\p{L})(più|è|perché|pubblicat\p{L}*|postat\p{L}*|quant[ie]|quale|questa|questo|quella|quello|opere|immagin[ei]|disegnat\p{L}*|chi ha|ha pubblicato|ha postato|ha creato|ha fatto|è stat[oa])(?!\p{L})/iu],
  ["es", /[¿¡ñ]|(?<!\p{L})(publicó|publicad[oa]s?|más|cuánt[oa]s?|quién|cuál|obras|imágenes|dibujó|creó|hizo|esta imagen|esta obra)(?!\p{L})/iu],
  ["fr", /(?<!\p{L})(œuvres?|a-t-il|a-t-elle|est-ce|combien|publiée?s?|postée?s?|pourquoi|quand|cette image|cette œuvre|ça)(?!\p{L})|qu['’]/iu],
  ["de", /(?<!\p{L})(gepostet|veröffentlicht|kunstwerke?|bilder|wie viele|welche[rs]?|warum|wurde|dieses bild|wer hat)(?!\p{L})/iu],
];

/** v4's language of a question: v3's guess, unless only another language's own words are in it. */
export function questionLang(question: string, guessed: Lang): Lang {
  // the words of a quoted title or an account are not the question's (“Ça va”, «Roma è bella», @piu)
  const own = withoutQuoted(question).replace(/@[\p{L}\p{N}._-]+/gu, " ");
  const hits = LANG_HINTS.filter(([, re]) => re.test(own)).map(([l]) => l);
  if (!hits.length || hits.includes(guessed)) return guessed;
  return hits.length === 1 ? hits[0] : guessed;
}

/** The deepest mode a caller may ask for. */
function maxModeFor(env: Env, admin: boolean): Mode {
  if (admin) return "expert";
  const m = String(env.SEARCH_MAX_MODE ?? "deep") as Mode;
  return ORDER.includes(m) ? m : "deep";
}

function pickModel(env: Env, a: Pick<AskRequestV4, "model" | "admin">, route: Pick<QueryRoute, "band">, notes: string[]): string {
  const wanted = a.model?.trim();
  if (wanted) {
    const allowed = a.admin || list(env.SEARCH_PUBLIC_MODELS).includes(wanted);
    if (allowed && isModelId(wanted, env)) return wanted;
    notes.push(allowed ? `unknown model ${wanted}: the configured one is used` : `model ${wanted} is not available to this caller: the configured one is used`);
  }
  return modelFor(env, "reasoning", route.band);
}

/** The answer style of a request: its own, else SEARCH_ANSWER_STYLE, else rich. */
export function answerStyle(env: Env, requested?: AnswerStyle): AnswerStyle {
  if (requested) return requested;
  const s = String(env.SEARCH_ANSWER_STYLE ?? "rich").trim().toLowerCase();
  return isAnswerStyle(s) ? s : "rich";
}

const r3 = (x: number) => Math.round(x * 1000) / 1000;

/** The context notes the model needs to read the cards right. */
function contextNotes(cards: EvidenceCard[], outcomes: StepOutcome[]): string[] {
  const out: string[] = [];
  if (cards.some((c) => c.type === "query_image")) {
    out.push("I1 is the image the question is about. Each artwork card's identity says how it relates to I1: exact (the same bytes), perceptual.near_identical (a re-encoded copy: same shapes and colours), visual (similarity only: never the same artwork), historical (whether it traces back to the same first post).");
    if (cards.some((c) => c.type === "query_image" && c.description)) out.push("I1.description is a vision model's description of the uploaded image, not a fact from the index.");
  }
  if (cards.some((c) => c.type === "result")) out.push('"result" cards are exact answers computed by the index from the evidence; your answer must agree with them.');
  if (cards.some((c) => c.type === "result" && !c.complete)) out.push("A result with complete: false is a lower bound: more matching posts may exist.");
  if (cards.some((c) => (c.type === "artwork" || c.type === "post") && c.history_exact === false)) out.push("Cards with history_exact false have a history inferred from dates, not read from chain operations.");
  if (cards.some((c) => (c.type === "artwork" || c.type === "post") && c.deleted)) out.push("Cards with deleted true are posts their authors deleted; they are evidence of what was posted, not posts anyone can see.");
  if (cards.some((c) => c.type === "conflict")) out.push('"conflict" cards report evidence that disagrees: report the disagreement.');
  if (cards.some((c) => (c.type === "artwork" || c.type === "post") && c.ai_caption)) out.push("ai_caption is a model's description of the image, not the author's.");
  if (outcomes.some((o) => o.status === "no_match")) out.push("A result with answer null or 0 means the index has no matching post.");
  return out;
}

/** The status of a deterministic outcome. */
function statusOf(o: StepOutcome, conflicts: ConflictCard[]): AskStatus {
  if (o.status === "not_found") return "not_found";
  if (o.status === "ambiguous") return "clarify";
  if (o.status === "failed") return "insufficient_evidence";
  if (o.status === "no_match") return "no_match";
  return conflicts.length ? "conflict" : "answered";
}

/**
 * Context compression (spec §43): which artwork cards the model sees, deciding evidence first,
 * then the candidates that passed verification, ordered by their verification score blended with
 * the cross-encoder's (SEARCH_RERANK_BLEND, as /search: alone it ranks worse, see reranker.ts).
 */
function selectCards(outcomes: StepOutcome[], profile: ExecutionProfile, rerank: Map<number, number> | null, beta = 0.2): Verified[] {
  const picked = new Map<number, Verified>();
  const add = (x: Verified | undefined) => {
    if (x && !picked.has(x.row.id)) picked.set(x.row.id, x);
  };
  const final = outcomes[outcomes.length - 1];
  for (const x of final.result.evidence.slice(0, 6)) add(x);
  for (const o of outcomes) for (const x of o.result.evidence.slice(0, 4)) add(x);
  for (const o of outcomes) for (const x of o.result.alternatives ?? []) add(x);
  // the rest of the candidates that passed, by the cross-encoder when it ran, else by their verification
  const rest = outcomes.flatMap((o) => o.scope?.verified ?? []).filter((x) => !picked.has(x.row.id));
  const key = (x: Verified) => (rerank && rerank.has(x.row.id) ? (1 - beta) * x.v.score + beta * rerank.get(x.row.id)! : x.v.score);
  rest.sort((a, b) => key(b) - key(a) || b.v.score - a.v.score || a.row.id - b.row.id);
  for (const x of rest) {
    if (picked.size >= profile.cards) break;
    add(x);
  }
  return [...picked.values()].slice(0, Math.max(profile.cards, final.result.evidence.length ? 1 : 0));
}

// ---- the model's reply, applied (shared by the inline and the deferred paths) ---------------------------------

/** Everything the model's reply is judged against, frozen when the answer is deferred. */
interface LeadInput {
  l: AnswerLang;
  resultText: string;
  /** the index answered, so the model only explains */
  hasDeterministic: boolean;
  /** the results the deterministic answer states: an explanation must state their values too */
  shownCards: ResultCard[];
  /** the cards that passed verification: what the model saw and what its claims are checked against */
  usable: EvidenceCard[];
  /** the accounts of the evidence (names without "@" are read as accounts) */
  accounts: string[];
  strictClaims: boolean;
  verifyClaimsOn: boolean;
  /** "describe" for "what is this image?", else null */
  task: string | null;
  finalStatus: string;
  /** the status before the model: answered, conflict … */
  status: AskStatus;
  /** the conflict sentences to append to a model answer that replaces nothing */
  conflictText: string;
  routeClass: QueryClass;
  hasPremise: boolean;
  /** rich answers: the direct answer is read by people, its citations stay in the claims and the body */
  stripCitations: boolean;
}

const stripCites = (s: string) => s.replace(/\s*\[(?:[A-Za-z]{1,2}\d{1,9}|\d{1,2})\]/g, "").replace(/\s+/g, " ").trim();

interface LeadOutcome {
  status: AskStatus;
  answerText: string;
  explanation?: string;
  rationale?: string;
  modelUsed: boolean;
  cv: ClaimVerification | null;
  /** the lead contradicted the index's result: nothing else of the reply is trusted */
  contradicted: boolean;
  notes: string[];
}

/** spec §27: when the index answers, its answer is said first and the model only explains it. */
function applyLead(f: LeadInput, reply: ReasoningResponse, answerText0: string): LeadOutcome {
  const notes: string[] = [];
  let status = f.status;
  let answerText = answerText0;
  let explanation: string | undefined;
  let rationale: string | undefined;
  let modelUsed = false;
  let contradicted = false;
  const accounts = new Set(f.accounts);
  const cv = f.verifyClaimsOn ? verifyClaims(reply, f.usable, { strict: f.strictClaims, authors: accounts }) : null;
  const agreement = f.hasDeterministic && cv ? agreesWithResult(reply.answer, f.shownCards, f.usable, { authors: accounts }) : { ok: true, problems: [] as string[] };
  const answerOk = !cv || cv.answer.status === "supported" || (cv.answer.status === "qualified" && !f.strictClaims);
  const explains = f.hasDeterministic || (f.task === "describe" && f.finalStatus === "no_match");
  if (reply.status === "insufficient_evidence") {
    if (explains) notes.push("the model judged the evidence insufficient; the index's own answer stands");
    else {
      status = "insufficient_evidence";
      answerText = say("insufficient", f.l);
    }
  } else if (answerOk && agreement.ok && reply.answer.trim()) {
    modelUsed = true;
    if (explains) {
      explanation = f.stripCitations ? stripCites(reply.answer) : reply.answer.trim();
      answerText = `${f.resultText} ${explanation}`.trim();
      // "what is this image?" is answered by the description even when no indexed artwork is the same image
      if (f.task === "describe" && status === "no_match") status = "answered";
    } else {
      answerText = f.stripCitations ? stripCites(reply.answer) : reply.answer;
      if (f.conflictText && !/\d{4}-\d{2}-\d{2}.*\d{4}-\d{2}-\d{2}/.test(answerText)) answerText = `${answerText} ${f.conflictText}`;
    }
    // a conflict is the evidence verifier's to find: the model's word alone does not change the status
    if (reply.status === "conflict" && !f.conflictText) notes.push("the model reported a conflict the evidence verifier did not find");
  } else {
    const why = !answerOk ? `${cv!.answer.status} (${cv!.answer.problems.join("; ").slice(0, 200)})` : !agreement.ok ? `it does not state the index's answer (${agreement.problems.join("; ").slice(0, 200)})` : "empty";
    notes.push(`the model's ${explains ? "explanation" : "answer"} was not used: ${why}`);
    contradicted = !!cv && cv.answer.status === "contradicted";
    if (!explains) {
      const kept = cv ? keptClaims(cv, f.strictClaims).filter((c) => c.kind !== "inference" || c.status === "supported") : [];
      if (kept.length) {
        modelUsed = true;
        answerText = kept.map((c) => c.text).join(" ");
      } else {
        status = "insufficient_evidence";
        answerText = say("insufficient", f.l);
      }
    }
  }
  // the rationale of a reply that was used, checked as strictly as the answer
  const rationaleOk = !cv || cv.rationale?.status === "supported" || (!f.strictClaims && cv.rationale?.status === "qualified");
  if (modelUsed && reply.rationale && rationaleOk) rationale = reply.rationale;
  // "why …?": the index holds what happened, not why. Without a grounded explanation from the
  // model, the facts are given as facts and the question is not answered (a premise check is).
  if (f.routeClass === "EXPLANATORY" && status === "answered" && !explanation && !(modelUsed && !f.hasDeterministic) && !f.hasPremise) {
    status = "insufficient_evidence";
    answerText = `${say("insufficient", f.l)} ${f.resultText}`.trim();
  }
  return { status, answerText, explanation, rationale, modelUsed, cv, contradicted, notes };
}

// ---- the long-form answer -------------------------------------------------------------------------------------

interface ComposeInput {
  env: Env;
  lang: Lang;
  question: string;
  style: AnswerStyle;
  /** the direct answer (answer_text): the lead */
  lead: string;
  digest: Digest;
  /** the model's reply, when one was used for the lead (or its body alone may be used) */
  reply: ReasoningResponse | null;
  /** the lead contradicted the result: the reply's body is not shown */
  contradicted: boolean;
  usable: EvidenceCard[];
  accounts: Set<string>;
  knownAuthors: Set<string>;
  strict: boolean;
  target: number | null;
  notes: string[];
}

interface ComposedAnswer {
  sections: AnswerSections;
  full: string;
  markdown: string;
  words: number;
  /** the body's sentence checks, for the trace */
  body: { removed: number; egs: number; citation_accuracy: number; claims: VerifiedClaim[] } | null;
  removed: { thinking: number; caveats: number; follow_ups: number; searches: number };
}

/** The sections of the answer: the lead, the model's verified body and trail, the digest, the suggestions. */
async function composeAnswer(x: ComposeInput): Promise<ComposedAnswer> {
  const sections: AnswerSections = { lead: x.lead, facts: [], overview: [], thinking: [], caveats: [], follow_ups: [], searches: [] };
  let body: ComposedAnswer["body"] = null;
  const removed = { thinking: 0, caveats: 0, follow_ups: 0, searches: 0 };
  if (x.style === "brief") {
    return { sections, full: x.lead, markdown: x.lead, words: countWords(x.lead), body, removed };
  }
  // a reply whose lead contradicted the result, or that judged the evidence insufficient, has no body to show
  const reply = x.reply && !x.contradicted && x.reply.status !== "insufficient_evidence" ? x.reply : null;
  if (x.reply && x.contradicted) x.notes.push("the model's body and reasoning trail were not used: its answer contradicted the index's result");
  if (reply?.body) {
    const v = verifyMarkdown(reply.body, x.usable, { strict: x.strict, authors: x.accounts });
    body = { removed: v.removed, egs: v.egs, citation_accuracy: v.citation_accuracy, claims: v.claims };
    if (v.removed) x.notes.push(`${v.removed} sentence(s) of the body removed: not supported by the evidence`);
    // a body that only repeats the lead adds nothing
    if (v.text && fold(v.text) !== fold(x.lead)) sections.body = v.text;
  }
  if (reply?.thinking?.length) {
    const v = verifyList(reply.thinking, x.usable, { strict: x.strict, authors: x.accounts });
    removed.thinking = v.removed;
    if (v.removed) x.notes.push(`${v.removed} step(s) of the reasoning trail removed: not supported by the evidence`);
    if (v.kept.length >= 2) sections.thinking = v.kept.map((s, i) => ({ n: i + 1, text: s.text, evidence: s.evidence, status: s.status }));
  }
  sections.overview = x.digest.overview;
  sections.facts = x.digest.facts;
  const caveats = [...x.digest.caveats];
  if (reply?.caveats?.length) {
    const v = verifyList(reply.caveats, x.usable, { strict: x.strict, authors: x.accounts });
    removed.caveats = v.removed;
    for (const c of v.kept) if (!caveats.some((d) => fold(d) === fold(c.text))) caveats.push(c.text);
  }
  sections.caveats = caveats.slice(0, 8);
  const s = await suggestions({ env: x.env, authors: x.knownAuthors, lang: x.lang, cards: x.usable, question: x.question }, reply ? { followUps: reply.followUps, searches: reply.searches } : null, x.digest, { limit: followUpLimit(x.env), notes: x.notes });
  sections.follow_ups = s.follow_ups;
  sections.searches = s.searches;
  removed.follow_ups = s.removed.follow_ups;
  removed.searches = s.removed.searches;
  const markdown = assembleMarkdown({ lang: x.lang, sections });
  return { sections, full: toPlainText(markdown), markdown, words: countWords(sections.body ?? "") + countWords(x.lead), body, removed };
}

// ---- deferred elaboration ---------------------------------------------------------------------------------------

/** The frozen context of a deferred answer, kept in KV until GET /ask/elaboration/:qid runs the model. */
interface FrozenElaboration {
  v: 1;
  query_id: string;
  question: string;
  lang: Lang;
  mode: Mode;
  reasoning: ReasoningLevel;
  maxOutputTokens: number;
  strictClaims: boolean;
  model: string;
  words: number;
  modelCards: EvidenceCard[];
  context: string[];
  lead: LeadInput;
  answerText0: string;
  digest: Digest;
  /** full: answer_text carries the whole answer (as the first answer did) */
  text: TextMode;
  /** the first answer's notes: the elaboration's answer carries them too */
  notes: string[];
  /** the parts of the confidence the model does not change, and the operator's confidence the cap follows */
  confidence: { parts: Record<string, number | null>; planConfidence: number; operator: number | null; operatorConfidence: number; hasDeterministic: boolean };
  versions: Record<string, string | null>;
  cacheTtl: number;
  at: number;
  /** model calls made so far (the next one is refused past ELABORATION_ATTEMPTS) */
  attempts: number;
}

/** Model calls a deferred answer may make before its elaboration is given up (transient failures). */
export const ELABORATION_ATTEMPTS = 3;

/** The follow-ups and searches of an answer, for GET /suggest?after=<query_id> (search/suggest.ts). */
function keepSuggestions(env: Env, qid: string, s: { follow_ups: SuggestionItem[]; searches: SuggestionItem[] }): Promise<void> {
  const body: AnswerSuggestions = { follow_ups: s.follow_ups.map((f) => ({ text: f.text, route: f.route })), searches: s.searches.map((x) => ({ text: x.text, route: "search" })) };
  return env.CACHE.put(answerSuggestionsKey(qid), JSON.stringify(body), { expirationTtl: elaborationTtl(env) }).catch(() => {});
}

const elabKey = (qid: string) => `elab:${qid}`;
const elabDoneKey = (qid: string) => `elab:${qid}:done`;
const elabRunKey = (qid: string) => `elab:${qid}:run`;
const elaborationTtl = (env: Env) => Math.max(120, int(env.SEARCH_ELABORATION_TTL, 1800));
const elaborationUrl = (qid: string) => `/ask/elaboration/${qid}`;
/** Seconds a running elaboration holds its marker: longer than the model's timeout, so a second call never doubles a call in flight. */
const RUN_MARKER_TTL = 120;

/**
 * GET /ask/elaboration/:qid: the model's part of a deferred rich answer. Runs the model on the
 * frozen context once — a marker holds while it runs, so calls that overlap are told to wait
 * (pending), and the result, or a final failure, is kept — and returns the fields it changed.
 * One /ask may thus cost at most ELABORATION_ATTEMPTS model calls, however often it is polled.
 */
export async function elaborate(env: Env, qid: string): Promise<ElaborationResponse> {
  const t0 = Date.now();
  if (!/^[A-Za-z0-9_-]{6,64}$/.test(qid)) return { query_id: qid, status: "unknown", error: "no such answer", took_ms: 0 };
  const done = (await env.CACHE.get(elabDoneKey(qid), "json").catch(() => null)) as ElaborationResponse | null;
  if (done && (done.status === "ready" || done.status === "failed")) return { ...done, took_ms: Date.now() - t0 };
  const frozen = (await env.CACHE.get(elabKey(qid), "json").catch(() => null)) as FrozenElaboration | null;
  if (!frozen || frozen.v !== 1) return { query_id: qid, status: "unknown", error: "no deferred answer with this id (it may have expired: answers wait SEARCH_ELABORATION_TTL seconds)", took_ms: Date.now() - t0 };
  if (await env.CACHE.get(elabRunKey(qid)).catch(() => null)) return { query_id: qid, status: "pending", error: "the model is answering: ask again in a moment", took_ms: Date.now() - t0 };
  const attempts = (frozen.attempts ?? 0) + 1;
  const keep = (r: ElaborationResponse) => env.CACHE.put(elabDoneKey(qid), JSON.stringify(r), { expirationTtl: elaborationTtl(env) }).catch(() => {});
  if (attempts > ELABORATION_ATTEMPTS) {
    const out: ElaborationResponse = { query_id: qid, status: "failed", error: `the model did not answer in ${ELABORATION_ATTEMPTS} attempts`, retry: false, took_ms: Date.now() - t0 };
    await keep(out);
    await env.CACHE.delete(elabKey(qid)).catch(() => {});
    return out;
  }
  // the marker, and the attempt counted, before the model is called
  await env.CACHE.put(elabRunKey(qid), "1", { expirationTtl: RUN_MARKER_TTL }).catch(() => {});
  await env.CACHE.put(elabKey(qid), JSON.stringify({ ...frozen, attempts }), { expirationTtl: elaborationTtl(env) }).catch(() => {});
  const notes: string[] = [...(frozen.notes ?? [])];
  const timings: Record<string, number> = {};
  const ctx = await loadContext(env);
  let reply: (ReasoningResponse & { cached?: boolean }) | null = null;
  let failure: { message: string; final: boolean } | null = null;
  let tp = Date.now();
  try {
    reply = await generateCached(
      env,
      reasoningModel(env, frozen.model),
      { question: frozen.question, evidence: frozen.modelCards as unknown as Array<{ evidence_id: string; type: string }>, reasoning: frozen.reasoning, maxTokens: frozen.maxOutputTokens, lang: frozen.lang, context: frozen.context, task: "compose", words: frozen.words },
      frozen.cacheTtl,
    );
  } catch (e) {
    const transient = e instanceof LlmError && e.retryable;
    failure = { message: `reasoning model unavailable (${frozen.model}): ${e instanceof Error ? e.message : String(e)}${transient ? " (transient)" : e instanceof UnusableReply ? " (unusable reply)" : ""}`, final: !transient };
  } finally {
    await env.CACHE.delete(elabRunKey(qid)).catch(() => {});
  }
  timings.model = Date.now() - tp;
  if (reply?.notes?.length) notes.push(...reply.notes);
  if (!reply) {
    // a transient failure is tried again by the next call (up to the attempts); a reply the
    // engine cannot use, or the last attempt, is final: the first answer stands as it was
    const final = failure!.final || attempts >= ELABORATION_ATTEMPTS;
    const out: ElaborationResponse = { query_id: qid, status: "failed", error: failure!.message, retry: !final, took_ms: Date.now() - t0 };
    if (final) {
      await keep(out);
      await env.CACHE.delete(elabKey(qid)).catch(() => {});
    }
    return out;
  }
  tp = Date.now();
  const lead = applyLead(frozen.lead, reply, frozen.answerText0);
  notes.push(...lead.notes);
  timings.claims = Date.now() - tp;
  tp = Date.now();
  const accounts = new Set(frozen.lead.accounts);
  const composed = await composeAnswer({ env, lang: frozen.lang, question: frozen.question, style: "rich", lead: lead.answerText, digest: frozen.digest, reply, contradicted: lead.contradicted, usable: frozen.lead.usable, accounts, knownAuthors: ctx.authors, strict: frozen.strictClaims, target: frozen.words, notes });
  timings.compose = Date.now() - tp;
  const cv = lead.cv;
  const c = frozen.confidence;
  const conf = combineConfidence(
    {
      retrieval: c.parts.retrieval ?? 0,
      reranking: c.parts.reranking ?? null,
      agreement: c.parts.agreement ?? 1,
      verification: c.hasDeterministic ? (lead.modelUsed && cv ? 0.5 + 0.5 * cv.egs : 1) : lead.modelUsed ? (cv ? (cv.answer.status === "supported" ? 0.5 + 0.5 * cv.egs : 0.5 * cv.egs) : 0.6) : 0.3,
      model: lead.modelUsed ? (reply.confidence ?? null) : null,
      planConfidence: c.planConfidence,
    },
    confidenceWeights(env),
  );
  // the caps, as the inline path applies them once the lead settled the status
  let confidence = conf.value;
  if (c.hasDeterministic && c.operatorConfidence > 0) confidence = r3(Math.min(confidence, c.operatorConfidence));
  if (lead.status === "no_match") confidence = r3(Math.min(confidence, c.operatorConfidence || confidence));
  if (lead.status === "insufficient_evidence" || lead.status === "clarify" || lead.status === "not_found") confidence = r3(Math.min(confidence, 0.3));
  const usage = reply.usage ? { input_tokens: reply.usage.inputTokens, output_tokens: reply.usage.outputTokens, ...(reply.usage.reasoningTokens !== undefined ? { reasoning_tokens: reply.usage.reasoningTokens } : {}), cost_usd: reply.costUsd, model_ms: reply.latencyMs, ...(reply.cached ? { cached: true } : {}) } : undefined;
  const answer: NonNullable<ElaborationResponse["answer"]> = {
    status: lead.status,
    answer_text: frozen.text === "full" ? composed.full : lead.answerText,
    answer_short: lead.answerText,
    answer_full: composed.full,
    answer_markdown: composed.markdown,
    ...(lead.explanation ? { explanation: lead.explanation } : {}),
    sections: composed.sections,
    thinking: composed.sections.thinking,
    suggestions: { follow_ups: composed.sections.follow_ups, searches: composed.sections.searches },
    claims: cv?.claims ?? [],
    ...(lead.rationale ? { rationale: lead.rationale } : {}),
    ...(cv ? { grounding: { egs: cv.egs, citation_accuracy: cv.citation_accuracy, answer: cv.answer.status, counts: cv.counts } } : {}),
    ...(usage ? { usage } : {}),
    model: reply.model,
    reasoning: reply.reasoning,
    confidence,
    confidence_parts: { ...conf.parts, ...(c.operator !== null ? { operator: c.operator } : {}) },
    length: { words: composed.words, target: frozen.words },
    elaboration: { status: "ready" },
    versions: { ...frozen.versions, reasoning_model: reply.model, prompt: PROMPT_VERSION },
    timings,
    notes: [...new Set(notes)],
  };
  const out: ElaborationResponse = { query_id: qid, status: "ready", answer, took_ms: Date.now() - t0 };
  await keep(out);
  await keepSuggestions(env, qid, answer.suggestions);
  await env.CACHE.delete(elabKey(qid)).catch(() => {});
  await env.DB.prepare("UPDATE ask_log SET model = ?, status = ?, confidence = ?, egs = ?, claims = ?, claims_supported = ?, input_tokens = ?, output_tokens = ?, cost_usd = ?, model_ms = ? WHERE qid = ?")
    .bind(reply.model, lead.status, confidence, cv?.egs ?? null, cv?.claims.length ?? 0, cv?.claims.filter((x) => x.status === "supported").length ?? 0, usage?.input_tokens ?? null, usage?.output_tokens ?? null, usage?.cost_usd ?? null, usage?.model_ms ?? null, qid)
    .run()
    .catch(() => {});
  return out;
}

// ---- /ask -----------------------------------------------------------------------------------------------------

export async function ask(env: Env, a: AskRequestV4, exec?: Pick<ExecutionContext, "waitUntil">): Promise<AskResponseV4 | (AskResponse & { mode: "v3" })> {
  if (a.mode === "v3") return { ...(await askV3(env, a)), mode: "v3" as const };
  const t0 = Date.now();
  const timings: Record<string, number> = {};
  const mark = (k: string, since: number) => (timings[k] = (timings[k] ?? 0) + Date.now() - since);
  const notes: string[] = [];
  const queryId = randomId();
  const ctx = await loadContext(env);
  const original = a.question.slice(0, 300);
  // instructions to the system inside the question are not part of it (spec §42)
  const cleaned = withoutInstructions(original);
  const question = cleaned.text;
  if (cleaned.dropped.length) notes.push(`instructions inside the question were ignored: ${cleaned.dropped.map((x) => `“${x.slice(0, 80)}”`).join(" ")}`);
  const limit = Math.min(50, Math.max(1, a.limit ?? 10));
  const nsfw: NsfwMode = a.nsfw ?? "exclude";
  const style = answerStyle(env, a.style);
  const rich = style === "rich";

  // ---- plan, program, route, mode ----------------------------------------------------------------
  let tp = Date.now();
  const plannerBefore = notes.length;
  // a question about an image is about the image: the rules plan its language and filters, no planner model
  const plan = await planQuestion(env, question, { ...a, ...(a.image ? { planner: "rules" } : {}), ...(cleaned.dropped.length ? { plan: undefined } : {}) }, ctx.authors, notes, { v4: true });
  const plannerUsed = plan.source !== "rules" || notes.length > plannerBefore;
  // the rules guess the language from the question's own words, not from a title it quotes (“Il Duomo è bello”)
  const lang: Lang = questionLang(question, plan.source === "rules" ? guessLang(withoutQuoted(question).trim() || question) : (plan.lang ?? guessLang(question)));
  const pctx = { authors: ctx.authors, lang, now: ctx.now };
  let program: QueryProgram = a.image
    ? { steps: [{ id: "q1", op: "image", text: question }], final: "q1", source: "rules", pattern: "image", titles: [], extraEntities: 0, notes: [] }
    : decomposeRules(question, plan, pctx);
  let route = routeQuestion(question, plan, program, { env, image: !!a.image });
  const task = a.image ? imageTask(question) : null;
  const needsSynthesis = route.class === "EXPLANATORY" || SUMMARY.test(fold(question)) || task === "describe";
  // the caller's deepest mode caps everything; the search box's ceiling only what auto picks; rich
  // answers raise what auto picks to SEARCH_RICH_MIN_MODE, so the model elaborates
  const most = maxModeFor(env, !!a.admin);
  const chosen = chooseMode(env, a.mode, route, { needsSynthesis, ceiling: a.ceiling, max: most, floor: rich ? richMinMode(env) : undefined });
  const profile: ExecutionProfile = { ...PROFILES[chosen.mode] };
  const reasoningAsked = a.reasoning && a.reasoning !== "auto" && isReasoningLevel(a.reasoning) ? a.reasoning : null;
  const envReasoning = isReasoningLevel(env.SEARCH_REASONING) ? (env.SEARCH_REASONING as ReasoningLevel) : null;
  if (reasoningAsked) profile.reasoning = reasoningAsked;
  else if (envReasoning && profile.reasoning !== "none") profile.reasoning = envReasoning;
  const cap = int(env.SEARCH_MAX_OUTPUT_TOKENS, 6000);
  // rich answers: the long-form reply's room (COMPOSE_TOKENS); brief: v4's
  if (rich) profile.maxOutputTokens = COMPOSE_TOKENS[profile.mode];
  profile.maxOutputTokens = Math.min(cap, Math.max(64, a.max_output_tokens ? Math.min(a.max_output_tokens, profile.maxOutputTokens * 2) : profile.maxOutputTokens));
  if (!bool(env.SEARCH_RERANK, true)) profile.reranking = false;
  // the evidence the model reads: at most SEARCH_FINAL_K artwork cards, whatever the mode
  profile.cards = Math.max(1, Math.min(profile.cards, int(env.SEARCH_FINAL_K, 20)));
  // one planner call per question: the decomposition model only when the rules planner did not call it
  if (!a.image && !plannerUsed && profile.llmDecomposition && program.pattern === "single" && looksMultiStep(question) && plan.intent !== "similar" && plan.intent !== "duplicate") {
    const p = await decompose(env, question, plan, pctx, { allowModel: true, notes });
    if (p.steps.length > 1) {
      program = p;
      route = routeQuestion(question, plan, program, { env });
    }
  }
  mark("plan", tp);

  // ---- execute --------------------------------------------------------------------------------------
  tp = Date.now();
  let identities: Map<number, ImageIdentity> | null = null;
  let findings: ImageFindings | null = null;
  let imageCard: QueryImageCard | null = null;
  let outcomes: StepOutcome[];
  if (a.image) {
    // the uploaded image as a vision model sees it: in the deeper modes, or when asked what it shows
    const describeIt = a.image.describe && !a.image.description && (profile.mode === "deep" || profile.mode === "expert" || (task === "describe" && profile.mode !== "fast"));
    const [io, description] = await Promise.all([
      imageOutcome(env, program.steps[0], a.image, { lang, nsfw, plan, limit }),
      describeIt
        ? a.image.describe!().catch((e) => {
            notes.push(`image description unavailable: ${e instanceof Error ? e.message : String(e)}`);
            return null;
          })
        : Promise.resolve(null),
    ]);
    identities = io.identities;
    findings = io.findings;
    imageCard = queryImageCard(description ? { ...a.image, description } : a.image, io.findings);
    outcomes = [io];
  } else {
    outcomes = await executeProgram(
      { env, ctx, lang, limit, nsfw, type: a.type, threshold: a.threshold ?? 0.5, depth: depthFor(env, profile.retrieval), notes },
      program,
    );
  }
  mark("retrieve", tp);
  const det = composeDeterministic(program, outcomes);
  const final = det.final;

  // ---- reranking of the evidence the model will read ----------------------------------------------------
  let rerank: Map<number, number> | null = null;
  let rerankModel: string | null = null;
  const candidates = outcomes.flatMap((o) => o.scope?.verified ?? []);
  if (profile.reranking && candidates.length > profile.cards / 2) {
    tp = Date.now();
    const head = [...candidates].sort((x, y) => y.v.score - x.v.score).slice(0, int(env.SEARCH_RERANK_K, 50));
    try {
      const r = await rerankRows(env, final.plan?.residual || question, head.map((x) => x.row));
      if (r) (rerank = r.scores), (rerankModel = r.model);
    } catch (e) {
      notes.push(`reranker unavailable: ${e instanceof Error ? e.message : String(e)}`);
    }
    mark("rerank", tp);
  }

  // ---- evidence cards, graph, verification ------------------------------------------------------------
  tp = Date.now();
  let chosenRows = selectCards(outcomes, profile, rerank, rerankBlend(env));
  // an exact count over the filters ("how many artworks did @alice post?") keeps no rows: a rich
  // answer shows, and gives the model, the newest posts of the set, so there is something to say
  let metaRows: Verified[] = [];
  let metaExtremes: { first?: Row | null; top?: Row | null } | undefined;
  if (rich && !chosenRows.length && final.status === "ok" && final.scope?.kind === "metadata" && final.plan && !a.image) {
    // the newest rows, and the set's oldest and most voted read exactly (three small SQL reads)
    const meta = final.plan;
    const req = final.scope.req;
    const [rows, first, top] = await Promise.all([
      metadataRows(env, { ...meta, intent: "find_last" }, req, Math.max(profile.cards, 24)).catch(() => [] as Row[]),
      metadataRows(env, { ...meta, intent: "find_first" }, req, 1).catch(() => [] as Row[]),
      metadataRows(env, { ...meta, intent: "top", sort: "votes" }, req, 1).catch(() => [] as Row[]),
    ]);
    metaRows = rows.map((row) => ({ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["matches the filters"] } }));
    metaExtremes = { first: first[0] ?? null, top: top[0] ?? null };
    chosenRows = metaRows.slice(0, profile.cards);
  }
  const histories = new Map<string, HistoryFacts>();
  for (const o of outcomes) if (o.history) histories.set(o.history.post, o.history);
  const conceptOf = (o: StepOutcome) => o.plan?.concepts ?? [];
  const provenance = new Map<number, string[]>();
  for (const o of outcomes) for (const [id, p] of o.scope?.provenance ?? []) provenance.set(id, [...new Set([...(provenance.get(id) ?? []), ...p.sources])]);
  const artCards: ArtworkCard[] = chosenRows.map((x) => {
    const o = outcomes.find((oo) => oo.scope?.verified.includes(x) || oo.result.evidence.includes(x)) ?? final;
    const card = artworkCard(x.row, { v: x.v, provenance: provenance.has(x.row.id) ? { sources: provenance.get(x.row.id)!, scores: {} } : undefined, concepts: conceptOf(o), rerank: rerank?.get(x.row.id), history: histories.get(`/@${x.row.author}/${x.row.permlink}`) ?? null });
    const identity = identities?.get(x.row.id);
    if (identity) card.identity = identity;
    return card;
  });
  // posts named by the evidence but not among it: a first sighting in a deleted post, a history appearance
  const named = new Set<string>();
  for (const c of artCards) if (c.first_seen_in && c.first_seen_in !== c.path) named.add(c.first_seen_in);
  for (const h of histories.values()) for (const ap of h.appearances.slice(0, 6)) named.add(`/@${ap.author}/${ap.permlink}`);
  for (const o of outcomes) for (const h of o.scope?.history ?? []) named.add(`/@${h.author}/${h.permlink}`);
  const have = new Set(artCards.map((c) => c.path));
  const extra = await rowsByPath(env, [...named].filter((p) => !have.has(p)).slice(0, 12)).catch(() => new Map());
  // only posts the nsfw setting shows: a post named by a history is still a post of the index
  for (const row of extra.values()) if (visibleUnder(row, nsfw)) artCards.push(artworkCard(row, { history: histories.get(`/@${row.author}/${row.permlink}`) ?? null }));
  const idOf = new Map(artCards.map((c) => [c.artwork_id, c.evidence_id]));
  const results = outcomes.map((o, i) => resultCard(i + 1, o.step.id, o.result, { question: o.step.text, over: o.result.evidence.map((x) => idOf.get(x.row.id)).filter((x): x is string => !!x) }));
  let cards: EvidenceCard[] = [...(imageCard ? [imageCard] : []), ...results, ...artCards];
  const ev = await verifyEvidence(env, cards);
  cards = applyVerification(cards, ev);
  notes.push(...ev.notes);
  // a conflict about the evidence the answer rests on is said in the answer (spec §24)
  const decisive = new Set(final.result.evidence.slice(0, 1).map((x) => `E${x.row.id}`));
  const relevantConflicts = ev.conflicts.filter((c) => c.evidence.some((id) => decisive.has(id)));
  const invalidDecisive = [...decisive].filter((id) => ev.invalid.has(id));
  if (invalidDecisive.length) notes.push(`the deciding evidence failed verification (${invalidDecisive.map((id) => `${id}: ${ev.invalid.get(id)!.join(", ")}`).join("; ")})`);
  const graph = a.graph || profile.mode === "deep" || profile.mode === "expert" ? buildGraph(cards, histories) : undefined;
  mark("evidence", tp);

  const l = answerLang(lang);
  const pathOfCard = (id: string) => artCards.find((c) => c.evidence_id === id)?.path;
  let status: AskStatus = statusOf(final, relevantConflicts);
  if (invalidDecisive.length && status === "answered") status = "insufficient_evidence";
  // "who posted this?" with nothing that says what "this" is: ask back rather than answer about every artwork
  const unnamed = route.class === "AMBIGUOUS" && route.signals.ambiguity.includes("refers to something the question does not name") && !a.image;
  if (unnamed) status = "clarify";
  // nothing may be answered from: the question is asked back, or the deciding evidence failed verification (spec §16)
  const withheld = unnamed || invalidDecisive.length > 0;
  let resultText = det.text;
  const conflictText = relevantConflicts.map((c) => conflictSentence(c, l, pathOfCard)).join(" ");
  if (relevantConflicts.length) resultText = `${resultText} ${conflictText}`;

  const hasDeterministic = det.complete && final.status === "ok" && !withheld;
  // the results the deterministic answer states: an explanation must state their values too
  const shownCards = det.shown.map((o) => results[outcomes.indexOf(o)]).filter((c): c is ResultCard => !!c);

  // ---- reasoning ------------------------------------------------------------------------------------------
  let reply: (ReasoningResponse & { cached?: boolean }) | null = null;
  let model: string | null = null;
  const explicit = chosen.explicit || !!reasoningAsked;
  // worth its cost (spec §53): asked for, a synthesis no operator gives, evidence no operator could
  // use — or, for rich answers, any answer the model may elaborate
  const worth = explicit || needsSynthesis || rich || (!hasDeterministic && final.status === "ok" && cards.some((c) => c.type === "artwork"));
  // the model explains what the index found: never where it found nothing, a step failed, or the question is asked back
  const explainable = (status === "answered" || status === "conflict" || (task === "describe" && status === "no_match")) && !withheld;
  const useModel = profile.reasoning !== "none" && worth && explainable;
  // the model sees only evidence that passed verification (spec §16)
  const usable = cards.filter((c) => !((c.type === "artwork" || c.type === "post") && c.valid === false));
  const modelCards = usable.map((c) => modelView(c) as unknown as EvidenceCard);
  const context = contextNotes(usable, outcomes);
  if (a.contextOnly) {
    return contextResponse();
  }
  const words = rich ? wordTarget(env, profile.mode, a.length) : null;
  const deferred = useModel && rich && !!a.defer;
  if (useModel && !deferred) {
    model = pickModel(env, a, route, notes);
    tp = Date.now();
    try {
      reply = await generateCached(
        env,
        reasoningModel(env, model),
        { question, evidence: modelCards as unknown as Array<{ evidence_id: string; type: string }>, reasoning: profile.reasoning, maxTokens: profile.maxOutputTokens, lang, context, task: rich ? "compose" : "answer", ...(words ? { words } : {}) },
        a.noCache ? 0 : int(env.SEARCH_ANSWER_CACHE_TTL, 86400),
      );
    } catch (e) {
      notes.push(`reasoning model unavailable (${model}): ${e instanceof Error ? e.message : String(e)}${e instanceof LlmError && e.retryable ? " (transient)" : e instanceof UnusableReply ? " (unusable reply)" : ""}`);
    }
    // what the model layer had to adapt (a reply cut off and salvaged, a reasoning level mapped): said, as the trace says it
    if (rich && reply?.notes?.length) notes.push(...reply.notes);
    mark("reason", tp);
  }

  // ---- claim verification and the final answer -------------------------------------------------------------
  // spec §27: when the index answers, its answer is said first and the model only explains it; the
  // explanation is shown when its claims pass verification and it states the result's own values
  const answerText0 = unnamed ? say("clarify_reference", l) : invalidDecisive.length ? say("insufficient", l) : resultText;
  let answerText = answerText0;
  let explanation: string | undefined;
  let rationale: string | undefined;
  let modelUsed = false;
  let contradicted = false;
  let cv: ClaimVerification | null = null;
  const accounts = new Set<string>();
  for (const c of cards) if (c.type === "artwork" || c.type === "post") accounts.add(c.author);
  const leadInput: LeadInput = {
    l,
    resultText,
    hasDeterministic,
    shownCards,
    usable,
    accounts: [...accounts],
    strictClaims: profile.strictClaims,
    verifyClaimsOn: bool(env.SEARCH_VERIFY_CLAIMS, true),
    task,
    finalStatus: final.status,
    status,
    conflictText,
    routeClass: route.class,
    hasPremise: program.steps.some((s) => s.op === "premise"),
    stripCitations: rich,
  };
  if (reply) {
    tp = Date.now();
    const lead = applyLead(leadInput, reply, answerText0);
    ({ status, answerText, explanation, rationale, modelUsed, cv, contradicted } = lead);
    notes.push(...lead.notes);
    mark("claims", tp);
  } else if (!hasDeterministic && final.status === "ok" && status === "answered") {
    status = "insufficient_evidence";
  }
  // "why …?": the index holds what happened, not why (applyLead decides it when a model replied)
  if (!reply && route.class === "EXPLANATORY" && status === "answered" && !program.steps.some((s) => s.op === "premise")) {
    status = "insufficient_evidence";
    answerText = `${say("insufficient", l)} ${resultText}`.trim();
  }

  // ---- confidence -------------------------------------------------------------------------------------
  const decidingRows = final.result.evidence.slice(0, 3);
  const retrieval = final.result.exact ? 1 : decidingRows.length ? decidingRows.reduce((s, x) => s + x.v.score, 0) / decidingRows.length : final.result.confidence;
  const truncated = outcomes.some((o) => o.result.truncated);
  const inferred = decidingRows.some((x) => x.row.history_exact === 0);
  const ties = outcomes.some((o) => Array.isArray((o.result.details as any)?.tied) && (o.result.details as any).tied.length > 1);
  const rerankScore = rerank && decidingRows.length ? decidingRows.map((x) => rerank!.get(x.row.id)).filter((x): x is number => typeof x === "number") : [];
  const confParts = {
    retrieval: Math.min(1, retrieval) * (truncated ? 0.85 : 1) * (inferred ? 0.9 : 1),
    reranking: rerankScore.length ? rerankScore.reduce((s, x) => s + x, 0) / rerankScore.length : null,
    agreement: Math.max(0, 1 - 0.4 * relevantConflicts.length - (ties ? 0.5 : 0) - 0.1 * Math.max(0, ev.conflicts.length - relevantConflicts.length)),
    // a question the rules decomposed into explicit operators is planned with certainty; v3's single-intent confidence otherwise
    planConfidence: program.pattern === "single" ? plan.confidence : Math.max(plan.confidence, program.source === "rules" ? 0.9 : 0.75),
  };
  const conf = combineConfidence(
    {
      ...confParts,
      // what is shown: the index's answer (exact), with an explanation only when it passed verification
      verification: hasDeterministic ? (modelUsed && cv ? 0.5 + 0.5 * cv.egs : 1) : modelUsed ? (cv ? (cv.answer.status === "supported" ? 0.5 + 0.5 * cv.egs : 0.5 * cv.egs) : 0.6) : 0.3,
      model: modelUsed ? (reply?.confidence ?? null) : null,
    },
    confidenceWeights(env),
  );
  let confidence = conf.value;
  // a deterministic answer is never surer than its operator (v3's confidence: ties, lower bounds,
  // inferred histories), whatever the model added to it
  let confidenceCap: number | null = null;
  if (hasDeterministic && final.result.confidence > 0) confidenceCap = final.result.confidence;
  if (status === "no_match") confidenceCap = final.result.confidence || confidence;
  if (confidenceCap !== null) confidence = r3(Math.min(confidence, confidenceCap));
  if (status === "insufficient_evidence" || status === "clarify" || status === "not_found") confidence = r3(Math.min(confidence, 0.3));

  // ---- the long-form answer (v4.8) -----------------------------------------------------------------------------
  tp = Date.now();
  const validArt = artCards.filter((c) => c.valid !== false);
  const hidden = (findings?.hidden ?? 0) + [...histories.values()].reduce((s, h) => s + (h.hidden ?? 0), 0);
  const noDigest = withheld || status === "clarify" || status === "not_found" || !rich;
  const digest: Digest = noDigest
    ? { facts: [], overview: [], caveats: [], follow_ups: [], searches: [], stats: { posts: 0, read: 0, authors: [], tags: [], colors: [], exact: true }, about: [] }
    : buildDigest({ lang, question, plan, program, outcomes, final, cards: validArt, results, histories, conflicts: relevantConflicts.map((c) => conflictSentence(c, l, pathOfCard)), hidden, rows: metaRows.length ? metaRows.map((x) => x.row) : undefined, extremes: metaExtremes });
  // nothing matched: questions about the subject lead nowhere; the searches may
  if (status === "no_match") digest.follow_ups = [];
  const composed = await composeAnswer({ env, lang, question, style, lead: answerText, digest, reply, contradicted, usable, accounts, knownAuthors: ctx.authors, strict: profile.strictClaims, target: words, notes });
  mark("compose", tp);
  // the model a deferred answer will run (its notes, e.g. a model not available to this caller, belong to this answer)
  const deferredModel = deferred ? pickModel(env, a, route, notes) : null;
  const elaboration: AskResponseV4["elaboration"] = deferred
    ? { status: "pending", url: elaborationUrl(queryId) }
    : reply
      ? { status: "inline" }
      : { status: "none", reason: !rich ? "brief style" : !explainable ? "nothing to explain" : profile.reasoning === "none" ? "fast mode: no model" : useModel ? "the model did not answer" : "not worth a model call" };

  // ---- response ------------------------------------------------------------------------------------------
  const fplan = final.plan ?? plan;
  const items: SearchItem[] = (final.result.searchItems as SearchItem[] | undefined) ?? dedupeRows([...(final.result.items ?? []), ...outcomes.flatMap((o) => (o === final ? [] : (o.result.items ?? o.result.evidence).slice(0, 3)))]).slice(0, limit).map((x) => rowToItem(x.row));
  const evidence: Array<Evidence & { evidence_id: string }> = final.result.evidence.map((x) => ({ ...evidenceOf(x.row, x.v), evidence_id: `E${x.row.id}` }));
  notes.push(...outcomes.flatMap((o) => o.result.notes));
  const usage = reply?.usage ? { input_tokens: reply.usage.inputTokens, output_tokens: reply.usage.outputTokens, ...(reply.usage.reasoningTokens !== undefined ? { reasoning_tokens: reply.usage.reasoningTokens } : {}), cost_usd: reply.costUsd, model_ms: reply.latencyMs, ...(reply.cached ? { cached: true } : {}) } : undefined;
  const versions = {
    retrieval: RETRIEVAL_VERSION,
    answer: ANSWER_VERSION,
    index: `${ctx.stats.posts}.${ctx.stats.artworks}.${ctx.stats.maxCreated}`,
    ranker: ctx.weights.version,
    reranker: rerankModel ? rerankerVersion(env) : null,
    planner: program.source === "llm" || plan.source !== "rules" ? modelFor(env, "planner") : "rules",
    reasoning_model: reply ? reply.model : null,
    prompt: reply ? PROMPT_VERSION : null,
  };
  if (deferred) {
    const frozen: FrozenElaboration = {
      v: 1,
      query_id: queryId,
      question,
      lang,
      mode: profile.mode,
      reasoning: profile.reasoning,
      maxOutputTokens: profile.maxOutputTokens,
      strictClaims: profile.strictClaims,
      model: deferredModel!,
      words: words ?? wordTarget(env, profile.mode, a.length),
      modelCards,
      context,
      lead: leadInput,
      answerText0,
      digest,
      text: a.text === "full" ? "full" : "short",
      notes: [...new Set(notes)],
      confidence: { parts: { retrieval: confParts.retrieval, reranking: confParts.reranking, agreement: confParts.agreement }, planConfidence: confParts.planConfidence, operator: hasDeterministic ? r3(final.result.confidence) : null, operatorConfidence: final.result.confidence, hasDeterministic },
      versions,
      cacheTtl: a.noCache ? 0 : int(env.SEARCH_ANSWER_CACHE_TTL, 86400),
      at: now(),
      attempts: 0,
    };
    // written before the answer goes out: the box fetches the elaboration right away
    await env.CACHE.put(elabKey(queryId), JSON.stringify(frozen), { expirationTtl: elaborationTtl(env) }).catch((e) => {
      notes.push(`the elaboration could not be deferred: ${e instanceof Error ? e.message : String(e)}`);
    });
  }
  timings.total = Date.now() - t0;
  if (reply) timings.model = reply.latencyMs;
  const res: AskResponseV4 = {
    question: original,
    status,
    answer: withheld ? null : (final.result.answer as string | number | null),
    answer_type: withheld ? "none" : final.result.answerType,
    answer_text: a.text === "full" ? composed.full : answerText,
    // a question asked back, or evidence that failed verification: no result, evidence or items to show
    ...(!withheld && (hasDeterministic || final.status === "no_match") ? { result_text: resultText } : {}),
    ...(explanation ? { explanation } : {}),
    confidence,
    confidence_parts: { ...conf.parts, ...(hasDeterministic ? { operator: r3(final.result.confidence) } : {}) },
    intent: fplan.intent,
    output: fplan.output,
    plan: compactPlan(plan),
    mode: profile.mode,
    reasoning: reply?.reasoning ?? (useModel ? profile.reasoning : "none"),
    model: reply ? reply.model : null,
    class: route.class,
    complexity: route.complexity,
    band: route.band,
    subqueries: subqueries(program),
    steps: withheld ? [] : outcomes.map((o) => ({ id: o.step.id, op: o.step.op === "ask" ? (o.plan?.intent ?? "ask") : o.step.op, status: o.status, answer: o.result.answer, answer_type: o.result.answerType, text: o.result.text, evidence: o.result.evidence.slice(0, 6).map((x) => `E${x.row.id}`), ms: o.ms })),
    claims: cv?.claims ?? [],
    ...(rationale ? { rationale } : {}),
    contradictions: ev.conflicts,
    // asked back: nothing was named, so no evidence; failed verification: the evidence (flagged), not the results computed from it
    cards: unnamed ? [] : withheld ? cards.filter((c) => c.type !== "result") : cards,
    ...(graph && !withheld ? { graph } : {}),
    ...(cv ? { grounding: { egs: cv.egs, citation_accuracy: cv.citation_accuracy, answer: cv.answer.status, counts: cv.counts } } : {}),
    evidence: withheld ? [] : evidence,
    ...(final.result.alternatives && !withheld ? { alternatives: final.result.alternatives.map((x) => evidenceOf(x.row, x.v)) } : {}),
    ...(final.result.counts && !withheld ? { counts: final.result.counts } : {}),
    items: withheld ? [] : items,
    ...(final.result.verified !== undefined && !withheld ? { verified: final.result.verified } : {}),
    versions,
    ...(usage ? { usage } : {}),
    timings,
    notes: [...new Set(notes)],
    query_id: queryId,
    ...(findings
      ? {
          image: {
            task: task ?? "origin",
            identity: imageCard?.identity ?? "none",
            first_seen: findings.origin ? { author: findings.origin.author, post: `/@${findings.origin.author}/${findings.origin.permlink}`, at: findings.origin.at, match: findings.origin.match, deleted: findings.origin.deleted } : null,
            matches: findings.matches.slice(0, 24).map((m) => ({ post_id: m.row.id as number, path: `/@${m.row.author}/${m.row.permlink}`, identity: m.identity })),
            hidden: findings.hidden,
          },
        }
      : {}),
    took_ms: Date.now() - t0,
    // v4.8
    style,
    answer_short: answerText,
    answer_full: composed.full,
    answer_markdown: composed.markdown,
    sections: composed.sections,
    thinking: composed.sections.thinking,
    suggestions: { follow_ups: composed.sections.follow_ups, searches: composed.sections.searches },
    digest: { stats: digest.stats, about: digest.about },
    length: { words: composed.words, target: useModel && rich ? words : null },
    elaboration,
  };
  const wantTrace = a.trace && (a.admin || bool(env.SEARCH_TRACE, false));
  const trace = {
    route: { class: route.class, complexity: route.complexity, band: route.band, signals: route.signals, reason: route.reason },
    mode: { mode: profile.mode, explicit: chosen.explicit, needs_synthesis: needsSynthesis, style, profile },
    program: { pattern: program.pattern, source: program.source, steps: program.steps.map((s) => ({ id: s.id, op: s.op, text: s.text, refs: s.refs })) },
    steps: outcomes.map((o) => ({ id: o.step.id, status: o.status, legs: o.scope?.legs ?? {}, candidates: o.scope?.all.length ?? 0, verified: o.scope?.verified.length ?? 0, ms: o.ms, top: o.result.evidence.slice(0, 5).map((x) => ({ id: x.row.id, score: x.v.score, signals: x.v.signals })) })),
    rerank: rerank ? { model: rerankModel, scores: [...rerank.entries()].slice(0, 20).map(([id, s]) => ({ id, score: r3(s) })) } : null,
    evidence: { cards: cards.length, invalid: Object.fromEntries(ev.invalid), conflicts: ev.conflicts.length, duplicates: ev.duplicates },
    reasoning: reply ? { model: reply.model, status: reply.status, usage: reply.usage, finish: reply.finishReason, cached: !!reply.cached, salvaged: !!reply.salvaged, notes: reply.notes } : null,
    claims: cv ? { counts: cv.counts, egs: cv.egs, answer: cv.answer } : null,
    compose: { words: composed.words, target: words, body: composed.body ? { removed: composed.body.removed, egs: composed.body.egs, citation_accuracy: composed.body.citation_accuracy, sentences: composed.body.claims.map((c) => ({ text: c.text.slice(0, 120), status: c.status, problems: c.problems })) } : null, removed: composed.removed, deferred },
    image: findings ? { task, legs: findings.legs, origin: findings.origin, appearances: findings.appearances.length, hidden: findings.hidden, described: !!imageCard?.description } : null,
    timings,
    // what the model was given and what was kept of its reply: a fine-tuning example once verified (GET /admin/ask/export-sft)
    ...(reply ? { model_input: { cards: modelCards, context, lang }, model_output: { used: modelUsed, status: reply.status, answer: modelUsed ? (explanation ?? answerText) : null, claims: (cv ? keptClaims(cv, profile.strictClaims) : reply.claims).map((c) => ({ text: c.text, evidence: c.evidence, kind: c.kind })), rationale: rationale ?? null } } : {}),
  };
  if (wantTrace) res.trace = trace;
  const after: Promise<unknown>[] = [logAsk(env, res, trace).catch(() => {})];
  // the box's dropdown continues the conversation: GET /suggest?after=<query_id>
  if (rich && (res.suggestions.follow_ups.length || res.suggestions.searches.length)) after.push(keepSuggestions(env, queryId, res.suggestions));
  const settled = Promise.all(after);
  if (exec) exec.waitUntil(settled);
  else await settled;
  return res;

  function contextResponse(): AskResponseV4 {
    timings.total = Date.now() - t0;
    const lead = resultText;
    return {
      question: original,
      status,
      answer: final.result.answer as string | number | null,
      answer_type: final.result.answerType,
      answer_text: resultText,
      result_text: resultText,
      confidence: final.result.confidence,
      confidence_parts: {},
      intent: (final.plan ?? plan).intent,
      output: (final.plan ?? plan).output,
      plan: compactPlan(plan),
      mode: profile.mode,
      reasoning: profile.reasoning,
      model: null,
      class: route.class,
      complexity: route.complexity,
      band: route.band,
      subqueries: subqueries(program),
      steps: outcomes.map((o) => ({ id: o.step.id, op: o.step.op, status: o.status, answer: o.result.answer, answer_type: o.result.answerType, text: o.result.text, evidence: o.result.evidence.slice(0, 6).map((x) => `E${x.row.id}`), ms: o.ms })),
      claims: [],
      contradictions: ev.conflicts,
      cards,
      evidence: final.result.evidence.map((x) => ({ ...evidenceOf(x.row, x.v), evidence_id: `E${x.row.id}` })),
      items: [],
      versions: { retrieval: RETRIEVAL_VERSION, answer: ANSWER_VERSION, index: `${ctx.stats.posts}.${ctx.stats.artworks}.${ctx.stats.maxCreated}`, prompt: PROMPT_VERSION },
      timings,
      notes,
      query_id: queryId,
      took_ms: Date.now() - t0,
      trace: { context, model_cards: modelCards, lang, profile, has_deterministic: hasDeterministic, shown: shownCards.map((c) => c.evidence_id), style, words: rich ? wordTarget(env, profile.mode, a.length) : null },
      style,
      answer_short: lead,
      answer_full: lead,
      answer_markdown: lead,
      sections: { lead, facts: [], overview: [], thinking: [], caveats: [], follow_ups: [], searches: [] },
      thinking: [],
      suggestions: { follow_ups: [], searches: [] },
      digest: { stats: { posts: 0, read: 0, authors: [], tags: [], colors: [], exact: true }, about: [] },
      length: { words: 0, target: null },
      elaboration: { status: "none", reason: "context only" },
    };
  }
}

function dedupeRows(xs: Verified[]): Verified[] {
  const seen = new Set<number>();
  return xs.filter((x) => (seen.has(x.row.id) ? false : (seen.add(x.row.id), true)));
}

/** A trace as JSON within `max` characters: the heaviest parts are dropped first, never cut mid-way. */
export function fitTrace(trace: Record<string, unknown>, max = 32_000): string {
  let t = { ...trace };
  let json = JSON.stringify(t);
  for (const k of ["model_input", "model_output", "compose", "rerank", "steps", "evidence", "program"]) {
    if (json.length <= max) break;
    t = { ...t, [k]: "(dropped: trace too large)" };
    json = JSON.stringify(t);
  }
  return json.length <= max ? json : JSON.stringify({ route: trace.route, mode: trace.mode, dropped: "trace too large" });
}

/** One row per question in ask_log (spec §52); the trace for a sample (SEARCH_TRACE_SAMPLE). */
async function logAsk(env: Env, r: AskResponseV4, trace: Record<string, unknown>): Promise<void> {
  const keep = Math.random() < Math.min(1, Math.max(0, num(env.SEARCH_TRACE_SAMPLE, 0.1)));
  const json = keep ? fitTrace(trace) : null;
  await env.DB.prepare(
    `INSERT INTO ask_log (qid, q, lang, class, complexity, mode, reasoning, model, status, answer, confidence, egs, claims, claims_supported, input_tokens, output_tokens, cost_usd, model_ms, took_ms, versions, trace, at)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
  )
    .bind(
      r.query_id,
      r.question,
      r.plan.lang ?? null,
      r.class,
      r.complexity,
      r.mode,
      r.reasoning,
      r.model,
      r.status,
      r.answer === null || r.answer === undefined ? null : String(r.answer).slice(0, 200),
      r.confidence,
      r.grounding?.egs ?? null,
      r.claims.length,
      r.claims.filter((c) => c.status === "supported").length,
      r.usage?.input_tokens ?? null,
      r.usage?.output_tokens ?? null,
      r.usage?.cost_usd ?? null,
      r.usage?.model_ms ?? null,
      r.took_ms,
      JSON.stringify(r.versions),
      json,
      now(),
    )
    .run();
}

export type { SearchContext, QueryPlan };
