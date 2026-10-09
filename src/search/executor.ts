// Runs a question's program (query-planner.ts) step by step through the deterministic operators,
// passing earlier answers on (an author, a post, a time), and composes the deterministic answer.
// Nothing here calls a model.

import type { Env } from "../env";
import { buildFilter } from "./sql";
import type { SearchContext } from "./context";
import type { NsfwMode } from "./params";
import type { QueryPlan } from "./planner";
import { parseList, requestFromPlan, resolveScope, type RetrievalDepth, type SubjectRetrieval } from "./retrieval";
import {
  aggregateOp,
  authorOfOp,
  compareCounts,
  compareMetric,
  compareTotals,
  durationOp,
  groupMetadata,
  groupResult,
  historyFacts,
  historyOp,
  identifyOp,
  resolveTitle,
  runIntent,
  sequenceOp,
  similarOp,
  type HistoryFacts,
  type OpContext,
  type OperatorResult,
  type Row,
  type Scope,
  type TimedEvent,
  type Verified,
} from "./operators";
import { answerLang, capitalize, firstLabel, fmtDate, quote, say, subjectLabel, subjectPlural, yesNo, type AnswerLang } from "./answer-text";
import type { QueryProgram, Step, StepRef } from "./query-planner";
import { afterColon } from "./operators";
import { editDistance, fold, isStopword, type Lang } from "../lib/text";
import { suggest as suggestSpelling } from "./spell";
import { planQuery } from "./planner";
import { hydrateRows } from "./service";
import { emptyRequest } from "./params";

export type StepStatus = "ok" | "no_match" | "not_found" | "ambiguous" | "failed";

export interface StepOutcome {
  step: Step;
  status: StepStatus;
  result: OperatorResult & { searchItems?: unknown[] };
  /** the plan it ran, with the earlier answers injected */
  plan?: QueryPlan;
  scope?: Scope & { history?: SubjectRetrieval["history"] };
  /** what later steps may take from it */
  values: { author?: string; post?: string; time?: number; title?: string; row?: Row; n?: number };
  history?: HistoryFacts | null;
  ms: number;
}

export interface ExecContext {
  env: Env;
  ctx: SearchContext;
  lang: Lang;
  limit: number;
  nsfw: NsfwMode;
  type?: "artwork" | "blog";
  threshold: number;
  depth: RetrievalDepth;
  notes: string[];
}

const path = (r: Row) => `/@${r.author}/${r.permlink}`;
const PATH = /^\/@([a-z0-9][a-z0-9.-]{1,31})\/(\S+)$/;
/** "When was … deleted?": the history step answers with a date */
const WHEN = /^\s*(?:when|what date|on what day|quand|a quelle date|wann|an welchem tag|cuando|en que fecha|quando|in che data)\b/;

function oc(e: ExecContext, plan: QueryPlan | null): OpContext {
  return { env: e.env, ctx: e.ctx, lang: e.lang, v3: false, limit: e.limit, subject: plan ? subjectLabel(plan, e.lang) : "" };
}

function valuesOf(r: OperatorResult): StepOutcome["values"] {
  const top = r.evidence[0]?.row;
  const d = (r.details ?? {}) as Record<string, any>;
  const author = r.answerType === "author" && typeof r.answer === "string" ? r.answer : typeof d.first_author === "string" ? d.first_author : top?.author;
  const post = typeof d.first_post === "string" && r.op === "find_first" ? d.post : typeof d.post === "string" ? d.post : r.answerType === "post" && typeof r.answer === "string" ? r.answer : top ? path(top) : undefined;
  const time = typeof d.time === "number" ? d.time : top ? top.created : undefined;
  return { author, post, time, title: typeof d.title === "string" ? d.title : top?.title, row: top, n: typeof r.answer === "number" ? r.answer : undefined };
}

function failed(step: Step, text: string, status: StepStatus = "failed"): StepOutcome {
  return { step, status, result: { op: "resolve", answer: null, answerType: "none", text, confidence: 0, evidence: [], notes: [] }, values: {}, ms: 0 };
}

function withInjected(p: QueryPlan, step: Step, done: Map<string, StepOutcome>): QueryPlan | null {
  const filters = { ...p.filters };
  const inj = step.inject ?? {};
  if (inj.authors) {
    const a = done.get(inj.authors.step)?.values.author;
    if (!a) return null;
    filters.authors = [a];
  }
  if (inj.from) {
    const t = done.get(inj.from.step)?.values.time;
    if (t === undefined) return null;
    filters.from = t + 1;
  }
  if (inj.to) {
    const t = done.get(inj.to.step)?.values.time;
    if (t === undefined) return null;
    filters.to = t;
  }
  return { ...p, filters };
}

/** How an event is named in a sentence: “Title”, or "the first cat artwork". */
function eventLabel(o: StepOutcome, l: AnswerLang): string {
  if (o.step.op === "resolve_title" || o.step.op === "author_of") return quote(o.values.title ?? o.step.title ?? "", l);
  if (o.plan && (o.plan.intent === "find_first" || o.plan.intent === "find_last")) return firstLabel(subjectLabel(o.plan, l), o.plan.intent === "find_first", l);
  return quote(o.values.title ?? "", l);
}

/** "@laura's artworks", « les œuvres de @laura », "cat artworks": what a total is over. */
export function scopeLabel(plan: QueryPlan, lang: Lang): string {
  const l = answerLang(lang);
  const a = plan.filters.authors?.[0];
  const s = plan.residual || plan.concepts.length ? subjectPlural(plan, l) : "";
  if (a) {
    const own: Record<AnswerLang, string> = {
      en: s ? `@${a}'s ${s}` : `@${a}'s artworks`,
      fr: s ? `Les ${s} de @${a}` : `Les œuvres de @${a}`,
      de: s ? `Die ${s} von @${a}` : `Die Kunstwerke von @${a}`,
      es: s ? `Las ${s} de @${a}` : `Las obras de @${a}`,
      it: s ? `I ${s} di @${a}` : `Le opere di @${a}`,
    };
    return own[l];
  }
  const all: Record<AnswerLang, string> = { en: "All artworks", fr: "Toutes les œuvres", de: "Alle Kunstwerke", es: "Todas las obras", it: "Tutte le opere" };
  return s ? capitalize(s) : all[l];
}

async function aggregateMetadata(e: ExecContext, scope: Scope, metric: "net_votes" | "payout", agg: "sum" | "avg" | "max" | "min", label: string): Promise<OperatorResult> {
  const f = buildFilter(scope.req, { tones: scope.plan.filters.tones });
  const col = metric === "payout" ? "p.payout" : "p.net_votes";
  const fn = agg === "sum" ? "SUM" : agg === "avg" ? "AVG" : agg === "max" ? "MAX" : "MIN";
  const r = await e.env.DB.prepare(`SELECT ${fn}(${col}) AS v, COUNT(*) AS n FROM posts p LEFT JOIN artworks a ON a.post_id = p.id WHERE ${f.sql}`).bind(...f.params).first<{ v: number | null; n: number }>();
  return aggregateOp([], metric, agg, oc(e, scope.plan), label, { value: Number(r?.v ?? 0), n: Number(r?.n ?? 0) });
}

/** What a search names, without its phrasing: "Show me artworks of Miracle dream town" → "miracle dream town". */
export function titleCandidate(question: string): string | null {
  const ART = String.raw`(?:artworks?|arts?|images?|pictures?|posts?|drawings?|paintings?|oeuvres?|œuvres?|dessins?|tableaux|bilder|kunstwerke?|obras?|imagenes|dibujos|opere|immagini|disegni)`;
  let t = fold(question).replace(/[?!.,;:¿¡"“”«»„]+/g, " ").replace(/['’]/g, " ").replace(/\s+/g, " ").trim();
  if (t.split(" ").length > 10) return null;
  t = t.replace(/^(?:show me|show|find me|find|search for|search|look for|looking for|i m looking for|i am looking for|im looking for|i want|give me|are there any|are there|is there an?|is there|do you have|any|montre moi|montre|trouve moi|trouve|je cherche|cherche|y a t il des|zeig mir|zeige mir|finde|ich suche|suche|gibt es|muestrame|busca|busco|hay|mostrami|trova|cerco|ci sono)\s+/, "");
  t = t.replace(new RegExp(String.raw`^(?:the |some |any |all |des |les |la |le |l |d |die |der |das |los |las |el |gli |i |il )?${ART}\s+(?:of|with|about|called|titled|named|showing|de|du|des|d|avec|appele\w*|intitule\w*|von|mit|uber|namens|con|del|de la|di|della|chiamat\w*)\s+`), "");
  t = t.replace(new RegExp(String.raw`\s+${ART}$`), "").trim();
  return t.length >= 3 ? t : null;
}

/**
 * Posts titled exactly as the search names them, first, marked as such (spec §9: an exact metadata
 * match is a candidate source) — only those the search's own filters keep: "@bob swan artworks" is
 * not @alice's “Swan”, a search for artworks is not a blog post titled so.
 */
async function withExactTitle(e: ExecContext, question: string, p: QueryPlan, r: OperatorResult, c: OpContext, l: AnswerLang): Promise<OperatorResult> {
  const t = titleCandidate(question);
  if (!t) return r;
  const m = await resolveTitle(e.env, t, e.nsfw).catch(() => null);
  if (!m?.rows.length) return r;
  // the filters a title cannot be part of (colour words in a title are the title's, not a filter)
  const req = requestFromPlan(p, { type: e.type, nsfw: e.nsfw });
  const kept = await hydrateRows(e.env.DB, m.rows.map((x) => x.id), emptyRequest({ type: req.type, authors: req.authors, tags: req.tags, from: req.from, to: req.to, nsfw: e.nsfw })).catch(() => new Map<number, Row>());
  const rows = m.rows.filter((x) => kept.has(x.id));
  if (!rows.length) return r;
  const named: Verified[] = rows.slice(0, 4).map((row) => ({ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["the title names this post"] } }));
  const ids = new Set(named.map((x) => x.row.id));
  const before = new Set((r.evidence ?? []).map((x) => x.row.id));
  const evidence = [...named, ...(r.evidence ?? []).filter((x) => !ids.has(x.row.id))].slice(0, c.limit);
  const items = [...named, ...(r.items ?? r.evidence ?? []).filter((x) => !ids.has(x.row.id))].slice(0, c.limit);
  const top = named[0].row;
  const added = named.filter((x) => !before.has(x.row.id)).length;
  const n = (typeof r.answer === "number" ? r.answer : 0) + added;
  return {
    ...r,
    op: "search",
    answer: n,
    answerType: "count",
    text: say("post", l, { title: top.title, author: top.author, date: fmtDate(top.created) }),
    confidence: Math.max(r.confidence, 0.9),
    evidence,
    items,
    verified: (r.verified ?? 0) + added,
    details: { ...(r.details ?? {}), n, title_match: named.map((x) => path(x.row)) },
    notes: [...r.notes, `${named.length} post(s) titled “${top.title}” first`],
  };
}

/**
 * Run every step in order. A step that cannot run (a title not found, an earlier step with no
 * answer) stops the program; a title no post has, in a question that also reads as a subject
 * ("Who posted the first “sunset” artwork?"), runs as that subject (`orElse`).
 */
export async function executeProgram(e: ExecContext, program: QueryProgram): Promise<StepOutcome[]> {
  const out = await runSteps(e, program);
  if (program.orElse && out[0]?.status === "not_found") return runSteps(e, program.orElse);
  return out;
}

async function runSteps(e: ExecContext, program: QueryProgram): Promise<StepOutcome[]> {
  const done = new Map<string, StepOutcome>();
  const out: StepOutcome[] = [];
  const l = answerLang(e.lang);
  for (const step of program.steps) {
    const t0 = Date.now();
    let o: StepOutcome;
    const ref = (r: StepRef | undefined) => (r ? done.get(r.step) : undefined);
    try {
      switch (step.op) {
        case "ask":
        case "exists":
        case "premise":
        case "aggregate":
        case "group": {
          let p = step.plan ? withInjected(step.plan, step, done) : null;
          if (!p) {
            o = failed(step, say("insufficient", l));
            break;
          }
          if (step.op === "premise") p = { ...p, filters: { ...p.filters, authors: undefined }, intent: "find_first", output: "author" };
          if (p.intent === "similar" || p.intent === "duplicate") {
            const r = await similarOp(e.env, p.similarTo?.id, p.intent === "duplicate", oc(e, p), e.nsfw);
            o = { step, status: r.answer === null ? "failed" : "ok", result: r, plan: p, values: { n: typeof r.answer === "number" ? r.answer : undefined }, ms: 0 };
            break;
          }
          const req = requestFromPlan(p, { type: e.type, nsfw: e.nsfw });
          let scope = await resolveScope(e.env, p, req, e.ctx, { threshold: e.threshold, depth: e.depth, notes: e.notes });
          // nothing for a subject the corpus does not spell that way ("dargons", "galantus"): once more with the
          // corpus's spelling, one edit away (a different word two edits away, "wagon" → "dragon", is not read),
          // and said in the answer. Only the subject changes: the intent, the filters and what earlier steps gave stay.
          let respelled: Array<{ from: string; to: string }> = [];
          if (step.op === "ask" && scope.kind === "subject" && !scope.verified.length && p.residual) {
            // a correction to a function word ("teen" → "then") is no reading of the subject
            const oneEdit = (f: { from: string; to: string; distance: number }) => f.distance <= 1 || (f.from.endsWith("s") && editDistance(f.from.slice(0, -1), f.to, 1) <= 1);
            const fixes = (await suggestSpelling(e.env.DB, p.residual).catch(() => [])).filter((f) => oneEdit(f) && f.from.length >= 5 && !isStopword(f.to) && f.to.length >= 4);
            if (fixes.length) {
              const fixed = fixes.reduce((t, f) => t.replace(new RegExp(`(?<![\\p{L}])${f.from}(?![\\p{L}])`, "giu"), f.to), fold(p.residual));
              const re = planQuery(fixed, { authors: e.ctx.authors, mode: "ask", now: e.ctx.now, v4: true });
              const p2: QueryPlan = {
                ...p,
                residual: re.residual,
                lexicalTerms: re.lexicalTerms,
                concepts: re.concepts,
                conceptMatches: re.conceptMatches,
                filters: { ...p.filters, colors: p.filters.colors ?? re.filters.colors, tones: p.filters.tones ?? re.filters.tones },
              };
              const s2 = await resolveScope(e.env, p2, requestFromPlan(p2, { type: e.type, nsfw: e.nsfw }), e.ctx, { threshold: e.threshold, depth: e.depth, notes: e.notes });
              if (s2.verified.length) {
                e.notes.push(`spelling: ${fixes.map((f) => `“${f.from}” read as “${f.to}”`).join(", ")}`);
                respelled = fixes;
                p = p2;
                scope = s2;
              }
            }
          }
          const c = oc(e, p);
          let r: OperatorResult;
          if (step.op === "aggregate") {
            const label = scopeLabel(p, e.lang);
            r = scope.kind === "metadata" ? await aggregateMetadata(e, scope, step.metric ?? "net_votes", step.agg ?? "sum", label) : aggregateOp(scope.verified, step.metric ?? "net_votes", step.agg ?? "sum", c, label);
          } else if (step.op === "group") {
            const by = step.by ?? "month";
            const label = scopeLabel(p, e.lang);
            // "who has the most votes?": per author, the total of the metric over their artworks
            const metric = by === "author" ? step.metric : undefined;
            if (scope.kind === "metadata") r = await groupMetadata(scope, by, c, label, metric);
            else {
              const groups = new Map<string, number>();
              for (const x of scope.verified) {
                const keys = by === "author" ? [x.row.author] : by === "tag" ? parseList(x.row.tags_json) : [new Date(x.row.created * 1000).toISOString().slice(0, 7)];
                for (const k of new Set(keys)) groups.set(k, (groups.get(k) ?? 0) + (metric ? Number(x.row[metric] ?? 0) : 1));
              }
              const ranked = [...groups.entries()].map(([k, n]) => ({ k, n })).sort((a, b) => b.n - a.n || a.k.localeCompare(b.k));
              r = { ...groupResult(ranked, by, c, label, scope.verified.length, false, metric), evidence: scope.verified.slice(0, e.limit), truncated: scope.truncated };
              if (r.confidence > 0) r.confidence = Math.min(r.confidence, 0.9);
            }
          } else {
            r = await runIntent(scope, c);
            // "who posted the most liked artwork?": its author
            if (program.pattern === "top_author" && p.intent === "top" && typeof r.details?.author === "string") r = { ...r, answer: r.details.author, answerType: "author" };
            // a post whose title is exactly what a search names comes first ("Blue Church artworks")
            if (step.op === "ask" && p.intent === "search" && program.pattern === "single" && !respelled.length) r = await withExactTitle(e, step.text ?? "", p, r, c, l);
            // a respelled subject is said, and the answer is no surer than the reading
            if (respelled.length) r = { ...r, text: `${respelled.map((f) => say("spelling_read", l, f)).join(" ")} ${r.text}`, confidence: Math.min(r.confidence, 0.75) };
            if (step.op === "exists") {
              const n = typeof r.answer === "number" ? r.answer : 0;
              const author = p.filters.authors?.[0] ?? step.claimed ?? "";
              r = { ...r, answer: n > 0, answerType: "boolean", text: yesNo(n > 0, l) + afterColon(n > 0 ? say("exists_yes", l, { author, n, s: n === 1 ? subjectLabel(p, l) : subjectPlural(p, l) }) : say("exists_no", l, { author, s: subjectLabel(p, l) })), details: { ...(r.details ?? {}), n } };
            } else if (step.op === "premise" && r.answer !== null) {
              const v = valuesOf(r);
              const claimed = step.claimed ?? "";
              if (v.author && v.author !== claimed) r = { ...r, text: say("premise_false", l, { claimed, actual: v.author, s: subjectLabel(p, l), date: fmtDate(v.time ?? 0), title: v.title ?? "" }), details: { ...(r.details ?? {}), premise: false, claimed } };
              else r = { ...r, text: `${r.text} ${say("insufficient", l)}`, details: { ...(r.details ?? {}), premise: true, claimed } };
            }
          }
          const status: StepStatus = r.answer === null || (r.answer === 0 && r.answerType === "count" && step.op === "ask") ? "no_match" : "ok";
          o = { step, status, result: r, plan: p, scope, values: valuesOf(r), ms: 0 };
          break;
        }
        case "resolve_title":
        case "author_of":
        case "identify": {
          // a post a history step asks about may have been deleted since
          const forHistory = program.steps.some((x) => x.op === "history" && x.target?.step === step.id);
          const m = await resolveTitle(e.env, step.title ?? "", e.nsfw, { includeDeleted: forHistory });
          if (m.status === "not_found") {
            o = failed(step, say("not_found_title", l, { title: step.title ?? "" }), "not_found");
            break;
          }
          if (m.status === "ambiguous") {
            const list = m.rows.slice(0, 4).map((r) => `${quote(r.title, l)} @${r.author} (${fmtDate(r.created)})`).join(", ");
            o = { ...failed(step, say("clarify_title", l, { title: step.title ?? "", list }), "ambiguous"), result: { op: "resolve", answer: null, answerType: "none", text: say("clarify_title", l, { title: step.title ?? "", list }), confidence: 0.3, evidence: m.rows.slice(0, 4).map((row) => ({ row, v: { score: 1, lexical: 1, semantic: null, text: null, signals: ["has this title"] } })), notes: [], details: { candidates: m.rows.slice(0, 6).map((r) => path(r)) } } };
            break;
          }
          const row = m.rows[0];
          let r = step.op === "identify" ? identifyOp(row, oc(e, null)) : authorOfOp(row, oc(e, null));
          // "when was “T” posted?": the date, said with who posted it
          if (step.op === "author_of" && step.field === "time") r = { ...r, answer: fmtDate(row.created), answerType: "date", details: { ...(r.details ?? {}), time: row.created } };
          // "was “T” posted by @x?": yes or no, and who did post it
          if (step.op === "author_of" && step.claimed) {
            const claimed = step.claimed.toLowerCase();
            const yes = row.author === claimed;
            r = { ...r, answer: yes, answerType: "boolean", text: yesNo(yes, l) + afterColon(yes ? r.text : say("author_not", l, { title: row.title, author: row.author, date: fmtDate(row.created), claimed })), details: { ...(r.details ?? {}), claimed, actual: row.author } };
          }
          o = { step, status: "ok", result: r, values: { author: row.author, post: path(row), time: row.created, title: row.title, row }, ms: 0 };
          break;
        }
        case "fact": {
          const src = ref(step.target);
          if (!src?.values.post) {
            o = failed(step, say("insufficient", l));
            break;
          }
          const v = src.values;
          const r: OperatorResult =
            step.field === "time"
              ? { op: "resolve", answer: v.time ? fmtDate(v.time) : null, answerType: "date", text: v.time ? `${capitalize(fmtDate(v.time))}.` : say("insufficient", l), confidence: src.result.confidence, evidence: src.result.evidence.slice(0, 1), notes: [], details: { time: v.time, post: v.post } }
              : { op: "resolve", answer: v.author ?? null, answerType: "author", text: v.author ? `@${v.author}.` : say("insufficient", l), confidence: src.result.confidence, evidence: src.result.evidence.slice(0, 1), notes: [], details: { author: v.author, post: v.post } };
          o = { step, status: r.answer === null ? "failed" : "ok", result: r, values: v, ms: 0 };
          break;
        }
        case "history": {
          const src = ref(step.target);
          const m = src?.values.post ? PATH.exec(src.values.post) : null;
          if (!m) {
            o = failed(step, say("insufficient", l));
            break;
          }
          // the post whose image's history is asked: for "the first X", where the image first appeared
          const h = await historyFacts(e.env, { author: m[1], permlink: m[2] }, { nsfw: e.nsfw });
          if (!h) {
            o = failed(step, say("insufficient", l));
            break;
          }
          // "did they post it again": the earlier step's author; "reposted by @carol": the one named
          const by = step.actor ?? (step.byAuthor ? ref(step.byAuthor)?.values.author : undefined);
          // "when was it deleted?", and "was it edited, and when?"
          const asked = fold(step.text ?? "").replace(/[?!.,;:¿¡]+/g, " ").trim();
          const r = historyOp(h, step.relation ?? "all", oc(e, null), { byAuthor: by, when: WHEN.test(asked) || /\b(?:and|et|und|y|e) (?:when|quand|wann|cuando|quando)$/.test(asked) });
          o = { step, status: "ok", result: { ...r, evidence: src!.result.evidence.slice(0, 1) }, values: { ...src!.values }, history: h, ms: 0 };
          break;
        }
        case "compare_counts": {
          const a = done.get(step.a ?? "");
          const b = done.get(step.b ?? "");
          if (!a || !b || typeof a.result.answer !== "number" || typeof b.result.answer !== "number") {
            o = failed(step, say("insufficient", l));
            break;
          }
          const subject = a.plan ? subjectPlural(a.plan, l) : "";
          const authorA = a.plan?.filters.authors?.[0] ?? "";
          const authorB = b.plan?.filters.authors?.[0] ?? "";
          const evidence = [...a.result.evidence.slice(0, 3), ...b.result.evidence.slice(0, 3)];
          if (step.metric) {
            // "does @a have more votes than @b?": the totals of their artworks
            const r = compareTotals({ author: authorA, v: a.result.answer }, { author: authorB, v: b.result.answer }, step.metric, oc(e, null), { want: step.want === "less" ? "less" : "more", yesNo: step.yesNo, claimed: step.claimed });
            o = { step, status: "ok", result: { ...r, evidence }, values: {}, ms: 0 };
            break;
          }
          if (a.result.answer === 0 && b.result.answer === 0) {
            // neither has any: nothing to compare (and a subject read wrong would otherwise be "the same number, 0 each")
            o = { step, status: "no_match", result: { op: "compare", answer: null, answerType: "none", text: say("neither_count", l, { a: authorA, b: authorB, s: subject }), confidence: Math.min(a.result.confidence, b.result.confidence, 0.9), evidence: [], notes: [], details: { a: { author: authorA, n: 0 }, b: { author: authorB, n: 0 } } }, values: {}, ms: 0 };
            break;
          }
          const r = compareCounts({ author: authorA, n: a.result.answer }, { author: authorB, n: b.result.answer }, oc(e, a.plan ?? null), { subject, want: step.want === "less" ? "less" : "more", yesNo: step.yesNo, claimed: step.claimed });
          r.truncated = !!(a.result.truncated || b.result.truncated);
          o = { step, status: "ok", result: { ...r, evidence }, values: {}, ms: 0 };
          break;
        }
        case "compare_metric": {
          const a = done.get(step.a ?? "");
          const b = done.get(step.b ?? "");
          if (!a?.values.row || !b?.values.row) {
            o = failed(step, say("insufficient", l));
            break;
          }
          const metric = step.metric ?? "net_votes";
          const r = compareMetric({ label: quote(a.values.row.title, l), value: Number(a.values.row[metric] ?? 0), post: path(a.values.row) }, { label: quote(b.values.row.title, l), value: Number(b.values.row[metric] ?? 0), post: path(b.values.row) }, metric, oc(e, null), { yesNo: step.yesNo });
          o = { step, status: "ok", result: { ...r, evidence: [...a.result.evidence.slice(0, 1), ...b.result.evidence.slice(0, 1)] }, values: {}, ms: 0 };
          break;
        }
        case "sequence":
        case "duration": {
          const a = done.get(step.a ?? "");
          const b = done.get(step.b ?? "");
          if (a?.values.time === undefined || b?.values.time === undefined) {
            o = failed(step, say("insufficient", l));
            break;
          }
          const ea: TimedEvent = { label: eventLabel(a, l), at: a.values.time, post: a.values.post };
          const eb: TimedEvent = { label: eventLabel(b, l), at: b.values.time, post: b.values.post };
          const r = step.op === "sequence" ? sequenceOp(ea, eb, oc(e, null), { yesNo: step.yesNo ? (step.want === "after" ? "after" : "before") : undefined }) : durationOp(ea, eb, oc(e, null), { unit: step.unit });
          o = { step, status: "ok", result: { ...r, evidence: [...a.result.evidence.slice(0, 1), ...b.result.evidence.slice(0, 1)] }, values: {}, ms: 0 };
          break;
        }
        default:
          o = failed(step, say("insufficient", l));
      }
    } catch (err) {
      e.notes.push(`step ${step.id} (${step.op}) failed: ${err instanceof Error ? err.message : String(err)}`);
      o = failed(step, say("insufficient", l));
    }
    o.ms = Date.now() - t0;
    out.push(o);
    done.set(step.id, o);
    if (o.status === "failed" || o.status === "not_found" || o.status === "ambiguous") break;
  }
  return out;
}

/** The deterministic answer of a program: the final step's sentence, after the steps it builds on when they say something of their own. */
/** The steps whose results the deterministic answer says, in order (the others only lead to them). */
export function shownOutcomes(program: QueryProgram, outcomes: StepOutcome[]): StepOutcome[] {
  const last = outcomes[outcomes.length - 1];
  switch (program.pattern) {
    case "conjunction":
    case "nested_author":
    case "relative_time":
      return outcomes;
    case "planner_model":
      return outcomes.filter((o) => o.step.op !== "resolve_title" && (o === last || o.step.op === "ask"));
    default:
      return [last];
  }
}

export function composeDeterministic(program: QueryProgram, outcomes: StepOutcome[]): { text: string; final: StepOutcome; complete: boolean; shown: StepOutcome[] } {
  const last = outcomes[outcomes.length - 1];
  const complete = outcomes.length === program.steps.length && last.status !== "failed" && last.status !== "not_found" && last.status !== "ambiguous";
  if (!complete) return { text: last.result.text, final: last, complete, shown: [last] };
  const shown = shownOutcomes(program, outcomes);
  return { text: shown.map((o) => o.result.text).join(" ").replace(/\s+/g, " ").trim(), final: last, complete, shown };
}
