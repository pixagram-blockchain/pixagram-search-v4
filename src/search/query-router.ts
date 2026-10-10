// The v4 query router (spec §6-8, §20): what kind of question this is, how complex, and so how
// much computation it gets. Rules only, no model call.
//
//   class       EXACT FACTUAL SEMANTIC VISUAL TEMPORAL COMPARATIVE AGGREGATION MULTI_HOP
//               EXPLANATORY AMBIGUOUS UNKNOWN
//   complexity  [0, 1] from the question's length, entities, predicates, temporal and comparative
//               expressions, aggregation, ambiguity, and the retrieval operations and relations
//               its decomposition needs (query-planner.ts)
//   band        trivial < 0.2 ≤ simple < 0.4 ≤ normal < 0.65 ≤ complex < 0.85 ≤ deep
//   mode        fast (trivial, simple) · balanced (normal) · deep (complex) · expert (deep)
//
// The search box's own router (router.ts: search, ask or help) runs before this one; this one only
// sees questions about the artworks.

import type { Env } from "../env";
import { bool } from "../env";
import { fold, isStopword } from "../lib/text";
import type { ReasoningLevel } from "../llm/provider";
import type { ComplexityBand } from "../llm/router";
import type { QueryPlan } from "./planner";
import type { QueryProgram } from "./query-planner";

export type QueryClass = "EXACT" | "FACTUAL" | "SEMANTIC" | "VISUAL" | "TEMPORAL" | "COMPARATIVE" | "AGGREGATION" | "MULTI_HOP" | "EXPLANATORY" | "AMBIGUOUS" | "UNKNOWN";
export const QUERY_CLASSES: QueryClass[] = ["EXACT", "FACTUAL", "SEMANTIC", "VISUAL", "TEMPORAL", "COMPARATIVE", "AGGREGATION", "MULTI_HOP", "EXPLANATORY", "AMBIGUOUS", "UNKNOWN"];

export type Mode = "fast" | "balanced" | "deep" | "expert";
export type ModeRequest = Mode | "auto" | "v3";
export const MODES: Mode[] = ["fast", "balanced", "deep", "expert"];
export const isModeRequest = (x: unknown): x is ModeRequest => typeof x === "string" && (["auto", "v3", ...MODES] as string[]).includes(x);

/** What a mode spends (spec §8). */
export interface ExecutionProfile {
  mode: Mode;
  retrieval: "normal" | "expanded" | "multi-stage";
  reranking: boolean;
  reasoning: ReasoningLevel;
  maxOutputTokens: number;
  /** claim verification on (always, when a model answers); strict removes qualified claims too */
  strictClaims: boolean;
  /** evidence cards given to the reasoning model */
  cards: number;
  /** whether the planner model may decompose a question the rules could not */
  llmDecomposition: boolean;
}

export const PROFILES: Record<Mode, ExecutionProfile> = {
  fast: { mode: "fast", retrieval: "normal", reranking: false, reasoning: "none", maxOutputTokens: 800, strictClaims: false, cards: 8, llmDecomposition: false },
  balanced: { mode: "balanced", retrieval: "expanded", reranking: true, reasoning: "low", maxOutputTokens: 2000, strictClaims: false, cards: 12, llmDecomposition: true },
  deep: { mode: "deep", retrieval: "multi-stage", reranking: true, reasoning: "medium", maxOutputTokens: 4000, strictClaims: false, cards: 18, llmDecomposition: true },
  expert: { mode: "expert", retrieval: "multi-stage", reranking: true, reasoning: "high", maxOutputTokens: 6000, strictClaims: true, cards: 24, llmDecomposition: true },
};

/**
 * The visible reply of a rich answer (v4.8): the long-form reply (body, reasoning trail, claims,
 * follow-ups) needs room — about 1,500 tokens for a 250-word body with its claims, 3,000 for 800
 * words. v4's budgets above stay for the brief style.
 */
export const COMPOSE_TOKENS: Record<Mode, number> = { fast: 800, balanced: 3000, deep: 5000, expert: 8000 };

export interface RouteSignalsV4 {
  words: number;
  entities: number;
  predicates: string[];
  temporal: boolean;
  comparative: boolean;
  aggregation: boolean;
  explanatory: boolean;
  ambiguity: string[];
  operations: number;
  relations: number;
  image: boolean;
}

export interface QueryRoute {
  class: QueryClass;
  complexity: number;
  band: ComplexityBand;
  signals: RouteSignalsV4;
  reason: string;
}

// Patterns on folded text (lower case, no accents).
const EXPLAIN = /^(why|how come|explain|pourquoi|explique|warum|wieso|weshalb|erklar|por que|explica|perche|spiega)\b|\b(why is|why are|why does|why did|explain why|pourquoi est|warum ist|por que es|perche e)\b/;
const COMPARATIVE = /\b(more|fewer|less|than|versus|vs|compared? (to|with)|plus de|moins de|que @|davantage|mehr|weniger|als|mas|menos|que|piu|meno|di @)\b|\b(or|ou|oder|o)\s+@/;
const MORE_THAN = /\b(more|fewer|less)\b.*\bthan\b|\b(plus|moins)\b.*\bque\b|\b(mehr|weniger)\b.*\bals\b|\b(mas|menos)\b.*\bque\b|\b(piu|meno)\b.*\b(di|che)\b/;
const AGGREGATE = /\b(how many|number of|count|total|in total|average|on average|sum|combien|nombre|au total|en moyenne|wie viele|anzahl|insgesamt|durchschnitt|cuantos|cuantas|en total|promedio|quanti|quante|in totale|media)\b/;
const TEMPORAL = /\b(first|earliest|oldest|last|latest|newest|before|after|between|during|since|until|when|how long|sequence|then|later|earlier|premier|premiere|dernier|derniere|avant|apres|entre|pendant|depuis|quand|combien de temps|erste|letzte|neueste|vor|nach|zwischen|wahrend|seit|wann|wie lange|primero|primera|ultimo|ultima|antes|despues|cuando|cuanto tiempo|primo|prima|ultimo|ultima|dopo|tra|quando|quanto tempo)\b/;
const HISTORY = /\b(repost\w*|re-post\w*|re-?upload\w*|again|edit\w*|modif\w*|delet\w*|remov\w*|history|supprim\w*|republi\w*|a nouveau|de nouveau|bearbeit\w*|geandert|losch\w*|geloscht|erneut|wieder|editad\w*|borrad\w*|elimin\w*|otra vez|de nuevo|modificat\w*|cancellat\w*|di nuovo|ripubblic\w*)\b/;
const DURATION = /\b(how long|how many (days|hours|weeks)|combien de (jours|temps|heures)|wie lange|wie viele tage|cuanto tiempo|cuantos dias|quanto tempo|quanti giorni)\b/;
const EXACT = /\b(id|identifier|permalink|link|url|path|identifiant|lien|kennung|verlinkung|enlace|identificador|collegamento)\b/;
/** what follows a demonstrative that makes it a time ("this month", "este mes", "questa settimana"), not a thing */
const TIME_AFTER = String.raw`(?! (?:month|week|year|morning|evening|afternoon|weekend|summer|winter|spring|autumn|fall|time|season|mois|semaine|annee|matin|soir|ete|hiver|printemps|automne|monat|woche|jahr|morgen|abend|sommer|winter|fruhling|herbst|mes|semana|ano|manana|tarde|noche|verano|invierno|primavera|otono|mese|settimana|anno|mattina|sera|estate|inverno|autunno)\b)`;
const DEICTIC = new RegExp(
  [
    String.raw`^(it|they|this|that|these|those|he|she|ca|cela|celui|celle|es|dies|esto|eso|questo|quello)\b${TIME_AFTER}`,
    // "das" opens a question as a pronoun ("Das ist von wem?"), not as an article ("Das erste Kunstwerk …")
    String.raw`^das(?= (?:ist|war|wurde|hat|da|hier)\b|$)`,
    String.raw`\b(this|that) (one|artwork|image|picture|post|piece)\b`,
    String.raw`\b(cette|cet|ce) (oeuvre|œuvre|image|post|dessin)\b`,
    String.raw`\b(dieses|diese|dieser) (bild|kunstwerk|beitrag)\b`,
    // "posted this", but not "posted this month", nor the cleft "who was it that posted the first artwork"
    String.raw`\b(posted|made|drew|created|uploaded|published|is|was) (this|that|it)\b(?! (?:that|who|which)\b)${TIME_AFTER}`,
    String.raw`\b(a poste|a publie|a fait|a cree|a dessine|a partage) (ca|cela|ceci)\b`,
    String.raw`\bhat (das|dies|dieses) (gepostet|gemacht|erstellt|gezeichnet)\b`,
    String.raw`\b(publico|hizo|creo|dibujo|subio) (esto|eso)\b`,
    String.raw`\b(esta|este|esa|ese) (imagen|obra|publicacion|foto|dibujo)\b`,
    String.raw`\bha (pubblicato|fatto|creato|disegnato|postato) (questo|quello)\b${TIME_AFTER}`,
    String.raw`\b(questa|questo|quella|quello) (immagine|opera|foto|post|disegno)\b`,
    String.raw`\bdi chi e (questo|questa|quello|quella)\b${TIME_AFTER}`,
    String.raw`\b(pubblicat[oa]|postat[oa]|creat[oa]|fatt[oa]|disegnat[oa]|caricat[oa]) (questo|questa|quello|quella)\b${TIME_AFTER}`,
    String.raw`\b(publicad[oa]|publico|hizo|creo|subio|dibujo) (esto|eso|esta|este)\b${TIME_AFTER}`,
    String.raw`\b(postee?|publiee?|creee?|faite?|dessinee?) (ca|cela|ceci)\b`,
    String.raw`\b(gepostet|gemacht|erstellt|gezeichnet) (das|dies|dieses)\b${TIME_AFTER}`,
    // "c'est de qui, ça ?", "de qui est-ce ?"
    String.raw`\bc est (?:de|a) qui\b|\bde qui est (?:ce|ca|cela|ceci)\b|^de qui est ce$`,
    // "von wem ist das?", but "von wem ist das erste Kunstwerk?" is the article
    String.raw`\bvon wem (?:ist|stammt|kommt) (?:das|dies|dieses)$`,
  ].join("|"),
);

/**
 * Words of a question about "this" that name nothing: verbs of posting and of history, their
 * auxiliaries, the nouns that only say "artwork", the demonstratives themselves. A question whose
 * words are all of these (and stopwords) does not say what it is about.
 */
const UNNAMING = new Set(
  (
    "posted post made make created create drew draw drawn published publish uploaded upload shared share painted repost reposted edited edit deleted delete removed changed modified updated one artwork image picture piece this that it these those " +
    "fait cree creee dessine dessinee publie publiee poste postee partage modifie modifiee supprime supprimee ete oeuvre œuvre image dessin ca cela ceci celle celui " +
    "gemacht erstellt gepostet veroffentlicht gezeichnet hochgeladen bearbeitet geloscht geandert bild kunstwerk beitrag wurde dies dieses diese dieser das wem stammt kommt " +
    "hizo creo dibujo publico subio publicada publicado editado borrado eliminado imagen obra foto se esto eso esta este esa ese " +
    "fatto creato disegnato pubblicato pubblicata postato postata caricato caricata modificato modificata eliminato eliminata cancellato stato stata immagine opera disegno questo questa quello quella di chi e"
  ).split(" "),
);

/** Complexity weights per signal (sum of the maxima = 1). */
const W = { words: 0.1, entities: 0.15, predicates: 0.15, temporal: 0.08, comparative: 0.12, aggregation: 0.05, ambiguity: 0.1, operations: 0.15, relations: 0.1 };

/** Band bounds (SEARCH_COMPLEXITY_BANDS, four ascending numbers). */
export function bandBounds(env?: Env): [number, number, number, number] {
  const parts = String(env?.SEARCH_COMPLEXITY_BANDS ?? "").split(",").map((s) => Number(s.trim()));
  if (parts.length === 4 && parts.every((x, i) => Number.isFinite(x) && x > 0 && x < 1 && (i === 0 || x > parts[i - 1]))) return parts as [number, number, number, number];
  return [0.2, 0.4, 0.65, 0.85];
}

export function bandOf(complexity: number, env?: Env): ComplexityBand {
  const [a, b, c, d] = bandBounds(env);
  return complexity < a ? "trivial" : complexity < b ? "simple" : complexity < c ? "normal" : complexity < d ? "complex" : "deep";
}

/** Classify a question and score its complexity (spec §6-7). */
export function routeQuestion(question: string, plan: QueryPlan, program: QueryProgram, opts: { image?: boolean; env?: Env } = {}): QueryRoute {
  const f = fold(question).replace(/['’]/g, " ").replace(/[?!.,;:()"“”«»„¿¡]+/g, " ").replace(/\s+/g, " ").trim();
  const words = f.split(" ").filter(Boolean).length;
  const steps = program.steps;
  const ops = new Set(steps.map((s) => s.op));
  const predicates = new Set<string>();
  for (const s of steps) {
    if (s.op === "ask") predicates.add(s.plan?.intent ?? "search");
    else predicates.add(s.op);
  }
  if (HISTORY.test(f)) predicates.add("history");
  const entities =
    (plan.filters.authors?.length ?? 0) + (plan.filters.tags?.length ?? 0) + plan.concepts.length + (plan.filters.colors?.length ? 1 : 0) + (plan.filters.tones?.length ? 1 : 0) + program.titles.length + (plan.similarTo?.id ? 1 : 0) + program.extraEntities;
  const temporal = TEMPORAL.test(f) || plan.filters.from !== undefined || plan.filters.to !== undefined || ops.has("sequence") || ops.has("duration");
  const comparative = ops.has("compare_counts") || ops.has("compare_metric") || ops.has("sequence") || (COMPARATIVE.test(f) && (MORE_THAN.test(f) || /\b(or|ou|oder|o)\b/.test(f)) && entities >= 2);
  const aggregation = AGGREGATE.test(f) || ops.has("aggregate") || ops.has("group") || steps.some((s) => s.plan?.intent === "count" || s.plan?.intent === "compare");
  const explanatory = EXPLAIN.test(f);
  const ambiguity: string[] = [];
  // "this", "that", "ça", "dieses Bild": with nothing else that names an artwork, the question does not say which
  const words4 = steps.flatMap((x) => (x.plan?.residual ?? "").split(" ")).filter(Boolean).map((w) => w.replace(/^["“”«»„]+|["“”«»„]+$/g, ""));
  // a concept read from a word that only conjugates ("été": been, or summer) names nothing here
  const naming = (p: QueryPlan) => p.conceptMatches?.some((m) => !m.alias.split(" ").every((w) => UNNAMING.has(w))) ?? p.concepts.length > 0;
  // an account, a title, a subject, a colour or a tag name what "this" is; a date or "the first" do
  // not ("who posted this in September?", "is this the first artwork?" still ask about "this")
  const names =
    program.titles.length > 0 || !!plan.filters.authors?.length || naming(plan) || !!plan.filters.colors?.length || !!plan.filters.tags?.length ||
    steps.some((x) => x.title || (x.plan && naming(x.plan)) || x.plan?.filters.authors?.length) ||
    words4.some((w) => !UNNAMING.has(w) && !isStopword(w) && w.length > 1);
  // "who posted the first artwork and was it reposted?": "it" is the first step's answer; but in
  // "who posted this and when?" the "this" is in the first step's own clause, and names nothing
  const clause1 = fold(steps[0]?.text ?? "").replace(/['’]/g, " ").replace(/[?!.,;:()"“”«»„¿¡]+/g, " ").replace(/\s+/g, " ").trim();
  const refersBack = steps.length > 1 && steps.some((x) => x.refs?.length) && !DEICTIC.test(clause1);
  const unnamed = DEICTIC.test(f) && !opts.image && !plan.similarTo?.id && !names && !refersBack;
  if (unnamed) ambiguity.push("refers to something the question does not name");
  if (plan.confidence < 0.6 && plan.source === "rules" && program.pattern === "single") ambiguity.push("the rules are unsure of the question");
  if (!plan.residual && !plan.concepts.length && plan.intent === "search" && !plan.filters.authors?.length && !opts.image && !program.titles.length) ambiguity.push("no subject and no intent");
  if (program.ambiguous) ambiguity.push(program.ambiguous);
  const relations = steps.reduce((n, s) => n + (s.refs?.length ?? 0), 0);

  const score =
    W.words * Math.min(1, words / 25) +
    W.entities * Math.min(1, entities / 4) +
    W.predicates * Math.min(1, Math.max(0, predicates.size - 1) / 2) +
    W.temporal * (temporal ? (DURATION.test(f) || ops.has("sequence") ? 1 : 0.6) : 0) +
    W.comparative * (comparative ? 1 : 0) +
    W.aggregation * (aggregation ? 1 : 0) +
    W.ambiguity * Math.min(1, ambiguity.length / 2) +
    W.operations * Math.min(1, Math.max(0, steps.length - 1) / 3) +
    W.relations * Math.min(1, relations / 2);
  const complexity = Math.round(Math.min(1, score) * 1000) / 1000;

  const signals: RouteSignalsV4 = { words, entities, predicates: [...predicates], temporal, comparative, aggregation, explanatory, ambiguity, operations: steps.length, relations, image: !!opts.image };
  const done = (c: QueryClass, reason: string): QueryRoute => ({ class: c, complexity, band: bandOf(complexity, opts.env), signals, reason });

  if (opts.image || plan.intent === "similar" || plan.intent === "duplicate") return done("VISUAL", opts.image ? "a question about an image" : "similar artworks or duplicates");
  if (unnamed) return done("AMBIGUOUS", ambiguity.join("; "));
  if (explanatory) return done("EXPLANATORY", "asks why, or for an explanation");
  if (relations > 0 && steps.length > 1 && !ops.has("compare_counts") && !ops.has("compare_metric") && !ops.has("sequence") && !ops.has("duration")) return done("MULTI_HOP", "later steps use earlier answers");
  if (ops.has("compare_counts") || ops.has("compare_metric") || plan.intent === "compare" || (comparative && entities >= 2)) return done("COMPARATIVE", "compares artists or posts");
  if (ops.has("identify") || (EXACT.test(f) && program.titles.length > 0)) return done("EXACT", "asks for the identity of a named post");
  if (ops.has("sequence") || ops.has("duration") || ops.has("history")) return done("TEMPORAL", "order, duration or history of posts");
  // a program of explicit operators (a total, a grouping, an existence check) says what it asks
  if (ops.has("aggregate") || ops.has("group")) return done("AGGREGATION", "counts, totals or groups");
  if (ambiguity.length && !steps.some((s) => s.plan?.residual || s.title || s.op !== "ask")) return done("AMBIGUOUS", ambiguity.join("; "));
  if (plan.intent === "count") return done("AGGREGATION", "counts or totals");
  if (plan.intent === "find_first" || plan.intent === "find_last") {
    if (plan.output === "author") return done("FACTUAL", "who posted a first or latest artwork");
    return done("TEMPORAL", "the first or latest artwork");
  }
  if (plan.intent === "top" || ops.has("author_of") || ops.has("resolve_title")) return done("FACTUAL", "a fact about posts");
  if (plan.residual || plan.concepts.length || plan.filters.colors?.length || plan.filters.tones?.length || plan.filters.authors?.length) return done("SEMANTIC", "artworks about a subject");
  if (ambiguity.length) return done("AMBIGUOUS", ambiguity.join("; "));
  return done("UNKNOWN", "nothing the index can answer from");
}

const ORDER: Mode[] = ["fast", "balanced", "deep", "expert"];
const maxMode = (a: Mode, b: Mode) => (ORDER.indexOf(a) >= ORDER.indexOf(b) ? a : b);
const minMode = (a: Mode, b: Mode) => (ORDER.indexOf(a) <= ORDER.indexOf(b) ? a : b);
const asMode = (v: unknown, d: Mode): Mode => (typeof v === "string" && (MODES as string[]).includes(v) ? (v as Mode) : d);

/** The mode for a band, before any reason to think more (spec §8, §20). */
export function modeForBand(band: ComplexityBand): Mode {
  return band === "trivial" || band === "simple" ? "fast" : band === "normal" ? "balanced" : band === "complex" ? "deep" : "expert";
}

/**
 * Which mode runs. An explicit mode is honoured up to the caller's ceiling; "auto" follows the band,
 * one step up when the question needs a written synthesis that no operator can give (a fast
 * question that needs one runs balanced: a small model, low reasoning).
 */
/**
 * The mode of a question. `max`: the deepest this caller may have at all (an explicit request is
 * capped by it); `ceiling`: the deepest auto may pick by itself (the search box: an explicit
 * request is not capped by it); `floor`: the shallowest auto may pick (rich answers: balanced, so
 * the model elaborates; an explicit request is not raised by it).
 */
export function chooseMode(env: Env, requested: ModeRequest | undefined, route: QueryRoute, opts: { needsSynthesis: boolean; ceiling?: Mode; max?: Mode; floor?: Mode }): { mode: Mode; explicit: boolean } {
  const max = opts.max ?? "expert";
  const ceiling = minMode(opts.ceiling ?? max, max);
  // the caller's choice is explicit (it may call the model where the index alone would answer);
  // the deployment's default mode is not
  if (requested && requested !== "auto" && requested !== "v3") return { mode: minMode(requested, max), explicit: true };
  const dflt = String(env.SEARCH_DEFAULT_MODE ?? "auto").trim();
  if (!requested && isModeRequest(dflt) && dflt !== "auto" && dflt !== "v3") return { mode: minMode(dflt, ceiling), explicit: false };
  if (!bool(env.SEARCH_AUTO_COMPLEXITY, true)) return { mode: minMode(asMode(env.SEARCH_DEFAULT_MODE, "balanced"), ceiling), explicit: false };
  let mode = modeForBand(route.band);
  if (opts.needsSynthesis) mode = maxMode(mode, "balanced");
  if (opts.floor) mode = maxMode(mode, opts.floor);
  return { mode: minMode(mode, ceiling), explicit: false };
}

/** The shallowest mode rich answers run in by themselves (SEARCH_RICH_MIN_MODE, default balanced). */
export function richMinMode(env: Env): Mode {
  return asMode(String(env.SEARCH_RICH_MIN_MODE ?? "balanced").trim(), "balanced");
}
