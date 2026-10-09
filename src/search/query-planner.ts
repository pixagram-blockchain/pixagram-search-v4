// Query decomposition (spec §10, §49): a complex question becomes a short program of atomic,
// deterministic steps. Each step is a one-intent question planned by the v3 rules (planner.ts) or a
// v4 operator (history, comparison, sequence, duration, aggregate, group, title resolution), and
// later steps may take an answer from an earlier one ("they" → the earlier step's author).
//
//   "Who posted the first black-and-white cat artwork and was it later reposted?"
//     q1  ask      who posted the first black-and-white cat artwork    → author, post, time
//     q2  history  repost relationship of q1's post, by q1's author    → yes / no
//
//   "Did @alice post more cats than @bob?"
//     q1  ask  how many cats, author alice     q2  ask  how many cats, author bob
//     q3  compare_counts q1 q2, yes/no, more
//
// Rules first (English, French, German, Spanish, Italian). A question that looks like several steps
// but that no rule decomposes may be decomposed by the planner model (one call, JSON, validated):
// it only names operators and their arguments, which are turned back into canonical questions for
// the rule planner. The planner never sees a result and never produces one (spec §10).

import type { Env } from "../env";
import { now as nowSec } from "../env";
import { fold, isStopword, type Lang } from "../lib/text";
import { COLOR_NAMES } from "../enrich/color";
import { complete, replyObject } from "../llm/provider";
import { modelFor } from "../llm/router";
import { planQuery, type QueryPlan } from "./planner";
import type { Agg, HistoryRelation } from "./operators";

export type StepOp =
  | "ask" //            a one-intent question (v3: first, latest, count, top, compare, search)
  | "resolve_title" //  the live post with this exact title
  | "author_of" //      who posted the post with this title
  | "identify" //       the id and link of the post with this title
  | "fact" //           a field of an earlier step's answer (its date, author, post)
  | "history" //        edits, deletion, reposts, first and last appearance of a post's image
  | "compare_counts" // two counts: who has more
  | "compare_metric" // two posts by votes or payout
  | "sequence" //       which of two events came first
  | "duration" //       time between two events
  | "aggregate" //      sum / average / max / min of votes or payout over a set
  | "group" //          posts per month, per author, or per tag (what someone's artworks are about)
  | "exists" //         did @a post X: a count, as yes or no
  | "premise" //        "why did @a post the first X?": is it true that @a did
  | "image"; //         a question about an uploaded image (image-question.ts)

export interface StepRef {
  step: string;
  field: "author" | "post" | "time" | "title";
}

export interface Step {
  id: string;
  op: StepOp;
  /** the clause this step answers */
  text?: string;
  /** ask, aggregate, group, exists, premise: the sub-question as the rules planned it */
  plan?: QueryPlan;
  title?: string;
  /** what it takes from earlier steps */
  refs?: StepRef[];
  /** filters taken from earlier answers */
  inject?: { authors?: StepRef; from?: StepRef; to?: StepRef };
  /** history, fact, identify: the post it is about */
  target?: StepRef;
  field?: StepRef["field"];
  relation?: HistoryRelation;
  /** history repost: "did they post it again" (that author), else anyone */
  byAuthor?: StepRef;
  /** history: the account the question names as the one who did it ("reposted by @carol") */
  actor?: string;
  /** duration: the unit the question asks in ("how many hours …") */
  unit?: "hours" | "days" | "weeks";
  a?: string;
  b?: string;
  want?: "more" | "less" | "before" | "after" | "which";
  yesNo?: boolean;
  /** compare_counts and premise: the author the question asserts */
  claimed?: string;
  metric?: "net_votes" | "payout";
  agg?: Agg;
  by?: "month" | "author" | "tag";
  /** how this step's event is named in a sentence ("the first cat artwork", “Swan”) */
  label?: string;
}

/** The program in the spec's §10 form: typed atomic retrieval operations. */
export interface Subquery {
  id: string;
  type: "semantic" | "visual" | "temporal" | "history" | "aggregation" | "comparison" | "filter" | "title" | "fact";
  query?: string;
  operation?: string;
  step: string;
}

export interface QueryProgram {
  steps: Step[];
  /** the step whose result is the answer */
  final: string;
  source: "rules" | "llm";
  /** which rule (or the model) decomposed it */
  pattern: string;
  titles: string[];
  extraEntities: number;
  /** a reason the question cannot be planned without asking back */
  ambiguous?: string;
  notes: string[];
  /** what runs instead when the first step finds no post with the title (a quoted subject, not a title) */
  orElse?: QueryProgram;
}

export interface PlannerContext {
  authors: Set<string>;
  lang: Lang;
  now?: number;
}

// ---- text helpers -------------------------------------------------------------------------------

const QUOTES = /“([^”]{1,120})”|"([^"]{1,120})"|«\s*([^»]{1,120}?)\s*»|„([^“”]{1,120})[“”]|‘([^’]{1,120})’/g;

/** Titles written in quotes, in order. */
export function quotedTitles(raw: string): string[] {
  return [...raw.matchAll(QUOTES)].map((m) => (m[1] ?? m[2] ?? m[3] ?? m[4] ?? m[5] ?? "").trim()).filter(Boolean);
}

/** The question without what it quotes (a title's words are not the question's: “Ça va”, «Roma è bella»). */
export function withoutQuoted(raw: string): string {
  return raw.replace(QUOTES, " ");
}

/**
 * Folded text for the patterns: quoted titles become __t0__, __t1__…, punctuation spaces, "@" kept.
 * An apostrophe splits words, as in v3 ("l'auteur" → "l auteur").
 */
export function patternText(raw: string): { s: string; titles: string[] } {
  const titles: string[] = [];
  const replaced = raw.replace(QUOTES, (_m, a, b, c, d, e) => {
    titles.push(String(a ?? b ?? c ?? d ?? e ?? "").trim());
    return ` __t${titles.length - 1}__ `;
  });
  const s = fold(replaced)
    .replace(/œ/g, "oe")
    .replace(/æ/g, "ae")
    // an account keeps its dots and hyphens ("@top.witness"), its possessive aside
    .replace(/@([a-z0-9][a-z0-9.-]{0,30}[a-z0-9])/g, (_m, a: string) => `@${a.replace(/\./g, "\u0001").replace(/-/g, "\u0002")}`)
    .replace(/['’`´]/g, " ")
    // French questions: "a-t-il", "est-ce que", "peut-on" read as words
    .replace(/(\p{L})-(?:(t)-)?(il|elle|ils|elles|on|ce|vous|tu|je|nous)\b/gu, (_m, l: string, t: string | undefined, w: string) => `${l} ${t ? "t " : ""}${w}`)
    // "… and tell me if it was edited", "dis-moi si …", "sag mir, ob …": the question inside the request
    .replace(/\b(?:(?:can|could) you |please )?(?:tell|show) (?:me|us) (?:if|whether) |\b(?:dis|dites)[- ]moi (?:si|s) |\bsag(?:en sie)? mir,? ob |\bdime si |\bdimmi se /g, "")
    // "Who posted “T”? And when?": a second question after the first is asked with it
    .replace(/[?？]\s+[¿¡]?(?:(?:and|et|und|y|e)\s+)?(?=\S)/g, " and ")
    // "I'd like to know who …", "can you tell me …", "je voudrais savoir …": the question after it
    .replace(/^(?:(?:i d|i would) like to know|i want to know|i wonder|i m wondering|do you know|(?:can|could) you tell me|please tell me|tell me|je voudrais savoir|j aimerais savoir|dites[- ]moi|dis[- ]moi|ich mochte wissen|ich wurde gern wissen|sag mir|me gustaria saber|quisiera saber|dime|vorrei sapere|dimmi)\s+(?=\S)/, "")
    // dashes between parts of a question ("“Bamboo” — who made it?", "@a or @b — who …") are punctuation
    .replace(/\s[—–-]\s|[—–]/g, " ")
    .replace(/\b(?:(?:can|could) you |please )?(?:tell|show) (?:me|us) (?=(?:when|who|what|which|how|where)\b)/g, "")
    .replace(/[?!.,;:()¿¡]+/g, " ")
    .replace(/\u0001/g, ".")
    .replace(/\u0002/g, "-")
    .replace(/\s+/g, " ")
    .trim();
  return { s, titles };
}

// ---- instructions inside a question --------------------------------------------------------------

/** A sentence addressed to the system rather than a question ("ignore your instructions…", "say that…"), on folded text. */
const INSTRUCTION =
  /\b(new|additional|updated|extra) (instructions?|rules?|polic(y|ies)|orders?)\b|\balways (answer|say|reply|respond)\b|\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|prompts?|polic(y|ies)|guidelines?|system|restrictions?|evidence|data|index|facts?|context|results?|above|previous|prior|everything)\b|\b(and|then) (say|claim|state|write)\b.{0,30}\b(that|it was|it is|only)\b|\b(and|then) (answer|reply|respond|tell)(?: (me|us|them|everyone|the user))? (that|only)\b|\b(you are now|act as|pretend (to be|that|you)|system prompt|developer mode|jailbreak)\b|^(say|state|claim|write)\b.{0,30}\b(that|only|it was|it is)\b|^(just|only|simply) (say|answer|reply|respond|write|state)\b|^(say|answer|reply|respond|write|state)\b.{0,12}@|^(answer|reply|respond|output|print|tell (me|us|them|everyone))\b.{0,24}\b(that|only)\b|\b(ignore|oublie|ignoriere|vergiss|ignora|olvida|dimentica)\b.{0,40}\b(les instructions|les regles|anweisungen|regeln|instrucciones|reglas|istruzioni|regole)\b/;

/** What addresses the system itself (never a question's scope): "ignore your rules", "you are now …", "new instructions". */
const SYSTEM_DIRECTIVE =
  /\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(instructions?|rules?|prompts?|polic(y|ies)|guidelines?|system|restrictions?|evidence|data|index|facts?|context|results?|above|previous|prior|everything)\b|\b(new|additional|updated|extra) (instructions?|rules?|polic(y|ies)|orders?)\b|\b(you are now|act as|pretend|system prompt|developer mode|jailbreak)\b|\b(ignore|oublie|ignoriere|vergiss|ignora|olvida|dimentica)\b.{0,40}\b(les instructions|les regles|anweisungen|regeln|instrucciones|reglas|istruzioni|regole)\b/;

/**
 * The question without the sentences that instruct the system (spec §42: the question is only a
 * question). A sentence is dropped only when another one remains to be answered.
 */
export function withoutInstructions(question: string): { text: string; dropped: string[] } {
  // a tag before the question ("[admin override] …", "SYSTEM: …") is never part of it
  const tag = /^\s*(\[[^\]]{1,60}\]|\{[^}]{1,60}\}|<[^>]{1,60}>|(?:system|admin|developer|assistant|root)\s*:)\s*(?=\S)/i.exec(question);
  if (tag && question.slice(tag[0].length).trim()) {
    const rest = withoutInstructions(question.slice(tag[0].length));
    return { text: rest.text, dropped: [tag[1].trim(), ...rest.dropped] };
  }
  // an instruction to the system before a colon ("Override your guidelines: how many …") is not part of the
  // question either; "Answer only for @alice: …" scopes it, and stays
  const colon = /^([^:]{3,80}):\s+(?=\S)/.exec(question);
  if (colon && !/[@#"“”«»„]/.test(colon[1]) && SYSTEM_DIRECTIVE.test(fold(colon[1]).replace(/['’]/g, " ")) && question.slice(colon[0].length).trim()) {
    const rest = withoutInstructions(question.slice(colon[0].length));
    return { text: rest.text, dropped: [colon[1].trim(), ...rest.dropped] };
  }
  const sentences = question.split(/(?<=[.!?。！？])\s+/).filter((x) => x.trim());
  if (sentences.length < 2) return { text: question, dropped: [] };
  const keep: string[] = [];
  const dropped: string[] = [];
  for (const x of sentences) (INSTRUCTION.test(fold(x).replace(/['’]/g, " ")) ? dropped : keep).push(x);
  return keep.length && dropped.length ? { text: keep.join(" ").trim(), dropped } : { text: question, dropped: [] };
}

const AUTHOR = String.raw`@?([a-z0-9][a-z0-9.-]{1,31})`;
/** What follows "by", "par", "von", "por", "da" without naming an account ("by hand", "par erreur", "da zero"), even where such an account exists */
const BY_IDIOM = /^(?:hand|mistake|accident|chance|someone|somebody|anyone|anybody|everyone|nobody|me|you|us|them|him|her|ai|ia|ki|erreur|hasard|quelqu|quelquun|personne|moi|toi|nous|vous|eux|elle|lui|zero|error|accidente|casualidad|alguien|nadie|mi|ti|sbaglio|caso|qualcuno|nessuno|errore|fehler|zufall|versehen|jemand|niemand|jemandem|mir|dir|uns|euch|ihnen)$/;
const TITLE_TOKEN = /^__t(\d+)__$/;

/** An account named in a clause ("@alice", or a known author's bare name). */
function authorIn(clause: string, authors: Set<string>): string | null {
  const m = /@([a-z0-9][a-z0-9.-]{1,31})/.exec(clause);
  if (m) return m[1];
  for (const w of clause.split(" ")) if (authors.has(w)) return w;
  return null;
}

function titleOf(token: string | undefined, titles: string[]): string | null {
  const m = token ? TITLE_TOKEN.exec(token.trim()) : null;
  return m ? (titles[Number(m[1])] ?? null) : null;
}

/** A clause turned back into text for the rule planner (titles restored in quotes, with a question mark). */
function clauseText(clause: string, titles: string[]): string {
  return `${clause.replace(/__t(\d+)__/g, (_m, i) => `"${titles[Number(i)] ?? ""}"`).trim()}?`;
}

const plan = (text: string, ctx: PlannerContext): QueryPlan => {
  const p = planQuery(text, { authors: ctx.authors, mode: "ask", now: ctx.now ?? nowSec(), v4: true });
  return { ...p, lang: ctx.lang };
};

// ---- vocabularies (folded) ---------------------------------------------------------------------------

const PERSON = /\b(they|them|their|he|him|his|she|her|ils|elles|il|elle|leur|leurs|lui|er|sie|ihm|ihr|ihre|ihren|ihrem|ellos|ellas|ella|su|sus|lei|loro|suo|sua|suoi)\b/;
const THING = /\b(it|its|this|that|le|la|l|lo|es|ihn|questo|quello|esto|eso|ce|ca)\b/;
const FOLLOW =
  "it was|it has|it is|it got|they did|they have|they posted|i think|i guess|maybe|perhaps|probably|by whom|il a|elle a|il est|elle est|did|was|were|has|have|had|is|are|when|how|what|which|who|where|then|later|a t il|a t elle|ont ils|l a|l ont|est ce|quand|combien|quel|quelle|qui|hat|haben|wurde|wurden|ist|wann|wie|welche|welcher|wer|lo|la|cuando|cuantos|cuantas|quien|ha|hanno|e stato|quando|quanti|quante|chi|cosa|l ha|l hanno|lo ha|la ha|le ha|lo volvio|la volvio|lo ha|se";
const CONJ = new RegExp(String.raw`\s(?:and|et|und|y|e)\s(?=(?:${FOLLOW})\b)`);

const REPOST = /\b(repost\w*|re post\w*|re ?upload\w*|post(?:ed)? (?:it |that |the same image )?again|upload(?:ed)? (?:it )?again|publish(?:ed)? (?:it )?again|share(?:d)? (?:it )?again|republi\w*|reposte\w*|a nouveau|de nouveau|erneut|wieder|nochmal|noch einmal|de nuevo|otra vez|di nuovo|ripubblic\w*|ricaric\w*)\b/;
const EDIT = /\b(edit\w*|modif\w*|chang\w*|updat\w*|bearbeit\w*|geander\w*|aktualisier\w*|editad\w*|editó|cambiad\w*|modificat\w*|aggiornat\w*)\b/;
const DELETE = /\b(delet\w*|remov\w*|supprim\w*|efface\w*|losch\w*|geloscht|entfern\w*|elimin\w*|borrad\w*|cancellat\w*|rimoss\w*)\b/;
const STILL =
  /\b(still (online|visible|there|up|shown|exists?|available)|(?:toujours|encore) (en ligne|visible|la|disponible)|existe (?:toujours|encore)|immer noch (online|sichtbar|da)|noch (online|sichtbar|da|vorhanden)|existiert (?:noch|immer noch)|todavia (en linea|visible|existe|disponible)|sigue (en linea|visible|existiendo|disponible)|ancora (online|visibile|disponibile|li)|esiste ancora)\b/;
const FIRST_APPEAR = /\b(first (appear\w*|seen|shown)|(?:apparu\w*|vue?) (?:pour la )?premiere fois|zuerst (erschien\w*|aufgetaucht|gesehen)|erstmals|aparecio por primera vez|apparsa per la prima volta)\b/;
/** "the first …" / "the latest …" in five languages */
const FIRST_LAST = /\b(first|earliest|oldest|premier\w*|erste[nrsm]?|primer\w*|prim[oa]|last|latest|newest|dernier\w*|letzte[nrsm]?|neueste[nrsm]?|ultim[oa])\b/;
/** the words that name a history relation, to take out of the event they are about */
const RELATION_WORDS = new RegExp([REPOST, EDIT, DELETE, STILL, FIRST_APPEAR].map((r) => r.source).join("|"), "g");
/**
 * Auxiliaries of history questions ("has it been deleted", "a-t-elle été modifiée", "ist … gelöscht
 * worden", "è mai stato modificato"). Italian "mai" only before "stato": in French it is May.
 */
const HISTORY_AUX = /\b(been|ever|later|since|then|ete|deja|jamais|schon|jemals|worden|wurde|ist|sido|stato|stata|ya|gia)\b|\bmai(?= stat[oaie]\b)/g;
const MORE = /\b(more|plus|mehr|mas|piu)\b/;
const LESS = /\b(fewer|less|moins|weniger|menos|meno)\b/;
const THAN = String.raw`(?:than|que|als|di|de)`;
const OR = String.raw`(?:or|ou|oder|o|vs|versus|contre|gegen|contra)`;
const POSTED = String.raw`(?:posted|post|made|make|created|create|drew|draw|drawn|published|publish|uploaded|upload|shared|poste|postee|postes|publie|publiee|cree|dessine|fait|realise|peint|gepostet|postet|erstellt|gemacht|veroffentlicht|gezeichnet|gemalt|hochgeladen|publicado|publico|creado|dibujado|hecho|hizo|pintado|subido|subio|pubblicato|creato|disegnato|fatto|dipinto|realizzato|postato|caricato)`;
const WHO = String.raw`(?:who|qui|wer|quien|chi)`;
const METRIC_VOTES = /\b(votes?|voted|likes?|liked|upvotes?|upvoted|hearts?|popular|popularity|populaire|beliebt\w*|stimmen|votos?|votad[oa]|popular|voti|votat[oa]|popolare|j aime|me gusta|mi piace)\b/;
const METRIC_PAYOUT = /\b(payout|payouts|pxs|rewards?|earn\w*|paid|recompense\w*|gagn\w*|belohnung\w*|verdien\w*|recompensa\w*|ganad\w*|ricompens\w*|guadagn\w*)\b/;

const HISTORY_Q = /^(?:was|has|have|did|is|were|a t il|a t elle|est ce que|est il|est elle|wurde|ist|hat|fue|se|ha sido|ha|e stato|e stata|when was|when did|quand|wann|cuando|quando)\b/;
/** "the most …" that orders posts rather than counting them: "most recent", "most liked", "le plus récent", "die neuesten" */
const NOT_A_COUNT =
  /^(?:the |di |de |d )?(?:recemment|recent\w*|latest|newest|last|oldest|earliest|first|liked|loved|popular|voted|upvoted|rewarded|paid|viewed|commented|beautiful|famous|(?:le |la |les )?(?:recent\w*|dernier\w*|premier\w*|populaire\w*|aime\w*|vote\w*|ancien\w*)|neueste[nrsm]?|letzte[nrsm]?|erste[nrsm]?|alteste[nrsm]?|beliebt\w*|reciente\w*|ultim[oaie]s?|primer\w*|popular\w*|votad\w*|antigu\w*|recent[ei]|popolar[ei]|votat\w*|amat\w*|vecchi\w*)\b/;

/** "how many days / hours / weeks", in five languages */
const HOW_MANY_UNITS = String.raw`how many (?:days|hours|weeks)|combien (?:de |d )(?:jours|heures|semaines)|wie viele (?:tage|stunden|wochen)|cuant[oa]s (?:dias|horas|semanas)|quant[ie] (?:giorni|ore|settimane)`;

/** The unit a duration question asks in ("how many hours …"), on the pattern text (titles are __tN__). */
function unitOf(s: string): "hours" | "days" | "weeks" | undefined {
  const m = /\b(?:how many|combien de|combien d|wie viele|cuant[oa]s|quant[ie]) (days|hours|weeks|jours|heures|semaines|tage|stunden|wochen|dias|horas|semanas|giorni|ore|settimane)\b/.exec(s);
  if (!m) return undefined;
  return /^(hours|heures|stunden|horas|ore)$/.test(m[1]) ? "hours" : /^(weeks|semaines|wochen|semanas|settimane)$/.test(m[1]) ? "weeks" : "days";
}

// ---- the rules -----------------------------------------------------------------------------------------

function single(p: QueryPlan, titles: string[], pattern = "single"): QueryProgram {
  return { steps: [{ id: "q1", op: "ask", plan: p, text: p.text }], final: "q1", source: "rules", pattern, titles, extraEntities: 0, notes: [] };
}

function relationOf(clause: string): HistoryRelation | null {
  if (STILL.test(clause)) return "last_seen";
  if (FIRST_APPEAR.test(clause)) return "first_seen";
  if (REPOST.test(clause)) return "repost";
  // "has anyone posted “T” again?": the verb and "again" around what was posted
  if (/\b(post\w*|upload\w*|publish\w*|share\w*|poste\w*|publie\w*|gepostet|veroffentlicht|hochgeladen|publicad\w*|publico|subid\w*|pubblicat\w*|postat\w*|caricat\w*)\b.*\b(again|a nouveau|de nouveau|erneut|wieder|nochmal|de nuevo|otra vez|di nuovo)\b/.test(clause)) return "repost";
  // "encore" and "ancora" are "again" only next to the verb ("a encore posté", "l'ha ripostato ancora"); else "still"
  if (/\b(?:a|ont|l a|l ont|ha|hanno|l ha|l hanno) (?:encore|ancora) (?:poste\w*|publie\w*|pubblicat\w*|postat\w*|caricat\w*)\b|\b(?:poste\w*|publie\w*|pubblicat\w*|postat\w*|caricat\w*) (?:encore|ancora)$/.test(clause)) return "repost";
  if (DELETE.test(clause)) return "delete";
  if (EDIT.test(clause)) return "edit";
  return null;
}

const RELATION_ANY = new RegExp([REPOST, EDIT, DELETE].map((r) => r.source).join("|"));
const RELATION_THEN_TITLE = new RegExp(`(?:${[REPOST, EDIT, DELETE].map((r) => r.source).join("|")}) __t\\d+__$`);
const RELATION_END = new RegExp(`(?:${[REPOST, EDIT, DELETE].map((r) => r.source).join("|")}|again|a nouveau|de nouveau|erneut|wieder|de nuevo|otra vez|di nuovo)$`);

/**
 * Who did what a history question asks about, when it says so after the verb: "reposted by @carol",
 * "supprimé par @x", "von @x gelöscht". "The first cat by @carol" names carol's cat instead (no verb
 * before "by"). The clause without that part is returned with it.
 */
function actorOf(s: string, authors: Set<string>): { actor: string; clause: string } | null {
  let at: string | undefined;
  let name: string | undefined;
  let clause: string | undefined;
  const m = /^(.*\S)\s+(?:by|par|por|da|von)\s+(@?)([a-z0-9][a-z0-9.-]{1,31})$/.exec(s);
  // "… reposted by @x", and with the verb first: "¿fue republicado «T» por @x?", "è stato ripubblicato «T» da @x?"
  if (m && (RELATION_END.test(m[1].replace(/\s+(?:been|ete|stato|stata|sido|worden)$/, "")) || RELATION_THEN_TITLE.test(m[1]))) [clause, at, name] = [m[1], m[2], m[3]];
  // the one who did it first: "did @carol repost the first cat?", "est-ce que @carol a reposté …", "hat @carol … erneut gepostet?"
  const lead = clause ? null : /^(did|does|has|have|est ce que|hat|ha)\s+(@?)([a-z0-9][a-z0-9.-]{1,31})\s+(.+)$/.exec(s);
  if (lead && RELATION_ANY.test(lead[4])) [clause, at, name] = [`${lead[1]} ${lead[4]}`, lead[2], lead[3]];
  // German puts the verb last: "wurde „T“ von @x gelöscht", "… von @x erneut gepostet"
  const g = m ? null : /^(.*\S)\s+von\s+(@?)([a-z0-9][a-z0-9.-]{1,31})\s+((?:erneut |wieder |nochmal )?(?:gepostet|veroffentlicht|hochgeladen)|bearbeit\w*|geander\w*|geloscht|entfernt)(?:\s+worden)?$/.exec(s);
  if (g && (g[4] !== "gepostet" && g[4] !== "veroffentlicht" && g[4] !== "hochgeladen")) [clause, at, name] = [`${g[1]} ${g[4]}`, g[2], g[3]];
  if (!clause || !name) return null;
  // an account is written with "@" or known; "edited by hand" names nobody
  if (!at && !authors.has(name)) return null;
  return { actor: name, clause };
}

/** a history question with its subject first: "the first cat was reposted?", "le premier chat a-t-il été republié ?", "il primo gatto è stato ripubblicato?" */
const SUBJECT_FIRST =
  /^(?:the |le |la |l |il |lo |el |der |die |das )?(?:very )?(?:first|earliest|oldest|last|latest|newest|premier\w*|derni\w*|erste[nrsm]?|letzte[nrsm]?|neueste[nrsm]?|primer\w*|ultim[oaie]s?|prim[oa])\b.*\s(?:e|a t il|a t elle|a|est|fue|ha|wurde|ist|was|has|is)(?:\s(?:mai|jamais|deja|gia|ya|ever|schon|jemals|ete|stato|stata|sido|been))*\s\S+$/;

const ARTWORK_NOUN = /\b(?:artworks?|posts?|pieces?|images?|pictures?|drawings?|oeuvres?|dessins?|kunstwerke?|bilder?|beitrage?|obras?|imagen(?:es)?|dibujos?|opere|opera|immagin[ei]|disegn[oi])\b/;

/** An event named in a clause: a quoted title, "the first/latest X", or "@a's first X". */
function eventStep(clause: string, id: string, titles: string[], ctx: PlannerContext): Step | null {
  const t = clause.trim();
  const title = titleOf(t.split(" ").find((w) => TITLE_TOKEN.test(w)), titles);
  if (title && /^(?:the |la |le |l |der |die |das |el |il |lo )?(?:post |artwork |oeuvre |bild |obra |opera )?__t\d+__$/.test(t)) return { id, op: "resolve_title", title, text: `“${title}”`, label: title };
  if (/\b(first|earliest|oldest|premier|premiere|erste[nrsm]?|primer\w*|prim[oa])\b/.test(t) || /\b(last|latest|newest|dernier\w*|letzte[nrsm]?|neueste[nrsm]?|ultim[oa])\b/.test(t)) {
    const q = plan(`what is ${t}`, ctx);
    // "the first cat", "@a's latest", or "the first artwork" (Pixagram's very first): not a bare "first"
    if ((q.intent === "find_first" || q.intent === "find_last") && (q.residual || q.filters.authors?.length || q.concepts.length || ARTWORK_NOUN.test(t))) return { id, op: "ask", plan: q, text: t };
  }
  return null;
}

/** Rule decomposition. A question no rule matches is one step (v3's single intent). */
export function decomposeRules(question: string, base: QueryPlan, ctx: PlannerContext): QueryProgram {
  const { s, titles } = patternText(question);
  const A = ctx.authors;

  // ---- a title said first, then asked about: "“Bamboo” — who made it?", "« F-16 », qui l'a posté ?" ------------
  {
    const m = /^(__t\d+__) (.+?) (?:it|this|that|ca|cela|es|das|lo|la|esto|questo|quello)$/.exec(s) ?? /^(__t\d+__) (qui|wer|quien|chi) (?:l |lo |la |es |l )?(a|hat|ha|lo ha|la ha) (.+)$/.exec(s);
    if (m && titles.length === 1) {
      const rest = m.length === 3 ? `${m[2]} ${m[1]}` : `${m[2]} ${m[3]} ${m[4]} ${m[1]}`;
      const again = decomposeRules(clauseText(rest, titles), base, ctx);
      if (again.pattern !== "single") return again;
    }
  }

  // ---- EXACT: the id / link of a named post -------------------------------------------------------
  {
    const where = /^where (?:can i|do i|could i|would i|to) find (?:the )?(?:post |artwork )?(__t\d+__)$|^(?:ou|wo|donde|dove) (?:puis je |trouver |kann ich |finde ich |puedo |posso )?(?:trouver |finden |encontrar |trovare )?(?:le |la |l |den |die |das |el |il )?(?:post |bild |kunstwerk |oeuvre |obra |opera )?(__t\d+__)(?: finden)?$/.exec(s);
    const wt = where ? titleOf(where[1] ?? where[2], titles) : null;
    if (wt) return { steps: [{ id: "q1", op: "identify", title: wt, text: question, label: wt }], final: "q1", source: "rules", pattern: "identify", titles: [wt], extraEntities: 0, notes: [] };
    const m = /^(?:(?:can|could) you (?:please )?|please |pouvez vous |peux tu |kannst du |konnen sie |puedes |podrias |puoi |potresti )?(?:what is |whats |what s |give me |show me |send me |tell me |quel est |donne moi |donnez moi |was ist |wie lautet |gib mir |nenne mir |cual es |dame |qual e |dammi )?(?:the |l |le |la |die |der |das |el |il )?(?:id|identifier|identifiant|kennung|identificador|identificativo|link|lien|enlace|collegamento|url|permalink|address|adresse|path)(?: number| numero| nummer)? (?:of|for|to|de|du|des|von|vom|zu|fur|del|di|per) (?:the |l |le |la |die |der |das |el |il )?(?:post |artwork |oeuvre |bild |obra |opera )?(.+)$/.exec(s);
    if (m) {
      const title = titleOf(m[1], titles) ?? m[1].replace(/\s*__t\d+__\s*/g, " ").trim();
      if (title) return { steps: [{ id: "q1", op: "identify", title, text: question, label: title }], final: "q1", source: "rules", pattern: "identify", titles: [title], extraEntities: 0, notes: [] };
    }
  }

  // ---- "was “T” posted by @x?", "did @x post “T”?": the title's author against the one named --------------
  if (titles.length === 1) {
    const forms: Array<[RegExp, number, number]> = [
      [new RegExp(String.raw`^(?:was|is|were) (__t\d+__) (?:(?:${POSTED}|drawn|painted) )?by ${AUTHOR}$`), 1, 2],
      [new RegExp(String.raw`^(?:is|was) ${AUTHOR} the (?:author|artist|creator|maker|owner) of (?:the post |the artwork )?(__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(?:is|was) (__t\d+__) (?:an? )?(?:artwork|post|piece|drawing|work|painting) (?:by|from|of) ${AUTHOR}$`), 1, 2],
      [new RegExp(String.raw`^(?:est ce que )?${AUTHOR} (?:est il |est elle |est )?l (?:auteur|auteure|artiste|createur|createurice) de (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(?:ist|war) ${AUTHOR} (?:der|die) (?:autor|autorin|kunstler|kunstlerin|urheber|urheberin) von (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(?:es|fue) ${AUTHOR} (?:el|la) (?:autor|autora|artista|creador|creadora) de (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^${AUTHOR} (?:e|era) (?:l |la |il )?(?:autore|autrice|artista|creatore|creatrice) di (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(?:did|has) ${AUTHOR} (?:${POSTED}) (?:the |a )?(?:post |artwork |image )?(__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(?:est ce que )?(__t\d+__) (?:a t il |a t elle |a )?(?:ete )?(?:${POSTED}) par ${AUTHOR}$`), 1, 2],
      [new RegExp(String.raw`^(?:est ce que )?${AUTHOR} a (?:${POSTED}) (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^wurde (__t\d+__) von ${AUTHOR} (?:gepostet|veroffentlicht|erstellt|gemacht|gezeichnet|hochgeladen)$`), 1, 2],
      [new RegExp(String.raw`^(?:hat|ist) ${AUTHOR} (?:der autor von |die autorin von )?(__t\d+__)(?: gepostet| veroffentlicht| erstellt| gemacht| gezeichnet)?$`), 2, 1],
      [new RegExp(String.raw`^(?:fue |se )?(__t\d+__) (?:fue )?(?:publicad[oa]|subid[oa]|cread[oa]|dibujad[oa]|publico) por ${AUTHOR}$`), 1, 2],
      [new RegExp(String.raw`^(?:publico|subio|creo|dibujo|ha publicado) ${AUTHOR} (__t\d+__)$`), 2, 1],
      [new RegExp(String.raw`^(__t\d+__) e stat[oa] (?:pubblicat[oa]|postat[oa]|creat[oa]|disegnat[oa]|fatt[oa]) da ${AUTHOR}$`), 1, 2],
      [new RegExp(String.raw`^${AUTHOR} ha (?:pubblicato|postato|creato|disegnato|fatto) (__t\d+__)$`), 2, 1],
    ];
    for (const [re, ti, ai] of forms) {
      const m = re.exec(s);
      const title = m ? titleOf(m[ti], titles) : null;
      // "qui a posté « T »" is a question, not a claim about @qui; "made by AI", "posted by mistake",
      // "créé par erreur" name nobody: the account is written with "@", or is a known author's bare name
      const named = m ? m[ai] : "";
      const isAccount = !!named && !/^(?:who|qui|wer|quien|chi|what|que|was|cosa|which)$/.test(named) && (new RegExp(`@${named.replace(/[.-]/g, "\\$&")}(?![a-z0-9.-])`).test(s) || (A.has(named) && !BY_IDIOM.test(named)));
      if (m && title && isAccount) return { steps: [{ id: "q1", op: "author_of", title, claimed: named, text: question, label: title }], final: "q1", source: "rules", pattern: "author_check", titles, extraEntities: 1, notes: [] };
    }
  }

  // ---- who posted “T” (a quoted title, or "the artwork titled …") -----------------------------------
  {
    const m =
      new RegExp(String.raw`^${WHO} (?:has |have |a |hat |ha )?${POSTED} (?:the |l |le |la |der |die |das |el |il )?(?:post |artwork |image |picture |oeuvre |bild |obra |opera )?(?:titled |called |named |intitule\w* |appele\w* |betitelt\w* |genannt |titulad\w* |llamad\w* |intitolat\w* |chiamat\w* )?(__t\d+__)$`).exec(s) ??
      // "who is the author of “T”", "qui est l'auteur de « T »", "wer ist der Autor von „T“", "¿quién es el autor de «T»?", "chi è l'autore di «T»?"
      /^(?:who s|who is|whos|who was) behind (?:the |this )?(?:post |artwork |piece )?(__t\d+__)$/.exec(s) ??
      /^(?:who is|who s|whos|who was|what is|whats|what s|qui est|qui etait|quel est|quelle est|c est qui|wer ist|wer war|was ist|quien es|quien fue|quien era|cual es|chi e|chi era|chi e stato|chi e stata|qual e|quale e) (?:the |l |le |la |der |die |el |il )?(?:author|artist|creator|maker|auteur|auteure|artiste|createur|createurice|kunstler(?:in)?|autor(?:in|a)?|urheber(?:in)?|artista|autore|autrice|creatore|creatrice|creador|creadora) (?:of|behind|de|du|von|vom|del|di|dell) (?:the |l |le |la |der |die |das |el |il )?(?:post |artwork |image |oeuvre |bild |kunstwerk |obra |opera )?(__t\d+__)$/.exec(s) ??
      // German puts the verb last: "wer hat „T“ gepostet"
      /^wer hat (?:den |die |das )?(?:beitrag |bild |kunstwerk )?(__t\d+__) (?:gepostet|veroffentlicht|erstellt|gemacht|gezeichnet|hochgeladen|geteilt)$/.exec(s) ??
      // "“T” was made by whom?", "by whom was “T” posted?", "von wem ist „T“?", "de qui est « T » ?", "di chi e «T»?", "¿de quién es «T»?"
      new RegExp(String.raw`^(__t\d+__) (?:was |is )?(?:${POSTED}|drawn|painted) by whom$`).exec(s) ??
      new RegExp(String.raw`^by whom (?:was|is) (__t\d+__) (?:${POSTED}|drawn|painted)$`).exec(s) ??
      /^(?:von wem (?:ist|stammt|kommt)|de qui est|di chi e|de quien es) (?:der |die |das |le |la |l |il |el )?(?:beitrag |bild |kunstwerk |oeuvre |opera |obra )?(__t\d+__)$/.exec(s);
    const titled = /\b(?:titled|called|named|intitulee?|appelee?|betitelt|genannt|titulad[oa]|llamad[oa]|intitolat[oa]|chiamat[oa]) (.+)$/.exec(s);
    const title = m ? titleOf(m[1], titles) : titled && new RegExp(`^${WHO}\\b`).test(s) ? (titleOf(titled[1], titles) ?? titled[1]) : null;
    if (title) return { steps: [{ id: "q1", op: "author_of", title, text: question, label: title }], final: "q1", source: "rules", pattern: "author_of", titles: [title], extraEntities: 0, notes: [] };
  }

  // ---- when a named post was posted: "When was “T” posted?", "Wann wurde „T“ gepostet?" (v3 read the title as a subject) --
  if (titles.length === 1) {
    const PUB = String.raw`(?:posted|published|created|uploaded|made|shared|drawn|released)`;
    const m =
      new RegExp(String.raw`^(?:when|what date|on what date|on what day|which day) (?:was|did|were) (__t\d+__) (?:get |go |come )?(?:${PUB}|up|out|online)?$`).exec(s) ??
      new RegExp(String.raw`^when did ${AUTHOR} (?:post|publish|upload|create|make|share) (__t\d+__)$`).exec(s) ??
      /^(?:quand|a quelle date|quel jour) (?:est ce que )?(__t\d+__) (?:a t il |a t elle |a |est )?(?:ete )?(?:poste|publie|cree|mis en ligne|partage)e?s?$/.exec(s) ??
      /^(?:quelle est la date de publication de|date de publication de) (__t\d+__)$/.exec(s) ??
      /^(?:wann|an welchem tag) (?:wurde|ist) (__t\d+__) (?:gepostet|veroffentlicht|erstellt|hochgeladen|erschienen)(?: worden)?$/.exec(s) ??
      /^(?:cuando|que dia|en que fecha) (?:se )?(?:publico|subio|creo|fue publicad[oa]|fue subid[oa]) (__t\d+__)$/.exec(s) ??
      /^(?:cuando|en que fecha) (__t\d+__) (?:fue|se) (?:publicad[oa]|subid[oa]|publico)$/.exec(s) ??
      /^(?:quando|che giorno|in che data) (?:e stato|e stata|fu|venne) (?:pubblicat[oa]|postat[oa]|creat[oa]|caricat[oa]) (__t\d+__)$/.exec(s) ??
      /^(?:quando|in che data) (__t\d+__) (?:e stato|e stata|fu) (?:pubblicat[oa]|postat[oa]|creat[oa]|caricat[oa])$/.exec(s);
    const title = m ? titleOf(m[m.length - 1], titles) : null;
    // "when did @x post “T”": the date, whoever posted it (the answer says who)
    if (title) return { steps: [{ id: "q1", op: "author_of", title, field: "time", text: question, label: title }], final: "q1", source: "rules", pattern: "date_of", titles: [title], extraEntities: 0, notes: [] };
  }

  // ---- history of one named post: "Was “T” edited?", "When was “T” deleted?", "Is “T” still online?" --
  // ("was “T” posted by @bob and was it edited?" is two questions: the conjunction below)
  const conjAt = s.search(CONJ);
  if (titles.length === 1 && HISTORY_Q.test(s) && !(conjAt > 0 && !relationOf(s.slice(0, conjAt)))) {
    // "was “T” reposted by @carol?": by carol, not by anyone
    const who = actorOf(s, A);
    const rel = relationOf(who ? who.clause : s);
    if (rel) {
      const steps: Step[] = [
        { id: "q1", op: "resolve_title", title: titles[0], text: `“${titles[0]}”`, label: titles[0] },
        { id: "q2", op: "history", relation: rel, target: { step: "q1", field: "post" }, refs: [{ step: "q1", field: "post" }], text: question, ...(who ? { actor: who.actor } : {}) },
      ];
      return { steps, final: "q2", source: "rules", pattern: "history_title", titles, extraEntities: who ? 1 : 0, notes: [] };
    }
  }

  // ---- history of the first or latest of something: "Was @matus's first artwork edited?", "When was the latest cat deleted?" --
  // (also with the subject first, as Italian, French and Spanish ask: "il primo gatto è mai stato ripubblicato?")
  if (!titles.length && (HISTORY_Q.test(s) || SUBJECT_FIRST.test(s)) && !CONJ.test(s)) {
    // "was the first cat reposted by @carol?": the first cat, reposted by carol ("the first cat by @carol" is carol's)
    const who = actorOf(s, A);
    const clause = who ? who.clause : s;
    const rel = relationOf(clause);
    if (rel && FIRST_LAST.test(clause) && (HISTORY_Q.test(clause) || (SUBJECT_FIRST.test(clause) && RELATION_END.test(clause)))) {
      const event = clause
        .replace(HISTORY_Q, " ")
        .replace(RELATION_WORDS, " ")
        .replace(HISTORY_AUX, " ")
        .replace(/^\s*(?:a|ha|hat|has|have|did|does)\s+/, " ")
        .replace(/(?:\s+(?:e|a|ha|hanno|est|sont|fue|fueron|ist|wurde|was|is|has|a t il|a t elle|ont ils|ont elles|se))+\s*$/, "")
        .replace(/\s+/g, " ")
        .trim();
      const ev = eventStep(event, "q1", titles, ctx);
      if (ev && ev.op === "ask") {
        return {
          steps: [ev, { id: "q2", op: "history", relation: rel, target: { step: "q1", field: "post" }, refs: [{ step: "q1", field: "post" }], text: question, ...(who ? { actor: who.actor } : {}) }],
          final: "q2",
          source: "rules",
          pattern: "history_event",
          titles,
          extraEntities: 0,
          notes: [],
        };
      }
    }
  }

  // ---- the author of the top artwork: "who posted the most liked artwork?", "qui a posté l'œuvre la plus aimée ?" ----
  {
    const forms: Array<[RegExp, (m: RegExpExecArray) => [string, string]]> = [
      [/^(?:who|which (?:artist|author|user|account|person)) (?:has |had |posted |made |created |published |drew |owns |is behind )?(?:the )?(?:most|best|top|highest) (liked|voted|upvoted|popular|loved|rewarded|paid|rated) (.+)$/, (m) => [m[1], m[2]]],
      [/^qui a (?:poste|publie|cree|fait|dessine|partage) (?:l |la |le |les )(.+?) (?:le |la |les )?plus (aime\w*|vote\w*|populaire\w*|recompense\w*|paye\w*)$/, (m) => [m[2], m[1]]],
      [/^wer hat (?:das|den|die) (beliebteste\w*|meistgelikt\w*|popularste\w*|meistbelohnt\w*) (.+?)(?: gepostet| veroffentlicht| erstellt| gemacht| gezeichnet| hochgeladen)?$/, (m) => [m[1], m[2]]],
      [/^quien (?:publico|subio|creo|hizo|dibujo|ha publicado|ha subido|tiene) (?:la |el |los |las )(.+?) mas (votad\w*|popular\w*|gustad\w*|premiad\w*|pagad\w*)$/, (m) => [m[2], m[1]]],
      [/^chi ha (?:pubblicato|postato|creato|fatto|disegnato|caricato) (?:l |la |il |lo |le |i |gli )(.+?) piu (votat\w*|popolar\w*|amat\w*|apprezzat\w*|premiat\w*|pagat\w*)$/, (m) => [m[2], m[1]]],
    ];
    for (const [re, pick] of forms) {
      const w = re.exec(s);
      if (!w) continue;
      const [word, subject] = pick(w);
      const payout = /^(?:rewarded|paid|recompense|paye|meistbelohnt|premiad|pagad|premiat|pagat)/.test(word);
      const q = plan(`which ${subject} has the ${payout ? "highest payout" : "most likes"}?`, ctx);
      if (q.intent === "top") return { steps: [{ id: "q1", op: "ask", plan: { ...q, output: "author" }, text: question }], final: "q1", source: "rules", pattern: "top_author", titles, extraEntities: 0, notes: [] };
    }
  }

  // ---- the most prolific: "which artist has posted the most artworks?", "who has the most cat artworks?" -----
  // (and "who has the most votes?": the author whose artworks have the most votes in total)
  {
    const GOT = String.raw`(?:posted |made |drawn |created |published |uploaded |shared |received |earned |got |gotten |collected )`;
    const m =
      new RegExp(String.raw`^(?:which|what) (?:artist|author|user|creator|account|person|member)s? (?:has |have |had )?${GOT}?(?:the )?most (.*)$`).exec(s) ??
      new RegExp(String.raw`^who (?:has|have|had) ${GOT}?(?:the )?most (.*)$`).exec(s) ??
      // ("who posted the most artworks?" stays v3's own count by author)
      /^who (?:received|earned|got|gets|collected) (?:the )?most ?(.*)$/.exec(s) ??
      /^(?:quel|quelle) (?:artiste|auteur|utilisateur|membre|compte) a (?:poste |publie |cree |dessine |partage |recu |obtenu |gagne )?le plus (?:de |d )?(.*)$/.exec(s) ??
      /^qui a (?:poste |publie |cree |dessine |partage |recu |obtenu |gagne )?le plus (?:de |d )?(.*)$/.exec(s) ??
      /^welche[rs]? (?:kunstler(?:in)?|autor(?:in)?|nutzer(?:in)?|mitglied|konto) hat (?:die meisten|am meisten) ?(.*?)(?: gepostet| veroffentlicht| erstellt| gemacht| gezeichnet| bekommen| erhalten| gesammelt)?$/.exec(s) ??
      /^wer hat (?:die meisten|am meisten) ?(.*?)(?: gepostet| veroffentlicht| erstellt| gemacht| gezeichnet| bekommen| erhalten| gesammelt)?$/.exec(s) ??
      /^(?:que|cual) (?:artista|autor|usuario|miembro|cuenta) (?:ha |tiene )?(?:publicado |publico |subido |creado |dibujado |recibido |recibio |ganado |gano )?(?:mas|la mayor cantidad de) (.*)$/.exec(s) ??
      /^quien (?:ha |tiene )?(?:publicado |publico |subido |creado |dibujado |recibido |recibio |ganado |gano )?(?:mas|la mayor cantidad de) (.*)$/.exec(s) ??
      /^(?:quale|che) (?:artista|autore|utente|membro|account) ha (?:pubblicato |postato |creato |disegnato |ricevuto |ottenuto |guadagnato )?(?:piu|il maggior numero di) (.*)$/.exec(s) ??
      /^chi ha (?:pubblicato |postato |creato |disegnato |ricevuto |ottenuto |guadagnato )?(?:piu|il maggior numero di) (.*)$/.exec(s);
    const what = m?.[1] ?? "";
    // "who has the most votes (in total)?": a total per author; "who earned the most?": payouts
    const earned = !what && /^(?:who|which \S+) (?:has |have |had )?(?:earned|earns|gagne|verdient|gano|ganado|guadagnato)\b/.test(s);
    const metric = METRIC_VOTES.test(what) ? "net_votes" : METRIC_PAYOUT.test(what) || earned ? "payout" : null;
    if (m && metric && !what.replace(METRIC_WORDS, " ").replace(/\b(?:overall|altogether|insgesamt|au total|en total|in totale)\b/g, " ").trim() && !/\b(?:than|que|als|di|de) @/.test(s)) {
      const all = plan("how many artworks?", ctx);
      return { steps: [{ id: "q1", op: "group", plan: { ...all, intent: "count" }, by: "author", metric, text: question }], final: "q1", source: "rules", pattern: "group_author_total", titles, extraEntities: 0, notes: [] };
    }
    // "who has the most recent cat?", "who has the most liked artwork?", "chi ha pubblicato più di recente?": not a number of posts
    const notACount = m && (NOT_A_COUNT.test(what) || metric);
    if (m && !notACount && !/\b(?:than|que|als|di|de) @/.test(s)) {
      const subject = what.replace(/\b(?:artworks?|art|posts?|images?|pictures?|drawings?|oeuvres?|creations?|dessins?|kunstwerke?|bilder|obras?|imagenes|opere|immagini)\b/g, " ").replace(/\s+/g, " ").trim();
      const q = plan(`how many ${subject || "artworks"}?`, ctx);
      return { steps: [{ id: "q1", op: "group", plan: { ...q, intent: "count" }, by: "author", text: question }], final: "q1", source: "rules", pattern: "group_author", titles, extraEntities: 0, notes: [] };
    }
  }

  // ---- why did @a post the first X? (the premise is checked, the motive is not in the index) ---------
  {
    const m = new RegExp(String.raw`^(?:why|pourquoi|warum|wieso|por que|perche) (?:did|has|a|hat|ha) ${AUTHOR} (?:t il |t elle )?${POSTED} (.+)$`).exec(s);
    // "the first artwork" is Pixagram's first; "her first artwork" is the author's own, not a premise
    if (m && A.has(m[1]) && /\b(first|earliest|premier|premiere|erste[nrsm]?|primer\w*|prim[oa])\b/.test(m[2]) && !/\b(her|his|their|its|sa|son|ses|leur|ihr\w*|sein\w*|su|sus|suo|sua|suoi|sue)\b/.test(m[2])) {
      const q = plan(`who posted ${m[2]}`, ctx);
      if (q.intent === "find_first") {
        return { steps: [{ id: "q1", op: "premise", plan: q, claimed: m[1], text: question }], final: "q1", source: "rules", pattern: "premise", titles, extraEntities: 1, notes: [] };
      }
    }
  }

  // ---- comparisons between two named authors ----------------------------------------------------------
  {
    // "did @a post more X than @b", "has @a posted more X than @b", "@a a posté plus de X que @b"
    const m =
      new RegExp(String.raw`^(?:did |does |do |has |have |est ce que |hat |ha |publico |posteo |subio |creo |ha publicado |ha subido |tiene )?${AUTHOR} (?:a t il |a t elle |a |hat |ha |have |has |got |get |gets |received |receive |receives |earned |earn |earns |gotten |tiene )?(?:${POSTED} )?(?:more|fewer|less|plus|davantage|moins|mehr|weniger|mas|menos|piu|meno) (?:de |d )?(.*?) ?(?:${POSTED} )?${THAN} ${AUTHOR}$`).exec(s) ??
      new RegExp(String.raw`^(?:did |has |have |hat |ha )?${AUTHOR} (?:${POSTED} )?(?:more|fewer|less|mehr|weniger|mas|menos|piu|meno) (.*?) (?:${POSTED} )?${THAN} ${AUTHOR}(?: ${POSTED})?$`).exec(s);
    if (m && A.has(m[1]) && A.has(m[3]) && m[1] !== m[3]) {
      let subj = m[2].replace(new RegExp(`\\b${POSTED}\\b`, "g"), " ").replace(/\s+/g, " ").trim();
      // "did @a earn more than @b?": payouts; "did @a get more than @b?" says not what
      const verb = new RegExp(String.raw`^(?:did |does |do |has |have |est ce que |hat |ha |tiene )?${AUTHOR} (?:a t il |a t elle |a |hat |ha )?(\S+)`).exec(s)?.[2] ?? "";
      const earned = /^(?:earn|earns|earned|gagne|verdient|verdiente|gano|ganado|guadagna|guadagnato)$/.test(verb);
      const got = /^(?:get|gets|got|gotten|receive|receives|received)$/.test(verb);
      if (!subj && earned) subj = "payout";
      const c = !subj && got ? null : comparison(m[1], m[3], subj, LESS.test(s) ? "less" : "more", true, ctx, titles, question);
      if (c) return c;
    }
    // "is @a more prolific than @b?": more artworks
    {
      const pro = new RegExp(String.raw`^(?:is|was|est ce que|ist|es|e) ${AUTHOR} (?:more prolific|more active|plus prolifique|plus actif|plus active|produktiver|aktiver|mas prolific[oa]|mas activ[oa]|piu prolific[oa]|piu attiv[oa]) (?:than|que|als|di) ${AUTHOR}$`).exec(s);
      const c = pro && A.has(pro[1]) && A.has(pro[2]) && pro[1] !== pro[2] ? comparison(pro[1], pro[2], "artworks", "more", true, ctx, titles, question) : null;
      if (c) return c;
    }
    // "who posted more X, @a or @b", "@a or @b: who posted more X", "which of @a and @b posted more X"
    const MOREW = String.raw`(?:the )?(?:more|most|fewer|less|plus|le plus|moins|mehr|die meisten|weniger|mas|menos|piu|meno)`;
    const forms: Array<[RegExp, (m: RegExpExecArray) => [string, string, string]]> = [
      [new RegExp(String.raw`^${WHO} (?:has |have |a |hat |ha )?(?:${POSTED} )?${MOREW} (?:de |d )?(.*?) ?${AUTHOR} ${OR} ${AUTHOR}$`), (m) => [m[2], m[3], m[1]]],
      [new RegExp(String.raw`^${AUTHOR} ${OR} ${AUTHOR} ${WHO} (?:has |have |a |hat |ha )?(?:${POSTED} )?${MOREW} ?(?:de |d )?(.*)$`), (m) => [m[1], m[2], m[3]]],
      [new RegExp(String.raw`^(?:between|entre|zwischen|tra|fra) ${AUTHOR} (?:and|et|und|y|e) ${AUTHOR} ${WHO} (?:has |have |a |hat |ha )?(?:${POSTED} )?${MOREW} ?(?:de |d )?(.*)$`), (m) => [m[1], m[2], m[3]]],
      [new RegExp(String.raw`^(?:which|who) of ${AUTHOR} (?:and|et|und|y|e) ${AUTHOR} (?:has |have )?(?:${POSTED} )?${MOREW} (.*)$`), (m) => [m[1], m[2], m[3]]],
    ];
    for (const [re, pick] of forms) {
      const w = re.exec(s);
      if (!w) continue;
      const [a, b, subj] = pick(w);
      const c = A.has(a) && A.has(b) && a !== b ? comparison(a, b, (subj ?? "").replace(new RegExp(`\\b${POSTED}\\b`, "g"), " ").replace(/\s+/g, " ").trim(), LESS.test(s) ? "less" : "more", false, ctx, titles, question) : null;
      if (c) return c;
    }
  }

  // ---- two posts by votes or payout: "which has more votes, “A” or “B”?", "does “A” have more likes than “B”?" ----
  if (titles.length === 2 && (MORE.test(s) || LESS.test(s)) && (METRIC_VOTES.test(s) || METRIC_PAYOUT.test(s))) {
    const metric = METRIC_PAYOUT.test(s) && !METRIC_VOTES.test(s) ? "payout" : "net_votes";
    const yesNo = /^(?:does|did|has|is|a t il|est ce que|hat|ist|tiene|ha)\b/.test(s);
    const steps: Step[] = [
      { id: "q1", op: "resolve_title", title: titles[0], label: titles[0] },
      { id: "q2", op: "resolve_title", title: titles[1], label: titles[1] },
      { id: "q3", op: "compare_metric", a: "q1", b: "q2", metric, yesNo, refs: [{ step: "q1", field: "post" }, { step: "q2", field: "post" }], text: question },
    ];
    return { steps, final: "q3", source: "rules", pattern: "compare_metric", titles, extraEntities: 0, notes: [] };
  }

  // ---- duration: "how long after A was B?", "how many days between A and B?" --------------------------
  {
    const m =
      new RegExp(String.raw`^(?:${HOW_MANY_UNITS}) (?:separate|lie between|are there between|separent|liegen zwischen|separan|separano) (.+?) (?:and|from|et|und|y|e|da) (.+)$`).exec(s) ??
      new RegExp(String.raw`^(?:how long|how much time|combien de temps|wie lange|wie viel zeit|cuanto tiempo|quanto tempo|${HOW_MANY_UNITS})(?: has| have| had| did)?(?: passed| elapsed| went by| was there| is there| se sont ecoules| sont passes| s est ecoule| vergingen| verging| liegen| lag| pasaron| paso| sono passati| e passato)? (?:between|entre|zwischen|tra|fra) (.+?) (?:and|et|und|y|e) (.+)$`).exec(s) ??
      /^(?:how long|combien de temps|wie lange|cuanto tiempo|quanto tempo) (?:after|apres|nach|despues de|dopo) (.+?) (?:was|were|did|came|a ete|est|wurde|kam|fue|llego|e stato|e arrivato) (.+?)(?: posted| published| created| poste| publie| gepostet| publicado| pubblicato)?$/.exec(s) ??
      // "how much later than A did B come out?", "how many days after A was B posted?"
      new RegExp(String.raw`^(?:how much|how long|${HOW_MANY_UNITS}) (?:later than|earlier than|after|before) (.+?) (?:was|were|did|is|came) (.+?)(?: (?:come out|came out|go up|get posted|posted|published|created|uploaded|appear|appeared|released))?$`).exec(s);
    if (m) {
      const a = eventStep(m[1], "q1", titles, ctx);
      const b = eventStep(m[2], "q2", titles, ctx);
      if (a && b) {
        return {
          steps: [a, b, { id: "q3", op: "duration", a: "q1", b: "q2", refs: [{ step: "q1", field: "time" }, { step: "q2", field: "time" }], text: question, ...(unitOf(s) ? { unit: unitOf(s) } : {}) }],
          final: "q3",
          source: "rules",
          pattern: "duration",
          titles,
          extraEntities: 0,
          notes: [],
        };
      }
    }
  }

  // ---- sequence: "which came first, A or B?", "was A posted before B?" --------------------------------------
  {
    const which =
      /^(?:which|what|lequel|laquelle|welche\w*|cual|quale) (?:was|came|appeared|est|a ete|kam|erschien|wurde|fue|llego|e|e venuto)? ?(?:posted |published |created |poste |publie |gepostet |publicado |pubblicato )?(?:first|earlier|en premier|d abord|zuerst|als erstes|primero|antes|prima)(?: posted| published)? (.+?) (?:or|ou|oder|o) (.+)$/.exec(s) ??
      /^(.+?) (?:or|ou|oder|o) (.+?) (?:which|what|lequel|welche\w*|cual|quale) (?:was|came|est|kam|fue|e)? ?(?:posted |published )?(?:first|earlier|en premier|zuerst|primero|prima)$/.exec(s);
    const was = /^(?:was|were|did|does|is|est ce que|a t il|wurde|kam|ist|fue|e stato) (.+?) (?:posted |published |created |appear\w* |come |come out |came out |go up |released |poste\w* |publie\w* |gepostet |publicad\w* |pubblicat\w* )?(before|after|earlier than|later than|avant|apres|vor|nach|antes de|antes que|despues de|prima di|dopo) (.+?)(?: posted| published| was posted| poste\w*| publie\w*| gepostet)?$/.exec(s);
    // "«A» è stato pubblicato prima di «B»?", "est-ce que « A » a été posté avant « B » ?", "¿«A» se publicó antes que «B»?", "wurde „A“ vor „B“ gepostet?"
    const was2 = was
      ? null
      : (/^(?:est ce que )?(.+?) (?:a t il |a t elle |est il |est elle |a |e |est |fue |se |was |is )?(?:ete |stato |stata |sido )?(?:posted|published|created|poste\w*|publie\w*|sorti\w*|paru\w*|apparu\w*|mis en ligne|publicad\w*|publico|salio|pubblicat\w*|postat\w*|creat\w*|uscit[oa]) (before|after|avant|apres|antes de|antes que|despues de|prima di|dopo) (.+)$/.exec(s) ??
        /^(?:se publico|fue publicad[oa]|fue subid[oa]|fue) (.+?) (antes de|antes que|despues de) (.+)$/.exec(s) ??
        /^(?:wurde|ist|kam|war) (.+?) (vor|nach|fruher als|spater als|eher als) (.+?) (?:gepostet|veroffentlicht|erstellt|hochgeladen|erschienen|gekommen|online gegangen)(?: worden)?$/.exec(s));
    const yes = was ?? was2;
    // the event phrases without the auxiliaries around them ("« A » a été" → « A »)
    const bare = (x: string) =>
      x
        .replace(/^(?:(?:a|ete|e|est|stato|stata|wurde|fue|se|was|is|been|ha|sido|ist|hat)\s+)+/, "")
        .replace(/(?:\s+(?:come out|came out|appear\w*|go up|went up|get posted|got posted|released|release|published|posted|uploaded|gepostet|veroffentlicht|erstellt|hochgeladen|erschienen|gekommen|a|ete|e|est|stato|stata|wurde|fue|se|was|is|been|ha|sido|ist|hat))+$/, "")
        .trim();
    const pair = which ? [bare(which[1]), bare(which[2])] : yes ? [bare(yes[1]), bare(yes[3])] : null;
    if (pair) {
      const a = eventStep(pair[0], "q1", titles, ctx);
      const b = eventStep(pair[1], "q2", titles, ctx);
      if (a && b) {
        const yesNo = yes ? (/^(before|earlier than|avant|vor|fruher als|eher als|antes|prima)/.test(yes[2]) ? "before" : "after") : undefined;
        return {
          steps: [a, b, { id: "q3", op: "sequence", a: "q1", b: "q2", want: yesNo, yesNo: !!yesNo, refs: [{ step: "q1", field: "time" }, { step: "q2", field: "time" }], text: question }],
          final: "q3",
          source: "rules",
          pattern: "sequence",
          titles,
          extraEntities: 0,
          notes: [],
        };
      }
    }
  }

  // ---- after / before an event: "how many artworks were posted after the first cat?" ----------------------
  {
    const m = /^(.+?) (?:posted |published |created |poste\w* |publie\w* |gepostet |publicad\w* |pubblicat\w* )?(after|before|since|apres|avant|depuis|nach|vor|seit|despues de|antes de|desde|dopo|prima di|da) (the (?:first|earliest|last|latest) .+|(?:le |la )?(?:premier|premiere|dernier|derniere) .+|(?:dem |der |das )?(?:erste[nrsm]?|letzte[nrsm]?) .+|(?:el |la )?(?:primer\w*|ultim[oaie]s?) .+|(?:il |la )?(?:prim[oa]|ultim[oa]) .+|__t\d+__)$/.exec(s);
    if (m && /\b(how many|number|count|which|what|who|combien|quel|qui|wie viele|welche|wer|cuant|cual|quien|quant|quale|chi)\b/.test(m[1])) {
      const ev = eventStep(m[3], "q1", titles, ctx);
      if (ev) {
        const after = /^(after|since|apres|depuis|nach|seit|despues|desde|dopo|da)/.test(m[2]);
        const q = plan(clauseText(m[1], titles), ctx);
        return {
          steps: [ev, { id: "q2", op: "ask", plan: q, text: m[1], inject: after ? { from: { step: "q1", field: "time" } } : { to: { step: "q1", field: "time" } }, refs: [{ step: "q1", field: "time" }] }],
          final: "q2",
          source: "rules",
          pattern: "relative_time",
          titles,
          extraEntities: 0,
          notes: [],
        };
      }
    }
  }

  // ---- nested reference: "… by the author of “T”", "… did the artist who posted the first X …" ------------------
  {
    const m = new RegExp(String.raw`\b(?:by |from |de |par |von |vom |del |di |da )?(?:the |l |le |la |der |die |el |il |lo )?(?:author|artist|creator|maker|person|user|auteur|artiste|createur|createur|kunstler|autor|urheber|artista|autore|creatore)(?:in)? (?:of|behind|de|du|des|von|vom|del|di) (__t\d+__)`).exec(s) ??
      new RegExp(String.raw`\b(?:by |from |de |par |von |del |di |da )?(?:the |l |le |la |der |die |el |il |lo )?(?:author|artist|creator|maker|person|user|one|auteur|artiste|personne|kunstler|autor|person|artista|autore|persona)(?:in)? (?:who|that|qui|der|die|que|che) (?:has |have |a |hat |ha )?(?:${POSTED}|drew|painted) (__t\d+__)`).exec(s) ??
      // "… of whoever made “T”", "… de celui qui a posté « T »", "… von dem, der „T“ gepostet hat"
      new RegExp(String.raw`\b(?:by |from |of |de |par |von |del |di |da )?(?:whoever|whomever|the one who|celui qui|celle qui|chiunque abbia|quien sea que) (?:has |have |had |a |hat |ha )?(?:${POSTED}|drew|painted) (__t\d+__)`).exec(s) ??
      new RegExp(String.raw`\b(?:by |from |de |par |von |del |di |da )?(?:the |l |le |la |der |die |el |il |lo )?(?:author|artist|creator|person|user|auteur|artiste|kunstler|autor|artista|autore)(?:in)? (?:who|that|qui|qu|der|die|que|che) (?:has |have |a |hat |ha )?${POSTED} (?:the |le |la |l |den |die |das |el |il )?((?:first|earliest|last|latest|premier|premiere|dernier|derniere|erste[nrsm]?|letzte[nrsm]?|neueste[nrsm]?|primer\w*|ultim[oaie]s?|prim[oa]) \S+(?: \S+)?)`).exec(s);
    if (m) {
      const title = titleOf(m[1], titles);
      const first: Step | null = title ? { id: "q1", op: "author_of", title, label: title, text: `the author of “${title}”` } : eventStep(m[1], "q1", titles, ctx);
      const rest = s.replace(m[0], " ").replace(/\s+/g, " ").trim();
      // "quel est l'auteur de « T » ?": nothing more is asked of the author than who it is
      if (first && first.op === "author_of" && /^(?:(?:who|what|which|qui|quel|quelle|wer|was|welche\w*|quien|cual|chi|qual|quale)(?: (?:is|s|was|est|etait|ist|war|es|fue|era|e))?(?: (?:the|l|le|la|der|die|das|el|il|lo))?)?$/.test(rest)) {
        return { steps: [first], final: "q1", source: "rules", pattern: "author_of", titles: [first.title ?? ""], extraEntities: 0, notes: [] };
      }
      if (first && rest && first.op !== "resolve_title") {
        const restPlan = plan(clauseText(rest, titles), ctx);
        if (first.op === "ask" && first.plan) first.plan = { ...first.plan, output: "author" };
        return {
          steps: [first, { id: "q2", op: "ask", plan: restPlan, text: rest, inject: { authors: { step: "q1", field: "author" } }, refs: [{ step: "q1", field: "author" }] }],
          final: "q2",
          source: "rules",
          pattern: "nested_author",
          titles,
          extraEntities: 1,
          notes: [],
        };
      }
    }
  }

  // ---- totals and averages: "how many votes did @a get in total?", "average payout of cat artworks" -------------
  {
    const metric = METRIC_PAYOUT.test(s) && !/\bhow many artworks\b/.test(s) ? "payout" : METRIC_VOTES.test(s) ? "net_votes" : null;
    const agg: Agg | null = /\b(average|on average|mean|moyenne|en moyenne|durchschnitt\w*|im schnitt|promedio|de media|media|in media)\b/.test(s)
      ? "avg"
      : /\b(total|in total|sum|altogether|au total|en tout|insgesamt|zusammen|en total|in totale|how much|combien de|wie viel)\b/.test(s) || /^(how many (votes|likes)|combien de (votes|j aime)|wie viele (stimmen|likes)|cuantos (votos|me gusta)|quanti (voti|mi piace))\b/.test(s)
        ? "sum"
        : /\b(highest|most|maximum|max|le plus|meiste\w*|maximal|mas alto|massimo)\b/.test(s) && metric
          ? null
          : null;
    if (metric && agg && !/\b(first|latest|premier|dernier|erste|letzte|primer|ultim|prim[oa])\b/.test(s)) {
      const scopeText = s
        .replace(METRIC_VOTES, " ")
        .replace(METRIC_PAYOUT, " ")
        .replace(/\b(how many|how much|what is|what s|whats|the|total|in total|sum of|sum|average|on average|mean|of|for|did|does|do|get|got|receive\w*|have|has|had|earn\w*|make|made|combien de|combien|au total|en moyenne|moyenne|quel est|le|la|les|de|des|du|a|ont|recu\w*|wie viele|wie viel|insgesamt|durchschnitt\w*|im|schnitt|hat|haben|bekommen|erhalten|cuantos|cuantas|cuanto|en total|promedio|de media|tiene|tienen|recibio|quanti|quante|quanto|in totale|in media|media|ha|hanno|ricevuto)\b/g, " ")
        .replace(/\s+/g, " ")
        .trim();
      const q = plan(`how many ${scopeText || "artworks"}?`, ctx);
      return { steps: [{ id: "q1", op: "aggregate", plan: { ...q, intent: "count" }, metric, agg, text: question }], final: "q1", source: "rules", pattern: "aggregate", titles, extraEntities: 0, notes: [] };
    }
  }

  // ---- groups: "which month had the most cat artworks?", "how many artworks per month?" --------------------------
  {
    const month = /\b(which|what|in which|quel|quelle|en quel|welche\w*|in welchem|cual|en que|quale|in quale) (month|mois|monat|mes|mese)\b/.test(s) || /\b(per|each|by|par|pro|je|por|ogni|al) (month|mois|monat|mes|mese)\b/.test(s);
    if (month) {
      const scopeText = s.replace(/\b(busiest|most active|most productive|le plus actif|le plus productif|aktivste\w*|mas activ[oa]|piu attiv[oa]|which|what|in|during|quel|quelle|en|pendant|durant|welche\w*|in welchem|wahrend|cual|que|durante|quale|month|mois|monat|mes|mese|had|has|saw|have|did|do|does|the|most|le plus|de|die meisten|meisten|hatte|tuvo|mas|ha avuto|piu|per|each|by|par|pro|je|por|ogni|al|how many|combien|wie viele|cuantos|cuantas|quanti|quante|were|was|posted|post|publish|published|upload|uploaded|poste\w*|publie\w*|gepostet|veroffentlicht|publicad\w*|publico|pubblicat\w*|il y a eu|y a t il eu|gab es|hubo|ci sono stati)\b/g, " ").replace(/\s+/g, " ").trim();
      const q = plan(`how many ${scopeText || "artworks"}?`, ctx);
      return { steps: [{ id: "q1", op: "group", plan: { ...q, intent: "count" }, by: "month", text: question }], final: "q1", source: "rules", pattern: "group_month", titles, extraEntities: 0, notes: [] };
    }
  }

  // ---- what someone's art is about: "what kind of art does @a make?", "describe @a's artworks" --------------------
  {
    const KIND = String.raw`(?:kind|kinds|type|types|sort|sorts|style|styles)`;
    const ART = String.raw`(?:art|artworks?|things?|pictures?|images?|stuff|work)`;
    const m =
      new RegExp(String.raw`^what ${KIND} of ${ART} (?:does|did|do) ${AUTHOR} (?:make|post|draw|create|paint|do|publish)$`).exec(s) ??
      new RegExp(String.raw`^(?:what does|what did) ${AUTHOR} (?:usually |mostly )?(?:make|post|draw|create|paint|publish)$`).exec(s) ??
      new RegExp(String.raw`^(?:describe|summari[sz]e|tell me about) ${AUTHOR}(?: s)? (?:art|artworks?|work|style|posts?)$`).exec(s) ??
      new RegExp(String.raw`^(?:quel|quelle|quels|quelles) (?:genre|type|types|style|sorte) d (?:art|oeuvres?|images?) (?:fait|poste|dessine|cree|publie) ${AUTHOR}$`).exec(s) ??
      new RegExp(String.raw`^(?:was fur|welche) (?:kunst|bilder|kunstwerke|art von (?:kunst|bildern)) (?:macht|postet|zeichnet|erstellt) ${AUTHOR}$`).exec(s) ??
      new RegExp(String.raw`^que (?:tipo|clase) de (?:arte|obras?|imagenes) (?:hace|publica|dibuja|crea) ${AUTHOR}$`).exec(s) ??
      new RegExp(String.raw`^che (?:tipo|genere) di (?:arte|opere|immagini) (?:fa|pubblica|disegna|crea) ${AUTHOR}$`).exec(s);
    if (m && A.has(m[1])) {
      const q = plan(`how many artworks by @${m[1]}?`, ctx);
      const p: QueryPlan = { ...q, intent: "count", output: "count", residual: "", concepts: [], conceptMatches: [], lexicalTerms: [], filters: { ...q.filters, authors: [m[1]] } };
      return { steps: [{ id: "q1", op: "group", plan: p, by: "tag", text: question }], final: "q1", source: "rules", pattern: "group_tags", titles, extraEntities: 0, notes: [] };
    }
  }

  // ---- existence: "did @a post a dragon?", "has @a ever posted a cat?", "are there any cats by @a?" ------------------
  {
    const m =
      new RegExp(String.raw`^(?:did|has|have) ${AUTHOR} (?:ever |already )?(?:${POSTED}) (?:a |an |any |some )?(.+)$`).exec(s) ??
      new RegExp(String.raw`^${AUTHOR} (?:a t il|a t elle) (?:deja |jamais )?(?:${POSTED}) (?:un |une |des |de |d )?(.+)$`).exec(s) ??
      new RegExp(String.raw`^(?:est ce qu |est ce que )${AUTHOR} a (?:deja |jamais )?(?:${POSTED}) (?:un |une |des |de |d )?(.+)$`).exec(s) ??
      new RegExp(String.raw`^hat ${AUTHOR} (?:schon |jemals |je )?(?:ein |eine |einen |irgendein\w* )?(.+?) (?:gepostet|veroffentlicht|gezeichnet|erstellt)$`).exec(s) ??
      new RegExp(String.raw`^(?:ha publicado|publico) ${AUTHOR} (?:algun |alguna |un |una )?(.+)$`).exec(s) ??
      new RegExp(String.raw`^${AUTHOR} ha (?:mai |gia )?(?:pubblicato|creato|disegnato) (?:un |una |uno |qualche )?(.+)$`).exec(s);
    if (m && A.has(m[1]) && m[2] && !/\b(first|latest|most|more|premier|dernier|plus|davantage|erste|letzte|mehr|primer|ultim|mas|prim[oa]|piu)\b/.test(m[2]) && !/\b(?:than|que|als|di|de) @/.test(s)) {
      const q = plan(`how many ${m[2]}?`, ctx);
      if (q.residual || q.concepts.length || q.filters.colors?.length || q.filters.tones?.length) {
        const p: QueryPlan = { ...q, intent: "count", output: "count", filters: { ...q.filters, authors: [m[1]] } };
        return { steps: [{ id: "q1", op: "exists", plan: p, claimed: m[1], text: question }], final: "q1", source: "rules", pattern: "exists", titles, extraEntities: 1, notes: [] };
      }
    }
  }

  // ---- two questions in one: "who posted the first cat and did they later repost it?" -----------------------
  {
    const k = s.search(CONJ);
    if (k > 0) {
      const left = s.slice(0, k).trim();
      const right = s.slice(k).replace(/^\s(?:and|et|und|y|e)\s/, "").trim();
      const lp = plan(clauseText(left, titles), ctx);
      // "who posted “T” and was it edited?": the title's own step (v3's planner reads a quoted title as a subject)
      const leftRules = /__t\d+__/.test(left) ? decomposeRules(clauseText(left, titles), lp, ctx) : null;
      const titleStep: Step | null = leftRules && leftRules.steps.length === 1 && (leftRules.steps[0].op === "author_of" || leftRules.steps[0].op === "identify") ? { ...leftRules.steps[0], id: "q1", text: left } : null;
      // the first question alone, when the second cannot be read (never v3's reading of the title as a subject)
      const leftOnly = (): QueryProgram | null => (titleStep && leftRules ? { ...leftRules, steps: [titleStep] } : null);
      const lhasIntent = !!titleStep || lp.intent !== "search" || lp.output === "author" || lp.output === "date";
      if (lhasIntent && right) {
        const steps: Step[] = [titleStep ?? { id: "q1", op: "ask", plan: lp, text: left }];
        const person = PERSON.test(right);
        const thing = THING.test(right) && !person;
        const rel = relationOf(right);
        // "who posted “T”? was it @alice?", "… i think it was @alice": a check of the first answer
        const claim = new RegExp(String.raw`^(?:(?:was|is) it|it was|(?:i think|i guess|maybe|perhaps|probably) it was|c etait|etait ce|est ce|war es|ist es|fue|era|e stato) (?:by |posted by |made by |par |von |por |da )?(@?)([a-z0-9][a-z0-9.-]{1,31})$`).exec(right);
        if (claim && titleStep?.op === "author_of" && !titleStep.claimed && !titleStep.field && (claim[1] || (A.has(claim[2]) && !BY_IDIOM.test(claim[2])))) {
          return { steps: [{ ...titleStep, claimed: claim[2], text: question }], final: "q1", source: "rules", pattern: "author_check", titles, extraEntities: 1, notes: [] };
        }
        // a second question about another title: asked on its own ("who posted “A”? who posted “B”?")
        const rightRules = /__t\d+__/.test(right) && !/\b(?:it|this|that|they|them|their)\b/.test(right) ? decomposeRules(clauseText(right, titles), plan(clauseText(right, titles), ctx), ctx) : null;
        if (rightRules && rightRules.pattern !== "single") {
          if (rightRules.steps.length !== 1) return leftOnly() ?? single(base, titles);
          steps.push({ ...rightRules.steps[0], id: "q2", text: right });
        } else if (rel) {
          // "did they post it again": by the same author; "was it reposted": by anyone
          steps.push({ id: "q2", op: "history", relation: rel, target: { step: "q1", field: "post" }, byAuthor: person ? { step: "q1", field: "author" } : undefined, refs: [{ step: "q1", field: "post" }], text: right });
        } else if (/^(?:when|quand|wann|cuando|quando)(?:\s+(?:was|it|did|they|il|elle|a|t|ete|wurde|es|fue|e|stato|stata|ha|been|is|se))*(?:\s+(?:posted|published|made|created|uploaded|poste\w*|publie\w*|gepostet|veroffentlicht|publicad\w*|publico|pubblicat\w*))?$/.test(right)) {
          steps.push({ id: "q2", op: "fact", target: { step: "q1", field: "post" }, field: "time", refs: [{ step: "q1", field: "time" }], text: right });
        } else if (/^(?:by whom|(?:who|qui|wer|quien|chi)(?: (?:posted|made|created|a poste|hat|publico|ha pubblicato)(?: it| l| es| lo)?)?)$/.test(right)) {
          steps.push({ id: "q2", op: "fact", target: { step: "q1", field: "post" }, field: "author", refs: [{ step: "q1", field: "author" }], text: right });
        } else {
          const rp = plan(clauseText(right.replace(PERSON, " ").replace(/\s+/g, " "), titles), ctx);
          const usesAuthor = person || (!rp.filters.authors?.length && /\b(their|leur|leurs|ihre\w*|sus?|loro)\b/.test(right));
          if (rp.intent === "search" && !rp.residual && !usesAuthor) return leftOnly() ?? fallback(base, titles, question);
          steps.push({ id: "q2", op: "ask", plan: rp, text: right, ...(usesAuthor ? { inject: { authors: { step: "q1", field: "author" } }, refs: [{ step: "q1", field: "author" }] } : {}) });
          if (thing && !usesAuthor) steps[1].refs = [{ step: "q1", field: "post" }];
        }
        return { steps, final: steps[steps.length - 1].id, source: "rules", pattern: "conjunction", titles, extraEntities: 0, notes: [] };
      }
      const only = leftOnly();
      if (only) return only;
    }
  }

  return fallback(base, titles, question);
}

/**
 * The last reading, v3's single intent — except that a question about one quoted title which v3
 * reads as "the first artwork about those words" ("Who exactly posted “Lake”?", "When was “Lake”
 * posted on Pixagram?") is about the post with that title; if no post has it, the executor reads
 * the words as a subject after all (`orElse`).
 */
function fallback(base: QueryPlan, titles: string[], question: string): QueryProgram {
  const first = base.intent === "find_first" || base.intent === "find_last";
  if (titles.length === 1 && first && (base.output === "author" || base.output === "date")) {
    const said = new Set(fold(base.residual).replace(/["“”«»„‘’]/g, " ").split(/\s+/).filter(Boolean));
    const words = fold(titles[0]).split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 1 && !isStopword(w));
    if (words.length && words.every((w) => said.has(w))) {
      return {
        steps: [{ id: "q1", op: "author_of", title: titles[0], ...(base.output === "date" ? { field: "time" as const } : {}), text: question, label: titles[0] }],
        final: "q1",
        source: "rules",
        pattern: "title_reading",
        titles,
        extraEntities: 0,
        notes: [],
        orElse: single(base, titles),
      };
    }
  }
  return single(base, titles);
}

/** "@a posts more often than @b": more artworks */
const OFTEN = /^(?:often|frequently|regularly|souvent|frequemment|regulierement|ofter|haufiger|regelmassiger|a menudo|frecuentemente|con frecuencia|con mas frecuencia|spesso|frequentemente|regolarmente|di frequente)$/;
const METRIC_WORDS = new RegExp(`${METRIC_VOTES.source}|${METRIC_PAYOUT.source}|\\b(?:in total|total|au total|insgesamt|en total|in totale|altogether|got|get|received|recu\\w*|bekommen|erhalten|recibid\\w*|ricevut\\w*|artworks?|posts?|oeuvres?|kunstwerke?|obras?|opere|de|d|of|von|di)\\b`, "g");

function comparison(a: string, b: string, subj: string, want: "more" | "less", yesNo: boolean, ctx: PlannerContext, titles: string[], question: string): QueryProgram | null {
  let subject = subj.replace(/^(?:of |de |d |von |di )/, "").trim();
  if (OFTEN.test(subject)) subject = "";
  const withAuthor = (p: QueryPlan, author: string): QueryPlan => ({ ...p, intent: "count", output: "count", filters: { ...p.filters, authors: [author] } });
  // "does @a have more votes than @b?": the totals of their artworks, not a number of "vote" artworks
  if (METRIC_VOTES.test(subject) || METRIC_PAYOUT.test(subject)) {
    const metric = METRIC_VOTES.test(subject) ? "net_votes" : "payout";
    // a total over part of their work ("more votes on cats") is not read here
    if (subject.replace(METRIC_WORDS, " ").trim()) return null;
    const all = plan("how many artworks?", ctx);
    return {
      steps: [
        { id: "q1", op: "aggregate", plan: withAuthor(all, a), metric, agg: "sum", text: `${metric} of @${a}'s artworks` },
        { id: "q2", op: "aggregate", plan: withAuthor(all, b), metric, agg: "sum", text: `${metric} of @${b}'s artworks` },
        { id: "q3", op: "compare_counts", a: "q1", b: "q2", want, yesNo, claimed: a, metric, refs: [{ step: "q1", field: "author" }, { step: "q2", field: "author" }], text: question },
      ],
      final: "q3",
      source: "rules",
      pattern: yesNo ? "compare_totals_yes_no" : "compare_totals",
      titles,
      extraEntities: 2,
      notes: [],
    };
  }
  subject ||= "artworks";
  const qa = plan(`how many ${subject}?`, ctx);
  return {
    steps: [
      { id: "q1", op: "ask", plan: withAuthor(qa, a), text: `how many ${subject} by @${a}` },
      { id: "q2", op: "ask", plan: withAuthor(qa, b), text: `how many ${subject} by @${b}` },
      { id: "q3", op: "compare_counts", a: "q1", b: "q2", want, yesNo, claimed: a, refs: [{ step: "q1", field: "author" }, { step: "q2", field: "author" }], text: question },
    ],
    final: "q3",
    source: "rules",
    pattern: yesNo ? "compare_authors_yes_no" : "compare_authors",
    titles,
    extraEntities: 2,
    notes: [],
  };
}

/** Whether a question that the rules left in one step still reads like several (worth the planner model). */
export function looksMultiStep(question: string): boolean {
  const { s } = patternText(question);
  const families = [
    /\b(first|earliest|oldest|premier|erste|primer|prim[oa])\b/,
    /\b(last|latest|newest|dernier|letzte|neueste|ultim[oa])\b/,
    /\b(how many|number of|combien|wie viele|cuant|quant)\b/,
    /\b(most|le plus|meisten|mas|piu)\b/,
    REPOST,
    EDIT,
    DELETE,
    /\b(before|after|between|avant|apres|entre|vor|nach|zwischen|antes|despues|prima|dopo)\b/,
  ].filter((re) => re.test(s)).length;
  return families >= 2 || CONJ.test(s) || /\b(than|que|als)\b/.test(s);
}

// ---- the planner model's decomposition ------------------------------------------------------------------

const LLM_OPS = ["find_first", "find_latest", "count", "count_by_author", "top", "search", "author_of_title", "history", "compare_counts", "compare_metric", "sequence", "duration", "aggregate", "group"] as const;
type LlmOp = (typeof LLM_OPS)[number];

const PROGRAM_SCHEMA = {
  type: "object",
  required: ["steps"],
  properties: {
    steps: {
      type: "array",
      items: {
        type: "object",
        required: ["id", "op"],
        properties: {
          id: { type: "string", description: "q1, q2, …" },
          op: { type: "string", enum: LLM_OPS },
          subject_en: { type: "string", description: "what the artworks show, in English, a few words; empty if none" },
          title: { type: "string", description: "an exact post title the question names" },
          authors: { type: "array", items: { type: "string" } },
          author_from: { type: "string", description: "id of an earlier step whose answer is the author to use" },
          post_from: { type: "string", description: "id of an earlier step whose post this step is about" },
          after: { type: "string", description: "id of an earlier step: only posts after its date" },
          before: { type: "string", description: "id of an earlier step: only posts before its date" },
          colors: { type: "array", items: { type: "string", enum: COLOR_NAMES } },
          tones: { type: "array", items: { type: "string", enum: ["dark", "light", "greyscale", "monochrome", "colorful", "pastel", "high_contrast", "minimal"] } },
          output: { type: "string", enum: ["author", "date", "post", "count", "boolean"] },
          relation: { type: "string", enum: ["repost", "edit", "delete", "first_seen", "last_seen"] },
          a: { type: "string" },
          b: { type: "string" },
          want: { type: "string", enum: ["more", "less", "before", "after", "which"] },
          metric: { type: "string", enum: ["net_votes", "payout"] },
          agg: { type: "string", enum: ["sum", "avg", "max", "min"] },
          by: { type: "string", enum: ["month", "author"] },
        },
      },
    },
  },
};

export function programPrompt(question: string, today: string): string {
  return [
    "You split questions about artworks on Pixagram (a pixel-art social network on its own blockchain) into at most 6 steps.",
    "Each step is one operation: find_first, find_latest, count, count_by_author (who posted the most), top (most liked), search,",
    "author_of_title (who posted the post with this exact title), history (relation: repost, edit, delete, first_seen, last_seen of the post of post_from),",
    "compare_counts (a and b are two count steps), compare_metric (a and b are two author_of_title steps; metric net_votes or payout),",
    "sequence (which of steps a and b came first; want before/after for a yes/no question), duration (time between steps a and b),",
    "aggregate (metric and agg over the posts), group (by month or author).",
    "Later steps refer to earlier ones by id: author_from (use that step's author), post_from, after, before.",
    "subject_en: what must be depicted, in English, without words like image/artwork/first/posted. Never answer the question; only plan it.",
    `Dates relative to today (${today}). Answer with JSON only.`,
    `Question: ${question}`,
  ].join("\n");
}

const STEP_ID = /^q[1-6]$/;

/** The planner model's program, validated and turned into rule-planned steps. Null when unusable. */
export function programFromLlm(raw: unknown, question: string, ctx: PlannerContext): QueryProgram | null {
  const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : null;
  const items = Array.isArray(o?.steps) ? (o!.steps as any[]).slice(0, 6) : [];
  if (!items.length) return null;
  const seen = new Set<string>();
  const steps: Step[] = [];
  const titles: string[] = [];
  const earlier = (id: unknown): string | null => (typeof id === "string" && seen.has(id) ? id : null);
  const subject = (x: any) => (typeof x?.subject_en === "string" ? fold(x.subject_en).replace(/[^\p{L}\p{N}\s'-]/gu, " ").replace(/\s+/g, " ").trim().slice(0, 60) : "");
  for (const x of items) {
    const id = typeof x?.id === "string" && STEP_ID.test(x.id) && !seen.has(x.id) ? x.id : `q${steps.length + 1}`;
    const op = (LLM_OPS as readonly string[]).includes(x?.op) ? (x.op as LlmOp) : null;
    if (!op) return null;
    const authors = (Array.isArray(x.authors) ? x.authors : []).map((a: unknown) => String(a).toLowerCase().replace(/^@/, "")).filter((a: string) => ctx.authors.has(a));
    const colors = (Array.isArray(x.colors) ? x.colors : []).filter((c: unknown) => COLOR_NAMES.includes(String(c))).join(" ");
    const tones = (Array.isArray(x.tones) ? x.tones : []).map((t: unknown) => String(t).replace("high_contrast", "high contrast").replace("greyscale", "black and white")).join(" ");
    const subj = [colors, tones, subject(x)].filter(Boolean).join(" ");
    const by = authors.length ? ` by @${authors[0]}` : "";
    const inject: Step["inject"] = {};
    const refs: StepRef[] = [];
    const af = earlier(x.author_from);
    if (af) (inject.authors = { step: af, field: "author" }), refs.push(inject.authors);
    const aft = earlier(x.after);
    if (aft) (inject.from = { step: aft, field: "time" }), refs.push(inject.from);
    const bef = earlier(x.before);
    if (bef) (inject.to = { step: bef, field: "time" }), refs.push(inject.to);
    const canonical: Partial<Record<LlmOp, string>> = {
      find_first: `${x.output === "date" ? "when was the first" : x.output === "post" ? "what is the first" : "who posted the first"} ${subj || "artwork"}${by}`,
      find_latest: `${x.output === "date" ? "when was the latest" : x.output === "post" ? "what is the latest" : "who posted the latest"} ${subj || "artwork"}${by}`,
      count: `how many ${subj || "artworks"}${by}`,
      count_by_author: `who posted the most ${subj || "artworks"}`,
      top: `what is the most liked ${subj || "artwork"}${by}`,
      search: `${subj || "artworks"}${by}`,
      aggregate: `how many ${subj || "artworks"}${by}`,
      group: `how many ${subj || "artworks"}${by}`,
    };
    let step: Step;
    if (canonical[op]) {
      const p = plan(`${canonical[op]}?`, ctx);
      step = { id, op: op === "aggregate" ? "aggregate" : op === "group" ? "group" : "ask", plan: op === "aggregate" || op === "group" ? { ...p, intent: "count" } : p, text: canonical[op] };
      if (op === "aggregate") (step.metric = x.metric === "payout" ? "payout" : "net_votes"), (step.agg = ["sum", "avg", "max", "min"].includes(x.agg) ? x.agg : "sum");
      if (op === "group") step.by = x.by === "author" ? "author" : "month";
    } else if (op === "author_of_title") {
      const title = typeof x.title === "string" ? x.title.trim().slice(0, 120) : "";
      if (!title) return null;
      titles.push(title);
      step = { id, op: "author_of", title, label: title };
    } else if (op === "history") {
      const pf = earlier(x.post_from);
      if (!pf) return null;
      const rel = ["repost", "edit", "delete", "first_seen", "last_seen"].includes(x.relation) ? x.relation : "all";
      step = { id, op: "history", relation: rel, target: { step: pf, field: "post" }, byAuthor: rel === "repost" ? { step: pf, field: "author" } : undefined };
      refs.push({ step: pf, field: "post" });
    } else {
      const a = earlier(x.a);
      const b = earlier(x.b);
      if (!a || !b || a === b) return null;
      refs.push({ step: a, field: op === "sequence" || op === "duration" ? "time" : "post" }, { step: b, field: op === "sequence" || op === "duration" ? "time" : "post" });
      if (op === "compare_counts") step = { id, op, a, b, want: x.want === "less" ? "less" : "more", yesNo: false };
      else if (op === "compare_metric") step = { id, op, a, b, metric: x.metric === "payout" ? "payout" : "net_votes" };
      else if (op === "sequence") step = { id, op, a, b, want: x.want === "after" ? "after" : x.want === "before" ? "before" : undefined, yesNo: x.want === "before" || x.want === "after" };
      else step = { id, op: "duration", a, b };
    }
    if (Object.keys(inject).length) step.inject = inject;
    if (refs.length) step.refs = [...(step.refs ?? []), ...refs];
    seen.add(id);
    steps.push(step);
  }
  return { steps, final: steps[steps.length - 1].id, source: "llm", pattern: "planner_model", titles, extraEntities: 0, notes: ["decomposed by the planner model"] };
}

/** One call to the planner model for a program; null when it fails or its reply is unusable. */
export async function decomposeWithModel(env: Env, question: string, ctx: PlannerContext, notes: string[]): Promise<QueryProgram | null> {
  try {
    const r = await complete(env, {
      model: modelFor(env, "planner"),
      messages: [
        { role: "system", content: "You output only JSON that matches the schema." },
        { role: "user", content: programPrompt(question, new Date((ctx.now ?? nowSec()) * 1000).toISOString().slice(0, 10)) },
      ],
      json: { name: "question_program", schema: PROGRAM_SCHEMA },
      reasoning: "none",
      maxOutputTokens: 500,
      temperature: 0,
    });
    const p = programFromLlm(replyObject(r), question, ctx);
    if (!p) notes.push("the planner model's decomposition was unusable: one step used");
    return p;
  } catch (e) {
    notes.push(`planner model unavailable for decomposition: ${e instanceof Error ? e.message : String(e)}`);
    return null;
  }
}

// ---- the spec's view of a program (§10) ---------------------------------------------------------------------

export function subqueries(p: QueryProgram): Subquery[] {
  const out: Subquery[] = [];
  for (const s of p.steps) {
    const add = (type: Subquery["type"], x: { query?: string; operation?: string }) => out.push({ id: `${s.id}${out.filter((o) => o.step === s.id).length ? `.${out.filter((o) => o.step === s.id).length}` : ""}`, type, step: s.id, ...x });
    if (s.plan && (s.op === "ask" || s.op === "aggregate" || s.op === "group" || s.op === "exists" || s.op === "premise")) {
      const pl = s.plan;
      if (pl.residual) add("semantic", { query: pl.residual });
      if (pl.filters.colors?.length || pl.filters.tones?.length || pl.filters.background?.length) add("visual", { query: [...(pl.filters.colors ?? []), ...(pl.filters.tones ?? []), ...(pl.filters.background ?? [])].join(" / ") });
      if (pl.filters.authors?.length || pl.filters.tags?.length) add("filter", { query: [...(pl.filters.authors ?? []).map((a) => `@${a}`), ...(pl.filters.tags ?? []).map((t) => `#${t}`)].join(" ") });
      if (pl.filters.from !== undefined || pl.filters.to !== undefined) add("filter", { operation: "date_range" });
      if (pl.intent === "find_first") add("temporal", { operation: "first_seen" });
      else if (pl.intent === "find_last") add("temporal", { operation: "latest" });
      else if (pl.intent === "count" && s.op === "ask") add("aggregation", { operation: pl.countOf === "authors" ? "count_authors" : "count" });
      else if (pl.intent === "compare") add("aggregation", { operation: "count_by_author" });
      else if (pl.intent === "top") add("aggregation", { operation: `top_${pl.sort ?? "votes"}` });
      if (s.op === "aggregate") add("aggregation", { operation: `${s.agg}_${s.metric}` });
      if (s.op === "group") add("aggregation", { operation: `group_by_${s.by}` });
      if (s.op === "exists") add("aggregation", { operation: "exists" });
      if (s.op === "premise") add("temporal", { operation: "first_seen_check" });
    } else if (s.op === "resolve_title" || s.op === "author_of" || s.op === "identify") add("title", { query: s.title, operation: s.op });
    else if (s.op === "history") add("history", { operation: `${s.relation}_relationship` });
    else if (s.op === "fact") add("fact", { operation: s.field });
    else add("comparison", { operation: s.op });
  }
  return out;
}

/** The program for a question: the rules, else (when allowed and it looks like several steps) the planner model. */
export async function decompose(env: Env, question: string, base: QueryPlan, ctx: PlannerContext, opts: { allowModel: boolean; notes: string[] }): Promise<QueryProgram> {
  const rules = decomposeRules(question, base, ctx);
  if (rules.pattern !== "single" || !opts.allowModel || !looksMultiStep(question)) return rules;
  const llm = await decomposeWithModel(env, question, ctx, opts.notes);
  return llm && llm.steps.length > 1 ? llm : rules;
}
