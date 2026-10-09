// Concept matching over the multilingual vocabulary (vocab.ts).
//
// Text is folded and tokenised exactly like the vocabulary aliases, then matched longest-first
// ("space invader" wins over "space"), with a singular fallback ("dragons" → "dragon").

import { fold, isStopword, singular } from "../lib/text";
import { VOCAB_TSV } from "./vocab";

const LANGS = ["en", "fr", "de", "es", "it", "pt", "ja", "zh"] as const;
export type VocabLang = (typeof LANGS)[number];

export interface ConceptDef {
  id: string;
  parents: string[];
  aliases: Record<VocabLang, string[]>;
}

export interface ConceptMatch {
  concept: string;
  alias: string;
  /** token span in the matched text */
  start: number;
  end: number;
  langs: VocabLang[];
}

/** Tokens for matching: folded words; apostrophes split; hyphenated words kept whole. */
export function matchTokens(s: string): string[] {
  return fold(s)
    .split(/[^\p{L}\p{N}\-]+/u)
    .map((t) => t.replace(/^-+|-+$/g, ""))
    .filter(Boolean);
}

interface Index {
  defs: Map<string, ConceptDef>;
  aliases: Map<string, Array<{ concept: string; lang: VocabLang }>>;
  maxLen: number;
  cjkAliases: Array<{ alias: string; concept: string; lang: VocabLang }>;
}

let INDEX: Index | null = null;

function register(map: Index["aliases"], key: string, concept: string, lang: VocabLang): void {
  const arr = map.get(key) ?? [];
  if (!arr.some((x) => x.concept === concept && x.lang === lang)) arr.push({ concept, lang });
  map.set(key, arr);
}

/** Top-level grouping ids that say nothing useful about an artwork on their own. */
export const STRUCTURAL = new Set(["style", "event", "object", "place", "accessory", "season"]);

function build(): Index {
  const defs = new Map<string, ConceptDef>();
  const aliases = new Map<string, Array<{ concept: string; lang: VocabLang }>>();
  const cjkAliases: Index["cjkAliases"] = [];
  let maxLen = 1;
  for (const line of VOCAB_TSV.split("\n")) {
    if (!line.trim()) continue;
    const cols = line.split("|").map((c) => c.trim());
    const id = cols[0];
    const parents = cols[1] ? cols[1].split(",").map((p) => p.trim()).filter(Boolean) : [];
    const def: ConceptDef = { id, parents, aliases: { en: [], fr: [], de: [], es: [], it: [], pt: [], ja: [], zh: [] } };
    LANGS.forEach((lang, k) => {
      const raw = (cols[2 + k] ?? "").split(",").map((a) => a.trim()).filter(Boolean);
      if (lang === "en" && !raw.includes(id)) raw.unshift(id);
      for (const a of raw) {
        const toks = matchTokens(a);
        if (!toks.length) continue;
        // A one-word alias that is a function word somewhere ("the" = French th\u00e9) would fire on
        // every sentence; such words are only usable inside longer aliases.
        if (toks.length === 1 && isStopword(toks[0]) && toks[0] !== id) continue;
        const key = toks.join(" ");
        def.aliases[lang].push(key);
        register(aliases, key, id, lang);
        maxLen = Math.max(maxLen, toks.length);
        if (/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/.test(key)) cjkAliases.push({ alias: key, concept: id, lang });
        // Tags are often glued words ("spaceinvader", "hellokitty"): also register the
        // concatenation of multi-word Latin aliases.
        if (toks.length > 1 && toks.every((t) => /^[a-z0-9]+$/.test(t))) register(aliases, toks.join(""), id, lang);
      }
    });
    defs.set(id, def);
  }
  // Parents that are not concepts of their own (e.g. "accessory", "vehicle") still exist as ids.
  for (const d of [...defs.values()]) {
    for (const p of d.parents) {
      if (!defs.has(p)) {
        defs.set(p, { id: p, parents: [], aliases: { en: [p], fr: [], de: [], es: [], it: [], pt: [], ja: [], zh: [] } });
        const arr = aliases.get(p) ?? [];
        if (!arr.length) aliases.set(p, [{ concept: p, lang: "en" }]);
      }
    }
  }
  cjkAliases.sort((a, b) => b.alias.length - a.alias.length);
  return { defs, aliases, maxLen, cjkAliases };
}

function idx(): Index {
  return (INDEX ??= build());
}

export function conceptDef(id: string): ConceptDef | undefined {
  return idx().defs.get(id);
}

export function allConceptIds(): string[] {
  return [...idx().defs.keys()];
}

/** Supported vocabulary language for a UI or text language ("fr", "fr-CH" → "fr"; unknown → "en"). */
export function vocabLang(lang: string | null | undefined): VocabLang {
  const l = String(lang ?? "").toLowerCase().slice(0, 2);
  return (LANGS as readonly string[]).includes(l) ? (l as VocabLang) : "en";
}

/** A concept's name in a language: its first alias there, else its first English one, else its id. */
export function conceptLabel(id: string, lang: string): string {
  const d = idx().defs.get(id);
  if (!d) return id;
  return d.aliases[vocabLang(lang)][0] ?? d.aliases.en[0] ?? id;
}

let SORTED: Array<{ alias: string; concept: string; lang: VocabLang }> | null = null;

/** Every alias as written in the vocabulary (no glued forms), sorted, for prefix lookups. */
function sortedAliases(): Array<{ alias: string; concept: string; lang: VocabLang }> {
  if (SORTED) return SORTED;
  const out: Array<{ alias: string; concept: string; lang: VocabLang }> = [];
  for (const d of idx().defs.values()) for (const lang of LANGS) for (const alias of d.aliases[lang]) out.push({ alias, concept: d.id, lang });
  out.sort((a, b) => (a.alias < b.alias ? -1 : a.alias > b.alias ? 1 : a.concept < b.concept ? -1 : a.concept > b.concept ? 1 : 0));
  return (SORTED = out);
}

/** Aliases (folded) that start with `prefix`, with their concept and language: completions for a word being typed. */
export function aliasesStartingWith(prefix: string, limit = 60): Array<{ alias: string; concept: string; lang: VocabLang }> {
  const p = fold(prefix);
  if (!p) return [];
  const list = sortedAliases();
  let lo = 0;
  let hi = list.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid].alias < p) lo = mid + 1;
    else hi = mid;
  }
  const out: Array<{ alias: string; concept: string; lang: VocabLang }> = [];
  for (let i = lo; i < list.length && out.length < limit && list[i].alias.startsWith(p); i++) out.push(list[i]);
  return out;
}

/** Concepts mentioned in a text, longest alias first, no overlapping spans. */
export function matchConcepts(text: string): ConceptMatch[] {
  const { aliases, maxLen, cjkAliases } = idx();
  const toks = matchTokens(text);
  const out: ConceptMatch[] = [];
  let i = 0;
  while (i < toks.length) {
    let matched = false;
    for (let n = Math.min(maxLen, toks.length - i); n >= 1 && !matched; n--) {
      const span = toks.slice(i, i + n);
      const candidates = [span.join(" ")];
      // singular fallback on the last word ("mountain lakes" → "mountain lake", "cats" → "cat")
      const sing = [...span.slice(0, -1), singular(span[n - 1])].join(" ");
      if (sing !== candidates[0]) candidates.push(sing);
      for (const key of candidates) {
        const hits = aliases.get(key);
        if (!hits) continue;
        const byConcept = new Map<string, VocabLang[]>();
        for (const h of hits) byConcept.set(h.concept, [...(byConcept.get(h.concept) ?? []), h.lang]);
        for (const [concept, langs] of byConcept) out.push({ concept, alias: key, start: i, end: i + n, langs });
        i += n;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const t = toks[i];
    // A hyphenated token that did not match whole: try its parts.
    if (t.includes("-")) {
      for (const part of t.split("-")) {
        const hits = aliases.get(part) ?? aliases.get(singular(part));
        if (hits) for (const h of hits) out.push({ concept: h.concept, alias: part, start: i, end: i + 1, langs: [h.lang] });
      }
    } else if (/[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]/.test(t)) {
      // CJK: find aliases inside the run ("猫の王様" contains "猫").
      let rest = t;
      for (const a of cjkAliases) {
        if (rest.includes(a.alias)) {
          out.push({ concept: a.concept, alias: a.alias, start: i, end: i + 1, langs: [a.lang] });
          rest = rest.split(a.alias).join(" ");
        }
      }
    }
    i += 1;
  }
  // de-duplicate concept ids, keep the first (longest-span) occurrence
  const seen = new Set<string>();
  return out.filter((m) => (seen.has(m.concept) ? false : (seen.add(m.concept), true)));
}

/** Ancestors of a concept with their depth (cat → pet 1, animal 1), nearest first. */
export function ancestorsWithDepth(id: string, maxDepth = 3): Array<{ id: string; depth: number }> {
  const out: Array<{ id: string; depth: number }> = [];
  const seen = new Set([id]);
  let frontier = [id];
  for (let d = 1; d <= maxDepth && frontier.length; d++) {
    const next: string[] = [];
    for (const c of frontier) {
      for (const p of conceptDef(c)?.parents ?? []) {
        if (seen.has(p)) continue;
        seen.add(p);
        out.push({ id: p, depth: d });
        next.push(p);
      }
    }
    frontier = next;
  }
  return out;
}

export function ancestors(id: string, maxDepth = 3): string[] {
  return ancestorsWithDepth(id, maxDepth).map((a) => a.id);
}

/** All folded aliases of a concept (every language), e.g. for lexical verification. */
export function aliasesOf(id: string): string[] {
  const d = conceptDef(id);
  if (!d) return [id];
  return [...new Set(Object.values(d.aliases).flat())];
}

// ---- per-artwork extraction -----------------------------------------------------------------

export interface ConceptSourceInput {
  tags: string[];
  title: string;
  description: string;
  ai?: { subjects: string[]; objects: string[]; tags: string[]; caption: string; style: string | null } | null;
}

export interface ArtworkConcept {
  concept: string;
  confidence: number;
  source: "tag" | "title" | "description" | "vlm" | "parent";
}

const W = { tag: 0.95, title: 0.85, description: 0.6, vlm_subject: 0.85, vlm_tag: 0.75, vlm_object: 0.7, vlm_caption: 0.65, vlm_style: 0.6 };

/** Concepts of one artwork with a confidence per concept (max over sources) and parents at 0.8×. */
export function extractArtworkConcepts(x: ConceptSourceInput, max = 48): ArtworkConcept[] {
  const acc = new Map<string, ArtworkConcept>();
  const put = (concept: string, confidence: number, source: ArtworkConcept["source"]) => {
    const cur = acc.get(concept);
    if (!cur || cur.confidence < confidence) acc.set(concept, { concept, confidence, source });
  };
  const from = (texts: string[], w: number, source: ArtworkConcept["source"]) => {
    for (const t of texts) for (const m of matchConcepts(t)) put(m.concept, w, source);
  };
  // Tags are single keywords, sometimes glued ("greeneyes", "spaceinvader"; the glued forms of
  // multi-word aliases are registered) or with underscores.
  from(x.tags.map((t) => t.replace(/[_]+/g, " ")), W.tag, "tag");
  from([x.title], W.title, "title");
  if (x.description && x.description.trim().length > 2) from([x.description], W.description, "description");
  if (x.ai) {
    from(x.ai.subjects, W.vlm_subject, "vlm");
    from(x.ai.tags, W.vlm_tag, "vlm");
    from(x.ai.objects, W.vlm_object, "vlm");
    if (x.ai.caption) from([x.ai.caption], W.vlm_caption, "vlm");
    if (x.ai.style) from([x.ai.style], W.vlm_style, "vlm");
  }
  for (const c of [...acc.values()]) {
    for (const a of ancestorsWithDepth(c.concept)) put(a.id, Math.round(c.confidence * Math.pow(0.8, a.depth) * 1000) / 1000, "parent");
  }
  return [...acc.values()]
    .filter((c) => !STRUCTURAL.has(c.concept))
    .sort((a, b) => b.confidence - a.confidence || a.concept.localeCompare(b.concept))
    .slice(0, max);
}
