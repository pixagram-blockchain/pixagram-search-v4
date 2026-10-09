// Text utilities shared by indexing and querying: normalisation, tokenisation, stopwords,
// light singularisation, edit distance, trigrams and a tiny language guesser.

/** Lowercase, NFKC, strip diacritics (é → e, ü → u), keep CJK as is. */
export function fold(s: string): string {
  return s
    .normalize("NFKC")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .normalize("NFC");
}

const CJK = /[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]/;

export function hasCjk(s: string): boolean {
  return CJK.test(s);
}

/**
 * Word tokens of folded text. Letters and digits only; apostrophes split ("l'avion" → "l", "avion");
 * hyphenated words are kept whole and as parts ("sponge-bob" → "sponge-bob", "sponge", "bob").
 */
export function tokens(s: string, opts: { keepHyphenated?: boolean } = {}): string[] {
  const out: string[] = [];
  for (const raw of fold(s).split(/[^\p{L}\p{N}\-]+/u)) {
    const t = raw.replace(/^-+|-+$/g, "");
    if (!t) continue;
    if (t.includes("-")) {
      if (opts.keepHyphenated !== false) out.push(t);
      for (const p of t.split("-")) if (p) out.push(p);
    } else out.push(t);
  }
  return out;
}

// Function words of the languages Pixagram users write in. Used to measure query coverage and to
// relax queries; never removed from the full-text index itself.
const STOP = {
  en: "a an the and or of to in on at by for with from is are was were be been it its this that these those as into over under about i me my we you your he she they them his her their our who what which when where how why do does did has have had not no yes so if than then there here very just also any all some ever image images picture pictures photo artwork artworks art post posts posted upload uploaded drawing pixel pixels",
  fr: "le la les un une des du de d l et ou au aux en dans sur sous par pour avec sans est sont etait c ce cet cette ces qui que quoi quel quelle quels quelles quand ou comment je j me m mon ma mes tu te ton ta tes il elle ils elles nous vous leur leurs son sa ses ne pas plus tres a y image images dessin oeuvre oeuvres œuvre œuvres publie publiee poste",
  de: "der die das den dem des ein eine einen einem einer und oder zu im in am an auf aus bei mit von vom zum zur fur ist sind war waren wer was welche welcher welches wann wo wie ich du er sie es wir ihr mein meine dein sein ihre nicht kein keine sehr auch bild bilder kunstwerk gepostet",
  es: "el la los las un una unos unas de del y o en con por para sin es son era que quien quienes cual cuando donde como yo tu el ella nosotros mi mis su sus no muy tambien imagen imagenes obra publico",
  it: "il lo la i gli le un uno una di del della dei degli delle e o in con per su da che chi quale quando dove come io tu lui lei noi mio mia suo sua non molto anche immagine immagini opera",
} as const;

export const STOPWORDS: Record<keyof typeof STOP, Set<string>> = Object.fromEntries(
  Object.entries(STOP).map(([k, v]) => [k, new Set(v.split(/\s+/))]),
) as Record<keyof typeof STOP, Set<string>>;

const ALL_STOP = new Set<string>(Object.values(STOP).flatMap((v) => v.split(/\s+/)));

export function isStopword(t: string): boolean {
  return ALL_STOP.has(t);
}

/** Content tokens: tokens minus stopwords; if that leaves nothing, the tokens themselves. */
export function contentTokens(s: string): string[] {
  const t = tokens(s, { keepHyphenated: false });
  const c = t.filter((x) => !ALL_STOP.has(x));
  return c.length ? c : t;
}

/** Light English/French/German plural folding, for matching only (cats → cat, butterflies → butterfly). */
export function singular(t: string): string {
  if (t.length <= 3 || hasCjk(t)) return t;
  if (/ies$/.test(t) && t.length > 4) return t.slice(0, -3) + "y";
  if (/(ches|shes|sses|xes|zes)$/.test(t)) return t.slice(0, -2);
  if (/(ss|us|is|ous)$/.test(t)) return t;
  if (/s$/.test(t)) return t.slice(0, -1);
  if (/x$/.test(t) && /(eaux|aux)$/.test(t)) return t.slice(0, -1); // gâteaux → gateau
  return t;
}

/** Restricted Damerau–Levenshtein distance (adjacent transpositions), early exit above `max`. */
export function editDistance(a: string, b: string, max = 3): number {
  if (a === b) return 0;
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const n = a.length;
  const m = b.length;
  let prev2 = new Array<number>(m + 1).fill(0);
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  let cur = new Array<number>(m + 1).fill(0);
  let prevMin = 0;
  for (let i = 1; i <= n; i++) {
    cur[0] = i;
    let rowMin = cur[0];
    for (let j = 1; j <= m; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    // A transposition can reach back two rows, so stop only when both recent rows exceed max.
    if (rowMin > max && prevMin > max) return max + 1;
    prevMin = rowMin;
    [prev2, prev, cur] = [prev, cur, prev2];
  }
  return Math.min(prev[m], max + 1);
}

/** Character trigrams of a term padded with "^" and "$" ("cat" → ^ca, cat, at$). */
export function trigrams(term: string): string[] {
  const s = `^${term}$`;
  const out = new Set<string>();
  for (let i = 0; i + 3 <= s.length; i++) out.add(s.slice(i, i + 3));
  return [...out];
}

export type Lang = "en" | "fr" | "de" | "es" | "it" | "ja" | "zh" | "ko" | "ru";

/** Best guess at the language of a short query; "en" when nothing points elsewhere. */
export function guessLang(s: string): Lang {
  if (/[\u3040-\u30ff]/.test(s)) return "ja";
  if (/[\uac00-\ud7af]/.test(s)) return "ko";
  if (/[\u4e00-\u9fff]/.test(s)) return "zh";
  if (/[\u0400-\u04ff]/.test(s)) return "ru";
  const raw = s.toLowerCase();
  const t = tokens(s, { keepHyphenated: false });
  const score: Record<string, number> = { en: 0, fr: 0, de: 0, es: 0, it: 0 };
  for (const x of t) for (const [lang, set] of Object.entries(STOPWORDS)) if (set.has(x)) score[lang] += 1;
  // Diacritics and spellings that only one of the languages uses.
  if (/[\u00e0\u00e2\u00e7\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u0153\u00f9\u00fb]|qu'|l'|d'/.test(raw)) score.fr += 1.5;
  if (/[\u00e4\u00f6\u00fc\u00df]/.test(raw)) score.de += 1.5;
  if (/[\u00f1\u00bf\u00a1]/.test(raw)) score.es += 1.5;
  // Words shared by several languages count less ("de" is fr/es/it, "la" fr/es/it).
  const best = Object.entries(score).sort((a, b) => b[1] - a[1]);
  return best[0][1] > 0 && best[0][1] > best[1][1] ? (best[0][0] as Lang) : "en";
}

/** Short stable hash (FNV-1a, 32-bit, hex) for cache keys and change detection. */
export function fnv1a(s: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, "0");
}

/** Random id for query logs (12 chars, base36). */
export function randomId(): string {
  const b = new Uint8Array(8);
  crypto.getRandomValues(b);
  let s = "";
  for (const x of b) s += (x % 36).toString(36);
  return s + Date.now().toString(36).slice(-4);
}
