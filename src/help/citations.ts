// Citations and the safety of help answers' text (v3, unchanged): a model's answer becomes plain
// text whose only addresses are ones the documentation excerpts contain, and whose citations
// point at excerpts that exist.

// ---- the answer's text ------------------------------------------------------------------------
//
// The model's answer becomes plain text that stays harmless however a client shows it: as text,
// through a linkifier, as Markdown (CommonMark or GFM, raw HTML on or off) or even as HTML. The
// only addresses it can contain are ones written in the excerpts given to the model; links to the
// excerpts themselves are in `sources`. Two layers:
//   1. readable rewriting: links become "text (address)" or "text", images their alt text, HTML
//      tags and reference definitions go, citations keep the excerpt numbers that exist;
//   2. guarantees that do not depend on parsing Markdown right: every address (URL, "//host",
//      "www.", domain name, IPv4, e-mail, mailto:) that the excerpts do not contain is removed,
//      after entities, backslash escapes and invisible characters are resolved, so none can be
//      smuggled in pieces (Chinese and Japanese text first gets a space next to Latin letters,
//      so an address ends where the sentence resumes); then "](", "]:", "<" + letter and
//      entity syntax are broken up, which is all that inline links, reference definitions,
//      autolinks, HTML and entities need.

export const MAX_ANSWER_CHARS = 1500;

/**
 * Anything a Markdown renderer or a linkifier could turn into a link. A host runs to the next space,
 * "/", "?", "#" or CJK punctuation, "<" and ">" included: linkifiers read "https://a.com<x@b.com"
 * as user "a.com<x" at host b.com. An ideographic or full-width full stop before a Latin letter or
 * digit is a dot (URL parsers read "a.com。xyz" as a.com.xyz); elsewhere it ends the sentence. A
 * path runs to the next space or CJK punctuation, where a Chinese or Japanese sentence resumes.
 */
const CJK_PUNCT_CLASS = String.raw`\u3000-\u303f\uff01-\uff0f\uff1a-\uff20\uff3b-\uff40\uff5b-\uff65`;
const HOST = String.raw`(?:[^\s/?#${CJK_PUNCT_CLASS}]|[\u3002\uff0e\uff61](?=[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a]))*`;
const PATH = String.raw`(?:[/?#][^\s${CJK_PUNCT_CLASS}]*)?`;
const ADDRESS = new RegExp(
  [
    String.raw`[a-z][a-z0-9+.-]*:\/\/${HOST}${PATH}`, // scheme://host/path
    String.raw`(?:mailto|xmpp):\S+`,
    String.raw`(?<![\p{L}\p{N}_])(?:javascript|vbscript|data):\S+`,
    String.raw`\/\/[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+${HOST}${PATH}`, // //host/path
    String.raw`[\p{L}\p{N}._%+-]+@[\p{L}\p{N}-]+(?:\.[\p{L}\p{N}-]+)+`, // e-mail
    String.raw`www\.${HOST}${PATH}`,
    String.raw`\d{1,3}(?:\.\d{1,3}){3}(?::\d{1,5})?${PATH}`, // IPv4
    String.raw`(?=[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a-]*[A-Za-z\uff21-\uff3a\uff41-\uff5a])[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a](?:[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a-]*[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a])?(?:[\u3002\uff0e\uff61][A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a](?:[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a-]*[A-Za-z0-9\uff10-\uff19\uff21-\uff3a\uff41-\uff5a])?)*[\u3002\uff0e\uff61][A-Za-z\uff21-\uff3a\uff41-\uff5a]{2,63}(?![A-Za-z\uff21-\uff3a\uff41-\uff5a])${PATH}`, // name。tld: URL parsers read ideographic and full-width full stops as dots
    String.raw`(?:[\p{L}\p{N}](?:[\p{L}\p{N}-]*[\p{L}\p{N}])?\.)+(?:\p{L}{2,63}|xn--[\p{L}\p{N}-]{1,59})(?!\p{L})(?::\d{1,5})?${PATH}`, // name.tld, IDN too
  ].join("|"),
  "giu",
);

/** An address without what GFM leaves out of an autolink: trailing punctuation and an unbalanced ")". */
function trimAddress(t: string): string {
  let s = t;
  for (;;) {
    const before = s;
    s = s.replace(/[.,:;!?'"*_~\]}]+$/, "");
    if (s.endsWith(")") && (s.match(/\)/g) ?? []).length > (s.match(/\(/g) ?? []).length) s = s.slice(0, -1);
    if (s === before) return s;
  }
}

/** What two addresses are compared on: lower case, no "https://", "//", "mailto:" or "www.", no trailing "/". */
function addressKey(address: string): string {
  return address
    .toLowerCase()
    .replace(/^(?:https?:)?\/\//, "")
    .replace(/^mailto:/, "")
    .replace(/^www\./, "")
    .replace(/\/$/, "");
}

/**
 * The addresses an answer may contain: those written in the excerpts (and the sources' own pages),
 * the same pages without their anchor, and the bare host names of those pages and e-mail addresses.
 */
export function allowedAddresses(allowedText: string): Set<string> {
  const keys = new Set<string>();
  for (const m of spaceCjk(decodeEntities(allowedText)).matchAll(ADDRESS)) {
    const key = addressKey(trimAddress(m[0]));
    if (!key) continue;
    keys.add(key);
    keys.add(key.replace(/#.*$/, ""));
    const email = /^[^@/?#:]+@([^@/?#:]+)$/.exec(key);
    const host = email ? email[1] : /^[^/?#]+/.exec(key)?.[0];
    if (host && !host.includes("@")) keys.add(host.replace(/:\d+$/, ""));
  }
  return keys;
}

const ENTITIES: Record<string, string> = {
  amp: "&", AMP: "&", lt: "<", LT: "<", gt: ">", GT: ">", quot: '"', QUOT: '"', apos: "'", nbsp: " ", NonBreakingSpace: " ",
  excl: "!", num: "#", dollar: "$", percnt: "%", lpar: "(", rpar: ")", ast: "*", midast: "*", plus: "+", comma: ",",
  period: ".", sol: "/", bsol: "\\", colon: ":", semi: ";", equals: "=", quest: "?", commat: "@", lsqb: "[", lbrack: "[",
  rsqb: "]", rbrack: "]", Hat: "^", lowbar: "_", UnderBar: "_", grave: "`", DiacriticalGrave: "`", lcub: "{", lbrace: "{",
  rcub: "}", rbrace: "}", verbar: "|", vert: "|", VerticalLine: "|", Tab: "\t", NewLine: "\n", hyphen: "‐", dash: "‐",
  ndash: "–", mdash: "—", hellip: "…", laquo: "«", raquo: "»", ldquo: "“", rdquo: "”", lsquo: "‘", rsquo: "’", copy: "©",
  reg: "®", trade: "™", euro: "€", deg: "°", middot: "·", times: "×",
};
const ENTITY = /&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});/g;

/**
 * Numeric entities and the named ones above, once, as a renderer reads them. Whatever is left of
 * entity syntax is broken up at the end, so a renderer cannot decode more than was checked.
 */
function decodeEntities(s: string): string {
  return s.replace(ENTITY, (m, e: string) => {
    if (e[0] !== "#") return ENTITIES[e] ?? m;
    const cp = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
    return cp > 0 && cp <= 0x10ffff && (cp < 0xd800 || cp > 0xdfff) ? String.fromCodePoint(cp) : "\ufffd";
  });
}

/** Controls, zero-width and bidirectional formatting characters: they can hide or reorder an address. */
const INVISIBLE = /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f\u00ad\u061c\u180e\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb]/g;

/**
 * Chinese, Japanese and Korean attach words to an address ("请访问pixagram.com", "pixagram.com에서"): a
 * space goes between them, so the address ends where the sentence resumes, and a linkifier does
 * not read "请访问pixagram.com" as one IDN host (another domain). Only in runs that look like an
 * address ("." or "@"): other Latin words keep their particles ("PXS를").
 */
const CJK_SCRIPT = String.raw`\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}`;
const CJK_LATIN = new RegExp(`(?<=[${CJK_SCRIPT}])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[${CJK_SCRIPT}])`, "gu");
const CJK_LETTER = new RegExp(`[${CJK_SCRIPT}]`, "u");
const spaceCjk = (s: string) => s.replace(/\S+/g, (run) => (/[.@]/.test(run) && CJK_LETTER.test(run) ? run.replace(CJK_LATIN, " ") : run));

/** Link reference definitions, wherever they are (CommonMark honours them in block quotes and list items). */
const REF_DEF = /\[(?:[^[\]\\]|\\[\s\S]){1,999}\]:[ \t]*\n?[ \t]*(?:<[^<>\n]*>|[^\s]*[:/.#@][^\s]*)(?:[ \t]*\n?[ \t]*(?:"(?:[^"\\]|\\[\s\S])*"|'(?:[^'\\]|\\[\s\S])*'|\((?:[^()\\]|\\[\s\S])*\)))?/g;

const CITATION = /^\s*\d{1,2}\s*(?:[,;–-]\s*\d{1,2}\s*)*$/;

/** The index after the code span opened by the backtick run at i, or -1 when it is not closed. */
function codeSpanEnd(s: string, i: number): number {
  let k = i;
  while (s[k] === "`") k++;
  const n = k - i;
  for (let j = k; j < s.length; ) {
    if (s[j] !== "`") {
      j++;
      continue;
    }
    let e = j;
    while (s[e] === "`") e++;
    if (e - j === n) return e;
    j = e;
  }
  return -1;
}

const blankLineAt = (s: string, j: number) => /^\n[ \t]*\n/.test(s.slice(j, j + 64));

/** The "]" that closes the "[" at i, as CommonMark reads link text: nested brackets, escapes, code spans; -1 if none. */
function closeBracket(s: string, i: number): number {
  let depth = 0;
  for (let j = i; j < s.length && j - i < 2000; j++) {
    const c = s[j];
    if (c === "\\") {
      j++;
      continue;
    }
    if (c === "`") {
      const e = codeSpanEnd(s, j);
      if (e > 0) j = e - 1;
      else while (s[j + 1] === "`") j++;
      continue;
    }
    if (c === "\n" && blankLineAt(s, j)) return -1;
    if (c === "[") depth++;
    else if (c === "]" && --depth === 0) return j;
  }
  return -1;
}

/** The "(destination "title")" of an inline link, read from just after its "(": the destination and the index after ")". */
function linkTail(s: string, i: number): { dest: string; end: number } | null {
  let j = i;
  const space = () => {
    let breaks = 0;
    while (j < s.length && (s[j] === " " || s[j] === "\t" || (s[j] === "\n" && breaks++ === 0))) j++;
  };
  space();
  let dest: string;
  if (s[j] === "<") {
    const e = s.indexOf(">", j + 1);
    if (e < 0 || /[\n<]/.test(s.slice(j + 1, e))) return null;
    dest = s.slice(j + 1, e);
    j = e + 1;
  } else {
    const start = j;
    let depth = 0;
    for (; j < s.length; j++) {
      const c = s[j];
      if (c === "\\") j++;
      else if (c === "(") {
        if (++depth > 32) return null;
      } else if (c === ")") {
        if (depth === 0) break;
        depth--;
      } else if (/[\s\u0000-\u001f\u007f]/.test(c)) break;
    }
    if (depth) return null;
    dest = s.slice(start, j);
  }
  const afterDest = j;
  space();
  if (j > afterDest && (s[j] === '"' || s[j] === "'" || s[j] === "(")) {
    const close = s[j] === "(" ? ")" : s[j];
    let k = j + 1;
    for (; k < s.length; k++) {
      if (s[k] === "\\") k++;
      else if (s[k] === close) break;
      else if (close === ")" && s[k] === "(") return null;
      else if (s[k] === "\n" && blankLineAt(s, k)) return null;
    }
    if (k >= s.length) return null;
    j = k + 1;
    space();
  }
  return s[j] === ")" ? { dest, end: j + 1 } : null;
}

/**
 * Links and images as text: "[text](address)" → "text (address)" when `show` gives the address,
 * else "text"; "![alt](…)" → "alt"; "[text][label]" → "text" ("text [n]" when the label is a
 * citation). Link text is rewritten too (links inside links, images inside links).
 */
function rewriteLinks(s: string, show: (dest: string) => string | null, depth = 0): string {
  let out = "";
  for (let i = 0; i < s.length; ) {
    const c = s[i];
    if (c === "\\") {
      out += s.slice(i, i + 2);
      i += 2;
      continue;
    }
    if (c === "`") {
      let e = codeSpanEnd(s, i);
      if (e < 0) for (e = i; s[e] === "`"; ) e++;
      out += s.slice(i, e);
      i = e;
      continue;
    }
    const image = c === "!" && s[i + 1] === "[";
    if (c === "[" || image) {
      const open = image ? i + 1 : i;
      const close = closeBracket(s, open);
      if (close > 0) {
        const text = s.slice(open + 1, close);
        const label = depth < 4 ? rewriteLinks(text, show, depth + 1) : text;
        if (s[close + 1] === "(") {
          const tail = linkTail(s, close + 2);
          if (tail) {
            const shown = image ? null : show(tail.dest);
            out += shown ? `${label} (${shown})` : label;
            i = tail.end;
            continue;
          }
        }
        if (s[close + 1] === "[" && (image || !CITATION.test(text))) {
          const end = closeBracket(s, close + 1);
          const ref = end > 0 ? s.slice(close + 2, end) : "";
          if (end > 0 && !/[[\]]/.test(ref)) {
            out += CITATION.test(ref) && !image ? `${label} [${ref}]` : label;
            i = end + 1;
            continue;
          }
        }
        if (image) {
          out += label;
          i = close + 1;
          continue;
        }
      }
    }
    out += c;
    i++;
  }
  return out;
}

/** "[2-4]" → "[2][3][4]", "[1, 9]" → "[1]" for 3 excerpts; numbers of three digits or more ("[2021]") are not citations. */
function normalizeCitations(s: string, n: number): string {
  return s.replace(/\[(\s*\d{1,2}\s*(?:[,;–-]\s*\d{1,2}\s*)*)\]/g, (_m, inner: string) => {
    const out: number[] = [];
    for (const part of inner.split(/[,;]/)) {
      const [a, b] = part.split(/[–-]/).map((x) => Number(x.trim()));
      const hi = b === undefined ? a : b;
      for (let k = a; k <= hi && k - a < 10; k++) if (Number.isInteger(k) && k >= 1 && k <= n && !out.includes(k)) out.push(k);
    }
    return out.map((k) => `[${k}]`).join("");
  });
}

/**
 * The model's answer as plain text (see above). Addresses stay only when the excerpts contain them
 * (`allowedText`); a link to one becomes "text (address)". Citations keep the excerpt numbers that
 * exist (`n` excerpts).
 */
export function sanitizeAnswer(answer: string, n: number, allowedText: string): string {
  const allowed = allowedAddresses(allowedText);
  const keep = (address: string) => {
    const key = addressKey(trimAddress(address));
    if (!key) return false;
    if (allowed.has(key)) return true;
    // the same page at another anchor (an anchor, not "#evil.example", which a linkifier would link)
    const hash = key.indexOf("#");
    return hash > 0 && /^#[\p{L}\p{N}_-]*$/u.test(key.slice(hash)) && allowed.has(key.slice(0, hash));
  };
  // what a reader would see: entities decoded, invisible characters and backslash escapes removed
  let s = decodeEntities(answer.slice(0, 6000))
    .replace(/\r\n?|[\u2028\u2029]/g, "\n")
    .replace(INVISIBLE, "");
  for (let round = 0; round < 3; round++) {
    const next = s.replace(/\\([!-/:-@[-`{-~])/g, "$1");
    if (next === s) break;
    s = next;
  }
  s = s
    .replace(REF_DEF, " ")
    // autolinks before HTML, which would take them for tags: the address alone, judged below
    .replace(/<((?:[a-z][a-z0-9+.-]{1,31}:|[\w.!#$%&'*+/=?^`{|}~-]+@)[^\s<>]*)>/gi, " $1 ")
    .replace(/<!--[\s\S]*?-->|<![A-Za-z[][^>]*>|<\?[\s\S]*?\?>|<\/?[A-Za-z][A-Za-z0-9-]*(?:[\s/][^<>]*)?>/g, " ");
  s = rewriteLinks(s, (dest) => (keep(dest) ? trimAddress(dest).replace(/^mailto:/i, "") : null));
  s = normalizeCitations(s, n)
    .replace(/[ \t]+/g, " ")
    .replace(/ +([.,;:!?])/g, "$1")
    .replace(/ *\n */g, "\n")
    .trim();
  if (s.length > MAX_ANSWER_CHARS) {
    const cut = s.slice(0, MAX_ANSWER_CHARS - 1).search(/\s\S*$/);
    s = `${s.slice(0, cut > MAX_ANSWER_CHARS / 2 ? cut : MAX_ANSWER_CHARS - 1).trimEnd()} …`;
  }
  // After every rewrite that can join two pieces of text (and the cut): no address is left in pieces.
  // A kept URL that CJK punctuation follows gets a space: GFM would take what follows into its link.
  const cjkPunct = new RegExp(`^[${CJK_PUNCT_CLASS}]`, "u");
  s = spaceCjk(s).replace(new RegExp(`( *)(?:${ADDRESS.source})`, ADDRESS.flags), (m: string, lead: string, offset: number, all: string) => {
    const address = m.slice(lead.length);
    const core = trimAddress(address);
    const rest = address.slice(core.length);
    if (core && keep(core)) {
      const autolinked = /^(?:[a-z][a-z0-9+.-]*:\/\/|\/\/|www\.)/i.test(core);
      const gap = autolinked && !rest && cjkPunct.test(all.slice(offset + m.length, offset + m.length + 1)) ? " " : "";
      return lead + (/^mailto:/i.test(core) ? core.slice(7) + rest : address) + gap;
    }
    return /^[.,;:!?]/.test(rest) ? rest : `${lead} ${rest}`;
  });
  s = s
    .replace(/\(\s*[,;:]?\s*\)|\[\s*\]/g, " ")
    .replace(/^[ \t]*(?:(?:>|[-*+]|\d{1,9}[.)])[ \t]*)+$/gm, "")
    .replace(/[ \t]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
  // Nothing left can make a link, a definition, an autolink, a tag or an entity.
  return s
    .replace(/\]\(/g, "] (")
    .replace(/\]:/g, "] :")
    .replace(/<(?=[A-Za-z/!?])/g, "< ")
    .replace(/&(?=(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});)/g, "& ");
}

/**
 * Check a model reply: citations must point at given excerpts, links must come from the excerpts.
 * null when the reply is unusable.
 */
export function validateHelpReply(raw: unknown, n: number, allowedText: string): { answerable: boolean; answer: string; sources: number[] } | null {
  let o: any = raw;
  if (typeof raw === "string") {
    try {
      o = JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1));
    } catch {
      return null;
    }
  }
  if (!o || typeof o !== "object" || typeof o.answerable !== "boolean") return null;
  const listed = (Array.isArray(o.sources) ? o.sources : []).map(Number).filter((x: number) => Number.isInteger(x) && x >= 1 && x <= n);
  const answer = sanitizeAnswer(typeof o.answer === "string" ? o.answer : "", n, allowedText);
  if (!o.answerable || !answer) return { answerable: false, answer: "", sources: [] };
  const cited = [...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1]));
  return { answerable: true, answer, sources: [...new Set<number>([...listed, ...cited])].sort((a, b) => a - b) };
}

