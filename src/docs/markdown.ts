// Markdown files of the documentation repository → title, language, sections → chunks.
//
// A section is a heading of level 1–3 and the text under it (levels 4–6 stay inside their
// section). Long sections are cut at paragraph boundaries. Every chunk keeps its heading path and
// the GitHub anchor of its section, so an answer can link to the exact place on github.com.
//
// Front matter (optional, between --- lines at the top):
//   title:    the document's title (else the first level-1 heading, else the file name)
//   lang:     its language (else a /fr/ folder or a .fr.md suffix, else guessed from the text)
//   keywords: [a, b] extra words people may search with, indexed with the first chunk
//   draft: true  |  search: false  |  noindex: true    leaves the file out

import { fold, guessLang } from "../lib/text";

export const MAX_CHUNK_WORDS = 250;
const LANGS = ["en", "fr", "de", "es", "it", "pt", "ja", "zh", "ko", "ru"];

export interface DocMeta {
  title: string | null;
  lang: string | null;
  description: string | null;
  keywords: string[];
  /** draft: true, search: false or noindex: true */
  skip: boolean;
}

export interface DocSection {
  /** 0 for the text before the first heading */
  level: number;
  /** heading texts from level 1 down to this section */
  path: string[];
  /** GitHub anchor of the section's heading ('' for level 0) */
  anchor: string;
  text: string;
}

export interface ParsedDoc {
  title: string;
  lang: string;
  meta: DocMeta;
  sections: DocSection[];
}

export interface DocChunk {
  ord: number;
  /** heading path inside the document, without its title: "Royalties › How they are paid" */
  heading: string;
  anchor: string;
  text: string;
}

// ---- front matter -----------------------------------------------------------------------------

const unquote = (v: string) => v.trim().replace(/^(["'])([\s\S]*)\1$/, "$2").trim();

/** A YAML subset: `key: value`, `key: [a, b]`, `key:` followed by `- item` lines, booleans. */
export function splitFrontMatter(src: string): { meta: Record<string, string | string[] | boolean>; body: string } {
  const s = src.replace(/^\ufeff/, "");
  const m = /^---[ \t]*\r?\n([\s\S]*?)\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/.exec(s);
  if (!m) return { meta: {}, body: s };
  const meta: Record<string, string | string[] | boolean> = {};
  let listKey: string | null = null;
  for (const line of m[1].split(/\r?\n/)) {
    const item = /^\s*-\s+(.*)$/.exec(line);
    if (item && listKey) {
      (meta[listKey] as string[]).push(unquote(item[1]));
      continue;
    }
    const kv = /^([A-Za-z_][\w-]*)\s*:\s*(.*)$/.exec(line);
    listKey = null;
    if (!kv) continue;
    const key = kv[1].toLowerCase();
    const raw = kv[2].trim();
    if (raw === "") {
      meta[key] = [];
      listKey = key;
    } else if (/^\[.*\]$/.test(raw)) meta[key] = raw.slice(1, -1).split(",").map(unquote).filter(Boolean);
    else if (/^(true|false|yes|no)$/i.test(raw)) meta[key] = /^(true|yes)$/i.test(raw);
    else meta[key] = unquote(raw);
  }
  return { meta, body: s.slice(m[0].length) };
}

function docMeta(fm: Record<string, string | string[] | boolean>): DocMeta {
  const str = (v: unknown) => (typeof v === "string" && v.trim() ? v.trim() : null);
  const list = (v: unknown) => (Array.isArray(v) ? v : typeof v === "string" ? v.split(",") : []).map((x) => String(x).trim()).filter(Boolean);
  return {
    title: str(fm.title),
    lang: str(fm.lang ?? fm.language)?.toLowerCase().slice(0, 2) ?? null,
    description: str(fm.description),
    keywords: [...list(fm.keywords), ...list(fm.tags), ...list(fm.aliases)],
    skip: fm.draft === true || fm.search === false || fm.noindex === true,
  };
}

// ---- inline text ------------------------------------------------------------------------------

export interface InlineOptions {
  /** keep where links to web pages and e-mail addresses go: "text (https://…)" (body text; not headings, whose anchors GitHub makes from their text) */
  links?: boolean;
  /** the document's link reference definitions, by refLabel(label) */
  refs?: Map<string, string>;
}

/** A link label as CommonMark matches it: case and runs of spaces do not count. */
export const refLabel = (label: string) => label.trim().replace(/\s+/g, " ").toLowerCase();

/** The address a link shows in the text: an absolute http(s) URL or an e-mail address; null for anything else (relative links, anchors). */
function linkAddress(dest: string | undefined): string | null {
  const d = (dest ?? "").trim().replace(/^<([^>]*)>$/, "$1");
  if (/^https?:\/\/[^\s<>]+$/i.test(d)) return d;
  return /^mailto:([^\s<>?@]+@[^\s<>?@]+)/i.exec(d)?.[1] ?? null;
}

/**
 * Markdown inline syntax → plain text: links keep their text, images their alt text. With
 * `links`, a link to a web page or an e-mail address keeps it in parentheses, so that a help
 * answer can give it (the answer may only contain addresses the documentation contains).
 */
export function inlineText(s: string, opts: InlineOptions = {}): string {
  const link = (text: string, dest: string | undefined) => {
    const address = opts.links ? linkAddress(dest) : null;
    return address && address !== text ? `${text} (${address})` : text;
  };
  const ref = (label: string) => opts.refs?.get(refLabel(label));
  return s
    .replace(/!\[([^\]]*)\]\((?:[^()\s]|\([^)]*\))*(?:\s+"[^"]*")?\)/g, "$1")
    .replace(/!\[([^\]]*)\]\[[^\]]*\]/g, "$1")
    .replace(/\[([^\]]+)\]\(\s*((?:[^()\s]|\([^)\s]*\))*)(?:\s+"[^"]*")?\s*\)/g, (_m, text: string, dest: string) => link(text, dest))
    .replace(/\[([^\]]+)\]\[([^\]]*)\]/g, (_m, text: string, label: string) => link(text, ref(label || text)))
    .replace(/\[([^\]]+)\](?![([:])/g, (m: string, label: string) => {
      const dest = ref(label);
      return dest === undefined ? m : link(label, dest); // a shortcut reference link, when defined
    })
    .replace(/<((?:https?|mailto):[^>\s]+)>/gi, (_m, url: string) => url.replace(/^mailto:/i, ""))
    .replace(/<([\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+)>/g, "$1") // an e-mail autolink, not a tag
    .replace(/<\/?[A-Za-z][^>]*>/g, " ")
    .replace(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g, "$2")
    .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, "$2")
    .replace(/(^|[\s(["'])([*_])(?=\S)([^*_\n]*?\S)\2(?=$|[\s)\].,;:!?"'])/g, "$1$3")
    .replace(/~~(?=\S)([\s\S]*?\S)~~/g, "$1")
    .replace(/&nbsp;/g, " ")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&amp;/g, "&")
    .replace(/[ \t]+/g, " ")
    .trim();
}

/** GitHub's heading anchors (github-slugger): lowercase, punctuation removed, spaces → "-", "-1", "-2" for repeats. */
export function githubSlug(heading: string, seen: Map<string, number>): string {
  const base = heading
    .toLowerCase()
    .trim()
    .replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, "")
    .replace(/ /g, "-");
  const n = seen.get(base) ?? 0;
  seen.set(base, n + 1);
  return n === 0 ? base : `${base}-${n}`;
}

// ---- blocks -----------------------------------------------------------------------------------

const CODE = "\u0000"; // marks lines inside fenced code blocks while a section collects them

function cleanBlock(lines: string[], refs: Map<string, string>): string {
  const out: string[] = [];
  for (const line of lines) {
    if (line.startsWith(CODE)) {
      const l = line.slice(1);
      if (!/^\s{0,3}(`{3,}|~{3,})/.test(l)) out.push(l.trimEnd());
      continue;
    }
    if (/^\s{0,3}\[[^\]]+\]:\s*\S/.test(line)) continue; // reference definition
    if (/^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(line)) continue; // table alignment row
    if (/^\s{0,3}([-*_])(\s*\1){2,}\s*$/.test(line)) continue; // thematic break
    let l = line.replace(/^(\s*>\s?)+/, ""); // blockquotes
    if (/^\s{0,3}\[[^\]]+\]:\s*\S/.test(l)) continue; // a reference definition in a block quote
    l = l.replace(/^\s{0,3}#{4,6}[ \t]+(.*?)[ \t]*#*[ \t]*$/, "$1"); // headings below level 3 stay in the text
    if (/^\s*\|.*\|\s*$/.test(l)) l = l.trim().slice(1, -1).split("|").map((c) => c.trim()).join(" | ");
    l = l.replace(/^(\s*)[*+](\s+)/, "$1-$2");
    out.push(inlineText(l, { links: true, refs }));
  }
  return out
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

// ---- document ---------------------------------------------------------------------------------

function titleFromPath(path: string): string {
  const parts = path.split("/");
  let name = parts.pop() ?? path;
  if (/^(readme|index)\.(mdx?|markdown|txt)$/i.test(name)) name = parts.pop() ?? "Overview";
  const t = name.replace(/\.(mdx?|markdown|txt)$/i, "").replace(/[-_]+/g, " ").trim();
  return t ? t[0].toUpperCase() + t.slice(1) : "Overview";
}

function langOf(meta: DocMeta, path: string, text: string): string {
  if (meta.lang && LANGS.includes(meta.lang)) return meta.lang;
  const dir = /(?:^|\/)(en|fr|de|es|it|pt|ja|zh|ko|ru)(?:\/)/i.exec(path);
  if (dir) return dir[1].toLowerCase();
  const suffix = /[._-](en|fr|de|es|it|pt|ja|zh|ko|ru)\.(mdx?|markdown|txt)$/i.exec(path);
  if (suffix) return suffix[1].toLowerCase();
  return guessLang(text.slice(0, 1500));
}

export function parseDoc(src: string, path: string): ParsedDoc {
  const { meta: fm, body } = splitFrontMatter(src);
  const meta = docMeta(fm);
  // HTML comments go first (they may span lines); the line count is kept.
  const lines = body
    .replace(/\r\n?/g, "\n")
    .replace(/<!--[\s\S]*?-->/g, (m) => m.replace(/[^\n]/g, ""))
    .split("\n");
  // link reference definitions, for "[text][label]" links anywhere in the document (the first one counts)
  const refs = new Map<string, string>();
  let inFence: string | null = null;
  for (const line of lines) {
    const f = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (inFence) {
      if (f && f[1][0] === inFence[0] && f[1].length >= inFence.length) inFence = null;
      continue;
    }
    if (f) {
      inFence = f[1];
      continue;
    }
    const d = /^\s{0,3}(?:>\s?)*\[([^\]]+)\]:\s*(<[^>\s]*>|\S+)/.exec(line);
    if (d && !refs.has(refLabel(d[1]))) refs.set(refLabel(d[1]), d[2].replace(/^<(.*)>$/, "$1"));
  }
  const sections: DocSection[] = [];
  const seen = new Map<string, number>();
  const stack: string[] = [];
  let cur = { level: 0, path: [] as string[], anchor: "", lines: [] as string[] };
  let fence: string | null = null;
  let firstH1: string | null = null;
  const flush = () => {
    const text = cleanBlock(cur.lines, refs);
    if (text || cur.level > 0) sections.push({ level: cur.level, path: cur.path, anchor: cur.anchor, text });
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const f = /^\s{0,3}(`{3,}|~{3,})/.exec(line);
    if (fence) {
      if (f && f[1][0] === fence[0] && f[1].length >= fence.length && /^\s{0,3}(`{3,}|~{3,})\s*$/.test(line)) fence = null;
      cur.lines.push(CODE + line);
      continue;
    }
    if (f) {
      fence = f[1];
      cur.lines.push(CODE + line);
      continue;
    }
    let level = 0;
    let raw = "";
    const atx = /^\s{0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*(?:[ \t]#+)?[ \t]*$/.exec(line);
    if (atx) {
      level = atx[1].length;
      raw = atx[2] ?? "";
    } else if (
      i + 1 < lines.length &&
      line.trim() &&
      !/^\s{0,3}([-*+]|\d+[.)])\s/.test(line) &&
      !/^\s*\|/.test(line) &&
      (i === 0 || !lines[i - 1].trim()) &&
      /^\s{0,3}(=+|-+)\s*$/.test(lines[i + 1])
    ) {
      level = lines[i + 1].trim().startsWith("=") ? 1 : 2;
      raw = line;
      i++;
    }
    if (!level) {
      cur.lines.push(line);
      continue;
    }
    const text = inlineText(raw);
    const anchor = githubSlug(text, seen); // every heading counts for GitHub's de-duplication
    if (level > 3) {
      cur.lines.push(text);
      continue;
    }
    flush();
    if (level === 1 && firstH1 === null) firstH1 = text;
    stack.length = level - 1;
    stack[level - 1] = text;
    cur = { level, path: stack.filter((x) => x !== undefined && x !== ""), anchor, lines: [] };
  }
  flush();
  const title = meta.title ?? firstH1 ?? titleFromPath(path);
  const sample = sections.map((s) => s.text).join("\n");
  return { title, lang: langOf(meta, path, sample), meta, sections };
}

/** Split text into pieces of at most `max` words: paragraphs first, then sentences, then words. */
export function splitWords(text: string, max = MAX_CHUNK_WORDS): string[] {
  const out: string[] = [];
  let cur: string[] = [];
  let n = 0;
  const words = (s: string) => s.split(/\s+/).filter(Boolean).length;
  const flush = () => {
    if (cur.length) out.push(cur.join("\n\n"));
    cur = [];
    n = 0;
  };
  const add = (piece: string) => {
    const w = words(piece);
    if (!w) return;
    if (n && n + w > max) flush();
    if (w > max) {
      const ws = piece.split(/\s+/).filter(Boolean);
      for (let i = 0; i < ws.length; i += max) out.push(ws.slice(i, i + max).join(" "));
      return;
    }
    cur.push(piece);
    n += w;
  };
  for (const p of text.split(/\n{2,}/)) {
    if (words(p) <= max) add(p);
    else {
      for (const s of p.split(/(?<=[.!?。！？])\s+/)) add(s);
      flush();
    }
  }
  flush();
  return out;
}

/** Sections → chunks. A heading without text adds nothing: it lives on in its children's paths. */
export function chunkDoc(doc: ParsedDoc, max = MAX_CHUNK_WORDS): DocChunk[] {
  const out: DocChunk[] = [];
  const titleKey = fold(doc.title).trim();
  for (const s of doc.sections) {
    const path = s.path.length && fold(s.path[0]).trim() === titleKey ? s.path.slice(1) : s.path;
    if (!s.text.trim()) continue;
    for (const part of splitWords(s.text, max)) out.push({ ord: out.length, heading: path.join(" › "), anchor: s.anchor, text: part });
  }
  // Keywords from the front matter make the document findable by the words people use for it.
  if (out.length && doc.meta.keywords.length) out[0] = { ...out[0], text: `${out[0].text}\n\n${doc.meta.keywords.join(", ")}` };
  if (!out.length && doc.meta.description) out.push({ ord: 0, heading: "", anchor: "", text: doc.meta.description });
  return out;
}
