// GitHub access for the documentation repository, without the REST API.
//
// Workers share their outbound addresses, and GitHub allows 60 anonymous API requests an hour per
// address: an API-based sync would fail at random. It uses the two routes every `git clone` and
// every "Download ZIP" uses instead, which have no such limit:
//   head     git's ref advertisement, https://github.com/<repo>.git/info/refs (a few hundred bytes)
//   files    the commit's archive, https://codeload.github.com/<repo>/tar.gz/<commit>, read as a
//            stream: Markdown files are kept, everything else is skipped without being stored

import type { Env } from "../env";
import { list } from "../env";

export interface RepoRef {
  owner: string;
  repo: string;
  branch: string;
}

export interface ArchiveFile {
  /** path inside the repository */
  path: string;
  size: number;
  /** sha-256 of the content (hex, 40 characters) */
  hash: string;
  /** null when the file is too large or not UTF-8 text */
  text: string | null;
}

export class DocsSourceError extends Error {
  constructor(msg: string, readonly retryable: boolean) {
    super(msg);
    this.name = "DocsSourceError";
  }
}

export const DEFAULT_DOCS_REPO = "pixagram-blockchain/information";
const TIMEOUT_MS = 20_000;
/** Larger files are not documentation pages (skipped, not failed). */
export const MAX_DOC_BYTES = 512 * 1024;
/** The archive is read as a stream; past this many compressed bytes the sync gives up. */
export const MAX_ARCHIVE_BYTES = 64 * 1024 * 1024;

export function repoRef(env: Env): RepoRef | null {
  const full = (env.DOCS_REPO ?? DEFAULT_DOCS_REPO).trim();
  if (!full || full === "off") return null;
  const m = /^([A-Za-z0-9_.-]+)\/([A-Za-z0-9_.-]+)$/.exec(full);
  if (!m) return null;
  return { owner: m[1], repo: m[2], branch: (env.DOCS_BRANCH ?? "main").trim() || "main" };
}

export const repoName = (r: RepoRef) => `${r.owner}/${r.repo}`;

const encodePath = (p: string) => p.split("/").map(encodeURIComponent).join("/");

/** The page of a file (and section) on github.com. */
export function blobUrl(r: RepoRef, path: string, anchor = ""): string {
  return `https://github.com/${r.owner}/${r.repo}/blob/${encodeURIComponent(r.branch)}/${encodePath(path)}${anchor ? `#${anchor}` : ""}`;
}

const UA = { "user-agent": "pixagram-search-v3 (git/2)" };

async function get(url: string): Promise<Response> {
  try {
    return await fetch(url, { headers: UA, signal: AbortSignal.timeout(TIMEOUT_MS) });
  } catch (e) {
    throw new DocsSourceError(`GitHub unreachable: ${e instanceof Error ? e.message : String(e)}`, true);
  }
}

function failure(res: Response, what: string): DocsSourceError {
  if (res.status === 404 || res.status === 401) return new DocsSourceError(`${what}: not found (is the repository public?)`, false);
  return new DocsSourceError(`${what}: HTTP ${res.status}`, res.status >= 500 || res.status === 403 || res.status === 429);
}

/** Parse git's ref advertisement (pkt-lines, lengths in bytes) into ref name → commit. */
export function parseRefs(body: Uint8Array): Map<string, string> {
  const refs = new Map<string, string>();
  const utf8 = new TextDecoder();
  let i = 0;
  while (i + 4 <= body.length) {
    const len = parseInt(utf8.decode(body.subarray(i, i + 4)), 16);
    if (!Number.isFinite(len)) break;
    if (len === 0) {
      i += 4; // flush packet
      continue;
    }
    if (len < 4 || i + len > body.length) break;
    const line = utf8.decode(body.subarray(i + 4, i + len)).replace(/\n$/, "");
    i += len;
    const m = /^([0-9a-f]{40}) ([^\0]+)/.exec(line);
    if (m) refs.set(m[2], m[1]);
  }
  return refs;
}

/** The branch's head commit. */
export async function headCommit(env: Env, r: RepoRef): Promise<string> {
  const res = await get(`https://github.com/${r.owner}/${r.repo}.git/info/refs?service=git-upload-pack`);
  if (!res.ok) throw failure(res, `refs of ${repoName(r)}`);
  const refs = parseRefs(new Uint8Array(await res.arrayBuffer()));
  const sha = refs.get(`refs/heads/${r.branch}`);
  if (!sha) throw new DocsSourceError(`${repoName(r)} has no branch ${r.branch}`, false);
  return sha;
}

/** Which files are documentation: Markdown and text, outside tooling folders and licences, under DOCS_PATHS when set. */
export function isDocFile(path: string, include: string[] = []): boolean {
  if (!/\.(md|mdx|markdown|txt)$/i.test(path)) return false;
  if (/(^|\/)(\.github|\.git|node_modules|vendor)\//.test(path)) return false;
  if (/(^|\/)(license|licence|copying)(\.[a-z]+)?\.(md|txt|markdown)$/i.test(path)) return false;
  return !include.length || include.some((p) => path === p || path.startsWith(p.endsWith("/") ? p : `${p}/`));
}

export const docPaths = (env: Env) => list(env.DOCS_PATHS);

// ---- tar ------------------------------------------------------------------------------------------

/** A pull reader over a byte stream. */
class Bytes {
  private buf = new Uint8Array(0);
  private off = 0;
  private ended = false;
  constructor(private readonly reader: ReadableStreamDefaultReader<Uint8Array>) {}
  private async fill(n: number): Promise<boolean> {
    while (this.buf.length - this.off < n && !this.ended) {
      const { value, done } = await this.reader.read();
      if (done) {
        this.ended = true;
        break;
      }
      const rest = this.buf.subarray(this.off);
      const next = new Uint8Array(rest.length + value.length);
      next.set(rest);
      next.set(value, rest.length);
      this.buf = next;
      this.off = 0;
    }
    return this.buf.length - this.off >= n;
  }
  async take(n: number): Promise<Uint8Array | null> {
    if (!(await this.fill(n))) return null;
    const out = this.buf.slice(this.off, this.off + n);
    this.off += n;
    return out;
  }
  async skip(n: number): Promise<boolean> {
    while (n > 0) {
      if (this.buf.length - this.off === 0 && !(await this.fill(1))) return false;
      const k = Math.min(n, this.buf.length - this.off);
      this.off += k;
      n -= k;
    }
    return true;
  }
}

const dec = new TextDecoder();
const field = (b: Uint8Array, start: number, len: number) => {
  const s = b.subarray(start, start + len);
  const end = s.indexOf(0);
  return dec.decode(end < 0 ? s : s.subarray(0, end));
};
const octal = (b: Uint8Array, start: number, len: number) => {
  if (b[start] & 0x80) {
    let n = 0; // base-256 (GNU, for files of 8 GB and more)
    for (let i = start + 1; i < start + len; i++) n = n * 256 + b[i];
    return n;
  }
  return parseInt(field(b, start, len).trim() || "0", 8);
};

/** PAX extended header records: "<length in bytes> key=value\n", values in UTF-8. */
export function paxRecords(data: Uint8Array): Map<string, string> {
  const out = new Map<string, string>();
  let i = 0;
  while (i < data.length) {
    const sp = data.indexOf(0x20, i);
    if (sp < 0) break;
    const len = parseInt(dec.decode(data.subarray(i, sp)), 10);
    if (!Number.isFinite(len) || len <= sp - i || i + len > data.length) break;
    const rec = dec.decode(data.subarray(sp + 1, i + len)).replace(/\n$/, "");
    const eq = rec.indexOf("=");
    if (eq > 0) out.set(rec.slice(0, eq), rec.slice(eq + 1));
    i += len;
  }
  return out;
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", new Uint8Array(bytes)));
  return [...d.slice(0, 20)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * The archive as a tar stream: gunzipped when it starts with gzip's magic bytes (codeload sends
 * application/x-gzip without Content-Encoding; a proxy that already decompressed it is fine too).
 */
async function gunzipIfNeeded(stream: ReadableStream<Uint8Array>): Promise<ReadableStream<Uint8Array>> {
  const reader = stream.getReader();
  // the first two bytes decide, whatever the chunk sizes
  let first: Uint8Array | undefined;
  while (!first || first.length < 2) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value.length) continue;
    if (!first) first = value;
    else {
      const joined = new Uint8Array(first.length + value.length);
      joined.set(first);
      joined.set(value, first.length);
      first = joined;
    }
  }
  const replay = new ReadableStream<Uint8Array>({
    start(ctl) {
      if (first) ctl.enqueue(first);
    },
    async pull(ctl) {
      const { value, done } = await reader.read();
      if (done) ctl.close();
      else ctl.enqueue(value);
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });
  const gunzip = new DecompressionStream("gzip") as unknown as TransformStream<Uint8Array, Uint8Array>;
  return first && first[0] === 0x1f && first[1] === 0x8b ? replay.pipeThrough(gunzip) : replay;
}

/**
 * Read a .tar.gz stream and return the files `keep` accepts (path without the archive's top-level
 * folder). Everything else is decompressed and skipped, never stored.
 */
export async function readTarGz(stream: ReadableStream<Uint8Array>, keep: (path: string) => boolean): Promise<ArchiveFile[]> {
  const reader = new Bytes((await gunzipIfNeeded(stream)).getReader());
  const out: ArchiveFile[] = [];
  const truncated = () => new DocsSourceError("the archive is truncated", true);
  let longName: string | null = null;
  let pax = new Map<string, string>();
  let entries = 0;
  for (;;) {
    const h = await reader.take(512);
    // An archive ends with zero blocks. Running out of bytes before that, or a block that is not a
    // tar header (an error page, a cut download), must not read as "the repository is empty".
    if (!h) throw truncated();
    if (h.every((x) => x === 0)) break;
    if (!validChecksum(h)) throw new DocsSourceError("not a tar archive (bad header checksum)", true);
    entries++;
    const type = String.fromCharCode(h[156] || 48);
    let size = octal(h, 124, 12);
    const prefix = field(h, 345, 155);
    let name = field(h, 0, 100);
    if (field(h, 257, 5) === "ustar" && prefix) name = `${prefix}/${name}`;
    if (pax.has("path")) name = pax.get("path")!;
    if (longName) name = longName;
    if (pax.has("size")) size = Number(pax.get("size"));
    const padded = Math.ceil(size / 512) * 512;
    if (type === "x" || type === "L" || type === "g") {
      const data = await reader.take(padded);
      if (!data) throw truncated();
      if (type === "x") pax = paxRecords(data.subarray(0, size));
      else if (type === "L") longName = field(data, 0, size);
      continue; // the header applies to the next entry ("g" applies to all, and is not needed)
    }
    pax = new Map();
    longName = null;
    const path = name.split("/").slice(1).join("/"); // "<repo>-<sha>/docs/a.md" → "docs/a.md"
    if ((type === "0" || type === "7") && path && keep(path)) {
      if (size > MAX_DOC_BYTES) {
        if (!(await reader.skip(padded))) throw truncated();
        out.push({ path, size, hash: `size-${size}`, text: null });
        continue;
      }
      const data = await reader.take(padded);
      if (!data) throw truncated();
      const bytes = data.subarray(0, size);
      let text: string | null = null;
      if (!bytes.includes(0)) {
        try {
          text = new TextDecoder("utf-8", { fatal: true, ignoreBOM: false }).decode(bytes);
        } catch {
          text = null;
        }
      }
      out.push({ path, size, hash: await sha256Hex(bytes), text });
    } else if (!(await reader.skip(padded))) throw truncated();
  }
  if (!entries) throw new DocsSourceError("the archive is empty", true);
  return out;
}

/** The header checksum: the sum of its bytes, the checksum field counted as spaces. */
function validChecksum(h: Uint8Array): boolean {
  const stored = parseInt(field(h, 148, 8).trim() || "x", 8);
  if (!Number.isFinite(stored)) return false;
  let sum = 0;
  for (let i = 0; i < 512; i++) sum += i >= 148 && i < 156 ? 32 : h[i];
  return sum === stored;
}

/** The documentation files of a commit, from its archive. */
export async function fetchDocFiles(env: Env, r: RepoRef, commit: string): Promise<ArchiveFile[]> {
  const res = await get(`https://codeload.github.com/${r.owner}/${r.repo}/tar.gz/${commit}`);
  if (!res.ok || !res.body) throw failure(res, `archive of ${repoName(r)}@${commit.slice(0, 7)}`);
  const declared = Number(res.headers.get("content-length") ?? "");
  if (Number.isFinite(declared) && declared > MAX_ARCHIVE_BYTES) {
    await res.body.cancel().catch(() => {});
    throw new DocsSourceError(`archive of ${repoName(r)} is ${Math.round(declared / 1048576)} MB: more than the ${MAX_ARCHIVE_BYTES / 1048576} MB this sync reads`, false);
  }
  let seen = 0;
  const capped = res.body.pipeThrough(
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, ctl) {
        seen += chunk.byteLength;
        if (seen > MAX_ARCHIVE_BYTES) ctl.error(new DocsSourceError(`archive of ${repoName(r)} exceeds ${MAX_ARCHIVE_BYTES / 1048576} MB`, false));
        else ctl.enqueue(chunk);
      },
    }),
  );
  try {
    return await readTarGz(capped, (p) => isDocFile(p, docPaths(env)));
  } catch (e) {
    if (e instanceof DocsSourceError) throw e;
    throw new DocsSourceError(`archive of ${repoName(r)}@${commit.slice(0, 7)} unreadable: ${e instanceof Error ? e.message : String(e)}`, true);
  }
}

/** HMAC-SHA256 signature of a GitHub webhook delivery (X-Hub-Signature-256), compared in constant time. */
export async function verifyWebhook(secret: string, body: Uint8Array, signature: string | null): Promise<boolean> {
  if (!signature?.startsWith("sha256=")) return false;
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = new Uint8Array(await crypto.subtle.sign("HMAC", key, new Uint8Array(body)));
  const expected = [...mac].map((b) => b.toString(16).padStart(2, "0")).join("");
  const given = signature.slice(7).toLowerCase();
  if (given.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < expected.length; i++) diff |= expected.charCodeAt(i) ^ given.charCodeAt(i);
  return diff === 0;
}
