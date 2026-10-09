// A fake GitHub for the documentation sync: git's ref advertisement and codeload archives
// (ustar + PAX headers, gzip), built from an in-memory repository.

import { vi } from "vitest";

const enc = new TextEncoder();

function header(name: string, size: number, type: string): Uint8Array {
  const h = new Uint8Array(512);
  const put = (s: string, off: number, len: number) => h.set(enc.encode(s).subarray(0, len), off);
  put(name, 0, 100);
  put("0000644\0", 100, 8);
  put("0000000\0", 108, 8);
  put("0000000\0", 116, 8);
  put(`${size.toString(8).padStart(11, "0")}\0`, 124, 12);
  put("15000000000\0", 136, 12);
  put("        ", 148, 8);
  h[156] = type.charCodeAt(0);
  put("ustar\0", 257, 6);
  put("00", 263, 2);
  let sum = 0;
  for (const b of h) sum += b;
  put(`${sum.toString(8).padStart(6, "0")}\0 `, 148, 8);
  return h;
}

function padded(data: Uint8Array): Uint8Array[] {
  const pad = (512 - (data.length % 512)) % 512;
  return [data, new Uint8Array(pad)];
}

/** A PAX record: "<length> key=value\n", the length counting itself, in bytes. */
function paxRecord(key: string, value: string): string {
  const body = ` ${key}=${value}\n`;
  const n = enc.encode(body).length;
  let len = n + 1;
  while (len !== n + String(len).length) len = n + String(len).length;
  return `${len}${body}`;
}

/** A gzip'd tar of the files under "<prefix>/", as git archive (and GitHub) writes it. */
export async function makeTarGz(files: Record<string, string | Uint8Array>, prefix: string): Promise<Uint8Array<ArrayBuffer>> {
  const parts: Uint8Array[] = [];
  const comment = enc.encode(paxRecord("comment", prefix));
  parts.push(header("pax_global_header", comment.length, "g"), ...padded(comment));
  parts.push(header(`${prefix}/`, 0, "5"));
  for (const [path, content] of Object.entries(files)) {
    const data = typeof content === "string" ? enc.encode(content) : content;
    const name = `${prefix}/${path}`;
    if (enc.encode(name).length > 100) {
      const pax = enc.encode(paxRecord("path", name));
      parts.push(header("PaxHeader", pax.length, "x"), ...padded(pax));
    }
    parts.push(header(name, data.length, "0"), ...padded(data));
  }
  parts.push(new Uint8Array(1024));
  const tar = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) (tar.set(p, o), (o += p.length));
  const gz = new Response(new Blob([tar]).stream().pipeThrough(new CompressionStream("gzip")));
  return new Uint8Array(await gz.arrayBuffer());
}

const pkt = (s: string) => `${(s.length + 4).toString(16).padStart(4, "0")}${s}`;

export function refAdvertisement(head: string, branch = "main"): string {
  return `${pkt("# service=git-upload-pack\n")}0000${pkt(`${head} HEAD\0multi_ack symref=HEAD:refs/heads/${branch}\n`)}${pkt(`${head} refs/heads/${branch}\n`)}0000`;
}

export interface FakeRepo {
  owner: string;
  repo: string;
  head: string;
  files: Record<string, string | Uint8Array>;
  /** answer the archive request with this HTTP status instead */
  archiveStatus?: number;
  calls: string[];
}

export function installGitHub(r: FakeRepo) {
  vi.stubGlobal("fetch", async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input instanceof Request ? input.url : input);
    r.calls.push(url);
    if (!new Headers(init?.headers).get("user-agent")) return new Response("User-Agent required", { status: 403 });
    if (url === `https://github.com/${r.owner}/${r.repo}.git/info/refs?service=git-upload-pack`) {
      return new Response(refAdvertisement(r.head), { status: 200, headers: { "content-type": "application/x-git-upload-pack-advertisement" } });
    }
    const m = new RegExp(`^https://codeload\\.github\\.com/${r.owner}/${r.repo}/tar\\.gz/([0-9a-f]{40})$`).exec(url);
    if (m) {
      if (r.archiveStatus) return new Response("unavailable", { status: r.archiveStatus });
      const body = await makeTarGz(r.files, `${r.repo}-${m[1]}`);
      return new Response(body, { status: 200, headers: { "content-type": "application/x-gzip", "content-length": String(body.length) } });
    }
    throw new Error(`unexpected fetch ${url}`);
  });
}
