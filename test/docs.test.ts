// The documentation index and help answers: GitHub sync (conditional head check, incremental
// re-chunking, deletions, failures, budget, lock, model change), retrieval, grounded answers with
// checked citations, caching per commit, the gaps log, and the webhook — against a fake GitHub and
// fake Workers AI, with SQLite standing in for D1.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { FakeExec, FakeVectorize, makeEnv, type TestEnv } from "./harness/fakes";
import { installGitHub, makeTarGz, type FakeRepo } from "./harness/github";
import { singular, tokens } from "../src/lib/text";
import { isDocFile, parseRefs, readTarGz } from "../src/docs/github";
import { syncDocs, docsStatus, DOCS_SETTINGS } from "../src/docs/sync";
import { answerHelp, validateHelpReply } from "../src/help/answer";
import { retrieveDocs, coverage, questionTerms } from "../src/help/retrieve";
import { app } from "../src/api";

const DIM = 64;
const EMBED = "@cf/baai/bge-m3";
const HELP = "@cf/meta/llama-3.3-70b-instruct-fp8-fast";
const TOKEN = "admin-secret";

/** A bag-of-words embedder: texts that share words are close. */
function bow(text: string): number[] {
  const v = new Array<number>(DIM).fill(0);
  for (const t of tokens(text, { keepHyphenated: false })) {
    if (t.length < 3) continue;
    let h = 7;
    for (const ch of singular(t)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
    v[h % DIM] += 1;
  }
  const n = Math.hypot(...v) || 1;
  return v.map((x) => x / n);
}

const FILES: Record<string, string> = {
  "README.md": "# Information\n\nStart with the fees page and the wallet page.",
  "guides/fees.md": "---\ntitle: Fees\n---\n# Fees\n\nMinting an artwork is free.\n\n## Selling\n\nA sale on the marketplace pays a 5% fee to the platform.\n\n## Royalties\n\nArtists receive royalties on every resale.",
  "fr/portefeuille.md": "# Portefeuille\n\nLe portefeuille garde vos PXS et vos œuvres.",
  "drafts/next.md": "---\ndraft: true\n---\n# Next\n\nNot published yet.",
  LICENSE: "MIT License",
  "img/logo.png": "\u0000PNG",
};

const commitSha = (n: number) => n.toString(16).padStart(40, "a");

type HelpReply = (input: any) => unknown;

function setup(over: Record<string, string> = {}) {
  const env = makeEnv({ DOCS_REPO: "pixa/info", DOCS_BRANCH: "main", ADMIN_TOKEN: TOKEN, ...over });
  const vec = new FakeVectorize([]);
  (env as any).VEC_DOCS = vec;
  const repo: FakeRepo = { owner: "pixa", repo: "info", head: commitSha(1), files: { ...FILES }, calls: [] };
  installGitHub(repo);
  let helpReply: HelpReply = () => ({ response: { answerable: false, answer: "", sources: [] } });
  env._ai.handler = (model, input) => {
    if (model === EMBED || model.startsWith("@cf/test/")) return { shape: [input.text.length, DIM], data: input.text.map(bow) };
    if (model === HELP) return helpReply(input);
    throw new Error(`unexpected model ${model}`);
  };
  return { env, vec, repo, setHelp: (f: HelpReply) => (helpReply = f) };
}

const REFS = "https://github.com/pixa/info.git/info/refs?service=git-upload-pack";
const ARCHIVE = (sha: string) => `https://codeload.github.com/pixa/info/tar.gz/${sha}`;

const chunkIds = async (env: TestEnv, path: string) =>
  ((await env.DB.prepare("SELECT id, heading FROM doc_chunks WHERE path = ? ORDER BY ord").bind(path).all<{ id: number; heading: string }>()).results ?? []);

afterEach(() => vi.unstubAllGlobals());

describe("reading GitHub without its API", () => {
  it("parses git's ref advertisement (a real one, from the information repository)", () => {
    const refs = parseRefs(new Uint8Array(readFileSync(new URL("./fixtures/information.refs", import.meta.url))));
    expect(refs.get("refs/heads/main")).toBe("6d6647d1a7478ee7e180abfca4a1199a40efa27a");
    expect(refs.get("HEAD")).toBe("6d6647d1a7478ee7e180abfca4a1199a40efa27a");
  });

  it("reads archives written by git archive (as codeload serves them): prefix stripped, binaries and licences out", async () => {
    const stream = (f: string) => new Blob([readFileSync(new URL(`./fixtures/${f}`, import.meta.url))]).stream();
    const info = await readTarGz(stream("information.tar.gz"), (p) => isDocFile(p));
    expect(info.map((f) => [f.path, f.size, f.text])).toEqual([["README.md", 46, "# information\nInformation about the ecosystem\n"]]);
    const files = await readTarGz(stream("docs-archive.tar.gz"), (p) => isDocFile(p));
    const byPath = new Map(files.map((f) => [f.path, f]));
    expect([...byPath.keys()].sort()).toEqual([
      "big.md",
      "fr/sécurité du portefeuille.md",
      "guides/a-folder-name-long-enough-that-the-path-goes-beyond-the-hundred-bytes-of-a-ustar-name-field/page.md",
      "latin.md",
    ]);
    expect(byPath.get("fr/sécurité du portefeuille.md")?.text).toContain("Gardez vos clés privées");
    expect(byPath.get("big.md")).toMatchObject({ size: 600008, text: null }); // over 512 KB: skipped unread
    expect(byPath.get("latin.md")?.text).toBeNull(); // not UTF-8
    expect(byPath.get("guides/a-folder-name-long-enough-that-the-path-goes-beyond-the-hundred-bytes-of-a-ustar-name-field/page.md")?.hash).toMatch(/^[0-9a-f]{40}$/);
  });

  it("also reads a tar that arrives already decompressed", async () => {
    const gz = readFileSync(new URL("./fixtures/information.tar.gz", import.meta.url));
    const tar = new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip") as unknown as TransformStream<Uint8Array, Uint8Array>);
    expect((await readTarGz(tar, (p) => isDocFile(p))).map((f) => f.path)).toEqual(["README.md"]);
  });

  it("reads the archives the fake GitHub writes (PAX long names, unicode)", async () => {
    const tgz = await makeTarGz({ "a.md": "# A", [`${"d/".repeat(60)}deep.md`]: "# Deep", "é/ü.md": "# U" }, "info-x");
    const files = await readTarGz(new Blob([tgz]).stream(), () => true);
    expect(files.map((f) => f.path)).toEqual(["a.md", `${"d/".repeat(60)}deep.md`, "é/ü.md"]);
  });
});

describe("docs sync from GitHub", () => {
  let s: ReturnType<typeof setup>;
  beforeEach(() => {
    s = setup();
  });

  it("indexes the Markdown files, skips drafts and non-documents, records the commit, embeds every chunk", async () => {
    const r = await syncDocs(s.env);
    expect(r.status).toBe("synced");
    expect([...r.indexed].sort()).toEqual(["README.md", "fr/portefeuille.md", "guides/fees.md"]);
    expect(r.skipped).toEqual([{ path: "drafts/next.md", reason: "draft or noindex" }]);
    expect(r.files).toBe(4);
    expect(r.commit).toBe(commitSha(1));
    const st = await docsStatus(s.env);
    expect(st.commit).toBe(commitSha(1));
    expect(st.chunks).toBe(5); // readme 1, fees 3, portefeuille 1
    expect(st.embedded).toBe(5);
    expect(s.vec.v.size).toBe(5);
    expect(r.embedded).toBe(5);
    const fees = await chunkIds(s.env, "guides/fees.md");
    expect(fees.map((c) => c.heading)).toEqual(["", "Selling", "Royalties"]);
    // two requests, both with a User-Agent (the fake refuses others), none to the REST API
    expect(s.repo.calls).toEqual([REFS, ARCHIVE(commitSha(1))]);
  });

  it("nothing changed: one small request and no other work", async () => {
    await syncDocs(s.env);
    s.repo.calls.length = 0;
    const r = await syncDocs(s.env);
    expect(r.status).toBe("unchanged");
    expect(s.repo.calls).toEqual([REFS]);
    expect(r.embedded).toBe(0);
  });

  it("an edited file is re-chunked: unchanged chunks keep their row and vector, the changed one is replaced", async () => {
    await syncDocs(s.env);
    const before = await chunkIds(s.env, "guides/fees.md");
    s.repo.files["guides/fees.md"] = String(s.repo.files["guides/fees.md"]).replace("pays a 5% fee", "pays a 4% fee");
    s.repo.head = commitSha(2);
    const r = await syncDocs(s.env);
    expect(r.status).toBe("synced");
    expect(r.indexed).toEqual(["guides/fees.md"]); // the other files' content is unchanged
    const after = await chunkIds(s.env, "guides/fees.md");
    expect(after[0].id).toBe(before[0].id);
    expect(after[2].id).toBe(before[2].id);
    expect(after[1].id).not.toBe(before[1].id);
    expect(s.vec.v.has(String(before[1].id))).toBe(false);
    expect(s.vec.v.has(String(after[1].id))).toBe(true);
    expect(r.embedded).toBe(1);
    const text = await s.env.DB.prepare("SELECT text FROM doc_chunks WHERE id = ?").bind(after[1].id).first<{ text: string }>();
    expect(text?.text).toContain("4% fee");
  });

  it("a file removed from the repository loses its chunks and vectors", async () => {
    await syncDocs(s.env);
    const ids = await chunkIds(s.env, "fr/portefeuille.md");
    delete s.repo.files["fr/portefeuille.md"];
    s.repo.head = commitSha(3);
    const r = await syncDocs(s.env);
    expect(r.removed).toEqual(["fr/portefeuille.md"]);
    expect(await chunkIds(s.env, "fr/portefeuille.md")).toEqual([]);
    expect(s.vec.v.has(String(ids[0].id))).toBe(false);
    expect((await s.env.DB.prepare("SELECT COUNT(*) AS n FROM docs WHERE path = ?").bind("fr/portefeuille.md").first<{ n: number }>())?.n).toBe(0);
    // and full text no longer finds it
    expect((await retrieveDocs(s.env, "portefeuille PXS", { vectors: false })).hits.map((h) => h.path)).not.toContain("fr/portefeuille.md");
  });

  it("a failed download leaves the commit unrecorded, and the next run does the work", async () => {
    s.repo.archiveStatus = 502;
    const r = await syncDocs(s.env);
    expect(r.status).toBe("error");
    expect(r.error).toContain("HTTP 502");
    expect((await docsStatus(s.env)).commit).toBeNull();
    s.repo.archiveStatus = undefined;
    expect((await syncDocs(s.env)).status).toBe("synced");
    expect((await docsStatus(s.env)).commit).toBe(commitSha(1));
  });

  it("files that are too large or not text are skipped, and stay skipped until they change", async () => {
    s.repo.files["big.md"] = `# Big\n\n${"word ".repeat(120_000)}`;
    s.repo.files["latin.md"] = new Uint8Array([0x23, 0x20, 0xff, 0xfe, 0x41]);
    const r = await syncDocs(s.env);
    expect(r.skipped).toEqual(expect.arrayContaining([{ path: "big.md", reason: "larger than 512 KB" }, { path: "latin.md", reason: "not UTF-8 text" }]));
    s.repo.head = commitSha(5);
    const again = await syncDocs(s.env);
    expect(again.status).toBe("synced");
    expect(again.skipped).toEqual([]);
    expect(again.indexed).toEqual([]);
  });

  it("a run indexes at most maxFiles; the next runs continue until the commit is complete", async () => {
    const r1 = await syncDocs(s.env, { maxFiles: 2 });
    expect(r1.status).toBe("partial");
    expect(r1.pending).toBe(2);
    expect((await docsStatus(s.env)).commit).toBeNull();
    const r2 = await syncDocs(s.env, { maxFiles: 2 });
    expect(r2.status).toBe("synced");
    expect(r2.indexed.length + r2.skipped.length).toBe(2);
    expect((await docsStatus(s.env)).commit).toBe(commitSha(1));
  });

  it("only one sync at a time; an expired lock is taken over", async () => {
    const t = Math.floor(Date.now() / 1000);
    await s.env.DB.prepare("INSERT INTO settings (k, v) VALUES (?, ?)").bind(DOCS_SETTINGS.lock, `${t + 60}.other`).run();
    expect((await syncDocs(s.env)).status).toBe("locked");
    await s.env.DB.prepare("UPDATE settings SET v = ? WHERE k = ?").bind(`${t - 1}.other`, DOCS_SETTINGS.lock).run();
    expect((await syncDocs(s.env)).status).toBe("synced");
    expect((await s.env.DB.prepare("SELECT COUNT(*) AS n FROM settings WHERE k = ?").bind(DOCS_SETTINGS.lock).first<{ n: number }>())?.n).toBe(0);
  });

  it("a new DOCS_EMBED_MODEL re-embeds every chunk; force re-indexes files but keeps unchanged vectors", async () => {
    await syncDocs(s.env);
    s.env.DOCS_EMBED_MODEL = "@cf/test/other";
    const r = await syncDocs(s.env);
    expect(r.status).toBe("unchanged");
    expect(r.embedded).toBe(5);
    const forced = await syncDocs(s.env, { force: true });
    expect(forced.status).toBe("synced");
    expect(forced.indexed.length).toBe(3);
    expect(forced.embedded).toBe(0);
  });

  it("a vector of the wrong size is refused (DOCS_EMBED_DIM): chunks stay searchable by their words", async () => {
    s.env.DOCS_EMBED_DIM = "1024";
    const r = await syncDocs(s.env);
    expect(r.status).toBe("synced");
    expect(r.embedded).toBe(0);
    expect(r.notes.join(" ")).toContain("gives 64 dimensions, DOCS_EMBED_DIM (and VEC_DOCS) 1024");
    expect((await retrieveDocs(s.env, "fee sale", { vectors: false })).hits[0].path).toBe("guides/fees.md");
  });

  it("DOCS_REPO=off disables the sync", async () => {
    s.env.DOCS_REPO = "off";
    expect((await syncDocs(s.env)).status).toBe("disabled");
    expect(s.repo.calls).toEqual([]);
  });
});

describe("retrieval", () => {
  it("question words without the question's function words, matched with light stemming", () => {
    expect(questionTerms("How do I upload an artwork to Pixagram?")).toEqual(["upload", "artwork", "pixagram"]);
    expect(questionTerms("Comment fonctionnent les redevances ?")).toEqual(["fonctionnent", "redevances"]);
    expect(coverage(["royalties"], "Artists receive a royalty on resales")).toBe(1);
    expect(coverage(["fonctionnent"], "Comment cela fonctionne")).toBe(1);
    expect(coverage(["mint", "pixagram"], "Pixagram is a network")).toBeCloseTo(1 / 3);
  });
});

describe("help answers", () => {
  let s: ReturnType<typeof setup>;
  beforeEach(() => {
    s = setup();
  });

  it("no documentation yet: says so, without a model call", async () => {
    const r = await answerHelp(s.env, "What are the fees?");
    expect(r.status).toBe("no_docs");
    expect(r.answer_text).toBe("The Pixagram documentation is not available yet.");
    expect(s.env._ai.calls.length).toBe(0);
  });

  it("answers from the excerpts; invented links and citations are removed; sources link to the section", async () => {
    await syncDocs(s.env);
    let seen: any = null;
    s.setHelp((input) => {
      seen = input;
      return { response: { answerable: true, answer: "A sale pays a 5% fee [1]. See https://evil.example/x [9] or [the guide](https://evil.example/y).", sources: [1, 9] } };
    });
    const r = await answerHelp(s.env, "What fee does a sale pay?");
    expect(r.status).toBe("answered");
    expect(r.answer_text).toBe("A sale pays a 5% fee [1]. See or the guide.");
    expect(r.sources.map((x) => [x.n, x.url])).toEqual([[1, "https://github.com/pixa/info/blob/main/guides/fees.md#selling"]]);
    expect(r.docs_commit).toBe(commitSha(1));
    expect(seen.response_format.type).toBe("json_schema");
    expect(seen.messages[0].content).toContain("Write the answer in English");
    expect(seen.messages[1].content).toContain("[1] Fees — Selling\nA sale on the marketplace pays a 5% fee to the platform.");
    expect(seen.messages[1].content).toContain("Question: What fee does a sale pay?");
  });

  it("the same question comes from the cache until the documentation changes", async () => {
    await syncDocs(s.env);
    s.setHelp(() => ({ response: { answerable: true, answer: "Minting is free [1].", sources: [1] } }));
    const a = await answerHelp(s.env, "Is minting an artwork free?");
    expect(a.status).toBe("answered");
    const calls = s.env._ai.calls.filter((c) => c.model === HELP).length;
    const b = await answerHelp(s.env, "is minting an artwork free?");
    expect(b.cached).toBe(true);
    expect(b.answer_text).toBe("Minting is free [1].");
    expect(s.env._ai.calls.filter((c) => c.model === HELP).length).toBe(calls);
    s.repo.files["guides/fees.md"] = `${s.repo.files["guides/fees.md"]}\n\n## Gifts\n\nGifts are free too.`;
    s.repo.head = commitSha(9);
    await syncDocs(s.env);
    const c = await answerHelp(s.env, "Is minting an artwork free?");
    expect(c.cached).toBeUndefined();
    expect(s.env._ai.calls.filter((x) => x.model === HELP).length).toBe(calls + 1);
  });

  it("nothing relevant: not_found and no model call", async () => {
    await syncDocs(s.env);
    const r = await answerHelp(s.env, "How do I bake sourdough bread?");
    expect(r.status).toBe("not_found");
    expect(s.env._ai.calls.filter((c) => c.model === HELP)).toEqual([]);
  });

  it("the model finds no answer in the excerpts: not_found with the closest pages", async () => {
    await syncDocs(s.env);
    s.setHelp(() => ({ response: { answerable: false, answer: "", sources: [] } }));
    const r = await answerHelp(s.env, "Who receives royalties on a resale, and how much?");
    expect(r.status).toBe("not_found");
    expect(r.sources[0].path).toBe("guides/fees.md");
  });

  it("a failing or unusable model: the excerpts themselves, not cached", async () => {
    await syncDocs(s.env);
    s.setHelp(() => {
      throw new Error("model overloaded");
    });
    const r = await answerHelp(s.env, "What fee does a sale pay?");
    expect(r.status).toBe("excerpts");
    expect(r.sources[0].heading).toBe("Selling");
    expect(r.notes.join(" ")).toContain("model overloaded");
    s.setHelp(() => ({ response: "not json at all" }));
    expect((await answerHelp(s.env, "What fee does a sale pay?")).status).toBe("excerpts");
  });

  it("answers in the question's language", async () => {
    await syncDocs(s.env);
    let seen: any = null;
    s.setHelp((input) => {
      seen = input;
      return { response: { answerable: true, answer: "Dans le portefeuille [1].", sources: [1] } };
    });
    const r = await answerHelp(s.env, "Où sont gardés mes PXS ?");
    expect(r.lang).toBe("fr");
    expect(r.status).toBe("answered");
    expect(seen.messages[0].content).toContain("Write the answer in French");
    expect(r.sources[0].path).toBe("fr/portefeuille.md");
  });

  it("validates replies: answerable needs text, sources must exist", () => {
    expect(validateHelpReply({ answerable: true, answer: "", sources: [1] }, 2, "")).toEqual({ answerable: false, answer: "", sources: [] });
    expect(validateHelpReply('{"answerable": true, "answer": "Yes [2] [3].", "sources": [5]}', 2, "")).toEqual({ answerable: true, answer: "Yes [2].", sources: [2] });
    expect(validateHelpReply({ answer: "x" }, 2, "")).toBeNull();
    expect(validateHelpReply({ answerable: true, answer: "See https://github.com/pixa/info/blob/main/a.md.", sources: [] }, 1, "https://github.com/pixa/info/blob/main/a.md")).toEqual({
      answerable: true,
      answer: "See https://github.com/pixa/info/blob/main/a.md.",
      sources: [],
    });
  });
});

describe("help over HTTP: /help, the gaps log, the webhook, admin", () => {
  let s: ReturnType<typeof setup>;
  let exec: FakeExec;
  const call = async (path: string, init?: RequestInit & { admin?: boolean }) => {
    const headers = new Headers(init?.headers);
    if (init?.admin) headers.set("authorization", `Bearer ${TOKEN}`);
    const res = await app.fetch(new Request(`https://search.test${path}`, { ...init, headers }), s.env, exec as unknown as ExecutionContext);
    await exec.settle();
    const text = await res.text();
    return { status: res.status, body: text ? JSON.parse(text) : null };
  };
  const sign = async (secret: string, body: string) => {
    const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    return `sha256=${[...new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(body)))].map((b) => b.toString(16).padStart(2, "0")).join("")}`;
  };
  beforeEach(() => {
    s = setup({ GITHUB_WEBHOOK_SECRET: "hook-secret" });
    exec = new FakeExec();
  });

  it("admin sync and status; /help answers; unanswered questions are listed as gaps", async () => {
    const sync = await call("/admin/docs/sync", { method: "POST", admin: true });
    expect(sync.body.status).toBe("synced");
    expect((await call("/admin/docs", { admin: true })).body).toMatchObject({ repo: "pixa/info", branch: "main", commit: commitSha(1), chunks: 5, embedded: 5 });
    s.setHelp(() => ({ response: { answerable: true, answer: "Minting is free [1].", sources: [1] } }));
    const r = await call("/help?q=" + encodeURIComponent("Is minting free?"));
    expect(r.body.status).toBe("answered");
    await call("/help", { method: "POST", body: JSON.stringify({ question: "How do I bake bread?" }), headers: { "content-type": "application/json" } });
    await call("/help?q=" + encodeURIComponent("how do I bake bread?"));
    const gaps = (await call("/admin/docs/gaps", { admin: true })).body;
    expect(gaps.unanswered).toEqual([expect.objectContaining({ q: "how do i bake bread?", n: 2, status: "not_found" })]);
    expect(gaps.by_status).toEqual(expect.arrayContaining([{ status: "answered", n: 1 }, { status: "not_found", n: 2 }]));
  });

  it("the push webhook checks the signature, the repository and the branch, then syncs", async () => {
    const push = JSON.stringify({ ref: "refs/heads/main", after: commitSha(1), repository: { full_name: "pixa/info" } });
    expect((await call("/webhooks/github", { method: "POST", body: push, headers: { "x-github-event": "push", "x-hub-signature-256": "sha256=00" } })).status).toBe(401);
    const ping = JSON.stringify({ zen: "hi" });
    expect((await call("/webhooks/github", { method: "POST", body: ping, headers: { "x-github-event": "ping", "x-hub-signature-256": await sign("hook-secret", ping) } })).body).toEqual({ ok: true, pong: true });
    const other = JSON.stringify({ ref: "refs/heads/dev", repository: { full_name: "pixa/info" } });
    expect((await call("/webhooks/github", { method: "POST", body: other, headers: { "x-github-event": "push", "x-hub-signature-256": await sign("hook-secret", other) } })).body.ignored).toBe(
      "another repository or branch",
    );
    expect((await docsStatus(s.env)).chunks).toBe(0);
    const ok = await call("/webhooks/github", { method: "POST", body: push, headers: { "x-github-event": "push", "x-hub-signature-256": await sign("hook-secret", push) } });
    expect(ok.status).toBe(202);
    expect((await docsStatus(s.env)).chunks).toBe(5); // the sync ran in waitUntil
    expect(s.repo.calls).toEqual([ARCHIVE(commitSha(1))]); // the push named its commit: no ref lookup
  });

  it("without a webhook secret the route does not exist", async () => {
    s.env.GITHUB_WEBHOOK_SECRET = undefined;
    expect((await call("/webhooks/github", { method: "POST", body: "{}" })).status).toBe(404);
  });
});
