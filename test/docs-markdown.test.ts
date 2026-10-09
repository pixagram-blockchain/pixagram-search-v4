import { describe, expect, it } from "vitest";
import { chunkDoc, githubSlug, inlineText, parseDoc, splitFrontMatter, splitWords } from "../src/docs/markdown";
import { isDocFile, verifyWebhook } from "../src/docs/github";

const PAGE = `---
title: "Royalties"
lang: en
keywords: [resale, secondary sales]
---

Intro text before any heading.

# Royalties

Artists earn **royalties** on every resale, see [the marketplace](https://pixagram.com/market).

## How they are paid

Paid in PXS to the artist's wallet. ![a coin](coin.png)

\`\`\`
# not a heading inside code
pay --to artist
\`\`\`

### Edge cases

| case | result |
|------|--------|
| burned | nothing |

#### Details stay inside

Small print.

## How they are paid

A second section with the same title.
`;

describe("markdown: front matter, inline text, anchors", () => {
  it("reads the front matter subset and leaves the body", () => {
    const { meta, body } = splitFrontMatter(PAGE);
    expect(meta).toEqual({ title: "Royalties", lang: "en", keywords: ["resale", "secondary sales"] });
    expect(body.startsWith("\nIntro text")).toBe(true);
    expect(splitFrontMatter("---\ndraft: true\ntags:\n  - a\n  - b\n---\nx").meta).toEqual({ draft: true, tags: ["a", "b"] });
    expect(splitFrontMatter("no front matter").meta).toEqual({});
  });

  it("turns inline Markdown into text", () => {
    expect(inlineText("Earn **royalties** on [the market](https://x.y/a_(b)) with `code` and ![alt](i.png) <b>html</b> &amp; _emphasis_ snake_case_word")).toBe(
      "Earn royalties on the market with code and alt html & emphasis snake_case_word",
    );
  });

  it("makes GitHub's anchors, repeats numbered", () => {
    const seen = new Map<string, number>();
    expect(githubSlug("How they are paid", seen)).toBe("how-they-are-paid");
    expect(githubSlug("How they are paid", seen)).toBe("how-they-are-paid-1");
    expect(githubSlug("Qu'est-ce que PXS ?", new Map())).toBe("quest-ce-que-pxs-");
    expect(githubSlug("Frais & coûts (2026)", new Map())).toBe("frais--coûts-2026");
  });
});

describe("markdown: sections and chunks", () => {
  const doc = parseDoc(PAGE, "marketplace/royalties.md");

  it("cuts at headings 1-3, ignores # inside code, keeps 4-6 in their section", () => {
    expect(doc.title).toBe("Royalties");
    expect(doc.lang).toBe("en");
    expect(doc.sections.map((s) => [s.level, s.path.join(" › "), s.anchor])).toEqual([
      [0, "", ""],
      [1, "Royalties", "royalties"],
      [2, "Royalties › How they are paid", "how-they-are-paid"],
      [3, "Royalties › How they are paid › Edge cases", "edge-cases"],
      [2, "Royalties › How they are paid", "how-they-are-paid-1"],
    ]);
    const paid = doc.sections[2].text;
    expect(paid).toContain("Paid in PXS to the artist's wallet. a coin");
    expect(paid).toContain("# not a heading inside code");
    expect(paid).not.toContain("```");
    const edge = doc.sections[3].text;
    expect(edge).toContain("case | result");
    expect(edge).not.toContain("---");
    expect(edge).toContain("Details stay inside");
    expect(edge).toContain("Small print.");
  });

  it("chunks drop the document title from the heading path and carry the anchor; keywords go with the first chunk", () => {
    const chunks = chunkDoc(doc);
    expect(chunks.map((c) => [c.ord, c.heading, c.anchor])).toEqual([
      [0, "", ""],
      [1, "", "royalties"],
      [2, "How they are paid", "how-they-are-paid"],
      [3, "How they are paid › Edge cases", "edge-cases"],
      [4, "How they are paid", "how-they-are-paid-1"],
    ]);
    expect(chunks[0].text).toBe("Intro text before any heading.\n\nresale, secondary sales");
  });

  it("titles and languages from headings, file names, folders and suffixes", () => {
    expect(parseDoc("# Wallets\n\ntext", "guides/wallets.md").title).toBe("Wallets");
    expect(parseDoc("text only", "guides/getting-started.md").title).toBe("Getting started");
    expect(parseDoc("text only", "guides/README.md").title).toBe("Guides");
    expect(parseDoc("Le portefeuille est la clé de votre compte et de vos œuvres.", "fr/wallet.md").lang).toBe("fr");
    expect(parseDoc("text", "wallet.de.md").lang).toBe("de");
    expect(parseDoc("Comment créer un compte ? Il faut une clé et un portefeuille pour les œuvres.", "wallet.md").lang).toBe("fr");
    const setext = parseDoc("Fees\n====\n\nZero.\n\nPayouts\n-------\n\nWeekly.", "x.md");
    expect(setext.sections.map((s) => s.path.join("/"))).toEqual(["Fees", "Fees/Payouts"]);
  });

  it("drafts are skipped; long sections are cut at paragraphs, then sentences, then words", () => {
    expect(parseDoc("---\ndraft: true\n---\n# X\ny", "x.md").meta.skip).toBe(true);
    const para = (n: number, w: string) => Array.from({ length: n }, () => w).join(" ");
    const parts = splitWords([para(100, "a"), para(100, "b"), para(100, "c")].join("\n\n"), 250);
    expect(parts.map((p) => p.split(/\s+/).length)).toEqual([200, 100]);
    expect(splitWords(para(600, "w"), 250).map((p) => p.split(/\s+/).length)).toEqual([250, 250, 100]);
  });
});

describe("github helpers", () => {
  it("documentation files: Markdown and text outside tooling folders and licences, optionally under DOCS_PATHS", () => {
    expect(isDocFile("README.md")).toBe(true);
    expect(isDocFile("guides/wallet.mdx")).toBe(true);
    expect(isDocFile("LICENSE")).toBe(false);
    expect(isDocFile("LICENSE.md")).toBe(false);
    expect(isDocFile(".github/ISSUE_TEMPLATE.md")).toBe(false);
    expect(isDocFile("img/logo.png")).toBe(false);
    expect(isDocFile("guides/wallet.md", ["faq"])).toBe(false);
    expect(isDocFile("faq/fees.md", ["faq"])).toBe(true);
    expect(isDocFile("faq.md", ["faq.md"])).toBe(true);
  });

  it("verifies GitHub webhook signatures (HMAC-SHA256)", async () => {
    const body = new TextEncoder().encode('{"ref":"refs/heads/main"}');
    const key = await crypto.subtle.importKey("raw", new TextEncoder().encode("s3cret"), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
    const sig = [...new Uint8Array(await crypto.subtle.sign("HMAC", key, body))].map((b) => b.toString(16).padStart(2, "0")).join("");
    expect(await verifyWebhook("s3cret", body, `sha256=${sig}`)).toBe(true);
    expect(await verifyWebhook("other", body, `sha256=${sig}`)).toBe(false);
    expect(await verifyWebhook("s3cret", body, null)).toBe(false);
    expect(await verifyWebhook("s3cret", body, `sha1=${sig}`)).toBe(false);
  });
});
