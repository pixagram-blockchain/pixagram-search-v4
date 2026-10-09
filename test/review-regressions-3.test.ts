// Regressions for what the verification of the third review's fixes found (before the first
// deployment): links in help answers, platform questions routed to /ask, post questions routed to
// search, and "not found" help answers cached too long. Numbered as in the verification report;
// the last groups hold what later verifications of these fixes found.

import MarkdownIt from "markdown-it";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FakeExec, FakeVectorize, makeEnv } from "./harness/fakes";
import { installGitHub } from "./harness/github";
import { app } from "../src/api";
import { planQuery } from "../src/search/planner";
import { routeQuery, type Route } from "../src/search/router";
import { ANSWER_TTL, MAX_ANSWER_CHARS, NOT_FOUND_TTL, answerHelp, sanitizeAnswer, validateHelpReply } from "../src/help/answer";
import { inlineText, parseDoc } from "../src/docs/markdown";
import { syncDocs } from "../src/docs/sync";

const NOW = Date.UTC(2026, 9, 5, 12) / 1000;
const AUTHORS = new Set(["alice", "bob", "laura", "matias"]);

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

async function route(q: string, docs = 0): Promise<Route> {
  const plan = planQuery(q, { mode: "ask", authors: AUTHORS, now: NOW });
  return (await routeQuery(q, plan, { docsScore: async () => docs, isTitle: async () => false })).route;
}

// ---- 1 and 7: the answer's text ----------------------------------------------------------------

/** The sections given to the model: the only addresses an answer may contain. */
const SECTIONS = [
  "https://github.com/pixa/info/blob/main/wallet.md#settings",
  "Change it in https://pixagram.com/wallet/settings. Pixel art: https://en.wikipedia.org/wiki/Pixel_art_(disambiguation).",
  "Write to support@pixagram.com.",
].join("\n");
const clean = (answer: string, n = 2) => sanitizeAnswer(answer, n, SECTIONS);

// What a client might do with answer_text: CommonMark with raw HTML allowed and a linkifier that
// also takes bare domain names ("evil.com/x"), "www." and e-mail addresses.
const md = new MarkdownIt({ html: true, linkify: true });
const RENDERED_TAGS = /<(?!\/?(?:p|a|em|strong|code|pre|ul|ol|li|blockquote|h[1-6]|hr|br|s|table|thead|tbody|tr|th|td)\b)[a-z!?/]/i;

/** Where the rendered answer would send a click or a request. */
function targets(text: string): string[] {
  return [...md.render(text).matchAll(/\b(?:href|src|action|formaction|srcset|poster|data)\s*=\s*"([^"]*)"/gi)].map((m) => m[1]);
}
/**
 * A link target that goes where the documentation goes: mail to its domain, or a page on a host it
 * names. Linkifiers take more characters into an e-mail's local part (";", "$") or a host ("a
 * href=pixagram.com" links to "href=pixagram.com") than an address has: a host that no one can
 * register ("=" or "_" in it, a top-level domain of one letter or with digits) reaches nobody.
 */
const DOC_HOSTS = ["pixagram.com", "www.pixagram.com", "github.com", "www.github.com", "en.wikipedia.org"];
const reachable = (host: string) => /^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+([a-z]{2,63}|xn--[a-z0-9-]+)$/.test(host);
function fromDocs(href: string): boolean {
  if (/^mailto:/i.test(href)) {
    let h = href;
    try {
      h = decodeURIComponent(href);
    } catch {}
    const domain = h.replace(/^.*@/, "").toLowerCase().replace(/[^a-z0-9.-]+$/, "");
    return DOC_HOSTS.includes(domain) || !reachable(domain);
  }
  try {
    const u = new URL(href);
    const host = u.hostname.toLowerCase();
    return /^https?:$/.test(u.protocol) && (DOC_HOSTS.includes(host) || !reachable(host));
  } catch {
    return false;
  }
}
/** The rendered text, as a reader sees it. */
function seen(text: string): string {
  return md
    .render(text)
    .replace(/<[^>]*>/g, "")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, "&");
}
function expectSafe(input: string, out = clean(input)) {
  for (const t of targets(out)) expect(fromDocs(t), `${JSON.stringify(input)} → ${JSON.stringify(out)} links to ${t}`).toBe(true);
  expect(md.render(out), `${JSON.stringify(input)} → raw HTML`).not.toMatch(RENDERED_TAGS);
  expect(seen(out), `${JSON.stringify(input)} → shows an address`).not.toMatch(/evil(?:\.|&period;)[a-z]{2}/i);
}

const ZWSP = String.fromCharCode(0x200b);
const RLO = String.fromCharCode(0x202e);

describe("1. help answers: no link the documentation does not contain, whatever the syntax", () => {
  it("the report's inputs (GFM and CommonMark links) become text", () => {
    const cases: Array<[string, string]> = [
      ["Visit (https://evil.example/phish) now [1].", "Visit now [1]."],
      ["Visit (www.evil.example/phish) now [1].", "Visit now [1]."],
      ["[a [b] c](https://evil.example/phish)", "a [b] c"],
      ["[a\\]b](https://evil.example/phish)", "[a]b]"],
      ["[`]`](https://evil.example/phish)", "`]`"],
      ["[Pixagram\nsupport](https://evil.example/phish)", "Pixagram\nsupport"],
      ['[support](https://evil.example/phish "Official\nsupport")', "support"],
      ["[Pixagram support]x@y.zz(https://evil.example/phish)", "[Pixagram support]"],
      ["Contact [support] [1].\n\n> [support]:https://evil.example/phish", "Contact [support] [1]."],
      ["Contact [support] [1].\n\n- [support]:https://evil.example/phish", "Contact [support] [1]."],
      ["> [support]:javascript:alert(1)\n\nSee [support].", "See [support]."],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
    expect(validateHelpReply({ answerable: true, answer: "Visit (https://evil.example/phish) [1].", sources: [1] }, 1, SECTIONS)).toEqual({ answerable: true, answer: "Visit [1].", sources: [1] });
  });

  it("addresses cannot be smuggled in pieces: entities, escapes, invisible characters, joins", () => {
    const cases: Array<[string, string]> = [
      ["www&period;evil&period;com and www&#46;evil&#x2e;com", "and"],
      ["&#104;ttps://evil.com/x and https&colon;//evil.com", "and"],
      ["www&amp;#46;evil&amp;#46;com", "www& #46;evil& #46;com"],
      ["www\\.evil\\.com/x and evil\\.com", "and"],
      [`Go to evi${ZWSP}l.com now`, "Go to now"],
      [`Go to ${RLO}moc.live now`, "Go to now"],
      ["evil[9].com [1]", "[1]"],
      ["evil<b></b>.com", ""],
      ["evil .com", ""],
      ["https://pixagram.com@evil.com/ and https://pixagram.com'@evil.com/", "and"],
      ["HTTPS://EVIL.COM/X and Www.Evil.Com", "and"],
      ["see 203.0.113.9/login [1]", "see [1]"],
      ["mail x@evil.com or mailto:x@evil.com", "mail or"],
      ["Go to pixagram-support.com to recover your wallet [1].", "Go to to recover your wallet [1]."],
      ["See pixagram.com.evil.com and evil.com/pixagram.com", "See and"],
      ["<img src=x onerror=alert(1)> <script>alert(1)</script>", "alert(1)"],
      ["&lt;a href=&quot;https://evil.com&quot;&gt;x&lt;/a&gt;", "x"],
      ["Unknown &foo; entity", "Unknown & foo; entity"],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
  });

  it("what stays can never make a link, a definition, an autolink or a tag", () => {
    const out = clean("a](b c]:d <e </h <!-- x <?y &amp;lt;");
    expect(out).toBe("a] (b c] :d < e < /h < !-- x < ?y & lt;");
    expect(md.render(out)).not.toMatch(/<a |<e|<\/h|<!--|<\?y/);
  });

  it("random mixes of link syntax and addresses (2,000 cases)", () => {
    const pieces = [
      "[", "]", "(", ")", "<", ">", "!", "`", "\\", ":", "\n", "\n\n", " ", " ", "  ", '"', "'", "*", "_", "#", "&", ";", "@", ".", "/", "-", "> ", "- ", "1. ",
      "https://evil.example/x", "http://evil.com", "//evil.com/a", "www.evil.com", "evil.com", "evil.com/a(b)", "x@evil.com", "mailto:x@evil.com", "javascript:alert(1)",
      "&#46;", "&period;", "&lt;", "&gt;", "&amp;", "&#x3a;", "&colon;", "&sol;", ZWSP, RLO,
      "evil.xn--p1ai", "evil.\u0440\u0444", "\u00e9vil.com", "\uff45\uff56\uff49\uff4c.com", "evil.com-", "evi\u00adl.com", "Ftp://evil.com", "xmpp:a@evil.com", "mailto:a@b",
      "pixagram.com", "https://pixagram.com/wallet/settings", "support@pixagram.com", "https://en.wikipedia.org/wiki/Pixel_art_(disambiguation)",
      "[1]", "[2-3]", "[2021]", "text", "evil", "com", "www", "https", "a href=", "img src=x", "](", "]:", "![x](", "[x]: ", "<a href=\"", "</a>",
    ];
    let seed = 0x9e3779b9;
    const rand = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    for (let k = 0; k < 2000; k++) {
      let input = "";
      const len = 3 + Math.floor(rand() * 25);
      for (let i = 0; i < len; i++) input += pieces[Math.floor(rand() * pieces.length)];
      expectSafe(input);
    }
  });

  it("long answers are cut between words, never inside an address", () => {
    const answer = `${"Open the wallet settings at https://pixagram.com/wallet/settings and save. ".repeat(40)}`;
    const out = clean(answer);
    expect(out.length).toBeLessThanOrEqual(MAX_ANSWER_CHARS + 8);
    expect(out.endsWith("…")).toBe(true);
    for (const m of out.matchAll(/https:\/\/\S+/g)) expect(m[0]).toBe("https://pixagram.com/wallet/settings");
  });
});

describe("7. the documentation's own links and addresses stay, written out", () => {
  it("autolinks, links with parentheses, e-mail autolinks, mailto: links, bracketed years", () => {
    const cases: Array<[string, string]> = [
      ["See <https://pixagram.com/wallet/settings> [1].", "See https://pixagram.com/wallet/settings [1]."],
      ["[Wikipedia](https://en.wikipedia.org/wiki/Pixel_art_(disambiguation))", "Wikipedia (https://en.wikipedia.org/wiki/Pixel_art_(disambiguation))"],
      ["Mail <support@pixagram.com>", "Mail support@pixagram.com"],
      ["[Mail us](mailto:support@pixagram.com)", "Mail us (support@pixagram.com)"],
      ["It launched in [2021] [1].", "It launched in [2021] [1]."],
      ["Click [here](https://pixagram.com/wallet/settings#security) [1].", "Click here (https://pixagram.com/wallet/settings#security) [1]."],
      ["Use pixagram.com or https://pixagram.com/ [1].", "Use pixagram.com or https://pixagram.com/ [1]."],
      ["Source: https://github.com/pixa/info/blob/main/wallet.md#settings.", "Source: https://github.com/pixa/info/blob/main/wallet.md#settings."],
      ["[Fees][1] and [1][2], [3]", "Fees [1] and [1][2],"],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
    expect(targets(clean("[Wikipedia](https://en.wikipedia.org/wiki/Pixel_art_(disambiguation))"))).toEqual(["https://en.wikipedia.org/wiki/Pixel_art_(disambiguation)"]);
  });

  it("documentation links reach the model: absolute links keep their address in the indexed text", () => {
    const refs = new Map([["guide", "https://pixagram.com/guide"]]);
    expect(inlineText("Open [your wallet settings](https://pixagram.com/wallet/settings) or [the FAQ](faq.md#top).", { links: true })).toBe(
      "Open your wallet settings (https://pixagram.com/wallet/settings) or the FAQ.",
    );
    expect(inlineText("Write [to us](mailto:support@pixagram.com) or <help@pixagram.com>; see [the guide][Guide] and [guide].", { links: true, refs })).toBe(
      "Write to us (support@pixagram.com) or help@pixagram.com; see the guide (https://pixagram.com/guide) and guide (https://pixagram.com/guide).",
    );
    // headings keep their text: GitHub makes their anchors from it
    expect(inlineText("Settings [page](https://pixagram.com/settings)")).toBe("Settings page");
    const doc = parseDoc("# Wallet\n\n## Settings [page](https://x.y)\n\nOpen [the settings][s].\n\n> [s]: https://pixagram.com/wallet/settings\n", "wallet.md");
    expect(doc.sections.map((s) => [s.anchor, s.text])).toEqual([
      ["wallet", ""],
      ["settings-page", "Open the settings (https://pixagram.com/wallet/settings)."],
    ]);
  });

  it("end to end: an answer may give the address a section links to", async () => {
    const env = makeEnv({ DOCS_REPO: "pixa/info" });
    (env as any).VEC_DOCS = new FakeVectorize([]);
    installGitHub({ owner: "pixa", repo: "info", head: "7".repeat(40), files: { "wallet.md": "# Wallet\n\nChange your password in [the wallet settings](https://pixagram.com/wallet/settings)." }, calls: [] });
    env._ai.handler = (model, input) => {
      if (model === "@cf/baai/bge-m3") return { data: input.text.map(() => [1, 0, 0]) };
      return { response: { answerable: true, answer: "Open [the wallet settings](https://pixagram.com/wallet/settings) [1], not [this](https://evil.example).", sources: [1] } };
    };
    await syncDocs(env);
    const r = await answerHelp(env, "Where do I change my wallet password?");
    expect(r.status).toBe("answered");
    expect(r.answer_text).toBe("Open the wallet settings (https://pixagram.com/wallet/settings) [1], not this.");
  });
});

// ---- 2 to 6: the search box ---------------------------------------------------------------------

describe("2. platform words that are also artwork subjects: help, unless the question is plainly about posts", () => {
  const cases: Array<[string, Route]> = [
    ["which wallet is the best?", "help"],
    ["what is the best wallet?", "help"],
    ["best wallet for pixagram?", "help"],
    ["welche Wallet ist die beste?", "help"],
    ["how many fees are there?", "help"],
    ["what is the latest fee?", "help"],
    ["who is the top witness?", "help"],
    ["how many witnesses are there?", "help"],
    ["how many tokens do I get?", "help"],
    ["how many tokens do I get to post?", "help"],
    ["how many tokens does a post earn?", "help"],
    ["how much can I earn?", "help"],
    ["how many tokens can I earn?", "help"],
    ["wie viele Token bekomme ich?", "help"],
    ["combien de jetons ai-je ?", "help"],
    ["what is the latest PXS price?", "help"],
    ["how many PXS do I need?", "help"],
    ["who has the most PXS?", "help"],
    // people counted, in a permission question
    ["how many accounts can I create?", "help"],
    ["how many accounts can I have?", "help"],
    ["combien de comptes puis-je créer ?", "help"],
    ["wie viele Konten darf ich haben?", "help"],
    ["how many users can share a wallet?", "help"],
    // plainly about posts: the word is the subject
    ["who posted the first wallet?", "ask"],
    ["how many wallet artworks?", "ask"],
    ["what is the latest wallet post?", "ask"],
    ["who posted the first PXS artwork?", "ask"],
    ["how many accounts are there?", "ask"],
    // what the question ranks by is not its topic
    ["what is the best earning artwork?", "ask"],
    ["most earning cat?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));
});

describe("3. questions about posts are answered; advice is not a question about posts", () => {
  const cases: Array<[string, Route, number?]> = [
    ["what is the latest post?", "ask"],
    ["what are the latest posts?", "ask"],
    ["what is the most liked post?", "ask"],
    ["what is the most popular post?", "ask"],
    ["what is the first post?", "ask"],
    ["what is the most liked?", "ask"],
    ["what's the newest post?", "ask"],
    ["what's the latest?", "ask"],
    ["what is the best time to post?", "search"],
    ["quel est le meilleur moment pour poster ?", "search"],
    ["wann ist die beste Zeit zum Posten?", "search"],
    ["what is the best time to post?", "help", 0.8],
  ];
  for (const [q, want, docs] of cases) it(`${JSON.stringify(q)}${docs ? " (documented)" : ""} → ${want}`, async () => expect(await route(q, docs ?? 0)).toBe(want));
});

describe("4. phones and the team are platform words again", () => {
  const cases: Array<[string, Route]> = [
    ["does pixagram work on android?", "help"],
    ["does it work on iphone?", "help"],
    ["is pixagram available on android?", "help"],
    ["is there a mobile version?", "help"],
    ["est-ce que pixagram marche sur android ?", "help"],
    ["Funktioniert Pixagram auf dem iPhone?", "help"],
    ["ios version?", "help"],
    ["who is on the team?", "help"],
    ["who posted the first android?", "ask"],
    ["how many android artworks?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));
});

describe("5. 'on Pixagram' still names the platform", () => {
  const cases: Array<[string, Route]> = [
    ["is it free to post on pixagram?", "help"],
    ["is posting free on pixagram?", "help"],
    ["what is the minimum price on pixagram?", "help"],
    ["est-ce gratuit de poster sur pixagram ?", "help"],
    ["cats on pixagram?", "ask"],
    ["how many artists are on pixagram?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));
});

describe("6. 'forbidden' is a subject; a who/when opener alone is not about posts", () => {
  const cases: Array<[string, Route]> = [
    ["who posted the first forbidden fruit?", "ask"],
    ["how many forbidden forest artworks?", "ask"],
    ["qui a posté le premier fruit interdit ?", "ask"],
    ["is AI art allowed?", "help"],
    ["when was the last pixagram update?", "help"],
    ["when was the last block?", "search"],
    ["who is the newest member?", "search"],
    ["who posted the first avatar?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));
});

// ---- 8: the help cache --------------------------------------------------------------------------

/** Vectors that are written but not searchable yet, as just after a Vectorize upsert. */
class LateVectorize extends FakeVectorize {
  visible = false;
  override async query(vector: number[], opts: { topK?: number; filter?: Record<string, any> }) {
    return this.visible ? super.query(vector, opts) : { matches: [], count: 0 };
  }
}

describe("8. 'not found' is cached for minutes, answers for a day", () => {
  it("vectors that become searchable after the first question", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-10-05T12:00:00Z"));
    const env = makeEnv({ DOCS_REPO: "pixa/info" });
    const vec = new LateVectorize([]);
    (env as any).VEC_DOCS = vec;
    installGitHub({ owner: "pixa", repo: "info", head: "8".repeat(40), files: { "wallet.md": "# Wallet\n\nYour savings stay in your wallet." }, calls: [] });
    env._ai.handler = (model, input) => {
      if (model === "@cf/baai/bge-m3") return { data: input.text.map(() => [1, 0, 0]) };
      return { response: { answerable: true, answer: "In your wallet [1].", sources: [1] } };
    };
    await syncDocs(env);
    const q = "Où sont gardées mes économies ?";
    expect((await answerHelp(env, q)).status).toBe("not_found");
    const ttl = () => [...env._kv.expires.entries()].filter(([k]) => k.startsWith("help:")).map(([, at]) => (at - Date.now()) / 1000);
    expect(ttl()).toEqual([NOT_FOUND_TTL]);
    vec.visible = true;
    expect((await answerHelp(env, q)).cached).toBe(true); // within the few minutes
    vi.setSystemTime(new Date(Date.now() + (NOT_FOUND_TTL + 1) * 1000));
    const r = await answerHelp(env, q);
    expect(r.cached).toBeUndefined();
    expect(r.status).toBe("answered");
    expect(ttl()).toEqual([ANSWER_TTL]);
  });
});

// ---- the second verification ----------------------------------------------------------------------

describe("second verification: platform words beside 'post', compounds, German how-to, full stops", () => {
  const cases: Array<[string, Route]> = [
    // a platform word with "post" is still about the platform…
    ["how many tokens per post?", "help"],
    ["what is the fee per post?", "help"],
    ["what's the fee for a post?", "help"],
    ["PXS per post?", "help"],
    ["wallet for my posts?", "help"],
    ["what wallet should I use for my posts?", "help"],
    ["fees for posts?", "help"],
    ["is there a fee for each post?", "help"],
    ["how much is the fee for a post?", "help"],
    ["how many tokens do I get for a post?", "help"],
    ["how many tokens is a post worth?", "help"],
    // …unless the posts show it
    ["how many artworks of wallets?", "ask"],
    ["artworks with wallets?", "ask"],
    ["how many posts about tokens?", "ask"],
    ["combien d'œuvres de jetons ?", "ask"],
    ["who posted the first wallets?", "ask"],
    // hyphenated compounds
    ["Gibt es eine Android-App?", "help"],
    ["Gibt es eine iPhone-App?", "help"],
    ["Android-Version?", "help"],
    ["Hat Pixagram eine Mobile-App?", "help"],
    ["Wie ist die iOS-App?", "help"],
    // German how-to; a count is not one
    ["Wie lösche ich einen Beitrag?", "help"],
    ["wie viele Bilder habe ich gepostet?", "ask"],
    // "the most likes" ranks by votes
    ["which post has the most likes?", "ask"],
    ["welcher Beitrag hat die meisten Likes?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));

  it("'the most likes' is a most-liked question", () => {
    expect(planQuery("which post has the most likes?", { mode: "ask", authors: AUTHORS, now: NOW })).toMatchObject({ intent: "top", sort: "votes", residual: "" });
  });

  it("end to end: 'fees for posts?' is a help question, not the fairy artwork", async () => {
    const env = makeEnv({ PLANNER_BACKEND: "rules" });
    const exec = new FakeExec();
    const res = await app.fetch(new Request(`https://search.test/query?q=${encodeURIComponent("fees for posts?")}`), env, exec as unknown as ExecutionContext);
    await exec.settle();
    const body = await res.json<any>();
    expect(body.route).toBe("help");
    expect(body.answer.status).toBe("no_docs");
    expect(Array.isArray(body.results.items)).toBe(true);
  });

  it("domains written with ideographic or full-width full stops are addresses too", () => {
    const dots = [0x3002, 0xff0e, 0xff61].map((c) => String.fromCharCode(c));
    const wide = (t: string) => [...t].map((ch) => String.fromCharCode(ch.charCodeAt(0) + 0xfee0)).join("");
    for (const d of dots) {
      expect(clean(`Go to evil${d}com [1].`)).toBe("Go to [1].");
      expect(clean(`Go to www${d}evil${d}co${d}uk/login now`)).toBe("Go to now");
      expect(clean(`Go to ${wide("evil")}${d}${wide("com")} now`)).toBe("Go to now");
      expect(clean(`Use pixagram${d}com [1].`)).toBe("Use [1].");
    }
    // CJK sentences keep their full stops, and Latin words that are not addresses their particles
    const wallet = String.fromCharCode(0x30a6, 0x30a9, 0x30ec, 0x30c3, 0x30c8);
    const wa = String.fromCharCode(0x306f);
    expect(clean(`${wallet}${dots[0]}PXS${wa}${dots[0]} [1]`)).toBe(`${wallet}${dots[0]}PXS${wa}${dots[0]} [1]`);
  });
});

// ---- the third verification ---------------------------------------------------------------------

describe("third verification: what posts show, and addresses in Chinese and Japanese answers", () => {
  const cases: Array<[string, Route]> = [
    ["how many artworks show a wallet?", "ask"],
    ["latest artwork depicting a token?", "ask"],
    ["first drawing of an old iphone?", "ask"],
    ["how many artworks that show an android?", "ask"],
    ["first pixel art android?", "ask"],
    ["how many android robot artworks?", "ask"],
    ["quelle est la dernière œuvre android ?", "ask"],
    ["how many artworks of red wallets?", "ask"],
    ["combien d'œuvres avec un téléphone mobile ?", "ask"],
    ["pixel art of a mobile phone?", "ask"],
    // money after an artwork word, devices for drawing: the platform
    ["what are the artwork fees?", "help"],
    ["how many artwork tokens do I get?", "help"],
    ["is there an android app for posting art?", "help"],
    ["which iphone is best for pixel art?", "help"],
    ["how many posts mention fees?", "help"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));

  const h = (...c: number[]) => String.fromCharCode(...c);
  const DOT = h(0x3002);
  const [visit, then, supports, asOf, open, login, contact, example] = [
    h(0x8bf7, 0x8bbf, 0x95ee), h(0x7136, 0x540e), h(0x652f, 0x6301), h(0x622a, 0x81f3), h(0x8bf7, 0x6253, 0x5f00), h(0x91cd, 0x65b0, 0x767b, 0x5f55), h(0x8bf7, 0x8054, 0x7cfb), h(0x4f8b, 0x3048),
  ];
  it("the documentation's addresses survive Chinese and Japanese sentences", () => {
    const cases: Array<[string, string]> = [
      [`${visit} pixagram.com${DOT}Pixagram ${supports} PXS${DOT}[1]`, `${visit} pixagram.com${DOT}Pixagram ${supports} PXS${DOT}[1]`],
      [`${asOf}2026${DOT}Pixagram`, `${asOf}2026${DOT}Pixagram`],
      [`${open} https://pixagram.com/wallet/settings${DOT}${then}${login}${DOT} [1]`, `${open} https://pixagram.com/wallet/settings ${DOT}${then}${login}${DOT} [1]`],
      [`${visit}pixagram.com${DOT}${then}`, `${visit} pixagram.com${DOT}${then}`],
      [`${contact}support@pixagram.com${DOT}`, `${contact} support@pixagram.com${DOT}`],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
  });

  it("and other addresses do not, nor hosts that a full stop extends", () => {
    const cases: Array<[string, string]> = [
      [`${visit}evil.com${DOT}`, `${visit} ${DOT}`],
      [`https://pixagram.com${DOT}xyz [1]`, "[1]"],
      [`www.pixagram.com${DOT}xyz`, ""],
      [`https://pixagram.com/wallet/settings${DOT}evil.com`, `https://pixagram.com/wallet/settings ${DOT}`],
      [`${example}.com and pixagram${example}.com`, "and pixagram"],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
  });
});

// ---- the fourth verification --------------------------------------------------------------------

describe("fourth verification: the user's own things, the platform's products, fairies, CJK and Korean", () => {
  const cases: Array<[string, Route]> = [
    ["artworks with my wallet?", "help"],
    ["Bilder von meinem iPhone werden nicht angezeigt?", "help"],
    ["Bilder von meinem Android-Handy?", "help"],
    ["œuvres de mon téléphone mobile ?", "help"],
    ["is there a pixel art mobile version?", "help"],
    ["art mobile version?", "help"],
    ["how many fee artworks?", "help"],
    // the fairy, with its French accent or as a German noun; a picture of a phone
    ["qui a posté la première fée ?", "ask"],
    ["combien d'œuvres de fées ?", "ask"],
    ["wie viele Bilder mit einer Fee?", "ask"],
    ["What is the Fee per post?", "help"],
    ["Wie viele Bilder von einem Android-Handy?", "ask"],
  ];
  for (const [q, want] of cases) it(`${JSON.stringify(q)} → ${want}`, async () => expect(await route(q)).toBe(want));

  it("search and indexing keep the French and German fairy", async () => {
    const { matchConcepts } = await import("../src/concepts");
    for (const w of ["fée", "fées", "une fée", "die Fee", "Feen", "conte de fées"]) expect(matchConcepts(w).map((m) => m.concept), w).toContain("fantasy");
  });

  const h = (...c: number[]) => String.fromCharCode(...c);
  it("a URL without a path keeps its sentence in Chinese; addresses keep their Korean particles' sentence", () => {
    const DOT = h(0x3002);
    const COMMA = h(0xff0c);
    const COLON = h(0xff1a);
    const [visit, then, login, site] = [h(0x8bf7, 0x8bbf, 0x95ee), h(0x7136, 0x540e), h(0x91cd, 0x65b0, 0x767b, 0x5f55), h(0x5b98, 0x7f51)];
    const [settingsTopic, at, change, details, check, ask, toward, contact] = [
      h(0xc124, 0xc815, 0xc740), h(0xc5d0, 0xc11c), h(0xbcc0, 0xacbd, 0xd558, 0xc138, 0xc694), h(0xc790, 0xc138, 0xd55c, 0x20, 0xb0b4, 0xc6a9, 0xc740), h(0xd655, 0xc778, 0xd558, 0xc138, 0xc694), h(0xbb38, 0xc758), h(0xc73c, 0xb85c), h(0xc5f0, 0xb77d, 0xd558, 0xc138, 0xc694),
    ];
    const cases: Array<[string, string]> = [
      [`${visit} https://pixagram.com${DOT}${then}${login}${DOT} [1]`, `${visit} https://pixagram.com ${DOT}${then}${login}${DOT} [1]`],
      [`${visit} https://pixagram.com${DOT} [1]`, `${visit} https://pixagram.com ${DOT} [1]`],
      [`${site}${COLON}https://pixagram.com${COMMA}${then}${login}${DOT}[1]`, `${site}${COLON}https://pixagram.com ${COMMA}${then}${login}${DOT}[1]`],
      [`${settingsTopic} https://pixagram.com/wallet/settings${at} ${change}. [1]`, `${settingsTopic} https://pixagram.com/wallet/settings ${at} ${change}. [1]`],
      [`${details} pixagram.com${at} ${check}.`, `${details} pixagram.com ${at} ${check}.`],
      [`${ask}: support@pixagram.com${toward} ${contact}.`, `${ask}: support@pixagram.com ${toward} ${contact}.`],
      // a Korean particle after an address the documentation does not have
      [`evil.com${at} ${check}.`, `${at} ${check}.`],
    ];
    for (const [input, want] of cases) {
      expect(clean(input), JSON.stringify(input)).toBe(want);
      expectSafe(input);
    }
    expect(validateHelpReply({ answerable: true, answer: `${site}${COLON}https://pixagram.com${COMMA}${then}${login}${DOT}[1]`, sources: [1] }, 1, SECTIONS)?.sources).toEqual([1]);
  });
});
