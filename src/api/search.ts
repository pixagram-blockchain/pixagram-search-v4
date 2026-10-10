// Search routes: the index itself (/search, GET and POST), the single search box (/query),
// suggestions, similar artworks and duplicates, search by image, posts, history, concepts, images.

import type { Hono } from "hono";
import type { Env } from "../env";
import { int } from "../env";
import { cleanText, parseSearchRequest, searchParamsFromBody, type SearchRequest } from "../search/params";
import { getItem, hydrateOrdered, search } from "../search/service";
import { duplicates, duplicatesOfHash, similar } from "../search/visual";
import { ask, answerStyle } from "../search/ask";
import { searchOverview } from "../search/overview";
import { richOptions } from "./ask";
import { isModeRequest, MODES, type Mode } from "../search/query-router";
import { recordFeedback } from "../search/feedback";
import { knn } from "../search/vectors";
import { planQuery } from "../search/planner";
import { matchConcepts, ancestors, aliasesOf, conceptDef, vocabLang } from "../concepts";
import { getPostById, getArtwork } from "../db/posts";
import { decodeImage, sniff } from "../enrich/decode";
import { dhash, phash } from "../enrich/phash";
import { embeddingEnabled } from "../enrich/embed";
import { embedImageViews } from "../enrich/consumer";
import { COLOR_NAMES, NAMED_COLORS } from "../enrich/color";
import { SIZE_CLASSES } from "../enrich/stats";
import { knownAuthors } from "../search/context";
import { routeQuery, routeText, DOCS_KNOWS, type RouteDecision } from "../search/router";
import { answerHelp } from "../help/answer";
import { popularEnabled, recordSearcher, suggestAfter, suggestExamples, suggestFor, suggestText } from "../search/suggest";
import { lexicalDocs } from "../help/retrieve";
import { bodyObject, edgeCached, execOf, heavyAllowed, isAdmin, MAX_IMAGE_UPLOAD, MAX_QUERY_IMAGE_PIXELS, readBody, readJson, type Bindings } from "./common";
import { readUploadedImage } from "./image";

export const ENDPOINTS = [
  "/query", "/suggest", "/search", "/ask", "/ask/feedback", "/help", "/similar/:id", "/duplicates/:id", "/search-by-image", "/posts/:id", "/posts/:author/:permlink", "/history/:id",
  "/concepts", "/feedback", "/img/orig/:hash.:ext", "/img/up/:hash.png", "/vocab", "/healthz",
  "/copies/:id", "/copies-by-image", "/copies/:a/report/:b", "/paph/:id",
];

/** Whether the words are (part of) the title of a live post: "Come and fly with me" stays a search. */
async function isPostTitle(env: Env, q: string): Promise<boolean> {
  const phrase = routeText(q).replace(/"/g, " ").trim();
  if (phrase.split(" ").length < 2) return false;
  const r = await env.DB.prepare("SELECT 1 AS ok FROM posts_fts JOIN posts p ON p.id = posts_fts.rowid WHERE posts_fts MATCH ? AND p.deleted = 0 LIMIT 1")
    .bind(`title : "${phrase}"`)
    .first()
    .catch(() => null);
  return !!r;
}

/** Documentation pages that match a search's words closely (shown next to the results). */
async function helpLinks(env: Env, q: string): Promise<Array<{ title: string; heading: string; url: string }>> {
  const { hits } = await lexicalDocs(env, q, 10);
  const seen = new Set<string>();
  const out: Array<{ title: string; heading: string; url: string }> = [];
  for (const h of hits) {
    if (h.score < DOCS_KNOWS || seen.has(h.path)) continue;
    seen.add(h.path);
    out.push({ title: h.title, heading: h.heading, url: h.url });
    if (out.length === 3) break;
  }
  return out;
}

/** The deepest mode the search box picks by itself (SEARCH_QUERY_MAX_MODE; by default deep for rich answers, balanced for brief ones, as v4). */
export function queryMaxMode(env: Env, style: "rich" | "brief" = "brief"): Mode {
  const dflt: Mode = style === "rich" ? "deep" : "balanced";
  const m = String(env.SEARCH_QUERY_MAX_MODE ?? dflt);
  return (MODES as string[]).includes(m) ? (m as Mode) : dflt;
}

export function registerSearch(app: Hono<Bindings>): void {
  app.get("/", (c) => c.json({ name: "pixagram-search", version: 4, endpoints: ENDPOINTS }));

  app.get("/healthz", async (c) => {
    const row = await c.env.DB.prepare("SELECT COUNT(*) AS n FROM posts WHERE deleted = 0").first<{ n: number }>();
    return c.json({ ok: true, posts: row?.n ?? 0, semantic: embeddingEnabled(c.env), text_vectors: !!c.env.VEC_TEXT });
  });

  /** The filter vocabularies the UI needs to render controls. */
  app.get("/vocab", (c) =>
    c.json({
      colors: NAMED_COLORS.map((n) => ({ name: n.name, hex: n.hex })),
      size_classes: SIZE_CLASSES,
      orientations: ["portrait", "landscape", "square"],
      sorts: ["relevance", "newest", "oldest", "votes", "payout"],
      nsfw: ["exclude", "include", "only"],
      params: {
        q: "free text (title, tags, description, AI caption; semantic when the embedding endpoint is on). @author and #tag work inside q",
        type: "artwork | blog",
        author: "comma list or repeated",
        tag: "comma list or repeated (AND)",
        color: `primary colour, one of ${COLOR_NAMES.join("|")}`,
        has_color: "any palette bucket with weight >= min_color_weight (default 0.08)",
        size: `one of ${SIZE_CLASSES.join("|")}`,
        orientation: "portrait|landscape|square",
        monochrome: "true|false",
        background: "named colour of the backdrop, or transparent",
        concept: "canonical concept ids that must be present (see /concepts)",
        min_colors: "number", max_colors: "number",
        min_width: "px", max_width: "px", min_height: "px", max_height: "px",
        from: "date or unix seconds (inclusive)", to: "date or unix seconds (exclusive)",
        transparent: "true|false", nsfw: "exclude|include|only", listed: "true|false", ai_training: "true|false",
        sort: "relevance|newest|oldest|votes|payout", limit: "1..50", cursor: "from next_cursor", facets: "1 to include facet counts",
        semantic: "0 to disable vector search", rank: "v3 (default) | rrf (v2's fusion, for comparison)", expand: "0 to disable planner hints, spelling and concepts", explain: "1 to include features and the query plan",
        rerank: "1 to reorder the top of a relevance ranking with the cross-encoder (v4)",
      },
    }),
  );

  /**
   * /search: GET ?q=…, or POST {"query": "red pixel cat", "limit": 20, "rerank": true} with the same
   * parameters. The response carries how many candidates each retrieval family found (retrieval)
   * and whether the cross-encoder reordered it (reranked).
   */
  app.on(["GET", "POST"], "/search", async (c) => {
    const sp = c.req.method === "POST" ? searchParamsFromBody(await bodyObject(c)) : new URL(c.req.url).searchParams;
    const req = parseSearchRequest(sp);
    const exec = execOf(c);
    const notes: string[] = [];
    // the cross-encoder is a model call: it spends the answer budget
    if (req.rerank && !(await heavyAllowed(c))) {
      req.rerank = false;
      notes.push("rerank skipped: this client's answer budget is spent");
    }
    const res = await search(c.env, req, exec);
    if (notes.length) res.notes = [...(res.notes ?? []), ...notes];
    // overview=1: a text overview of the page (v4.8), in the language asked for (lang=) or the query's
    if (/^(1|true|yes|on)$/i.test(sp.get("overview") ?? "")) res.overview = searchOverview(res, sp.get("lang") ?? planQuery(req.q, { mode: "search" }).lang);
    // who ran it, for the popular searches /suggest may show (SUGGEST_POPULAR=on): a page-one text
    // search in the default safe mode that found something
    if (popularEnabled(c.env) && req.q.trim() && !req.cursor && req.nsfw === "exclude" && res.items.length > 0) {
      exec.waitUntil(recordSearcher(c.env, req.q, c.req.header("cf-connecting-ip")));
    }
    if (c.req.method === "GET") c.header("cache-control", "public, max-age=30");
    return c.json(res);
  });

  /**
   * The single search box (search/router.ts): GET /query?q=… plus any /search parameter.
   *   route "search" → results (a /search response), and help_links when the words are the platform's
   *   route "ask"    → answer (an /ask response, with the posts behind it in answer.items)
   *   route "help"   → answer (a /help response: text and GitHub sources)
   * When an answer comes back empty, results carry a search for the same text. route=search|ask|help
   * forces the destination (tabs in the UI); a cursor always means the next page of a search. v4: the
   * box answers in the mode the question needs, up to SEARCH_QUERY_MAX_MODE (mode= forces one).
   */
  app.get("/query", async (c) => {
    const t0 = Date.now();
    const sp = new URL(c.req.url).searchParams;
    const q = cleanText(sp.get("q") ?? "").slice(0, 300);
    const req: SearchRequest = parseSearchRequest(sp);
    const exec = execOf(c);
    const runSearch = () => search(c.env, req, exec);
    const forced = sp.get("route");
    // results=0: the caller runs its own search (the search box does); no search here, and no
    // search logged twice
    const withResults = sp.get("results") !== "0";
    // v4.8: rich answers (style, text, length, defer), and a text overview with every search
    const rich = richOptions((k) => sp.get(k));
    const richStyle = answerStyle(c.env, rich.style) === "rich";
    const overviewOf = (results: Awaited<ReturnType<typeof search>> | null, lang: string) => (richStyle && results ? searchOverview(results, lang) : undefined);
    const maybeSearch = async () => (withResults ? runSearch() : null);
    if (!q || sp.get("cursor") || forced === "search") {
      const reason = !q ? "nothing typed: browse" : sp.get("cursor") ? "next page" : "route=search";
      const results = await maybeSearch();
      return c.json({ q, route: "search", reason, results, overview: q && !sp.get("cursor") ? overviewOf(results, planQuery(q, { mode: "search" }).lang) : undefined, took_ms: Date.now() - t0 });
    }
    const plan = planQuery(q, { mode: "ask", authors: await knownAuthors(c.env) });
    let decision: RouteDecision;
    if (forced === "ask" || forced === "help") {
      decision = await routeQuery(q, plan, { docsScore: async () => 0, isTitle: async () => false });
      decision = { ...decision, route: forced, reason: `route=${forced}` };
    } else {
      decision = await routeQuery(q, plan, {
        docsScore: async () => (await lexicalDocs(c.env, q, 10)).hits[0]?.score ?? 0,
        isTitle: () => isPostTitle(c.env, q),
      });
    }
    const notes: string[] = [];
    let budget: "spent" | undefined;
    if (decision.route !== "search" && !(await heavyAllowed(c))) {
      notes.push(`would have answered (${decision.route}), but this client's answer budget is spent: showing results`);
      decision = { ...decision, route: "search" };
      budget = "spent"; // a client may ask again later: this is no verdict about the text
    }
    const meta = { q, route: decision.route, reason: decision.reason, signals: decision.signals, notes, answer_budget: budget };
    // An answer that fails is reported in the notes; the box then shows a search.
    const fallback = async (e: unknown) => {
      console.error("query answer failed", decision.route, e);
      notes.push(`the ${decision.route} answer failed (${e instanceof Error ? e.message : String(e)}): showing results`);
      const results = await maybeSearch();
      return c.json({ ...meta, route: "search", results, overview: overviewOf(results, plan.lang), took_ms: Date.now() - t0 });
    };
    const askedMode = sp.get("mode");
    if (decision.route === "ask") {
      try {
        const answer = await ask(
          c.env,
          {
            question: q,
            plan,
            type: req.type ?? undefined,
            nsfw: req.nsfw,
            limit: Math.min(req.limit, 24),
            mode: isModeRequest(askedMode) ? askedMode : undefined,
            ceiling: queryMaxMode(c.env, richStyle ? "rich" : "brief"),
            admin: isAdmin(c),
            ...rich,
          },
          exec,
        );
        // nothing found (null), or a zero ("0 cats", "0 similar artworks"): show what a search finds
        const results = answer.answer === null || answer.answer === 0 ? await maybeSearch() : undefined;
        return c.json({ ...meta, answer, results, overview: results ? overviewOf(results, plan.lang) : undefined, took_ms: Date.now() - t0 });
      } catch (e) {
        return fallback(e);
      }
    }
    if (decision.route === "help") {
      try {
        const answer = await answerHelp(c.env, q, { lang: plan.lang, style: rich.style, length: rich.length, ...(richStyle && isModeRequest(askedMode) && askedMode !== "auto" && askedMode !== "v3" ? { mode: askedMode } : {}) });
        const results = answer.status === "answered" || answer.status === "excerpts" ? undefined : await maybeSearch();
        return c.json({ ...meta, answer, results, overview: results ? overviewOf(results, plan.lang) : undefined, took_ms: Date.now() - t0 });
      } catch (e) {
        return fallback(e);
      }
    }
    const showLinks = decision.signals.platform.length > 0 || decision.signals.platformName;
    const [results, help_links] = await Promise.all([maybeSearch(), showLinks ? helpLinks(c.env, q).catch(() => []) : Promise.resolve(undefined)]);
    return c.json({ ...meta, results, overview: overviewOf(results, plan.lang), help_links: help_links?.length ? help_links : undefined, took_ms: Date.now() - t0 });
  });

  /**
   * Suggestions for the search box (search/suggest.ts): GET /suggest?q=…&lang=…&limit=… →
   * { completion, suggestions: [{ text, kind, route, query?, n?, source?, post? }] }. Without q:
   * { examples } for the placeholder, in the UI language. A trailing space in q means the last word
   * is finished.
   */
  app.get("/suggest", async (c) => {
    const sp = new URL(c.req.url).searchParams;
    const q = suggestText(sp.get("q"));
    const lang = vocabLang(sp.get("lang"));
    const limit = Math.max(1, Math.min(12, int(sp.get("limit") ?? undefined, 8)));
    // after=<query_id> (v4.8): the follow-up questions and searches of that answer come first, so
    // the box continues the conversation; the rest of the list is the usual one
    const after = (sp.get("after") ?? "").trim();
    if (after) {
      const key = `suggest/after/${encodeURIComponent(after)}/${lang}/${limit}/${encodeURIComponent(q)}`;
      return edgeCached(c, key, 30, async () => {
        const mine = await suggestAfter(c.env, after, q, Math.min(limit, 4));
        if (!q.trim()) {
          const ex = await suggestExamples(c.env, lang);
          return { ...ex, examples: [...mine, ...ex.examples.filter((e) => !mine.some((m) => m.text === e.text))].slice(0, Math.max(limit, ex.examples.length)) };
        }
        const rest = await suggestFor(c.env, q, { lang, limit });
        return { ...rest, suggestions: [...mine, ...rest.suggestions.filter((e) => !mine.some((m) => m.text === e.text))].slice(0, limit) };
      });
    }
    if (!q.trim()) return edgeCached(c, `suggest/examples/${lang}`, 300, () => suggestExamples(c.env, lang));
    return edgeCached(c, `suggest/q/${lang}/${limit}/${encodeURIComponent(q)}`, 60, () => suggestFor(c.env, q, { lang, limit }));
  });

  /** Click and engagement events for learning to rank: {query_id, post_id, rank?, action, dwell_ms?}. */
  app.post("/feedback", async (c) => {
    const r = await recordFeedback(c.env, await readJson(c.req.raw, 2048));
    return r.ok ? c.body(null, 204) : c.json({ error: r.error }, 400);
  });

  app.get("/similar/:id", async (c) => {
    const id = Number(c.req.param("id"));
    if (!Number.isFinite(id)) return c.json({ error: "bad id" }, 400);
    const sp = new URL(c.req.url).searchParams;
    const limit = Math.min(50, Math.max(1, int(sp.get("limit") ?? undefined, 24)));
    // Always filtered like /search: NSFW stays out unless nsfw=include|only is asked for.
    const r = await similar(c.env, id, limit, parseSearchRequest(sp));
    return c.json({ id, method: r.method, items: r.items });
  });

  app.get("/duplicates/:id", async (c) => {
    const id = Number(c.req.param("id"));
    if (!Number.isFinite(id)) return c.json({ error: "bad id" }, 400);
    const sp = new URL(c.req.url).searchParams;
    const max = Math.min(24, Math.max(0, int(sp.get("max_distance") ?? undefined, 8)));
    const r = await duplicates(c.env, id, max, Math.min(100, int(sp.get("limit") ?? undefined, 20)), parseSearchRequest(sp));
    return c.json({ id, phash: r.phash, max_distance: max, method: r.method, items: r.items });
  });

  /**
   * Search by an uploaded image: multipart field "image", or JSON {"image": "<base64 or data URI>"}.
   * Returns semantic neighbours (when the embedding endpoint is on) and pHash near-duplicates.
   */
  app.post("/search-by-image", async (c) => {
    const raw = await readBody(c.req.raw, MAX_IMAGE_UPLOAD);
    const up = await readUploadedImage(c.req.url, c.req.header("content-type") ?? "", raw);
    if ("error" in up) return c.json({ error: up.error }, up.status);
    const { bytes, info } = up;
    if (!info.width || !info.height || info.width * info.height > MAX_QUERY_IMAGE_PIXELS) {
      return c.json({ error: `image too large: ${info.width ?? "?"}x${info.height ?? "?"} (1024x1024 max)` }, 413);
    }
    const sp = new URL(c.req.url).searchParams;
    const req = parseSearchRequest(sp);
    const limit = Math.min(50, Math.max(1, int(sp.get("limit") ?? undefined, 24)));

    const img = await decodeImage(bytes, info, MAX_QUERY_IMAGE_PIXELS);
    const ph = phash(img);
    const dup = await duplicatesOfHash(c.env, ph, Math.min(24, int(sp.get("max_distance") ?? undefined, 8)), 20, undefined, dhash(img), req);

    let items: Awaited<ReturnType<typeof hydrateOrdered>> = [];
    let method: "vector" | "none" = "none";
    let note: string | undefined;
    if (embeddingEnabled(c.env)) {
      try {
        // Same views as the indexed artworks (EMBED_VIEWS; xBRZ by default).
        const emb = await embedImageViews(c.env, img);
        const hits = await knn(c.env, "image", emb.vector, req, limit);
        items = await hydrateOrdered(c.env.DB, hits.map((h) => h.id), req);
        const scores = new Map(hits.map((h) => [h.id, h.score]));
        for (const it of items) it.score = { fused: scores.get(it.id) ?? 0, ranks: { vec: hits.findIndex((h) => h.id === it.id) + 1 } };
        method = "vector";
      } catch (e) {
        note = `semantic search unavailable: ${e instanceof Error ? e.message : String(e)}`;
      }
    } else note = "semantic search disabled (HF_EMBED_URL not set)";
    return c.json({ method, phash: ph, width: img.width, height: img.height, items, duplicates: dup.items, note });
  });

  app.get("/posts/:id", async (c) => {
    const id = Number(c.req.param("id"));
    if (!Number.isFinite(id)) return c.json({ error: "bad id" }, 400);
    const it = await getItem(c.env, { id });
    return it ? c.json(it) : c.json({ error: "not found" }, 404);
  });

  app.get("/posts/:author/:permlink", async (c) => {
    const it = await getItem(c.env, { author: c.req.param("author").replace(/^@/, ""), permlink: c.req.param("permlink") });
    return it ? c.json(it) : c.json({ error: "not found" }, 404);
  });

  /** Every recorded version of a post (create, edits, delete-by-edit) and where its image first appeared. */
  app.get("/history/:id", async (c) => {
    const id = Number(c.req.param("id"));
    if (!Number.isFinite(id)) return c.json({ error: "bad id" }, 400);
    const p = await getPostById(c.env.DB, id);
    if (!p) return c.json({ error: "not found" }, 404);
    const versions = await c.env.DB
      .prepare("SELECT block_num, trx_id, op_in_trx, at, kind, body_kind, content_hash, title, source FROM post_versions WHERE author = ? AND permlink = ? ORDER BY at, block_num, op_in_trx")
      .bind(p.author, p.permlink)
      .all();
    const art = await getArtwork(c.env.DB, id);
    const sameImage = art?.content_hash
      ? await c.env.DB.prepare("SELECT author, permlink, MIN(at) AS first_at, COUNT(*) AS versions FROM post_versions WHERE content_hash = ? AND source != 'snapshot' GROUP BY author, permlink ORDER BY first_at").bind(art.content_hash).all()
      : null;
    return c.json({
      id,
      author: p.author,
      permlink: p.permlink,
      created: p.created,
      updated: p.updated,
      deleted: p.deleted === 1,
      image_since: art?.image_since ?? null,
      first_seen: art?.first_seen ?? null,
      versions: versions.results,
      same_image_in: sameImage?.results ?? [],
    });
  });

  /** How a word maps onto the concept vocabulary (debugging, UI chips). */
  app.get("/concepts", (c) => {
    const q = new URL(c.req.url).searchParams.get("q") ?? "";
    const matches = matchConcepts(q).map((m) => {
      const d = conceptDef(m.concept);
      return { concept: m.concept, matched: m.alias, langs: m.langs, parents: ancestors(m.concept), aliases: d ? Object.fromEntries(Object.entries(d.aliases).filter(([, v]) => v.length)) : {} };
    });
    return c.json({ q, matches, alias_count: q ? undefined : aliasesOf("cat").length });
  });

  /** Serve originals / upscaled previews from R2 (immutable, content-addressed). */
  app.get("/img/:kind{orig|up}/:file", async (c) => {
    const key = `${c.req.param("kind")}/${c.req.param("file")}`;
    if (!/^(orig|up)\/[0-9a-f]{64}\.(webp|png)$/.test(key)) return c.json({ error: "bad key" }, 400);
    // The bucket is shared and content-addressed: serve an image only while a live artwork of this
    // index shows it, so a deleted artwork's image stops being served (caches expire within a day).
    const hash = key.slice(key.indexOf("/") + 1, key.indexOf("/") + 65);
    const live = await c.env.DB.prepare("SELECT 1 AS ok FROM artworks a JOIN posts p ON p.id = a.post_id WHERE a.content_hash = ? AND p.deleted = 0 LIMIT 1").bind(hash).first();
    if (!live) return c.json({ error: "not found" }, 404);
    const obj = await c.env.ART.get(key);
    if (!obj) return c.json({ error: "not found" }, 404);
    const h = new Headers();
    obj.writeHttpMetadata(h);
    h.set("etag", obj.httpEtag);
    h.set("cache-control", "public, max-age=86400");
    return new Response(obj.body, { headers: h });
  });
}
