// Copy detection routes (src/paph):
//
//   GET  /copies/:id?min=copy|suspected|identical&limit=…   the stored verdicts of an artwork, both
//        directions (relation: the listed work is earlier or later). D1 only. live=1 (admin token)
//        checks again now, in every shard.
//   POST /copies-by-image?min=…&limit=…&semantic=1           an upload (multipart "image", or JSON
//        {"image": base64}): sub-second, answered from a day-long cache for the same pixels.
//   GET  /copies/:a/report/:b?wires=1                        the PAPH-X report of a pair, with
//        comparator 42's beside it, and optionally both wires to re-run it anywhere.
//   GET  /paph/:id                                           where an artwork stands in the index.
//
// Listings filter like /search (nsfw stays out unless nsfw=include|only); `hidden` counts verdicts
// not listed for that reason (or because the post was deleted or its image changed since).

import type { Hono } from "hono";
import { int } from "../env";
import { decodeImage } from "../enrich/decode";
import { embeddingEnabled } from "../enrich/embed";
import { embedImageViews } from "../enrich/consumer";
import { parseSearchRequest } from "../search/params";
import { copiesOf, copiesOfImage, liveCopiesOf, pairReport, paphInfo } from "../paph/copies";
import { paphEnabled } from "../paph/shards";
import { heavyAllowed, isAdmin, MAX_IMAGE_UPLOAD, MAX_QUERY_IMAGE_PIXELS, readBody, type Bindings } from "./common";
import { readUploadedImage } from "./image";

const OFF = { error: "copy detection is not configured (no PAPH binding, or PAPH_ENABLED=false)" };

export function registerCopies(app: Hono<Bindings>): void {
  app.get("/copies/:id", async (c) => {
    if (!paphEnabled(c.env)) return c.json(OFF, 503);
    const id = Number(c.req.param("id"));
    if (!Number.isSafeInteger(id) || id <= 0) return c.json({ error: "bad id" }, 400);
    const sp = new URL(c.req.url).searchParams;
    const o = { min: sp.get("min") ?? undefined, limit: int(sp.get("limit") ?? undefined, 24), req: parseSearchRequest(sp) };
    if (sp.get("live") === "1") {
      if (!isAdmin(c)) return c.json({ error: "live=1 needs the admin token" }, 403);
      return c.json(await liveCopiesOf(c.env, id, o));
    }
    c.header("cache-control", "public, max-age=60");
    return c.json(await copiesOf(c.env, id, o));
  });

  app.post("/copies-by-image", async (c) => {
    const startedAt = Date.now();
    if (!paphEnabled(c.env)) return c.json(OFF, 503);
    const raw = await readBody(c.req.raw, MAX_IMAGE_UPLOAD);
    const up = await readUploadedImage(c.req.url, c.req.header("content-type") ?? "", raw);
    if ("error" in up) return c.json({ error: up.error }, up.status);
    const { bytes, info, fields } = up;
    if (!info.width || !info.height || info.width * info.height > MAX_QUERY_IMAGE_PIXELS) {
      return c.json({ error: `image too large: ${info.width ?? "?"}x${info.height ?? "?"} (1024x1024 max)` }, 413);
    }
    const sp = new URL(c.req.url).searchParams;
    const param = (k: string) => sp.get(k) ?? fields[k] ?? undefined;
    const img = await decodeImage(bytes, info, MAX_QUERY_IMAGE_PIXELS);
    // the embedding's neighbours: opt-in, and computed only when the answer is not cached already
    // (the embedding call alone can take longer than the check)
    const wantSemantic = param("semantic") === "1";
    const semantic = wantSemantic && embeddingEnabled(c.env);
    const res = await copiesOfImage(c.env, img, {
      min: param("min"),
      limit: int(param("limit"), 24),
      semantic,
      embed: semantic ? async () => (await embedImageViews(c.env, img)).vector : undefined,
      req: parseSearchRequest(new URLSearchParams({ nsfw: param("nsfw") ?? "exclude" })),
      startedAt,
    });
    return c.json(wantSemantic && !semantic ? { ...res, notes: [...(res.notes ?? []), "semantic candidates disabled (HF_EMBED_URL not set)"] } : res);
  });

  app.get("/copies/:a/report/:b", async (c) => {
    if (!paphEnabled(c.env)) return c.json(OFF, 503);
    const a = Number(c.req.param("a")), b = Number(c.req.param("b"));
    if (!Number.isSafeInteger(a) || !Number.isSafeInteger(b) || a <= 0 || b <= 0 || a === b) return c.json({ error: "bad ids" }, 400);
    // a comparison on demand: the per-client budget of the expensive routes
    if (!(await heavyAllowed(c))) return c.json({ error: "too many requests, slow down" }, 429, { "retry-after": "60" });
    const rep = await pairReport(c.env, a, b, new URL(c.req.url).searchParams.get("wires") === "1");
    return rep ? c.json(rep) : c.json({ error: "one of the two works is not in the copy index, or was deleted or edited since" }, 404);
  });

  app.get("/paph/:id", async (c) => {
    if (!paphEnabled(c.env)) return c.json(OFF, 503);
    const id = Number(c.req.param("id"));
    if (!Number.isSafeInteger(id) || id <= 0) return c.json({ error: "bad id" }, 400);
    const info = await paphInfo(c.env, id);
    return info ? c.json(info) : c.json({ error: "no such artwork" }, 404);
  });
}
