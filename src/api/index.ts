// HTTP API (Hono). Public routes in three modules, admin routes behind the token:
//
//   search.ts  /, /healthz, /vocab, /search (GET, POST), /query, /suggest, /feedback, /similar/:id,
//              /duplicates/:id, /search-by-image, /posts/…, /history/:id, /concepts, /img/…
//   ask.ts     /ask (GET, POST, with an image), /ask/feedback
//   help.ts    /help, /webhooks/github
//   copies.ts  /copies/:id, /copies-by-image, /copies/:a/report/:b, /paph/:id (copy detection)
//   admin.ts   /admin/…
//
// Every public route has a per-client budget (Workers Rate Limiting, optional): search-like routes
// share RL_PUBLIC, the expensive ones (answers, image uploads) RL_HEAVY, suggestions RL_SUGGEST.

import { Hono } from "hono";
import { cors } from "hono/cors";
import { BodyTooLarge, rateLimited, type Bindings } from "./common";
import { registerSearch } from "./search";
import { registerAsk } from "./ask";
import { registerHelp } from "./help";
import { registerCopies } from "./copies";
import { admin } from "./admin";

export const app = new Hono<Bindings>();

app.use("*", cors({ origin: "*", allowMethods: ["GET", "POST", "OPTIONS"], maxAge: 86400 }));

app.onError((err, c) => {
  if (err instanceof BodyTooLarge) return c.json({ error: "request body too large" }, 413);
  console.error("api error", err);
  return c.json({ error: err.message ?? "internal error" }, 500);
});

// the elaboration of a deferred answer is on the public budget: its model call was paid for by the /ask that deferred it
for (const path of ["/search", "/query", "/feedback", "/ask/feedback", "/ask/elaboration/*", "/similar/*", "/duplicates/*", "/posts/*", "/history/*", "/concepts", "/copies/*", "/paph/*"]) app.use(path, rateLimited("public"));
// suggestions come with every pause in typing: a budget of their own, so typing never starves searching
app.use("/suggest", rateLimited("suggest"));
for (const path of ["/ask", "/help", "/search-by-image", "/copies-by-image"]) app.use(path, rateLimited("heavy"));

registerSearch(app);
registerAsk(app);
registerHelp(app);
registerCopies(app);
app.route("/admin", admin);

export type { Bindings };
