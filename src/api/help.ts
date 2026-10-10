// /help: questions about Pixagram itself, answered from the documentation repository
// (help/answer.ts), and the GitHub webhook that keeps that index current.

import type { Hono } from "hono";
import { cleanText } from "../search/params";
import { answerHelp, isHelpMode } from "../help/answer";
import { isReasoningLevel } from "../llm/provider";
import { syncDocs } from "../docs/sync";
import { repoName, repoRef, verifyWebhook } from "../docs/github";
import { bodyObject, isAdmin, readBody, type Bindings } from "./common";
import { richOptions } from "./ask";

export function registerHelp(app: Hono<Bindings>): void {
  /**
   * GET ?q= or POST {"question"}. v4: mode=fast|balanced|deep|expert (default HELP_MODE: by the
   * question's length) and reasoning=none|low|medium (high needs the admin token). v4.8:
   * style=rich|brief and length=short|medium|long (README "Rich answers").
   */
  app.on(["GET", "POST"], "/help", async (c) => {
    const body = await bodyObject(c);
    const sp = new URL(c.req.url).searchParams;
    const question = cleanText(String(body.question ?? sp.get("q") ?? sp.get("question") ?? "")).slice(0, 300);
    if (!question) return c.json({ error: "give a question: POST {\"question\": \"...\"} or GET /help?q=..." }, 400);
    const mode = body.mode ?? sp.get("mode");
    const reasoning = body.reasoning ?? sp.get("reasoning");
    const r = isReasoningLevel(reasoning) && (reasoning !== "high" || isAdmin(c)) ? reasoning : undefined;
    // expert is the admin's, as for /ask
    const m = isHelpMode(mode) && (mode !== "expert" || isAdmin(c)) ? mode : undefined;
    // v4.8: style=rich|brief (default HELP_STYLE), length=short|medium|long
    const { style, length } = richOptions((k) => body[k] ?? sp.get(k));
    return c.json(await answerHelp(c.env, question, { mode: m, reasoning: r, style, length }));
  });

  /**
   * GitHub push webhook of the documentation repository (secret GITHUB_WEBHOOK_SECRET; content type
   * application/json; event "push"). Starts a sync at once; the 10-minute cron is the fallback.
   */
  app.post("/webhooks/github", async (c) => {
    const secret = c.env.GITHUB_WEBHOOK_SECRET;
    if (!secret) return c.json({ error: "webhook not configured" }, 404);
    const body = await readBody(c.req.raw, 2 * 1024 * 1024);
    if (!(await verifyWebhook(secret, body, c.req.header("x-hub-signature-256") ?? null))) return c.json({ error: "bad signature" }, 401);
    const event = c.req.header("x-github-event") ?? "";
    if (event === "ping") return c.json({ ok: true, pong: true });
    if (event !== "push") return c.json({ ok: true, ignored: event || "no event" });
    let payload: { ref?: string; after?: string; repository?: { full_name?: string } } | null = null;
    try {
      payload = JSON.parse(new TextDecoder().decode(body));
    } catch {
      return c.json({ error: "body must be JSON (set the webhook's content type to application/json)" }, 400);
    }
    const ref = repoRef(c.env);
    if (!ref || payload?.repository?.full_name?.toLowerCase() !== repoName(ref).toLowerCase() || payload?.ref !== `refs/heads/${ref.branch}`) {
      return c.json({ ok: true, ignored: "another repository or branch" });
    }
    c.executionCtx.waitUntil(
      syncDocs(c.env, { reason: "webhook", budgetMs: 15_000, commit: typeof payload.after === "string" ? payload.after : undefined })
        .then((r) => console.log("docs sync (webhook)", JSON.stringify({ status: r.status, commit: r.commit, indexed: r.indexed.length, removed: r.removed.length, failed: r.failed.length, pending: r.pending })))
        .catch((e) => console.error("docs sync (webhook) failed", e)),
    );
    return c.json({ ok: true, syncing: true }, 202);
  });
}
