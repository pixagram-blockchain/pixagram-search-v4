// Worker entry: HTTP API, queue consumer, cron, plus the Durable Object and Workflow classes.

import WEBP_DEC_WASM from "@jsquash/webp/codec/dec/webp_dec.wasm";
// @ts-expect-error the package ships a wasm-bindgen .d.ts for this file; wrangler turns the import into a WebAssembly.Module
import PNG_WASM from "@jsquash/png/codec/pkg/squoosh_png_bg.wasm";
import PAPH_WASM from "@pixagram/paph-x/wasm/paph.wasm";
import { app } from "./api";
import type { Env, EnrichMessage } from "./env";
import { num } from "./env";
import { initCodecs } from "./enrich/decode";
import { handleEnrichBatch } from "./enrich/consumer";
import { sweep, sweepBudget } from "./enrich/sweeper";
import { indexerStub } from "./chain/indexer-do";
import { refreshCounters } from "./chain/refresh";
import { reconcileFull, reconcileRecent } from "./chain/reconcile";
import { refreshBackground } from "./search/background";
import { pruneLogs } from "./search/feedback";
import { syncDocs } from "./docs/sync";
import { setPaphModule } from "./paph/engine";
import { paphGcPass, paphHealWire3 } from "./paph/copies";
import { paphEnabled } from "./paph/shards";

export { ChainIndexer } from "./chain/indexer-do";
export { BackfillWorkflow } from "./chain/backfill";
export { PaphShard } from "./paph/shard-do";

// Compile-once WASM codecs (WebP decode, PNG decode/encode) for this isolate.
initCodecs({ webpDecode: WEBP_DEC_WASM, png: PNG_WASM });
// PAPH-X (copy detection), for the Worker and the PaphShard Durable Objects alike; instantiated on first use.
setPaphModule(PAPH_WASM);

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> | Response {
    return app.fetch(request, env, ctx);
  },

  async queue(batch: MessageBatch<EnrichMessage>, env: Env): Promise<void> {
    await handleEnrichBatch(batch, env);
  },

  async scheduled(event: ScheduledController, env: Env, ctx: ExecutionContext): Promise<void> {
    if (event.cron === "*/10 * * * *") {
      // Sweeper: re-drive enrichment that exhausted its queue retries (e.g. a long Space outage),
      // as much per run as the Space's CPUs embed (sweepBudget); while the Space is not ready it
      // holds the embedding stages back and logs why.
      ctx.waitUntil(
        sweepBudget(env)
          .then((max) => sweep(env, max))
          .then((r) => (r.enqueued || r.held) && console.log("sweeper", JSON.stringify(r)))
          .catch((e) => console.error("sweeper failed", e)),
      );
      // Documentation: a conditional request to GitHub (free when nothing changed), then whatever
      // changed, plus vectors still missing. The push webhook, when configured, is faster.
      ctx.waitUntil(
        syncDocs(env, { reason: "cron", budgetMs: 120_000 })
          .then((r) => r.status !== "unchanged" && r.status !== "disabled" && console.log("docs sync", JSON.stringify({ status: r.status, commit: r.commit, indexed: r.indexed.length, removed: r.removed.length, failed: r.failed, pending: r.pending, embedded: r.embedded, error: r.error })))
          .catch((e) => console.error("docs sync failed", e)),
      );
      // Watchdog: a running indexer always has an alarm armed and ticks; one never started (a fresh
      // deployment) is started. One stopped with POST /admin/indexer/stop stays stopped.
      try {
        const w = await indexerStub(env).watchdog();
        if (w.action !== "none") console.warn(`indexer ${w.action} by the watchdog: ${w.why}`);
      } catch (e) {
        console.error("watchdog failed", e);
      }
      // Posts the tail set aside because their ingestion kept failing: tried again.
      try {
        const r = await indexerStub(env).retryParked();
        if (r.retried) console.log("indexer parked retry", JSON.stringify(r));
      } catch (e) {
        console.error("parked retry failed", e);
      }
      // v4.8.3: Hivemind's newest posts of the last RECONCILE_HOURS against the index; anything the
      // tail missed is ingested now (logged by reconcileRecent). v4.8.4: then the next pages of the
      // rolling pass over the whole listing (RECONCILE_FULL_PAGES), one after the other so the two
      // never ingest the same post at once.
      ctx.waitUntil(
        reconcileRecent(env)
          .catch((e) => console.error("reconcile failed", e))
          .then(() => reconcileFull(env))
          .then((r) => r && (r.ingested || r.pass_done || r.error) && console.log("reconcile full pass", JSON.stringify(r)))
          .catch((e) => console.error("reconcile full pass failed", e)),
      );
      return;
    }
    // Nightly: counters of posts inside their payout window, background samples, log retention.
    ctx.waitUntil(refreshCounters(env));
    // v4.8.3: the reconciliation over two days (RECONCILE_DEEP_HOURS), for anything older than the
    // ten-minute runs look at.
    ctx.waitUntil(
      reconcileRecent(env, { hours: num(env.RECONCILE_DEEP_HOURS, 48), maxPages: 200, maxIngest: 300, budgetMs: 600_000 })
        .then((r) => console.log("reconcile nightly", JSON.stringify(r)))
        .catch((e) => console.error("reconcile nightly failed", e)),
    );
    ctx.waitUntil(Promise.all([refreshBackground(env, "image"), refreshBackground(env, "text")]).catch((e) => console.error("background refresh failed", e)));
    ctx.waitUntil(pruneLogs(env).catch((e) => console.error("prune failed", e)));
    // copy detection: works whose posts were deleted leave the shards (their removal is best effort);
    // then the verdicts reached on wire 3 that no re-check will replace are compared again
    if (paphEnabled(env)) {
      ctx.waitUntil(
        paphGcPass(env)
          .then((r) => (r.removed || r.reset || r.failed.length) && console.log("paph gc", JSON.stringify(r)))
          .catch((e) => console.error("paph gc failed", e))
          .then(() => paphHealWire3(env, 200))
          .then((r) => r.found && console.log("paph heal", JSON.stringify(r)))
          .catch((e) => console.error("paph heal failed", e)),
      );
    }
  },
} satisfies ExportedHandler<Env, EnrichMessage>;
