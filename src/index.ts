// Worker entry: HTTP API, queue consumer, cron, plus the Durable Object and Workflow classes.

import WEBP_DEC_WASM from "@jsquash/webp/codec/dec/webp_dec.wasm";
// @ts-expect-error the package ships a wasm-bindgen .d.ts for this file; wrangler turns the import into a WebAssembly.Module
import PNG_WASM from "@jsquash/png/codec/pkg/squoosh_png_bg.wasm";
import PAPH_WASM from "@pixagram/paph-x/wasm/paph.wasm";
import { app } from "./api";
import type { Env, EnrichMessage } from "./env";
import { initCodecs } from "./enrich/decode";
import { handleEnrichBatch } from "./enrich/consumer";
import { sweep } from "./enrich/sweeper";
import { indexerStub } from "./chain/indexer-do";
import { refreshCounters } from "./chain/refresh";
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
      // Sweeper: re-drive enrichment that exhausted its queue retries (e.g. a long Space outage).
      ctx.waitUntil(sweep(env, 25).then((r) => r.enqueued && console.log("sweeper", JSON.stringify(r))).catch((e) => console.error("sweeper failed", e)));
      // Documentation: a conditional request to GitHub (free when nothing changed), then whatever
      // changed, plus vectors still missing. The push webhook, when configured, is faster.
      ctx.waitUntil(
        syncDocs(env, { reason: "cron", budgetMs: 120_000 })
          .then((r) => r.status !== "unchanged" && r.status !== "disabled" && console.log("docs sync", JSON.stringify({ status: r.status, commit: r.commit, indexed: r.indexed.length, removed: r.removed.length, failed: r.failed, pending: r.pending, embedded: r.embedded, error: r.error })))
          .catch((e) => console.error("docs sync failed", e)),
      );
      // Watchdog: a running indexer must always have an alarm armed.
      try {
        if (await indexerStub(env).ensureAlarm()) console.warn("indexer alarm re-armed by watchdog");
      } catch (e) {
        console.error("watchdog failed", e);
      }
      return;
    }
    // Nightly: counters of posts inside their payout window, background samples, log retention.
    ctx.waitUntil(refreshCounters(env));
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
