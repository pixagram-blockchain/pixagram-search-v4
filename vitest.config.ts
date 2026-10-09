import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// Unit tests cover the pure modules (colour, hashing, features, planner, concepts, ranker, SQL)
// and the WebP/PNG codecs. Integration tests run the Worker's own code paths — ingestion,
// enrichment, search, /ask, duplicates, history — against a real SQLite (node:sqlite, with the
// D1 migrations applied) and in-memory stand-ins for KV, R2, Queues, Vectorize and Workers AI
// (test/harness). Only the Cloudflare runtime itself is not exercised here: see README for
// `wrangler deploy --dry-run` and the live evaluation (scripts/eval.py).
export default defineConfig({
  resolve: {
    alias: { "cloudflare:workers": fileURLToPath(new URL("./test/harness/cloudflare-workers.ts", import.meta.url)) },
  },
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
    testTimeout: 60000,
  },
});
