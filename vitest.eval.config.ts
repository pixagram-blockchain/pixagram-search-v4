import { defineConfig } from "vitest/config";
import { fileURLToPath } from "node:url";

// Offline evaluation (eval/offline/run.test.ts): slow, needs a chain snapshot and a local
// embedding server; not part of `npm test`.
export default defineConfig({
  resolve: {
    alias: { "cloudflare:workers": fileURLToPath(new URL("./test/harness/cloudflare-workers.ts", import.meta.url)) },
  },
  test: {
    include: ["eval/offline/**/*.test.ts"],
    environment: "node",
    testTimeout: 3_600_000,
    hookTimeout: 3_600_000,
    fileParallelism: false,
  },
});
