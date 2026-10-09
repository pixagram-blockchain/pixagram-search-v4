import { afterEach, describe, expect, it, vi } from "vitest";
import { embedTexts } from "../src/enrich/embed";

const MODEL = "google/siglip2-base-patch16-256";
const env = (over: Record<string, unknown> = {}) =>
  ({ HF_EMBED_URL: "https://space.example/embed", EMBED_MODEL: MODEL, EMBED_DIM: "768", ...over }) as any;
const unit = (d: number) => Array.from({ length: d }, (_, i) => (i === 0 ? 1 : 0));
const ok = (model = MODEL, d = 768) =>
  new Response(JSON.stringify({ model, dim: d, embeddings: [unit(d)] }), { status: 200, headers: { "content-type": "application/json" } });

afterEach(() => vi.unstubAllGlobals());

describe("embedding client", () => {
  it("accepts vectors from EMBED_MODEL", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => ok()));
    await expect(embedTexts(env(), ["cow"])).resolves.toMatchObject({ model: MODEL, dim: 768 });
  });

  it("refuses vectors from a model other than EMBED_MODEL, even with the same dimension", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => ok("google/siglip-base-patch16-256-multilingual")));
    await expect(embedTexts(env(), ["cow"])).rejects.toThrow(
      "embedding endpoint serves google/siglip-base-patch16-256-multilingual but EMBED_MODEL is google/siglip2-base-patch16-256",
    );
    // retryable: during a rollout the Space or the Worker catches up within minutes
    await expect(embedTexts(env(), ["cow"])).rejects.toMatchObject({ name: "EmbedUnavailable", retryable: true });
  });

  it("still rejects a dimension mismatch, permanently", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => ok(MODEL, 1152)));
    await expect(embedTexts(env(), ["cow"])).rejects.toMatchObject({ retryable: false });
  });

  it("accepts the stub model used for wiring tests", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => ok("stub")));
    await expect(embedTexts(env(), ["cow"])).resolves.toMatchObject({ dim: 768, model: "stub" });
  });

  it("treats a Hugging Face edge 502 as transient", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(async () => new Response("<!DOCTYPE html>", { status: 502 })));
    await expect(embedTexts(env(), ["cow"])).rejects.toMatchObject({ retryable: true });
  });
});
