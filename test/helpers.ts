import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { initCodecs } from "../src/enrich/decode";
import { paphRuntime, setPaphModule, type PaphRuntime } from "../src/paph/engine";

const require = createRequire(import.meta.url);

let ready: Promise<void> | null = null;

/** Compile the jSquash WASM codecs from node_modules (what wrangler does at build time). */
export function codecsReady(): Promise<void> {
  if (!ready) {
    ready = (async () => {
      const webp = await WebAssembly.compile(readFileSync(require.resolve("@jsquash/webp/codec/dec/webp_dec.wasm")));
      const png = await WebAssembly.compile(readFileSync(require.resolve("@jsquash/png/codec/pkg/squoosh_png_bg.wasm")));
      await initCodecs({ webpDecode: webp, png });
    })();
  }
  return ready;
}

export function fixture(name: string): Uint8Array {
  return new Uint8Array(readFileSync(new URL(`./fixtures/${name}`, import.meta.url)));
}

export function fixtureJson<T = any>(name: string): T {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8")) as T;
}

let paph: Promise<PaphRuntime> | null = null;

/** Compile PAPH-X from node_modules and register it (what the Worker's entry point does with wrangler's module). */
export function paphReady(): Promise<PaphRuntime> {
  if (!paph) {
    paph = (async () => {
      setPaphModule(await WebAssembly.compile(readFileSync(require.resolve("@pixagram/paph-x/wasm/paph.wasm"))));
      return paphRuntime();
    })();
  }
  return paph;
}
