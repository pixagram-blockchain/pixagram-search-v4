// The chain tail's control calls against a tick in flight: a Durable Object delivers other calls
// while alarm() awaits the chain, so start()/stop() must win over the tick's stale state.
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChainIndexer } from "../src/chain/indexer-do";
import { makeEnv } from "./harness/fakes";

class FakeStorage {
  m = new Map<string, any>();
  alarm: number | null = null;
  async get(k: string) {
    return structuredClone(this.m.get(k));
  }
  async put(k: string, v: any) {
    this.m.set(k, structuredClone(v));
  }
  async getAlarm() {
    return this.alarm;
  }
  async setAlarm(t: number) {
    this.alarm = t;
  }
  async deleteAlarm() {
    this.alarm = null;
  }
}

/** A chain whose get_block_range waits until released, so a control call can land mid-tick. */
function gatedChain() {
  let release!: () => void;
  const gate = new Promise<void>((r) => (release = r));
  let entered!: () => void;
  const inRange = new Promise<void>((r) => (entered = r));
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_u: unknown, init: any) => {
      const { method, id } = JSON.parse(init.body);
      const ok = (result: unknown) => new Response(JSON.stringify({ jsonrpc: "2.0", id, result }));
      if (method === "database_api.get_dynamic_global_properties") return ok({ head_block_number: 2000, last_irreversible_block_num: 1990, time: "2026-10-04T00:00:00" });
      if (method === "block_api.get_block_range") {
        entered();
        await gate;
        return ok({ blocks: [{ timestamp: "2026-10-04T00:00:00", transactions: [] }] });
      }
      throw new Error(`unexpected ${method}`);
    }),
  );
  return { release, inRange };
}

afterEach(() => vi.unstubAllGlobals());

describe("ChainIndexer", () => {
  it("a stop() during a tick stays stopped, with no alarm", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release, inRange } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    const tick = doo.alarm();
    await inRange;
    await doo.stop();
    release();
    await tick;
    const after = await doo.status();
    expect(after.running).toBe(false);
    expect(after.alarmAt).toBeNull();
    expect(after.ticks).toBe(1); // the tick's counters still count
  });

  it("a start(from) replay during a tick keeps its cursor", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release, inRange } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    const tick = doo.alarm();
    await inRange;
    await doo.start(500);
    release();
    await tick;
    const after = await doo.status();
    expect(after.cursor).toBe(500);
    expect(after.running).toBe(true);
    expect(after.alarmAt).not.toBeNull();
  });

  it("an undisturbed tick advances the cursor and re-arms", async () => {
    const storage = new FakeStorage();
    const doo = new ChainIndexer({ storage } as any, makeEnv() as any);
    const { release } = gatedChain();
    await storage.put("state", { running: true, cursor: 1000 });
    release();
    await doo.alarm();
    const after = await doo.status();
    expect(after.cursor).toBe(1001);
    expect(after.alarmAt).not.toBeNull();
  });
});
