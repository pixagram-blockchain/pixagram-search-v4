// The sentence of a "first" / "latest" answer is built around what the question asks: who posted
// it, when, or which artwork it is (and, when the question names the author, the artwork they
// posted). French agrees with "œuvre"; German writes the noun with a capital.

import { beforeAll, describe, expect, it } from "vitest";
import { makeEnv, type TestEnv } from "./harness/fakes";
import { upsertPost } from "../src/db/posts";
import { ask } from "../src/search/ask";

const T0 = Date.UTC(2026, 8, 1) / 1000;
let env: TestEnv;

beforeAll(async () => {
  env = makeEnv({ PLANNER_BACKEND: "rules" });
  // one artwork a day: alice's dragon, bob's cat, alice's cat, bob's dragon
  const rows: Array<[string, string, string, string]> = [
    ["alice", "red-dragon", "Red Dragon", "dragon"],
    ["bob", "black-cat", "Black cat", "cat"],
    ["alice", "cat-nap", "Cat nap", "cat"],
    ["bob", "dragon-two", "Dragon Two", "dragon"],
  ];
  let i = 0;
  for (const [author, permlink, title, concept] of rows) {
    const created = T0 + i++ * 86400;
    const { id } = await upsertPost(
      env,
      { author, permlink, type: "artwork", title, description: "", body: "", bodyLength: 0, category: "pixagram", tags: [concept], app: "pixagram/3", nsfw: false, aiTraining: null, licenseJson: null, royaltyPct: null, created, updated: created, deleted: false, netVotes: 1, payout: 0, children: 0, image: null },
      null,
    );
    await env.DB.prepare("INSERT INTO artworks (post_id, content_hash, mime, bytes, ai_nsfw, updated) VALUES (?, ?, 'image/png', 10, 0.01, ?)").bind(id, `h${id}`, created).run();
    await env.DB.prepare("INSERT INTO artwork_concepts (post_id, concept, confidence, source) VALUES (?, ?, 0.9, 'tag')").bind(id, concept).run();
  }
});

const say = async (question: string) => (await ask(env, { question, planner: "rules" })).answer_text;

describe("first and latest, worded by what is asked", () => {
  it("which artwork, from a named author: the artwork they posted", async () => {
    expect(await say("what is the first artwork from @alice")).toBe("The first artwork posted by @alice is “Red Dragon” (2026-09-01).");
    expect(await say("what is the latest artwork from @bob?")).toBe("The latest artwork posted by @bob is “Dragon Two” (2026-09-04).");
    expect(await say("quelle est la première œuvre de @alice ?")).toBe("La première œuvre postée par @alice est « Red Dragon » (2026-09-01).");
    expect(await say("was ist das erste Kunstwerk von @alice?")).toBe("Das erste „Kunstwerk“ von @alice ist „Red Dragon“ (2026-09-01).");
  });

  it("which artwork: the artwork, then who posted it", async () => {
    expect(await say("what is the first cat?")).toBe("The first cat artwork is “Black cat”, posted by @bob on 2026-09-02.");
    expect(await say("show me the latest artwork")).toBe("The latest artwork is “Dragon Two”, posted by @bob on 2026-09-04.");
    expect(await say("quel est le premier chat ?")).toBe("Le premier « chat » est « Black cat », posté par @bob le 2026-09-02.");
  });

  it("who, and when: the author, the date", async () => {
    expect(await say("who posted the first cat?")).toBe("The first cat artwork was posted by @bob on 2026-09-02: “Black cat”.");
    expect(await say("when was the first dragon posted?")).toBe("The first dragon artwork appeared on 2026-09-01 (“Red Dragon” by @alice).");
    expect(await say("qui a posté la première œuvre de chat ?")).toBe("Le premier « chat » a été posté par @bob le 2026-09-02 : « Black cat ».");
    expect(await say("wer hat die erste Katze gepostet?")).toBe("Das erste „Katze“ hat @bob am 2026-09-02 gepostet: „Black cat“.");
  });
});
