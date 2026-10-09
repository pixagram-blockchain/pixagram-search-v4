import { describe, expect, it } from "vitest";
import { captionKeywords, describeImage, EmptyDescription, isUsable, MOONDREAM_MODEL, normalizeDescription, replyText, SCOUT_MODEL } from "../src/enrich/describe";
import { maxEdits, pickCorrection, spellTokens, suggest } from "../src/search/spell";
import { docTerms, updateVocab } from "../src/db/posts";
import { makeEnv } from "./harness/fakes";

const IMG = "data:image/png;base64,AAAA";

describe("VLM replies (the v2 empty-caption bug)", () => {
  it("replyText reads the task's field and never the envelope", () => {
    const envelope = { answer: null, finish_reason: "stop", metrics: { tokens: 12 } };
    expect(replyText(envelope, "answer")).toBeNull(); // v2 stringified this and stored "" as done
    expect(replyText({ answer: '{"caption":"x"}' }, "answer")).toBe('{"caption":"x"}');
    expect(replyText({ result: { caption: "A cat." } }, "caption")).toBe("A cat.");
    expect(replyText({ response: { caption: "obj" } }, "response")).toBe('{"caption":"obj"}'); // JSON mode object
    expect(replyText({ choices: [{ message: { content: "hi" } }] }, "response")).toBe("hi");
    expect(replyText("plain", "answer")).toBe("plain");
  });

  it("a description must say something", () => {
    expect(isUsable(normalizeDescription({ caption: "", tags: [] }))).toBe(false);
    expect(isUsable(normalizeDescription({ caption: "A cat." }))).toBe(false); // 4 letters
    expect(isUsable(normalizeDescription({ caption: "A cat sleeping on a red sofa." }))).toBe(true);
    expect(isUsable(normalizeDescription({ caption: "", tags: ["cat", "sofa"] }))).toBe(true);
    expect(normalizeDescription({ caption: "x", tags: "Cat, SOFA; cat" })!.tags).toEqual(["cat", "sofa"]);
    expect(captionKeywords("A girl with green eyes wearing a red hat in the rain")).toEqual(["girl", "green", "eyes", "red", "hat", "rain"]);
  });

  it("moondream: query task first, caption task when the JSON is unusable, error when both are empty", async () => {
    const env = makeEnv();
    env._ai.handler = (_m, input) =>
      input.task === "query" ? { answer: '```json\n{"caption":"A knight on a horse.","subjects":["Knight"],"tags":["knight","horse"],"nsfw":0}\n```' } : { caption: "unused" };
    const ok = await describeImage(env, "moondream", IMG, { title: "Sir" });
    expect(ok).toMatchObject({ model: MOONDREAM_MODEL, status: "ok", description: { caption: "A knight on a horse.", subjects: ["knight"] } });
    expect(env._ai.calls.map((c) => c.input.task)).toEqual(["query"]);

    env._ai.calls.length = 0;
    env._ai.handler = (_m, input) => (input.task === "query" ? { answer: null, finish_reason: "length" } : { caption: "A small red dragon breathing fire over a castle." });
    const fb = await describeImage(env, "moondream", IMG, {});
    expect(fb.status).toBe("caption_only");
    expect(fb.description.tags).toEqual(expect.arrayContaining(["dragon", "castle"]));
    expect(env._ai.calls.map((c) => c.input.task)).toEqual(["query", "caption"]);

    env._ai.handler = () => ({ answer: null, caption: "" });
    await expect(describeImage(env, "moondream", IMG, {})).rejects.toBeInstanceOf(EmptyDescription);
  });

  it("scout uses a JSON schema and fails loudly on an empty reply", async () => {
    const env = makeEnv();
    env._ai.handler = (model) => (model === SCOUT_MODEL ? { response: { caption: "A lighthouse at night by the sea.", tags: ["lighthouse", "night"] } } : {});
    const r = await describeImage(env, "scout", IMG, {});
    expect(r.description.tags).toEqual(["lighthouse", "night"]);
    expect(env._ai.calls[0].input.response_format.type).toBe("json_schema");
    env._ai.handler = () => ({ response: "" });
    await expect(describeImage(env, "scout", IMG, {})).rejects.toBeInstanceOf(EmptyDescription);
  });
});

describe("spelling", () => {
  it("only checks words that could be typos", () => {
    expect(maxEdits(4)).toBe(1);
    expect(maxEdits(8)).toBe(2);
    expect(maxEdits(12)).toBe(3);
    // concept aliases, colour/tone words, stopwords and short words are never corrected
    expect(spellTokens("beach chien pastel with the dog elodrado")).toEqual(["elodrado"]);
    expect(spellTokens("猫 cat")).toEqual([]);
  });

  it("picks the closest term, ties to the more frequent, nothing for a known word", () => {
    expect(pickCorrection("elodrado", [{ term: "eldorado", df: 1 }, { term: "colorado", df: 9 }])).toMatchObject({ to: "eldorado", distance: 1 });
    expect(pickCorrection("dragn", [{ term: "dragon", df: 2 }, { term: "drag", df: 5 }])).toMatchObject({ to: "drag", distance: 1 }); // both 1 edit: more frequent
    expect(pickCorrection("wizard", [{ term: "wizard", df: 1 }])).toBeNull();
    expect(pickCorrection("zzzzzz", [{ term: "pizza", df: 3 }])).toBeNull();
  });

  it("suggest() works off the vocabulary that indexing maintains", async () => {
    const env = makeEnv();
    const doc = { title: "Eldorado", description: "the golden city", tags: "gold city", ai_caption: "", ai_tags: "", author: "wang" };
    await updateVocab(env.DB, new Set(), docTerms(doc));
    const s = await suggest(env.DB, "elodrado city");
    expect(s).toEqual([expect.objectContaining({ from: "elodrado", to: "eldorado" })]);
    // when the document goes away the term stops being suggested
    await updateVocab(env.DB, docTerms(doc), new Set());
    expect(await suggest(env.DB, "elodrado")).toEqual([]);
  });
});
