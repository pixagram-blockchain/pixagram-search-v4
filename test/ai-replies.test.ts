// Workers AI replies in every form the binding returns them: parsed JSON (content-type exactly
// application/json), or the raw body stream for anything else (charset parameters, server-sent
// events), as partner models such as Moondream answer.

import { describe, expect, it } from "vitest";
import { aiReply, parseAiText, replyShape, UnreadableReply } from "../src/lib/ai";
import { describeImage, EmptyDescription, MOONDREAM_MODEL } from "../src/enrich/describe";
import { makeEnv } from "./harness/fakes";

const stream = (text: string) => new Response(text).body!;
const IMG = "data:image/png;base64,AAAA";

describe("reading replies", () => {
  it("objects pass through; streams, Responses and bytes are read as JSON", async () => {
    const o = { answer: "x" };
    expect(await aiReply(o)).toBe(o);
    expect(await aiReply(stream('{"answer":"A cat."}'))).toEqual({ answer: "A cat." });
    expect(await aiReply(new Response('{"caption":"A dog."}'))).toEqual({ caption: "A dog." });
    expect(await aiReply(new TextEncoder().encode('{"response":"hi"}'))).toEqual({ response: "hi" });
    expect(await aiReply("plain")).toBe("plain"); // what the binding parsed itself is left alone
  });

  it("a body that is neither JSON nor events is an error, never text to use", async () => {
    await expect(aiReply(stream(""))).rejects.toThrow(new UnreadableReply(""));
    await expect(aiReply(stream("<html>Bad gateway</html>"))).rejects.toThrow('unreadable reply: text "<html>Bad gateway</html>"');
  });

  it("server-sent events: text pieces joined in order, OpenAI deltas into `response`", () => {
    expect(parseAiText('data: {"answer":"A red"}\n\ndata: {"answer":" dragon."}\n\ndata: {"finish_reason":"stop"}\n\ndata: [DONE]\n')).toEqual({ answer: "A red dragon.", finish_reason: "stop" });
    expect(parseAiText('data: {"choices":[{"delta":{"content":"{\\"a\\":"}}]}\ndata: {"choices":[{"delta":{"content":"1}"}}]}\n')).toEqual({ response: '{"a":1}' });
    expect(parseAiText("just words")).toBe("just words");
  });

  it("error messages name what came back", () => {
    expect(replyShape({ answer: null, metrics: {} })).toBe("{answer,metrics}");
    expect(replyShape(stream("x"))).toBe("ReadableStream");
    expect(replyShape(null)).toBe("object");
    expect(replyShape("")).toBe("empty reply");
    expect(replyShape("  Internal error  ")).toBe('text "Internal error"');
  });
});

describe("descriptions from stream replies", () => {
  it("Moondream's JSON answer arriving as a stream is a description", async () => {
    const env = makeEnv();
    env._ai.handler = (_m, input) => stream(JSON.stringify(input.task === "query" ? { answer: '{"caption":"A knight on a horse at dawn.","tags":["knight","horse"]}', finish_reason: "stop" } : { caption: "unused" }));
    const r = await describeImage(env, "moondream", IMG, {});
    expect(r).toMatchObject({ model: MOONDREAM_MODEL, status: "ok", description: { caption: "A knight on a horse at dawn.", tags: ["knight", "horse"] } });
    expect(env._ai.calls.map((c) => c.input.task)).toEqual(["query"]);
  });

  it("the caption task streamed as server-sent events", async () => {
    const env = makeEnv();
    env._ai.handler = (_m, input) => (input.task === "query" ? stream('{"answer":null}') : stream('data: {"caption":"A lighthouse"}\n\ndata: {"caption":" on a cliff at night."}\n\n'));
    const r = await describeImage(env, "caption", IMG, {});
    expect(r.description.caption).toBe("A lighthouse on a cliff at night.");
  });

  it("an unreadable reply fails loudly, saying what came back, and is never stored as a caption", async () => {
    const env = makeEnv();
    env._ai.handler = () => stream("");
    await expect(describeImage(env, "moondream", IMG, {})).rejects.toThrow("empty reply");
    env._ai.handler = () => stream("upstream unavailable");
    await expect(describeImage(env, "caption", IMG, {})).rejects.toThrow('unreadable reply: text "upstream unavailable"');
    env._ai.handler = () => ({ answer: null, caption: "" }); // parsed, but empty
    await expect(describeImage(env, "moondream", IMG, {})).rejects.toBeInstanceOf(EmptyDescription);
  });
});
