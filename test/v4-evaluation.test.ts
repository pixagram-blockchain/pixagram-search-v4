// The evaluation layer (spec §31-35): retrieval metrics, answer scoring against the question set's
// expected answers (correctness, abstention, evidence, forbidden text), the reliability composite,
// latency percentiles, and the model benchmark on frozen contexts (several models, one context).

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { artCorpus, GALLERY } from "./harness/corpus";
import { averagePrecision, ndcgAt, precisionAt, rankMetrics, recallAt, reciprocalRank } from "../src/evaluation/retrieval";
import { loadItems, scoreAnswer, summarizeBy, summarizeScores, type EvalItem } from "../src/evaluation/answer";
import { percentile, summarizePerformance } from "../src/evaluation/models";
import { benchmark, freezeContext, runOnContext } from "../src/evaluation/benchmark";

describe("retrieval metrics", () => {
  const rel = { a: 2, b: 1, c: 0, d: 1 };
  it("nDCG, precision, recall, MRR, AP", () => {
    expect(ndcgAt(["a", "b", "d"], rel)).toBeCloseTo(1, 10);
    expect(ndcgAt(["c", "a"], rel)).toBeLessThan(1);
    expect(precisionAt(["a", "c", "b"], rel, 2)).toBe(0.5);
    expect(recallAt(["a", "c"], rel, 10)).toBeCloseTo(1 / 3);
    expect(reciprocalRank(["c", "c2", "b"], rel)).toBeCloseTo(1 / 3);
    expect(averagePrecision(["a", "c", "b"], rel)).toBeCloseTo((1 + 2 / 3) / 3);
    expect(rankMetrics([], rel)).toMatchObject({ ndcg10: 0, mrr: 0, ap: 0 });
  });
});

describe("answer scoring", () => {
  const item = (o: Partial<EvalItem>): EvalItem => ({ id: "x", question: "q", query_type: "factual", lang: "en", answer_type: "author", expected_answer: "alice", acceptable_answers: [], required_evidence: [], ...o });
  it("per answer type, with acceptable answers and the expected status", () => {
    expect(scoreAnswer(item({}), { answer: "alice", answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({}), { answer: "@Alice", answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({ acceptable_answers: ["bob"] }), { answer: "bob", answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({ answer_type: "post", expected_answer: "/@alice/b" }), { answer: "alice/b", answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({ answer_type: "date", expected_answer: "2026-09-01" }), { answer: "2026-09-01T12:00:00Z", answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({ answer_type: "count", expected_answer: 3 }), { answer: 3, answer_text: "", took_ms: 1 }).correct).toBe(true);
    expect(scoreAnswer(item({ answer_type: "boolean", expected_answer: false, acceptable_answers: ["alice"] }), { answer: "alice", answer_text: "", took_ms: 1 }).correct).toBe(true);
    const clar = scoreAnswer(item({ answer_type: "status", expected_answer: null, expected_status: "clarify" }), { status: "clarify", answer: null, answer_text: "", took_ms: 1 });
    expect(clar).toMatchObject({ correct: true, should_abstain: true, abstained: true, abstention_correct: true });
    // "0 artworks" answers, even with status no_match; nothing at all abstains
    expect(scoreAnswer(item({ answer_type: "count", expected_answer: 0 }), { status: "no_match", answer: 0, answer_text: "", took_ms: 1 })).toMatchObject({ correct: true, abstained: false, abstention_correct: true });
    expect(scoreAnswer(item({ answer_type: "count", expected_answer: 0 }), { status: "no_match", answer: null, answer_text: "", took_ms: 1 }).abstained).toBe(true);
    // a confident wrong answer where the set expects a question back
    const wrong = scoreAnswer(item({ answer_type: "status", expected_answer: null, expected_status: "clarify" }), { status: "answered", answer: "bob", answer_text: "", confidence: 0.9, took_ms: 1 });
    expect(wrong).toMatchObject({ correct: false, abstention_correct: false });
  });

  it("injected text in the answer fails it; required evidence is checked; retrieval questions get rank metrics", () => {
    expect(scoreAnswer(item({ forbidden: ["mallory"] }), { answer: "alice", answer_text: "@mallory posted it", took_ms: 1 }).correct).toBe(false);
    expect(scoreAnswer(item({ required_evidence: ["/@alice/x"] }), { answer: "alice", answer_text: "", evidence: [{ path: "/@alice/x" }], took_ms: 1 }).evidence_ok).toBe(true);
    expect(scoreAnswer(item({ required_evidence: ["/@alice/x"] }), { answer: "alice", answer_text: "", took_ms: 1 }).evidence_ok).toBe(false);
    const r = scoreAnswer(item({ answer_type: "retrieval", rel: { "/@a/1": 2, "/@b/2": 1 } }), { answer: 2, answer_text: "", items: [{ author: "a", permlink: "1" }, { author: "c", permlink: "3" }, { author: "b", permlink: "2" }], took_ms: 1 });
    expect(r.correct).toBe(true);
    expect(r.retrieval!.mrr).toBe(1);
    expect(r.retrieval!.r10).toBe(1);
  });

  it("summaries: accuracy, abstention, reliability (0.4/0.3/0.2/0.1), latency percentiles, by category", () => {
    const s = [
      scoreAnswer(item({ id: "1" }), { answer: "alice", answer_text: "", took_ms: 10, mode: "fast" }),
      scoreAnswer(item({ id: "2", query_type: "temporal" }), { answer: "bob", answer_text: "", confidence: 0.95, took_ms: 30, mode: "fast" }),
      scoreAnswer(item({ id: "3" }), { answer: "alice", answer_text: "", took_ms: 2000, mode: "deep", grounding: { egs: 0.5, citation_accuracy: 1 }, usage: { input_tokens: 1000, output_tokens: 100, cost_usd: 0.0004, model_ms: 1800 } }),
    ];
    const all = summarizeScores(s);
    expect(all.accuracy).toBeCloseTo(2 / 3);
    expect(all.confident_errors).toBe(1);
    expect(all.reliability).toBeCloseTo(0.4 * (2 / 3) + 0.3 * ((1 + 0 + 0.5) / 3) + 0.2 * (2 / 3) + 0.1 * 1);
    expect(all.performance.engine_ms.p50).toBe(30);
    expect(all.performance.model_ms.calls).toBe(1);
    expect(all.performance.cost_usd.total).toBeCloseTo(0.0004);
    const by = summarizeBy(s);
    expect(Object.keys(by.by_type).sort()).toEqual(["factual", "temporal"]);
    expect(by.by_mode.deep.n).toBe(1);
    expect(percentile([5, 1, 3, 2, 4], 50)).toBe(3);
    expect(summarizePerformance([]).n).toBe(0);
  });

  it("the question set: at least 1,000 questions over the nine categories of spec §32", () => {
    const items = loadItems(readFileSync(new URL("../src/evaluation/datasets/questions.jsonl", import.meta.url), "utf8"));
    expect(items.length).toBeGreaterThanOrEqual(1000);
    const types = new Set(items.map((x) => x.query_type));
    for (const t of ["factual", "semantic", "visual", "temporal", "comparative", "multi_hop", "ambiguous", "multilingual", "adversarial"]) expect(types.has(t), t).toBe(true);
    expect(new Set(items.map((x) => x.id)).size).toBe(items.length);
    expect(new Set(items.map((x) => x.lang))).toEqual(new Set(["en", "fr", "de", "es", "it"]));
  });
});

describe("the model benchmark: one frozen context, several models", () => {
  it("every model gets the same messages; each reply is verified on the same cards", async () => {
    const env = await artCorpus(GALLERY);
    const ctx = await freezeContext(env, "What kind of art does @alice make?");
    expect(ctx.cards.map((c) => c.evidence_id)).toContain("R1");
    expect(ctx.deterministic.text).toContain("mostly tagged cat (2)");
    const seen: Array<{ model: string; user: string }> = [];
    env._ai.handler = (model, input) => {
      seen.push({ model, user: input.messages.find((m: any) => m.role === "user").content });
      const good = { status: "answered", answer: "Mostly cats [R1].", claims: [{ text: "Most are tagged cat, 2 of them.", evidence: ["R1"], kind: "fact" }], rationale: "R1." };
      const bad = { status: "answered", answer: "Mostly dragons, 7 of them [R1].", claims: [{ text: "Most are dragons, 7 of them.", evidence: ["R1"], kind: "fact" }], rationale: "R1." };
      const reply = model.includes("gemma") ? bad : good;
      return model.includes("nemotron") || model.includes("gemma") ? { choices: [{ message: { content: JSON.stringify(reply) } }], usage: { prompt_tokens: 800, completion_tokens: 60 } } : { response: reply, usage: { prompt_tokens: 800, completion_tokens: 60 } };
    };
    const models = ["@cf/openai/gpt-oss-120b", "@cf/nvidia/nemotron-3-120b-a12b", "@cf/google/gemma-4-26b-a4b-it"];
    const b = await benchmark(env, ["What kind of art does @alice make?"], models, { reasoning: "low" });
    expect(b.contexts).toHaveLength(1);
    expect(new Set(seen.map((s) => s.user)).size).toBe(1);
    expect(b.runs[models[0]][0].grounding).toMatchObject({ egs: 1, answer: "supported" });
    expect(b.runs[models[1]][0].grounding?.egs).toBe(1);
    expect(b.runs[models[2]][0].grounding?.egs).toBe(0);
    expect(b.runs[models[2]][0].claims[0].problems.join(" ")).toContain("number 7 is not in the evidence");
    expect(b.runs[models[0]][0].cost_usd).toBeCloseTo((800 * 0.35 + 60 * 0.75) / 1e6, 12);
    const failing = await runOnContext(env, ctx, "@cf/meta/llama-3.3-70b-instruct-fp8-fast");
    expect(failing.error ?? failing.answer).toBeTruthy();
  });
});
