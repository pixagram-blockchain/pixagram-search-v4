// Moondream on Workers AI: not a chat model but a set of vision tasks ("query" answers a question
// about the image, "caption" describes it). Used by the describe stage (enrich/describe.ts).
//
// Each task returns its own field (answer, caption) inside an envelope ({finish_reason, metrics, …}).
// Only that field is read: v2 stringified the whole envelope when the answer was missing and stored
// the empty description that came out as done.

import type { Env } from "../../env";
import { aiReply, replyShape } from "../../lib/ai";

export const MOONDREAM_MODEL = "@cf/moondream/moondream3.1-9B-A2B";

type AiRunner = { run: (model: string, input: unknown) => Promise<unknown> };

/** Text of a Workers AI reply for a given field, never the envelope. */
export function replyText(r: unknown, field: "answer" | "caption" | "response"): string | null {
  if (typeof r === "string") return r;
  if (!r || typeof r !== "object") return null;
  const o = r as Record<string, any>;
  const direct = o[field] ?? o.result?.[field];
  if (typeof direct === "string") return direct;
  if (direct && typeof direct === "object") return JSON.stringify(direct); // JSON mode may return the object itself
  const choice = o.choices?.[0]?.message?.content;
  if (typeof choice === "string") return choice;
  return null;
}

export interface MoondreamReply {
  text: string | null;
  /** what the reply looked like, for error messages */
  shape: string;
}

/** The "query" task: a question about the image (the describe prompt asks for JSON). */
export async function moondreamQuery(env: Env, pngDataUri: string, question: string): Promise<MoondreamReply> {
  const ai = env.AI as unknown as AiRunner;
  const r = await aiReply(await ai.run(MOONDREAM_MODEL, { task: "query", image: pngDataUri, question, reasoning: false, temperature: 0.2, max_tokens: 700 }));
  return { text: replyText(r, "answer"), shape: replyShape(r) };
}

/** The "caption" task: always prose, never JSON. */
export async function moondreamCaption(env: Env, pngDataUri: string): Promise<MoondreamReply> {
  const ai = env.AI as unknown as AiRunner;
  const r = await aiReply(await ai.run(MOONDREAM_MODEL, { task: "caption", image: pngDataUri, caption_length: "normal", temperature: 0.2, max_tokens: 300 }));
  return { text: replyText(r, "caption"), shape: replyShape(r) };
}
