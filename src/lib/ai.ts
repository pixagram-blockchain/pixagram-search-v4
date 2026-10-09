// Reading Workers AI replies.
//
// The AI binding (workerd's Ai.run) parses a reply only when its content-type is exactly
// "application/json"; anything else ("application/json; charset=utf-8", "text/event-stream", …)
// comes back as the raw ReadableStream. Cloudflare's own models answer with the exact type, but
// partner models need not. In production every Moondream reply looked like an object with no
// fields ({} in the error), which is what such a stream looks like, and every description failed
// as "no usable caption".

/** A reply that is neither JSON nor server-sent events (an empty body, an error page…). */
export class UnreadableReply extends Error {
  constructor(public readonly text: string) {
    super(text.trim() ? `unreadable reply: text "${text.trim().slice(0, 160)}"` : "empty reply");
    this.name = "UnreadableReply";
  }
}

/**
 * The reply as JSON whatever form the binding returned it in. A body that is not JSON is an
 * UnreadableReply, never text to use: an error page must not become a caption or an answer.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any -- replies are validated by each caller
export async function aiReply(r: unknown): Promise<any> {
  let text: string | null = null;
  if (typeof ReadableStream !== "undefined" && r instanceof ReadableStream) text = await new Response(r).text();
  else if (typeof Response !== "undefined" && r instanceof Response) text = await r.text();
  else if (r instanceof ArrayBuffer || ArrayBuffer.isView(r)) text = new TextDecoder().decode(r as ArrayBuffer | ArrayBufferView);
  if (text === null) return r;
  const t = text.trim();
  try {
    return JSON.parse(t);
  } catch {
    // not one JSON document
  }
  const events = parseEvents(t);
  if (events) return events;
  throw new UnreadableReply(t);
}

/** JSON, server-sent events (see parseEvents), or else the text itself. */
export function parseAiText(text: string): unknown {
  const t = text.trim();
  try {
    return JSON.parse(t);
  } catch {
    return parseEvents(t) ?? t;
  }
}

/**
 * Server-sent events ("data: {...}" lines, as a streamed reply arrives): their string fields are
 * joined in order (OpenAI-style deltas into `response`), other fields keep the last value. Null
 * when the text has no event.
 */
export function parseEvents(t: string): Record<string, unknown> | null {
  const events = t
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trim())
    .filter((d) => d && d !== "[DONE]");
  if (!events.length) return null;
  const out: Record<string, unknown> = {};
  for (const d of events) {
    let o: any;
    try {
      o = JSON.parse(d);
    } catch {
      continue;
    }
    if (!o || typeof o !== "object") continue;
    const delta = o.choices?.[0]?.delta?.content;
    if (typeof delta === "string") {
      out.response = String(out.response ?? "") + delta;
      continue;
    }
    for (const [k, v] of Object.entries(o)) out[k] = typeof v === "string" && typeof out[k] === "string" ? (out[k] as string) + v : v;
  }
  return out;
}

/** What a reply looked like, for error messages: its keys, its type, or the start of its text. */
export function replyShape(r: unknown): string {
  if (typeof r === "string") return r.trim() ? `text "${r.trim().slice(0, 80)}"` : "empty reply";
  if (!r || typeof r !== "object") return typeof r;
  const name = Object.getPrototypeOf(r)?.constructor?.name;
  if (name && name !== "Object") return name;
  return `{${Object.keys(r as object).slice(0, 8).join(",")}}`;
}
