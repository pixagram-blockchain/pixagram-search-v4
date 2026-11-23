// Minimal JSON-RPC client for the Pixa chain (Hive-compatible).
// Primary endpoint with fallbacks; retries on network errors and 5xx; no retry on RPC asserts.
//
// v4.8.4: a failed call moves on through every node (before: the primary and the first two
// fallbacks only, so three nodes down stopped the tail while three others answered), and the
// node that answered is tried first for the next ten minutes (per isolate), so a primary that hangs
// costs its timeout once instead of on every call. The timeout now covers reading the reply too: a
// node that sent its headers and then stalled used to hold a call until the invocation was killed.

export interface RpcOptions {
  url: string;
  fallbacks?: string[];
  timeoutMs?: number;
  /**
   * Tries after the first. Unset: every node once (at least three tries). Set: exactly that many
   * more, so a request path that allows itself one try (retries: 0) keeps its time bound.
   */
  retries?: number;
}

/** The node that answered last, per list of nodes (this isolate), and since when it is preferred. */
const PREFERRED = new Map<string, { url: string; since: number }>();
/** How long a fallback that answered is tried first; then the configured order (the primary) again. */
export const PREFER_MS = 10 * 60_000;
/** Forget the preferred nodes (tests). */
export function resetPreferredNodes(): void {
  PREFERRED.clear();
}

export class RpcError extends Error {
  constructor(
    message: string,
    public readonly method: string,
    public readonly code?: number,
    public readonly data?: unknown,
  ) {
    // Hivemind puts the actual reason in `data` behind a generic message ("Invalid parameters"):
    // keep it in the message, which is what logs and the indexer's lastError show.
    super(typeof data === "string" && data && !message.includes(data) ? `${message}: ${data}` : message);
    this.name = "RpcError";
  }
}

/**
 * The node's refusal to return a post it does not hold. Hivemind asserts instead of answering an
 * empty object: "Post a/p was deleted 1 time(s)" (code -31999, the reason in `data`) for a deleted
 * post, "Assert Exception:Post a/p does not exist" (code -32602) for one never created.
 */
export function isMissingPost(e: unknown, author: string, permlink: string): boolean {
  return missingReason(e, author, permlink) !== null;
}

/**
 * Why the node refused a post: "deleted" (Hivemind says it was deleted: final) or "unknown" (it
 * does not hold the post: never created, or created in a block Hivemind has not processed yet).
 * null: the error says nothing about this post's absence.
 */
export function missingReason(e: unknown, author: string, permlink: string): "deleted" | "unknown" | null {
  if (!(e instanceof RpcError)) return null;
  const text = `${e.message} ${typeof e.data === "string" ? e.data : JSON.stringify(e.data ?? "")}`;
  if (!text.includes(`Post ${author}/${permlink} `)) return null;
  if (/\bwas deleted\b/i.test(text)) return "deleted";
  if (/\bdoes not exist\b/i.test(text)) return "unknown";
  return null;
}

/** What get_content says about a post: its current state, or why the node does not return it. */
export type ContentState = { post: CondenserPost } | { absent: "deleted" | "unknown" };

export class ChainRpc {
  private readonly urls: string[];
  private readonly timeoutMs: number;
  private readonly attempts: number;
  private id = 1;

  constructor(opts: RpcOptions) {
    this.urls = [...new Set([opts.url, ...(opts.fallbacks ?? [])].filter(Boolean))];
    this.timeoutMs = opts.timeoutMs ?? 30_000;
    this.attempts = opts.retries === undefined ? Math.max(3, this.urls.length) : Math.max(1, opts.retries + 1);
  }

  /** The nodes in the order a call tries them: the one that answered lately first, then the configured order. */
  nodes(): string[] {
    const p = PREFERRED.get(this.urls.join(" "));
    if (!p || Date.now() - p.since > PREFER_MS || !this.urls.includes(p.url)) return this.urls;
    return [p.url, ...this.urls.filter((u) => u !== p.url)];
  }

  /** How many distinct nodes this client knows. */
  nodeCount(): number {
    return this.urls.length;
  }

  /** The node that answered the last call (a result or an assert); null before any. */
  lastNode: string | null = null;

  private answered(url: string): void {
    const key = this.urls.join(" ");
    if (url === this.urls[0]) PREFERRED.delete(key);
    else if (PREFERRED.get(key)?.url !== url) PREFERRED.set(key, { url, since: Date.now() });
  }

  /**
   * `o.from`: start with the from-th configured node (modulo their number) instead of the
   * preferred one, and leave the preference alone: v4.8.4's deferred posts ask a different node
   * on every retry, so one node whose Hivemind has stalled cannot hold a post back.
   */
  async call<T = unknown>(method: string, params: unknown = [], o: { from?: number } = {}): Promise<T> {
    let lastErr: unknown;
    const k = o.from === undefined ? 0 : ((Math.trunc(o.from) % this.urls.length) + this.urls.length) % this.urls.length;
    const order = o.from === undefined ? this.nodes() : [...this.urls.slice(k), ...this.urls.slice(0, k)];
    const remember = o.from === undefined;
    // by default every node once (at least three tries: with one node, that one again)
    const attempts = this.attempts;
    for (let attempt = 0; attempt < attempts; attempt++) {
      const url = order[attempt % order.length];
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), this.timeoutMs);
      try {
        const res = await fetch(url, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ jsonrpc: "2.0", id: this.id++, method, params }),
          signal: ctrl.signal,
        });
        if (res.status >= 500 || res.status === 429) {
          throw new Error(`HTTP ${res.status} from ${url}`);
        }
        // still under the timeout: a node that stalls after its headers is a failed attempt too
        const json = (await res.json()) as { result?: T; error?: { code: number; message: string; data?: unknown } };
        if (json.error) {
          // Assert exceptions are deterministic; do not retry them.
          throw new RpcError(json.error.message, method, json.error.code, json.error.data);
        }
        this.lastNode = url;
        if (remember) this.answered(url);
        return json.result as T;
      } catch (e) {
        if (e instanceof RpcError) {
          // the node works: it answered about the request
          this.lastNode = url;
          if (remember) this.answered(url);
          throw e;
        }
        lastErr = (e as { name?: string })?.name === "AbortError" ? new Error(`timed out after ${this.timeoutMs / 1000} s: ${url}`) : e;
        if (attempt < attempts - 1) await new Promise((r) => setTimeout(r, Math.min(1_000, 250 * (attempt + 1))));
      } finally {
        clearTimeout(t);
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
  }

  // ---- typed helpers -------------------------------------------------------

  getDynamicGlobalProperties() {
    return this.call<DynamicGlobalProperties>("database_api.get_dynamic_global_properties", {});
  }

  /** Blocks [start, start+count) in the modern block_api shape. */
  async getBlockRange(startingBlockNum: number, count: number): Promise<Block[]> {
    const r = await this.call<{ blocks: Block[] }>("block_api.get_block_range", {
      starting_block_num: startingBlockNum,
      count,
    });
    return r.blocks ?? [];
  }

  /**
   * Full current state of a post (json_metadata is a string here); null when the chain no longer
   * holds it (deleted) or never did. Legacy condenser nodes answer an empty object for those,
   * Hivemind asserts (see isMissingPost): both are an answer about the post, not a failure.
   */
  async getContent(author: string, permlink: string): Promise<CondenserPost | null> {
    const s = await this.getContentState(author, permlink);
    return "post" in s ? s.post : null;
  }

  /**
   * get_content, telling a deleted post ("was deleted": final) from one the node does not hold
   * ("does not exist", or a legacy node's empty object). The second is also what Hivemind answers
   * for a post created in a block it has not processed yet: the tail must not take it for a
   * deletion (src/chain/indexer-do.ts).
   */
  async getContentState(author: string, permlink: string, o: { from?: number } = {}): Promise<ContentState> {
    let r: CondenserPost;
    try {
      r = await this.call<CondenserPost>("condenser_api.get_content", [author, permlink], o);
    } catch (e) {
      const why = missingReason(e, author, permlink);
      if (why) return { absent: why };
      throw e;
    }
    if (!r || !r.author) return { absent: "unknown" };
    return { post: r };
  }

  /** Account names after `start` (inclusive), up to `limit` (max 1000). */
  lookupAccounts(start: string, limit = 1000): Promise<string[]> {
    return this.call<string[]>("condenser_api.lookup_accounts", [start, limit]);
  }

  /**
   * An account's own top-level posts, newest first, 20 per page (Hivemind limit).
   * Pass the last item of the previous page as start_author/start_permlink; it is excluded.
   */
  getAccountPosts(account: string, start?: { author: string; permlink: string }, limit = 20) {
    return this.call<BridgePost[]>("bridge.get_account_posts", {
      sort: "posts",
      account,
      limit,
      ...(start ? { start_author: start.author, start_permlink: start.permlink } : {}),
    });
  }

  /**
   * Newest posts chain-wide, 20 per page (Hivemind's limit). Pixa's Hivemind starts the page after
   * `start` (Hive's condenser repeats it). Used by the reconciliation (src/chain/reconcile.ts) to
   * spot anything the tail missed; `truncateBody` keeps an artwork's data URI out of the page.
   */
  getDiscussionsByCreated(start?: { author: string; permlink: string }, limit = 20, truncateBody?: number) {
    return this.call<CondenserPost[]>("condenser_api.get_discussions_by_created", [
      {
        tag: "",
        limit,
        ...(truncateBody ? { truncate_body: truncateBody } : {}),
        ...(start ? { start_author: start.author, start_permlink: start.permlink } : {}),
      },
    ]);
  }

  /**
   * One page of an account's history, newest first from `start` (-1 = latest), filtered by the
   * operation bitmask (comment_operation = bit 1 → 2). The node scans at most 2000 operations per
   * call; when the filter finds nothing in that window it answers with an assert that names the
   * sequence to continue from — returned here as `{ history: [], next: N }`.
   */
  async getAccountHistory(account: string, start: number, limit: number, operationFilterLow?: number): Promise<{ history: AccountHistoryEntry[]; next: number | null }> {
    // The node asserts start >= limit - 1 (start is a 0-based sequence number).
    if (start >= 0) limit = Math.min(limit, start + 1);
    try {
      const r = await this.call<{ history: AccountHistoryEntry[] }>("account_history_api.get_account_history", {
        account,
        start,
        limit,
        ...(operationFilterLow ? { operation_filter_low: operationFilterLow } : {}),
      });
      const h = r.history ?? [];
      // Fewer than `limit` results can also mean the 2000-operation scan cap was hit, so keep going
      // below the oldest result; an empty page without the assert means the start was reached.
      const minSeq = h.length ? Math.min(...h.map((e) => e[0])) : null;
      return { history: h, next: minSeq === null || minSeq <= 0 ? null : minSeq - 1 };
    } catch (e) {
      const m = e instanceof RpcError ? /set start=(\d+)/.exec(e.message + JSON.stringify(e.data ?? "")) : null;
      if (m) return { history: [], next: Number(m[1]) };
      throw e;
    }
  }
}

/** [sequence, entry] as returned by account_history_api. */
export type AccountHistoryEntry = [
  number,
  { trx_id: string; block: number; trx_in_block: number; op_in_trx: number; virtual_op: boolean; timestamp: string; op: Operation },
];

// ---- chain types (subset) ----------------------------------------------------

export interface DynamicGlobalProperties {
  head_block_number: number;
  last_irreversible_block_num: number;
  time: string; // "2026-09-27T20:48:03"
}

export interface Block {
  block_id: string;
  previous: string;
  timestamp: string;
  witness: string;
  transactions: Transaction[];
  transaction_ids?: string[];
}

export interface Transaction {
  operations: Operation[];
}

/** block_api shape: { type: "comment_operation", value: {...} }. condenser shape: ["comment", {...}]. */
export type Operation = { type: string; value: Record<string, any> } | [string, Record<string, any>];

export interface CommentOp {
  parent_author: string;
  parent_permlink: string;
  author: string;
  permlink: string;
  title: string;
  body: string;
  json_metadata: string;
}

export interface CustomJsonOp {
  required_auths: string[];
  required_posting_auths: string[];
  id: string;
  json: string;
}

export interface CondenserPost {
  author: string;
  permlink: string;
  parent_author: string;
  parent_permlink: string;
  category?: string;
  title: string;
  body: string;
  body_length?: number;
  json_metadata: string | Record<string, unknown>;
  created: string;
  last_update?: string;
  updated?: string;
  depth: number;
  children: number;
  net_votes?: number;
  active_votes?: unknown[];
  pending_payout_value?: string; // "14.469 PXS"
  total_payout_value?: string;
  curator_payout_value?: string;
  author_payout_value?: string;
  payout?: number;
}

export type BridgePost = CondenserPost;

/** "14.469 PXS" -> 14.469 */
export function parseAsset(v: string | number | undefined | null): number {
  if (v === undefined || v === null) return 0;
  if (typeof v === "number") return v;
  const n = Number.parseFloat(v.split(" ")[0]);
  return Number.isFinite(n) ? n : 0;
}

/** Chain timestamps are UTC without a zone suffix. */
export function chainTime(s: string | undefined): number {
  if (!s) return 0;
  const t = Date.parse(s.endsWith("Z") ? s : `${s}Z`);
  return Number.isFinite(t) ? Math.floor(t / 1000) : 0;
}

/** Normalise both operation encodings to { type, value } with the short type name. */
export function normalizeOp(op: Operation): { type: string; value: Record<string, any> } {
  if (Array.isArray(op)) return { type: op[0], value: op[1] };
  return { type: op.type.replace(/_operation$/, ""), value: op.value };
}
