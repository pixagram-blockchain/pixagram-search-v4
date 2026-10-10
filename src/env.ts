/// <reference types="@cloudflare/workers-types" />

export interface EnrichMessage {
  /** posts.id */
  postId: number;
  author: string;
  permlink: string;
  /** Which stages to (re)run. Default: all that apply to the post type. */
  stages?: Stage[];
  /** Force recomputation even when the content hash is unchanged. */
  force?: boolean;
  /** Why it was enqueued — for logs only. */
  reason?: string;
}

/**
 * Enrichment stages, in dependency order.
 *   stats     decode → native stats, features, pHash/dHash, colours           (artworks)
 *   paph      PAPH fingerprint → copy-detection index, checked for copies     (artworks, no AI)
 *   embed     SigLIP image vector                                             (artworks)
 *   describe  VLM description → D1 + FTS                                      (artworks)
 *   concepts  normalised concepts from tags, title and the description        (artworks, no AI)
 *   text      SigLIP text vector of title/tags/description/caption, or blog   (artworks + blogs)
 */
export type Stage = "stats" | "paph" | "embed" | "describe" | "concepts" | "text";
export const ALL_STAGES: Stage[] = ["stats", "paph", "embed", "describe", "concepts", "text"];
export const BLOG_STAGES: Stage[] = ["text"];

export interface Env {
  DB: D1Database;
  ART: R2Bucket;
  CACHE: KVNamespace;
  /** Image vectors (SigLIP image tower), one per artwork, id = post id. */
  VEC: VectorizeIndex;
  /** Text vectors (SigLIP text tower) of artworks (title, caption, tags) and blog posts, id = post id. Optional. */
  VEC_TEXT?: VectorizeIndex;
  /** Vectors of the documentation chunks (DOCS_EMBED_MODEL), id = doc_chunks.id. Optional: without it help is lexical. */
  VEC_DOCS?: VectorizeIndex;
  AI: Ai;
  ENRICH_QUEUE: Queue<EnrichMessage>;
  INDEXER: DurableObjectNamespace;
  BACKFILL: Workflow;
  /** Copy detection: the PaphShard Durable Objects (src/paph). Optional: without it the paph stage and /copies are off. */
  PAPH?: DurableObjectNamespace;

  RPC_URL: string;
  RPC_FALLBACK_URLS?: string;
  APP_PREFIXES?: string;
  HF_EMBED_URL?: string;
  HF_TOKEN?: string; // secret
  EMBED_MODEL?: string;
  EMBED_DIM?: string;
  /** Image fed to SigLIP: "xbrz" (default, measured best), "nearest", "native", or a weighted mix "nearest:0.65,xbrz:0.35". */
  EMBED_VIEWS?: string;
  /** NaFlex models: the Space's MAX_NUM_PATCHES. Checked against the Space and part of the views label (a change re-embeds). */
  EMBED_PATCHES?: string;
  /** "on" (default) embeds title/caption/tags (artworks) and title/body (blogs) into VEC_TEXT. */
  TEXT_VECTORS?: string;
  VLM_BACKEND?: "chat" | "gemma" | "moondream" | "caption" | "scout" | "off" | string;
  /** The vision chat model of the "chat" backend (a model id the model table knows as vision-capable, e.g. @cf/zai-org/glm-5.3-flash). */
  VLM_MODEL?: string;
  /** Long side of the image sent to the VLM (default 512). */
  VLM_TARGET?: string;
  SCALER?: "xbrz" | "nearest" | string;
  UPSCALE_TARGET?: string;
  /** Stages skipped when the PIXA license says ai-training=false, e.g. "describe" or "describe,embed,text". */
  AI_TRAINING_FALSE_BLOCKS?: string;
  /** v2 compatibility: "true" = AI_TRAINING_FALSE_BLOCKS="describe". */
  RESPECT_AI_TRAINING_FLAG?: string;
  STORE_IN_R2?: string;
  MARKET_CUSTOM_JSON_IDS?: string;
  TAIL_BLOCKS_PER_TICK?: string;
  TAIL_IDLE_SECONDS?: string;
  /** Walk each account's comment history during the backfill (post_versions). Default "true". */
  HISTORY_BACKFILL?: string;
  /** Max account_history calls per account in one backfill step (each scans ≤ 2000 ops). Default 300. */
  HISTORY_MAX_CALLS?: string;
  /** Query planner for /ask: "rules", "llm", or "auto" (rules, LLM only when the rules are unsure). Default auto. */
  PLANNER_BACKEND?: string;
  PLANNER_MODEL?: string;
  /** Share of searches whose candidate features are logged for learning-to-rank (0..1). Default 1. */
  RANK_LOG_SAMPLE?: string;
  /** Platform documentation: "owner/repo" on GitHub (default pixagram-blockchain/information; "off" disables help). */
  DOCS_REPO?: string;
  DOCS_BRANCH?: string;
  /** Comma-separated folders (or files) of the repository to index; empty = every Markdown file. */
  DOCS_PATHS?: string;
  /** Workers AI embedding model of the documentation chunks (VEC_DOCS dimensions must match). */
  DOCS_EMBED_MODEL?: string;
  /** Dimensions of DOCS_EMBED_MODEL = those of VEC_DOCS (checked on every embedding). */
  DOCS_EMBED_DIM?: string;
  /** The retrieval task a question is embedded with on Qwen3-Embedding (docs/vectors.ts has the default). */
  DOCS_QUERY_INSTRUCTION?: string;
  /** Cosine above which a documentation chunk counts as relevant (default 0.5, read on bge-m3; re-read after a model change). */
  DOCS_MIN_SCORE?: string;
  /** Workers AI model that writes help answers from the documentation (JSON mode). */
  HELP_MODEL?: string;
  /**
   * "on": /suggest also proposes popular searches (three people on two days; see search/suggest.ts),
   * and /search records who ran what for it. Off by default: shown to everybody, they can be planted.
   */
  SUGGEST_POPULAR?: string;
  GITHUB_WEBHOOK_SECRET?: string; // secret, optional: enables POST /webhooks/github
  ADMIN_TOKEN?: string; // secret

  // ---- v4: models (src/llm) -------------------------------------------------------------------
  /** Model that plans questions the rules cannot (falls back to PLANNER_MODEL). */
  SEARCH_PLANNER_MODEL?: string;
  /** Model that reasons over the evidence (/ask); per complexity band below, else this one. */
  SEARCH_REASONING_MODEL?: string;
  SEARCH_REASONING_MODEL_SIMPLE?: string;
  SEARCH_REASONING_MODEL_NORMAL?: string;
  SEARCH_REASONING_MODEL_COMPLEX?: string;
  SEARCH_REASONING_MODEL_DEEP?: string;
  /** Model that writes /help answers (falls back to HELP_MODEL). */
  SEARCH_HELP_MODEL?: string;
  /** Default /help mode: auto (fast for short questions, else balanced) | fast | balanced | deep | expert. */
  HELP_MODE?: string;
  /** "on": in the modes above fast, English help questions get their excerpts reordered by the cross-encoder. Off by default (not evaluated on the documentation yet). */
  HELP_RERANK?: string;
  /** Cross-encoder reranker (Workers AI id), or "http" with SEARCH_RERANK_URL. */
  SEARCH_RERANKER_MODEL?: string;
  /** A TEI-style reranker endpoint (POST {query, texts} → [{index, score}]), used when the reranker model is "http". */
  SEARCH_RERANK_URL?: string;
  /** JSON: OpenAI-compatible endpoints, {"name": {"base_url", "api_key_env", "json", "reasoning"}} (llm/model.ts). */
  SEARCH_LLM_ENDPOINTS?: string;
  /** JSON: facts about models the built-in table does not know (style, reasoning control, price…). */
  SEARCH_MODEL_OVERRIDES?: string;
  /** Reasoning tokens allowed per level on top of the answer, "low:1024,medium:4096,high:12288". */
  SEARCH_REASONING_TOKENS?: string;
  /** Models a public /ask request may name (comma-separated); any other choice needs the admin token. */
  SEARCH_PUBLIC_MODELS?: string;
  SEARCH_LLM_TIMEOUT_MS?: string;

  // ---- v4: execution ----------------------------------------------------------------------------
  /** Default reasoning: auto (by mode) | none | low | medium | high. */
  SEARCH_REASONING?: string;
  /** Upper bound of the visible answer, in tokens, whatever the mode asks for. */
  SEARCH_MAX_OUTPUT_TOKENS?: string;
  /** Default /ask mode: auto | fast | balanced | deep | expert | v3. */
  SEARCH_DEFAULT_MODE?: string;
  /** The deepest mode a public request may ask for (default deep; expert needs the admin token). */
  SEARCH_MAX_MODE?: string;
  /** The deepest mode the search box (/query) chooses by itself (default balanced). */
  SEARCH_QUERY_MAX_MODE?: string;
  /** "false": no complexity routing; every question runs SEARCH_DEFAULT_MODE (or balanced). */
  SEARCH_AUTO_COMPLEXITY?: string;
  /** Complexity band bounds, "0.2,0.4,0.65,0.85" (trivial < simple < normal < complex < deep). */
  SEARCH_COMPLEXITY_BANDS?: string;
  /** Candidate pools of the /ask retrieval legs. */
  SEARCH_FTS_K?: string;
  SEARCH_SEMANTIC_K?: string;
  SEARCH_VISUAL_K?: string;
  SEARCH_CONCEPT_K?: string;
  SEARCH_HISTORY_K?: string;
  /** Cross-encoder reranking in the modes that use it (default true) and on /search when asked. */
  SEARCH_RERANK?: string;
  /** Candidates the cross-encoder sees, and how many it keeps. */
  SEARCH_RERANK_K?: string;
  SEARCH_FINAL_K?: string;
  /** /search with rerank=1: weight of the cross-encoder in the blended score (0..1). */
  SEARCH_RERANK_BLEND?: string;
  /** Verify the reasoning model's claims against the evidence (default true). */
  SEARCH_VERIFY_CLAIMS?: string;
  /** Confidence weights, "retrieval:0.25,reranking:0.2,agreement:0.2,verification:0.2,model:0.15". */
  SEARCH_CONFIDENCE_WEIGHTS?: string;
  /** Query expansion for /ask retrieval: rules (concept aliases, default) | off. */
  SEARCH_EXPANSION?: string;
  /** Return traces to callers that ask (trace=1); admin callers always may. */
  SEARCH_TRACE?: string;
  /** Share of /ask requests whose trace is stored in ask_log (0..1, default 0.1). */
  SEARCH_TRACE_SAMPLE?: string;
  /** Seconds a reasoned /ask answer is cached (keyed on the evidence, the model and the prompt). */
  SEARCH_ANSWER_CACHE_TTL?: string;

  // ---- v4.8: long-form answers (search/compose.ts, search/digest.ts, README "Rich answers") -------
  /**
   * rich (default): every answer carries a deterministic digest (facts, overview, caveats, follow-up
   * questions, related searches), and in balanced and above the model writes a long body, a
   * reasoning trail and its own follow-ups, verified sentence by sentence. brief: v4's answers.
   */
  SEARCH_ANSWER_STYLE?: string;
  /** Body length targets per mode, in words: "balanced:250,deep:450,expert:800". */
  SEARCH_ANSWER_WORDS?: string;
  /** rich: the mode auto never picks below (default balanced), so questions from the search box reach the model. */
  SEARCH_RICH_MIN_MODE?: string;
  /** Follow-up questions shown with an answer, at most (default 6). */
  SEARCH_FOLLOW_UPS?: string;
  /** Seconds a deferred elaboration's frozen context and its result are kept (default 1800). */
  SEARCH_ELABORATION_TTL?: string;
  /** rich (default): /help answers at length, with follow-up questions and related sections; brief: v4's. */
  HELP_STYLE?: string;
  /** /help body length targets per mode, in words: "fast:120,balanced:200,deep:350,expert:600". */
  HELP_ANSWER_WORDS?: string;

  // ---- copy detection (src/paph, README-V4 "Copy detection") -------------------------------------
  /** "false" turns the paph stage and the /copies routes off while keeping the binding. Default true. */
  PAPH_ENABLED?: string;
  /** Post ids per shard (default 100000). Part of every shard's name: choose it once. */
  PAPH_SHARD_SIZE?: string;
  /** Overrides of the interactive budget per shard, e.g. "verify:96,deadline_ms:600" (src/paph/budget.ts). */
  PAPH_BUDGET_QUERY?: string;
  /** Overrides of the enrichment stage's budget per shard. */
  PAPH_BUDGET_STAGE?: string;
  /** Wall-clock budget of an interactive check, from the request's start (default 900 ms). */
  PAPH_QUERY_MS?: string;
  /** Seconds an upload's answer is kept (default 86400; an answer with a shard missing: 600). */
  PAPH_CACHE_TTL?: string;
  /** Verdicts the stage stores, at or above (default suspected; the API lists copy and up unless asked). */
  PAPH_MIN_VERDICT?: string;
  /** XRank policy: safe (default: comparator 42's verdict on every pair it reports) | exact | fast. */
  PAPH_POLICY?: string;
  /** Hashing budget in pixels (default 768²); larger images are divided or box-filtered first. */
  PAPH_MAX_PIXELS?: string;
  /** pHash neighbours within this Hamming distance are checked too (default 10; 0 turns the channel off). */
  PAPH_PHASH_DISTANCE?: string;
  /** How many neighbours the pHash and embedding channels add (default 24 each). */
  PAPH_CHANNEL_K?: string;
  /** Per-client request limits (Workers Rate Limiting bindings, optional): search-like routes / expensive routes / suggestions while typing. */
  RL_PUBLIC?: RateLimit;
  RL_HEAVY?: RateLimit;
  RL_SUGGEST?: RateLimit;
}

export function bool(v: string | undefined, dflt = false): boolean {
  if (v === undefined || v === "") return dflt;
  return /^(1|true|yes|on)$/i.test(v);
}

export function int(v: string | undefined, dflt: number): number {
  const n = Number.parseInt(v ?? "", 10);
  return Number.isFinite(n) ? n : dflt;
}

export function num(v: string | undefined, dflt: number): number {
  const n = Number.parseFloat(v ?? "");
  return Number.isFinite(n) ? n : dflt;
}

export function list(v: string | undefined): string[] {
  return (v ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export const now = (): number => Math.floor(Date.now() / 1000);

/** Stages that the license flag ai-training=false blocks (explicit, instead of v2's single flag). */
export function aiTrainingBlocks(env: Env): Set<Stage> {
  // copy detection protects the artist and trains nothing: never blocked
  const explicit = list(env.AI_TRAINING_FALSE_BLOCKS).filter((s): s is Stage => s !== "paph" && (ALL_STAGES as string[]).includes(s));
  if (explicit.length) return new Set(explicit);
  return new Set(bool(env.RESPECT_AI_TRAINING_FLAG, false) ? (["describe"] as Stage[]) : []);
}
