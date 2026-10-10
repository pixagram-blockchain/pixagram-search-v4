# pixagram-search v4

Search, evidence-based answers and platform help for [Pixagram](https://pixagram.com), behind one
search box. **v4.8** turns the answers into long-form ones: a deterministic digest, the model's
body and reasoning trail verified sentence by sentence, follow-up questions, related searches, a
text overview of every search, and an elaboration the search box fetches once the index has
answered — see [Rich answers](#rich-answers-v48) below. **v4.8.1** moves the models to the
stronger ones Workers AI serves in October 2026 — GLM-5.3-Flash for reasoning and help, GLM-5.3
for the deep band, Qwen3-Embedding for the documentation vectors — and gives the description
stage a configurable chat model; README-V4 "Models, October 2026" says what changed and what to
do after deploying it (re-embed the documentation, set `DOCS_MIN_SCORE`).

v4 keeps v3's deterministic search as the system of record and builds a retrieval, verification
and reasoning engine around it. The Pixagram index decides what happened; a language model may
only explain it, and every claim it makes is checked against the evidence before it reaches the
answer:

```
                  PIXAGRAM DATA (the chain, indexed)  ─ authoritative source of truth
                             │
                       SEARCH ENGINE  ─ v3's retrieval, verification and deterministic operators
                             │
                         EVIDENCE     ─ cards and a graph, verified against the index
                             │
                     REASONING MODEL  ─ any model, behind one interface; optional, per question
                             │
                        EXPLANATION   ─ claims verified: unsupported ones removed, contradictions rejected
```

- **Built on Cloudflare:** Workers, D1, Queues, Vectorize, Workers AI, R2, Durable Objects,
  Workflows and Rate Limiting. Models are Workers AI by default; any OpenAI-compatible endpoint
  can be configured (`SEARCH_LLM_ENDPOINTS`).
- **Embeddings:** one small Hugging Face Space (`hf/app.py`) serves SigLIP 2 NaFlex image and
  text embeddings, as in v3.
- **Sources of truth:** the Pixa chain (artworks, blog posts and their history) and the public
  repository [pixagram-blockchain/information](https://github.com/pixagram-blockchain/information)
  (questions about the platform).
- **Its own stack:** Worker `pixagram-search-v4`, its own D1, KV, Vectorize indexes, queues and
  Space. The v3 stack (used by the Pixagram UI), production and v2 are not touched; the scripts
  refuse their resource names.

**What v4 adds, how the specification maps onto the code, and what was measured:
[README-V4.md](README-V4.md).** v3's own notes and evaluation: [README-V3.md](README-V3.md).

```
Pixa chain (api.pixagram.com)
   │  block tail (Durable Object alarm)   +   backfill (Workflow): accounts → posts → edit history
   ▼
D1: posts, artworks (+ features), post_versions, colours, pHash bands, concepts, FTS5, vocabulary, jobs,
    logs (queries, ranked results, feedback, help questions, v4: answers with traces and votes)
   │  Queue: one message per (re)ingested post; a 10-minute sweeper re-drives missing/stale stages
   ▼
consumer: decode WebP/PNG → stats, features, pHash/dHash, colours ─────────────► D1         (stats)
          xBRZ view → HF Space (SigLIP 2 NaFlex, 576 patches) ─────────────────► Vectorize  (embed)
          VLM (Workers AI Gemma 4) caption, tags, NSFW estimate ────────────────► D1 + FTS   (describe)
          tags + title + description + AI fields → multilingual concepts ───────► D1         (concepts)
          title + caption + tags (artworks), title + body (blogs) → SigLIP text ► Vectorize  (text)
   ▲
API Worker (Hono, src/api/):
   /query    the search box: rules route the text to search, ask or help (search/router.ts)
   /suggest  what the box proposes while typing, and its placeholder examples (D1 only, cached)
   /search   v3: full text ∪ concepts ∪ colours/tones ∪ image kNN ∪ text kNN → feature ranker
             v4: POST too; rerank=1 blends a cross-encoder into the top of the ranking
   /ask      v4: plan → route (class, complexity) → mode → decompose → hybrid retrieval → operators
             → evidence cards + graph → verification → [reasoning model] → claim verification
             v4.8: → digest (facts, overview, caveats, follow-ups, searches) → the model's body,
             reasoning trail and follow-ups verified sentence by sentence → answer_full / answer_markdown
             mode=v3: v3's /ask, unchanged; defer=1: the model's part from /ask/elaboration/:id
   /help     documentation sections → the help model answers from them only → every sentence checked
             v4.8: at length, with follow-up questions the documentation answers and related sections
   /similar, /duplicates, /search-by-image, /history, /feedback, /ask/feedback, /webhooks/github, /admin/*
   ▲
GitHub pixagram-blockchain/information (every 10 min, or at once through the push webhook)
```

## What is on chain (verified Sept–Oct 2026)

| | |
|---|---|
| artwork vs blog | `json_metadata.format` = `"image"` / `"markdown"`. Otherwise a body that *is* an image means an artwork |
| image payload | the whole body is `data:image/webp;base64,…` (lossless VP8L today; PNG handled) |
| metadata | `tags[]`, `description`, `nsfw`, `license` (PIXA_LICENSE incl. `visitorRights["ai-training"]`, `royaltyPercentage`), `app: "pixagram/x.y.z"` |
| deletion | the body is edited to the literal `deleted`. `delete_comment` is honoured too |
| history | `account_history_api` with `operation_filter_low = 2` (comment operations). Each call scans ≤ 2000 operations and answers `set start=N` to continue. It asserts `start >= limit - 1` |
| timing | a post's `created` is the head block time when its operation was applied: the operation's block is ~3 s later |
| sizes | typically 200–430 px on the long side; 2–370 KB |
| APIs | `condenser_api.get_content`, `bridge.get_account_posts` (20/page), `block_api.get_block_range`, `condenser_api.lookup_accounts`, `account_history_api.get_account_history` |
| marketplace | no `custom_json` ops yet; `src/chain/market.ts` is a documented hook |

## Layout

```
migrations/0001…0004          v1–v3 schema (unchanged)
migrations/0005_v4.sql        ask_log (every /ask answer: route, mode, model, grounding, tokens, cost, versions, sampled trace),
                              answer_feedback (votes on answers), help_log += mode, model, grounding
migrations/0006_paph.sql      copy detection: paph_matches (verdicts, a = earlier post), paph_progress (shards a
                              partial check finished), artworks.paph_hash / paph_engine
src/paph/                     copy detection (PAPH-X): engine.ts (module, profiles, identity, hashing budget),
                              budget.ts (per-shard bounds), shard-store.ts (a shard on SQLite: wires, keys, PAPH-SI,
                              XRank), shard-do.ts (PaphShard Durable Object), shards.ts (range sharding, parallel
                              checks), copies.ts (the paph stage, D1 verdicts, listings, uploads cached a day, reports)
src/api/copies.ts             /copies/:id, /copies-by-image, /copies/:a/report/:b, /paph/:id
src/index.ts                  Worker entry: fetch / queue / scheduled (sweeper, watchdog, nightly refresh and pruning)
src/api/                      Hono routes: index.ts (CORS, errors, rate limits), search.ts, ask.ts, help.ts, admin.ts,
                              common.ts (bodies, budgets, admin check, edge cache), image.ts (uploaded images)
src/llm/                      the model layer: provider.ts (one completion API), model.ts (what each model takes and costs),
                              adapters/ (Workers AI, OpenAI-compatible, Moondream), router.ts (model per role and band),
                              prompts.ts (policy + task + evidence + schema, versioned), reasoning.ts (ReasoningModel)
src/search/query-router.ts    v4: class, complexity, band, mode
src/search/query-planner.ts   v4: decomposition into deterministic steps (rules; the planner model when they cannot)
src/search/retrieval.ts       v4: v3's candidate collection with provenance, depth per mode, the history leg
src/search/operators/         v3's operators (first, latest, count, top, compare, search) + v4's (history, comparisons,
                              sequence, duration, aggregate, group, titles, identity)
src/search/executor.ts        v4: runs a program through the operators, passing answers between steps
src/search/evidence.ts        v4: evidence cards (E artworks, R results, D documentation, C conflicts, I1 an uploaded image), graph
src/search/verifier.ts        v4: evidence verification against the index, contradictions
src/search/claims.ts          v4: claim extraction and verification (accounts, dates, numbers, titles, paths, superlatives)
src/search/confidence.ts      v4: confidence from the evidence (weights configurable)
src/search/reranker.ts        v4: cross-encoder (Workers AI bge-reranker-base, or a TEI-style endpoint), cached
src/search/image-question.ts  v4: questions about an uploaded image: exact / perceptual / historical identity, similarity
src/search/ask.ts             v4: the /ask orchestrator; ask-v3.ts: v3's /ask (mode=v3)
src/search/answer-text.ts     every deterministic sentence in English, French, German, Spanish, Italian
src/help/                     /help: retrieve.ts, answer.ts (modes, sentence checks), citations.ts (link and citation safety)
src/evaluation/               retrieval.ts, answer.ts (scoring, reliability), models.ts (latency, cost), benchmark.ts
                              (frozen contexts, several models); datasets/questions.jsonl (1,608 questions)
src/search/{planner,lexicon,sql,service,ranker,spell,background,vectors,visual,router,suggest,feedback}.ts   v3 search (kept)
src/chain/, src/db/, src/concepts/, src/enrich/, src/docs/   v3: chain, storage, concepts, enrichment, documentation sync
hf/                           the Space (unchanged from v3)
scripts/deploy.sh, teardown.sh, lib.sh, create-resources.sh, admin.sh
scripts/eval.py               v3's live evaluation (search, two stacks side by side, /ask)
scripts/eval_v4.py            v4's question set against a deployed stack
scripts/benchmark.py          reasoning models on frozen contexts (spec §31)
scripts/train-ranker.py       ranker weights from click data
eval/                         judged queries, v3 /ask questions; offline/ harness (snapshot index, embed and rerank servers,
                              run.test.ts, rerank.test.ts, v4.test.ts, v3-fixture.test.ts); v4/ question generator, checker,
                              and three sets of reworded questions (paraphrases*.jsonl: probes of phrasings the rules were not written from)
test/fixtures/v3-answers.json v3's own answers on the test gallery (scripts/v3_fixture.sh), for the parity test
test/                         vitest: v3's tests (unit, integration, regressions) and v4's (v4-*.test.ts)
.github/workflows/ci.yml      typecheck, tests, dry-run bundle, question-set check on every push
```

## Setup

Prerequisites: Node 22.13+ (the tests use `node:sqlite`), Python 3.10+ with `huggingface_hub`, a
Cloudflare account on Workers Paid, a Hugging Face account with a write token.

```bash
npm ci
pip install -U huggingface_hub && hf auth login
npx wrangler login
scripts/deploy.sh          # everything: Space, resources, schema, Worker, secrets, backfill, tail, evaluation
```

`deploy.sh` creates the v4 stack (names in `wrangler.jsonc`: Worker, D1 `pixagram-search-v4`, KV
`pixagram-search-v4-cache`, Vectorize `pixagram-art-v4`, `pixagram-text-v4`, `pixagram-docs-v4`,
queues `pixagram-enrich-v4(-dlq)`, Workflow `pixagram-backfill-v4`, Space
`primerz/pixagram-search-v4`), applies the migrations, deploys, sets `ADMIN_TOKEN`, `HF_TOKEN`
and `GITHUB_WEBHOOK_SECRET` (kept in `~/.pixagram-search-v4.json`), backfills the chain, starts the
tail, syncs the documentation, waits for enrichment, then compares v3 and v4 on the judged queries
and runs the v4 question set in the deterministic mode. It is safe to re-run. `scripts/teardown.sh`
removes the stack. The R2 bucket `pixagram-art` is shared (content-addressed, written only when
missing, never deleted).

### Tests and evaluation

```bash
npm test                   # 690 tests: v3's (unit, integration, regressions), v4's and copy detection's
npm run typecheck          # src, tests and eval
python3 -I eval/v4/check.py src/evaluation/datasets/questions.jsonl
scripts/v3_fixture.sh      # re-freeze v3's own answers (test/fixtures/v3-answers.json) after changing the test gallery
```

`test/v4-v3-parity.test.ts` checks that `mode=v3` and `/search` give, field for field, the
answers of v3's own code (commit 16061e6, frozen by `scripts/v3_fixture.sh`) on the test gallery.

Against a deployed stack:

```bash
python3 scripts/eval.py --compare https://pixagram-search-v3.p1x4.workers.dev https://pixagram-search-v4.<you>.workers.dev
python3 scripts/eval_v4.py https://pixagram-search-v4.<you>.workers.dev --modes fast,auto --admin-token "$ADMIN_TOKEN"
ADMIN_TOKEN=… python3 scripts/benchmark.py https://pixagram-search-v4.<you>.workers.dev --reasoning medium --limit 60
```

Offline, with the real code, the real models and a snapshot of the chain:

```bash
python3 eval/offline/snapshot.py --out eval/snapshot                   # chain_posts.json + chain_history.json
pip install torch transformers pillow
MODEL_ID=google/siglip2-base-patch16-naflex MAX_NUM_PATCHES=576 python3 eval/offline/embed_server.py --port 7861 &
python3 eval/offline/rerank_server.py --port 7862 &                    # BAAI/bge-reranker-base
SNAPSHOT_DIR=eval/snapshot STATE=/tmp/v4.sqlite npm run eval:offline   # v3 search and /ask (judged queries)
SNAPSHOT_DIR=eval/snapshot STATE=/tmp/v4.sqlite npm run eval:rerank    # the cross-encoder, blends and depths
SNAPSHOT_DIR=eval/snapshot STATE=/tmp/v4.sqlite MODES=auto,fast,v3 npm run eval:v4   # the v4 question set
DATASET=eval/v4/paraphrases-2.jsonl SNAPSHOT_DIR=eval/snapshot STATE=/tmp/v4.sqlite npm run eval:v4   # reworded questions
python3 -I eval/v4/generate.py --snapshot eval/snapshot                # regenerate the question set from a snapshot
```

### Local development

```bash
npm run db:migrate:local
echo 'ADMIN_TOKEN=devtoken' > .dev.vars
npm run dev                          # http://127.0.0.1:8787
BASE=http://127.0.0.1:8787 ADMIN_TOKEN=devtoken scripts/admin.sh ingest matus swan-1790532192509
curl 'http://127.0.0.1:8787/ask?q=who+posted+the+first+cat'
```

Vectorize and Workers AI have no local emulation; add `"remote": true` to those bindings to use
the real ones. The Space runs locally with `cd hf && pip install -r requirements.txt && python app.py`.

## Configuration (`wrangler.jsonc` → `vars`)

v3's variables keep their meaning:

| var | default | meaning |
|---|---|---|
| `RPC_URL`, `RPC_FALLBACK_URLS` | api.pixagram.com, … | Hive-compatible JSON-RPC nodes |
| `APP_PREFIXES` | `pixagram` | index posts whose `json_metadata.app` starts with one of these; empty = all |
| `HF_EMBED_URL`, `HF_TOKEN` (secret) | the v4 Space | `https://<owner>-<space>.hf.space/embed`; empty disables semantic search |
| `EMBED_MODEL`, `EMBED_DIM`, `EMBED_PATCHES`, `EMBED_VIEWS` | SigLIP 2 NaFlex, 768, 576, `xbrz` | as v3 (checked against the Space on every call) |
| `TEXT_VECTORS` | `on` | SigLIP text vectors of artworks and blogs in `VEC_TEXT` |
| `VLM_BACKEND`, `VLM_MODEL`, `VLM_TARGET` | `gemma`, `@cf/zai-org/glm-5.3-flash`, 512 | `gemma` (Gemma 4) · `chat` (any vision chat model the model table knows, named by `VLM_MODEL`; v4.8.1) · `scout` · `moondream` · `caption` · `off`; also describes uploaded images in /ask's deeper modes |
| `AI_TRAINING_FALSE_BLOCKS` | (empty) | stages skipped when the license says `ai-training: false` |
| `SCALER`, `UPSCALE_TARGET`, `STORE_IN_R2` | `xbrz`, 800, `true` | previews and originals in R2 |
| `HISTORY_BACKFILL`, `HISTORY_MAX_CALLS` | `true`, 300 | walk each author's comment history in the backfill |
| `PLANNER_BACKEND` | `auto` | `rules`, `llm`, or `auto` (the planner model only when the rules are unsure) |
| `RANK_LOG_SAMPLE` | 0.25 | share of searches whose shown results are logged with features |
| `DOCS_REPO`, `DOCS_BRANCH`, `DOCS_PATHS` | `pixagram-blockchain/information`, `main`, (all) | the repository behind `/help`; `off` disables help |
| `DOCS_EMBED_MODEL`, `DOCS_EMBED_DIM`, `DOCS_MIN_SCORE`, `DOCS_QUERY_INSTRUCTION` | `@cf/qwen/qwen3-embedding-0.6b`, 1024, 0.4, (built in) | documentation vectors and relevance (v4.8.1: Qwen3-Embedding, questions embedded with the retrieval instruction; README-V4 "Models, October 2026" on re-embedding and the threshold) |
| `SUGGEST_POPULAR` | `off` | popular searches in `/suggest` (see README-V3) |
| `MARKET_CUSTOM_JSON_IDS`, `TAIL_BLOCKS_PER_TICK`, `TAIL_IDLE_SECONDS` | —, 100, 3 | marketplace hook, tail pacing |
| `ADMIN_TOKEN`, `GITHUB_WEBHOOK_SECRET` (secrets) | — | `/admin/*` bearer token; the documentation push webhook |

v4's (spec §57-58; every model is configuration):

| var | deployed | meaning |
|---|---|---|
| `SEARCH_PLANNER_MODEL` | `@cf/google/gemma-4-26b-a4b-it` | plans what the rules cannot, and decomposes multi-step questions (falls back to v3's `PLANNER_MODEL`) |
| `SEARCH_REASONING_MODEL` | `@cf/zai-org/glm-5.3-flash` | reasons over the evidence (v4.8.1; v4: gpt-oss-120b); `SEARCH_REASONING_MODEL_SIMPLE`, `_NORMAL`, `_COMPLEX`, `_DEEP` per complexity band — deployed: `_DEEP` = `@cf/zai-org/glm-5.3` |
| `SEARCH_HELP_MODEL` | `@cf/zai-org/glm-5.3-flash` | writes `/help` answers (v4.8.1; v4: Nemotron 3 Super; falls back to v3's `HELP_MODEL`) |
| `SEARCH_RERANKER_MODEL`, `SEARCH_RERANK_URL` | `@cf/baai/bge-reranker-base` | the cross-encoder; `http` with a TEI-style `SEARCH_RERANK_URL`; `off` |
| `SEARCH_LLM_ENDPOINTS` | — | JSON `{"name": {"base_url", "api_key_env", "json", "reasoning", "price"}}`: OpenAI-compatible models, named `name:model` |
| `SEARCH_MODEL_OVERRIDES` | — | JSON facts about models the built-in table does not know (style, reasoning control, context, price) |
| `SEARCH_PUBLIC_MODELS` | (none) | models a public `/ask` may name; any other choice needs the admin token |
| `SEARCH_REASONING` | `auto` | by mode (fast none, balanced low, deep medium, expert high), or one level for every mode that reasons |
| `SEARCH_MAX_OUTPUT_TOKENS` | 9000 | the visible answer's bound, apart from the reasoning budget (brief answers: 2000/4000/6000 by mode; rich: 3000/5000/8000) |
| `SEARCH_REASONING_TOKENS` | `low:2048,medium:6144,high:16384` | reasoning tokens allowed per level, on top of the answer (raised in v4.8.1: GLM reasons in every mode) |
| `SEARCH_DEFAULT_MODE`, `SEARCH_MAX_MODE`, `SEARCH_QUERY_MAX_MODE` | `auto`, `deep`, `deep` | the default; the deepest a public caller may ask for; the deepest the search box picks (v4 and brief answers: balanced) |
| `SEARCH_AUTO_COMPLEXITY`, `SEARCH_COMPLEXITY_BANDS` | `true`, `0.2,0.4,0.65,0.85` | complexity routing and its band bounds |
| `SEARCH_FTS_K`, `SEARCH_CONCEPT_K`, `SEARCH_SEMANTIC_K`, `SEARCH_VISUAL_K`, `SEARCH_HISTORY_K` | 2000, 2000, 100, 64, 50 | retrieval pools of the deeper modes (never below v3's) |
| `SEARCH_RERANK`, `SEARCH_RERANK_K`, `SEARCH_FINAL_K`, `SEARCH_RERANK_BLEND` | `true`, 50, 20, 0.2 | the cross-encoder on /ask's evidence (balanced and deeper); how many it reads, how many cards the model gets; /search's blend weight |
| `SEARCH_VERIFY_CLAIMS` | `true` | claim verification of every model answer |
| `SEARCH_CONFIDENCE_WEIGHTS` | `retrieval:0.25,reranking:0.2,agreement:0.2,verification:0.2,model:0.15` | confidence components |
| `SEARCH_EXPANSION` | `rules` | the concepts' aliases in every language (v3); `off` |
| `SEARCH_TRACE`, `SEARCH_TRACE_SAMPLE` | `false`, 0.1 | traces to callers that ask (the admin always may); share of answers whose trace is kept in `ask_log` |
| `SEARCH_ANSWER_CACHE_TTL`, `SEARCH_LLM_TIMEOUT_MS` | 86400, 120000 | cache of model answers (keyed on the exact evidence, model, prompt version, task, length); model timeout (long-form replies take longer than v4's three sentences; GLM-5.3 at high effort longer still) |
| `HELP_MODE`, `HELP_RERANK` | `auto`, `off` | /help's mode (auto: fast up to twelve words, else balanced; rich: never below balanced); the cross-encoder on its excerpts |

v4.8's (the long-form answers; "Rich answers" below):

| var | deployed | meaning |
|---|---|---|
| `SEARCH_ANSWER_STYLE` | `rich` | `rich`: every answer carries the digest and, in balanced and above, the model's body, reasoning trail, caveats and follow-ups; `brief`: v4's answers (a request may ask for either: `style=`) |
| `SEARCH_ANSWER_WORDS` | `balanced:250,deep:450,expert:800` | the body's target length per mode, in words (`length=short` halves it, `long` nearly doubles it) |
| `SEARCH_RICH_MIN_MODE` | `balanced` | rich answers never run below it by themselves (fast has no model): every question from the box reaches the model, at low reasoning |
| `SEARCH_FOLLOW_UPS` | 6 | follow-up questions shown with an answer (the model's that pass, then the templates') |
| `SEARCH_ELABORATION_TTL` | 1800 | seconds a deferred answer's frozen context waits for `GET /ask/elaboration/:id`, and its result is kept |
| `HELP_STYLE`, `HELP_ANSWER_WORDS` | `rich`, `fast:120,balanced:200,deep:350,expert:600` | /help's style and the answer's target length per mode |

Copy detection (`@pixagram/paph-x` 1.2.0: wire 4, CAL-007, profiles X3 and SI4; `src/paph`; README-V4 "Copy detection", and "Moving a store to 1.2.0" for a store indexed before):

| var | default | meaning |
|---|---|---|
| `PAPH` (binding) | `PaphShard` | the shards; without it the `paph` stage is skipped and `/copies*` answer 503 |
| `PAPH_ENABLED` | `true` | `false` turns the stage and the routes off, keeping the binding |
| `PAPH_SHARD_SIZE` | 100000 | post ids per shard; part of each shard's name — choose it once |
| `PAPH_BUDGET_QUERY`, `PAPH_BUDGET_STAGE` | (presets) | per-shard bounds over `src/paph/budget.ts`'s presets, e.g. `verify:128,deadline_ms:700` (`per_family`, `df_cap`, `key_postings`, `si_postings`, `si_reach`, `si_top`, `si_min_score`, `verify`, and `deadline_ms`: from a check's arrival in the shard, waiting included) |
| `PAPH_QUERY_MS` | 900 | an interactive check's wall-clock budget, from the request's start (with `semantic=1`, from the embedding) |
| `PAPH_CACHE_TTL` | 86400 | seconds an upload's answer and a pair report are kept (an answer with a shard missing: 600) |
| `PAPH_MIN_VERDICT` | `suspected` | verdicts the stage stores (listings show `copy` and up unless `min=` asks) |
| `PAPH_POLICY` | `safe` | XRank's policy: `safe` (every Copy is comparator 42's or certified), `exact`, `fast` |
| `PAPH_MAX_PIXELS` | 589824 (768²) | hashing budget; larger images are divided (exact blow-ups) or box-filtered first |
| `PAPH_PHASH_DISTANCE`, `PAPH_CHANNEL_K` | 10, 24 | pHash and embedding neighbours checked beside the index's own nominations |

Rate limits (`ratelimits`, per client IP and location; namespaces 41xx, apart from v3's 31xx; the
admin token is exempt): 120 a minute on search-like routes (`/query`, `/search`, `/feedback`,
`/ask/feedback`, `/similar`, `/duplicates`, `/posts`, `/history`, `/concepts`, `/copies`, `/paph`),
600 on `/suggest`, 20 answers a minute (`/ask`, `/help`, `/search-by-image`, `/copies-by-image`,
pair reports, `/query` when it answers, `/search` with `rerank=1`).

## API

All responses are JSON, and CORS is open. v3's routes keep their parameters and fields; v4 adds
fields beside them. Search items have v3's shape (README-V3.md); blog posts have `artwork: null`.

### `GET /query`: the search box

As v3 (search, ask or help, chosen by rules with no model call). v4: a question routed to ask is
answered in the mode it needs, up to `SEARCH_QUERY_MAX_MODE` (`mode=` forces one). `answer` is an
/ask v4 response; `results` carries a search when the answer comes back empty. v4.8: `style=`,
`text=`, `length=` and `defer=1` pass through to the answer (see "Rich answers"); every search
answered in the rich style carries an `overview` (a text summary of the page, below).

### `GET /suggest`

As v3: completions, questions, titles, documentation sections while typing (D1 and the concept
vocabulary only), and placeholder examples without `q`. See README-V3.md. v4.8: `after=<query_id>`
puts the follow-up questions and related searches of that answer first (`kind: "followup"`, with
the `route` the box should send them to), with nothing typed and while typing, for half an hour
after the answer: the box continues the conversation.

### `GET /search`, `POST /search`

v3's parameters (`q`, `type`, `author`, `tag`, `color`, `has_color`, `background`, `orientation`,
`monochrome`, `transparent`, `concept`, `size`, `min_colors` …, `from`, `to`, `nsfw`, `listed`,
`ai_training`, `sort`, `limit`, `cursor`, `facets`, `semantic`, `expand`, `rank`, `explain`), as a
query string or a JSON body (`{"query": "red pixel dragon", "limit": 20, "rerank": true}`). v4:

- `rerank=1`: the cross-encoder reads the top `SEARCH_RERANK_K` of a relevance ranking (title,
  tags, description, AI caption, author: text only), and its score is blended with the feature
  ranker's (`SEARCH_RERANK_BLEND`), which carries the image similarity. Off by default: the
  default ranking is v3's (see README-V4 for the evaluation).
- The response adds `retrieval` (candidates per retrieval family: `fts`, `concept`, `visual`,
  `semantic`, `color`, …), `reranked`, `reranker`, and `score.rerank` per item when it ran.
- v4.8: `overview=1` (and `lang=`) adds a text overview of the page: `{text, lines, stats}` (see
  "Rich answers"). `/query` adds it to every search by itself.

### `/ask`

`POST /ask {"question": "…"}` or `GET /ask?q=…`, with optionally:

| field | |
|---|---|
| `mode` | `auto` (default: by complexity) · `fast` · `balanced` · `deep` · `expert` (admin) · `v3` (v3's /ask, unchanged) |
| `reasoning` | `auto` · `none` · `low` · `medium` · `high` (admin): how much the model may think, apart from the answer's length |
| `model` | a reasoning model for this request (`SEARCH_PUBLIC_MODELS`, or the admin token) |
| `max_output_tokens` | bound of the visible answer (capped by `SEARCH_MAX_OUTPUT_TOKENS`) |
| `trace`, `graph` | the internal trace (admin, or `SEARCH_TRACE=true`); the evidence graph (always in deep and expert) |
| `image` | base64 or a data URI (or a multipart `image` field with `question`): a question about that image |
| `nsfw`, `type`, `limit`, `threshold`, `planner=rules` | as v3 |
| `style` | v4.8: `rich` (default, `SEARCH_ANSWER_STYLE`) · `brief` (v4's answer, field for field) |
| `text` | v4.8: `short` (default: `answer_text` is the direct answer, as v4) · `full` (`answer_text` carries the whole long-form answer) |
| `length` | v4.8: `short` · `medium` (default) · `long`: the body's length against the mode's target |
| `defer` | v4.8: `1`: answer now with the digest, the model's part from `GET /ask/elaboration/:query_id` |

```jsonc
// "Who posted the first cat artwork and was it later reposted?"
{ "question": "…", "status": "answered",   // answered | no_match | insufficient_evidence | conflict | clarify | not_found
  "answer": true, "answer_type": "boolean",
  "answer_text": "The first cat artwork was posted by @alice on 2026-09-01: “Black cat”. Yes: the same image was posted again on 2026-09-25 in /@carol/found-cat.",
  "result_text": "…",                       // the index's own answer: answer_text starts with it
  "explanation": "…",                       // a model's explanation after it, when one ran, its claims held and it states the result's values
  "confidence": 0.95, "confidence_parts": { "retrieval": 1, "agreement": 1, "verification": 1, "plan": 1, "operator": 0.95 },
  "class": "MULTI_HOP", "complexity": 0.305, "band": "simple", "mode": "fast", "reasoning": "none", "model": null,
  "subqueries": [{ "id": "s1", "type": "temporal", "operation": "find_first", "step": "q1" }, { "type": "history", … }],
  "steps": [{ "id": "q1", "op": "find_first", "status": "ok", "answer": "alice", "text": "…", "evidence": ["E1"] }, { "id": "q2", "op": "history", … }],
  "cards": [{ "evidence_id": "R1", "type": "result", … }, { "evidence_id": "E1", "type": "artwork", "path": "/@alice/black-cat", "created_at": "…", "first_seen_at": "…", … }],
  "claims": [], "contradictions": [], "grounding": null,          // a model's claims, verified, when one answered
  "versions": { "retrieval": "4.0.0", "index": "…", "ranker": "…", "reranker": null, "planner": "rules", "reasoning_model": null, "prompt": null },
  "usage": null,                            // tokens, cost_usd, model_ms when a model answered
  "timings": { "plan": 1, "retrieve": 12, "evidence": 3, "total": 19 },
  "query_id": "…",                          // for POST /ask/feedback {query_id, rating: 1|-1, reason?}
  /* and v3's fields: intent, output, plan, evidence, alternatives, counts, items, verified, notes, took_ms */ }
```

- **Deterministic first.** Questions are decomposed into steps the operators answer exactly
  (first, latest, count, top, compare, search, history, comparisons, sequence, duration, totals,
  groups, titles, existence, author checks, premises). "How many artworks did @alice publish?"
  never calls a model. A model runs only when asked for (`mode`/`reasoning`), when a written
  synthesis is needed ("why …", "what kind of art does @a make"), or when the evidence holds no
  operator's answer. When the index answers, its answer comes first (`result_text`) and the
  model's text is only shown after it, as `explanation`, when every claim in it holds and it
  states the result's own values (the same @author first, date, number, post, "Yes"/"No"); the
  confidence stays the operator's. Nothing the index did not find, or found on evidence that
  failed verification, is answered by a model (an uploaded image the index does not hold is
  still described, after the index's own answer).
- **Questions back.** "Who posted this?" with nothing named: `clarify`. A title several posts
  share: `clarify` with the candidates. "Why did @b post the first cat?" when @a did: the index's
  answer. "Why …?" with no model available: the facts, and that they do not explain it.
- **Images.** With `image`: `class: VISUAL`, an `I1` card, `image: {task, identity, first_seen,
  matches}`. `exact_identity` (the same bytes, in any post or version, deleted ones included),
  `perceptual_identity` (pHash ≤ 4 and the same colours: a rescaled or re-encoded copy),
  `historical_identity` (where it first appeared), `visual_similarity` (never identity).
- **Languages.** Answers in English, French, German, Spanish and Italian (`mode=v3`: English,
  French, German, as v3).

### Rich answers (v4.8)

v4 answered in one sentence and let a model add one to three more, dropped whole when a single
claim failed. v4.8 keeps v4's rule — the index decides, the model explains, nothing unverified
reaches the reader — and makes the answer a real one: longer, structured, with the reasoning
shown, and with somewhere to go next. Every v4 field keeps its meaning; these are added:

| field | |
|---|---|
| `style` | `rich` or `brief` |
| `answer_text` | the direct answer: `result_text` then the model's verified `explanation` (its citations removed) — as v4, unless `text=full` |
| `answer_short` | always the direct answer |
| `answer_full` | the whole answer as plain text: the direct answer, the body, "From the index" (overview and facts), "How this was worked out", "Keep in mind", "You may also ask", "Related searches" — section titles in the question's language (EN, FR, DE, ES, IT), no Markdown marks, no citations |
| `answer_markdown` | the same as Markdown, citations kept (`[E12]`, `[R1]`) for a UI that links them to `cards` |
| `sections` | the parts apart: `lead`, `body` (the model's, verified, Markdown), `overview[]`, `facts[]`, `thinking[]`, `caveats[]`, `follow_ups[]`, `searches[]` |
| `thinking` | the model's reasoning trail: `[{n, text, evidence, status}]`, each step verified like a claim and renumbered; the model's private chain of thought is never shown (spec §21) — this is a written account the reader can check |
| `suggestions` | `follow_ups: [{text, route: ask\|search\|help, source: model\|rules}]` and `searches: [{text, route: search, source}]` |
| `digest` | `stats` (posts, authors with counts, from, to, top_voted, tags, colours, `exact`) and `about` (the evidence ids the facts describe) |
| `length` | `{words, target}`: the body's words and the target the model was given |
| `elaboration` | `{status: inline}` the model's part is in this answer · `{status: pending, url}` fetch it (`defer=1`) · `{status: none, reason}` there is none (fast mode, brief style, nothing to explain) |

**What is deterministic and what is the model's.** The digest — the facts about the deciding
posts (title, author, date, tags, votes, payout, where the image first appeared, edits, deletion,
later posts of the same image, the AI caption labelled as a model's description), the overview of
the set (how many posts by whom, the range of dates, the most voted, the frequent tags), the
caveats (lower bounds, ties, inferred histories, deleted or hidden posts) and the template
follow-ups — is written from computed values only, in the question's language, with no model
call: it is there in fast mode, when the model is unavailable, and while a deferred elaboration
runs. An exact count over the filters ("how many artworks did @alice post?") kept no rows in v4;
the rich answer reads the set's newest posts as evidence cards (and its oldest and most voted,
exactly), so the digest and the model have something to say.

The model's part (balanced and above; `SEARCH_RICH_MIN_MODE` raises what auto picks so the box's
questions reach it): the `compose` task asks for the direct answer, a body of about
`SEARCH_ANSWER_WORDS` words in Markdown with every factual sentence citing its cards, a reasoning
trail of 3–8 steps, caveats, 3–6 follow-up questions and 2–4 searches. Then:

- the direct answer is judged as in v4 (claims verified; it must state the index's own value);
- the body is verified **sentence by sentence** (`search/compose.ts`): a sentence whose account,
  date, number, title or path the cards do not state is removed, one that contradicts a result
  card is removed, one with an address or an accusation the evidence does not make is removed,
  a cited interpretation stays (qualified when few of its words are the card's); headings are
  checked the same way; the rest is kept — a long answer is never thrown away whole for one bad
  sentence, as v4's explanation was;
- the reasoning trail and the caveats are checked step by step;
- a follow-up question is kept only when it names accounts, titles, numbers and dates the
  evidence holds, carries no address and no accusation, is not the question itself, and leads
  somewhere: an /ask the rules plan with confidence whose subject is made of the evidence's
  words, a search of such words, or a help question the documentation covers; a search is kept
  when every word of it is the evidence's; the templates fill up to `SEARCH_FOLLOW_UPS`;
- a lead that contradicts the index's result drops the whole reply (the digest stays); a reply
  cut off at the token limit is salvaged up to its last whole sentence; a model that judges the
  evidence insufficient leaves the index's answer as it is.

**Deferred elaboration (`defer=1`).** The index's answer with the digest comes back in
milliseconds with `elaboration: {status: "pending", url}`; the frozen context (the cards, the
notes, the result) waits in KV. `GET /ask/elaboration/:query_id` runs the model the first time
it is called (the caller waits there instead of on `/ask`) and returns `{status: "ready",
answer: {…}}` — the fields the model's part changed (`answer_text`, `explanation`, `answer_full`,
`answer_markdown`, `sections`, `thinking`, `suggestions`, `claims`, `grounding`, `usage`,
`model`, `confidence`, `versions`, `notes` — the first answer's included), to merge over the
first answer; `{status: "pending"}` while another call runs it; `{status: "failed", retry}` when
the model did not answer (a transient failure is tried again by the next call, three times at
most; a reply the engine cannot use is final: the first answer stands); 404 `unknown` after
`SEARCH_ELABORATION_TTL`. One answer costs at most three model calls, however often it is
polled. The search box: `GET /query?q=…&defer=1`, show `answer`, fetch `answer.elaboration.url`,
merge; `GET /suggest?after=<query_id>` for the dropdown meanwhile.

**Searches.** A search has an `overview` too (`/query` always, `/search?overview=1`): "“red
dragon”: 37 artworks match (showing 20). Most by @nova (12) and @pix (8). The most voted is
“Fire” by @nova (120 votes, 2026-05-02). Posted between … Frequent tags: dragon, fire, red." —
from the page and, when the request had `facets=1`, from the facet counts (which cover every
match); when the page is not the whole set, its superlatives are said of the page ("Of the 20
shown: …") and the tags only when the facets count them.

**For the UI.** Show `answer_markdown` (or `answer_full`) in place of `answer_text`; render
`thinking` as a collapsible "How this was worked out"; `suggestions.follow_ups` as chips that
send their `text` to the box with their `route`; `overview.text` above a result grid; with
`defer=1`, show the first answer at once and merge the elaboration when it arrives. Costs: one
reasoning call per uncached rich answer in balanced and above (GLM-5.3-Flash: about
$0.001–0.004 for a 250–450-word body with its trail and the reasoning behind it, cached on
identical evidence for a day; the deep band's GLM-5.3 about ten times that); the digest, the
overview and the suggestions make no model call.

### `/help`

`POST /help {"question": "…", "mode"?, "reasoning"?, "style"?, "length"?}` or `GET /help?q=…`: as v3 (answers from the
documentation only, checked citations, safe text, cached per commit, gaps log), plus:

- **Every sentence is checked** against the excerpts it cites: a number, date, account, path or
  address the documentation does not state removes the sentence; nothing left → the excerpts
  themselves (`status: excerpts`). A translated quoted label stays, qualified.
- `mode`: `fast` (5 excerpts, no reasoning: v3), `balanced` (6), `deep` (8, low reasoning),
  `expert` (10, medium reasoning, admin; a sentence that cites no excerpt is removed); `auto`.
- Response fields added: `mode`, `reasoning`, `claims` (each sentence, its status and sources),
  `grounding` (`egs`, `citation_accuracy`, `removed`), `versions`, `usage`.
- v4.8, `style=rich` (default, `HELP_STYLE`): the model writes a complete answer — a direct
  sentence, then the details, the steps as a numbered list, the limits and fees the excerpts
  state, about `HELP_ANSWER_WORDS` words by mode (`length=short|long`), every sentence still
  citing its excerpt and checked; auto never runs below balanced. Added: `follow_ups` (the
  model's questions, kept only when the documentation covers them: one lexical lookup each),
  `related` (the other sections of the pages the answer cites, with their links), `length`,
  `style`. `style=brief`: v4's five sentences.

### Copy detection

| route | |
|---|---|
| `GET /copies/:id?min=copy\|suspected\|identical&limit=24` | the stored verdicts of an artwork about the two works' current images, both directions: each item is a search item with `copy: {verdict, state, certifiable, certificate, structural: [lo, hi], geometry, inliers, mirrored, execution, relation: earlier\|later, same_author, via, engine, computed}`; `indexed` says the check completed for the current image under the current engine; `hidden` counts verdicts the filters (nsfw…) leave out or whose images changed since. D1 only. `live=1` (admin token) checks again now, in every shard |
| `POST /copies-by-image?min=…&limit=…&semantic=1&nsfw=…` | multipart `image` or JSON `{"image": base64}`: the copies of an uploaded image (not stored), sub-second; `as_of`, `cached`, `partial`, `engine`, `shards`, `stats`; kept a day for the same pixels. `semantic=1` adds the embedding's neighbours (one embedding call) |
| `GET /copies/:a/report/:b?wires=1` | PAPH-X's full report of a pair with comparator 42's beside it (`report.fallback`), and with `wires=1` both wires (base64) to re-run it anywhere; live posts' current images only; kept a day |
| `GET /paph/:id` | where an artwork stands: its shard, indexed, check complete under the current engine, image current, derived by the current engine (404 for an id D1 does not know) |

Admin: `GET /admin/paph` (shards: works, size, derivation, postings, queued checks; verdicts by kind
and engine; the stage's progress under the current identity; identity and budgets),
`GET /admin/paph/alerts?days=30` (cross-author copies, earlier → later), `POST
/admin/paph/rederive?limit=200` (after a release; repeat until `remaining` is 0), `GET|POST
/admin/paph/stale[?purge=1][&force=1]` (verdicts of another engine or policy; the purge is refused
while artworks still wait for their re-check), `POST /admin/paph/gc?limit=500` (indexed works of
deleted posts leave the shards, a page per shard, and `reset` counts works whose check was marked
complete although their entry shows another image or is missing — they go back to the stage; a
nightly pass goes through every shard).
`/admin/reindex` takes the `paph` stage like the others.

### Other routes

| route | |
|---|---|
| `GET /similar/:id`, `GET /duplicates/:id`, `POST /search-by-image` | as v3 |
| `GET /history/:id`, `GET /posts/:id`, `/posts/:author/:permlink`, `GET /concepts`, `/img/…`, `/vocab`, `/healthz` | as v3 |
| `POST /feedback` | search engagement events (v3) |
| `POST /ask/feedback` | `{query_id, rating: 1 \| -1, reason?: wrong \| unsupported \| incomplete \| other}`, for a recent answer |
| `GET /ask/elaboration/:query_id` | v4.8: the model's part of an answer asked with `defer=1` ("Rich answers") |
| `POST /webhooks/github` | documentation push events (v3) |

Admin routes (`Authorization: Bearer $ADMIN_TOKEN`; `scripts/admin.sh` wraps them): v3's
(`stats`, `indexer`, `backfill`, `ingest`, `reindex`, `sweep`, `background`, `vocab/rebuild`,
`jobs/failed`, `queries`, `ranker/weights`, `ltr/export`, `docs`, `docs/sync`, `docs/reembed`,
`docs/gaps`, `debug/describe/:id`, `debug/help`), and v4's:

| route | |
|---|---|
| `GET /admin/ask/log?days&status&mode&class&model` | answers with route, mode, model, grounding, tokens, cost; totals |
| `GET /admin/ask/trace/:qid` | one answer with its stored trace (classification, subqueries, candidates per leg, scores, reranking, evidence, claims, verification) and the votes on it |
| `POST /admin/ask/context` | a question's frozen context: the exact cards and notes a model would get, and the deterministic answer |
| `POST /admin/ask/reason` | one model on a frozen context, its claims verified: grounding, tokens, cost, latency; `task: "compose"` (v4.8) adds the long body's sentence checks and length (`scripts/benchmark.py --task compose`) |
| `GET /admin/models` | models per role and band, the model table (request style, reasoning control, context, prices) |
| `GET /admin/ltr/pairs` | pairwise preferences from engagement (a result acted on over those skipped above it) |
| `GET /admin/ask/export-sft` | fine-tuning examples: sampled answers whose claims all held and nobody voted down |

## How retrieval and ranking work

v3's, unchanged (README-V3.md): SQL filters (also pushed into Vectorize), candidates from FTS5
(title 6, description 3, body 1, tags 4, AI caption 2.5, AI tags 3, author 0.5), concepts in any
language, colour buckets and tones, SigLIP image and text kNN (time-sliced when filters are
restrictive); semantic cosines z-scored against a background sample; the feature ranker (exact
title, bm25, coverage, tag, author, concept, image z, text z, colour, tone, orientation) × bounded
Bayesian quality × freshness; near-duplicate diversity; a 60 s result cache. /ask collects the
same candidates with their provenance and verifies each against the subject (noisy-OR of lexical
and concept evidence with z-scored image and text similarity).

## Limits designed around

v3's (message and row sizes, D1's 100 bound parameters and 100-statement batches, Vectorize's 100
results per query and 10 metadata indexes, decoding up to 2048×2048 and 1024×1024 for uploads,
16 KB question bodies and 3 MB image bodies, subrequest budgets, the 5-minute CPU limit, the
Space's concurrency), plus: a trace kept in `ask_log` is at most 32 KB (heavy parts dropped
first); a model sees at most `SEARCH_FINAL_K` artwork cards; model calls time out after
`SEARCH_LLM_TIMEOUT_MS`; model answers are cached on the exact evidence. Copy detection: a
Durable Object holds 10 GB (a shard of 100,000 post ids needs ~3–5 GB: wires ~10–25 KB, keys ~4 KB and
~800 postings per artwork); an isolate has 128 MB, so images are hashed within 768² pixels
(~60 bytes per pixel of WebAssembly memory, which never shrinks); every per-shard step is
bounded, and a queue consumer run (15 minutes) starts no message after 12 and cuts the stage's
checks to end by then (README-V4 "Copy detection").

## Costs (order of magnitude)

- **Cloudflare:** Workers Paid $5/mo; D1, KV, R2, Queues and Vectorize usage at Pixagram's size
  costs cents.
- **Workers AI (October 2026 prices, per million tokens in / out):** descriptions (Gemma 4,
  $0.10 / $0.30) once per image; the planner (Gemma 4) only for questions the rules cannot plan;
  reasoning (GLM-5.3-Flash, $0.15 / $0.50; the deep band on GLM-5.3, $1.40 / $4.40) only where
  a model is worth its cost, with contexts of 2–6k tokens: about $0.001–0.004 per reasoned
  answer on Flash, cached on identical evidence; help (GLM-5.3-Flash) once per new question the
  documentation answers; the documentation vectors (Qwen3-Embedding-0.6B, $0.0118) once per
  chunk and per new question; the cross-encoder ($0.003 per million tokens) on /ask's evidence
  in the deeper modes and on `/search?rerank=1`.
- **HF Space:** free on CPU basic (sleeps after 48 h idle); paid CPU without sleep costs a few
  dollars a month.

## Notes

- `xbrz-js` is a GPL-3 port of Zenju's xBRZ. It runs server-side only.
- SVG artworks are indexed as posts but skipped by the image pipeline.
- The Hivemind endpoints cap page size at 20, hence the account-by-account backfill.
