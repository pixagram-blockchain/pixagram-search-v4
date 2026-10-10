# pixagram-search v4: what it adds, where the specification lives in the code, what was measured

**v4.8 (October 2026): the long-form answers.** The engine below is unchanged; what a reader gets
from it is not. See "v4.8: the long-form answers" at the end of this file, and README.md "Rich
answers" for the API.

v4 is v3's deterministic search with a retrieval, verification and reasoning engine built around
it (the "Pixagram Search v4 Specification", 63 sections). The rule it is built on (spec §61): the
Pixagram index decides what happened; a model may only explain it. Concretely:

- every question is planned into deterministic steps first; most questions never reach a model
  ("How many artworks did @alice publish?" does not, spec §26);
- when the index answers, its answer is said first (`result_text`), and a model's text is shown
  after it only as an `explanation`, only when every claim in it holds against the evidence and
  it states the result's own values (spec §27);
- where the index finds nothing, a step fails, the evidence fails verification, or the question
  does not say what it is about, no model answers in its place;
- `mode=v3` and `/search` answer field for field as v3's own code does, checked against v3's
  frozen answers on every test run.

## The path of a question

```
POST /ask {"question", "mode", "reasoning", "image"?}                         src/api/ask.ts
  │  instructions inside the question are dropped (spec §42)                   query-planner.ts withoutInstructions
  ▼
plan: v3's rules planner (+ the planner model when the rules are unsure)      planner.ts, ask-v3.ts planQuestion
  ▼
decompose into steps: find_first/latest, count, top, compare, search,         query-planner.ts decomposeRules
  author_of, history, compare_counts/metric, sequence, duration, aggregate,      (the planner model only for what
  group, exists, premise, image                                                   no rule decomposes: decompose)
  ▼
route: class (EXACT … UNKNOWN), complexity (9 weighted signals), band, mode    query-router.ts
  ▼
execute: hybrid retrieval with provenance → v3 verification → operators       executor.ts, retrieval.ts, operators/
  ▼
cards (E artworks, R results, D docs, C conflicts, I1 an image) + graph       evidence.ts
  ▼
evidence verification against the index; conflicts become C-cards             verifier.ts
  ▼
cross-encoder orders what the model will read (balanced and deeper)           reranker.ts
  ▼
reasoning model, when the mode allows and it is worth its cost                llm/reasoning.ts, llm/router.ts
  ▼
claim verification; agreement with the index's answer                         claims.ts
  ▼
confidence from the evidence, capped by the operator's                        confidence.ts
  ▼
response (v3's fields + v4's), ask_log row, sampled trace                     ask.ts
```

## Specification → code

| § | requirement | where | notes |
|---|---|---|---|
| 1, 4, 61 | v3 deterministic search stays the system of record | `search/ask.ts`, `search/ask-v3.ts` | `mode=v3` is v3's /ask; parity test against v3's own answers |
| 6–7 | query router: classes, complexity, bands | `search/query-router.ts` | EXACT, FACTUAL, SEMANTIC, VISUAL, TEMPORAL, COMPARATIVE, MULTI_HOP, EXPLANATORY, AGGREGATION, AMBIGUOUS, UNKNOWN; bands `SEARCH_COMPLEXITY_BANDS` |
| 8 | fast / balanced / deep / expert | `query-router.ts` PROFILES, `chooseMode` | expert and `reasoning=high` need the admin token; `SEARCH_MAX_MODE` caps public callers |
| 9, 11 | hybrid retrieval, candidate generation with provenance | `search/retrieval.ts` | FTS, concepts, colours, tones, image and text kNN, metadata, history; per-candidate `retrieval` sources; pools per mode (`SEARCH_*_K`) |
| 10, 49 | decomposition, multi-hop | `search/query-planner.ts`, `executor.ts` | rules in five languages; the planner model (JSON schema) only when no rule applies; it plans, the executor computes |
| 12 | feature ranker | `search/ranker.ts` | v3's, unchanged |
| 13 | cross-encoder, evaluated | `search/reranker.ts`, `eval/offline/rerank.test.ts` | bge-reranker-base; see the measurements below |
| 14–15 | evidence graph, cards | `search/evidence.ts` | stable ids, `modelView` keeps internal fields from the model |
| 16, 24 | evidence verification, contradictions | `search/verifier.ts` | invalid deciding evidence → `insufficient_evidence`; C-cards; conflict sentences in five languages |
| 17–21 | reasoning layer, model abstraction, router, selection, budget | `src/llm/` | one `complete()` over Workers AI and OpenAI-compatible endpoints; model table (request style, reasoning control, prices); model per role and band; reasoning tokens apart from the answer's (`SEARCH_REASONING_TOKENS`, `SEARCH_MAX_OUTPUT_TOKENS`) |
| 22–23 | claim extraction and verification | `search/claims.ts` | atoms (accounts with or without "@", dates in five languages and numeric formats, typed numbers, titles and the numbers in them, paths, superlatives, yes/no); unsupported claims removed, qualified ones capped (removed in expert) |
| 25 | evidence-derived confidence | `search/confidence.ts` | weights `SEARCH_CONFIDENCE_WEIGHTS`; a deterministic answer never above its operator's |
| 26 | deterministic operators first-class | `search/operators/` | v3's + history, comparisons, sequence, duration, aggregate, group, titles, identity, author checks, premises |
| 27 | hybrid answers: result, then explanation | `ask.ts`, `claims.ts agreesWithResult` | `result_text` + `explanation` |
| 28 | `/ask` | `api/ask.ts`, `search/ask.ts` | GET and POST, multipart images; v3's fields kept |
| 29 | `/search` | `api/search.ts`, `search/service.ts` | POST bodies, `rerank=1`, `retrieval` counts per family; default ranking v3's |
| 30 | `/help` through reasoning and claim verification | `src/help/` | modes; every sentence checked; removed sentences never glue text into an address |
| 31 | model benchmarking on frozen contexts | `evaluation/benchmark.ts`, `/admin/ask/context`, `/admin/ask/reason`, `scripts/benchmark.py` | identical messages for every model; grounding, agreement with the index, tokens, cost, latency |
| 32–35 | evaluation set, metrics, EGS, reliability | `src/evaluation/`, `eval/v4/generate.py` | 1,608 questions; nDCG/MRR/MAP/P/R; correctness, abstention, EGS, citations, reliability (0.4/0.3/0.2/0.1), latency percentiles, tokens, cost |
| 36–37 | reranker training data, learning to rank | `search/feedback.ts`, `/admin/ltr/pairs` | pairwise preferences from engagement; no training run yet |
| 38–40 | embedding fine-tuning, LLM fine-tuning, LoRA | `/admin/ask/export-sft` | examples whose claims all held and nobody voted down; no fine-tuning (spec §62: after the baseline) |
| 41–42 | prompt architecture, evidence-first prompt | `src/llm/prompts.ts` | policy + task + question + evidence + schema; `PROMPT_VERSION` (v4.2) in every cache key |
| 43 | context compression | `ask.ts selectCards` | deciding evidence first, then verified candidates by verification blended with the cross-encoder; at most `SEARCH_FINAL_K` cards |
| 44 | query expansion | `retrieval.ts` | the concepts' aliases in every language (`SEARCH_EXPANSION=rules`), as v3 |
| 45 | multilingual | `answer-text.ts`, `query-planner.ts`, `query-router.ts` | EN, FR, DE, ES, IT: rules, deictics, answers |
| 46–47 | visual questions, image identity | `search/image-question.ts`, `api/image.ts` | exact (same bytes, any version), perceptual (pHash ≤ 4 and colours ≥ 0.8), historical (first appearance), visual similarity (never identity) |
| 48 | temporal reasoning | operators: sequence, duration, history, relative time | creation vs first appearance kept apart |
| 50–51 | caching, versions | `llm/reasoning.ts generateCached`, `help/answer.ts`, `ask.ts versions` | keys on model, prompt version, reasoning, tokens, language, question, evidence; `versions` in every answer |
| 52 | observability | `migrations/0005_v4.sql` (`ask_log`, `answer_feedback`), `/admin/ask/log`, `/admin/ask/trace/:qid` | traces sampled (`SEARCH_TRACE_SAMPLE`), at most 32 KB |
| 53 | cost control | `ask.ts` (`worth`), `llm/model.ts` (prices) | a model only when asked for, for a synthesis, or when no operator answers |
| 54 | latency | `evaluation/models.ts`, offline harness | engine P50/P95/P99 apart from model latency |
| 56 | backward compatibility | `test/v4-v3-parity.test.ts`, `test/fixtures/v3-answers.json` | v3's routes, parameters and fields unchanged |
| 57–58 | configuration, initial models | `wrangler.jsonc` | v4: gpt-oss-120b reasoning, Gemma 4 planner, Nemotron 3 help, bge-reranker-base; v4.8.1: GLM-5.3-Flash reasoning and help, GLM-5.3 for the deep band ("Models, October 2026" below) |

## Decisions worth knowing

- **The model explains, it never answers instead of the index.** Before the first independent review,
  a model's verified answer replaced the deterministic sentence. The review showed answers whose
  every atom existed somewhere in the evidence but whose statement was wrong ("Lake came before
  Swan", the right author for the wrong title, a bare "Mallory"). Now the index's sentence always
  comes first and the model's text is added only when it states the same values
  (`agreesWithResult`), so a wrong explanation can be dropped but never shown as the answer.
- **Claim atoms are typed.** A number is checked only against facts of its kind (votes, money,
  percentages, durations, sizes, counts); ids, scores, similarities, hashes and timestamps are
  not facts; votes and payouts must belong to the post the claim names; accounts come from the
  index's structured fields, never from titles or captions (which anyone can write).
- **v3 stays v3.** v4's planning additions (demonstratives as fillers, "¿"/"¡" as punctuation,
  "in May" read after other words, multilingual rules) apply only to v4's /ask; `mode=v3`, `/search` and `/suggest` plan exactly
  as v3, and a test compares them with v3's own code's answers.
- **The cross-encoder orders evidence; it does not rank /search by default.** Measured below:
  blended at 0.1–0.2 it changes nothing on the judged queries; alone it is far worse, because it
  reads text and most Pixagram queries are about images. `/search?rerank=1` blends it at
  `SEARCH_RERANK_BLEND` (0.2); /ask uses the same blend to choose which candidates the model reads.
- **Asking back is an answer.** "Who posted this?" with nothing named, "¿Cuándo se publicó esta
  imagen?", a title several posts share: `clarify`, with no result, evidence or items that would
  answer about every artwork.
- **The nsfw filter holds everywhere.** Image origins, appearances and posts named by a history
  follow it as /search does; the answer says how many posts it left out, never which.
- **A reading the question does not state is said, or not made.** A subject the corpus spells
  otherwise ("dargons") is read one edit away only when nothing matches it as written, keeps every
  filter of the question, and the answer says "read as “dragon”" (its confidence at most 0.75). A
  word after "by" is an account only when written with "@" or a known author's name: "made by AI"
  and "par erreur" claim nothing about an account.

## Measured

Everything below ran offline in this repository's harness: the Worker's own code on Node with
SQLite for D1, the chain snapshot of 2026-10-05 (eval/offline), real SigLIP 2 NaFlex vectors from
a local copy of the Space, and the real `BAAI/bge-reranker-base` for the cross-encoder. No
Workers AI model was reachable from here, so every number is the deterministic engine's.

### Tests

`npm test`: 690 tests, all passing — v3's own (unit, integration, the regressions of its reviews),
v4's (`test/v4-*.test.ts`: the router and modes, decomposition in five languages, operators,
evidence and verification, claims, the reasoning path with test doubles for the models, visual
questions, /search and /help, the evaluation code, the regressions of v4's two independent
reviews, and v3 parity) and copy detection's (`test/paph-*.test.ts`, below). `npm run typecheck`
covers the Worker, the tests and the evaluation harness; `wrangler deploy --dry-run` bundles the
Worker (2,196 KiB, 727 KiB gzipped, of which PAPH-X's WebAssembly is 777 KiB). CI runs all of it,
and checks the question set, on every push (`.github/workflows/ci.yml`).

### v3 compatibility (spec §56)

`test/v4-v3-parity.test.ts` runs 38 /ask questions (English, French, German, Spanish, Italian,
including "Qui a posté ça ?" and "¿…?") and 12 /search queries through v4's `mode=v3` and
`search()`, and compares every field with the answers of v3's own code (commit 16061e6, frozen by
`scripts/v3_fixture.sh`). They are identical; /search only adds `retrieval` and `reranked`.

### Retrieval: the judged queries (160, eight categories)

| ranking | nDCG@10 | MRR | R@50 | better / worse than v3 |
|---|---:|---:|---:|---:|
| v2's RRF | 0.875 | | 0.98 | |
| **v3's feature ranker (v4's default)** | **0.953** | **0.981** | **0.984** | — |
| + cross-encoder, blend 0.1 or 0.2 (top 20 or 50) | 0.953 | 0.981 | 0.984 | 3 / 4 |
| + cross-encoder, blend 0.3 | 0.951 | 0.978 | 0.984 | 3 / 7 |
| + cross-encoder, blend 0.5 | 0.948 | 0.973 | 0.984 | 4 / 10 |
| cross-encoder alone (top 20) | 0.787 | 0.842 | 0.984 | 6 / 74 |
| cross-encoder alone (top 50) | 0.738 | 0.810 | 0.984 | 4 / 80 |

Alone, the cross-encoder falls furthest on colour (0.411), multilingual (0.636) and visual
(0.707) queries: it reads titles, tags and captions, not images. The spec's targets (nDCG@10 ≥
0.97, R@50 ≥ 0.98) are met for recall, not yet for nDCG: colour queries (0.750) hold it back.
Its other three (evidence grounding ≥ 0.95, claim accuracy ≥ 0.95, citation accuracy ≥ 0.98) are
about model answers: the benchmark measures them once a model is reachable.

### The v4 question set (1,608 questions, spec §32)

Generated from the snapshot by `eval/v4/generate.py`, an oracle written apart from the Worker
(Python, from the raw chain data), in nine categories and five languages. `auto` is v4 as
deployed without a reachable model (the same as `fast` here); `v3` is v3's /ask on the same
questions (it has no image questions).

| category | n | v4 `auto` accuracy | reliability | abstention | v3's /ask accuracy |
|---|---:|---:|---:|---:|---:|
| factual | 195 | 1.000 | 1.00 | 1.00 | 0.513 |
| semantic | 220 | 0.905 | 0.89 | 0.97 | 0.814 |
| visual | 200 | 1.000 | 1.00 | 1.00 | — |
| temporal | 175 | 1.000 | 1.00 | 1.00 | 0.411 |
| comparative | 197 | 1.000 | 1.00 | 1.00 | 0.005 |
| multi_hop | 146 | 1.000 | 1.00 | 1.00 | 0.041 |
| ambiguous | 97 | 1.000 | 1.00 | 1.00 | 0.000 |
| multilingual | 204 | 1.000 | 1.00 | 1.00 | 0.328 |
| adversarial | 174 | 1.000 | 1.00 | 1.00 | 0.144 |
| **all** | **1608** | **0.987** | **0.98** | **1.00** | **0.320** (1408) |

By language (v4): en 0.984 (1352) · fr 1.000 (64) · de 1.000 (64) · es 1.000 (64) · it 1.000 (64); v3: en 0.332 · fr 0.203 · de 0.375 · es 0.000 · it 0.469.

| | v4 `auto` | v3 |
|---|---:|---:|
| false answers (an answer where the set expects none) | 0 | 53 |
| false abstentions | 7 | 393 |
| confident errors (wrong, confidence ≥ 0.8) | 9 | 248 |
| required evidence present | 0.967 | 0.508 |
| semantic questions: nDCG@10 · MRR · MAP · R@10 | 0.848 · 0.923 · 0.771 · 0.791 | 0.779 · 0.843 · 0.707 · 0.744 |
| engine latency P50 · P95 · P99 | 2 · 405 · 432 ms | 381 · 425 · 446 ms |

The first full run, before the fixes described below, scored 0.868.

**Reworded questions.** Three sets of the same questions in other words, written apart from the
templates (`eval/v4/paraphrases.jsonl`, 48; `eval/v4/paraphrases-2.jsonl`, 54;
`eval/v4/paraphrases-3.jsonl`, 56, written after the second review below), measure what the set
above cannot: phrasings the rules were not written from. Each was run once before any rule was
changed for it:

| | v4 `auto` | v3 | false answers (v4) | confident errors (v4) |
|---|---:|---:|---:|---:|
| reworded set 1 (48) | 0.771 | 0.167 | 0 | 0 |
| reworded set 2 (54) | 0.611 | 0.148 | 0 | 3 |
| reworded set 3 (56) | 0.750 | 0.196 | 1 | 2 |

That is the honest estimate for new phrasings: about two thirds to three quarters, offline,
without the planner model. The misses mostly fall back to a search or to "nothing found"; six
answered wrong or answered where the set expects a question back ("Count the artworks of @x" read
as a search; "the person who made “T”" read as a subject; "@a a-t-il publié davantage d'œuvres
que @b ?" read as whether @a posted any "davantage"; "C'est de qui, ça ?" answered instead of
asked back). All three sets were then fixed in general terms, with tests, and now score 1.000;
on Workers, a question the rules are unsure of also goes to the planner model
(`PLANNER_BACKEND=auto`), which no offline run can measure.

How to read these numbers:

- The question set was written by the same hand as the rules. The first full run (0.868) showed
  where v4's planner failed — accounts with dots, German and Italian quotes, French
  interrogatives, demonstratives in four languages, "Was “T” posted by @x?", "the first
  artwork" as a premise — and the rules were fixed for those forms, in general terms (not per
  question), with tests. The second run measures those fixes on the same set: it is a regression
  measure more than an estimate of quality on new phrasings. Real questions from `ask_log` and
  the votes on them are the next measure.
- A second independent review, of the planner rules alone, probed phrasings of its own and found
  12 misreadings: "Was “Lake” made by AI?" as a claim about an account @ai; "Does @alice have
  more votes than @bob?" as a count of "vote" artworks; "Who has the most recent cat artwork?" as
  the most prolific author; a misspelled subject respelled without the question's author filter;
  "reposted by @carol" as reposted by anyone; French "mai" (May) dropped as Italian "ever"; an
  Italian title making the answer Italian; "this month" and "das erste Kunstwerk" asked back as
  if they named nothing. Each is fixed in general terms with a regression test
  (`test/v4-review-planner.test.ts`). The same reviewer then probed the fixes and found 9 more,
  among them two the fixes had caused ("Who posted this in September?" answered as if the date
  named the artwork; "Qui a le plus de votes ?" answered with the single most liked post) and
  older ones ("Who posted “Lake”? Was it @alice?" read “Lake” as a subject; "Did @bob repost
  “Black cat”?" ignored @bob; "in May" after other words was not read as a month). Those are
  fixed and tested too. Through both rounds the 1,608-question set kept its score (0.987), with
  fewer confident errors (9, from 11).
- Semantic questions are scored by whether the first result is one the judgments mark relevant;
  the 21 semantic misses are the ones a subject search cannot settle from text and image similarity ("mirror selfie", "cozy room with a fireplace", "femme aux yeux verts", "電車"), plus a post the nsfw filter hides.
- Latency is the engine's on Node and SQLite with a CPU SigLIP server: the P95 is the text
  embedding of semantic questions (~0.4 s on this CPU). On Workers, D1, Vectorize and the Space
  have their own latencies; `took_ms` and `timings` in every answer measure them there.

## Copy detection (PAPH-X)

Every artwork is fingerprinted when it is indexed, checked against every artwork indexed before
it, and what the verifier concludes — `Identical`, `Copy`, `Suspected` — is stored with which of
the two came first on chain. `GET /copies/:id` lists a work's copies in milliseconds (a D1 read);
an uploaded image is checked across the whole index within a 900 ms budget (`POST
/copies-by-image`), and its answer is kept for a day. The engine is `@pixagram/paph-x` 1.2.0
(pinned): PAPH 4.2's wires in format 4 (sampling that commutes with mirrors and quarter turns) and
comparator 42 under CAL-007, PAPH-X's cascade (XRank, profile X3-PROVISIONAL) to reach the same
verdicts cheaply, and PAPH-SI (profile SI4-PROVISIONAL, fitted on the Pixa chain's own artworks
hashed in wire 4) beside the exact index keys to find the few stored works worth comparing.

```
queue consumer:  stats ─► paph ─► embed (─► vector pass) ─► describe ─► concepts ─► text
                           │ hash in WebAssembly (5–60 ms), or the stored wires of the same image
                           │ (and its wire-3 twin while some shard still holds works hashed before 1.2.0)
                           ▼
        home shard: PaphShard Durable Object holding post ids [i·S, (i+1)·S), S = PAPH_SHARD_SIZE
          store the wires, index them: local codes + descriptor bands (exact keys), PAPH-SI keys
                           │
                           ▼  every shard at once, each within the same budget (src/paph/budget.ts)
          nominate  codes and bands, rarest first, Σ 1/df, top K per family
                    ∪ PAPH-SI: probes rarest first, reached candidates scored exactly, best B
                    ∪ pHash neighbours (D1) ∪ embedding neighbours (Vectorize) ∪ previous partners
          verify    XRank, policy safe, copy scope; best candidates first, in batches, until the
                    deadline; each stored side against the query of its own wire format
                           ▼
        D1 paph_matches: (a = earlier by chain time, b = later), both images, verdict, evidence,
        engine identity; artworks.paph_hash / paph_engine: the image and identity checked
```

**Why an upload stays sub-second as the corpus grows.** Works are sharded by post-id range: the
corpus grows into new shards and nothing is ever moved. A check runs in every shard at once, and
what a shard does is bounded by counts, whatever its size: posting rows the key nomination may
read (rarest keys first — they carry the most weight), posting rows PAPH-SI may read, how many SI
candidates are scored exactly and kept, how many candidates XRank compares. Shards run in
parallel when Cloudflare places them in different isolates (Durable Objects sharing an isolate
share its thread), so a check's latency is that of one shard; its total work grows with the
number of shards, which is what the counts bound. The interactive budget reads ≤ 40,000 key and
≤ 30,000 SI postings and compares ≤ 96 candidates per shard. Time bounds a check
only through its deadline, measured from its arrival in the shard, waiting included: when its
caller stops waiting, the shard stops too and answers what it has, flagged `partial`. The whole
request is held to `PAPH_QUERY_MS` (900 ms) from its start (with `semantic=1`, from when the
embedding is in), and a shard that does not answer in time is reported (`shards.failed`).

One check runs at a time in a shard: interactive ones (uploads, live re-checks) in arrival
order, then background ones (the enrichment stage, re-derivation, the clean-up). A check runs in
slices — postings read 25,000 rows at a time, XRank in batches of 32 — and hands the turn back to
the runtime between them. On Cloudflare the clock moves only between turns, so this is what lets
a shard see a deadline at all; and it is where background work steps aside for a waiting upload,
which therefore waits for the slice under way (tens of milliseconds), for the uploads ahead of it
and, at most once a second, for a background turn of four slices: a background check that has
waited a second gets that turn ahead of the uploads, so background work completes under any
load. A queue holds at most 64 checks per kind (beyond, the shard answers busy and the check is
partial); a holder whose request was dropped loses the turn after 30 s (one timer per shard,
armed only while checks wait, so an idle shard can hibernate). At ~0.25–0.35 s a check
(below), a shard answers some 3–4 uploads a second; the day-long cache and the per-client limits
keep repeated uploads off it.

The stage's budget is five to ten times the interactive one (≤ 512 compared per shard, a
minute to answer, waiting included, plus 10 s of grace): the stored verdicts, which
`/copies/:id` lists, come from the thorough check. A check some shard did not finish is retried
on those shards only (a forced re-check starts over on every delivery). The ones that answered in
full are remembered in `paph_progress`, written in the same D1 batch as the verdicts they vouch
for and only while the post is live and shows that image; the row names the shard size its
numbers refer to, and a retry that continued from it writes its mark and its own progress only
while that row is still there — a deletion (or a forced run starting over) removes it, and the
work is checked in full again (two deliveries of the same retry both end complete). A shard that
answers under another engine identity (a deploy reaches the Worker and the shards at different
moments) counts as failed: what it found is not stored, and the retry asks it again. A run that
a deletion or a newer image overtook leaves no job record ("skipped" is for what is permanent
about an image: the sweeper never re-drives it).

Cloudflare stops a queue consumer run after 15 minutes. A run starts no message after 12
minutes (the rest are sent again, 30 s later, without using up a delivery), and the stage cuts
its check to end by then, or defers it when less than 5 s would be left (a work whose mark says
its check is complete is taken as it is); a call to the work's own shard (its entry, its wires,
the put) waits at most 20 s, and no longer than the run can spare. Without that, a hung shard
would hold every message of a batch for the full wait, twice with the embedding's own pass, and
with the other stages take the batch past the limit.

The index holds what a completed check vouches for. The put is a compare-and-set on the entry
the run read: of two runs for two images of the same post, the older cannot put its image back
over the newer's, and a run that fails takes out only the old image's entry, only while the post
still shows its own image. What no single step can rule out — an edit and its revert in flight
at once, a removal that raced a restore — the nightly clean-up finds: a live work marked
complete whose entry shows another image or is missing loses its mark and is sent back to the
stage. It handles a hundred such works a page (a shard that lost its entries wholesale heals over
several nights; `reindex-all paph` is faster). Rarely, it also sends back a work that a run had
just completed, which costs one more check.

A first backfill of a large corpus checks every work against every shard, so its total cost
grows with the corpus times the number of shards; `PAPH_BUDGET_STAGE` can lighten it, and a later
re-check (`scripts/admin.sh reindex-all paph force`) restores the thorough one.

**Kept for a day.** An upload's comparisons are cached under its fingerprint (the same pixels in
any encoding hit the same entry), the engine's identity and the budget, for `PAPH_CACHE_TTL`
(86,400 s), and looked up before anything else is computed (with `semantic=1`, before the
embedding too); a cached answer touches no shard. The listing is rebuilt from D1 on every
request, so a work deleted or edited since drops out (`hidden` counts it). An answer that some
shard or channel did not complete is kept for ten minutes only. Pair reports are kept a day under
both content hashes.

**Precise.** Only XRank decides; the keys, PAPH-SI, pHash and the embeddings only nominate. Under
the safe policy every `Copy` is comparator 42's or certified by PAPH-X's sparse geometry, and the
listings show `Copy` and `Identical` unless asked (`min=suspected` is the lowest threshold:
`Suspected` is for review — on the chain's works CAL-004 read it on 222 of 14,412 pairs of two
authors' works, on shared style alone; CAL-007 raises that one bar above the highest such pair
and reads it on none, `Copy` on none either, and moves no `Copy`: PAPH-X 1.2.0). XRank screens
the candidates before comparing them; under X3 (X2's schedule bound to CAL-007) the screen's
exits ask the structural channels first, and PAPH-X 1.2.0 measures XRank reading `Copy` on all
6,190 copy queries of the chain's works and 1,990 synthetic ones, and on none of the 31,148 chain
queries comparator 42 does not call `Copy`. A candidate the screen rejects was not compared, so
it never replaces a stored verdict. A verdict is about two images
and is written, replaced or listed only while both posts are live artworks showing them: a check
that raced an edit or a deletion can neither erase a newer check's verdicts nor resurrect old
ones, and a work deleted while it was being checked leaves the index again. A shard entry whose
post was deleted anyway (its removal is best effort) is cleaned up by a nightly pass through every
shard (each shard on its own: one that fails does not stop the others).

**Identity, and the next release.** Every verdict, every artwork's completed check and every
cached answer carries the identity of what produced it — `paph-x/1.2.0 c50
CAL-007-PROVISIO:741afad9252f2ccb X3-PROVISIONAL:8af84dd0abb12192 si:aa8ce6d311f4ce56 k1 w4`
(CAL-007's name field holds 16 characters) and the policy — so a new release, profile or
`PAPH_POLICY` makes the stage re-check every work (the sweeper finds them too). A first deployment
on 1.2.0 has nothing to re-derive; a later release that keeps the API (1.1.1, 1.1.2 and 1.2.0
did) is moved to by:

```bash
npm install --save-exact @pixagram/paph-x@<version>  # and PAPH_X_VERSION in src/paph/engine.ts (a test checks it)
npm test && npm run deploy
scripts/admin.sh paph-rederive 500                  # repeat until remaining is 0: keys and SI signatures from the
                                                    # stored wires, no image fetched (another wire format: rehash)
scripts/admin.sh reindex-all paph                   # re-check every work under the new identity: from the stored
                                                    # wires, or from the image when they are on another format
scripts/admin.sh paph-stale                         # verdicts no re-check has confirmed, and works still waiting
scripts/admin.sh paph-purge-stale                   # removes them once no work waits (refused before)
```

A new SI profile changes every stored signature (1.1.2's SI3 and 1.2.0's SI4 did);
`paph-rederive` re-derives them from the wires, keyed by the derivation stored with each work.
`setPaphProfiles()` in `src/paph/engine.ts` runs other profile artefacts than the package's
defaults (the earlier ones ship in its `docs/calibration/`; while the SI profile is bound to another
X profile, SI sits out and `/admin/paph` says why). They do not bring 1.1.2 back: this Worker hashes
wire 4, and SI3 and X2 were measured on wire 3 (the package's SPEC-W4 §9 path 2, staying on wire 3,
is not supported here).

**Moving a store to 1.2.0: wire 4.** 1.2.0 changed the wire itself: `hash()` writes format 4,
whose thumbnail, DCT, shape regions, radial profiles and silhouette are sampled so that a mirrored
or quarter-turned image's sections are the original's, moved (on the chain's works, the route's DCT
word matched its original's on 96 % of such copies with both sides multiples of 16 and on 25 % of
the others in wire 3, and on all of them in wire 4). A wire-3 side and a wire-4 side are never
compared (`Indeterminate`, `WIRE_MISMATCH`), so every work indexed before 1.2.0 is hashed again: its
identity moved, so the stage re-checks it, finds its stored wires on wire 3, and fetches and hashes
its image again — the one step of this move that costs a fetch per work (`reindex-all paph` speeds
it up). Meanwhile copy detection keeps working across both formats (the package's SPEC-W4 §9,
path 1):

- each entry records its wires' format (Tier 1's byte 4); the exact index keys are the same in both
  formats, so they keep nominating across the move; PAPH-SI holds wire 4's signatures only, and
  `paph-rederive` drops 1.1.2's (SI3) from the wire-3 entries without fetching anything, which
  spares the move's checks reading postings that cannot score;
- while some shard holds wire-3 entries, an upload and the stage (also when it reuses stored
  wires: a retry, a forced re-check, the image decoded for this alone) hash the image in wire 3 as
  well — its twin, about 5–60 ms more — and each stored side is compared with the query of its own
  format. Each check reports whether its shard still holds wire-3 entries; once a check that heard
  from every shard finds none, queries stop hashing twins (`/admin/paph`: `wire.legacy_works`,
  `wire.legacy_shards`, and `wire.twin` as the isolate that answered sees it);
- a verdict reached on wire-3 sides carries `w3` in its identity and is provisional: it never
  replaces or erases a verdict reached on wire 4 (the stored side's own re-check may write one while
  the provisional one is on its way), the stored side's re-check replaces it on wire 4, and the
  nightly clean-up compares again any it finds whose two works have both completed their check —
  written after that re-check (`scripts/admin.sh paph-heal` does it at once). `paph-stale` counts
  them apart (`wire3`) and the purge never removes them;
- a live re-check of a work not re-hashed yet compares it with the works on its own format and says
  how many it could not compare (`partial`, `note`); an upload whose check met wire-3 works without
  a twin is kept ten minutes, not a day; a pair report names both formats (and is keyed by them,
  so a re-hash is a new report). The embedding pass after the stage brings no twin: a wire-3
  candidate it meets is left to that work's own re-check.

The first start of each shard on this release indexes its entries by format (`works_wire`), reading
the small columns of every entry once: 23 ms for 20,000 entries here, with the pages cached. A work
whose image can no longer be fetched keeps its wire-3 entry, compared through twins, and its
provisional verdicts stay; `paph-gc` removes entries of deleted posts as before.

**Measured here** (Node, the real WebAssembly; `test/paph-store.test.ts`, `test/paph-flow.test.ts`,
80 tests):

- mirrored, cropped, 2× upscaled, channel-swapped and pasted copies of a real Pixagram artwork
  are found as `Copy` among 24 procedural distractors, each through more than one nominator;
  the distractors are never `Copy`;
- every statement of a check and of an index write runs on primary keys (query plans checked);
  postings read in chunks give the same nominations as in one; posting reads and the verify cap
  hold; on a clock that moves only between slices (as on Cloudflare) the deadline holds; the
  gate drops no candidate the ungated comparison calls a copy (six copies and recolours against
  every work); four concurrent checks run one after another, the first ones
  exactly as alone, the ones whose turn comes after their deadline doing no work; an interactive
  check overtakes a background one at its next slice, with the same verdicts; background work
  still completes under two overlapping streams of interactive checks; a full queue refuses; a
  turn whose holder was dropped comes round, with one lease timer however many wait (50 queued:
  one timer, none once they are served);
- the stage across three shards: arrival order, unchanged images skipped, re-checks from the
  stored wires without fetching the image, a policy change re-checking every work without force
  (and the purge refused until it has), a changed image's verdicts withdrawn, a newer image
  winning, a post deleted while it is checked leaving no trace, a failed shard or an unavailable
  channel leaving the stage incomplete — and its retry asking only the shard that failed; no
  progress kept for a post deleted during a partial check (restored, every shard is asked and the
  copy found), nor trusted once it was reset while a retry ran, nor across a change of shard
  size; two deliveries of the same retry both complete (and settle the job); a shard answering
  under another engine neither stored nor counted done; an older image's run refused over a
  newer one's entry, and a newer one putting its own over an older one; a failing run leaving a
  newer run's entry and mark alone; the clean-up sending back works marked complete whose entry
  shows another image or is missing, a hundred a page, in queue batches of 100 at most, resuming
  where it stopped; an overtaken run recording no job, so the sweeper still
  finds the work; in the consumer, a check's deadline cut to the run's time left (a configured
  short deadline still runs), a slow home shard given up at what the run can spare, a late check
  deferred (a complete one taken as it is) and a late message sent again;
- verdict replacement against races: a late check of an old image erases nothing a newer check
  established; nothing is written about a deleted post, a replaced image or a post that is no
  artwork any more; only compared works with a stored verdict are bound (5,000 compared, none
  stored: no delete at all); listings hide pairs whose images changed since, and count as hidden
  only what they passed over;
- an upload across three shards answers within the test's one-second bound; the same pixels again
  come from the cache without a shard call, with a TTL of 86,400 s (600 s when a shard or channel
  was missing); the embedding of a semantic upload is computed only on a cache miss; a shard
  slower than the request's budget is cut and reported; ids D1 does not know never reach a shard;
- the whole pipeline from the chain: ingest → stats → paph → `/copies` from both sides → a
  paph-only re-check that does not ask the chain → the copy deleted on chain leaves the index and
  the listings;
- the move to wire 4, on entries written as 1.1.2 wrote them (wire 3, SI3 signatures and
  postings): the format read from Tier 1's byte 4 (keypoints and index keys the same in both
  formats, a mixed pair refused); the re-derivation keeping their keys and dropping their SI3
  postings; a copy of a wire-3 original found only through the query's wire-3 twin, everything else
  compared on wire 4, a refused twin leaving those entries unread; a new work's copy of a work not
  re-hashed yet found at once, its verdict named `w3` until the original's re-check re-hashes it and
  replaces the verdict on wire 4; a provisional verdict arriving after that re-check replacing
  nothing, and one written after both checks compared again by the clean-up and never purged; a
  retry that reuses the stored wires still bringing the twin; twins dropped once no shard holds
  wire 3, and brought back by a check that meets one; a live re-check of a wire-3 work and a pair
  report of two formats saying what they could not compare.

A check's nomination, as a shard runs it, on a synthetic shard of 60,000 works (37.4 M band,
7.2 M code and 2.8 M SI postings with skewed key frequencies; 1.1 GB of SQLite on local disk;
8 queries of 10,240 keys each): the document frequencies of the query's keys in 30 ms, the key
nomination in 37 ms (≈ 11,000 postings read, max 57 ms), PAPH-SI's in 14 ms (max 29 ms) — about
0.1 s in all. Preparing a stored work for XRank costs 0.4–0.7 ms on the two real fixtures (151
and 208 keypoints; the chain's works have 512 at the median, so more there), and PAPH-X 1.2.0
measures XRank at 0.65 ms a candidate natively under X3 (0.72 under X2) on pairs of the chain's
own works, where the cascade runs whole (WebAssembly is about 1.2× native on its reference
workload): ≤ 96 comparisons
add ~0.15–0.25 s. A shard's interactive check is therefore ~0.25–0.35 s, inside its 600 ms
deadline.

**Not verified here.** Durable Object latency (cold starts, storage reads after eviction), how
Cloudflare places shards in isolates, that the runtime advances its clock across the slices'
yields (if it did not, the counts and the Worker's own timeout would still bound a check), and
XRank's cost on the whole corpus (PAPH-X 1.2.0 measured 177 of the chain's works, re-hashed in
wire 4: 648 µs a query natively under X3, against 717 under X2):
`/admin/paph` (queued checks per shard) and the `paph` log line of every stage (per-shard times,
waits) report them. Recall on real copies at scale is PAPH's to measure (its SPEC-SI §9); the
budgets are configuration (`PAPH_BUDGET_QUERY`, `PAPH_BUDGET_STAGE`).

## Not verified here

- **Live models.** No Workers AI call could be made from this environment: the reasoning model,
  the planner model, the help model, the vision model's descriptions of uploaded images and the
  Workers AI cross-encoder ran only against test doubles. Their request formats follow
  Cloudflare's documented schemas (`src/llm/adapters/workers-ai.ts`), and every failure path
  (timeouts, unusable replies, missing models) degrades to the deterministic answer.
- **The model benchmark (spec §31, §58; DoD: three models).** The benchmark runs and is tested
  (three models on one frozen context in `test/v4-evaluation.test.ts`), but the live comparison
  needs the deployed stack:

  ```bash
  ADMIN_TOKEN=… python3 scripts/benchmark.py https://pixagram-search-v4.<you>.workers.dev \
      --models @cf/zai-org/glm-5.3-flash,@cf/zai-org/glm-5.3,@cf/deepseek-ai/deepseek-v4-flash-0731,@cf/openai/gpt-oss-120b,@cf/qwen/qwen3.8-27b \
      --reasoning medium --limit 60 --out bench.json
  ```

  It reports per model: grounding (EGS), citation accuracy, answers rejected by claim
  verification, how often the answer states the index's result, agreement with the question
  set, abstentions, tokens, cost per 100 questions, P50/P95 latency.
- **Latency and cost on Cloudflare.** Measured per answer once deployed (`/admin/ask/log`
  totals tokens, cost and latency by mode, class and model).

## Definition of done (spec §62)

| item | | |
|---|---|---|
| Existing v3 deterministic tests pass | ✅ | v3's tests run unchanged in the suite; `mode=v3` and `/search` match v3's own answers field for field |
| Retrieval quality does not regress | ✅ | same ranking, same judged-query numbers (nDCG@10 0.953); the cross-encoder is opt-in |
| `/ask` supports adaptive reasoning | ✅ | four modes chosen by complexity; reasoning levels apart from the answer's length |
| Models are interchangeable | ✅ | one model layer (Workers AI, OpenAI-compatible endpoints); roles and bands are configuration |
| At least three reasoning models benchmarked | ⏳ | the benchmark runs on frozen contexts and is tested with three models on test doubles; the live run needs the deployed stack (command above) |
| Evidence cards | ✅ | E, R, D, C, I1 cards and the graph |
| Claim verification | ✅ | every model answer, /ask and /help |
| Contradictions detected | ✅ | first sightings against chain operations and between cards; C-cards; said in the answer |
| Complex questions decomposed | ✅ | 26 rule patterns in five languages on top of v3's single-intent plans; the planner model for the rest |
| Cross-encoder evaluated | ✅ | 160 judged queries, five blends, two depths (above) |
| Evaluation set ≥ 1,000 questions | ✅ | 1,608, nine categories, five languages; three reworded sets |
| Model and version provenance recorded | ✅ | `versions` in every answer and in `ask_log`; in every cache key |
| Unsupported claims rejected or qualified | ✅ | and a model's text is never the answer in the index's place |
| Cost and latency measured | ◐ | the engine's latency offline; tokens, cost and model latency recorded per answer and summarised by `/admin/ask/log` and the evaluation scripts once deployed |
| Regression tests run automatically | ✅ | CI on every push |
| Fine-tuning only after the baseline | ✅ | not started; the exports it would use exist (LTR pairs, SFT examples) |

## Next

0. Copy detection: after deploying, `scripts/admin.sh reindex-all paph` fingerprints and checks
   every artwork once (new ones go through the stage by themselves); `scripts/admin.sh paph`
   shows the shards, the verdicts and the stage's progress.
1. Deploy (`scripts/deploy.sh`), run `scripts/benchmark.py` with the five models of spec §58 and
   set `SEARCH_REASONING_MODEL` (and the per-band models) from it.
2. Run `scripts/eval_v4.py` against the stack in `fast` and `auto`, and compare with the offline
   numbers above.
3. Collect answers and votes (`ask_log`, `answer_feedback`), build a held-out set from real
   questions, and only then consider training the reranker on the LTR pairs or fine-tuning a
   model on the SFT exports (spec §36-40).
4. Colour queries (nDCG@10 0.750) are the weakest retrieval category.

## v4.8: the long-form answers

v4's answers were short by design: the index's one sentence, and a model's explanation of one to
three sentences that the engine dropped whole when a single claim failed. Most questions never
reached a model at all (fast mode), the model's reasoning never reached the reader (spec §21), and
nothing suggested what to ask next. v4.8 keeps the rule that made v4 trustworthy — the index
decides what happened, the model only explains it, and nothing unverified is shown — and builds
the answer people wanted on top of it. Nothing of the retrieval, the operators, the evidence
verification or the claim atoms changed; `mode=v3` and `style=brief` answer exactly as before (the
v3 and v4 suites run with `SEARCH_ANSWER_STYLE=brief` and pass unchanged, 701 of the 726 tests).

### What it adds

```
answer (as v4) ─► digest.ts      facts about the deciding posts, the set's overview, caveats,
                                  template follow-ups, searches — computed values only, five
                                  languages, no model call; always there
               ─► compose task   the model (balanced and above; rich raises auto to balanced):
                                  direct answer + body (Markdown, every sentence cited) + reasoning
                                  trail + caveats + follow-ups + searches, in one JSON reply
               ─► compose.ts     the body verified sentence by sentence against the cards (claims.ts
                                  atoms + contradictions, addresses, accusations; headings too), the
                                  trail and the caveats step by step; follow-ups and searches filtered
                                  (names in the evidence, no address, planned by the rules, words of
                                  the evidence, help only when the documentation covers it)
               ─► assembly       answer_markdown (citations kept) and answer_full (plain), sections,
                                  thinking, suggestions; answer_text stays the direct answer
defer=1        ─► the frozen context in KV; GET /ask/elaboration/:id runs the model once (marker,
                  three attempts at most), returns the changed fields; /suggest?after=:id continues
/query, /search ─► overview.ts: a text overview of a result page (five languages)
/help          ─► style rich: a complete answer at length, follow-ups the documentation covers,
                  the related sections of the cited pages
```

| what | where |
|---|---|
| the compose task, its schema and prompt (`PROMPT_VERSION` v4.8) | `src/llm/prompts.ts` |
| the reply parsed and bounded; a reply cut off at the token limit salvaged by a top-level partial-JSON scan (the model's status respected) | `src/llm/reasoning.ts` |
| the digest: facts, overview (exact oldest and most voted through SQL on the metadata path), caveats, templates per family and language, searches | `src/search/digest.ts` |
| sentence-level verification, the suggestion checks, the assembly and the section titles | `src/search/compose.ts` |
| `citedInterpretations` (a cited sentence without atoms: supported at ≥ 25 % word overlap, qualified below), `unknownNames`, the accusation vocabulary | `src/search/claims.ts` |
| the long-form flow, the mode floor, the set's rows as cards, the deferred path | `src/search/ask.ts` (`applyLead`, `composeAnswer`, `elaborate`) |
| the search overview | `src/search/overview.ts` |
| rich help: prompt, follow-ups, related sections | `src/help/answer.ts` |
| `/suggest?after=` | `src/search/suggest.ts suggestAfter` |
| the benchmark on the compose task (`body` grounding and length per model) | `src/evaluation/benchmark.ts`, `scripts/benchmark.py --task compose` |

### Decisions worth knowing

- **A sentence, not a reply, is the unit of verification.** v4 checked the explanation as one
  claim: one atom the evidence did not hold, and three sentences were gone. A 300-word body would
  almost always lose a sentence somewhere, so the body is split into sentences (list items and
  headings included), each verified with the same atoms, contradiction tests and ownership rules
  as a claim, and only the failing ones are removed. The review measured on the fixture: a body
  with a made-up post, an address and an empty heading keeps its seven good sentences and loses
  the three bad ones.
- **The index still speaks first.** `answer_text` is `result_text` then the model's direct
  answer, judged as in v4 (`agreesWithResult`); a direct answer that contradicts the result drops
  the whole reply, body and trail included, because a model that got the main fact wrong is not
  trusted on the rest. A model that only fails to restate the value keeps its verified body.
- **"Thinking" is a written trail, never the chain of thought.** Spec §21 stands: the adapters
  drop the models' reasoning. What the reader sees under "How this was worked out" is a list of
  steps the model wrote for the reader, each citing cards and each verified as a claim; steps that
  fail are removed and the rest renumbered. It is checkable, which the raw reasoning is not.
- **The digest can never be wrong about the index, so it says only what it read.** An exact count
  reads the set's newest 24 posts, and its oldest and most voted by SQL, so the range and the most
  voted are the set's; tags and colours are said of the rows read. A search overview says "Of the
  20 shown: …" when the page is not the whole set.
- **Follow-ups must lead somewhere, and must be safe to show.** A model's question is kept only
  when every account, title, number and date in it is in the evidence, it carries no address and
  no accusation, the rules plan it with confidence, and its subject is made of words the evidence
  carries (or, for a help question, the documentation covers it). The templates (per family —
  author, subject, named post — and language) are checked against the planner in the tests:
  every one of them is a question the index answers. In French, German, Spanish and Italian the
  subject is attached to a noun of fixed gender (« œuvre de chat », „Kunstwerk mit Katze“, «obra
  de gato», «opera di gatto»): a typed subject has no gender or number a template could know.
- **Brief is v4.** Token budgets, the search box's ceiling, prompts and fields of the brief style
  are v4's; only the prompt version moved (every cache key changes with it).
- **A deferred answer costs at most three model calls.** The frozen context is written before the
  answer goes out; a marker holds while the model runs, so overlapping polls wait; a transient
  failure is tried again by the next call up to three times, a reply the engine cannot use is
  final; the result and a final failure are kept for `SEARCH_ELABORATION_TTL`.

### Measured

Offline, in this repository's harness (Node, SQLite for D1, the models as test doubles that
answer with fixed replies): `npm test` — 726 tests, 25 of them v4.8's (`test/v4-rich.test.ts`,
and one in `test/v4-visual.test.ts`): the digest in five languages, every follow-up template
planned and answered (over 60 generated questions), the body verification on a reply with a
made-up post, an address, an accusation and an empty heading, the trail and the caveats, the
mode floor and the budgets per style, a contradicting lead, a salvaged reply, the deferred flow
over HTTP with its failures and its marker, `/query` and `/search` overviews, `/suggest?after=`,
rich `/help`, and an image question. `npm run typecheck` and `wrangler deploy --dry-run` (2,349
KiB, 772 KiB gzipped) pass. An independent review of the first cut found eleven issues (the
deferred route open to repeated model calls, a race on the frozen context, headings and
follow-ups reaching the reader unverified, page-level superlatives stated as the set's, two
budgets changed under the brief style, templates ungrammatical in four languages, a salvaged
reply ignoring the model's status, the abbreviation rule merging real sentences); all are fixed
and tested in this release.

**Not verified here:** the models' replies to the compose prompt (how long gpt-oss-120b's bodies
come out, how many sentences the verification removes on real questions, the follow-ups it
proposes) — `scripts/benchmark.py --task compose` measures them once deployed (`body` grounding,
words, share of sentences kept, per model); Workers AI latency on the longer replies
(`SEARCH_LLM_TIMEOUT_MS` is 60 s; `timings.model` in every answer measures it); KV propagation
delays between the deferred answer and its first poll (the frozen context is written and awaited
before the answer is returned; a poll that still finds nothing is a 404 the box may retry once).

### Next

1. Deploy, then `scripts/benchmark.py --task compose --words 250` on the models of v4.8.1
   (the script's default list): confirm `SEARCH_REASONING_MODEL` by body grounding (`body EGS`,
   `kept%`) as much as by agreement, and set `SEARCH_ANSWER_WORDS` from the lengths that come out.
2. Watch `ask_log` and the notes: the sentences removed per answer (`trace.compose.body`) say
   where the prompt or the evidence cards fall short; follow-ups dropped say which templates to
   add.
3. The UI: `answer_markdown`, the reasoning trail as a collapsible, follow-ups as chips, the
   overview above the grid, `defer=1` with `/suggest?after=`.

## v4.8.1: the models of October 2026

v4 made every model configuration; v4.8.1 changes the configuration where a stronger model is
certain, and teaches the engine what those models need. Nothing else moves: the index, the
evidence, the verification and the prompts are v4.8's.

### What changed

| role | v4 | v4.8.1 | why |
|---|---|---|---|
| reasoning, every band | gpt-oss-120b ($0.35 / $0.75) | **GLM-5.3-Flash** `@cf/zai-org/glm-5.3-flash` ($0.15 / $0.50) | the strongest small model Workers AI serves (Artificial Analysis index 42 against gpt-oss-120b's 12), at less than half the price; MIT; 1M context |
| reasoning, deep band | (the same) | **GLM-5.3** `@cf/zai-org/glm-5.3` ($1.40 / $4.40) | the full model (index 45) for the multi-hop and comparative questions (complexity ≥ 0.85); ten times the price per token and slower, on the rarest band; `SEARCH_REASONING_MODEL_DEEP` |
| help | Nemotron 3 Super ($0.50 / $1.50) | **GLM-5.3-Flash** | the better writer at a third of the price; help answers are checked sentence by sentence either way |
| documentation vectors | bge-m3 | **Qwen3-Embedding-0.6B** `@cf/qwen/qwen3-embedding-0.6b` | ahead of bge-m3 on the multilingual retrieval benchmarks, same 1024 dimensions (VEC_DOCS is kept), same price ($0.0118); questions are embedded with the retrieval instruction, as the model was trained |
| planner | Gemma 4 | Gemma 4 | plans with reasoning off, which GLM cannot; the follow-up templates of v4.8 were checked through it |
| descriptions | Gemma 4 | Gemma 4, with the `chat` backend ready | GLM-5.3-Flash is multimodal, but nothing says it describes pixel art better than Gemma 4: `VLM_BACKEND=chat` + `VLM_MODEL` switch it once `scripts/admin.sh describe <id> chat` has been compared on a few artworks |
| reranker, image and text vectors | bge-reranker-base, SigLIP 2 | unchanged | no better cross-encoder on Workers AI; SigLIP stays on the Space |

The change is in `wrangler.jsonc`: the code's own fallbacks for an unset variable stay v4's
(gpt-oss-120b, bge-m3, Gemma 4), so a stack that keeps its variables keeps its models.

What that does to a request: GLM reasons in every call (`reasoning_effort` low, high or max; it
cannot be switched off, so the engine's "none" runs as "low"), so `SEARCH_REASONING_TOKENS` is
raised to `low:2048,medium:6144,high:16384` and `SEARCH_LLM_TIMEOUT_MS` to 120 s. In balanced
mode (the search box's floor for rich answers) a question costs about $0.001–0.004 on Flash; a
deep-band question on GLM-5.3 about ten times that, and it may take a minute — which `defer=1`
hides from the box.

### What the engine learnt

- **A reasoning control in the model's own words** (`src/llm/model.ts`, `effort`): each level of
  the engine mapped to the model's value, and how "none" is asked for — an effort of its own
  (DeepSeek V4: `none`), the chat template (Qwen3.8), or not at all (GLM: "none" runs as "low").
  The table knows GLM-5.3, GLM-5.3-Flash, Qwen3.8-27B, DeepSeek V4 Flash and Pro with their
  prices; `/admin/models` lists them. An unknown `@cf/` id is still taken as a legacy chat
  model, so a new model is declared in the table or in `SEARCH_MODEL_OVERRIDES` before it is
  configured. The budget of a call adds the reasoning allowance whenever the model will reason,
  including at "none" on a model that cannot stop.
- **Qwen3-Embedding's two sides** (`src/docs/vectors.ts`): chunks are embedded as documents
  (`{text}`), questions as queries with the retrieval instruction (`{queries, instruction}`;
  `DOCS_QUERY_INSTRUCTION` overrides the built-in sentence). The question-vector cache key
  carries a version, so no bge-m3 vector is read back for a Qwen3 query.
- **The `chat` description backend** (`src/enrich/describe.ts`): any vision chat model the table
  knows (or an override declares), named by `VLM_MODEL`; a misconfigured one skips the stage
  with the reason in the job (`VLM_BACKEND=chat needs VLM_MODEL …`) instead of failing every
  image. The stage's staleness is by content hash, so a model change describes new and edited
  images only; `scripts/admin.sh reindex-all describe` redoes them all.
- **The retrieval on its own** (`GET /admin/debug/docs?q=…`, `scripts/admin.sh docs-retrieve`):
  each chunk's lexical coverage, raw cosine and combined score — the tool for `DOCS_MIN_SCORE`.

### After deploying, in this order

1. `scripts/deploy.sh` — the Worker with the new variables (the Space is untouched).
2. `scripts/admin.sh docs-reembed` — every chunk re-embedded on Qwen3 (2,000 in the first call,
   the rest by the following syncs; `scripts/admin.sh docs` shows what is pending). Until it is
   done, a question's Qwen3 vector meets bge-m3 chunk vectors: the vector leg is noise and help
   relies on its lexical leg (and does not cache its answers meanwhile).
3. Set `DOCS_MIN_SCORE`: `scripts/admin.sh docs-retrieve "<question>"` for a dozen questions
   the documentation answers and a few it does not; the cosines of the right chunks sit above
   those of the wrong ones, and the line goes between them. It is the cosine at which a chunk
   counts as relevant by its vector alone (`help/retrieve.ts`: a cosine 0.15 below it scores 0,
   0.15 above it 1); 0.4 is where the configuration starts, not a measurement.
4. `scripts/benchmark.py --task compose --words 250` (its default list is v4.8.1's five models)
   confirms the reasoning model on Pixagram's own questions: body grounding, sentences kept,
   agreement, cost and latency per model.
5. Watch `scripts/admin.sh ask-log`: `timings.model` per mode and model, and replies whose
   reasoning used up the budget (`finish_reason: length` in the trace; the deterministic answer
   still shows). If deep-band answers time out, point `SEARCH_REASONING_MODEL_DEEP` at the Flash
   model or lower `SEARCH_REASONING_TOKENS`.
6. Optional: `scripts/admin.sh describe <id> chat` against `describe <id> gemma` on a handful of
   artworks; `VLM_BACKEND=chat` if GLM's descriptions read better.

### Measured

Offline: `npm test` — 743 tests, 17 of them v4.8.1's (`test/models-2026-10.test.ts`): the
request shape of every new model at every level, the budget at "none" on a model that cannot
stop reasoning, the provider's note and the cost, the deployed configuration read from
`wrangler.jsonc`, Qwen3's document and query inputs through a synced index and the versioned
cache key, the retrieval debug route, the `chat` backend through the enrichment pipeline and its
skip reason. `npm run typecheck` and `wrangler deploy --dry-run` pass.

### Not verified here

No Workers AI call could be made from this environment. The request shapes follow the models'
Workers AI input schemas as published in October 2026; what only a live call shows:
`reasoning_effort`'s accepted values on the binding (a refused field is retried without it,
with a note in the answer), where GLM returns its reasoning (the adapter reads
`choices[0].message.content` and skips `reasoning` parts), how many tokens each effort level
spends and how long it takes on Workers AI, Qwen3-Embedding's `{queries, instruction}` input
through the binding and the cosines it gives on the documentation (hence step 3 above), and
whether GLM-5.3-Flash describes pixel art better than Gemma 4 (hence step 6). The prices in
the table are the model pages' of October 2026.
