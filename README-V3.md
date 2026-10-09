# pixagram-search v3: what changed and why

v3 rebuilds the search engine around three stages, as the v2 review recommended:
**retrieval** (several candidate legs, sized to the query), **ranking** (a feature-based scorer
whose features mean something on their own), and **reasoning** (an `/ask` endpoint that answers
questions from verified evidence). On top of them:

- **One search box.** `/query` sends each text to search, to `/ask`, or to `/help`.
- **Suggestions while typing.** `/suggest` completes words and questions, proposes titles,
  documentation sections and popular searches, and gives the box examples for its placeholder.
- **Questions about the platform.** `/help` answers them from the public repository
  [pixagram-blockchain/information](https://github.com/pixagram-blockchain/information).
- **SigLIP 2 NaFlex.** Images keep their aspect ratio.

It runs as its own stack next to production and v2: Worker `pixagram-search-v3` and Space
`primerz/pixagram-search-v3`, with its own D1, KV, three Vectorize indexes, queues, Durable
Object and Workflow. Only the content-addressed R2 bucket is shared. `scripts/deploy.sh` creates
and indexes everything; `scripts/teardown.sh` removes it.

## Results

All figures use the same 160 judged queries (`eval/queries.jsonl`, graded relevance, 8
categories) and the same corpus: the Pixa chain as of 4 October 2026, with 133 artworks,
15 blog posts and 168 historical versions.

| category | n | v2 live (deployed) | v2 ranking on the v3 index | v3, SigLIP 2 256 px | **v3, SigLIP 2 NaFlex** |
|---|---:|---:|---:|---:|---:|
| title | 36 | 0.825 | 0.967 | 1.000 | **1.000** |
| tag | 14 | 0.821 | 0.946 | 0.984 | **0.979** |
| visual | 45 | 0.697 | 0.864 | 0.928 | **0.930** |
| multilingual | 32 | 0.732 | 0.843 | 0.971 | **0.975** |
| typo | 13 | 0.790 | 0.813 | 0.991 | **0.992** |
| colour | 7 | 0.340 | 0.502 | 0.732 | **0.746** |
| author | 5 | 0.797 | 0.896 | 0.968 | **0.966** |
| ambiguous | 8 | 0.796 | 0.920 | 0.980 | **0.973** |
| **all (nDCG@10)** | 160 | **0.743** | **0.874** | **0.958** | **0.959** |
| MRR | | 0.795 | 0.901 | 0.991 | 0.988 |
| R@50 | | 0.912 | 0.990 | 0.993 | 0.993 |

How each column was measured:

- **v2 live** is `scripts/eval.py` against `pixagram-search-v2.p1x4.workers.dev`.
- The other columns come from the offline harness (`eval/offline/run.test.ts`). It runs the
  real v3 code on the chain snapshot, with SQLite in place of D1 and the model served locally
  by the Space's own code (`hf/siglip.py`).
- **v2 ranking on the v3 index** is v2's reciprocal rank fusion and popularity boost (`rank=rrf`)
  applied to v3's NaFlex index. It isolates what the ranker adds (0.874 → 0.959) from what
  better indexing adds (0.743 → 0.874).
- **v3, SigLIP 2 256 px** is the same code with v2's model, for comparison. v3 ships NaFlex.
- The offline harness runs without a vision model (`VLM_BACKEND=off`), so no AI captions are
  included. With Moondream captions, a deployed v3 should score at least as well.

`/ask` (`eval/ask.jsonl`) answers all **24/24** questions as expected, with either model. The
questions cover first/latest posts, counts, the most liked artwork, the most active artist,
FR/DE phrasing, a deleted-then-reposted image and black-and-white artworks.

### SigLIP 2 NaFlex

NaFlex keeps each image's aspect ratio. It spends a budget of 16×16-pixel patches on the image
instead of squashing it into a 256×256 square, so a 274×183 artwork is seen as about 19×13
patches. The budget matters. Image-only retrieval (nDCG@10 of the image vectors alone, 160
queries), by the image fed to the model:

| model | native | nearest | xBRZ | CPU per image (2 vCPU) |
|---|---:|---:|---:|---:|
| SigLIP 2, 256 px (v2's) | 0.830 | 0.838 | 0.841 | 0.25 s |
| NaFlex, 256 patches (its default) | 0.810 | 0.815 | 0.828 | 0.21 s |
| **NaFlex, 576 patches** | 0.830 | 0.825 | **0.845** | 0.53 s |
| NaFlex, 1024 patches | 0.823 | 0.824 | 0.843 | 1.07 s |

- At its default budget NaFlex is worse than the fixed 256 px model. At 576 patches it is
  slightly better; at 1024 it is no better and costs twice the CPU.
- **v3 ships 576 patches and the xBRZ view** (`EMBED_PATCHES`, `EMBED_VIEWS`). The Space and the
  Worker check each other's budget; changing it re-embeds through the sweeper.
- In the full pipeline the gain is small: 0.958 → 0.959 overall, colour 0.732 → 0.746.
- One `/ask` answer changed with NaFlex. For "When was Good vibes first posted?", it saw "good
  vibes" in another post, "Good Things Take Time". Two `/ask` rules now prevent that: a subject
  that is exactly a post's title names that post, and matching only some words of a
  several-word subject counts for less.

### Larger or other embedding models (5 October 2026)

This comparison uses the same harness and the same 160 queries on a fresh snapshot: 136
artworks, 15 blog posts and 173 versions. Each model indexes the whole snapshot and answers
through the full v3 pipeline. The "vectors only" columns rank by cosine alone, without the
ranker (`eval/offline/vectors_only.py`). For the image column the query is compared with each
artwork's image vector. For the text column it is compared with each artwork's title and tags.

| model | licence | dim | full pipeline nDCG@10 | image vectors only | text vectors only | index build, 2 CPUs |
|---|---|---:|---:|---:|---:|---:|
| **SigLIP 2 base, NaFlex 576 (v3)** | Apache 2.0 | 768 | **0.948** | **0.834** | 0.572 | 167 s |
| SigLIP 2 so400m, NaFlex 576 | Apache 2.0 | 1152 | 0.944 | 0.834 | 0.599 | 406 s |
| JinaCLIP v2 | CC BY-NC 4.0 | 1024 | 0.935 | 0.790 | 0.697 | 941 s |
| JinaCLIP v2, query instruction | CC BY-NC 4.0 | 1024 | 0.936 | 0.791 | 0.781 | (same index) |

- **so400m** is three times the size of base and scores the same: 0.834 on image vectors alone
  and 0.944 vs 0.948 in the full pipeline. Its index build takes 2.4 times as long.
- **JinaCLIP v2** matches artworks to queries less well than SigLIP 2, both on image vectors
  (0.790) and in the full pipeline (0.935).
  - Its text tower, from jina-embeddings-v3, is the better text-to-text model: 0.781 vs 0.572
    on titles and tags. Full-text search already covers most of that.
  - Its weights are non-commercial. Commercial use goes through Jina's API or the cloud
    marketplaces.
  - Its remote code needs transformers 4 (`eval/offline/jina_embedder.py`).
- `/ask` answers 24/24 with every model.
- 136 artworks is a small corpus. Rerun this when there are a few thousand: start the embed
  server with the model, then run the harness with the same `EMBED_MODEL` and its `EMBED_DIM`
  (1152 for so400m, 1024 for JinaCLIP; `EMBED_PATCHES=` for a model without NaFlex).

### Why v2 scored 0.743

Three causes, all fixed in v3:

1. **No AI descriptions at all.**
   - When Moondream's reply had no usable `answer`, v2 fell back to `JSON.stringify(reply)`.
     It then parsed what that gave as a description with every field empty: the reply's
     envelope (`{finish_reason, metrics, answer: null}`), or `{}` for a reply the AI binding had
     left as a stream (see below).
   - That empty description was stored as "done", so every artwork on the live stack has an
     empty caption.
   - v3 reads the task's own field (`answer` for query, `caption` for caption) and never the
     envelope. It rejects descriptions with no caption or tags, and falls back to Moondream's
     caption task. An empty result is an error, retried later. Descriptions that v2 marked done
     but left empty are redone.
   - The first v3 deployment showed why the replies were empty: every Moondream reply arrived as
     an object with no fields at all. The AI binding parses a reply only when its content type
     is exactly `application/json`, and otherwise returns the raw body stream, which looks just
     like that. v3 now reads such streams (`src/lib/ai.ts`): JSON, or server-sent events. A body
     that is neither is an error that quotes it, never a caption.
   - An empty description is no longer retried through the queue, where each retry is a paid
     call and a message is retried up to 8 times. The sweeper tries again 10 minutes later, up to
     40 times.
   - v3 now describes with Gemma 4 (`VLM_BACKEND=gemma`). In the first live comparison, on
     @retro's "Hello Kitty! Hail Satan!", Gemma described the scene in detail (the figures, the
     cliff, the spire, the sky). Scout gave one short line, an "isometric" style the image does
     not have, and the author's own tags. Moondream gave nothing usable. Gemma is also the
     cheapest of the three per token. "Pixel art" is dropped from styles and tags, since every
     artwork is pixel art.
2. **21 of 133 artworks without a vector.**
   - The queue gave up after 8 retries while the Space was unavailable, and nothing re-drove them.
   - v3 adds a sweeper (every 10 minutes). It re-enqueues every stage that is missing or stale:
     a new image, another `EMBED_MODEL`, other `EMBED_VIEWS`, an empty caption, or a text vector
     whose metadata changed.
3. **Popularity decided most orderings.**
   - v2 multiplied RRF scores by up to ~1.5 for votes and recency, while adjacent RRF ranks
     differ by ~1.6 %.
   - v3 bounds popularity to ±10 %, using a Bayesian vote rate per day of exposure. It can break
     near-ties but cannot override relevance.

## The review's recommendations, item by item

| recommendation | in v3 |
|---|---|
| Two-stage retrieve + rerank instead of pure RRF | `service.ts retrieve()` unions the full-text legs (AND, OR, spelling), the concept, colour and tone legs, image kNN and text kNN. `ranker.ts` scores each candidate on 11 features (exact title, bm25 per leg, query coverage, tag, author, concept, image z-score, text z-score, colour share, tone, orientation) × a bounded quality term × freshness, with near-duplicate diversity. `rank=rrf` keeps v2's ordering for comparison |
| Adaptive candidate depth | Deeper full-text legs for long queries. Time-sliced kNN widens image recall when restrictive filters leave too few survivors. `/ask` splits any time slice that comes back full (`adaptiveKnn`) and reports counts as lower bounds when its budget runs out |
| pHash radius bug | v2's eight 8-bit chunks only guaranteed radius ≤ 7, while the API defaulted to 8 and `/similar` used 16. v3 uses four 16-bit bands, exact up to radius 15 (pigeonhole), with an in-SQL popcount for anything larger. Both paths compute the exact distance in SQL, and are tested against brute force |
| Query planner | `planner.ts` covers intent, output, subject, concepts, authors, colours, background, tones, orientation, dates and object type, in EN/FR/DE/ES/IT. The `llm-planner.ts` fallback (Nemotron 3, `PLANNER_MODEL`) only runs when the rules are unsure; its JSON is validated against fixed vocabularies, and the LLM never produces answers |
| Evidence-based `/ask` | Candidates come from concepts, full text over every alias, adaptive image kNN and text kNN. Each one is verified by noisy-OR of lexical/concept evidence and z-scored image and text similarity, with a threshold relative to the best match. The answer is computed deterministically: first, last, count, top, compare, similar, duplicates |
| Multilingual concept / synonym index | `concepts/vocab.ts` has ~182 concepts with aliases in 8 languages and a parent hierarchy. `artwork_concepts` stores per-artwork concepts with a confidence per source (tags, title, description, VLM). `/concepts?q=` shows how a word maps |
| Hybrid visual similarity | `/similar` scores 0.65 SigLIP + 0.15 pHash + 0.08 dHash + 0.07 colour histogram + 0.05 geometry, over the union of the kNN and pHash neighbourhoods |
| Historical / version index | `post_versions` holds every top-level comment operation (from the tail and an account-history walk in the backfill), with its exact block, transaction, image SHA-256, pHash and colours. `image_since` and `first_seen` follow an image to its first appearance, including deleted originals and re-encoded re-uploads by the same author. `/history/:id` exposes it |
| Native vs upscaled embeddings | Measured on the corpus (image-only nDCG@10). With SigLIP 2 256 px: native .830, nearest-neighbour .838, xBRZ .841. With NaFlex at 576 patches: native .830, nearest .825, **xBRZ .845** (see SigLIP 2 NaFlex above). xBRZ stays the default; `EMBED_VIEWS` can mix views, and the sweeper re-embeds when it or `EMBED_PATCHES` changes |
| Learning to rank | Every search returns a `query_id`. `POST /feedback` records clicks, dwell, likes and saves (validated against the query's results). A sample of rankings is logged with features in `rank_log`. `scripts/train-ranker.py` fits weights from clicks with inverse-propensity weighting and cross-validation, and installs them through `/admin/ranker/weights`. Fitting on the editorial judgments (`eval/offline/fit_weights.py`) did not beat the hand-set defaults on held-out queries (0.9515 vs 0.9559), so the defaults ship |
| Rich colour / geometry descriptors | Brightness, contrast, saturation, colourfulness (Hasler–Süsstrunk), monochrome, edge density, symmetry, foreground share and centre, aspect, orientation, palette entropy, background colour, mean Lab, and hue and luminance histograms. Saturated blues and dark greens are now named correctly: v2's single swatch per name called `#0000ff` "purple" |
| Bayesian popularity | Votes per day over the 7-day payout window, shrunk toward the corpus rate with a 2-day prior. Bounded to [0.9, 1.1] |
| Expanded Vectorize metadata | 10 indexed properties per index: orientation and transparency added. The metadata NSFW flag is the same predicate as SQL (author flag or AI estimate ≥ 0.7). Metadata follows flag changes, listing changes and image changes |
| Explicit AI permission flags | `AI_TRAINING_FALSE_BLOCKS` names the stages (`describe`, `embed`, `text`) skipped when the PIXA license says `ai-training: false`. The flag is also a vector filter. The default keeps v2's behaviour |
| Restructured enrichment graph | Five idempotent stages (stats, embed, describe, concepts, text), each keyed on a hash and recorded per stage in `jobs`. Each stage writes its "done" hash only after its side effects. Embedding failures no longer block descriptions, or vice versa |
| Search-quality eval sets | `eval/queries.jsonl` (160 judged queries) and `eval/ask.jsonl` (24 questions). Run them live with `scripts/eval.py`, offline with `eval/offline/run.test.ts`, and rebuild the snapshot with `eval/offline/snapshot.py` |
| Spelling | A vocabulary with trigrams is maintained incrementally by indexing. Corrections run as an extra query variant through the text and vector legs, and never replace the query (`did_you_mean`). Words the concept vocabulary knows ("chien", "beach") are never "corrected" |

## The search box: `/query`

One text field, three destinations, chosen by rules (`src/search/router.ts`, no model call):

- **search**: words and titles. "First snow", "top hat" and "the last samurai" are titles, not
  questions.
- **ask**: questions about artworks, artists and their history.
- **help**: questions about the platform.

The rules, in order:

1. "Similar to 42" and "duplicates of 42" → ask.
2. Not a question → search. A question mark makes a question. So does a question word at the
   start, unless the text is the title of a post: "Why I draw cats" is a blog post, not a
   question.
3. An intent word ("first", "how many", "most liked", "who posted the most") about posts (an
   artwork subject, a posting verb, "post", or no subject at all: "what's the latest?"), with
   no platform word → ask. Advice ("the best time to post") is not about posts.
4. Platform words (fees, wallet, PXS, royalties, mint, account, android, team…), how-to and
   "can I" phrasing, or the name Pixagram without an artwork subject → help.
5. "What is …" → help when the documentation has the words; otherwise → search.
6. Any other question → ask when it is about artworks, help when the documentation has the
   words, otherwise → search.

A platform word can also be what an artwork shows ("wallet", "token", "android"). It is the
subject, not the topic, only when the question says the posts show it: a posting verb, or a post
noun it qualifies ("who posted the first wallet?", "how many android artworks?", "œuvres de
jetons"), or the people counted ("how many accounts are there?"). "Which wallet is the best?",
"fees per post?", "does it work on android?" and "how many accounts can I create?" go to help.
How-to and permission questions and policy words ("allowed") are always about the platform.
Hyphenated words count by their parts ("Android-App").

Every decision returns its `reason` and `signals`. When an answer comes back empty, the
response also carries search results, so the box never shows nothing. When a client's answer
budget is spent (20 per minute), questions get search results instead of a 429. About 195
routing cases in English, French and German are tests (`test/router.test.ts`,
`test/review-regressions-2.test.ts`, `test/review-regressions-3.test.ts`).

Three answer gaps surfaced here and are fixed:

- "How many artists…" counts people, not artworks.
- "Who posted the most cats?" and "the most active artist" ask for an author ranking. They
  used to be read as "who posted the first cat".
- A tie for the first place names everyone in it ("@alice, @bob and @carol posted the most
  artworks (2 each)."), with half the confidence; it used to name the first in alphabetical
  order alone.

## Suggestions while typing: `/suggest`

The box asks `/suggest` at every pause in typing (README: "GET /suggest"). Everything it proposes
leads somewhere, because it comes from what the index holds:

- **Completions** of the word being typed, best first:
  - the usual names of concepts in the reader's language ("dragon", "chat" for French), by the
    artworks behind them; from two letters, from memory;
  - from three letters, words posts carry: other names of a concept ("dragonfly", counted by the
    posts that say it), names in another language an artist used, the artists' words (titles,
    tags), and for English readers words the descriptions use in three posts or more. A word one
    description used once ("catastrophically") or a name in a language nobody wrote ("cabelo" for
    an English reader) is never proposed. Once the word is a subject already ("cat"), only the
    artists' words extend it.
  - after other words, only a word some post writes right after them ("red dra" → "red dragon",
    never "cat interior"); a function word after others ("cat in") is not completed.
  - accents as written for French and German names ("cha" → "château"; the vocabulary is
    folded).
- **Questions**: a question being typed gets its subject completed ("who posted the first dra"
  → "… dragon?", "combien de ch" → "combien de chats ?", "wie viele kat" → "wie viele Katzen?").
  A question begun gets whole questions that start with it: an opener ("who po") with the
  subjects the artists post most, the general questions, and the platform questions the
  documentation answers ("how do" → "how do royalties work?"); its last word is never completed
  out of the vocabulary. For a one-word subject, questions about it in English and French, only
  when the subject is clear: the word typed is its concept's usual name, or the only usual name
  the letters can still become ("dra" → dragon). Never another concept a word belongs to
  ("helmet" names hats, "dragonfly" insects), and never something one does not post "a first" of
  (hair, crypto, a country, a season).
- **Titles** (with a thumbnail), **documentation sections** (with their page) and, when the
  operator turns them on, **popular searches**.
- **A correction** when nothing else matches.
- **`completion`**: the first suggestion that extends the text, which the box shows as ghost
  text. The letters already typed stay as typed.
- **Examples** without text, for the placeholder, in the UI language: the subjects the artists
  tag and title most (the AI descriptions' "hair" is not a subject), questions about them, a
  colour with a thing ("black cat", never "black landscape"), titles and documentation questions.

Nothing a suggestion shows comes from a post `/search` hides by default (NSFW by the author's
flag or the AI estimate): titles, concept counts and corpus words are all counted over the posts
it shows.

Popular searches are shown to everybody, so they are off unless `SUGGEST_POPULAR=on`. The search
log cannot tell people apart (and one search in the box is two or three requests), so when they
are on, `/search` records one row per search, person and day (`query_people`, migration 0004):
the person is six hex characters of a SHA-256 of the day's random salt and the address (IPv6 by
its /64), which tells people apart within a day and never who they are; the salt is deleted the
day after (out of reach of anyone reading the data later; the account holder can restore it within
D1 Time Travel's window). A search becomes popular with three people on the same day, on two days within 30
days, in the default safe mode, made of words the index knows, and nothing that looks like an
e-mail, a link or a number. The independent review caught two earlier versions: one counted runs
(one person searching twice across an hour mark made a suggestion for everybody), the next
counted daily values across days (one address on three days was three people). Three addresses
can still plant a phrase of corpus words, which is why it is the operator's call.

Cost: D1 index lookups only, most in parallel, the corpus's words and title prefixes from the
third letter on (a shorter last word after others is checked on the rows found for those), with the counts per concept, the documentation's presence and the popular searches cached (KV
and the isolate). On a custom domain each answer is also cached a minute per location (the Cache
API does nothing on `workers.dev`). It has its own rate budget (`RL_SUGGEST`, 600 a minute), so
typing never uses up searching. 29 tests (`test/suggest.test.ts`, `test/suggest-sense.test.ts`),
including one that sends every question template through the router, and the inputs that were
nonsense on the live stack in October 2026, on a corpus with AI descriptions like production's.

## Questions about the platform: `/help`

The source is the public repository `pixagram-blockchain/information`. Today it holds a
README and a licence, so `/help` answers that no documentation is available yet. Everything
else is ready for its content.

**Sync** (`src/docs/sync.ts`): every 10 minutes, and at once through a push webhook.

- **Reading GitHub without its REST API.** Workers share their outbound addresses, and GitHub
  allows 60 anonymous API requests an hour per address, so an API-based sync would fail at
  random. The sync reads the branch head from git's ref advertisement (a few hundred bytes) and
  streams the commit's archive from codeload, the "Download ZIP" route. Only the Markdown
  files of the archive are kept.
- **Incremental.**
  - Files whose content changed are cut into sections.
  - A section whose text did not change keeps its row and its vector.
  - Files gone from the repository lose theirs.
  - A failure, or a run cut short by its budget, leaves the commit unrecorded, so the next run
    finishes the work.
  - A lock keeps the cron and the webhook apart.

**Writing for it** (`src/docs/markdown.ts`):

- **Sections.** Every heading of level 1–3 starts a section. An answer links to its exact
  place on GitHub (`…/fees.md#selling`), so one topic per section works best.
- **Links.** Links to web pages and e-mail addresses keep their address in the indexed text
  ("the wallet settings (https://…)"), so an answer can give it. Relative links keep their text.
- **Optional front matter:**
  - `title`
  - `lang`: otherwise taken from a `fr/` folder, a `.fr.md` suffix, or guessed from the text
  - `keywords: [a, b]`: words people search with
  - `draft: true`: leaves the page out
- **Excluded:** files over 512 KB, licences, and files in `.github/`.

**Answering** (`src/docs/answer.ts`):

- **Retrieval.** Full text (FTS5) plus multilingual vectors (bge-m3 on Workers AI, in
  `VEC_DOCS`), so a French question finds an English page.
- **No relevant section:** `not_found`, without a model call.
- **Otherwise:** Nemotron 3 120B (`HELP_MODEL`; JSON mode, reasoning off) writes at most five sentences from the numbered
  sections only, in the question's language, citing them. Citations to sections it was not
  given are removed.
- **Plain text out.** The answer becomes plain text. A link becomes "text (address)"; any
  address (URL, domain name, IPv4, e-mail, `mailto:`) that the sections do not contain is
  removed, after entities, backslash escapes and invisible characters are resolved; then the
  characters that links, autolinks, HTML and entities need are broken up. So a question cannot
  make the answer carry a link of its choosing, however the client shows it (text, a linkifier,
  Markdown with or without HTML). Tests render 2,000 random mixes of link syntax with
  markdown-it (raw HTML and linkify on) and check where every link goes.
- **Caching:** answers are cached per documentation commit, for a day; "not found" for 10
  minutes, since new sections take a moment to become searchable.
- **Gaps:** every question is logged; `GET /admin/docs/gaps` lists the unanswered ones, most
  asked first. That list is the repository's to-do list.

## Hardening before the first deployment

Three independent reviews of the v3 code, a verification of the third review's fixes, and later
checks found these defects. All are fixed, each with a regression test
(`test/review-regressions.test.ts`, `test/review-regressions-2.test.ts`,
`test/review-regressions-3.test.ts`, `test/indexer.test.ts`, `test/retrieval.test.ts`,
`test/planner.test.ts`):

- **Search results and cache.**
  - The result cache now uses a SHA-256 key and stores the request it answers. A forged 32-bit
    hash collision could serve an `nsfw=only` ranking to an ordinary request. The page is also
    re-filtered.
  - An exact all-words match was scored as an "OR" match.
  - `de <author>` was read as an author filter ("coucher de soleil" with an author named soleil).
  - Facets now cover all results, not only full-text matches.
  - A query made only of `@author` or `#tag` returns its posts instead of nothing.
- **Indexing state.**
  - An image edit that kept the byte length was never re-indexed.
  - Stages marked themselves done before their side effects (R2 original, history, full-text caption).
  - An older backfill snapshot could undo a newer edit or bring a deleted post back.
  - A recolour by the same author counted as the same artwork in `first_seen`.
  - Vector metadata drifted after the AI NSFW estimate or a flag change.
  - A markdown post with an inline picture was classified as an artwork.
- **`/ask`.**
  - "When did @bob first post a cat?" answered with someone else's earlier post.
  - Tone words were ignored for first, last and top questions.
  - "post" (the verb) was read as "blog post".
  - Counts were not flagged as lower bounds when a time slice was full.
  - A question naming a post by its title ("When was Good vibes first posted?") could be
    answered with another post that shares one word of it and looks alike to the image model.
  - "last week", "le mois dernier" and "letzte Woche" also read as "the latest one": "how many
    cats last month" answered with a single artwork. "last year" was not read as a date.
  - Removing filter words cut longer words: "the first red armored knight" looked for "armo knight".
  - A question mark alone did not mark a question, so unrecognised questions never reached the
    LLM fallback.
  - "best paid" was read as "best" (most liked), so it ranked by votes instead of payout.
- **Public endpoints.**
  - Request bodies are capped. Uploaded images are size-checked on the header before decoding: a
    75 KB PNG declaring 4096×4096 pixels used to allocate 136 MB.
  - Questions are limited to 300 characters, and callers can no longer force the LLM planner.
  - Feedback must match a real query and one of its results.
  - Ranking logs are sampled and bounded, and rate limits apply per client IP (Workers Rate
    Limiting bindings).
  - The embedding call has a timeout.
  - `/similar`, `/duplicates` and `/ask similar` exclude NSFW by default, like `/search`.
  - Deleted posts and their images are no longer served.
- **Search box and help** (the third review).
  - **Links in help answers.** A question could have the model write a link of its choosing;
    only `https://` links in plain form were checked.
    - Every link form is now reduced to text unless its URL is written in the documentation:
      titles, upper-case schemes, `//host`, `javascript:`, `mailto:`, `www.`, autolinks,
      reference definitions, HTML and e-mail addresses.
    - Citations keep only existing excerpts.
  - **Errors in answers.**
    - A question whose filters matched nothing ("the first #unicorn?") answered with a 500.
    - Any failing answer now falls back to search results.
  - **"Like N" read as "similar to post #N".** "Party like 1999" was taken for that, and the box
    showed nothing. Only explicit phrases ("similar to 42", "like #42") count now.
  - **Documentation sync.**
    - An empty or broken archive read as an empty repository: the index was wiped and the commit
      recorded. Tar headers are now checksummed, and the archive must be complete.
    - A sync cut short, then a branch moved back, left the index half-updated.
    - A forced re-index could not span several runs.
    - Branch names in UTF-8 broke the head lookup.
  - **Counting people.** In "how many people / members / Leute…", the counted word stayed in the
    subject, so the count was 0.
  - **Routing.**
    - "Cats on Pixagram?" and "Katzen auf Pixagram?" went to help.
    - "The first bridge / android / avatar", "how many accounts" went to help too.
    - "œuvre" (spelled with œ) was not an artwork word.
    - "what's" left a stray "s".
    - Policy questions ("is AI art allowed?") and French permission phrasing ("je peux", "on
      peut") were missed.
  - **Caching.**
    - A "not found" given while the documentation vectors were missing was cached for a day.
    - Two long questions with the same first 256 characters shared an answer.
  - A control character in a query broke full text (500).
- **Search box and help, second pass** (the verification of those fixes).
  - **Links in help answers, again.** Pattern-based link checks still let links through:
    a URL in parentheses, link text with nested or escaped brackets, code spans or line breaks,
    titles over two lines, reference definitions inside quotes or lists, and an e-mail removal
    that joined `]` to `(`. Answers are now plain text, with every address checked wherever it
    is and the link syntax broken up (see "Plain text out" above).
  - **The documentation's own links were dropped:** autolinks, links whose address has
    parentheses, e-mail autolinks, `mailto:` links, and "[2021]" read as a citation.
  - **Platform questions answered as artwork questions.** Platform words that are also artwork
    subjects (wallet, token, PXS, witness, fees, android) and counted accounts sent "which
    wallet is the best?", "how many tokens do I get?" and "how many accounts can I create?" to
    /ask, which answered with artworks.
  - **Post questions sent to search:** "what is the latest post?", "what's the latest?".
  - **Phones and the team:** "does pixagram work on android?" went to /ask, "who is on the
    team?" to search.
  - **"… on Pixagram":** "is it free to post on pixagram?" lost the platform's name and went
    to search.
  - **Policy words as subjects:** "who posted the first forbidden fruit?" went to help;
    "when was the last pixagram update?" went to /ask.
  - **Caching:** a "not found" given before new vectors became searchable was kept for a day.
  - **A second verification** of these fixes found:
    - "Fees per post?", "how many tokens do I get for a post?": a platform word next to "post"
      still counted as the artworks' subject, and /ask answered with an artwork.
    - "Gibt es eine Android-App?": hyphenated compounds hid the platform word.
    - "evil。com": ideographic and full-width full stops, which URL parsers read as dots, kept a
      domain in the answer as text.
    - Also fixed on the way: "Wie lösche ich einen Beitrag?" is a how-to question, and "which
      post has the most likes?" asks for the most liked post.
  - **A third verification** found:
    - The narrower rule lost artwork questions: "how many artworks show a wallet?", "first
      drawing of an old iphone?", "first pixel art android?", "quelle est la dernière œuvre
      android ?" went to help. What a post shows is now read through "of / with / showing /
      depicting" (and FR/DE/ES/IT), a word or two apart, and after an artwork word for things
      that are not money ("pixel art android", but "artwork fees" stays a platform question).
    - The full-stop pattern removed the documentation's own address from Chinese and Japanese
      answers ("pixagram.com。Pixagram"), and a URL swallowed the sentence after it.
  - **A fourth verification** found:
    - The wider reading took the user's own things and the platform's products for what posts
      show: "artworks with my wallet?", "Bilder von meinem iPhone…", "is there a pixel art
      mobile version?" went to /ask. A possessive or a product word (version, app, site…) now
      keeps them platform questions.
    - In Chinese, Japanese and Korean answers, a URL without a path followed by "。" or "，", and
      an address followed by a Korean particle ("pixagram.com에서"), were removed with the rest
      of the sentence.
  - In a question, "fee" is the platform's word; the fairy ("fée", folded to the same letters)
    only with its French accent or as a German noun inside a sentence ("Bilder mit einer Fee").
    The vocabulary keeps the fairy for search and indexing.
- **Chain tail.** A `stop()` or `start(from)` that landed during a tick was overwritten by the tick.
- **Scripts.**
  - The deploy wait loop died on one HTTP 500.
  - The deploy wait loop stopped as soon as the backfill had queued its work: stats and vectors
    are written in the same queue message, so "every artwork has a vector" was true from the
    first artwork. It then sampled the background vectors and ran the evaluation on a corpus
    that was mostly still queued. It now waits for the queue.
  - Teardown reported "not found" for any error, then reset the config.
  - A recreated Space could stay without its API token.
  - Tokens appeared in process arguments.

## Deploying

```bash
pip install -U huggingface_hub && hf auth login      # token with write access
npx wrangler login                                    # the Workers Paid account
scripts/deploy.sh
```

The script runs these steps, all idempotent:

1. Checks the logins and refuses any production or v2 resource name.
2. Creates or updates the Space `primerz/pixagram-search-v3`:
   - model `google/siglip2-base-patch16-naflex`, with `MAX_NUM_PATCHES` = `EMBED_PATCHES`;
   - an `API_TOKEN` secret.
3. Creates D1, KV, the queues and the DLQ, and three Vectorize indexes:
   - images and texts, 768 dimensions, with their 10 metadata indexes each;
   - documentation, 1024 dimensions.
4. Applies the migrations, deploys the Worker and sets its secrets: `ADMIN_TOKEN`, `HF_TOKEN`,
   and a generated `GITHUB_WEBHOOK_SECRET`.
5. Waits for the Space (right model and patch budget). Then starts the backfill (posts, then
   each author's edit history), the live tail, and the first documentation sync.
6. Waits until the enrichment queue is empty (backfill complete, no job queued, nothing changing
   for 2.5 minutes), then takes the first background sample and evaluates against v2
   (`scripts/eval.py --compare`) and on `/ask`. If indexing takes longer than 40 minutes, it
   skips both and prints the commands to run later. Failed stages are listed by
   `scripts/admin.sh failed`.
7. Prints how to add the push webhook to the documentation repository, which is optional: the
   10-minute cron syncs it anyway. Adding a webhook takes admin rights on the repository, so the
   script does not do it.

The tokens are kept in `~/.pixagram-search-v3.json`. `SKIP_WAIT=1` stops after the deploy;
`OLD_BASE` picks what to compare against.

Things to know:

- **R2 is shared** with production and v2. Keys are content hashes, objects are written only
  when missing and never deleted.
- **Rate limits** use namespaces `3101` and `3102` (`wrangler.jsonc → ratelimits`). Change them
  if your account already uses those ids.
- **Migration 0002 also applies in place** on a v2 database. After that, run
  `scripts/admin.sh vocab-rebuild` once: older documents predate the spelling vocabulary.
- **Teardown:** `scripts/teardown.sh` asks you to type the Worker name. It stops and keeps the
  config ids if any deletion fails for a reason other than "not found".

## Operating it

| task | how |
|---|---|
| status | `scripts/admin.sh stats` (counts per stage, history, indexer, config) |
| failed stages | `scripts/admin.sh failed`; the sweeper retries missing/stale stages every 10 min, up to 40 consecutive failures |
| change the embedding views or model | set `EMBED_VIEWS` / `EMBED_MODEL`, deploy; the sweeper re-embeds, then `scripts/admin.sh background` |
| compare VLMs on one artwork | `scripts/admin.sh describe <post id> moondream`, then `… scout`, `… gemma` |
| index faster | upgrade the Space's hardware (Settings → Space hardware): it computes one embedding per CPU at once, and the Worker sees the new number within ten minutes (`/health` → `concurrency`) |
| change the VLM | `VLM_BACKEND`, deploy, `scripts/admin.sh reindex-all describe` |
| compare help models | `scripts/admin.sh help-ask "<question>" @cf/nvidia/nemotron-3-120b-a12b` (any Workers AI model id), then set `HELP_MODEL` |
| new concepts / aliases | edit `src/concepts/vocab.ts`, deploy, `scripts/admin.sh reindex-all concepts` |
| words people search that no concept knows | `scripts/admin.sh queries` → `words_without_concept` |
| learn ranker weights from clicks | `python3 scripts/train-ranker.py --base $BASE --token $ADMIN_TOKEN`, then `scripts/admin.sh weights-set weights.json` (revert: POST `{"reset": true}`) |
| replay the tail | `scripts/admin.sh start <block>` |
| edit history for one author | `scripts/admin.sh backfill-history <author>` |
| documentation status / sync now | `scripts/admin.sh docs`, `scripts/admin.sh docs-sync [force]` |
| what the documentation cannot answer yet | `scripts/admin.sh docs-gaps [days]` |
| instant documentation updates | GitHub → information → Settings → Webhooks: payload URL `<worker>/webhooks/github`, content type `application/json`, secret `GITHUB_WEBHOOK_SECRET` from `~/.pixagram-search-v3.json`, event "push" |
| change the documentation embedding model | `DOCS_EMBED_MODEL` (same dimensions as `VEC_DOCS`), deploy, `scripts/admin.sh docs-reembed` |
| see where the search box sends a text | `scripts/admin.sh query "how do I mint?"` |

## Known limits

- **Corpus size.**
  - `/admin/reindex all` and the sweeper write job rows 100 per batch. One request stays under
    the 1,000-subrequest limit up to roughly 10,000 posts × 5 stages; beyond that, reindex
    per author.
  - The background sample (256 vectors) and the concept vocabulary are sized for thousands of
    artworks, not millions.
- **Concepts.** Multilingual matching depends on the vocabulary. Unknown words still work
  through full text and SigLIP, which is multilingual. `words_without_concept` lists the
  candidates to add.
- **Colour queries** are the weakest category (0.746). Pixel-art palettes are small, and a
  colour word can name the subject ("red dragon") or the palette ("red").
- **Learning to rank** needs real click data. The pipeline is in place, but no learned weights
  ship.
- **Rate limits** are per Cloudflare location and approximate by design (Workers Rate Limiting).
- **Help answers** are only as good as the repository. Until it has content, `/help` says the
  documentation is not available yet, and the search box shows search results with that note.
- **Help answers in Chinese, Japanese and Korean**: an address written against the text
  ("请访问pixagram.com", "pixagram.com에서") gets a space on that side, so it ends where the
  sentence resumes. Two Latin words joined by an ideographic full stop ("PXS。Wallet") read as a
  domain name and are removed.
- **Routing** is rule-based, and a few texts are ambiguous. "How to draw a cat" goes to help, and
  falls back to search results when the documentation has no answer. The `reason` and `signals`
  of each decision are returned, so the UI can offer the other destination.
- **Not verified live** (no Cloudflare credentials here):
  - the Workers AI calls (bge-m3, the help model's JSON mode) and the `VEC_DOCS` index;
  - codeload, which this environment's proxy does not reach.

  The archive reader is tested on real `git archive` output, and the ref parser on a real
  ref advertisement of the repository.
