-- v4: one row per /ask question (spec §51-52): how it was routed, what it cost, how well its
-- claims were grounded, the versions behind the answer, and for a sample (SEARCH_TRACE_SAMPLE)
-- the whole trace: classification, subqueries, candidate counts, ranking and reranking scores,
-- evidence, claims, verification. GET /admin/ask/log and /admin/ask/trace/:qid read it; the
-- nightly prune keeps 120 days (search/feedback.ts).

CREATE TABLE IF NOT EXISTS ask_log (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  qid              TEXT    NOT NULL,
  q                TEXT    NOT NULL,
  lang             TEXT,
  class            TEXT,                   -- EXACT | FACTUAL | SEMANTIC | VISUAL | TEMPORAL | COMPARATIVE | AGGREGATION | MULTI_HOP | EXPLANATORY | AMBIGUOUS | UNKNOWN
  complexity       REAL,
  mode             TEXT,                   -- fast | balanced | deep | expert
  reasoning        TEXT,                   -- none | low | medium | high
  model            TEXT,                   -- the reasoning model, when one answered
  status           TEXT    NOT NULL,       -- answered | no_match | insufficient_evidence | conflict | clarify | not_found
  answer           TEXT,
  confidence       REAL,
  egs              REAL,                   -- evidence grounding score of the model's claims
  claims           INTEGER,
  claims_supported INTEGER,
  input_tokens     INTEGER,
  output_tokens    INTEGER,
  cost_usd         REAL,
  model_ms         INTEGER,
  took_ms          INTEGER,
  versions         TEXT,                   -- JSON: retrieval, index, ranker, reranker, planner, reasoning model, prompt
  trace            TEXT,                   -- JSON, for a sample only
  at               INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ask_log_at ON ask_log (at DESC);
CREATE INDEX IF NOT EXISTS ask_log_qid ON ask_log (qid);
CREATE INDEX IF NOT EXISTS ask_log_status ON ask_log (status, at DESC);

-- Feedback on answers (spec §36: explicit relevance, votes on an answer), next to /feedback's
-- search signals: the rows a reranker or a fine-tuning set can be built from later.
CREATE TABLE IF NOT EXISTS answer_feedback (
  id      INTEGER PRIMARY KEY AUTOINCREMENT,
  qid     TEXT    NOT NULL,
  rating  INTEGER NOT NULL,                -- 1 helpful, -1 not helpful
  reason  TEXT,                            -- wrong | unsupported | incomplete | other
  at      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS answer_feedback_qid ON answer_feedback (qid);

-- /help answers are verified sentence by sentence too (help/answer.ts): the mode they ran in, the
-- model, and the share of their sentences the documentation supports, next to v3's outcome.
ALTER TABLE help_log ADD COLUMN mode TEXT;
ALTER TABLE help_log ADD COLUMN model TEXT;
ALTER TABLE help_log ADD COLUMN egs REAL;
