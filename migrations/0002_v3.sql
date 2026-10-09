-- pixagram-search v3: additions on top of the v2 schema (0001_init.sql).
-- Applies to a fresh v3 database and, in place, to an existing v2 database. After an in-place
-- upgrade run `scripts/admin.sh reindex-all` once: the stats stage fills the new feature columns
-- and the pHash bands, and the later stages fill concepts and text vectors.

-- ---- artworks: image features (native image, computed with the stats stage) -------------------
ALTER TABLE artworks ADD COLUMN dhash TEXT;                 -- 64-bit difference hash, 16 hex chars
ALTER TABLE artworks ADD COLUMN phash_hi INTEGER;           -- pHash as two unsigned 32-bit halves, for the
ALTER TABLE artworks ADD COLUMN phash_lo INTEGER;           -- exact full-scan Hamming search (radius > 15)
ALTER TABLE artworks ADD COLUMN brightness REAL;            -- mean luminance of opaque pixels, 0..1
ALTER TABLE artworks ADD COLUMN contrast REAL;              -- luminance standard deviation, 0..0.5
ALTER TABLE artworks ADD COLUMN saturation REAL;            -- mean HSV saturation, 0..1
ALTER TABLE artworks ADD COLUMN colorfulness REAL;          -- Hasler–Süsstrunk M, ~0 for greyscale
ALTER TABLE artworks ADD COLUMN monochrome INTEGER;         -- 1 = greyscale or a single hue (sepia, duotone)
ALTER TABLE artworks ADD COLUMN edge_density REAL;          -- share of pixels on a colour edge
ALTER TABLE artworks ADD COLUMN symmetry_x REAL;            -- left/right mirror agreement, 0..1
ALTER TABLE artworks ADD COLUMN symmetry_y REAL;            -- top/bottom mirror agreement, 0..1
ALTER TABLE artworks ADD COLUMN foreground_share REAL;      -- opaque, non-background pixels / all pixels
ALTER TABLE artworks ADD COLUMN center_x REAL;              -- foreground centre of mass, 0..1
ALTER TABLE artworks ADD COLUMN center_y REAL;
ALTER TABLE artworks ADD COLUMN aspect REAL;                -- width / height
ALTER TABLE artworks ADD COLUMN orientation TEXT;           -- portrait | landscape | square
ALTER TABLE artworks ADD COLUMN palette_entropy REAL;       -- bits, over the ΔE-clustered palette
ALTER TABLE artworks ADD COLUMN background_name TEXT;       -- named colour of the backdrop, or 'transparent'
ALTER TABLE artworks ADD COLUMN background_share REAL;
ALTER TABLE artworks ADD COLUMN lab_l REAL;                 -- mean CIELAB of the counted pixels
ALTER TABLE artworks ADD COLUMN lab_a REAL;
ALTER TABLE artworks ADD COLUMN lab_b REAL;
ALTER TABLE artworks ADD COLUMN features_json TEXT;         -- everything above plus histograms, for the API
ALTER TABLE artworks ADD COLUMN features_hash TEXT;         -- content_hash the features were computed for

-- ---- artworks: semantic layer -------------------------------------------------------------------
ALTER TABLE artworks ADD COLUMN embed_views TEXT;           -- which image views the vector mixes, e.g. "xbrz"
ALTER TABLE artworks ADD COLUMN ai_objects_json TEXT;
ALTER TABLE artworks ADD COLUMN ai_status TEXT;             -- ok | caption_only | empty | error
ALTER TABLE artworks ADD COLUMN ai_raw TEXT;                -- last raw VLM reply (truncated), for debugging
ALTER TABLE artworks ADD COLUMN concepts_hash TEXT;         -- fingerprint of the inputs the concepts were derived from

-- ---- artworks: history (see post_versions) ---------------------------------------------------------
ALTER TABLE artworks ADD COLUMN image_since INTEGER;        -- first time this post showed its current image
ALTER TABLE artworks ADD COLUMN first_seen INTEGER;         -- first time these image bytes appeared on chain, any post
ALTER TABLE artworks ADD COLUMN first_seen_author TEXT;     -- ... and where (may be a post deleted since)
ALTER TABLE artworks ADD COLUMN first_seen_permlink TEXT;
ALTER TABLE artworks ADD COLUMN first_seen_match TEXT;      -- exact (same bytes) | near (same author, pHash <= 4, same colours) | self
ALTER TABLE artworks ADD COLUMN history_exact INTEGER;      -- 1 = from chain operations, 0 = inferred from created/updated

CREATE INDEX IF NOT EXISTS artworks_orientation ON artworks (orientation);
CREATE INDEX IF NOT EXISTS artworks_first_seen ON artworks (first_seen);

-- ---- posts: text vector bookkeeping (artworks and blogs) -----------------------------------------
ALTER TABLE posts ADD COLUMN text_hash TEXT;                -- hash of the text the vector was computed from
ALTER TABLE posts ADD COLUMN text_model TEXT;

-- ---- pHash multi-index hashing ----------------------------------------------------------------------
-- Four 16-bit bands. If two hashes are within Hamming distance r, one band differs in at most
-- floor(r / 4) bits (pigeonhole), so enumerating that band's neighbours finds every match for
-- r <= 15 with indexed equality lookups. v2's eight 8-bit chunks only guaranteed r <= 7 while the
-- API defaulted to 8 (and /similar used 16).
CREATE TABLE IF NOT EXISTS phash_bands (
  post_id INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  band    INTEGER NOT NULL,                     -- 0..3, 0 = most significant 16 bits
  val     INTEGER NOT NULL,                     -- 0..65535
  PRIMARY KEY (post_id, band)
);
CREATE INDEX IF NOT EXISTS phash_bands_lookup ON phash_bands (band, val);
DROP TABLE IF EXISTS phash_chunks;

-- ---- concepts ------------------------------------------------------------------------------------
-- Canonical concept ids (src/concepts/vocab.ts: "cat", "mountain", "woman", ...) per artwork, from
-- the author's tags and title and from the AI description, with their parents ("cat" → "animal").
CREATE TABLE IF NOT EXISTS artwork_concepts (
  post_id    INTEGER NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
  concept    TEXT    NOT NULL,
  confidence REAL    NOT NULL,                  -- 0..1
  source     TEXT    NOT NULL,                  -- tag | title | description | vlm | parent
  PRIMARY KEY (post_id, concept)
);
CREATE INDEX IF NOT EXISTS artwork_concepts_concept ON artwork_concepts (concept, confidence DESC);

-- ---- post history -------------------------------------------------------------------------------
-- One row per top-level comment operation (create, edit, delete-by-edit), from the live tail
-- (exact), from account history during the backfill (exact), or a snapshot when neither is known.
CREATE TABLE IF NOT EXISTS post_versions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  author       TEXT    NOT NULL,
  permlink     TEXT    NOT NULL,
  block_num    INTEGER NOT NULL,                -- 0 for snapshots
  trx_id       TEXT    NOT NULL,                -- 'snapshot' for snapshots
  op_in_trx    INTEGER NOT NULL,
  at           INTEGER NOT NULL,                -- block time, unix seconds
  kind         TEXT    NOT NULL,                -- create | edit | delete
  body_kind    TEXT    NOT NULL,                -- image | text | patch | deleted
  content_hash TEXT,                            -- sha256 of the decoded image (body_kind = image)
  phash        TEXT,                            -- pHash of that image, to link near-identical re-uploads
  buckets_json TEXT,                            -- its named-colour shares: pHash ignores colour, a recolour is a new artwork
  mime         TEXT,
  title        TEXT,
  source       TEXT    NOT NULL,                -- tail | history | snapshot
  UNIQUE (author, permlink, block_num, trx_id, op_in_trx)
);
CREATE INDEX IF NOT EXISTS post_versions_post ON post_versions (author, permlink, at);
CREATE INDEX IF NOT EXISTS post_versions_hash ON post_versions (content_hash, at);

-- ---- vocabulary (spelling suggestions) ----------------------------------------------------------
-- Document frequency of every token of titles, tags, descriptions, AI tags/captions and authors,
-- maintained incrementally by writeSearchDoc, plus its character trigrams for candidate lookup.
CREATE TABLE IF NOT EXISTS vocab (
  term TEXT PRIMARY KEY,
  df   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS vocab_grams (
  gram TEXT NOT NULL,
  term TEXT NOT NULL,
  PRIMARY KEY (gram, term)
);

-- ---- query log, ranking log, feedback (learning to rank) ---------------------------------------
ALTER TABLE query_log ADD COLUMN qid TEXT;                  -- returned to the client as query_id
ALTER TABLE query_log ADD COLUMN mode TEXT;
ALTER TABLE query_log ADD COLUMN plan_json TEXT;
ALTER TABLE query_log ADD COLUMN top_json TEXT;             -- first 20 result ids
CREATE INDEX IF NOT EXISTS query_log_qid ON query_log (qid);

-- Candidate features at serving time, for the shown results (sampled by RANK_LOG_SAMPLE).
CREATE TABLE IF NOT EXISTS rank_log (
  qid      TEXT    NOT NULL,
  post_id  INTEGER NOT NULL,
  rank     INTEGER NOT NULL,                    -- 1-based position served
  features TEXT    NOT NULL,                    -- JSON {name: value}
  at       INTEGER NOT NULL,
  PRIMARY KEY (qid, post_id)
);
CREATE INDEX IF NOT EXISTS rank_log_at ON rank_log (at);

CREATE TABLE IF NOT EXISTS feedback (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  qid      TEXT,
  post_id  INTEGER NOT NULL,
  rank     INTEGER,
  action   TEXT    NOT NULL,                    -- click | open | like | save | similar | dwell
  dwell_ms INTEGER,
  at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS feedback_qid ON feedback (qid);
CREATE INDEX IF NOT EXISTS feedback_at ON feedback (at);
