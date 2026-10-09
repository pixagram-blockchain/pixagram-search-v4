-- Copy detection (PAPH-X, src/paph): the verifier's verdicts.
--
-- The fingerprints (PAPH wires, ~5–25 KB per artwork), the index keys and PAPH-SI postings over
-- them live in the PaphShard Durable Objects, where the verifier runs next to them. D1 keeps only
-- what the verifier concluded, so the API lists and filters verdicts next to posts. Written by
-- the "paph" enrichment stage (jobs.stage = 'paph').

-- content_hash the paph stage last completed for (fingerprint indexed AND every shard checked
-- AND verdicts written), like stats_hash / embed_hash / describe_hash, and the identity it
-- completed under (engine, profiles, policy): a new release or policy re-checks every artwork
ALTER TABLE artworks ADD COLUMN paph_hash TEXT;
ALTER TABLE artworks ADD COLUMN paph_engine TEXT;

CREATE TABLE IF NOT EXISTS paph_matches (
  a             INTEGER NOT NULL,               -- the earlier post (chain time, then id)
  b             INTEGER NOT NULL,               -- the later post
  a_hash        TEXT    NOT NULL,               -- the images the verdict is about (content hashes):
  b_hash        TEXT    NOT NULL,               -- listings skip a pair whose images changed since
  verdict       TEXT    NOT NULL,               -- Suspected | Copy | Identical
  state         INTEGER NOT NULL,               -- 2 | 3 | 4, for ordering and thresholds
  certifiable   INTEGER NOT NULL,               -- the comparator stands behind the verdict
  certificate   INTEGER NOT NULL,               -- the sparse geometry alone certified it (PAPH-X §9.4)
  structural_lo INTEGER NOT NULL,               -- structural agreement interval, 0..10000
  structural_hi INTEGER NOT NULL,
  geometry      INTEGER NOT NULL,               -- geometric evidence, 0..10000
  inliers       INTEGER NOT NULL,               -- keypoint correspondences in the agreeing models
  mirrored      INTEGER NOT NULL,               -- the geometry is a reflection
  execution     TEXT    NOT NULL,               -- FAST | DEFERRED | FALLBACK | AUDIT: how PAPH-X reached it
  rescued       INTEGER NOT NULL,               -- 0 (PAPH-X 1.1.0's ungated second pass, gone since 1.1.2;
                                                -- kept so a database migrated by the first release reads alike)
  same_author   INTEGER NOT NULL,
  via           TEXT    NOT NULL,               -- channels that nominated the pair: codes,bands,si,phash,vector,previous
  engine        TEXT    NOT NULL,               -- identity of the engine, profiles and policy that reached it
  computed      INTEGER NOT NULL,               -- unix seconds
  PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS paph_matches_b    ON paph_matches (b);
CREATE INDEX IF NOT EXISTS paph_matches_feed ON paph_matches (computed DESC);

-- A check some shard did not finish: the shards that did, so that the stage's retry asks only the
-- others (same image, same identity, same shard size; anything else starts over). Written in the
-- same batch as the verdicts it vouches for, and only while the post is live and shows that image;
-- deleted with the post's verdicts, and when the check completes.
CREATE TABLE IF NOT EXISTS paph_progress (
  post_id      INTEGER PRIMARY KEY,
  content_hash TEXT    NOT NULL,
  engine       TEXT    NOT NULL,
  shard_size   INTEGER NOT NULL,                -- PAPH_SHARD_SIZE the shard numbers refer to
  token        TEXT    NOT NULL,                -- which run started the row: a retry that continued
                                                -- from it writes only while it is still there
  done         TEXT    NOT NULL,                -- JSON array of the shards that answered in full
  updated      INTEGER NOT NULL
);
