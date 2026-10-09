-- v3: the platform documentation (the public repository DOCS_REPO, pixagram-blockchain/information)
-- for questions about Pixagram itself, and the log of those questions.
--
-- docs        one row per Markdown file of the repository (path = repository path)
-- doc_chunks  the files cut at their headings; FTS5 over title, heading and text, and one vector
--             per chunk in VEC_DOCS (id = doc_chunks.id) when that index is bound
-- help_log    every help question with its outcome: the questions the documentation does not
--             answer yet are the to-do list for the repository

CREATE TABLE IF NOT EXISTS docs (
  path    TEXT PRIMARY KEY,
  sha     TEXT NOT NULL,                  -- sha-256 of the indexed content (from the commit's archive)
  title   TEXT NOT NULL,
  lang    TEXT,
  url     TEXT NOT NULL,                  -- https://github.com/<repo>/blob/<branch>/<path>
  chunks  INTEGER NOT NULL DEFAULT 0,
  status  TEXT NOT NULL,                  -- indexed | skipped (draft, too large, not text) | failed
  error   TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  updated INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_chunks (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  path     TEXT NOT NULL,
  ord      INTEGER NOT NULL,              -- position in the file
  title    TEXT NOT NULL,                 -- the document's title (repeated so FTS can weigh it)
  heading  TEXT NOT NULL,                 -- heading path inside the document, "Marketplace › Royalties"
  anchor   TEXT NOT NULL DEFAULT '',      -- GitHub anchor of the section ("royalties"), '' for the top
  text     TEXT NOT NULL,
  lang     TEXT,
  hash     TEXT NOT NULL,                 -- of title, heading and text: unchanged chunks keep their vector
  embedded TEXT                           -- model of its vector in VEC_DOCS; NULL = none yet
);
CREATE INDEX IF NOT EXISTS doc_chunks_path ON doc_chunks (path, ord);
CREATE INDEX IF NOT EXISTS doc_chunks_embedded ON doc_chunks (embedded);

CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_fts USING fts5(
  title, heading, text,
  content='doc_chunks',
  content_rowid='id',
  tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS doc_chunks_ai AFTER INSERT ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(rowid, title, heading, text) VALUES (new.id, new.title, new.heading, new.text);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_ad AFTER DELETE ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, title, heading, text) VALUES ('delete', old.id, old.title, old.heading, old.text);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_au AFTER UPDATE OF title, heading, text ON doc_chunks BEGIN
  INSERT INTO doc_chunks_fts(doc_chunks_fts, rowid, title, heading, text) VALUES ('delete', old.id, old.title, old.heading, old.text);
  INSERT INTO doc_chunks_fts(rowid, title, heading, text) VALUES (new.id, new.title, new.heading, new.text);
END;

CREATE TABLE IF NOT EXISTS help_log (
  id       INTEGER PRIMARY KEY AUTOINCREMENT,
  q        TEXT NOT NULL,
  lang     TEXT,
  status   TEXT NOT NULL,                 -- answered | excerpts | not_found | no_docs | disabled
  top_path TEXT,
  score    REAL,
  at       INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS help_log_at ON help_log (at DESC);
CREATE INDEX IF NOT EXISTS help_log_status ON help_log (status, at DESC);
