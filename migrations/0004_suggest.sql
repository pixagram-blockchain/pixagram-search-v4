-- v3: who ran which search, for the popular searches /suggest may show (SUGGEST_POPULAR=on).
--
-- One row per (search, client, day), written by /search for a page-one text search in the default
-- safe mode that found something. `client` is six hex characters of a SHA-256 of the day's random
-- salt and the client's address (IPv6 by its /64): it tells people apart within a day, never who
-- they are (about 256 IPv4 addresses share each value; the salt is deleted the day after; only
-- D1 Time Travel, for the account holder, can bring one back). A
-- search is popular with three people on one day, on two days within 30 days (search/suggest.ts).
-- Rows older than 30 days are pruned nightly (search/feedback.ts).

CREATE TABLE IF NOT EXISTS query_people (
  qn      TEXT    NOT NULL,              -- the search, folded (lower case, no accents, single spaces)
  q       TEXT    NOT NULL,              -- as shown: lower case, single spaces
  client  TEXT    NOT NULL,
  day     INTEGER NOT NULL,              -- unix day
  PRIMARY KEY (qn, client, day)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS query_people_day ON query_people (day);
