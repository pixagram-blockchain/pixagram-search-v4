-- Copy detection on @pixagram/paph-x 1.3.0: why each verdict was reached the way it was.
--
-- `execution` already says how (FAST, or FALLBACK: comparator 42 decided); `reason` says why — for a
-- fallback, why the cascade deferred: 'uncertified-geometry' (the sparse geometry would decide and
-- did not certify), 'certificate-unreachable' (a small or sparse work no certificate could hold
-- for: expected, never evidence against a copy), 'control-saturated', 'pool-truncated'; on the fast
-- path how it decided ('exact-evidence', 'bounded-evidence', 'route', 'identical'). NULL on verdicts
-- stored before 1.3.0 (the stage re-checks every work under the new identity, and writes it).
ALTER TABLE paph_matches ADD COLUMN reason TEXT;
