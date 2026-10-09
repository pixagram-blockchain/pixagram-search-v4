#!/usr/bin/env bash
# Freeze v3's answers on the test gallery (test/fixtures/v3-answers.json) from v3's own code: a
# git worktree of the last v3 commit, run against this repo's test harness. Re-run it when the
# test gallery (test/harness/corpus.ts) changes; the parity test (test/v4-v3-parity.test.ts)
# then checks that mode=v3 and /search still answer exactly as v3 did.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="${1:-16061e6}"
TREE="$PWD/.v3-tree"
rm -rf "$TREE"; git worktree prune
git worktree add --detach "$TREE" "$COMMIT" >/dev/null
trap 'git worktree remove --force "$TREE" >/dev/null 2>&1 || true' EXIT
ln -s "$PWD/node_modules" "$TREE/node_modules"
V3_TREE="$TREE" npx vitest run -c vitest.eval.config.ts eval/offline/v3-fixture.test.ts
echo "wrote test/fixtures/v3-answers.json from $COMMIT"
