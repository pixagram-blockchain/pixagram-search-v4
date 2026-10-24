#!/usr/bin/env bash
# Deploy the Worker alone, bound to the stack's EXISTING D1 database and KV namespace.
#
# wrangler.jsonc in the repository carries placeholders (REPLACE_D1_ID, REPLACE_KV_ID): the real
# ids belong to a Cloudflare account, not to the code. scripts/deploy.sh looks them up and writes
# them in before it deploys; this does the same and nothing else (no Space, no secrets, no
# backfill), for a code-only update. It never creates anything: when the database or the namespace
# is missing it stops (scripts/deploy.sh builds a stack from scratch).
#
#   npx wrangler login
#   scripts/deploy-worker.sh
#
# Overrides (env): CONFIG (wrangler.jsonc), CLOUDFLARE_ACCOUNT_ID (if your login has several accounts).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
CONFIG="${CONFIG:-wrangler.jsonc}"
WR=(npx --no-install wrangler)
# shellcheck source=scripts/lib.sh
source "$HERE/lib.sh"

step "Preflight"
for c in node npx python3; do command -v "$c" >/dev/null || die "$c is required"; done
[ -f "$CONFIG" ] || die "$CONFIG not found (run from the repository)"
[ -d node_modules/wrangler ] || { info "installing npm dependencies"; npm ci --no-audit --no-fund >/dev/null; }
WORKER=$(cfg name)
D1_NAME=$(cfg d1_databases.0.database_name)
KV_TITLE="${WORKER}-cache"
guard_name "Worker" "$WORKER"
guard_name "D1 database" "$D1_NAME"
CF_WHO=$("${WR[@]}" whoami --json 2>/dev/null) || die "not logged in to Cloudflare: run  npx wrangler login"
CF_ACCOUNTS=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("\n".join("{}  {}".format(a.get("id"), a.get("name")) for a in d.get("accounts") or []))' <<<"$CF_WHO")
if [ "$(grep -c . <<<"$CF_ACCOUNTS")" -gt 1 ] && [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  printf '%s\n' "$CF_ACCOUNTS" >&2
  die "several Cloudflare accounts: export CLOUDFLARE_ACCOUNT_ID=<id> and re-run"
fi
info "Worker $WORKER · Cloudflare: ${CLOUDFLARE_ACCOUNT_ID:-$(cut -c1-200 <<<"$CF_ACCOUNTS")}"

step "Ids of D1 $D1_NAME and KV $KV_TITLE"
D1_ID=$("${WR[@]}" d1 list --json 2>/dev/null | json_pick name "$D1_NAME" uuid)
[ -n "$D1_ID" ] || die "no D1 database named $D1_NAME in this account (scripts/deploy.sh creates the stack)"
KV_ID=$("${WR[@]}" kv namespace list 2>/dev/null | json_pick title "$KV_TITLE" id)
[ -n "$KV_ID" ] || die "no KV namespace titled $KV_TITLE in this account (scripts/deploy.sh creates the stack)"
patch_cfg d1 "$D1_ID"; info "database_id $D1_ID"
patch_cfg kv "$KV_ID"; info "KV id $KV_ID"

step "D1 migrations (only those not applied yet)"
"${WR[@]}" d1 migrations apply "$D1_NAME" --remote -c "$CONFIG" </dev/null

step "Deploy Worker $WORKER"
"${WR[@]}" deploy -c "$CONFIG"
