#!/usr/bin/env bash
# Remove the v4 stack created by scripts/deploy.sh: the Worker (with its Durable Object and
# Workflow), D1 (all v4 data: posts, history, documentation, answer logs), KV, the three Vectorize
# indexes, the queues and the v4 Space. The v3 stack, production, v2, the shared R2 bucket and the
# documentation repository on GitHub are not touched (scripts/lib.sh refuses their names).
#
#   scripts/teardown.sh            # asks you to type the Worker name
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

CONFIG="${CONFIG:-wrangler.jsonc}"
SPACE_ID="${SPACE_ID:-primerz/pixagram-search-v4}"
SECRETS_FILE="${SECRETS_FILE:-$HOME/.pixagram-search-v4.json}"
WR=(npx --no-install wrangler)

# shellcheck source=scripts/lib.sh
source "$HERE/lib.sh"

WORKER=$(cfg name)
D1_NAME=$(cfg d1_databases.0.database_name)
KV_ID=$(cfg kv_namespaces.0.id)
VEC_NAME=$(vector_index VEC)
VEC_TEXT_NAME=$(vector_index VEC_TEXT)
VEC_DOCS_NAME=$(vector_index VEC_DOCS)
QUEUE=$(cfg queues.consumers.0.queue)
DLQ=$(cfg queues.consumers.0.dead_letter_queue)
WORKFLOW=$(cfg workflows.0.name)
R2_BUCKET=$(cfg r2_buckets.0.bucket_name)
guard_name "Worker" "$WORKER"
guard_name "D1 database" "$D1_NAME"
guard_vector_index "$VEC_NAME"
[ -z "$VEC_TEXT_NAME" ] || guard_vector_index "$VEC_TEXT_NAME"
[ -z "$VEC_DOCS_NAME" ] || guard_vector_index "$VEC_DOCS_NAME"
guard_name "queue" "$QUEUE"; guard_name "queue" "$DLQ"; guard_name "Workflow" "$WORKFLOW"
guard_name "Space" "$SPACE_ID"

# Same login checks as deploy.sh: with several Cloudflare accounts and no CLOUDFLARE_ACCOUNT_ID,
# every wrangler call below would fail (and used to be reported as "not found").
CF_WHO=$("${WR[@]}" whoami --json 2>/dev/null) || die "not logged in to Cloudflare: run  npx wrangler login"
CF_ACCOUNTS=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("\n".join("{}  {}".format(a.get("id"), a.get("name")) for a in d.get("accounts") or []))' <<<"$CF_WHO")
if [ "$(grep -c . <<<"$CF_ACCOUNTS")" -gt 1 ] && [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  printf '%s\n' "$CF_ACCOUNTS" >&2
  die "several Cloudflare accounts: export CLOUDFLARE_ACCOUNT_ID=<id> and re-run"
fi
python3 -c "import huggingface_hub" 2>/dev/null || die "pip install -U huggingface_hub (needed to delete the Space)"

step "This permanently deletes"
info "Worker $WORKER (Durable Objects ChainIndexer and PaphShard — the copy-detection index —, Workflow $WORKFLOW)"
info "D1 $D1_NAME · KV $KV_ID · Vectorize $VEC_NAME${VEC_TEXT_NAME:+, $VEC_TEXT_NAME}${VEC_DOCS_NAME:+, $VEC_DOCS_NAME} · queues $QUEUE, $DLQ"
info "Space $SPACE_ID"
info "Not touched: R2 $R2_BUCKET (shared), the v3, production and v2 Workers and Spaces"
read -r -p "Type the Worker name ($WORKER) to confirm: " answer
[ "$answer" = "$WORKER" ] || die "aborted"

FAILED=()
# try_delete <label> <cmd...>: "not found" is fine (already gone); any other error is a failure.
# Prompts get their default "yes" (stdin is not a TTY).
try_delete() {
  local label="$1"; shift
  local out
  if out=$("$@" </dev/null 2>&1); then info "$label: deleted"; return 0; fi
  if grep -Eqi "not found|does not exist|doesn.t exist|couldn.t find|could not find|no such|code: 10007|status 404|404 Client Error" <<<"$out"; then
    info "$label: not found (already gone)"; return 0
  fi
  info "$label: FAILED: $(tail -n 3 <<<"$out" | tr '\n' ' ' | cut -c1-300)"
  FAILED+=("$label")
}
# Order matters: the Worker is the consumer of the queue, so it goes first.
try_delete "Worker $WORKER" "${WR[@]}" delete -c "$CONFIG" --force
try_delete "Workflow $WORKFLOW" "${WR[@]}" workflows delete "$WORKFLOW"
try_delete "queue $QUEUE" "${WR[@]}" queues delete "$QUEUE"
try_delete "queue $DLQ" "${WR[@]}" queues delete "$DLQ"
try_delete "Vectorize $VEC_NAME" "${WR[@]}" vectorize delete "$VEC_NAME" --force
[ -z "$VEC_TEXT_NAME" ] || try_delete "Vectorize $VEC_TEXT_NAME" "${WR[@]}" vectorize delete "$VEC_TEXT_NAME" --force
[ -z "$VEC_DOCS_NAME" ] || try_delete "Vectorize $VEC_DOCS_NAME" "${WR[@]}" vectorize delete "$VEC_DOCS_NAME" --force
if [ -n "$KV_ID" ] && [ "$KV_ID" != "REPLACE_KV_ID" ]; then
  try_delete "KV $KV_ID" "${WR[@]}" kv namespace delete --namespace-id "$KV_ID" -y
fi
try_delete "D1 $D1_NAME" "${WR[@]}" d1 delete "$D1_NAME" -y
try_delete "Space $SPACE_ID" env SPACE_ID="$SPACE_ID" python3 -c \
  'import os; from huggingface_hub import HfApi; HfApi().delete_repo(os.environ["SPACE_ID"], repo_type="space", missing_ok=True)'

if [ "${#FAILED[@]}" -gt 0 ]; then
  die "not deleted: ${FAILED[*]}. Fix the cause and re-run; $CONFIG keeps its ids meanwhile."
fi
patch_cfg d1 REPLACE_D1_ID; patch_cfg kv REPLACE_KV_ID
# A Space recreated later must get our API_TOKEN secret again: forget that this one had it.
if [ -f "$SECRETS_FILE" ]; then
  python3 - "$SECRETS_FILE" "$SPACE_ID" <<'PY'
import json, sys
path, sid = sys.argv[1], sys.argv[2]
d = json.load(open(path))
if d.get("SPACE_SECRET_SET_ON") == sid:
    del d["SPACE_SECRET_SET_ON"]
    json.dump(d, open(path, "w"), indent=1)
PY
fi
step "Done: $CONFIG reset to placeholders; production and v2 untouched"
