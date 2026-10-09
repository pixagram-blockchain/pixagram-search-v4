#!/usr/bin/env bash
# Create only the Cloudflare resources named in wrangler.jsonc (D1, KV, the three Vectorize indexes
# with their metadata indexes, queues, R2) and write the D1/KV ids into the config. For a manual
# setup; scripts/deploy.sh does this and everything else (Space, migrations, deploy, backfill).
# Requires: npx wrangler login on a Workers Paid account.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
CONFIG="${CONFIG:-wrangler.jsonc}"
WR=(npx --no-install wrangler)
# shellcheck source=scripts/lib.sh
source "$HERE/lib.sh"

WORKER=$(cfg name); D1_NAME=$(cfg d1_databases.0.database_name); EMBED_DIM=$(cfg vars.EMBED_DIM)
VEC_NAME=$(vector_index VEC); VEC_TEXT_NAME=$(vector_index VEC_TEXT); VEC_DOCS_NAME=$(vector_index VEC_DOCS)
DOCS_EMBED_DIM=$(cfg vars.DOCS_EMBED_DIM)
QUEUE=$(cfg queues.consumers.0.queue); DLQ=$(cfg queues.consumers.0.dead_letter_queue)
R2_BUCKET=$(cfg r2_buckets.0.bucket_name); KV_TITLE="${WORKER}-cache"
guard_name "Worker" "$WORKER"; guard_name "D1 database" "$D1_NAME"; guard_vector_index "$VEC_NAME"
[ -z "$VEC_TEXT_NAME" ] || guard_vector_index "$VEC_TEXT_NAME"
[ -z "$VEC_DOCS_NAME" ] || guard_vector_index "$VEC_DOCS_NAME"
guard_name "queue" "$QUEUE"; guard_name "queue" "$DLQ"

step "D1 $D1_NAME"
run_ok "D1 $D1_NAME" "${WR[@]}" d1 create "$D1_NAME"
D1_ID=$("${WR[@]}" d1 list --json 2>/dev/null | json_pick name "$D1_NAME" uuid)
[ -n "$D1_ID" ] && patch_cfg d1 "$D1_ID" && info "database_id $D1_ID written to $CONFIG"

step "KV $KV_TITLE"
run_ok "KV $KV_TITLE" "${WR[@]}" kv namespace create "$KV_TITLE"
KV_ID=$("${WR[@]}" kv namespace list 2>/dev/null | json_pick title "$KV_TITLE" id)
[ -n "$KV_ID" ] && patch_cfg kv "$KV_ID" && info "KV id $KV_ID written to $CONFIG"

step "R2 $R2_BUCKET (shared) · queues $QUEUE, $DLQ"
run_ok "R2 $R2_BUCKET" "${WR[@]}" r2 bucket create "$R2_BUCKET"
for q in "$QUEUE" "$DLQ"; do run_ok "queue $q" "${WR[@]}" queues create "$q"; done

# Metadata indexes must exist BEFORE vectors are inserted (max 10 per index).
for pair in "$VEC_NAME:image" "${VEC_TEXT_NAME:+$VEC_TEXT_NAME:text}"; do
  [ -n "$pair" ] || continue
  name="${pair%%:*}"; kind="${pair##*:}"
  step "Vectorize $name ($kind, $EMBED_DIM-d, cosine)"
  run_ok "Vectorize $name" "${WR[@]}" vectorize create "$name" --dimensions="$EMBED_DIM" --metric=cosine
  if [ "$kind" = image ]; then specs=("${IMAGE_METADATA_INDEXES[@]}"); else specs=("${TEXT_METADATA_INDEXES[@]}"); fi
  for spec in "${specs[@]}"; do
    run_ok "  metadata index ${spec%%:*}" "${WR[@]}" vectorize create-metadata-index "$name" --property-name="${spec%%:*}" --type="${spec##*:}"
  done
done

if [ -n "$VEC_DOCS_NAME" ]; then
  step "Vectorize $VEC_DOCS_NAME (documentation, ${DOCS_EMBED_DIM:?set vars.DOCS_EMBED_DIM}-d, cosine)"
  run_ok "Vectorize $VEC_DOCS_NAME" "${WR[@]}" vectorize create "$VEC_DOCS_NAME" --dimensions="$DOCS_EMBED_DIM" --metric=cosine
fi

step "Next"
info "npx wrangler d1 migrations apply $D1_NAME --remote"
info "npx wrangler deploy"
info "npx wrangler secret put ADMIN_TOKEN; npx wrangler secret put HF_TOKEN   # the Space's API_TOKEN"
info "optional: npx wrangler secret put GITHUB_WEBHOOK_SECRET   # enables the documentation repository's push webhook"
info "then: POST /admin/docs/sync (the documentation), POST /admin/backfill, POST /admin/indexer/start"
