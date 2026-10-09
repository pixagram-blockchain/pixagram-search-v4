#!/usr/bin/env bash
# Deploy the pixagram-search v4 stack: its own Worker (wrangler.jsonc) and its own Hugging Face
# Space running SigLIP 2 NaFlex (the same model as v3). The v3 stack (Worker pixagram-search-v3,
# which the Pixagram UI uses; Space primerz/pixagram-search-v3), production and v2 are never
# touched. Safe to re-run: every step checks what exists.
#
#   pip install -U huggingface_hub && hf auth login     # an HF token with write access
#   npx wrangler login                                   # the Workers Paid account
#   scripts/deploy.sh
#
# Overrides (env): SPACE_ID (primerz/pixagram-search-v4), CONFIG (wrangler.jsonc),
#   OLD_BASE (stack to compare against at the end; default the v3 Worker),
#   SECRETS_FILE (~/.pixagram-search-v4.json), CLOUDFLARE_ACCOUNT_ID (only if your Cloudflare
#   login has several accounts), SKIP_WAIT=1 (deploy only; skip indexing and the evaluation)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."

CONFIG="${CONFIG:-wrangler.jsonc}"
SPACE_ID="${SPACE_ID:-primerz/pixagram-search-v4}"
OLD_BASE="${OLD_BASE:-https://pixagram-search-v3.p1x4.workers.dev}"
SECRETS_FILE="${SECRETS_FILE:-$HOME/.pixagram-search-v4.json}"
WR=(npx --no-install wrangler)

# shellcheck source=scripts/lib.sh
source "$HERE/lib.sh"

# ---- 0. preflight -------------------------------------------------------------------------------
step "0/10 Preflight"
for c in node npx python3 curl; do command -v "$c" >/dev/null || die "$c is required"; done
[ -f "$CONFIG" ] || die "$CONFIG not found (run from the repository)"
[ -d node_modules/wrangler ] || { info "installing npm dependencies"; npm ci --no-audit --no-fund >/dev/null; }

WORKER=$(cfg name)
D1_NAME=$(cfg d1_databases.0.database_name)
VEC_NAME=$(vector_index VEC)
VEC_TEXT_NAME=$(vector_index VEC_TEXT)
VEC_DOCS_NAME=$(vector_index VEC_DOCS)
QUEUE=$(cfg queues.consumers.0.queue)
DLQ=$(cfg queues.consumers.0.dead_letter_queue)
WORKFLOW=$(cfg workflows.0.name)
R2_BUCKET=$(cfg r2_buckets.0.bucket_name)
MODEL_ID=$(cfg vars.EMBED_MODEL)
EMBED_DIM=$(cfg vars.EMBED_DIM)
EMBED_PATCHES=$(cfg vars.EMBED_PATCHES)
DOCS_EMBED_DIM=$(cfg vars.DOCS_EMBED_DIM)
DOCS_REPO=$(cfg vars.DOCS_REPO)
KV_TITLE="${WORKER}-cache"
[ -n "$VEC_NAME" ] || die "$CONFIG has no VEC binding"
guard_name "Worker" "$WORKER"
guard_name "D1 database" "$D1_NAME"
guard_vector_index "$VEC_NAME"
[ -z "$VEC_TEXT_NAME" ] || guard_vector_index "$VEC_TEXT_NAME"
[ "$VEC_TEXT_NAME" != "$VEC_NAME" ] || die "VEC and VEC_TEXT must be different indexes"
if [ -n "$VEC_DOCS_NAME" ]; then
  guard_vector_index "$VEC_DOCS_NAME"
  [ "$VEC_DOCS_NAME" != "$VEC_NAME" ] && [ "$VEC_DOCS_NAME" != "$VEC_TEXT_NAME" ] || die "VEC_DOCS must be its own index"
  [ -n "$DOCS_EMBED_DIM" ] || die "$CONFIG: set vars.DOCS_EMBED_DIM (the dimensions of DOCS_EMBED_MODEL) for $VEC_DOCS_NAME"
fi
guard_name "queue" "$QUEUE"; guard_name "queue" "$DLQ"; guard_name "Workflow" "$WORKFLOW"
guard_name "Space" "$SPACE_ID"
info "Worker $WORKER · D1 $D1_NAME · KV $KV_TITLE · Vectorize $VEC_NAME${VEC_TEXT_NAME:+ + $VEC_TEXT_NAME} ($EMBED_DIM-d)${VEC_DOCS_NAME:+ + $VEC_DOCS_NAME ($DOCS_EMBED_DIM-d)}"
info "queues $QUEUE, $DLQ · Workflow $WORKFLOW · R2 $R2_BUCKET (shared) · Space $SPACE_ID · model $MODEL_ID${EMBED_PATCHES:+ ($EMBED_PATCHES patches)}"
info "documentation: ${DOCS_REPO:-off}"

python3 -c "import huggingface_hub" 2>/dev/null || die "pip install -U huggingface_hub"
HF_USER=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) \
  || die "not logged in to Hugging Face: run  hf auth login  (token with write access)"
info "Hugging Face: $HF_USER"
CF_WHO=$("${WR[@]}" whoami --json 2>/dev/null) || die "not logged in to Cloudflare: run  npx wrangler login"
CF_ACCOUNTS=$(python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print("\n".join("{}  {}".format(a.get("id"), a.get("name")) for a in d.get("accounts") or []))' <<<"$CF_WHO")
if [ "$(grep -c . <<<"$CF_ACCOUNTS")" -gt 1 ] && [ -z "${CLOUDFLARE_ACCOUNT_ID:-}" ]; then
  printf '%s\n' "$CF_ACCOUNTS" >&2
  die "several Cloudflare accounts: export CLOUDFLARE_ACCOUNT_ID=<id> and re-run"
fi
info "Cloudflare: ${CLOUDFLARE_ACCOUNT_ID:-$(cut -c1-200 <<<"$CF_ACCOUNTS")}"

if [ ! -s "$SECRETS_FILE" ]; then
  (umask 077; python3 -c 'import json, secrets; print(json.dumps({"ADMIN_TOKEN": secrets.token_hex(32), "SPACE_API_TOKEN": secrets.token_hex(32)}, indent=1))' >"$SECRETS_FILE")
  info "generated tokens in $SECRETS_FILE (keep it; the stack uses them)"
fi
# The GitHub webhook secret is generated once too (files from before it existed get it added).
python3 - "$SECRETS_FILE" <<'PY2'
import json, secrets, sys
p = sys.argv[1]
d = json.load(open(p))
if not d.get("GITHUB_WEBHOOK_SECRET"):
    d["GITHUB_WEBHOOK_SECRET"] = secrets.token_hex(24)
    json.dump(d, open(p, "w"), indent=1)
PY2
ADMIN_TOKEN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ADMIN_TOKEN"])' "$SECRETS_FILE")
SPACE_API_TOKEN=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["SPACE_API_TOKEN"])' "$SECRETS_FILE")

# ---- 1. Hugging Face Space ----------------------------------------------------------------------
step "1/10 Hugging Face Space $SPACE_ID"
SPACE_URL=$(SPACE_ID="$SPACE_ID" MODEL_ID="$MODEL_ID" EMBED_PATCHES="$EMBED_PATCHES" SPACE_API_TOKEN="$SPACE_API_TOKEN" SECRETS_FILE="$SECRETS_FILE" python3 - <<'PY'
import json, os, sys
from huggingface_hub import HfApi
api = HfApi()
sid, model = os.environ["SPACE_ID"], os.environ["MODEL_ID"]
existed = api.repo_exists(sid, repo_type="space")
api.create_repo(sid, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
# Setting a variable or secret restarts the Space, so only do it when something changed.
have = api.get_space_variables(sid)
want = {"MODEL_ID": model}
# NaFlex: the patch budget per image must be the Worker's EMBED_PATCHES (checked on every call).
if os.environ.get("EMBED_PATCHES"):
    want["MAX_NUM_PATCHES"] = os.environ["EMBED_PATCHES"]
for k, v in want.items():
    if getattr(have.get(k), "value", None) != v:
        api.add_space_variable(sid, k, v)
        print(f"{k}={v} set", file=sys.stderr)
# /embed requires "Authorization: Bearer <API_TOKEN>"; the Space stays public so /health works.
# Secrets cannot be read back: remember in the tokens file which Space already has ours.
sec_path = os.environ["SECRETS_FILE"]
sec = json.load(open(sec_path))
# A Space created just now never has it, whatever the tokens file says (it may predate a teardown).
if not existed or sec.get("SPACE_SECRET_SET_ON") != sid:
    api.add_space_secret(sid, "API_TOKEN", os.environ["SPACE_API_TOKEN"])
    sec["SPACE_SECRET_SET_ON"] = sid
    json.dump(sec, open(sec_path, "w"), indent=1)
    print("API_TOKEN secret set", file=sys.stderr)
# No commit (and no rebuild) when the files did not change.
api.upload_folder(repo_id=sid, repo_type="space", folder_path="hf",
                  ignore_patterns=["__pycache__/*", "**/__pycache__/*", "*.pyc", ".DS_Store"],
                  commit_message="pixagram-search v4 embeddings (deploy.sh)")
owner, name = sid.split("/")
print(f"https://{owner}-{name}".replace("_", "-").replace(".", "-").lower() + ".hf.space")
PY
)
info "Space files in sync → $SPACE_URL"
patch_cfg hf_url "$SPACE_URL/embed"

# ---- 2. D1 --------------------------------------------------------------------------------------
step "2/10 D1 database $D1_NAME"
D1_ID=$("${WR[@]}" d1 list --json 2>/dev/null | json_pick name "$D1_NAME" uuid)
if [ -z "$D1_ID" ]; then
  run_ok "D1 $D1_NAME" "${WR[@]}" d1 create "$D1_NAME"
  D1_ID=$("${WR[@]}" d1 list --json 2>/dev/null | json_pick name "$D1_NAME" uuid)
else
  info "D1 $D1_NAME: already exists"
fi
[ -n "$D1_ID" ] || die "could not find the id of D1 $D1_NAME"
patch_cfg d1 "$D1_ID"; info "database_id $D1_ID"

# ---- 3. KV --------------------------------------------------------------------------------------
step "3/10 KV namespace $KV_TITLE"
KV_ID=$("${WR[@]}" kv namespace list 2>/dev/null | json_pick title "$KV_TITLE" id)
if [ -z "$KV_ID" ]; then
  run_ok "KV $KV_TITLE" "${WR[@]}" kv namespace create "$KV_TITLE"
  KV_ID=$("${WR[@]}" kv namespace list 2>/dev/null | json_pick title "$KV_TITLE" id)
else
  info "KV $KV_TITLE: already exists"
fi
[ -n "$KV_ID" ] || die "could not find the id of KV $KV_TITLE"
patch_cfg kv "$KV_ID"; info "id $KV_ID"

# ---- 4. Vectorize -------------------------------------------------------------------------------
# create_vector_index <name> <dimensions> <metadata specs...>: the index, then its metadata
# indexes, which must exist before the first vector is inserted (earlier vectors are not indexed).
create_vector_index() {
  local name="$1" dims="$2"; shift 2
  if [ -z "$("${WR[@]}" vectorize list --json 2>/dev/null | json_pick name "$name" name)" ]; then
    run_ok "Vectorize $name" "${WR[@]}" vectorize create "$name" --dimensions="$dims" --metric=cosine
  else
    info "Vectorize $name: already exists"
  fi
  local have spec prop typ
  have=$("${WR[@]}" vectorize list-metadata-index "$name" --json 2>/dev/null || true)
  for spec in "$@"; do
    prop="${spec%%:*}"; typ="${spec##*:}"
    if grep -q "\"$prop\"" <<<"$have"; then info "  metadata index $prop: already exists"; continue; fi
    run_ok "  metadata index $prop" "${WR[@]}" vectorize create-metadata-index "$name" --property-name="$prop" --type="$typ"
  done
}
step "4/10 Vectorize: $VEC_NAME (images)${VEC_TEXT_NAME:+, $VEC_TEXT_NAME (texts)}${VEC_DOCS_NAME:+, $VEC_DOCS_NAME (documentation)}"
create_vector_index "$VEC_NAME" "$EMBED_DIM" "${IMAGE_METADATA_INDEXES[@]}"
[ -z "$VEC_TEXT_NAME" ] || create_vector_index "$VEC_TEXT_NAME" "$EMBED_DIM" "${TEXT_METADATA_INDEXES[@]}"
[ -z "$VEC_DOCS_NAME" ] || create_vector_index "$VEC_DOCS_NAME" "$DOCS_EMBED_DIM"

# ---- 5. Queues + R2 -----------------------------------------------------------------------------
step "5/10 Queues $QUEUE, $DLQ · R2 $R2_BUCKET"
for q in "$QUEUE" "$DLQ"; do run_ok "queue $q" "${WR[@]}" queues create "$q"; done
# Default: the bucket production and v2 use, shared (content-addressed keys, written only when
# missing, never deleted). "already exists" is the expected answer here.
run_ok "R2 $R2_BUCKET" "${WR[@]}" r2 bucket create "$R2_BUCKET"

# ---- 6. Schema ----------------------------------------------------------------------------------
step "6/10 D1 migrations (0001 base schema, 0002 v3, 0003 documentation, 0004 suggestions, 0005 v4 answers, 0006 copy detection)"
"${WR[@]}" d1 migrations apply "$D1_NAME" --remote -c "$CONFIG" </dev/null

# ---- 7. Deploy + secrets ------------------------------------------------------------------------
step "7/10 Deploy Worker $WORKER"
DEPLOY_OUT=$("${WR[@]}" deploy -c "$CONFIG" 2>&1) || { printf '%s\n' "$DEPLOY_OUT" >&2; die "wrangler deploy failed"; }
printf '%s\n' "$DEPLOY_OUT" | grep -E "Uploaded|Deployed|workers\.dev|Current Version" | sed 's/^/    /' || true
NEW_BASE="${NEW_BASE:-$(grep -oE "https://$WORKER\.[A-Za-z0-9.-]+\.workers\.dev" <<<"$DEPLOY_OUT" | head -1)}"
[ -n "$NEW_BASE" ] || die "could not read the workers.dev URL from the deploy output; re-run with NEW_BASE=https://$WORKER.<subdomain>.workers.dev"
TMP_SECRETS=$(mktemp); trap 'rm -f "$TMP_SECRETS"' EXIT
# Tokens go from the tokens file to a private temp file, never through argv (readable with ps).
python3 - "$SECRETS_FILE" "$TMP_SECRETS" <<'PY2'
import json, sys
d = json.load(open(sys.argv[1]))
out = {"ADMIN_TOKEN": d["ADMIN_TOKEN"], "HF_TOKEN": d["SPACE_API_TOKEN"], "GITHUB_WEBHOOK_SECRET": d["GITHUB_WEBHOOK_SECRET"]}
json.dump(out, open(sys.argv[2], "w"))
print("    secrets " + ", ".join(out) + " set")
PY2
"${WR[@]}" secret bulk "$TMP_SECRETS" -c "$CONFIG" >/dev/null
info "→ $NEW_BASE"

# ---- 8. Wait for the Space, then index the chain ------------------------------------------------
step "8/10 Wait for $SPACE_URL, then backfill (posts + edit history) and live tail"
for i in $(seq 1 90); do
  H=$(curl -sS --max-time 20 "$SPACE_URL/health" 2>/dev/null || true)
  # ready, with our model and (NaFlex) our patch budget: a restarting Space answers with the old ones for a while
  if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("ready") and d.get("model")==sys.argv[2] and (not sys.argv[3] or str(d.get("max_num_patches"))==sys.argv[3]) else 1)' "$H" "$MODEL_ID" "$EMBED_PATCHES" 2>/dev/null; then
    info "Space ready: $H"; break
  fi
  STAGE=$(SPACE_ID="$SPACE_ID" python3 -c 'import os; from huggingface_hub import HfApi; print(HfApi().get_space_runtime(os.environ["SPACE_ID"]).stage)' 2>/dev/null || echo "?")
  case "$STAGE" in BUILD_ERROR|RUNTIME_ERROR|CONFIG_ERROR) die "Space $STAGE: see https://huggingface.co/spaces/$SPACE_ID?logs=container";; esac
  [ $((i % 6)) -eq 1 ] && info "Space stage: $STAGE … (up to ~5 min on the first build)"
  [ "$i" -eq 90 ] && die "Space not ready after 15 min: https://huggingface.co/spaces/$SPACE_ID"
  sleep 10
done
PROBE=$(http_json POST "$SPACE_URL/embed" "$SPACE_API_TOKEN" '{"inputs":{"texts":["A Cow","a cow"]}}' 2>/dev/null || true)
python3 -c '
import json, sys
d = json.loads(sys.argv[1]); a, b = d["embeddings"]
cos = sum(x * y for x, y in zip(a, b))
print("    /embed: model {}, dim {}, cos(A Cow, a cow) = {:.4f}, calibration {}".format(d["model"], d["dim"], cos, d.get("calibration")))
' "$PROBE" 2>/dev/null || info "/embed probe failed (Hugging Face edge 502s happen; the queue retries): ${PROBE:0:160}"

BF=$(http_json POST "$NEW_BASE/admin/backfill" "$ADMIN_TOKEN" '{"reason":"deploy"}') || die "could not start the backfill: ${BF:0:300}"
BF_ID=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["id"])' "$BF") || die "unexpected backfill answer: ${BF:0:300}"
IDX=$(http_json POST "$NEW_BASE/admin/indexer/start" "$ADMIN_TOKEN") || die "could not start the chain tail: ${IDX:0:300}"
info "backfill $BF_ID started, live tail started"
# The documentation (help answers): first sync now, then the 10-minute cron or the webhook.
DS=$(http_json POST "$NEW_BASE/admin/docs/sync" "$ADMIN_TOKEN" 2>/dev/null) || DS='{}'
python3 -c '
import json, sys
try:
    d = json.loads(sys.argv[1])
except ValueError:
    d = {}
print("    documentation {}: {} ({} files, {} indexed, {} skipped, {} chunks embedded){}".format(
    d.get("repo"), d.get("status", "sync failed; the cron retries"), d.get("files", "?"), len(d.get("indexed") or []),
    len(d.get("skipped") or []), d.get("embedded", "?"), "; " + d["error"] if d.get("error") else ""))
' "$DS" || true

if [ "${SKIP_WAIT:-}" = "1" ]; then
  step "Deployed (SKIP_WAIT=1)"; info "Worker: $NEW_BASE"; info "Space:  $SPACE_URL"
  info "after indexing: BASE=$NEW_BASE ADMIN_TOKEN=… scripts/admin.sh background"; exit 0
fi

# ---- 9. Wait for enrichment ---------------------------------------------------------------------
step "9/10 Wait for enrichment (vectors, descriptions, concepts, history)"
# The backfill only records the posts and queues their enrichment; the queue does the work. Done
# means: the backfill is complete, no job is queued, and nothing has changed for 2.5 minutes (a
# failed stage may be retried after one or two minutes). Stats and embedding run in the same queue
# message, so "every artwork has a vector" alone says nothing about the artworks still queued.
DONE=0
QUIET=0
PREV=""
REST="(status unavailable)"
for i in $(seq 1 160); do
  # A failed call (curl prints the error body, then fails) must not end the wait: under set -e
  # a failing command substitution in an assignment exits the script. Reset to {} on failure, and
  # parse defensively (a 5xx body may not be JSON).
  S=$(http_json GET "$NEW_BASE/admin/stats" "$ADMIN_TOKEN" 2>/dev/null) || S='{}'
  B=$(http_json GET "$NEW_BASE/admin/backfill/$BF_ID" "$ADMIN_TOKEN" 2>/dev/null) || B='{}'
  LINE=$(python3 -c '
import json, sys
def load(x):
    try:
        v = json.loads(x or "{}")
        return v if isinstance(v, dict) else {}
    except ValueError:
        return {}
s, b = load(sys.argv[1]), load(sys.argv[2])
if not isinstance(s.get("jobs"), list) or not s.get("artworks"):
    print("? -1 ? (stats unavailable)")
    sys.exit(0)
a = s.get("artworks") or {}
n, e, d, c = a.get("n") or 0, a.get("embedded") or 0, a.get("described") or 0, a.get("with_concepts") or 0
live = sum(p.get("n", 0) for p in s.get("posts", []) if p.get("type") == "artwork" and not p.get("deleted"))
jobs = s.get("jobs", [])
queued = sum(j.get("n", 0) for j in jobs if j.get("status") == "queued")
failed = sum(j.get("n", 0) for j in jobs if j.get("status") == "failed")
st = (b.get("status") or {}).get("status", "?")
versions = sum(v.get("n", 0) for v in s.get("versions", []))
print(f"{st} {queued} {n}:{e}:{d}:{c}:{failed} backfill={st} artworks={n}/{live} embedded={e} described={d} concepts={c} versions={versions} queued_jobs={queued} failed_jobs={failed}")
' "$S" "$B") || LINE="? -1 ? (status unavailable)"
  read -r ST Q SIG REST <<<"$LINE"
  [ $((i % 3)) -eq 1 ] && info "$REST"
  if [ "$ST" = "complete" ] && [ "$Q" = "0" ] && [ "$SIG" = "$PREV" ] && [ "${SIG%%:*}" != "0" ]; then QUIET=$((QUIET + 1)); else QUIET=0; fi
  PREV="$SIG"
  if [ "$QUIET" -ge 10 ]; then DONE=1; info "$REST"; break; fi
  case "$ST" in errored|terminated)
    ERR=$(python3 -c 'import json,sys; s=json.loads(sys.argv[1]).get("status") or {}; print((s.get("error") or {}).get("message") or json.dumps(s))' "$B" 2>/dev/null || printf '%s' "$B")
    info "backfill $ST: $ERR"
    die "fix the cause, then re-run scripts/deploy.sh: it starts a new backfill, and artworks already indexed are skipped";;
  esac
  sleep 15
done
case "$REST" in *"failed_jobs=0"*|*unavailable*) ;; *)
  info "some stages failed: BASE=$NEW_BASE ADMIN_TOKEN=… scripts/admin.sh failed shows why (the 10-minute sweeper retries them)";;
esac

if [ "$DONE" = 1 ]; then
  # Semantic scores are z-scores against a random sample of the corpus vectors; the nightly cron
  # refreshes it, this takes the first sample now that the vectors exist.
  BGR=$(http_json POST "$NEW_BASE/admin/background" "$ADMIN_TOKEN" 2>/dev/null) || BGR='(failed; the nightly cron will take it)'
  info "background sample: $BGR"

  # ---- 10. Evaluate -----------------------------------------------------------------------------
  step "10/10 Evaluate: $OLD_BASE vs $NEW_BASE (eval/queries.jsonl), /ask (eval/ask.jsonl), the v4 question set"
  python3 scripts/eval.py --compare "$OLD_BASE" "$NEW_BASE" || true
  python3 scripts/eval.py "$NEW_BASE" --ask || true
  # v4: the generated question set through /ask, deterministic modes only (no model cost), and
  # the frozen-context benchmark of the reasoning models (spec §31) on a sample of it
  python3 scripts/eval_v4.py "$NEW_BASE" --modes fast --limit 400 || true
  info "model benchmark (costs Workers AI tokens): ADMIN_TOKEN=… python3 scripts/benchmark.py $NEW_BASE"
else
  info "indexing is not finished after 40 min (the queue and the 10-minute sweeper keep going)."
  info "When BASE=$NEW_BASE ADMIN_TOKEN=… scripts/admin.sh stats shows no queued job, run:"
  info "  scripts/admin.sh background, then python3 scripts/eval.py --compare $OLD_BASE $NEW_BASE"
  info "(an evaluation on a half-indexed corpus would compare nothing useful)"
fi

step "Done"
info "v4 Worker:  $NEW_BASE"
info "v4 Space:   https://huggingface.co/spaces/$SPACE_ID  ($SPACE_URL)"
info "tokens:     $SECRETS_FILE  (ADMIN_TOKEN for $NEW_BASE/admin/*; SPACE_API_TOKEN = the Worker's HF_TOKEN)"
info "evaluate:   python3 scripts/eval.py --compare $OLD_BASE $NEW_BASE"
info "admin:      BASE=$NEW_BASE ADMIN_TOKEN=… scripts/admin.sh stats"
if [ -n "$DOCS_REPO" ] && [ "$DOCS_REPO" != "off" ]; then
  info "help:       $DOCS_REPO is synced every 10 minutes. For instant updates, add a webhook in"
  info "            https://github.com/$DOCS_REPO/settings/hooks : payload URL $NEW_BASE/webhooks/github,"
  info "            content type application/json, secret = GITHUB_WEBHOOK_SECRET in $SECRETS_FILE, event: push"
fi
info "remove:     scripts/teardown.sh"
