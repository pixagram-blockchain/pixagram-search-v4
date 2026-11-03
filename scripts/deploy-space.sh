#!/usr/bin/env bash
# Update the stack's Hugging Face Space alone: its hardware and sleep time, its variables, the files
# of hf/, then wait until it serves the Worker's model and patch budget. The Worker is not touched
# (scripts/deploy-worker.sh deploys it; scripts/deploy.sh builds a whole stack).
#
#   pip install -U huggingface_hub && hf auth login     # an HF token with write access to the Space
#   scripts/deploy-space.sh
#
# Hardware (v4.8.2): CPU upgrade (8 vCPU, 32 GB, $0.03 an hour) that never sleeps. On free CPU
# basic the Space sleeps after 48 hours without a request; the v4 Space stopped answering from
# 29 October to 3 November 2026 (found asleep), and the artworks posted meanwhile got no vectors. The Space computes one
# embedding per CPU at once and the Worker reads that from /health, so nothing else changes.
#
# Overrides (env): SPACE_ID (primerz/pixagram-search-v4), CONFIG (wrangler.jsonc),
#   SPACE_HARDWARE (cpu-upgrade; "keep" leaves it as it is), SPACE_SLEEP (-1 = never, else
#   seconds without a request; paid hardware only)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/.."
CONFIG="${CONFIG:-wrangler.jsonc}"
SPACE_ID="${SPACE_ID:-primerz/pixagram-search-v4}"
# shellcheck source=scripts/lib.sh
source "$HERE/lib.sh"

step "Preflight"
for c in python3 curl; do command -v "$c" >/dev/null || die "$c is required"; done
[ -f "$CONFIG" ] || die "$CONFIG not found (run from the repository)"
python3 -c "import huggingface_hub" 2>/dev/null || die "pip install -U huggingface_hub"
python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1 \
  || die "not logged in to Hugging Face: run  hf auth login  (token with write access)"
guard_name "Space" "$SPACE_ID"
MODEL_ID=$(cfg vars.EMBED_MODEL)
EMBED_PATCHES=$(cfg vars.EMBED_PATCHES)
[ -n "$MODEL_ID" ] || die "$CONFIG has no vars.EMBED_MODEL"
info "Space $SPACE_ID · model $MODEL_ID${EMBED_PATCHES:+ ($EMBED_PATCHES patches)}"

step "Hardware and sleep time"
python3 "$HERE/space_hardware.py" "$SPACE_ID" "${SPACE_HARDWARE:-cpu-upgrade}" "${SPACE_SLEEP:--1}"

step "Variables and files"
SPACE_URL=$(SPACE_ID="$SPACE_ID" MODEL_ID="$MODEL_ID" EMBED_PATCHES="$EMBED_PATCHES" python3 - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi()
sid = os.environ["SPACE_ID"]
if not api.repo_exists(sid, repo_type="space"):
    sys.exit(f"Space {sid} does not exist: scripts/deploy.sh creates the stack")
# Setting a variable restarts the Space, so only when something changed.
have = api.get_space_variables(sid)
want = {"MODEL_ID": os.environ["MODEL_ID"]}
if os.environ.get("EMBED_PATCHES"):
    want["MAX_NUM_PATCHES"] = os.environ["EMBED_PATCHES"]
for k, v in want.items():
    if getattr(have.get(k), "value", None) != v:
        api.add_space_variable(sid, k, v)
        print(f"    {k}={v} set", file=sys.stderr)
# No commit (and no rebuild) when the files did not change.
api.upload_folder(repo_id=sid, repo_type="space", folder_path="hf",
                  ignore_patterns=["__pycache__/*", "**/__pycache__/*", "*.pyc", ".DS_Store"],
                  commit_message="pixagram-search v4 embeddings (deploy-space.sh)")
owner, name = sid.split("/")
print(f"https://{owner}-{name}".replace("_", "-").replace(".", "-").lower() + ".hf.space")
PY
)
info "files in sync → $SPACE_URL"

step "Wait until it serves $MODEL_ID"
for i in $(seq 1 90); do
  H=$(curl -sS --max-time 20 "$SPACE_URL/health" 2>/dev/null || true)
  if python3 -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("ready") and d.get("model")==sys.argv[2] and (not sys.argv[3] or str(d.get("max_num_patches"))==sys.argv[3]) else 1)' "$H" "$MODEL_ID" "$EMBED_PATCHES" 2>/dev/null; then
    python3 -c '
import json, sys
d = json.loads(sys.argv[1])
print("    ready: {} CPUs, {} embeddings at once, {} PyTorch thread(s) each, up {} s".format(d.get("cpus"), d.get("concurrency"), d.get("threads"), d.get("uptime_s")))
' "$H"
    break
  fi
  STAGE=$(SPACE_ID="$SPACE_ID" python3 -c 'import os; from huggingface_hub import HfApi; print(HfApi().get_space_runtime(os.environ["SPACE_ID"]).stage)' 2>/dev/null || echo "?")
  case "$STAGE" in BUILD_ERROR|RUNTIME_ERROR|CONFIG_ERROR) die "Space $STAGE: see https://huggingface.co/spaces/$SPACE_ID?logs=container";; esac
  [ $((i % 6)) -eq 1 ] && info "Space stage: $STAGE … (a hardware change or a rebuild takes a few minutes)"
  [ "$i" -eq 90 ] && die "Space not ready after 15 min: https://huggingface.co/spaces/$SPACE_ID"
  sleep 10
done
info "the Worker reads the new concurrency from /health within ten minutes; check with"
info "  curl -s https://<worker>/healthz   (status, embedding, vectors)"
