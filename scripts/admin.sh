#!/usr/bin/env bash
# Thin wrapper around the admin API of a deployed stack.
#   BASE=https://pixagram-search-v4.<you>.workers.dev ADMIN_TOKEN=... scripts/admin.sh stats
# (scripts/deploy.sh keeps ADMIN_TOKEN in ~/.pixagram-search-v4.json)
set -euo pipefail
BASE="${BASE:-http://127.0.0.1:8787}"
: "${ADMIN_TOKEN:?set ADMIN_TOKEN}"
# The token reaches curl through a header file (printf is a builtin): never in a process's
# arguments, where any local user could read it with ps.
curl() { command curl -H @<(printf 'Authorization: Bearer %s\n' "$ADMIN_TOKEN") "$@"; }
H=(-sS -H "Content-Type: application/json")

# stages_json "embed,text" -> ["embed","text"] (default: every stage)
stages_json() {
  local s="${1:-stats,paph,embed,describe,concepts,text}" out="" x
  IFS=',' read -r -a parts <<<"$s"
  for x in "${parts[@]}"; do
    case "$x" in stats|paph|embed|describe|concepts|text) out+="${out:+,}\"$x\"";; *) echo "unknown stage: $x" >&2; exit 2;; esac
  done
  printf '[%s]' "$out"
}
bool_json() { case "${1:-false}" in 1|true|yes|force) echo true;; *) echo false;; esac; }

case "${1:-help}" in
  stats)            curl "${H[@]}" "$BASE/admin/stats" ;;
  indexer)          curl "${H[@]}" "$BASE/admin/indexer" ;;
  start)            curl "${H[@]}" -X POST "$BASE/admin/indexer/start${2:+?from=$2}" ;;     # start [from_block]
  stop)             curl "${H[@]}" -X POST "$BASE/admin/indexer/stop" ;;
  backfill)         curl "${H[@]}" -X POST "$BASE/admin/backfill" -d "{\"reason\":\"cli\"${2:+,\"authors\":[\"$2\"]}}" ;;                        # backfill [author]
  backfill-history) curl "${H[@]}" -X POST "$BASE/admin/backfill" -d "{\"reason\":\"cli\",\"historyOnly\":true${2:+,\"authors\":[\"$2\"]}}" ;;  # edit history only
  backfill-status)  curl "${H[@]}" "$BASE/admin/backfill/${2:?backfill id}" ;;
  ingest)           curl "${H[@]}" -X POST "$BASE/admin/ingest/${2:?author}/${3:?permlink}" ;;
  reindex-all)      curl "${H[@]}" -X POST "$BASE/admin/reindex" -d "{\"all\":true,\"stages\":$(stages_json "${2:-}"),\"force\":$(bool_json "${3:-}")}" ;;
  reindex-author)   curl "${H[@]}" -X POST "$BASE/admin/reindex" -d "{\"author\":\"${2:?author}\",\"stages\":$(stages_json "${3:-}"),\"force\":$(bool_json "${4:-}")}" ;;
  reindex-post)     curl "${H[@]}" -X POST "$BASE/admin/reindex" -d "{\"post_id\":${2:?post id},\"stages\":$(stages_json "${3:-}"),\"force\":true}" ;;
  sweep)            curl "${H[@]}" -X POST "$BASE/admin/sweep?max=${2:-200}" ;;
  background)       curl "${H[@]}" -X POST "$BASE/admin/background" ;;
  vocab-rebuild)    curl "${H[@]}" -X POST "$BASE/admin/vocab/rebuild" ;;
  failed)           curl "${H[@]}" "$BASE/admin/jobs/failed" ;;
  queries)          curl "${H[@]}" "$BASE/admin/queries?days=${2:-7}" ;;
  weights)          curl "${H[@]}" "$BASE/admin/ranker/weights" ;;
  weights-set)      curl "${H[@]}" -X POST "$BASE/admin/ranker/weights" --data-binary "@${2:?weights.json}" ;;
  ltr-export)       curl "${H[@]}" "$BASE/admin/ltr/export?days=${2:-30}" ;;
  describe)         curl "${H[@]}" -X POST "$BASE/admin/debug/describe/${2:?post id}${3:+?backend=$3}" ;;      # describe id [chat|gemma|scout|moondream|caption]
  docs)             curl "${H[@]}" "$BASE/admin/docs" ;;
  docs-sync)        curl "${H[@]}" -X POST "$BASE/admin/docs/sync$( [ "$(bool_json "${2:-}")" = true ] && echo '?force=1')" ;;  # docs-sync [force]
  docs-reembed)     curl "${H[@]}" -X POST "$BASE/admin/docs/reembed" ;;
  docs-retrieve)    curl "${H[@]}" -G "$BASE/admin/debug/docs" --data-urlencode "q=${2:?question}" --data-urlencode "k=${3:-10}" ;;  # what retrieval finds (lexical, cosine, score): DOCS_MIN_SCORE calibration
  docs-gaps)        curl "${H[@]}" "$BASE/admin/docs/gaps?days=${2:-30}" ;;
  paph)             curl "${H[@]}" "$BASE/admin/paph" ;;                                         # copy detection: shards, verdicts, identity, budgets
  paph-alerts)      curl "${H[@]}" "$BASE/admin/paph/alerts?days=${2:-30}" ;;                  # cross-author copies, earlier → later
  paph-rederive)    curl "${H[@]}" -X POST "$BASE/admin/paph/rederive?limit=${2:-200}" ;;     # after a paph-x release: repeat until remaining is 0
  paph-stale)       curl "${H[@]}" "$BASE/admin/paph/stale" ;;                                 # verdicts of another engine or policy
  paph-purge-stale) curl "${H[@]}" -X POST "$BASE/admin/paph/stale?purge=1" ;;                 # refused while artworks await their re-check
  paph-gc)          curl "${H[@]}" -X POST "$BASE/admin/paph/gc?limit=${2:-500}" ;;            # works of deleted posts leave the shards (and stale entries lose their mark)
  paph-heal)        curl "${H[@]}" -X POST "$BASE/admin/paph/heal?limit=${2:-100}" ;;          # verdicts reached on wire 3 that no re-check will replace: compared again
  copies)           curl "${H[@]}" "$BASE/copies/${2:?post id}?min=${3:-copy}" ;;              # stored verdicts of an artwork
  copies-live)      curl "${H[@]}" "$BASE/copies/${2:?post id}?min=${3:-copy}&live=1" ;;       # checked again now, in every shard
  copies-image)     command curl -sS -F "image=@${2:?image file}" "$BASE/copies-by-image?min=${3:-copy}" ;;  # an upload (sub-second; cached a day)
  query)            command curl -sS -G "$BASE/query" --data-urlencode "q=${2:?text}" ;;   # what the search box does with a text
  ask)              curl "${H[@]}" -X POST "$BASE/ask" -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "mode": sys.argv[2], "trace": True}))' "${2:?question}" "${3:-auto}")" ;;  # ask "<question>" [mode]
  ask-log)          curl "${H[@]}" "$BASE/admin/ask/log?days=${2:-7}${3:+&status=$3}" ;;          # ask-log [days] [status]
  ask-trace)        curl "${H[@]}" "$BASE/admin/ask/trace/${2:?query id}" ;;
  models)           curl "${H[@]}" "$BASE/admin/models" ;;
  ltr-pairs)        curl "${H[@]}" "$BASE/admin/ltr/pairs?days=${2:-30}" ;;
  export-sft)       curl "${H[@]}" "$BASE/admin/ask/export-sft?days=${2:-30}" ;;
  help-ask)         if [ -n "${3:-}" ]; then curl "${H[@]}" -G "$BASE/admin/debug/help" --data-urlencode "q=${2:?question}" --data-urlencode "model=$3"   # help-ask "<question>" [model]
                    else command curl -sS -G "$BASE/help" --data-urlencode "q=${2:?question}"; fi ;;
  *) cat <<'USAGE'
usage: scripts/admin.sh <command> [args]
  stats | indexer | start [from_block] | stop
  backfill [author] | backfill-history [author] | backfill-status <id> | ingest <author> <permlink>
  reindex-all [stages] [force] | reindex-author <author> [stages] [force] | reindex-post <id> [stages]
      stages: comma list of stats,paph,embed,describe,concepts,text (default: all)
  paph | paph-alerts [days] | paph-rederive [limit] | paph-stale | paph-purge-stale | paph-gc [limit] | paph-heal [limit]   copy detection
  copies <id> [min] | copies-live <id> [min] | copies-image <file> [min]             copies of a work / an image
  sweep [max] | background | vocab-rebuild | failed | queries [days]
  weights | weights-set <file.json> | ltr-export [days] | describe <id> [backend]
  docs | docs-sync [force] | docs-reembed | docs-gaps [days]       the documentation behind /help
  docs-retrieve "<question>" [k]                                    its retrieval alone (cosines, for DOCS_MIN_SCORE)
  ask "<question>" [mode] | ask-log [days] [status] | ask-trace <query id>   v4 answers, with traces
  models | ltr-pairs [days] | export-sft [days]                    models; learning-to-rank and fine-tuning data
  query "<text>" | help-ask "<question>" [model]                     public routes, for a quick look;
      help-ask with a model id answers with that model instead of HELP_MODEL (admin)
USAGE
  ;;
esac
echo
