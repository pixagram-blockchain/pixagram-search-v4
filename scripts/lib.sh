# shellcheck shell=bash
# Helpers shared by scripts/deploy.sh and scripts/teardown.sh (sourced, not executed).
# Expects CONFIG to point at the wrangler config of the v4 stack.

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\033[31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Names that belong to the production, v2 and v3 stacks (v3 is the one the Pixagram UI uses).
# Nothing in this repository may create, deploy to or delete them.
PROTECTED_NAMES=(
  pixagram-search pixagram-search-v2 pixagram-search-v3   # Workers and D1 databases
  pixagram-search-v3-cache                                # v3 KV namespace
  pixagram-art-siglip2                                    # v2 Vectorize index
  pixagram-art-v3 pixagram-text-v3 pixagram-docs-v3       # v3 Vectorize indexes
  pixagram-enrich pixagram-enrich-dlq pixagram-enrich-v2 pixagram-enrich-v2-dlq pixagram-enrich-v3 pixagram-enrich-v3-dlq
  pixagram-backfill pixagram-backfill-v2 pixagram-backfill-v3
  primerz/pixagram-siglip-endpoint primerz/pixagram-siglip2 primerz/pixagram-search-v3
)
# The production Vectorize index is also called pixagram-art, like the shared R2 bucket, so it is
# checked separately (guard_vector_index).

guard_name() {  # guard_name <kind> <name>
  local p
  for p in "${PROTECTED_NAMES[@]}"; do
    [ "$2" != "$p" ] || die "$1 '$2' belongs to the production, v2 or v3 stack; refusing (edit $CONFIG)"
  done
}

guard_vector_index() {
  [ "$1" != "pixagram-art" ] || die "Vectorize index 'pixagram-art' is production's; refusing (edit $CONFIG)"
  guard_name "Vectorize index" "$1"
}

# ---- config -------------------------------------------------------------------------------------

# cfg <dotted.path>: read a value from the JSONC config (comments stripped outside strings).
# Prints nothing (and succeeds) when the path does not exist.
cfg() {
  python3 - "$CONFIG" "$1" <<'PY'
import json, re, sys
def strip_jsonc(s):
    out, i, n, in_str = [], 0, len(s), False
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(s[i + 1]); i += 2; continue
            if c == '"': in_str = False
            i += 1; continue
        if c == '"': in_str = True; out.append(c); i += 1; continue
        if s.startswith("//", i):
            j = s.find("\n", i); i = n if j < 0 else j; continue
        if s.startswith("/*", i):
            j = s.find("*/", i + 2); i = n if j < 0 else j + 2; continue
        out.append(c); i += 1
    return re.sub(r",(\s*[}\]])", r"\1", "".join(out))
v = json.loads(strip_jsonc(open(sys.argv[1]).read()))
try:
    for k in sys.argv[2].split("."):
        v = v[int(k)] if isinstance(v, list) else v[k]
except (KeyError, IndexError, ValueError):
    sys.exit(0)
print(json.dumps(v) if isinstance(v, (dict, list)) else v)
PY
}

# vector_index <binding>: index_name of the Vectorize binding (empty when absent).
vector_index() {
  python3 -c '
import json, sys
for b in json.loads(sys.argv[1] or "[]"):
    if b.get("binding") == sys.argv[2]:
        print(b.get("index_name", "")); break
' "$(cfg vectorize)" "$1"
}

# patch_cfg <d1|kv|hf_url> <value>: rewrite one value in place, keeping comments and layout.
patch_cfg() {
  python3 - "$CONFIG" "$1" "$2" <<'PY'
import re, sys
path, what, val = sys.argv[1:4]
pats = {
    "d1": r'("database_id"\s*:\s*")[^"]*(")',
    "kv": r'("binding"\s*:\s*"CACHE"\s*,\s*"id"\s*:\s*")[^"]*(")',
    "hf_url": r'("HF_EMBED_URL"\s*:\s*")[^"]*(")',
}
s = open(path).read()
s2, n = re.subn(pats[what], lambda m: m.group(1) + val + m.group(2), s, count=1)
if n != 1:
    sys.exit(f"could not find the {what} entry in {path}")
if s2 != s:
    open(path, "w").write(s2)
PY
}

# ---- wrangler output ----------------------------------------------------------------------------

# json_pick <match-field> <wanted> <out-field>: read a JSON array (possibly surrounded by log
# lines) on stdin and print <out-field> of the item whose <match-field> equals <wanted>, or
# ends with "-<wanted>" (wrangler may prefix KV titles with the Worker name).
json_pick() {
  python3 -c '
import json, sys
raw = sys.stdin.read()
items, dec, i = [], json.JSONDecoder(), raw.find("[")
while i >= 0:  # first "[" that starts a JSON array of objects (skips "[WARNING]" style log lines)
    try:
        v, _ = dec.raw_decode(raw[i:])
        if isinstance(v, list) and all(isinstance(x, dict) for x in v):
            items = v; break
    except ValueError:
        pass
    i = raw.find("[", i + 1)
field, wanted, out = sys.argv[1:4]
for it in items:
    v = str(it.get(field, ""))
    if v == wanted or v.endswith("-" + wanted):
        print(it.get(out, "")); break
' "$@"
}

# run_ok <label> <cmd...>: run a create command; "already exists" counts as success.
run_ok() {
  local label="$1"; shift
  local out
  if out=$("$@" 2>&1); then info "$label: created"; return 0; fi
  if grep -Eqi "already (exists|taken|in use)|name is taken|code: 11009|code: 10004" <<<"$out"; then info "$label: already exists"; return 0; fi
  printf '%s\n' "$out" >&2
  die "$label failed"
}

# http_json <method> <url> [bearer] [json-body] -> body on stdout, fails on non-2xx.
# The bearer token goes in through a header file (printf is a builtin), so it never appears in a
# process's arguments, where any local user could read it with ps.
http_json() {
  local method="$1" url="$2" bearer="${3:-}" body="${4:-}"
  local args=(-sS --max-time 120 -X "$method" "$url" -H "accept: application/json")
  [ -n "$body" ] && args+=(-H "content-type: application/json" --data "$body")
  if [ -n "$bearer" ]; then
    curl --fail-with-body "${args[@]}" -H @<(printf 'authorization: Bearer %s\n' "$bearer")
  else
    curl --fail-with-body "${args[@]}"
  fi
}

# Vectorize metadata indexes (max 10 per index; they must exist before the first insert).
# Keep in sync with vectorMetadata() in src/enrich/consumer.ts and vectorFilter() in
# src/search/vectors.ts. (Used by the scripts that source this file.)
# shellcheck disable=SC2034
IMAGE_METADATA_INDEXES=(author:string primary_color:string size_class:string created:number color_count:number nsfw:boolean listed:boolean ai_training:boolean orientation:string transparent:boolean)
# shellcheck disable=SC2034
TEXT_METADATA_INDEXES=(author:string primary_color:string size_class:string created:number type:string nsfw:boolean listed:boolean ai_training:boolean orientation:string transparent:boolean)
