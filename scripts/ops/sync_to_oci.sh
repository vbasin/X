#!/usr/bin/env bash
# sync_to_oci.sh — Incremental OCI Object Storage sync
#
# Configuration: edit /opt/vtech/config/oci_sync.conf (targets, region, parallelism)
# Credentials:   ~/.oci/config (OCI standard — not managed here)
#
# Usage:
#   ./scripts/sync_to_oci.sh              # sync changed/new files only
#   ./scripts/sync_to_oci.sh --dry-run    # preview without uploading
#   ./scripts/sync_to_oci.sh --delete     # also remove remote objects deleted locally

set -euo pipefail

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------
CONF_FILE="/opt/vtech/config/runtime/oci_sync.conf"
if [ ! -f "$CONF_FILE" ]; then CONF_FILE="/opt/vtech/config/oci_sync.conf"; fi
[[ -f "${CONF_FILE}" ]] || { echo "ERROR: config not found: ${CONF_FILE}" >&2; exit 1; }
# shellcheck source=/opt/vtech/config/oci_sync.conf
source "${CONF_FILE}"

# ---------------------------------------------------------------------------
# Logging — format matches project convention (feature-builder, live-streamer)
# stdout → oci-sync.log, stderr → oci-sync.err (via systemd append: directive)
# ---------------------------------------------------------------------------
_ts()  { date -u '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "$(_ts) | oci-sync | INFO  | $*"; }
warn() { echo "$(_ts) | oci-sync | WARN  | $*" >&2; }
err()  { echo "$(_ts) | oci-sync | ERROR | $*" >&2; }

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
DRY_RUN=false
DELETE=false
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=true ;;
    --delete)  DELETE=true ;;
  esac
done

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
command -v oci >/dev/null 2>&1     || { err "oci CLI not found in PATH"; exit 1; }
[[ -f "${OCI_CONFIG_FILE}" ]]      || { err "OCI config not found: ${OCI_CONFIG_FILE}"; exit 1; }

log "starting  dry_run=${DRY_RUN}  delete=${DELETE}"

# ---------------------------------------------------------------------------
# Sync flags array — built once, reused per target
# ---------------------------------------------------------------------------
sync_flags=()
[[ "${DRY_RUN}" == true ]] && sync_flags+=(--dry-run)
[[ "${DELETE}"  == true ]] && sync_flags+=(--delete)

# ---------------------------------------------------------------------------
# Sync loop
# ---------------------------------------------------------------------------
total_up=0; total_sk=0; total_fail=0

for target in "${SYNC_TARGETS[@]}"; do
  IFS='|' read -r local_dir bucket prefix <<< "${target}"

  if [[ ! -d "${local_dir}" ]]; then
    warn "local dir missing, skipping: ${local_dir}"
    continue
  fi

  file_count=$(find "${local_dir}" -type f | wc -l)
  log "syncing  local=${local_dir}  bucket=${bucket}  prefix=${prefix}  files=${file_count}"

  sync_out=$(oci os object sync \
    --config-file "${OCI_CONFIG_FILE}" \
    --region      "${OCI_REGION}" \
    --bucket-name "${bucket}" \
    --src-dir     "${local_dir}" \
    --prefix      "${prefix}" \
    --parallel-operations-count "${OCI_PARALLEL_OPS}" \
    "${sync_flags[@]+"${sync_flags[@]}"}" \
    2>&1) || {
      err "sync failed  local=${local_dir}  bucket=${bucket}"
      err "${sync_out}"
      total_fail=$(( total_fail + 1 ))
      continue
    }

  # Parse the JSON summary OCI sync appends to its output.
  # OCI CLI emits the opening '{' on its own line — use that as the anchor
  # rather than rfind('{') which would match inner nested braces.
  # uploaded-objects: dict{name → {etag,...}}  — len = upload count
  # skipped-objects:  list[name]               — len = skip count
  # upload-failures:  dict{name → error}       — len = failure count
  IFS=' ' read -r up sk fail < <(echo "${sync_out}" | python3 -c "
import sys, json, re
text = sys.stdin.read()
up = sk = fail = 0
m = re.search(r'^\{$', text, re.MULTILINE)
if m:
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, m.start())
        up   = len(obj.get('uploaded-objects', {}))
        sk   = len(obj.get('skipped-objects',  []))
        fail = len(obj.get('upload-failures',  {}))
    except (json.JSONDecodeError, ValueError):
        pass
print(up, sk, fail)
" 2>/dev/null)

  total_up=$(( total_up + up ))
  total_sk=$(( total_sk + sk ))

  if [[ "${fail}" -gt 0 ]]; then
    err "upload failures  count=${fail}  bucket=${bucket}  prefix=${prefix}"
    total_fail=$(( total_fail + fail ))
  elif [[ "${up}" -gt 0 ]]; then
    log "done  uploaded=${up}  skipped=${sk}"
  else
    log "in sync  skipped=${sk}"
  fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "finished  total_uploaded=${total_up}  total_skipped=${total_sk}  errors=${total_fail}"

[[ "${total_fail}" -eq 0 ]] || exit 1
