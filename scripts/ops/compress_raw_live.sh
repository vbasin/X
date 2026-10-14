#!/usr/bin/env bash
# compress_raw_live.sh — Compress leftover .dbn files under data/sources/raw/live/
#
# Rationale: vtech-live-streamer.service writes wire-format .dbn via
# databento.Live.add_stream(). The rest of the data pipeline expects .dbn.zst,
# so uncompressed .dbn files under raw/live/ are pure redundancy (parsed rows
# are already in TimescaleDB). This script compresses them in place.
#
# Safety:
#   * Skips the current UTC-date directory (the streamer may be writing there).
#   * Skips any file modified within the last 30 minutes (defence-in-depth
#     against a streamer that started late in the previous UTC day and is still
#     appending past midnight).
#   * Uses `zstd --rm` so the original is removed only after successful
#     compression; failure leaves the .dbn untouched.
#   * Processes files serially — never balloons disk usage by more than one
#     output file at a time.
#
# Usage:
#   ./scripts/ops/compress_raw_live.sh              # normal
#   VTECH_LIVE_DIR=/some/path ./compress_raw_live.sh
#   VTECH_COMPRESS_LEVEL=19 ./compress_raw_live.sh  # override level (default 13)
#
# Exit codes: 0 on success (including no-op), 1 on precondition failure.
set -euo pipefail

LIVE_DIR="${VTECH_LIVE_DIR:-/opt/vtech/data/sources/raw/live}"
LEVEL="${VTECH_COMPRESS_LEVEL:-13}"
TODAY_UTC="$(date -u '+%Y-%m-%d')"
# Also skip the Chicago session-date dir (streamer archives there, not UTC).
SESSION_DATE="$(
  PYTHONPATH=/opt/vtech /opt/vtech/venv/bin/python -c \
    'from src.acquisition.options_universe import live_option_session_date; print(live_option_session_date())' \
    2>/dev/null || true
)"

_ts()  { date -u '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "$(_ts) | compress-raw-live | INFO  | $*"; }
warn() { echo "$(_ts) | compress-raw-live | WARN  | $*" >&2; }
err()  { echo "$(_ts) | compress-raw-live | ERROR | $*" >&2; }

command -v zstd >/dev/null 2>&1 || { err "zstd not installed"; exit 1; }
[[ -d "${LIVE_DIR}" ]] || { warn "live dir missing: ${LIVE_DIR} (nothing to do)"; exit 0; }

log "starting  live_dir=${LIVE_DIR}  today_utc=${TODAY_UTC}  session_date=${SESSION_DATE:-unknown}  level=${LEVEL}"

used_before_kb="$(df --output=used /opt/vtech/data | tail -1 | tr -d ' ')"

# Collect candidates (portable; no bash-4 mapfile-with-nul quirk on old shells).
find_args=( -type f -name '*.dbn' ! -name '*.dbn.zst' -not -path "*/${TODAY_UTC}/*" )
if [[ -n "${SESSION_DATE}" && "${SESSION_DATE}" != "${TODAY_UTC}" ]]; then
  find_args+=( -not -path "*/${SESSION_DATE}/*" )
fi
mapfile -d '' candidates < <(
  find "${LIVE_DIR}" "${find_args[@]}" -mmin +30 -print0
)

n_candidates="${#candidates[@]}"
if (( n_candidates == 0 )); then
    log "no eligible .dbn files found (skipping today's dir and files newer than 30min)"
    exit 0
fi

log "found  n=${n_candidates}"

n_ok=0
n_fail=0
total_saved_bytes=0

for f in "${candidates[@]}"; do
    orig_size="$(stat -c '%s' -- "${f}")"
    log "compressing  file=${f}  orig_bytes=${orig_size}"
    if zstd -q -T0 "-${LEVEL}" --rm -- "${f}"; then
        if [[ -f "${f}.zst" ]]; then
            new_size="$(stat -c '%s' -- "${f}.zst")"
            saved=$(( orig_size - new_size ))
            total_saved_bytes=$(( total_saved_bytes + saved ))
            log "compressed   file=${f}.zst  new_bytes=${new_size}  saved_bytes=${saved}"
            n_ok=$(( n_ok + 1 ))
        else
            warn "post-compress missing output: ${f}.zst"
            n_fail=$(( n_fail + 1 ))
        fi
    else
        warn "zstd failed for ${f} (leaving original in place)"
        n_fail=$(( n_fail + 1 ))
    fi
done

used_after_kb="$(df --output=used /opt/vtech/data | tail -1 | tr -d ' ')"
delta_kb=$(( used_before_kb - used_after_kb ))
log "done  ok=${n_ok}  fail=${n_fail}  saved_bytes=${total_saved_bytes}  df_delta_kb=${delta_kb}"

exit 0
