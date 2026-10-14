#!/usr/bin/env bash
# =============================================================================
# pg_backup.sh — pgBackRest daily backup wrapper
# =============================================================================
# Schedule:  Sunday → full backup
#            Mon–Sat → incremental backup
#
# Requires:  /etc/pgbackrest/pgbackrest.conf configured with OCI S3 credentials
#            pgBackRest stanza must already exist: pgbackrest --stanza=vtech stanza-create
#
# Run as:    postgres user (systemd: User=postgres)
# Logs:      /opt/vtech/logs/pg-backup.log  (stdout)
#            /opt/vtech/logs/pg-backup.err   (stderr)
# =============================================================================
set -euo pipefail

STANZA="vtech"
LOG_TAG="pg-backup"

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S') | ${LOG_TAG} | INFO  | $*"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') | ${LOG_TAG} | WARN  | $*" >&2; }
err()  { echo "$(date '+%Y-%m-%d %H:%M:%S') | ${LOG_TAG} | ERROR | $*" >&2; }

log "=== pg_backup.sh started ==="

# Determine backup type: Sunday (dow=7) → full, all other days → incr
DOW=$(date +%u)  # 1=Mon … 7=Sun
if [[ "$DOW" -eq 7 ]]; then
    BACKUP_TYPE="full"
else
    BACKUP_TYPE="incr"
fi

log "day_of_week=${DOW} backup_type=${BACKUP_TYPE} stanza=${STANZA}"

# Run the backup
START_TS=$(date +%s)
if pgbackrest --stanza="${STANZA}" backup --type="${BACKUP_TYPE}"; then
    END_TS=$(date +%s)
    ELAPSED=$(( END_TS - START_TS ))
    log "backup=complete type=${BACKUP_TYPE} elapsed_s=${ELAPSED}"
else
    END_TS=$(date +%s)
    ELAPSED=$(( END_TS - START_TS ))
    err "backup=FAILED type=${BACKUP_TYPE} elapsed_s=${ELAPSED}"
    exit 1
fi

log "=== pg_backup.sh finished ==="
