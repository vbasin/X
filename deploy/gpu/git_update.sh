#!/usr/bin/env bash
# deploy/gpu/git_update.sh — check for upstream commits and pull if newer.
#
# Called at boot by vtech-gpu-runner.service (ExecStartPre) and optionally
# by bootstrap.sh for fresh provisioning pulls.
#
# Exit codes:
#   0  — already up-to-date or successfully pulled
#   1  — fetch/pull failed (non-fatal: callers use `|| true`)
#
# Usage:
#   bash deploy/gpu/git_update.sh [repo_dir]
#
#   repo_dir defaults to the directory containing this script's parent (i.e.
#   the repo root when the script lives at deploy/gpu/).

set -euo pipefail

REPO_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LOG_TAG="git_update"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] [$LOG_TAG] $*"; }

cd "$REPO_DIR"

# Verify this is actually a git repo.
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    log "ERROR: $REPO_DIR is not a git repository."
    exit 1
fi

BEFORE=$(git rev-parse HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)

log "Checking for upstream updates (branch: $BRANCH, current: ${BEFORE:0:8})"

# Publication checkout updates never discard local work or publish changes.
if [[ -n "$(git status --porcelain)" ]]; then
    log "ERROR: working tree has local changes; review them before updating."
    exit 1
fi
if [[ "$BRANCH" == "HEAD" ]]; then
    log "ERROR: detached HEAD; select the intended branch manually before updating."
    exit 1
fi

log "Checking for upstream updates (branch: $BRANCH, current: ${BEFORE:0:8})"

# Fetch — fail fast if network is unavailable (service will retry via RestartSec).
if ! git fetch --quiet origin "$BRANCH" 2>&1; then
    log "WARNING: git fetch failed — skipping update."
    exit 1
fi

REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null || echo "")

if [[ -z "$REMOTE" ]]; then
    log "WARNING: could not resolve origin/$BRANCH — skipping update."
    exit 0
fi

if [[ "$BEFORE" == "$REMOTE" ]]; then
    log "Already up-to-date at ${BEFORE:0:8}."
    exit 0
fi

# New commits are available — pull (fast-forward only; never auto-merge).
log "New commits available: ${BEFORE:0:8} → ${REMOTE:0:8}"
if git pull --ff-only --quiet origin "$BRANCH" 2>&1; then
    AFTER=$(git rev-parse HEAD)
    log "Updated: ${BEFORE:0:8} → ${AFTER:0:8}"
    # Show a one-line summary of what changed (useful in journalctl).
    git log --oneline "${BEFORE}..${AFTER}" | sed "s/^/  /"
else
    log "WARNING: git pull --ff-only failed (diverged history or dirty tree) — skipping."
    exit 1
fi
