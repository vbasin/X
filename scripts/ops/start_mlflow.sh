#!/usr/bin/env bash
# start_mlflow.sh — wrapper called by vtech-mlflow.service
#
# Systemd loads /opt/vtech/.env via EnvironmentFile= before invoking this script,
# so PGUSER, PGPASSWORD, PGHOST, PGPORT, MLFLOW_S3_ENDPOINT_URL,
# AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY are all available as shell vars.
#
# Using exec so systemd tracks the mlflow PID directly (no shell wrapper process).
set -euo pipefail

exec /opt/vtech/venv/bin/mlflow server \
    --backend-store-uri "postgresql+psycopg://${PGUSER}:${PGPASSWORD}@${PGHOST:-127.0.0.1}:${PGPORT:-5432}/mlflow" \
    --default-artifact-root "/opt/vtech/data/ops/mlflow_artifacts/" \
    --host 127.0.0.1 \
    --port 5000 \
    --workers 2
