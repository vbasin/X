#!/usr/bin/env bash
# GPU job progress snapshot (ADR-073). Operator entry: make gpu-monitor
set -uo pipefail
REPO="${VTECH_REPO:-/opt/vtech}"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source "$REPO/venv/bin/activate"
set -a && source "$REPO/.env" && set +a
export PYTHONPATH="$REPO"

echo "---- health probe ----"
python -m src.gpu_runner.health_monitor --once || true

state="$(python -c "import os; from src.common.gpu_provider import GpuProvider; print(GpuProvider()._state(os.environ['GPU_INSTANCE_OCID']))" 2>/dev/null || echo unknown)"
echo
echo "---- GPU progress.json (current run) ----"
if [ "$state" != "RUNNING" ]; then
  echo "(GPU is ${state} — skip SSH; start a job or pass --start only if you intend to bill)"
else
  python -m src.gpu_runner.health_monitor --dump || echo "(GPU SSH failed)"
fi

echo
echo "---- local latest.json ----"
root="${VTECH_DATA_DIR:-/opt/vtech/data}"
latest="$root/ops/gpu_health/latest.json"
if [ -f "$latest" ]; then cat "$latest"; else echo "(no $latest)"; fi
