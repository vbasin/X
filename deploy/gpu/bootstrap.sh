#!/usr/bin/env bash
# deploy/gpu/bootstrap.sh — one-time provisioning for the GPU VM
#
# Run once after the initial `git clone` on the GPU VM:
#
#   git clone https://github.com/vbasin/X.git /opt/vtech
#   cd /opt/vtech
#   bash deploy/gpu/bootstrap.sh
#
# What this script does:
#   1. Safety-checks we are NOT on the CPU VM.
#   2. Applies git sparse-checkout so only the GPU-relevant subset is checked out.
#      Crucially this EXCLUDES systemd/ — so none of the CPU-side timers
#      (vtech-model-trainer, vtech-oci-sync, vtech-pg-backup, etc.) are ever
#      visible in the working tree on the GPU VM.
#   3. Installs Python dependencies (GPU extras).
#   4. Installs and enables ONLY the gpu-runner systemd service.
#
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${REPO_ROOT}/venv"
SERVICE_SRC="${REPO_ROOT}/deploy/gpu/vtech-gpu-runner.service"
SERVICE_DST="/etc/systemd/system/vtech-gpu-runner.service"

# ---------------------------------------------------------------------------
# 0. Safety check — refuse to run on a host named like the CPU VM
# ---------------------------------------------------------------------------
HOSTNAME=$(hostname -s)
if [[ "$HOSTNAME" == *"chi"* || "$HOSTNAME" == *"cpu"* ]]; then
    echo "ERROR: This script looks like it's running on the CPU VM (hostname=$HOSTNAME)."
    echo "       Run it only on the GPU VM."
    exit 1
fi

echo "==> Bootstrap starting on host: $HOSTNAME"
echo "    Repo root: $REPO_ROOT"

# ---------------------------------------------------------------------------
# 0b. Pull latest code from origin before provisioning.
#     Ensures bootstrap installs the most recent package version.
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking for upstream code updates ..."
bash "${REPO_ROOT}/deploy/gpu/git_update.sh" "${REPO_ROOT}" || true

# ---------------------------------------------------------------------------
# 0c. Bind-mount REPO_ROOT onto /opt/vtech so the service file can use the
#     canonical /opt/vtech paths while the repo lives anywhere on this host.
#     Files under /opt/ receive usr_t/bin_t SELinux labels (readable and
#     executable by confined system services) instead of user_home_t.
# ---------------------------------------------------------------------------
if [[ "$REPO_ROOT" != "/opt/vtech" ]]; then
    echo ""
    echo "==> Setting up /opt/vtech bind mount ..."
    sudo mkdir -p /opt/vtech
    if mountpoint -q /opt/vtech; then
        echo "    Already mounted — skipping."
    else
        sudo mount --bind "$REPO_ROOT" /opt/vtech
        echo "    Mounted: $REPO_ROOT → /opt/vtech"
    fi
    # Persist across reboots
    if ! grep -q "$REPO_ROOT.*bind" /etc/fstab; then
        echo "$REPO_ROOT  /opt/vtech  none  bind  0 0" | sudo tee -a /etc/fstab >/dev/null
        echo "    Added fstab entry."
    fi
    # Fix SELinux labels so services can read/exec from /opt/vtech
    echo "    Running restorecon on /opt/vtech ..."
    sudo restorecon -R /opt/vtech 2>/dev/null || true
    echo "    Done."
fi

# ---------------------------------------------------------------------------
# 1. Git sparse-checkout (cone mode)
#    Included:  src/  config/  deploy/  + all top-level files
#    Excluded:  systemd/  scripts/  docs/  research_cases/  data/  logs/  mlruns/
# ---------------------------------------------------------------------------
echo ""
echo "==> Applying git sparse-checkout (cone mode) ..."
cd "$REPO_ROOT"
git sparse-checkout init --cone
git sparse-checkout set src config deploy

echo "    Sparse-checkout active. Included paths:"
git sparse-checkout list | sed 's/^/      /'

echo "    Verifying systemd/ is excluded ..."
if [[ -d "${REPO_ROOT}/systemd" ]]; then
    echo "WARNING: systemd/ directory still exists in working tree."
    echo "         This may mean sparse-checkout did not take effect."
    echo "         Check: git sparse-checkout list"
else
    echo "    OK — systemd/ is not present in working tree."
fi

# ---------------------------------------------------------------------------
# 2. Python environment + GPU dependencies
#    The codebase requires Python >=3.10 (union-type syntax etc.). A stale
#    system python3 (e.g. 3.6) cannot install modern setuptools and will fail.
# ---------------------------------------------------------------------------
echo ""
echo "==> Locating a Python >=3.10 interpreter ..."
PYTHON_BIN=""
for cand in python3.12 python3.11 python3.10; do
    if command -v "$cand" >/dev/null 2>&1; then
        PYTHON_BIN="$cand"; break
    fi
done
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: no Python >=3.10 found (system python3 is $(python3 --version 2>&1))."
    echo "       Install one, then re-run this script:"
    echo "         Oracle Linux:  sudo dnf install -y python3.12 python3.12-devel"
    echo "         Ubuntu/Debian: sudo apt-get install -y python3.12 python3.12-venv python3.12-dev"
    exit 1
fi
echo "    Using: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# Recreate the venv if it's missing or built with a too-old interpreter.
NEED_VENV=1
if [[ -x "${VENV}/bin/python" ]]; then
    if "${VENV}/bin/python" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)' 2>/dev/null; then
        NEED_VENV=0
    fi
fi
if [[ "$NEED_VENV" == "1" ]]; then
    if [[ -d "$VENV" ]]; then
        BACKUP="${VENV}.old.$(date +%s)"
        echo "    Existing venv is too old (<3.10) — moving aside to ${BACKUP}"
        mv "$VENV" "$BACKUP"
    fi
    echo "    Creating fresh venv with $PYTHON_BIN ..."
    "$PYTHON_BIN" -m venv "$VENV"
else
    echo "    Existing venv is Python >=3.10 — reusing $VENV"
fi

source "${VENV}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet

echo "==> Installing package (GPU extras: training + agents) ..."
pip install -e "${REPO_ROOT}[gpu]" --quiet

# ---------------------------------------------------------------------------
# 3. .env file — must be provided separately (secrets are never in git)
# ---------------------------------------------------------------------------
if [[ ! -f "${REPO_ROOT}/.env" ]]; then
    echo ""
    echo "WARNING: ${REPO_ROOT}/.env is missing."
    echo "         Copy it from the CPU VM or a secrets store before starting the runner."
    echo "         Required keys:"
    echo "           MLFLOW_S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,"
    echo "           AWS_DEFAULT_REGION, VTECH_DATA_DIR (point at the block volume, e.g. /data)"
fi

# ---------------------------------------------------------------------------
# 3b. Feature data — pull parquet files from OCI Object Storage
#     The GPU VM never stores parquet in git (data/ is in .gitignore and the
#     sparse-checkout excludes data/).  Instead they live in the vtech_hot
#     bucket under parquet_raw1s/, parquet_path1s/, parquet_pathlabels/,
#     parquet_labels_15s/, parquet_labels_5s/, and timeframe=*/ (feature
#     hive, including 5s), synced there by the CPU VM's oci_sync / vtech-oci-sync timer.
#     The runner service repeats this at every boot via ExecStartPre, so this
#     initial pull just pre-warms the cache to avoid a cold-start delay on
#     the first job wave.
#
#     Destination honours VTECH_DATA_DIR (the GPU VM's block volume, e.g.
#     /data) so parquet lands on the large volume, not the small boot disk.
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    echo ""
    echo "==> Initial parquet sync from OCI Object Storage ..."
    # shellcheck disable=SC1091
    set -a && source "${REPO_ROOT}/.env" && set +a
    DATA_DIR="${VTECH_DATA_DIR:-${REPO_ROOT}/data}"
    FEATURES_ROOT="${DATA_DIR}/sources/features"
    mkdir -p "${FEATURES_ROOT}"
    echo "    Destination: ${FEATURES_ROOT} (VTECH_DATA_DIR=${VTECH_DATA_DIR:-<unset>})"
    if "${VENV}/bin/python" "${REPO_ROOT}/deploy/gpu/sync_parquet.py" "${FEATURES_ROOT}"; then
        echo "    Parquet sync complete."
    else
        echo "WARNING: Parquet sync failed — runner ExecStartPre will retry at boot."
    fi
else
    echo ""
    echo "SKIP: .env not present yet — parquet sync skipped."
    echo "      After placing .env, run manually:"
    echo "        source .env && \${VENV}/bin/python \${REPO_ROOT}/deploy/gpu/sync_parquet.py \"\${VTECH_DATA_DIR:-data}/sources/features\""
fi

# ---------------------------------------------------------------------------
# 4. systemd service — gpu-runner only, nothing else
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing vtech-gpu-runner.service ..."
if [[ ! -f "$SERVICE_SRC" ]]; then
    echo "ERROR: $SERVICE_SRC not found. Did sparse-checkout complete?"
    exit 1
fi

# The service file uses /opt/vtech (canonical path, now always mapped via the
# bind mount above). Only substitute the service user with the actual running user.
CURRENT_USER="$(id -un)"
sed -e "s|User=ubuntu|User=${CURRENT_USER}|g" \
    "$SERVICE_SRC" | sudo tee "$SERVICE_DST" >/dev/null
echo "    Installed service (User=ubuntu → User=${CURRENT_USER}, paths stay /opt/vtech via bind mount)"
sudo systemctl daemon-reload
sudo systemctl enable vtech-gpu-runner.service
echo "    Enabled: vtech-gpu-runner.service"
echo "    NOTE: service will NOT start now — it starts on next boot, or:"
echo "          sudo systemctl start vtech-gpu-runner.service"

# ---------------------------------------------------------------------------
# GPU idle self-stop (ADR-022 revised) is now OWNED BY THE RUNNER itself: after
# GPU_RUNNER_IDLE_SECS with no work it calls the OCI API to SOFTSTOP this VM
# (--stop-vm-on-idle), so the runner and billed compute halt together. No
# separate idle-shutdown timer/service is installed anymore (the old in-guest
# `shutdown` left the OCI instance RUNNING/billed). Requires the GPU's dynamic
# group to be allowed to use/manage instances in its compartment.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==> Bootstrap complete."
echo "    Next steps:"
echo "      1. Place .env at ${REPO_ROOT}/.env (copy from CPU VM, not committed)."
echo "      2. Confirm feature data is available (sources/features/timeframe=*/ via OCI sync)."
echo "      3. Start the runner:  sudo systemctl start vtech-gpu-runner.service"
echo "         (or reboot — it is enabled on boot)"
