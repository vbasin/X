#!/usr/bin/env bash
# ssh_gpu.sh — connect to the GPU VM by resolving its LIVE public IP via the OCI API.
#
# The GPU's public IP is EPHEMERAL and can change across stop/start (the box powers itself
# off when idle — ADR-022 — and gets a new ephemeral IP on the next START). Hardcoding the IP
# breaks after the first idle-shutdown, so this helper always resolves the current IP first.
#
# Auth: CPU-side ~/.oci/config (same as every other control-plane call). Reads GPU_INSTANCE_OCID
# and GPU_REGION from the environment — source .env first:
#   cd /opt/vtech && set -a && source .env && set +a
#
# Usage:
#   deploy/gpu/ssh_gpu.sh                       # interactive shell
#   deploy/gpu/ssh_gpu.sh 'journalctl -u vtech-gpu-runner -n 40 --no-pager'
#   deploy/gpu/ssh_gpu.sh 'cat /opt/vtech/data/ops/gpu_jobs/*/progress.json'
#   deploy/gpu/ssh_gpu.sh 'tail -f /opt/vtech/data/ops/gpu_jobs/*/job.log'
#   deploy/gpu/ssh_gpu.sh --start 'nvidia-smi'  # START the VM first if it is STOPPED
set -euo pipefail

REPO="${VTECH_REPO:-/opt/vtech}"
KEY="${GPU_SSH_KEY_PATH:-$HOME/.ssh/gpu_vm_key}"
SSH_USER="${GPU_SSH_USER:-opc}"

START=0
if [[ "${1:-}" == "--start" ]]; then START=1; shift; fi

: "${GPU_INSTANCE_OCID:?set GPU_INSTANCE_OCID (source .env first)}"

IP="$(START="$START" "$REPO/venv/bin/python" - <<'PY'
import os, time, oci
cfg = oci.config.from_file(os.path.expanduser(os.environ.get("OCI_CONFIG_FILE", "~/.oci/config")))
region = os.environ.get("GPU_REGION")
if region:
    cfg = {**cfg, "region": region}
ocid = os.environ["GPU_INSTANCE_OCID"]
comp = oci.core.ComputeClient(cfg)
net = oci.core.VirtualNetworkClient(cfg)
inst = comp.get_instance(ocid).data
if inst.lifecycle_state != "RUNNING" and os.environ.get("START") == "1":
    if inst.lifecycle_state == "STOPPED":
        comp.instance_action(ocid, "START")
    deadline = time.time() + 300
    while time.time() < deadline:
        if comp.get_instance(ocid).data.lifecycle_state == "RUNNING":
            time.sleep(20)  # let sshd come up
            break
        time.sleep(6)


def _ip():
    for a in comp.list_vnic_attachments(compartment_id=inst.compartment_id, instance_id=ocid).data:
        v = net.get_vnic(a.vnic_id).data
        if v.is_primary and v.public_ip:
            return v.public_ip
    return ""


print(_ip())
PY
)"

if [[ -z "$IP" ]]; then
  echo "ERROR: could not resolve a public IP for $GPU_INSTANCE_OCID (is it RUNNING? pass --start)" >&2
  exit 1
fi

exec ssh -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
     -o ConnectTimeout=15 "${SSH_USER}@${IP}" "$@"
