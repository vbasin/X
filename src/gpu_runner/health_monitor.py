"""CPU-side GPU health watchdog (ADR-022 companion) — SSH-based process monitoring.

**Why this exists.** The OCI Queue results poll only surfaces failures the runner *explicitly
reports* (a job that fails and `put`s a `status=failed` result message). It is blind to failures
that produce **no result message at all**:

* the runner service crashing, failing to start, or being **OOM-killed** mid-job;
* a job subprocess hung / deadlocked (no `job_done`, no failure — just silence until the wave
  timeout);
* CUDA / GPU-driver faults (`nvidia-smi` failing);
* the data disk filling up;
* the runner going **down while the VM keeps billing** (e.g. a self-stop that didn't take).

Whenever the GPU VM is `RUNNING`, this watchdog SSHes in on an interval and inspects the live
process / GPU / disk state, logging and recording anomalies the queue poll can never see. It runs
on the **CPU VM** (it needs `~/.oci/config` to resolve the GPU's live public IP + the SSH key).

Run as a daemon (systemd unit ``vtech-gpu-health-monitor`` on the CPU VM):

    python -m src.gpu_runner.health_monitor --daemon      # engage whenever the GPU is RUNNING
    python -m src.gpu_runner.health_monitor --once        # single check now (VM must be RUNNING)

Anomalies are logged at ERROR and appended to ``$VTECH_DATA_DIR/gpu_health/alerts.jsonl``; the
latest full report is written to ``$VTECH_DATA_DIR/gpu_health/latest.json``.
"""

from __future__ import annotations

from src.common.data_paths import data_root, gpu_health_dir

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("vtech.gpu_health")

_DATA_ROOT = data_root()
_HEALTH_DIR = gpu_health_dir()

# Health probe run on the GPU over `ssh … bash -s` (script fed via stdin → no quoting pain).
# `__SINCE__` is substituted with the journal look-back window (seconds).
_HEALTH_SCRIPT = r"""
set +e
echo "RUNNER_ACTIVE=$(systemctl is-active vtech-gpu-runner 2>/dev/null)"
echo "RUNNER_SUBSTATE=$(systemctl show -p SubState --value vtech-gpu-runner 2>/dev/null)"
echo "RUNNER_RESULT=$(systemctl show -p Result --value vtech-gpu-runner 2>/dev/null)"
echo "RUNNER_ERRORS=$(journalctl -u vtech-gpu-runner --since '__SINCE__ seconds ago' --no-pager 2>/dev/null | grep -icE 'traceback|job_failed|exception|cuda error|out of memory|fatal')"
echo "LASTERR=$(journalctl -u vtech-gpu-runner --since '__SINCE__ seconds ago' --no-pager 2>/dev/null | grep -iE 'traceback|job_failed|exception|error|fatal' | tail -1 | tr -d '\r' | cut -c1-180)"
echo "OOM_KILLS=$(journalctl -k --since '__SINCE__ seconds ago' --no-pager 2>/dev/null | grep -icE 'out of memory|killed process|oom-kill')"
if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi >/dev/null 2>&1 && echo "GPU_OK=1" || echo "GPU_OK=0"; else echo "GPU_OK=na"; fi
echo "GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
echo "GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')"
echo "JOBS_RUNNING=$(pgrep -cf 'gpu_runner.job_exec' 2>/dev/null || echo 0)"
echo "JOBS_CPU_PCT=$(ps -eo pcpu,args 2>/dev/null | grep 'gpu_runner.job_exec' | grep -v grep | awk '{s+=$1} END {printf \"%d\", s}')"
echo "LOAD1=$(cut -d' ' -f1 /proc/loadavg 2>/dev/null)"
echo "DISK_USE_PCT=$(df --output=pcent /opt/vtech/data 2>/dev/null | tail -1 | tr -dc '0-9')"
PROGRESS=$(python3 -c 'import json,pathlib,time
LIVE={"starting","training","fold_done"}
LIVE_AGE=900.0
now=time.time(); rows=[]
for root in (pathlib.Path("/opt/vtech/data/ops/gpu_jobs"), pathlib.Path("/opt/vtech/data/gpu_jobs")):
    if not root.is_dir(): continue
    for f in root.glob("*/progress.json"):
        try:
            p=json.loads(f.read_text()); mtime=f.stat().st_mtime
        except Exception:
            continue
        lf=p.get("last_fold") or {}
        rec={"run_id":p.get("run_id") or f.parent.name,"family":p.get("family"),"status":p.get("status"),"fold_i":p.get("fold_i"),"n_folds":p.get("n_folds"),"epoch_i":p.get("epoch_i"),"n_epochs":p.get("n_epochs"),"pct_complete":p.get("pct_complete"),"early_stop":p.get("early_stop"),"error":(p.get("error") or "")[:160],"last_fold_n":lf.get("n"),"hit_rate":lf.get("hit_rate"),"cm":(lf.get("confusion") or {}).get("rows_true_cols_pred"),"age_s":round(now-mtime,1),"_mtime":mtime}
        rows.append(rec)
rows.sort(key=lambda r: r.get("_mtime") or 0, reverse=True)
in_flight=[r for r in rows if (r.get("age_s") or 1e18)<=LIVE_AGE and str(r.get("status") or "") in LIVE]
tail=[r for r in rows if (r.get("age_s") or 1e18)<=LIVE_AGE and str(r.get("status") or "") in {"uploading","failed"}]
chosen=in_flight if in_flight else (tail[:1] if tail else rows[:1])
for r in chosen: r.pop("_mtime", None)
print(json.dumps({"n_on_disk":len(rows),"jobs":chosen},separators=(",",":")))' 2>/dev/null)
echo "PROGRESS=${PROGRESS:-[]}"
echo "REPORT_END=1"
"""

# Statuses that mean a job_exec is still working. ``uploading`` is also written
# when job_exec finishes and then left on disk forever — freshness (age_s)
# distinguishes a live upload from leftovers. 15 min covers a quiet XGB fold;
# a silent-but-newest training file still wins via the idle fallback.
LIVE_PROGRESS_STATUSES = frozenset({"starting", "training", "fold_done"})
LIVE_PROGRESS_AGE_S = 900.0

# Stdlib-only dump run on the GPU (SSH stdin is already the probe script, so
# gpu-monitor uses this as a separate bash -s payload).
_DUMP_SCRIPT = r"""
set +e
python3 -c 'import json,pathlib,time
LIVE={"starting","training","fold_done"}
LIVE_AGE=900.0
now=time.time(); rows=[]
for root in (pathlib.Path("/opt/vtech/data/ops/gpu_jobs"), pathlib.Path("/opt/vtech/data/gpu_jobs")):
    if not root.is_dir(): continue
    for f in root.glob("*/progress.json"):
        try:
            p=json.loads(f.read_text()); mtime=f.stat().st_mtime
        except Exception:
            continue
        rows.append((now-mtime, f, p))
rows.sort(key=lambda t: t[0])
in_flight=[t for t in rows if t[0]<=LIVE_AGE and str(t[2].get("status") or "") in LIVE]
tail=[t for t in rows if t[0]<=LIVE_AGE and str(t[2].get("status") or "") in {"uploading","failed"}]
chosen=in_flight if in_flight else (tail[:1] if tail else rows[:1])
if not chosen:
    print("(none — no job in flight or bundle predates ADR-073)")
else:
    print("(showing %d current / %d on disk)" % (len(chosen), len(rows)))
    for age, f, p in chosen:
        print("== %s  age=%.0fs  status=%s ==" % (f, age, p.get("status")))
        print(json.dumps(p, indent=2, default=str))
        print()
    print("---- fold checkpoints (current run) ----")
    n=0
    for age, f, p in chosen:
        d=f.parent / "folds"
        if not d.is_dir(): continue
        for ff in sorted(d.glob("fold_*.json")):
            print("== %s ==" % ff)
            print(ff.read_text())
            print()
            n+=1
    if n==0:
        print("(none yet)")
'
echo "REPORT_END=1"
"""


def parse_report(raw: str) -> dict[str, str]:
    """Parse the ``KEY=value`` probe output into a dict (values may themselves contain ``=``)."""
    report: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line:
            key, _, val = line.partition("=")
            report[key.strip()] = val.strip()
    return report


def select_current_progress(
    jobs: list[dict],
    *,
    live_age_s: float = LIVE_PROGRESS_AGE_S,
) -> list[dict]:
    """Keep in-flight heartbeat(s); drop leftover ``progress.json`` from prior runs.

    Live = ``starting`` / ``training`` / ``fold_done``, or ``uploading`` /
    ``failed`` with ``age_s`` within *live_age_s*. If nothing matches (idle GPU,
    or a payload without ``age_s``), return the single newest file so the
    operator still sees the last job instead of an empty line.
    """
    if not jobs:
        return []

    def _age(j: dict) -> float | None:
        v = j.get("age_s")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _mtime(j: dict) -> float:
        v = j.get("mtime")
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    ranked = sorted(
        jobs,
        key=lambda j: (_mtime(j), -(_age(j) if _age(j) is not None else 1e18)),
        reverse=True,
    )
    in_flight: list[dict] = []
    recent_tail: list[dict] = []
    for j in ranked:
        st = str(j.get("status") or "")
        age = _age(j)
        if age is None:
            if st in LIVE_PROGRESS_STATUSES:
                in_flight.append(j)
            continue
        if age > live_age_s:
            continue
        if st in LIVE_PROGRESS_STATUSES:
            in_flight.append(j)
        elif st in {"uploading", "failed"}:
            recent_tail.append(j)
    if in_flight:
        return in_flight
    if recent_tail:
        return recent_tail[:1]
    return ranked[:1]


def progress_payload(report: dict[str, str]) -> tuple[list[dict], int | None]:
    """Decode ``PROGRESS=...`` (list or ``{jobs, n_on_disk}``) and keep current jobs."""
    raw = report.get("PROGRESS") or "[]"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], None
    n_on_disk: int | None = None
    if isinstance(data, dict):
        jobs = data.get("jobs") if isinstance(data.get("jobs"), list) else []
        raw_n = data.get("n_on_disk")
        try:
            n_on_disk = int(raw_n) if raw_n is not None else None
        except (TypeError, ValueError):
            n_on_disk = None
    elif isinstance(data, list):
        jobs = data
    else:
        return [], None
    selected = select_current_progress(jobs)
    return selected, n_on_disk


def progress_jobs(report: dict[str, str]) -> list[dict]:
    """Decode the compact ``PROGRESS=...`` JSON and drop stale leftover jobs."""
    jobs, _ = progress_payload(report)
    return jobs


def format_progress_line(
    jobs: list[dict],
    *,
    n_on_disk: int | None = None,
) -> str:
    """One-line summary for the health log (empty string when nothing is in flight)."""
    jobs = select_current_progress(jobs)
    if not jobs:
        return ""
    idle = all(str(j.get("status") or "") not in LIVE_PROGRESS_STATUSES for j in jobs)
    bits: list[str] = []
    for j in jobs:
        fam = j.get("family") or j.get("run_id") or "?"
        n_folds = j.get("n_folds")
        fold_i = j.get("fold_i")
        epoch_i = j.get("epoch_i")
        n_epochs = j.get("n_epochs")
        pct = j.get("pct_complete")
        status = j.get("status") or "?"
        part = f"{fam} {status}"
        if n_folds is not None:
            # fold_i is 0-based while training the current fold
            shown = (int(fold_i) + 1) if fold_i is not None else 0
            part += f" fold={shown}/{n_folds}"
        if n_epochs:
            part += f" epoch={epoch_i or 0}/{n_epochs}"
        if pct is not None:
            part += f" {pct}%"
        hr = j.get("hit_rate")
        if hr is not None:
            part += f" last_hr={hr}"
        bits.append(part)
    line = "; ".join(bits)
    if idle:
        line = "last: " + line
    if n_on_disk is not None and n_on_disk > len(jobs):
        line += f" ({len(jobs)} shown / {n_on_disk} on disk)"
    return line


def evaluate(
    report: dict[str, str],
    *,
    inactive_streak: int = 0,
    inactive_streak_max: int = 3,
    disk_pct_max: int = 90,
) -> list[str]:
    """Return a list of human-readable anomaly strings (empty ⇒ healthy).

    ``inactive_streak`` is the number of *consecutive* prior checks that already found the runner
    down — used so a brief inactive window (boot / normal self-stop) doesn't false-alarm, but a
    runner that stays down while the VM keeps billing does.
    """
    anomalies: list[str] = []
    active = report.get("RUNNER_ACTIVE", "")
    result = report.get("RUNNER_RESULT", "")

    if active == "failed" or result not in ("", "success"):
        anomalies.append(f"runner unit unhealthy (active={active or '?'}, result={result or '?'})")
    if active in ("inactive", "failed") and inactive_streak >= inactive_streak_max:
        anomalies.append(
            f"runner down for {inactive_streak} consecutive checks while the VM is RUNNING — "
            f"probable crash or a self-stop that did not take (compute is still billing)")

    def _int(key: str) -> int:
        try:
            return int(report.get(key, "") or 0)
        except ValueError:
            return 0

    if _int("RUNNER_ERRORS") > 0:
        anomalies.append(
            f"{report['RUNNER_ERRORS']} error line(s) in the runner journal — last: "
            f"{report.get('LASTERR', '') or '(n/a)'}")
    if _int("OOM_KILLS") > 0:
        anomalies.append(
            f"{report['OOM_KILLS']} kernel OOM-kill(s) — a job likely died with no result message")
    if report.get("GPU_OK") == "0":
        anomalies.append("nvidia-smi failed — GPU / driver fault")
    if report.get("DISK_USE_PCT") and _int("DISK_USE_PCT") >= disk_pct_max:
        anomalies.append(f"data disk {report['DISK_USE_PCT']}% full (>= {disk_pct_max}%)")
    # Bad resource utilization: jobs alive but BOTH the GPU and the CPUs are near-idle —
    # a training process that is stalled/misconfigured quietly drags out billed GPU time.
    jobs = _int("JOBS_RUNNING")
    if jobs > 0 and report.get("GPU_UTIL") != "" and _int("GPU_UTIL") == 0 \
            and report.get("JOBS_CPU_PCT") != "" and _int("JOBS_CPU_PCT") < 50:
        anomalies.append(
            f"{jobs} job(s) running but GPU util 0% and job CPU {report.get('JOBS_CPU_PCT')}% — "
            f"stalled or misconfigured job is wasting billed GPU time")
    for job in progress_jobs(report):
        if str(job.get("status") or "") == "failed":
            anomalies.append(
                f"job progress status=failed family={job.get('family') or '?'} "
                f"run_id={job.get('run_id') or '?'} err={job.get('error') or '(n/a)'}")
    return anomalies


class SSHUnreachable(RuntimeError):
    """Raised when the GPU is RUNNING but SSH did not succeed (booting, hung, or network fault)."""


def _ssh_script(ip: str, script: str, *, key: str, user: str, timeout: int = 20) -> str:
    try:
        cp = subprocess.run(
            ["ssh", "-i", key, "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}", f"{user}@{ip}", "bash -s"],
            input=script, capture_output=True, text=True, timeout=timeout + 20,
        )
    except subprocess.TimeoutExpired as exc:
        raise SSHUnreachable(f"ssh timed out to {ip}") from exc
    if cp.returncode != 0 or "REPORT_END=1" not in cp.stdout:
        raise SSHUnreachable(f"ssh rc={cp.returncode}: {(cp.stderr or cp.stdout).strip()[:160]}")
    return cp.stdout


def _run_probe(ip: str, *, key: str, user: str, since_s: int, timeout: int = 20) -> dict[str, str]:
    script = _HEALTH_SCRIPT.replace("__SINCE__", str(int(since_s)))
    return parse_report(_ssh_script(ip, script, key=key, user=user, timeout=timeout))


def dump_progress() -> int:
    """SSH the GPU and print the current ``progress.json`` + fold CMs (not leftovers)."""
    from src.common.gpu_provider import GpuProvider  # noqa: PLC0415

    key = os.environ.get("GPU_SSH_KEY_PATH", os.path.expanduser("~/.ssh/gpu_vm_key"))
    user = os.environ.get("GPU_SSH_USER", "opc")
    prov = GpuProvider()
    ocid = os.environ["GPU_INSTANCE_OCID"]
    state = prov._state(ocid)
    if state != "RUNNING":
        print(f"(GPU is {state} — skip SSH; start a job or pass --start only if you intend to bill)")
        return 0
    ip = prov.public_ip(ocid)
    if not ip:
        print("(GPU RUNNING but public IP not resolved)")
        return 1
    try:
        raw = _ssh_script(ip, _DUMP_SCRIPT, key=key, user=user)
    except SSHUnreachable as exc:
        print(f"(GPU SSH failed: {exc})")
        return 1
    text = raw.replace("REPORT_END=1", "").rstrip()
    print(text)
    return 0


def _record(report: dict[str, str], anomalies: list[str]) -> None:
    _HEALTH_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    payload = {"ts": now, "report": report, "anomalies": anomalies}
    (_HEALTH_DIR / "latest.json").write_text(json.dumps(payload, indent=2))
    if anomalies:
        with (_HEALTH_DIR / "alerts.jsonl").open("a") as fh:
            fh.write(json.dumps(payload) + "\n")


def monitor(
    *,
    once: bool = False,
    interval_s: int = 45,
    idle_poll_s: int = 30,
    ssh_fail_max: int = 5,
    key: str | None = None,
    user: str | None = None,
) -> list[str]:
    """Watch the GPU: while it is RUNNING, SSH-probe every ``interval_s`` and report anomalies.

    In ``--daemon`` mode this loops forever, engaging automatically each time the GPU comes up and
    disengaging when it stops. In ``--once`` mode it runs a single probe (returning the anomalies)
    and requires the VM to be RUNNING. Returns the anomaly list from the last probe.
    """
    from src.common.gpu_provider import GpuProvider  # noqa: PLC0415

    key = key or os.environ.get("GPU_SSH_KEY_PATH", os.path.expanduser("~/.ssh/gpu_vm_key"))
    user = user or os.environ.get("GPU_SSH_USER", "opc")
    prov = GpuProvider()
    ocid = os.environ["GPU_INSTANCE_OCID"]

    ip: str | None = None
    inactive_streak = 0
    ssh_fail_streak = 0
    last_anomalies: list[str] = []

    while True:
        state = prov._state(ocid)
        if state != "RUNNING":
            if once:
                logger.info("GPU is %s — nothing to monitor", state)
                return []
            ip, inactive_streak, ssh_fail_streak = None, 0, 0
            time.sleep(idle_poll_s)
            continue

        if ip is None:                       # resolve the ephemeral public IP once per up-cycle
            ip = prov.public_ip(ocid)
            if ip is None:
                time.sleep(5)
                continue
            logger.info("GPU RUNNING at %s — engaging health watchdog", ip)

        try:
            report = _run_probe(ip, key=key, user=user, since_s=max(interval_s * 2, 90))
            ssh_fail_streak = 0
        except SSHUnreachable as exc:
            ssh_fail_streak += 1
            if ssh_fail_streak >= ssh_fail_max:
                last_anomalies = [f"GPU RUNNING but unreachable via SSH for {ssh_fail_streak} "
                                  f"checks ({exc}) — VM may be hung or the runner wedged"]
                logger.error("GPU HEALTH ANOMALY: %s", last_anomalies[0])
                _record({"RUNNER_ACTIVE": "unreachable"}, last_anomalies)
            else:
                logger.info("ssh probe failed (%d/%d) — VM may still be booting: %s",
                            ssh_fail_streak, ssh_fail_max, exc)
            ip = None                        # force re-resolve next loop (IP may have changed)
            if once:
                return last_anomalies
            time.sleep(interval_s)
            continue

        active = report.get("RUNNER_ACTIVE", "")
        inactive_streak = inactive_streak + 1 if active in ("inactive", "failed") else 0
        last_anomalies = evaluate(report, inactive_streak=inactive_streak)
        _record(report, last_anomalies)
        jobs, n_disk = progress_payload(report)
        prog = format_progress_line(jobs, n_on_disk=n_disk)
        if last_anomalies:
            logger.error("GPU HEALTH ANOMALY: %s", "; ".join(last_anomalies))
        elif prog:
            logger.info("gpu_health ok runner=%s gpu=%s util=%s%% disk=%s%% %s",
                        active, report.get("GPU_OK"), report.get("GPU_UTIL"),
                        report.get("DISK_USE_PCT"), prog)
        else:
            logger.info("gpu_health ok runner=%s gpu=%s util=%s%% disk=%s%%",
                        active, report.get("GPU_OK"), report.get("GPU_UTIL"),
                        report.get("DISK_USE_PCT"))
        if once:
            return last_anomalies
        time.sleep(interval_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU-side SSH health watchdog for the GPU VM (ADR-022)")
    ap.add_argument("--daemon", action="store_true", help="watch forever; engage while RUNNING")
    ap.add_argument("--once", action="store_true", help="one probe now (VM must be RUNNING)")
    ap.add_argument("--dump", action="store_true",
                    help="SSH current progress.json + fold CM (skip leftover runs)")
    ap.add_argument("--interval", type=int, default=int(os.environ.get("GPU_HEALTH_INTERVAL_S", "45")))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.dump:
        raise SystemExit(dump_progress())
    if not args.daemon and not args.once:
        args.once = True
    anomalies = monitor(once=args.once and not args.daemon, interval_s=args.interval)
    if args.once and not args.daemon:
        raise SystemExit(1 if anomalies else 0)


if __name__ == "__main__":
    main()
