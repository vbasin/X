#!/usr/bin/env python3
"""Apply RAM-derived Postgres settings and install systemd units from the repo.

systemd MemoryMax=N% is evaluated from physical RAM at boot, so unit files
do not embed a GB figure from a previous OCI shape. PostgreSQL does not
accept percentages — this script writes conf.d from get_vm_config().

  sudo python /opt/vtech/scripts/ops/apply_vm_resource_limits.py --apply-postgres --install-units --reload

Does not restart Postgres or vtech units unless --restart-postgres /
--restart-live is passed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/opt/vtech")
SYSTEMD_SRC = REPO / "systemd"
SYSTEMD_DST = Path("/etc/systemd/system")
PG_CONF_D = Path("/etc/postgresql/16/main/conf.d")
PG_OVERRIDE = PG_CONF_D / "z-vtech-vm-ram.conf"


def _cfg():
    sys.path.insert(0, str(REPO))
    from src.common.vm_config import get_vm_config
    get_vm_config.cache_clear()
    return get_vm_config()


def postgres_conf_text(cfg) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sb = cfg.postgres_shared_buffers_gb()
    ec = cfg.postgres_effective_cache_gb()
    mw = cfg.postgres_maintenance_work_mem_gb()
    return (
        f"# Generated {ts} by scripts/ops/apply_vm_resource_limits.py — do not edit.\n"
        f"# VM at generate time: {cfg.cpu_count} CPU, {cfg.total_ram_gb:.0f} GB RAM\n"
        f"# Re-run after an OCI resize (then restart postgresql@16-main).\n"
        f"shared_buffers = {sb}GB\n"
        f"effective_cache_size = {ec}GB\n"
        f"maintenance_work_mem = {mw}GB\n"
    )


def install_units() -> list[str]:
    """Symlink repo units into /etc/systemd/system so the tree is SSOT."""
    if not SYSTEMD_DST.is_dir():
        raise SystemExit(f"missing {SYSTEMD_DST}")
    changed = []
    for src in sorted(SYSTEMD_SRC.glob("vtech-*")):
        dest = SYSTEMD_DST / src.name
        if dest.is_symlink() and dest.resolve() == src.resolve():
            continue
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src)
        changed.append(src.name)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-postgres", action="store_true")
    parser.add_argument("--install-units", action="store_true")
    parser.add_argument("--reload", action="store_true", help="systemctl daemon-reload")
    parser.add_argument("--restart-postgres", action="store_true")
    parser.add_argument("--restart-live", action="store_true",
                        help="restart vtech-live-streamer + vtech-live-features")
    parser.add_argument("--enable-cme-timers", action="store_true",
                        help="enable Sunday-open / Friday-close timers")
    args = parser.parse_args()

    cfg = _cfg()
    print(cfg)
    print()
    print(postgres_conf_text(cfg), end="")

    if args.apply_postgres:
        if os.geteuid() != 0:
            print("ERROR: --apply-postgres requires root", file=sys.stderr)
            return 1
        PG_CONF_D.mkdir(parents=True, exist_ok=True)
        PG_OVERRIDE.write_text(postgres_conf_text(cfg))
        os.chmod(PG_OVERRIDE, 0o644)
        print(f"wrote {PG_OVERRIDE}")

    if args.install_units:
        if os.geteuid() != 0:
            print("ERROR: --install-units requires root", file=sys.stderr)
            return 1
        changed = install_units()
        print("symlinked:" if changed else "units already linked:",
              ", ".join(changed) if changed else "(none)")

    if args.reload or args.install_units:
        subprocess.check_call(["systemctl", "daemon-reload"])
        print("daemon-reload ok")

    if args.enable_cme_timers:
        subprocess.check_call([
            "systemctl", "enable", "--now",
            "vtech-live-streamer.timer",
            "vtech-live-streamer-stop.timer",
        ])
        print("enabled CME start/stop timers")

    if args.restart_postgres:
        subprocess.check_call(["systemctl", "restart", "postgresql@16-main"])
        print("restarted postgresql@16-main")

    if args.restart_live:
        subprocess.check_call([
            "systemctl", "restart",
            "vtech-live-streamer.service",
            "vtech-live-features.service",
        ])
        print("restarted live-streamer + live-features")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {exc}", file=sys.stderr)
        raise SystemExit(exc.returncode)
