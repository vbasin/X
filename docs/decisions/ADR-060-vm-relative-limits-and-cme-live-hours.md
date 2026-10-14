# ADR-060: VM-relative resource limits + CME-hours live streamer

## Status
Accepted — implemented 2026-08-26.

## Date
2026-08-26

## Context

On 2026-08-22 (Saturday) `data/sources/raw/live/2026-08-22/` was created. On
2026-08-25 (Tuesday) no live archive existed. Investigation:

1. **`vtech-live-streamer` is 24/7 `WantedBy=multi-user.target`**, not on a
   CME calendar. A kernel-upgrade reboot at Saturday 07:50 UTC started it
   while CME was shut (Fri 4pm–Sun 5pm CT). Combined with archive paths using
   the **UTC date**, that wrote a weekend-named folder. The 108 MB file in
   that folder ran until Monday 05:30 UTC (Sunday Globex + Monday overnight).

2. **Monday 05:13 UTC global OOM** killed Postgres. The kernel dump showed
   **8 GB physical RAM** on that boot (this host has since been resized).
   systemd `MemoryMax` values were **absolute GB from earlier shapes**
   (`live-features` 32G, `webui` 24G, `streamer` 2G, `feature-builder` 4G).
   `postgresql.conf` still had `shared_buffers = 32GB` /
   `effective_cache_size = 96GB` from the 376 GB era. On an 8 GB (or even
   47 GB) VM those caps do not track reality. MLflow `CPUQuota=50%` is 50%
   of **one** CPU, not of the VM, and does not follow `nproc`.

3. Streamer then crash-looped on `PoolTimeout` (Postgres down), hit
   `StartLimitBurst=5` / 600s, and **stayed failed** through all of Tuesday
   2026-08-25 until the Wednesday 04:18 UTC reboot.

`get_vm_config()` already sizes workers from `/proc/meminfo` and
`os.cpu_count()` at process start. systemd and Postgres did not.

## Options considered

| Option | Pros | Cons |
|--------|------|------|
| A. In-process sleep until Sunday 5pm CT | Handles Saturday reboot | Process looks "active" all weekend; 30s poll was rejected |
| B. systemd CME calendar + exit 0 if started while shut (chosen) | Weekend reboot writes nothing; Sunday timer starts us; weekday boot still works via `WantedBy` | Need `Restart=on-failure` so exit 0 is not bounced |
| C. Remove `WantedBy`, timer-only | Pure calendar | Weekday OCI resize/reboot would stay down until next Sunday |

Resource limits:

| Option | Pros | Cons |
|--------|------|------|
| D. Keep absolute `MemoryMax=32G` | Simple | Wrong after every resize |
| E. systemd `MemoryMax=N%` of host RAM at boot + Postgres conf.d from `get_vm_config()` (chosen) | Tracks resize on reboot; no invented CPU count | Postgres still needs a restart to pick up new `shared_buffers` |
| F. `CPUWeight` / generated `CPUQuota=nproc*k` | Caps CPU | `CPUWeight` is a share, not a core count; worker budgets already live in `vm_config.workers_for()` |

## Decision

1. **CME hours in systemd**, IANA `America/Chicago` (DST-safe):
   - `vtech-live-streamer.timer` — `Sun 17:00` start
   - `vtech-live-streamer-stop.timer` — `Fri 16:00` stop
   - Service stays `WantedBy=multi-user.target` so a **weekday** reboot
     resumes the stream.
   - `Restart=on-failure`. If a weekend boot or a Persistent timer fires
     while CME is shut, Python logs `live_cme_weekend_skip` and **exits 0**
     (no Databento, no `raw/live/` mkdir). Next Sunday timer starts us.

2. **Archive directory = Chicago option-session date**
   (`live_option_session_date`), not UTC. Sunday 5pm CT is still Sunday
   UTC; the folder must be Monday's trading date.

3. **`MemoryMax` is a percent of physical RAM at boot** on the live units.
   Drop MLflow `CPUQuota=50%`. CPU worker counts stay in `get_vm_config()`
   (re-read every process start; not a systemd CPU count).

4. **Postgres `shared_buffers` / `effective_cache_size` /
   `maintenance_work_mem`** are written to
   `/etc/postgresql/16/main/conf.d/z-vtech-vm-ram.conf` by
   `scripts/ops/apply_vm_resource_limits.py` from live RAM (25% / 50% / 8%,
   clamped). `vtech-vm-resources.service` runs before Postgres on boot.
   Changing `shared_buffers` still requires a Postgres restart.

5. **`StartLimitIntervalSec=0`** on the streamer and live-features so a
   Postgres-down burst cannot disable the unit for the rest of the week.
   Definition bootstrap retries then degrades; the DBN archive still opens.

## Consequences

- Saturday reboot no longer creates `raw/live/YYYY-MM-DD` for a weekend date.
- Tuesday-class gaps from start-limit after OOM should not recur.
- After an OCI resize: reboot (or run the apply script) so MemoryMax % and
  Postgres conf.d match the new shape. Restart Postgres to apply
  `shared_buffers`.
- Compress-raw-live skips the **session-date** directory as well as UTC today.

## Links

- [SYSTEM.md](../architecture/SYSTEM.md) — live VM facts via `get_vm_config()`
- [DATA_PIPELINE.md](../architecture/DATA_PIPELINE.md) — live archive path
- [TROUBLESHOOTING.md](../operations/TROUBLESHOOTING.md) — streamer start-limit / weekend files
