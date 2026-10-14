# ADR-022 — Autonomous GPU Worker: OCI Queue + Content-Addressed Code Bundles + Self-Shutdown

**Date**: 2026-07-10
**Status**: Accepted (foundation modules built; cloud provisioning + cutover pending)
**Relates to**: [ADR-015](./ADR-015-gpu-standard-host-job-queue.md), [ADR-016](./ADR-016-gpu-job-code-identity-contract.md),
[RESEARCH_DIRECTOR.md §3.1](../architecture/RESEARCH_DIRECTOR.md), [SYSTEM.md](../architecture/SYSTEM.md)
**Supersedes**: **ADR-016 in full** (git-sha code-identity contract) and the **lifecycle + queue
mechanics of ADR-015** (§2 Postgres/S3 hand-rolled queue; §1 CPU-owned START/STOP wave). ADR-015's
core decision — *the GPU VM is the single standard test-execution host, no Kubernetes* — is retained.

---

## Context

The queue/lifecycle/code-sync machinery designed in ADR-015/016 works but carries avoidable
cost and complexity, exposed while mapping the CPU⇄GPU workflow:

1. **Lifecycle race + no central owner.** `controller._dispatch_gpu_jobs` (and, independently,
   `research_director/conductor`) each call `GpuProvider.start/stop` per invocation. Two concurrent
   submissions let the first finisher `STOP` the VM out from under the other's running jobs — directly
   contradicting ADR-015's "multiple simultaneous ad-hoc tests" goal. No reaper exists.
2. **git/SHA code-identity tax.** ADR-016's contract requires GitHub reachable on the hot path, a
   producer push/dirty guard that blocks ad-hoc runs, detached-HEAD healing (`git_update.sh`), a
   review-branch round-trip (`push_local_changes.sh`/`poll_gpu_updates.sh`), and a **one-sha-per-runner-lifetime**
   limit (imported modules cannot reload) that blocks concurrent multi-version work.
3. **Hand-rolled S3-marker queue.** `src/gpu_runner/queue.py` re-implements pending/claimed/heartbeat/done
   as Object-Storage objects polled by `LIST`/`GET`, with manual claim races, heartbeats, and stale
   detection (~300 lines) and poll-latency idle-billing gaps.

**Operator-supplied simplifying assumption:** the GPU VM has an OS-level **idle self-shutdown timer
(~15 min)**. This removes the reason the CPU-side coordinator/reaper existed at all.

## Decision

Adopt an **autonomous GPU worker + stateless CPU trigger** model over managed cloud primitives:

1. **The runner owns STOP (via the OCI API).** The GPU runner stays up the whole time the VM is on
   (billing is per VM-`RUNNING`-hour regardless of whether the runner is up), keeps claiming jobs,
   and — after `GPU_RUNNER_IDLE_SECS` with an empty queue plus a final drain-recheck — calls the OCI
   API to **SOFTSTOP its own instance** (instance-principal auth), so the runner and the billed
   compute stop together. The CPU **never** issues STOP. This deletes the coordinator/reaper/lease-TTL
   requirement.

   > **Revision (2026-07-10, live-validated).** The original design used a separate OS-level
   > idle-shutdown timer (`gpu_idle_shutdown.sh` + `vtech-gpu-idle-shutdown.{service,timer}`) running
   > `shutdown -h now`. Live testing proved an **in-guest `shutdown` powers off the OS but leaves the
   > OCI instance in `RUNNING` state — still billed** (confirmed: OS down for 3 min, OCI still
   > `RUNNING`; an OCI-API `STOP` reached `STOPPED` in 6 s). It also opened a ~10-min window where the
   > VM was up but the runner had already exited, stranding jobs dispatched in that gap. Folding the
   > stop into the runner (OCI-API SOFTSTOP) fixes both; the separate script + units are **removed**.
   > **Requires** one new IAM grant: the GPU's dynamic group allowed to manage/use its own instance.

2. **CPU trigger = idempotent `ensure_running`.** `GpuProvider.ensure_running(ocid)` reads the
   instance state and issues `START` **only if `STOPPED`**; if the VM is `STOPPING` (runner mid
   self-stop) it briefly waits for `STOPPED` then `START`s, so a job dispatched during the power-off
   window is never stranded. Safe for any number of concurrent callers; replaces the per-invocation
   `start`/`finally: stop` ownership.

3. **Code identity = content-addressed bundle, not git.** At dispatch the CPU packs the runtime
   code (`src/` + `config/`) into a deterministic tar, zstd-compresses it, and keys it by the
   **sha256 of the raw tar**: `code/<digest>.tzst`, uploaded **only if absent**. The job message
   carries `code_digest`. The GPU fetches by digest (cached on the block volume), **verifies the
   hash**, extracts to `run/<digest>/`, and runs the job in a **subprocess** with that as `PYTHONPATH`.
   - Reproducibility is *stronger* than a git sha (the digest **is** the bytes; verified on download,
     divergence impossible). No GitHub on the hot path; the GPU needs no git or credentials.
   - Different digests coexist → **concurrent multi-version** jobs; the one-sha limit is gone.
   - Ad-hoc/unpushed trees are runnable *and* reproducible (digest recorded for lineage).
   - Slow-changing dependencies are baked into a **base custom image**; only the ~sub-MB code bundle
     moves per dispatch, and only when code changes.

4. **Control plane = OCI Queue (managed).** Replace the S3-marker FSM with OCI Queue: **jobs** channel
   (per `objective_id`) and a **results** channel. Managed **visibility timeout** (≤12 h, extendable)
   replaces `claimed/`+heartbeat and auto-redelivers on worker death; **DLQ** replaces manual retry;
   **long-poll** replaces busy `LIST`. Both VMs reach it over the same OCI API surface they already
   use — no inbound path between VMs. Large blobs (OOF parquet, diagnostics) stay in Object Storage;
   the ≤256 KB message carries pointers.

5. **Results return path is explicit.** On completion the GPU uploads OOF+diagnostics to Object
   Storage and `put`s a small `{run_id, status, score, oof_ptr}` message to the **results** channel.
   The CPU long-polls the results channel, downloads blobs, then runs the CPU-side chain
   (ensemble → champion gate → leaderboard → episodes → results index).

6. **Retained from ADR-015:** single GPU VM as the standard host; persistent-stopped, never
   terminated; shared Object Storage; **no Kubernetes** (Ray/OCI-batch remain the only re-eval trigger
   if the workload outgrows one GPU).

```mermaid
flowchart LR
  S["submitter (run_search/conductor/HPO)"] -->|put code/<digest>.tzst if absent| OS[(Object Storage)]
  S -->|put job {code_digest, ptrs}| Q[(OCI Queue: jobs)]
  S -->|ensure_running: START iff STOPPED| API[OCI Compute]
  API --> G[GPU VM worker]
  G -->|long-poll| Q
  G -->|fetch+verify+extract| OS
  G -->|OOF+diag| OS
  G -->|put result msg| QR[(OCI Queue: results)]
  QR -->|long-poll| S
  G -.->|idle: SOFTSTOP self via OCI API| G2[STOPPED]
```

## Consequences
- **Deleted:** CPU coordinator/reaper (never built), git-sha guard + checkout + `ran_sha` echo,
  `git_update.sh` detached-HEAD healing, `push_local_changes.sh`, `poll_gpu_updates.sh`, GPU-side git;
  the separate idle-shutdown path (`gpu_idle_shutdown.sh` + `vtech-gpu-idle-shutdown.{service,timer}`,
  superseded by the runner's OCI-API self-stop); and (post-cutover) the `src/gpu_runner/queue.py`
  marker FSM. The GPU VM needs no git/GitHub.
- **New deps at runtime:** an OCI Queue resource (10/tenancy default; channels give per-objective
  isolation) + a base image build when `requirements.lock` changes + **one IAM grant** (GPU dynamic
  group may manage/use its own instance, for self-stop). `zstandard`/`oci` already vendored.
- **New failure surface:** an idle self-stop race (GPU stops just as a job arrives). Mitigated by the
  runner's **final drain-recheck** immediately before it issues SOFTSTOP, plus `ensure_running`
  **waiting out `STOPPING`** and then `START`ing so any job enqueued during power-off is picked up.
- **Queue-poll blind spot → CPU-side health watchdog.** The results poll only surfaces failures the
  runner *reports*; it is blind to runner crash/OOM-kill, a hung subprocess, CUDA/driver faults, a
  full disk, or the runner going down while the VM keeps billing. `src/gpu_runner/health_monitor.py`
  (CPU systemd unit `vtech-gpu-health-monitor`) SSHes into the GPU whenever it is `RUNNING` and
  inspects the live runner/GPU/disk state on an interval, logging anomalies to ERROR +
  `$VTECH_DATA_DIR/gpu_health/alerts.jsonl`. Read-only; never acts on the VM.
  Fold/epoch/`%`/confusion live in `ops/gpu_jobs/<run_id>/progress.json`
  ([ADR-073](ADR-073-gpu-job-progress-heartbeat.md)); the watchdog scrapes that
  file on the same SSH probe.
- **No model/label/metric change** — leaderboards remain comparable.

## Rollout (status)
- **Built + live-validated:** `src/common/code_bundle.py`, `src/common/oci_queue.py`,
  `GpuProvider.ensure_running()` + `stop_self()` + `public_ip()`, the runner's `--stop-vm-on-idle`
  self-stop with final drain-recheck, the CPU-side SSH **health watchdog**
  (`src/gpu_runner/health_monitor.py` + `vtech-gpu-health-monitor.service`), + unit tests. Cross-VM
  smoke round-trip green (CPU→Queue→GPU→results). OCI-API self-stop reaches `STOPPED`; in-guest
  `shutdown` proven insufficient (see Decision §1 revision). Watchdog live-validated (healthy +
  runner-down-while-billing detection).
- **Pending (needs provisioning/approval):** confirm/add the GPU dynamic-group IAM grant for
  self-stop; base image; producer/consumer full cutover; deletion of the superseded
  `src/gpu_runner/queue.py` marker FSM; broader live A10 validation.
