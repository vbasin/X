# ADR-073: Live GPU job heartbeat + per-fold OOF checkpoints

## Status
Accepted — implemented 2026-08-27.

## Date
2026-08-27

## Context

GPU Custom Test / search jobs were a black box from the CPU VM:

1. Keras `fit(..., verbose=0)` on LSTM / TCN / CNN / Transformer / `path_tcn`.
2. The GPU runner started `job_exec` with `subprocess.run(capture_output=True)`, so even the one INFO line per fold never reached `journalctl` until the process exited.
3. `job_exec` did not configure logging, so plugin `logger.info` was dropped (default WARNING).
4. `_run_one` wrote the concatenated OOF parquet and `diagnostics_*.json` **once, at the end**. A crash mid-wave discarded every finished fold.

That last item already happened: 1s-TCN wave `20260821T201520Z` completed **16/24** OOF frames (fold 16 in epoch 5/15), then segfaulted. Nothing on disk. ROADMAP blocked a requeue until per-fold OOF was checkpointed.

The CPU health watchdog (`src/gpu_runner/health_monitor.py`, ADR-022) already SSHes every ~45s, but it only saw host health (runner up, nvidia-smi, disk, `job_exec` count) — not fold/epoch/%/confusion.

SSH from this VM already works (`deploy/gpu/ssh_gpu.sh`). The GPU just did not emit anything to read.

## Decision

1. **Heartbeat file, not Keras `verbose=1`.** Each job writes an atomic `progress.json` (fold, epoch, `%`, loss/val_loss, last error, last-fold confusion summary). `%` is a best-case estimate: `(fold_i * n_epochs + epoch_i) / (n_folds * n_epochs)` — EarlyStopping can finish a fold early.
2. **Per-fold checkpoint.** After each fold's OOF predict, write `folds/fold_KK.parquet` + `folds/fold_KK.json` (3×3 confusion, n_signals, hit_rate). Survives a later crash. Confusion is **per finished fold**, not per epoch (a per-epoch CM is a full val predict every step).
3. **Two write roots on the GPU.** Job temp dir (uploaded at success *or* failure) and `$VTECH_DATA_DIR/ops/gpu_jobs/<run_id>/` (SSH-visible while the job is running).
4. **Live log tee.** The runner `Popen`s `job_exec` with merged stdout/stderr, `PYTHONUNBUFFERED=1`, line-buffered tee to `job.log` **and** the systemd journal. `job_exec` configures logging at INFO.
5. **Health watchdog scrapes the heartbeat.** The existing SSH probe adds a compact `PROGRESS=` JSON object (`jobs` + `n_on_disk`). Leftover `ops/gpu_jobs/*/progress.json` from finished runs (`uploading` at 100% is left on disk) are **not** concatenated: only in-flight statuses (`starting` / `training` / `fold_done`, or a recent `uploading`/`failed` within 15 min) are shown; idle GPU falls back to the single newest file, prefixed `last:`. `make gpu-watch` dumps that current run + its fold CMs only. A current `status=failed` heartbeat is an anomaly.
6. **No YAML / label / loss / serving change.** Opt-in is automatic for any job that goes through `job_exec` → `_run_one` → a wired plugin. Plugins no-op when no session is active (unit tests).

Wired fold/epoch hooks: Attention-LSTM, TCN, CNN, Transformer, `path_tcn`, `fit_hazard_folds` (`tcn_hazard` / `patch_hazard`), XGBoost / LightGBM / other tabular (fold-level, no per-tree heartbeat).

## Consequences

* From the CPU VM, a live job is:

  ```bash
  make gpu-monitor    # SSH progress.json + fold CM + health probe
  make gpu-watch      # refresh every 15s
  make gpu-follow     # journalctl -f on the GPU runner
  ```

* A mid-wave crash still leaves `ops/gpu_jobs/<run_id>/folds/fold_*.json` on the GPU disk; the runner also publishes whatever is in the job dir on `status=failed` (no more empty `mkdtemp` on the failure path).
* **Heartbeat + fold checkpoints ride the code bundle** (next dispatch). **Live journal tee needs the GPU runner process updated** (next GPU boot `git_update.sh`, or SSH + pull + `systemctl restart vtech-gpu-runner`). An old runner still captures stdout, but `progress.json` on disk is enough for SSH / health_monitor.
* Closes the ADR-053 requeue blocker (“per-fold OOF is checkpointed”).

## Alternatives considered

**Keras `verbose=1` over SSH.** Rejected. `\r` epoch bars are unreadable in journald; they are not a structured `%` or confusion matrix; `capture_output=True` would still hide them until exit.

**Push heartbeats on the OCI Queue.** Rejected for v1. Queue messages are the job/result contract (ADR-022). A 15-epoch × 6-fold LSTM would spam the results channel. Disk file + existing SSH watchdog is enough.

**Per-epoch confusion matrix.** Rejected. Extra GPU time every epoch; noisy on a ~87% flat TBM book. `loss` / `val_loss` on the heartbeat; CM on fold OOF.

## Related

* [ADR-022](ADR-022-oci-queue-code-bundle-autonomous-gpu.md) — queue + health watchdog (host health only, until this ADR).
* [ADR-053](ADR-053-vol-normalized-path-labels-and-encoder-ablation.md) — 16/24 fold death that motivated per-fold checkpoints.
