# ADR-015 — GPU VM as the Standard Test-Execution Host + Postgres/S3 Job Queue

> **Partially superseded by [ADR-022](./ADR-022-oci-queue-code-bundle-autonomous-gpu.md) (2026-07-10):**
> the CPU-owned START/STOP wave (§1) and the Postgres/S3 hand-rolled job queue (§2) are replaced by an
> autonomous self-shutting-down GPU worker + idempotent `ensure_running` + managed OCI Queue. The
> retained core decision is *single GPU VM as the standard test host, no Kubernetes*.

**Date**: 2026-07-02  
**Status**: Accepted (design; build deferred until the GPU VM is provisioned)  
**Relates to**: [PHASE-23-MODEL-SELECTION-SEARCH.md](../roadmap/PHASE-23-MODEL-SELECTION-SEARCH.md) (§6 hardware mapping),
[RESEARCH_DIRECTOR.md](../architecture/RESEARCH_DIRECTOR.md) (§3.1 GPU execution plane),
[ADR-012](./ADR-012-gated-promotion-mlflow-lineage.md) (promotion gate the search feeds)  
**Supersedes**: the Phase 23 §6 "CPU families fan out on autoscaled OCPU, DL families on the GPU"
split (amended below — not a separate ADR)

---

## Context

Phase 23 W3's first full-fidelity search ran the 4 CPU supervised families on the CPU VM and took
~1h02m. Investigation showed the runtime is **training-bound** (RandomForest ~24 min; logistic
regression hit `lbfgs` non-convergence at `max_iter=500`; xgboost 500 trees × 3 folds), not
fold-build-bound (fold build is ~8s). The operator decided:

1. **All tests should run on the GPU VM**, whose CPU and RAM exceed the CPU VM's — so it becomes the
   *standard* test-execution host regardless of whether a candidate needs the GPU.
2. The plane must support **multiple simultaneous ad-hoc tests**, potentially across **multiple trade
   objectives** submitted at different times.

The existing plane primitives ([gpu_provider.py](../../src/common/gpu_provider.py)) implement a
**persistent-stopped** OCI GPU VM (`START → run → STOP`, never terminate; `git checkout <sha>` code
sync; shared OCI Object Storage), but: the GPU-side runner (`src/gpu_runner/`) is **unbuilt**
(Phase 20 W9), the search controller's GPU path is an in-process **stub**, dispatch is **FIFO
single-job**, and the GPU VM is **not yet provisioned** (`GPU_INSTANCE_OCID` / `GPU_SSH_HOST` unset).

Kubeflow/Kubernetes was evaluated and **rejected** for this scale (single GPU + a few OCPU + S3 +
GitHub): a standing K8s cluster is disproportionate cost/ops for a single-GPU, low-concurrency
workload. K8s + Kueue/Volcano/MIG belongs to the Variant C (multi-GPU continuous-NAS) north star.

## Decision

1. **The GPU VM is the standard test-execution host** for the model-selection search (and HPO). CPU
   families use its many cores + large RAM; `xgboost`/`lightgbm` use `device=cuda`; DL families use
   the GPU; pure-sklearn families (`random_forest`, `logistic_regression`, `hmm`, `gmm`, `kmeans`,
   `dbscan`) stay on CPU cores (or RAPIDS cuML later). This **supersedes** the §6 OCPU-fan-out split.
   Cost is bounded by `START`-on-wave / `STOP`-on-drain (scale-to-zero when idle).

2. **Queue = Postgres (coordinator) + S3 (payloads).** A `job_queue` table holds each job
   (`objective_id`, `run_id`, `git_sha`, spec pointer, resource needs, `status`, `priority`) with
   **atomic claiming** (`SELECT … FOR UPDATE SKIP LOCKED`) so exactly one worker takes a job. Job
   **specs and results** (spec JSON, OOF parquet, checkpoints, heartbeats) travel via the shared
   **S3** bucket both VMs already use. DB = coordination/locking; S3 = large blobs shared between VMs.

3. **Concurrency via resource-aware bin-packing on one host.** The runner admits N jobs concurrently
   sized by GPU memory (MPS / time-slice; MIG if the shape supports it) and CPU/RAM. Each job runs in
   a **resource-isolated unit** — a single GPU **container image** ([Dockerfile.gpu](../../Dockerfile.gpu))
   run per-job with cgroup caps, or (lighter) a cgroup-limited process. **Per-candidate images are
   NOT built** — every candidate is a plugin in one repo, so one image + `git checkout <sha>` gives
   code reproducibility; per-experiment images are reserved for Variant C (arbitrary mutated code).

4. **Multi-objective isolation is preserved by `objective_id` namespacing** (already in place):
   `config/objectives/<id>.yaml`, `data/evaluation/<id>/` (contract + leaderboard), `data/search/<id>/`,
   `data/hpo/<id>/`, `data/scoping/<id>/`. **Shared** across objectives: code, the parquet feature
   cache (features shared; **labels derived per-objective**), the GPU VM + queue, S3, GitHub, MLflow.
   The queue is objective-agnostic (jobs tagged with `objective_id`) with fair scheduling so one
   objective's wave cannot starve another.

5. **No Kubeflow / Kubernetes.** Reuse the persistent-stopped GPU VM + Postgres/S3 queue + `gpu_provider`.
   Re-evaluate only if the workload outgrows one GPU (next step would be Ray or OCI batch, not Kubeflow).

6. **Build is deferred until the GPU VM is provisioned.** Prerequisite: stand up the OCI GPU VM and
   set `GPU_INSTANCE_OCID` / `GPU_SSH_HOST` / `GPU_SSH_USER` / `GPU_SSH_KEY_PATH`. The shared
   `gpu_runner` (Phase 20 W9) is built once and reused by both training and the search.

## Consequences
- Simpler execution model (one host), leveraging the larger box; loses horizontal OCPU fan-out —
  acceptable at current scale, and the queue is written host-agnostic so a second worker can be added.
- Premium GPU-instance rate is paid for CPU-only work — mitigated by scale-to-zero START/STOP.
- Phase 23 GPU testing now **depends on Phase 20 W9** (shared `gpu_runner`) — sequence accordingly.
- A precursor local improvement landed independently: a **shared feature/label fold cache**
  (`src/search/fold_cache.py`, keyed by parquet fingerprint + label params, memory-mapped) so parallel
  workers / HPO trials / objectives reuse the load+label step (8s → 0.01s) — but the search runtime
  remains **training-bound**, so the GPU plane (this ADR) is the real throughput fix.
- Metrics/leaderboards remain per-`(objective_id, contract_version)`; no comparability change.
