"""GPU dispatch plane — ADR-022 (OCI Queue + content-addressed code bundles).

Shared producer/consumer helpers over the managed OCI Queue (``src/common/oci_queue.py``) +
Object Storage (``src/common/object_store.py``). Replaces the hand-rolled Object-Storage marker
FSM (``src/gpu_runner/queue.py``).

Planes
------
* **Control** — OCI Queue. Two logical channels: a single **jobs** channel (CPU→GPU) and a
  per-objective **results** channel (GPU→CPU). Small JSON only (≤256 KB).
* **Data** — Object Storage. Code bundles at ``code/<digest>.tzst``; per-run result artifacts
  (OOF parquet, diagnostics JSON) at ``results/<run_id>/<file>``.

Both the queue and the store honour the same local-backend env var (``VTECH_GPU_QUEUE_LOCAL``)
so the whole plane runs offline in tests; unset in production → OCI Queue + the ``vtech_hot``
bucket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.common.object_store import ObjectStore, make_object_store
from src.common.oci_queue import MessageQueue, make_message_queue

logger = logging.getLogger("vtech.gpu_runner.dispatch")

JOBS_CHANNEL = "jobs"
_RESULTS_PREFIX = "results"


def results_channel(objective_id: str) -> str:
    """Per-objective results channel (GPU→CPU)."""
    return f"{_RESULTS_PREFIX}.{objective_id}"


def make_plane() -> tuple[MessageQueue, ObjectStore]:
    """Return ``(queue, store)``. Local (``VTECH_GPU_QUEUE_LOCAL``) in tests, else OCI Queue +
    the shared ``vtech_hot`` bucket. Both honour the same local env so one dir toggles both."""
    queue, _ = make_message_queue()                                   # local via VTECH_GPU_QUEUE_LOCAL else OCI
    store, _ = make_object_store(local_env="VTECH_GPU_QUEUE_LOCAL")   # same local root else S3 vtech_hot
    return queue, store


# ── producer (CPU) ────────────────────────────────────────────────────────────

def enqueue_job(queue: MessageQueue, payload: dict[str, Any]) -> None:
    """Put one candidate job on the jobs channel. ``payload`` must carry ``objective_id``,
    ``run_id``, ``code_digest``, ``family``, ``spec_params``, ``parquet_dir`` (+ optional
    user-override ``candidate_id``/``feature_selection``/``search_max_rows``)."""
    queue.put(payload, channel=JOBS_CHANNEL)


def iter_job_artifacts(out_dir: str | Path) -> list[Path]:
    """Files the GPU→CPU handoff must ship: OOF/diagnostics, last-fold bundles, progress."""
    out = Path(out_dir)
    if not out.is_dir():
        return []
    files = sorted(out.glob("oof_*.parquet")) + sorted(out.glob("diagnostics_*.json"))
    files += sorted(out.glob("progress.json")) + sorted(out.glob("progress_*.json"))
    files += sorted(out.glob("job.log")) + sorted(out.glob("job_stderr.log"))
    folds = out / "folds"
    if folds.is_dir():
        files.extend(sorted(p for p in folds.rglob("*") if p.is_file()))
    models = out / "models"
    if models.is_dir():
        files.extend(sorted(p for p in models.rglob("*") if p.is_file()))
    return files


def upload_job_artifacts(store: ObjectStore, run_id: str, out_dir: str | Path) -> list[str]:
    """Put job artifacts at ``results/<run_id>/<rel>``. Returns relative keys."""
    out = Path(out_dir)
    files: list[str] = []
    for f in iter_job_artifacts(out):
        rel = f.relative_to(out).as_posix()
        store.put(f"{_RESULTS_PREFIX}/{run_id}/{rel}", f.read_bytes())
        files.append(rel)
    return files


def _result_rel_names(store: ObjectStore, run_id: str, body: dict[str, Any]) -> list[str]:
    """Union of the queue ``result_files`` list and whatever is actually in the outbox.

    The GPU VM runner process imports ``publish_result`` from its installed tree, not
    the job code bundle. A stale runner only globs root ``oof_*.parquet`` and leaves
    ``result_files`` empty when Custom Test nested weights under ``models/``. The
    bundle's ``job_exec`` uploads those objects; the CPU collector must list the
    prefix so they still land.
    """
    prefix = f"{_RESULTS_PREFIX}/{run_id}/"
    names = list(body.get("result_files") or [])
    try:
        for key in store.list(prefix):
            rel = key[len(prefix):].lstrip("/") if key.startswith(prefix) else ""
            if rel:
                names.append(rel)
    except Exception as exc:  # noqa: BLE001 — fall back to the queue list
        logger.warning("result_prefix_list_failed run_id=%s err=%s", run_id, exc)
    seen: set[str] = set()
    out: list[str] = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def collect_result(
    store: ObjectStore, body: dict[str, Any], out_dir: str | Path
) -> tuple[str | None, dict[str, Any] | None]:
    """Download a finished run's OOF + diagnostics from Object Storage into *out_dir*.

    Returns ``(oof_path, diagnostics_dict)``. Reads the object keys listed in
    ``body["result_files"]`` **and** ``results/<run_id>/`` (prefix list), then
    **deletes** each successfully-downloaded object — ``results/`` is a transient
    GPU→CPU handoff outbox, not an archive; the durable copy is the collected
    ``data/research/search/`` or ``research/custom_test/`` run dir.
    """
    run_id = body["run_id"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    oof_path: str | None = None
    diag: dict[str, Any] | None = None
    for name in _result_rel_names(store, run_id, body):
        key = f"{_RESULTS_PREFIX}/{run_id}/{name}"
        data = store.get(key)
        if data is None:
            continue
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        leaf = Path(name).name
        if leaf.startswith("oof_") or leaf == "oof.parquet":
            oof_path = str(dest)
        elif leaf.startswith("diagnostics_") or leaf == "diagnostics.json":
            try:
                diag = json.loads(dest.read_text())
            except Exception:
                diag = None
        try:
            store.delete(key)                     # verified local copy → clean the outbox
        except Exception:  # noqa: BLE001 — cleanup is best-effort; a leak is non-fatal
            pass
    return oof_path, diag


# ── consumer (GPU) ─────────────────────────────────────────────────────────────

def publish_result(
    queue: MessageQueue,
    store: ObjectStore,
    *,
    objective_id: str,
    run_id: str,
    code_digest: str,
    ran_digest: str,
    result: dict[str, Any],
    out_dir: str | Path,
) -> None:
    """Upload OOF+diagnostics from *out_dir* to Object Storage, then put a small result summary
    on the objective's results channel. Called by the GPU runner after a job finishes."""
    files = list(dict.fromkeys(
        list(result.get("result_files") or []) + upload_job_artifacts(store, run_id, out_dir)
    ))
    body = {
        "run_id": run_id,
        "objective_id": objective_id,
        "code_digest": code_digest,
        "ran_digest": ran_digest,
        "status": result.get("status"),
        "family": result.get("family"),
        "score": result.get("score") or {},
        "meets_target": bool(result.get("meets_target")),
        "error": result.get("error"),
        "elapsed_s": float(result.get("elapsed_s") or 0.0),
        "result_files": files,
        # Echo the EXECUTED candidate config (tuned params + feature selection) so the
        # CPU collector records reality instead of catalog defaults — required for the
        # champion-finalize bridge to reproduce the winner exactly.
        "spec_params": result.get("spec_params") or {},
        "feature_selection": result.get("feature_selection"),
    }
    queue.put(body, channel=results_channel(objective_id))
