"""gpu_runner.job_exec — run ONE candidate from a spec file (ADR-022 subprocess executor).

The runner invokes this as a **subprocess** with ``cwd``/``PYTHONPATH`` set to the extracted
content-addressed code bundle, so the job executes exactly the dispatched code version
(independent of the long-lived runner's own code, and safe for concurrent multi-version waves).

Reads a job spec JSON, reconstructs the catalog entry (honouring tuned params + USER-OVERRIDE
feature selection / row cap), runs the search controller's ``_run_one`` at full walk-forward
fidelity into ``<out_dir>``, and writes ``result.json`` alongside the ``oof_*.parquet`` /
``diagnostics_*.json`` artifacts the runner then uploads.

Usage::

    python -m src.gpu_runner.job_exec <spec.json> <out_dir>
"""

from __future__ import annotations

from src.common.data_paths import gpu_jobs_dir

import dataclasses
import json
import logging
import sys
from pathlib import Path


def reconstruct_catalog_entry(spec: dict) -> object:
    """Rebuild the CatalogEntry the CPU controller dispatched (params + opt-in HPO)."""
    from src.models.catalog import get_entry

    entry = get_entry(spec["family"])
    repl: dict = {}
    overrides = spec.get("spec_params") or {}
    if overrides:                                   # tuned/custom params
        repl["params_default"] = {**entry.params_default, **overrides}
    # USER-OVERRIDE candidate: reconstruct the exact candidate the CPU controller dispatched.
    if spec.get("candidate_id"):
        repl["name"] = spec["candidate_id"]
        repl["plugin_family"] = spec["family"]
        repl["feature_view"] = None                 # feature_selection supersedes a named view
        ss = spec.get("search_space") or {}
        repl["search_space"] = dict(ss) if ss else {}
    if "feature_selection" in spec:
        repl["feature_selection"] = spec["feature_selection"]
    if "search_max_rows" in spec:
        repl["search_max_rows"] = spec["search_max_rows"]
    return dataclasses.replace(entry, **repl) if repl else entry


def _sync_data_prefix(prefix: str, pdir: Path) -> None:
    """Size+mtime sync ``features_*.parquet`` from Object Storage into *pdir*."""
    if not prefix:
        return
    try:
        from src.gpu_runner.parquet_sync import pull_features
        pull_features(prefix, pdir)
    except Exception as exc:  # noqa: BLE001
        print(f"data_sync skipped: object store unavailable ({exc})", file=sys.stderr)


def _sync_labels_prefix(spec: dict) -> None:
    """Pull persisted TBM labels when the job carries ``labels_s3_prefix``."""
    pfx = str(spec.get("labels_s3_prefix") or "")
    kind = str(spec.get("labels_kind") or "")
    if not pfx:
        return
    try:
        from src.common.data_paths import data_root
        from src.gpu_runner.parquet_sync import pull_features
    except Exception as exc:  # noqa: BLE001
        print(f"labels_sync skipped: import ({exc})", file=sys.stderr)
        return
    dest_raw = str(spec.get("labels_s3_dest") or "")
    dest = Path(dest_raw) if dest_raw else data_root() / "sources" / "labels" / kind
    try:
        pull_features(pfx, dest, glob="labels_*.parquet")
        pull_features(pfx, dest, glob="_contract.json")
    except Exception as exc:  # noqa: BLE001
        print(f"labels_sync skipped: {exc}", file=sys.stderr)


def _sync_context_prefixes(spec: dict) -> None:
    """Pull 1h/1d HTF hives when the job carries ``context_s3_prefixes`` (ADR-102)."""
    prefixes = spec.get("context_s3_prefixes") or []
    if not prefixes:
        return
    try:
        from src.common.data_paths import native_tf_features_root
        from src.gpu_runner.parquet_sync import pull_features
    except Exception as exc:  # noqa: BLE001
        print(f"context_sync skipped: import ({exc})", file=sys.stderr)
        return
    root = native_tf_features_root()
    for raw in prefixes:
        pfx = str(raw or "").strip()
        if not pfx:
            continue
        if not pfx.endswith("/"):
            pfx += "/"
        dest = root / pfx.strip("/")
        try:
            pull_features(pfx, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"context_sync skipped prefix={pfx}: {exc}", file=sys.stderr)


def _sync_path1s_prefixes(spec: dict) -> None:
    """Pull 1s path + path-label parquets when the job carries those prefixes."""
    try:
        from src.common.data_paths import parquet_path1s_dir, parquet_pathlabels_dir
        from src.gpu_runner.parquet_sync import pull_features
    except Exception as exc:  # noqa: BLE001
        print(f"path1s_sync skipped: import ({exc})", file=sys.stderr)
        return
    p1 = spec.get("path1s_s3_prefix")
    pl = spec.get("pathlabels_s3_prefix")
    if p1:
        try:
            pull_features(str(p1), parquet_path1s_dir(), glob="path_*.parquet")
        except Exception as exc:  # noqa: BLE001
            print(f"path1s_sync skipped: {exc}", file=sys.stderr)
    if pl:
        try:
            pull_features(str(pl), parquet_pathlabels_dir(), glob="labels_*.parquet")
        except Exception as exc:  # noqa: BLE001
            print(f"pathlabels_sync skipped: {exc}", file=sys.stderr)


def _configure_logging() -> None:
    """INFO to stderr so the runner tee + journal see fold/epoch lines (ADR-073)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass


def _run(spec_path: str, out_dir: str) -> int:
    spec = json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _configure_logging()

    # No-ML connectivity check (ADR-022 smoke): prove bundle extraction + subprocess + result
    # publishing end-to-end without importing or training anything.
    if spec.get("smoke"):
        (out / "result.json").write_text(json.dumps({
            "candidate_id": spec.get("run_id", "smoke"), "family": "__smoke__",
            "status": "ok", "score": {"smoke": True}, "meets_target": False,
            "error": None, "elapsed_s": 0.0,
        }))
        return 0

    from src.objectives.loader import load_objective
    from src.search.controller import _run_one
    from src.training.job_progress import finish_session, mark_failed, start_session

    run_id = str(spec.get("run_id") or "unknown")
    durable = gpu_jobs_dir() / run_id
    start_session(
        run_id=run_id,
        family=spec.get("family"),
        objective_id=spec.get("objective_id"),
        out_dir=out,
        durable_dir=durable,
    )

    objective = load_objective(spec["objective_id"])
    if spec.get("eval_overlay"):
        from src.search.controller import _apply_eval_override
        objective = _apply_eval_override(objective, spec["eval_overlay"])
    if spec.get("topology_overlay"):
        from src.search.controller import _apply_topology_override
        objective = _apply_topology_override(objective, spec["topology_overlay"])
    entry = reconstruct_catalog_entry(spec)

    from src.features.stores import default_parquet_dir

    pdir = Path(spec.get("parquet_dir") or str(default_parquet_dir()))
    data_dest = Path(spec.get("data_s3_dest") or pdir)

    # Non-default data dirs (e.g. data/parquet_raw1s, hive timeframe={tf}/) are
    # not covered when the GPU VM was already RUNNING (ExecStartPre does not
    # re-run). job_exec hydrates the Object-Storage prefix into data_s3_dest
    # (hive root for timeframe={tf}/).
    _sync_data_prefix(str(spec.get("data_s3_prefix") or ""), data_dest)
    _sync_labels_prefix(spec)
    _sync_context_prefixes(spec)
    _sync_path1s_prefixes(spec)

    try:
        res = _run_one(
            entry, objective, pdir, out, False,
            int(spec.get("tune_trials", 0)),
            bool(spec.get("persist_bundle")),
            spec.get("search_run_ts"),
            spec.get("search_run_dir"),
        )
    except Exception as exc:  # noqa: BLE001 — persist heartbeat then let the runner publish failed
        mark_failed(exc)
        raise
    finish_session(status=res.status, error=res.error)

    from src.training.job_progress import note_status
    note_status("uploading")

    # Upload from the *bundle* process. The long-lived GPU runner imports
    # ``publish_result`` from its installed tree, which may predate ADR-036
    # (root oof_* glob only). Putting models/** here means the CPU collector
    # can prefix-list them even when the runner's result_files list is empty.
    result_files: list[str] = []
    run_id = spec.get("run_id")
    if run_id:
        try:
            from src.common.object_store import make_object_store
            from src.gpu_runner.dispatch import upload_job_artifacts
            store, _ = make_object_store(local_env="VTECH_GPU_QUEUE_LOCAL")
            result_files = upload_job_artifacts(store, str(run_id), out)
            print(f"artifact_upload run_id={run_id} n={len(result_files)}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — runner publish is the fallback
            print(f"artifact_upload_failed run_id={run_id} err={exc}", file=sys.stderr)

    (out / "result.json").write_text(json.dumps({
        "candidate_id": res.candidate_id,
        "family": res.family,
        "status": res.status,
        "score": res.score,
        "meets_target": getattr(res, "meets_target", False),
        "error": res.error,
        "elapsed_s": res.elapsed_s,
        # Executed config echo (tuned params survive; collector must not fall back
        # to catalog defaults — see ADR-024 provenance requirements).
        "spec_params": res.spec_params,
        "feature_selection": entry.feature_selection,
        "result_files": result_files,
        "bundle_path": res.bundle_path,
    }, default=str))
    return 0


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python -m src.gpu_runner.job_exec <spec.json> <out_dir>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(_run(sys.argv[1], sys.argv[2]))


if __name__ == "__main__":
    main()
