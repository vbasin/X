"""CPU ↔ GPU feature-parquet mirror via Object Storage.

GPU ``job_exec`` pulls matching parquets from the job's ``data_s3_prefix``
(and optional ``labels_s3_prefix``) on every candidate (size and remote
LastModified). The CPU controller pushes the local tree into that prefix
before enqueue so a Custom Test launched after a parquet patch does not
train on yesterday's ``vtech_hot``.

Native-TF hive files live under ``timeframe={tf}/symbol=/year=/month=/``.
Keys preserve that relative path (ADR-078). Same-named monthly parts
(``features_15s_part1.parquet``) must not flatten to the basename.
"""

from __future__ import annotations

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

_SLACK_S = 2.0
_SYNC_WORKERS = max(1, int(os.environ.get("VTECH_PARQUET_SYNC_WORKERS", "8")))


def _is_feature_parquet(name: str) -> bool:
    return name.startswith("features_") and name.endswith(".parquet")


def _match_parquet(name: str, glob: str) -> bool:
    """Tiny glob: ``prefix_*.parquet`` (also used for path_ / labels_)."""
    if glob == "features_*.parquet":
        return _is_feature_parquet(name)
    if "*" not in glob:
        return name == glob
    prefix, _, suffix = glob.partition("*")
    return name.startswith(prefix) and name.endswith(suffix)


def _store(store: Any | None = None):
    if store is not None:
        return store
    from src.common.object_store import make_object_store
    s, _kind = make_object_store(local_env="VTECH_GPU_QUEUE_LOCAL")
    return s


def _atomic_write(dest: Path, data: bytes) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)


def _rel_key(key: str, prefix: str) -> str:
    return key[len(prefix):].lstrip("/") if key.startswith(prefix) else key.lstrip("/")


def iter_local_matches(pdir: Path, glob: str) -> Iterator[Path]:
    """Yield files under *pdir* whose basename matches *glob* (recursive)."""
    pdir = Path(pdir)
    if not pdir.is_dir():
        return
    if "*" not in glob:
        p = pdir / glob
        if p.is_file():
            yield p
        return
    for src in sorted(pdir.rglob(glob)):
        if src.is_file() and not src.name.startswith(".") and _match_parquet(src.name, glob):
            yield src


def count_local_matches(pdir: Path, glob: str) -> int:
    return sum(1 for _ in iter_local_matches(pdir, glob))


def gpu_feature_mirror(pdir: Path) -> tuple[Path, str]:
    """``(push_root, s3_prefix)`` aligned with ``oci_sync.conf`` / GPU boot DATASETS.

    Training ``parquet_dir`` for hive stores is ``timeframe={tf}/symbol=NQ``.
    Object Storage keys start at ``timeframe={tf}/`` so hive relative paths
    match GPU boot pull into ``sources/features/timeframe={tf}``. Leftover
    ``parquet_5s`` day-file dirs remap to the 5s hive, not ``parquet/``.
    """
    from src.common.data_paths import (
        hive_root_from_path,
        is_default_5s_feature_dir,
        native_tf_features_root,
    )

    p = Path(pdir)
    root = hive_root_from_path(p)
    if root is not None:
        return root, f"{root.name}/"
    try:
        if is_default_5s_feature_dir(p):
            hive = native_tf_features_root() / "timeframe=5s"
            return hive, "timeframe=5s/"
    except OSError:
        pass
    return p, f"{p.name}/"


def gpu_context_prefixes(objective: dict[str, Any] | None) -> list[str]:
    """Object-store prefixes for slim 1h/1d HTF hives the GPU fold cache asofs.

    Training ``data_s3_prefix`` is the predictive store only (15s / 1m / 5m).
    ``attach_htf_to_frame`` needs ``timeframe=1h/`` and ``timeframe=1d/`` on
    the GPU disk (ADR-102). Empty when the objective has no topology.context.
    """
    from src.features.htf_context import wants_htf_on_store
    from src.features.stores import feature_store_id

    obj = objective or {}
    if not wants_htf_on_store(obj, feature_store_id(obj)):
        return []
    return ["timeframe=1h/", "timeframe=1d/"]


def gpu_label_mirror(store_id: str, objective: dict[str, Any]) -> tuple[Path, str, str] | None:
    """``(local_dir, s3_prefix, kind)`` for persisted labels of a feature store."""
    from src.common.data_paths import NATIVE_TF_TIMEFRAMES
    from src.training.label_store import label_contract_id, label_set_dir, tbm_contract_from_objective

    sid = str(store_id or "").strip()
    if sid in NATIVE_TF_TIMEFRAMES:
        contract = tbm_contract_from_objective(objective)
        contract_id = label_contract_id(contract)
        return label_set_dir("15s", contract), f"labels/15s/{contract_id}/", "15s"
    if sid == "5s":
        contract = tbm_contract_from_objective(objective)
        contract_id = label_contract_id(contract)
        return label_set_dir("5s", contract), f"labels/5s/{contract_id}/", "5s"
    return None


def _run_pool(items: list, fn) -> int:
    if not items:
        return 0
    workers = max(1, min(_SYNC_WORKERS, len(items)))
    n_ok = 0
    if workers == 1:
        for item in items:
            if fn(item):
                n_ok += 1
        return n_ok
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, item) for item in items]
        for fut in as_completed(futs):
            if fut.result():
                n_ok += 1
    return n_ok


def pull_features(prefix: str, pdir: Path, *, store: Any | None = None,
                  glob: str = "features_*.parquet") -> tuple[int, int]:
    """Download matching parquets under *prefix* into *pdir* (hive-relative keys)."""
    if not prefix:
        return 0, 0
    store = _store(store)
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    n_skip = 0
    todo: list[tuple[str, Path]] = []
    for key, size, remote_mtime in store.list_meta(prefix):
        rel = _rel_key(key, prefix)
        if not rel or rel.endswith("/"):
            continue
        name = Path(rel).name
        if not _match_parquet(name, glob):
            continue
        dest = pdir / rel
        if dest.exists():
            st = dest.stat()
            if st.st_size == size and (
                remote_mtime <= 0.0 or st.st_mtime + _SLACK_S >= remote_mtime
            ):
                n_skip += 1
                continue
        todo.append((key, dest))

    def _one(item: tuple[str, Path]) -> bool:
        key, dest = item
        data = store.get(key)
        if data is None:
            return False
        _atomic_write(dest, data)
        return True

    n_dl = _run_pool(todo, _one)
    print(f"data_sync: downloaded={n_dl} skipped={n_skip} prefix='{prefix}' -> {pdir}",
          file=sys.stderr)
    return n_dl, n_skip


def push_features(prefix: str, pdir: Path, *, store: Any | None = None,
                  glob: str = "features_*.parquet") -> tuple[int, int]:
    """Upload local matching parquets under *pdir* to *prefix* (recursive, relative keys)."""
    if not prefix:
        return 0, 0
    store = _store(store)
    pdir = Path(pdir)
    if not pdir.is_dir():
        return 0, 0
    remote: dict[str, tuple[int, float]] = {}
    for key, size, remote_mtime in store.list_meta(prefix):
        rel = _rel_key(key, prefix)
        if not rel or rel.endswith("/"):
            continue
        name = Path(rel).name
        if _match_parquet(name, glob):
            remote[rel] = (size, remote_mtime)
    pref = prefix.rstrip("/") + "/"
    todo: list[tuple[str, Path]] = []
    n_skip = 0
    for src in iter_local_matches(pdir, glob):
        rel = src.relative_to(pdir).as_posix()
        prev = remote.get(rel)
        st = src.stat()
        if prev is not None:
            rsize, rmtime = prev
            if st.st_size == rsize and (
                rmtime <= 0.0 or st.st_mtime <= rmtime + _SLACK_S
            ):
                n_skip += 1
                continue
        todo.append((pref + rel, src))

    def _one(item: tuple[str, Path]) -> bool:
        key, src = item
        store.put(key, src.read_bytes())
        return True

    n_up = _run_pool(todo, _one)
    print(f"data_push: uploaded={n_up} skipped={n_skip} prefix='{prefix}' <- {pdir}",
          file=sys.stderr)
    return n_up, n_skip


def push_job_training_data(
    objective: dict[str, Any],
    pdir: Path,
    *,
    store: Any | None = None,
) -> dict[str, str]:
    """Push feature (+ label) trees the GPU job will hydrate.

    Returns payload fields (``data_s3_prefix``, ``data_s3_dest``, optional
    ``labels_s3_prefix`` / ``labels_kind``). Raises ``RuntimeError`` when the
    local tree is empty or a successful list+push mirrored zero of N files
    (the native-TF ``symbol=NQ/`` + non-recursive glob failure mode).
    Object-store exceptions are logged and re-raised only when they leave
    the mirror empty; callers may still enqueue 5s jobs whose GPU disk
    already has a boot-time copy.
    """
    from src.features.stores import feature_store_id

    push_root, s3_prefix = gpu_feature_mirror(pdir)
    n_local = count_local_matches(push_root, "features_*.parquet")
    if n_local == 0:
        raise RuntimeError(
            f"no feature parquets under {push_root} (prefix {s3_prefix!r})"
        )
    native = s3_prefix.startswith("timeframe=")
    try:
        n_up, n_skip = push_features(s3_prefix, push_root, store=store)
    except Exception as exc:  # noqa: BLE001
        print(f"data_push failed prefix='{s3_prefix}' err={exc}", file=sys.stderr)
        if native:
            raise RuntimeError(
                f"parquet push failed prefix={s3_prefix!r} from {push_root}: {exc}"
            ) from exc
        n_up, n_skip = -1, 0
    if n_up >= 0 and n_up + n_skip == 0:
        raise RuntimeError(
            f"parquet push mirrored 0/{n_local} files to prefix {s3_prefix!r} "
            f"from {push_root}"
        )
    payload = {
        "data_s3_prefix": s3_prefix,
        "data_s3_dest": str(push_root),
    }

    lab = gpu_label_mirror(feature_store_id(objective), objective)
    if lab is None:
        return payload
    lab_dir, lab_prefix, lab_kind = lab
    n_lab = count_local_matches(lab_dir, "labels_*.parquet")
    if lab_kind != "5s" and n_lab == 0:
        raise RuntimeError(
            f"native-TF store needs labels under {lab_dir} "
            f"(prefix {lab_prefix!r})"
        )
    if n_lab == 0:
        return payload
    try:
        u, s = push_features(lab_prefix, lab_dir, store=store, glob="labels_*.parquet")
        push_features(lab_prefix, lab_dir, store=store, glob="_contract.json")
    except Exception as exc:  # noqa: BLE001
        print(f"data_push failed prefix='{lab_prefix}' err={exc}", file=sys.stderr)
        if lab_kind != "5s":
            raise RuntimeError(
                f"label push failed prefix={lab_prefix!r} from {lab_dir}: {exc}"
            ) from exc
        return payload
    if u + s == 0:
        raise RuntimeError(
            f"label push mirrored 0/{n_lab} files to prefix {lab_prefix!r} "
            f"from {lab_dir}"
        )
    payload["labels_s3_prefix"] = lab_prefix
    payload["labels_kind"] = lab_kind
    payload["labels_s3_dest"] = str(lab_dir)
    from src.common.data_paths import native_tf_features_root

    ctx: list[str] = []
    feats_root = native_tf_features_root()
    for prefix in gpu_context_prefixes(objective):
        hive = feats_root / prefix.rstrip("/")
        n_ctx = count_local_matches(hive, "features_*.parquet")
        if n_ctx == 0:
            print(f"data_push: skip empty context hive {hive}", file=sys.stderr)
            continue
        try:
            u, s = push_features(prefix, hive, store=store)
        except Exception as exc:  # noqa: BLE001
            print(f"data_push failed prefix='{prefix}' err={exc}", file=sys.stderr)
            raise RuntimeError(
                f"HTF context push failed prefix={prefix!r} from {hive}: {exc}"
            ) from exc
        if u + s == 0:
            raise RuntimeError(
                f"HTF context push mirrored 0/{n_ctx} files to prefix {prefix!r} "
                f"from {hive}"
            )
        ctx.append(prefix)
    if ctx:
        payload["context_s3_prefixes"] = ctx
    return payload

