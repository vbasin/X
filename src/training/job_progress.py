"""Live training heartbeat + per-fold OOF checkpoints (ADR-073).

Walk-forward plugins used to emit nothing until ``_run_one`` finished, so a
mid-wave crash (or a silent Keras ``verbose=0`` GPU job) left no proof that
any fold completed. This module is a process-local session:

* ``progress.json`` — atomic heartbeat (fold / epoch / %, last error, last-fold CM)
* ``folds/fold_K.parquet`` + ``folds/fold_K.json`` — OOF + confusion after each fold

GPU ``job_exec`` starts a session that writes both the job temp dir (uploaded
at the end) and ``$VTECH_DATA_DIR/ops/gpu_jobs/<run_id>/`` (SSH-visible while
the job is running). Plugins no-op when no session is active (CPU unit tests,
catalog smoke).
"""

from __future__ import annotations

import json
import logging
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.base import CANONICAL_DIRECTION

logger = logging.getLogger("vtech.training.job_progress")

_LABELS = ("down", "flat", "up")
_SESSION: "JobProgress | None" = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.replace(path)


def fold_confusion(oof: pd.DataFrame | None) -> dict[str, Any]:
    """Cheap 3-class confusion + hit counts from a canonical OOF frame."""
    if oof is None or oof.empty:
        return {"n": 0}
    n = int(len(oof))
    out: dict[str, Any] = {"n": n}
    if CANONICAL_DIRECTION not in oof.columns:
        return out
    pred = oof[CANONICAL_DIRECTION].astype(str)
    n_signals = int((pred != "flat").sum())
    out["n_signals"] = n_signals
    if "direction_true" not in oof.columns:
        return out
    true = oof["direction_true"].astype(str)
    out["confusion"] = {
        "labels": list(_LABELS),
        "rows_true_cols_pred": [
            [int(((true == t) & (pred == p)).sum()) for p in _LABELS]
            for t in _LABELS
        ],
    }
    n_hits = int(((true == pred) & (pred != "flat")).sum())
    out["n_hits"] = n_hits
    if n_signals:
        out["hit_rate"] = round(n_hits / n_signals, 4)
    from src.models.dual_binary import is_dual_binary_frame
    if is_dual_binary_frame(oof):
        pl = oof["p_long"].to_numpy(dtype="float64")
        ps = oof["p_short"].to_numpy(dtype="float64")
        out["head"] = "dual_binary"
        out["heads"] = {
            "long": {
                "n_pos": int((true == "up").sum()),
                "n_pred_pos": int((pred == "up").sum()),
                "p_mean": float(pl.mean()),
                "p_max": float(pl.max()),
            },
            "short": {
                "n_pos": int((true == "down").sum()),
                "n_pred_pos": int((pred == "down").sum()),
                "p_mean": float(ps.mean()),
                "p_max": float(ps.max()),
            },
        }
    return out


class JobProgress:
    """Process-local heartbeat writer. Safe to call from Keras callbacks."""

    def __init__(
        self,
        *,
        run_id: str,
        family: str | None,
        objective_id: str | None,
        paths: list[Path],
        fold_dirs: list[Path] | None = None,
    ) -> None:
        self.run_id = str(run_id)
        self.family = family
        self.objective_id = objective_id
        self.paths = [Path(p) for p in paths]
        self.fold_dirs = [Path(p) for p in (fold_dirs or [])]
        self.t0 = time.time()
        self.state: dict[str, Any] = {
            "run_id": self.run_id,
            "objective_id": self.objective_id,
            "family": self.family,
            "status": "starting",
            "fold_i": 0,
            "n_folds": None,
            "folds_done": 0,
            "epoch_i": 0,
            "n_epochs": None,
            "early_stop": False,
            "loss": None,
            "val_loss": None,
            "pct_complete": None,
            "elapsed_s": 0.0,
            "error": None,
            "last_fold": None,
            "ts": _now_iso(),
        }
        self.flush()

    def _pct(self) -> float | None:
        n_folds = int(self.state.get("n_folds") or 0)
        if n_folds <= 0:
            return None
        n_epochs = int(self.state.get("n_epochs") or 0)
        fold_i = int(self.state.get("fold_i") or 0)
        epoch_i = int(self.state.get("epoch_i") or 0)
        folds_done = int(self.state.get("folds_done") or 0)
        if n_epochs > 0:
            done = fold_i * n_epochs + epoch_i
            total = n_folds * n_epochs
            return round(100.0 * min(done, total) / total, 1)
        return round(100.0 * min(folds_done, n_folds) / n_folds, 1)

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is not None or k in ("error", "loss", "val_loss", "last_fold"):
                self.state[k] = v
        self.state["elapsed_s"] = round(time.time() - self.t0, 1)
        self.state["pct_complete"] = self._pct()
        self.state["ts"] = _now_iso()
        self.flush()

    def flush(self) -> None:
        payload = dict(self.state)
        for path in self.paths:
            try:
                _atomic_write_json(path, payload)
            except Exception as exc:  # noqa: BLE001 — never fail the trainer
                logger.warning("progress_write_failed path=%s err=%s", path, exc)

    def checkpoint_fold(self, fold_i: int, oof: pd.DataFrame | None) -> dict[str, Any]:
        meta = fold_confusion(oof)
        meta["fold"] = int(fold_i)
        meta["elapsed_s"] = round(time.time() - self.t0, 1)
        stem = f"fold_{int(fold_i):02d}"
        for d in self.fold_dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{stem}.json").write_text(json.dumps(meta, indent=2, default=str))
                if oof is not None and not oof.empty:
                    oof.to_parquet(d / f"{stem}.parquet", index=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("fold_checkpoint_failed dir=%s fold=%s err=%s", d, fold_i, exc)
        return meta


def current() -> JobProgress | None:
    return _SESSION


def reset_session() -> None:
    """Tests only — drop the process-local session."""
    global _SESSION
    _SESSION = None


def start_session(
    *,
    run_id: str,
    family: str | None = None,
    objective_id: str | None = None,
    out_dir: str | Path | None = None,
    durable_dir: str | Path | None = None,
    paths: list[Path] | None = None,
    fold_dirs: list[Path] | None = None,
) -> JobProgress:
    """Begin a heartbeat session. Replaces any previous session in this process."""
    global _SESSION
    write_paths: list[Path] = list(paths or [])
    fold_dirs_out: list[Path] = [Path(p) for p in (fold_dirs or [])]
    if out_dir is not None:
        od = Path(out_dir)
        write_paths.append(od / "progress.json")
        fold_dirs_out.append(od / "folds")
    if durable_dir is not None:
        dd = Path(durable_dir)
        write_paths.append(dd / "progress.json")
        fold_dirs_out.append(dd / "folds")
    # unique, stable order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in write_paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    fold_seen: set[str] = set()
    fold_uniq: list[Path] = []
    for p in fold_dirs_out:
        key = str(p)
        if key not in fold_seen:
            fold_seen.add(key)
            fold_uniq.append(p)
    _SESSION = JobProgress(
        run_id=str(run_id),
        family=family,
        objective_id=objective_id,
        paths=uniq,
        fold_dirs=fold_uniq,
    )
    return _SESSION


def ensure_session(
    *,
    run_id: str,
    family: str | None = None,
    objective_id: str | None = None,
    out_dir: str | Path | None = None,
) -> JobProgress:
    """Start if missing; otherwise keep the existing GPU job_exec session."""
    sess = current()
    if sess is not None:
        if family and not sess.family:
            sess.family = family
            sess.state["family"] = family
        if objective_id and not sess.objective_id:
            sess.objective_id = objective_id
            sess.state["objective_id"] = objective_id
        return sess
    return start_session(
        run_id=run_id, family=family, objective_id=objective_id, out_dir=out_dir,
    )


def note_status(status: str, **kwargs: Any) -> None:
    sess = current()
    if sess is None:
        return
    sess.update(status=status, **kwargs)


def note_train_start(
    fold_i: int,
    n_folds: int,
    *,
    n_epochs: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    sess = current()
    if sess is None:
        return
    payload: dict[str, Any] = {
        "status": "training",
        "fold_i": int(fold_i),
        "n_folds": int(n_folds),
        "epoch_i": 0,
        "early_stop": False,
        "loss": None,
        "val_loss": None,
    }
    if n_epochs is not None:
        payload["n_epochs"] = int(n_epochs)
    if extra:
        payload.update(extra)
    sess.update(**payload)
    logger.info(
        "progress fold=%d/%d epoch=0/%s family=%s",
        fold_i + 1, n_folds, n_epochs if n_epochs is not None else "-", sess.family,
    )


def note_epoch(
    *,
    fold_i: int,
    n_folds: int,
    epoch_i: int,
    n_epochs: int,
    loss: float | None = None,
    val_loss: float | None = None,
    early_stop: bool = False,
) -> None:
    sess = current()
    if sess is None:
        return
    sess.update(
        status="training",
        fold_i=int(fold_i),
        n_folds=int(n_folds),
        epoch_i=int(epoch_i),
        n_epochs=int(n_epochs),
        loss=_f(loss),
        val_loss=_f(val_loss),
        early_stop=bool(early_stop),
    )
    logger.info(
        "epoch fold=%d/%d epoch=%d/%d loss=%s val_loss=%s early_stop=%s pct=%s",
        fold_i + 1, n_folds, epoch_i, n_epochs,
        sess.state.get("loss"), sess.state.get("val_loss"),
        early_stop, sess.state.get("pct_complete"),
    )


def checkpoint_fold(
    fold_i: int,
    n_folds: int,
    oof: pd.DataFrame | None,
) -> None:
    sess = current()
    if sess is None:
        return
    meta = sess.checkpoint_fold(fold_i, oof)
    sess.update(
        status="fold_done",
        fold_i=int(fold_i),
        n_folds=int(n_folds),
        folds_done=int(fold_i) + 1,
        last_fold=meta,
    )
    cm = (meta.get("confusion") or {}).get("rows_true_cols_pred")
    logger.info(
        "fold_done fold=%d/%d n=%s n_signals=%s hit_rate=%s cm=%s pct=%s",
        fold_i + 1, n_folds, meta.get("n"), meta.get("n_signals"),
        meta.get("hit_rate"), cm, sess.state.get("pct_complete"),
    )


def finish_session(status: str = "ok", error: str | None = None, **kwargs: Any) -> None:
    sess = current()
    if sess is None:
        return
    if status == "ok":
        n_folds = sess.state.get("n_folds")
        if n_folds:
            sess.state["folds_done"] = n_folds
            sess.state["fold_i"] = max(int(n_folds) - 1, 0)
            n_epochs = sess.state.get("n_epochs")
            if n_epochs:
                sess.state["epoch_i"] = n_epochs
    sess.update(status=status, error=error, **kwargs)


def mark_failed(exc: BaseException | str) -> None:
    if isinstance(exc, BaseException):
        err = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        if tb and tb.strip() != "NoneType: None":
            err = (tb[-1500:] if len(tb) > 1500 else tb)
    else:
        err = str(exc)
    finish_session(status="failed", error=err[:1500])


def keras_epoch_callback(
    *,
    fold_i: int,
    n_folds: int,
    n_epochs: int,
) -> list[Any]:
    """Keras callback list (empty when no session — avoids a TF import in tests)."""
    if current() is None:
        return []
    from tensorflow.keras.callbacks import Callback

    class _Heartbeat(Callback):
        def on_epoch_end(self, epoch, logs=None):  # noqa: ANN001
            logs = logs or {}
            note_epoch(
                fold_i=fold_i,
                n_folds=n_folds,
                epoch_i=int(epoch) + 1,
                n_epochs=n_epochs,
                loss=logs.get("loss"),
                val_loss=logs.get("val_loss"),
            )

        def on_train_end(self, logs=None):  # noqa: ANN001
            sess = current()
            if sess is None:
                return
            if int(sess.state.get("epoch_i") or 0) < int(n_epochs):
                sess.update(early_stop=True)

    return [_Heartbeat()]
