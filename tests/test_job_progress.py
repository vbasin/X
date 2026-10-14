"""ADR-073 heartbeat + per-fold checkpoint (no Keras / no GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training.job_progress import (
    checkpoint_fold,
    finish_session,
    fold_confusion,
    keras_epoch_callback,
    mark_failed,
    note_epoch,
    note_train_start,
    reset_session,
    start_session,
)


def setup_function() -> None:
    reset_session()


def teardown_function() -> None:
    reset_session()


def test_fold_confusion_3x3() -> None:
    df = pd.DataFrame({
        "direction": ["up", "up", "flat", "down"],
        "direction_true": ["up", "down", "flat", "down"],
    })
    meta = fold_confusion(df)
    assert meta["n"] == 4
    assert meta["n_signals"] == 3
    assert meta["n_hits"] == 2  # correct up + correct down
    assert meta["hit_rate"] == 0.6667
    assert meta["confusion"]["labels"] == ["down", "flat", "up"]
    # rows = true down/flat/up; cols = pred down/flat/up
    cm = meta["confusion"]["rows_true_cols_pred"]
    assert cm[0][0] == 1  # true down pred down
    assert cm[0][2] == 1  # true down pred up
    assert cm[1][1] == 1  # true flat pred flat
    assert cm[2][2] == 1  # true up pred up


def test_fold_confusion_dual_binary() -> None:
    df = pd.DataFrame({
        "direction": ["up", "flat", "down", "flat"],
        "direction_true": ["up", "flat", "down", "up"],
        "p_long": [0.8, 0.1, 0.05, 0.4],
        "p_short": [0.1, 0.2, 0.7, 0.1],
    })
    meta = fold_confusion(df)
    assert meta["head"] == "dual_binary"
    assert meta["heads"]["long"]["n_pos"] == 2
    assert meta["heads"]["short"]["n_pos"] == 1
    assert meta["heads"]["long"]["n_pred_pos"] == 1
    assert meta["confusion"]["labels"] == ["down", "flat", "up"]


def test_fold_confusion_empty() -> None:
    assert fold_confusion(None) == {"n": 0}
    assert fold_confusion(pd.DataFrame()) == {"n": 0}


def test_session_writes_atomic_progress_and_fold_checkpoint(tmp_path: Path) -> None:
    durable = tmp_path / "gpu_jobs" / "run-1"
    sess = start_session(
        run_id="run-1",
        family="attention_lstm",
        objective_id="nq-50pt-tbm-opt",
        out_dir=tmp_path / "job",
        durable_dir=durable,
    )
    note_train_start(0, 6, n_epochs=15)
    note_epoch(fold_i=0, n_folds=6, epoch_i=5, n_epochs=15, loss=1.2, val_loss=1.4)
    payload = json.loads((tmp_path / "job" / "progress.json").read_text())
    assert payload["family"] == "attention_lstm"
    assert payload["fold_i"] == 0
    assert payload["n_folds"] == 6
    assert payload["epoch_i"] == 5
    assert payload["n_epochs"] == 15
    assert payload["pct_complete"] == 5.6  # 5 / (6*15) * 100
    assert payload["loss"] == 1.2
    assert (durable / "progress.json").is_file()

    oof = pd.DataFrame({
        "direction": ["up", "flat"],
        "direction_true": ["up", "flat"],
    })
    checkpoint_fold(0, 6, oof)
    fold_json = json.loads((tmp_path / "job" / "folds" / "fold_00.json").read_text())
    assert fold_json["n"] == 2
    assert fold_json["n_hits"] == 1
    assert (tmp_path / "job" / "folds" / "fold_00.parquet").is_file()
    assert (durable / "folds" / "fold_00.json").is_file()
    live = json.loads((durable / "progress.json").read_text())
    assert live["status"] == "fold_done"
    assert live["folds_done"] == 1
    assert live["last_fold"]["n"] == 2

    finish_session(status="ok")
    done = json.loads((tmp_path / "job" / "progress.json").read_text())
    assert done["status"] == "ok"
    assert done["pct_complete"] == 100.0
    assert sess is not None


def test_no_session_is_noop() -> None:
    reset_session()
    note_train_start(0, 3, n_epochs=10)
    note_epoch(fold_i=0, n_folds=3, epoch_i=1, n_epochs=10)
    checkpoint_fold(0, 3, None)
    assert keras_epoch_callback(fold_i=0, n_folds=3, n_epochs=10) == []


def test_mark_failed_records_error(tmp_path: Path) -> None:
    start_session(run_id="x", family="tcn", out_dir=tmp_path)
    try:
        raise ValueError("boom")
    except ValueError as exc:
        mark_failed(exc)
    payload = json.loads((tmp_path / "progress.json").read_text())
    assert payload["status"] == "failed"
    assert "ValueError" in (payload["error"] or "")


def test_keras_callback_empty_without_session() -> None:
    reset_session()
    assert keras_epoch_callback(fold_i=0, n_folds=2, n_epochs=3) == []
