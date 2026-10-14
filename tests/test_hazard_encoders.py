"""Path-label fold join + TCN/patch hazard plugins (ADR-053 Encoder A/B)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.catalog import get_entry
from src.models.heads import encode_hazard_class, n_hazard_classes
from src.objectives.loader import load_objective
from src.search.controller import _load_plugin, _subsample_folds
from src.search.uniqueness import apply_train_uniqueness, slice_train_rows
from src.validation.walk_forward import apply_schedule_to_arrays
from src.training.path_labels import CENSORED, STOP, TARGET


def test_objective_loads_and_names_both_encoders():
    obj = load_objective("nq-50pt-path-hazard")
    assert obj["labels"]["scheme"] == "path_vol_normalized"
    assert obj["eligible_model_families"] == ["tcn_hazard", "patch_hazard"]
    assert obj["eval"]["walk_forward"]["purge_seconds"] == 600
    assert obj["eval"]["date_range"]["start"] == "2018-01-02"
    assert obj["search"]["feature_views"]["tcn_hazard"] == "sequence_baseline"


def test_catalog_and_plugin_registry():
    tcn = get_entry("tcn_hazard")
    patch = get_entry("patch_hazard")
    assert tcn.compute_class == "gpu" and patch.needs_sequence
    assert tcn.params_default["class_balance"] == "sqrt"
    assert patch.params_default["n_layers"] == 4
    assert _load_plugin("tcn_hazard").name == "tcn_hazard"
    assert _load_plugin("patch_hazard").name == "patch_hazard"


def _label_frame(ts: pd.DatetimeIndex, *, usable: bool = True, r_value: float = 80.0) -> pd.DataFrame:
    n = len(ts)
    event_up = np.full(n, CENSORED, dtype=np.int8)
    bucket_up = np.full(n, 20, dtype=np.int16)
    event_up[::10] = TARGET
    bucket_up[::10] = 3
    event_down = np.full(n, STOP, dtype=np.int8)
    bucket_down = np.zeros(n, dtype=np.int16)
    label = np.ones(n, dtype=np.int32)
    label[::10] = 2
    return pd.DataFrame({
        "ts": ts, "label": label,
        "event_up": event_up, "bucket_up": bucket_up,
        "event_down": event_down, "bucket_down": bucket_down,
        "offset_up": 100, "offset_down": 5,
        "mfe_up_points": 1.0, "mae_up_points": 0.5,
        "mfe_down_points": 0.5, "mae_down_points": 1.0,
        "target_points": 0.743 * r_value, "stop_points": 0.0891 * r_value,
        "r_value": r_value, "stop_ticks": (0.0891 * r_value) / 0.25,
        "usable": usable, "ambiguous": False, "truncated": False,
        "mfe_up_r": 0.2, "mae_up_r": 0.1, "mfe_down_r": 0.1, "mae_down_r": 0.2,
    })


def test_fold_cache_joins_pathlabels_on_ts(tmp_path, monkeypatch):
    from src.search import fold_cache as fc

    feat_dir = tmp_path / "parquet"
    lab_dir = tmp_path / "labs"
    feat_dir.mkdir()
    lab_dir.mkdir()
    idx_1s = pd.date_range("2026-03-02 14:30:00", periods=120, freq="1s", tz="UTC")
    _label_frame(idx_1s).to_parquet(lab_dir / "labels_2026-03-02.parquet", index=False)
    idx_5s = pd.date_range("2026-03-02 14:30:00", periods=24, freq="5s", tz="UTC")
    rng = np.random.default_rng(0)
    pd.DataFrame(
        {"vw_x": rng.normal(size=24), "rv_y": rng.normal(size=24),
         "_close": np.linspace(20000, 20010, 24)},
        index=idx_5s,
    ).to_parquet(feat_dir / "features_2026-03-02.parquet")

    monkeypatch.setattr(fc, "parquet_pathlabels_dir", lambda: lab_dir)
    fc._CACHE_ROOT = tmp_path / "_fold_cache"
    obj = {
        "labels": {"move_threshold": 50, "mae_tolerance": 6, "scheme": "path_vol_normalized"},
        "horizon": {"max_trade_horizon": "10m"},
    }
    arrays = fc.build_or_load_fold_arrays(obj, feat_dir)
    assert arrays is not None
    X, y, ts, cols = arrays
    assert X.shape == (24, 2)
    assert set(cols) == {"vw_x", "rv_y"}
    extras = fc.load_fold_extras(obj, feat_dir)
    assert extras["y_up"].shape == (24,)
    assert extras["r_value"][0] == pytest.approx(80.0)
    # every 10th 1s bar is TARGET; 5s bars at 0,5,10,15,… so indices 0 and 10 of 1s
    # that land on the 5s grid are 0, 10, … → 5s rows 0 and 2
    assert extras["y_up"][0] == encode_hazard_class(TARGET, 3).item()
    assert extras["y_up"][1] == encode_hazard_class(CENSORED, 20).item()
    assert set(np.unique(y)).issubset({0, 1, 2})


def test_unusable_pathlabel_day_is_dropped(tmp_path, monkeypatch):
    from src.search import fold_cache as fc

    feat_dir = tmp_path / "parquet"
    lab_dir = tmp_path / "labs"
    feat_dir.mkdir()
    lab_dir.mkdir()
    idx_1s = pd.date_range("2026-03-02 14:30:00", periods=60, freq="1s", tz="UTC")
    _label_frame(idx_1s, usable=False, r_value=5.0).to_parquet(
        lab_dir / "labels_2026-03-02.parquet", index=False)
    idx_5s = pd.date_range("2026-03-02 14:30:00", periods=12, freq="5s", tz="UTC")
    pd.DataFrame({"f": np.ones(12), "_close": np.ones(12)}, index=idx_5s).to_parquet(
        feat_dir / "features_2026-03-02.parquet")
    monkeypatch.setattr(fc, "parquet_pathlabels_dir", lambda: lab_dir)
    fc._CACHE_ROOT = tmp_path / "_fold_cache"
    obj = {"labels": {"scheme": "path_vol_normalized", "move_threshold": 50, "mae_tolerance": 6},
           "horizon": {"max_trade_horizon": "10m"}}
    assert fc.build_or_load_fold_arrays(obj, feat_dir) is None


def test_schedule_and_uniqueness_keep_hazard_extras_aligned():
    n = 40
    X = np.arange(n * 2, dtype="float32").reshape(n, 2)
    y = np.zeros(n, dtype=np.int32)
    ts = pd.date_range("2026-03-02 14:30:00", periods=n, freq="5s", tz="UTC").to_numpy()
    extras = {"y_up": np.arange(n, dtype=np.int32), "r_value": np.full(n, 80.0)}
    sched = [{"fold": 0,
              "train": {"start": "2026-03-02", "end": "2026-03-02"},
              "oof": {"start": "2026-03-02", "end": "2026-03-02"},
              "val": None}]
    folds = apply_schedule_to_arrays(
        X, y, ts, sched, ["a", "b"], purge_seconds=0, extras=extras,
    )
    assert len(folds) == 1
    assert "y_up_train" in folds[0] and len(folds[0]["y_up_train"]) == n
    thinned = apply_train_uniqueness(folds, uniqueness_bars=10)
    assert len(thinned[0]["y_up_train"]) == len(thinned[0]["X_train"])
    cap = _subsample_folds(thinned, 4, contiguous=True)
    assert len(cap[0]["y_up_train"]) == 4
    assert len(cap[0]["r_value_train"]) == 4


def test_slice_train_rows_boolean_and_index():
    n = 8
    f = {"X_train": np.eye(n, dtype="float32"), "y_train": np.arange(n),
         "y_up_train": np.arange(n) + 10, "X_oof": np.ones((2, n))}
    g = slice_train_rows(f, np.array([True, False] * 4))
    assert list(g["y_up_train"]) == [10, 12, 14, 16]
    assert g["X_oof"].shape == (2, n)


def _hazard_fold(n: int = 40, f: int = 4, seq_len: int = 8) -> dict:
    rng = np.random.default_rng(0)
    n_cls = n_hazard_classes()
    cols = [f"vw_{i}" if i < 2 else f"rv_{i}" for i in range(f)]
    return {
        "X_train": rng.normal(size=(n, f)).astype("float32"),
        "y_train": rng.integers(0, 3, size=n, dtype=np.int32),
        "y_up_train": rng.integers(0, n_cls, size=n, dtype=np.int32),
        "y_down_train": rng.integers(0, n_cls, size=n, dtype=np.int32),
        "y_quantile_train": rng.random(size=(n, 4)).astype("float32"),
        "r_value_train": np.full(n, 80.0),
        "X_val": rng.normal(size=(n // 2, f)).astype("float32"),
        "y_val": rng.integers(0, 3, size=n // 2, dtype=np.int32),
        "y_up_val": rng.integers(0, n_cls, size=n // 2, dtype=np.int32),
        "y_down_val": rng.integers(0, n_cls, size=n // 2, dtype=np.int32),
        "y_quantile_val": rng.random(size=(n // 2, 4)).astype("float32"),
        "X_oof": rng.normal(size=(n, f)).astype("float32"),
        "y_oof": rng.integers(0, 3, size=n, dtype=np.int32),
        "y_up_oof": rng.integers(0, n_cls, size=n, dtype=np.int32),
        "y_down_oof": rng.integers(0, n_cls, size=n, dtype=np.int32),
        "r_value_oof": np.full(n, 80.0),
        "ts_oof": pd.date_range("2026-03-02", periods=n, freq="5s", tz="UTC").to_numpy(),
        "feature_cols": cols,
    }


def test_tcn_hazard_fits_and_emits_canonical():
    from tensorflow import keras

    keras.backend.clear_session()
    plugin = _load_plugin("tcn_hazard")
    plugin.build(
        __import__("src.models.base", fromlist=["ModelSpec"]).ModelSpec(
            family="tcn_hazard",
            params={"seq_len": 8, "epochs": 1, "batch_size": 8, "filters": 8,
                    "dilations": [1, 2], "include_quantile": False,
                    "class_balance": "sqrt"},
        ),
        {},
    )
    fold = _hazard_fold()
    fitted = plugin.fit(None, [fold])
    assert fitted.oof_preds is not None
    assert set(fitted.oof_preds["direction"]).issubset({"up", "down", "flat"})
    preds = plugin.predict(fitted, fold["X_oof"])
    assert len(preds) == len(fold["X_oof"]) - 8 + 1
    keras.backend.clear_session()


def test_patch_hazard_default_is_about_0_9m_params():
    from tensorflow import keras

    keras.backend.clear_session()
    plugin = _load_plugin("patch_hazard")
    model = plugin._rebuild(40, [f"vw_{i}" for i in range(10)]
                            + [f"of_{i}" for i in range(10)]
                            + [f"rv_{i}" for i in range(10)]
                            + [f"iv_{i}" for i in range(10)])
    n = int(model.count_params())
    assert 400_000 < n < 2_000_000, n
    keras.backend.clear_session()


def test_patch_hazard_fits_tiny_graph():
    from tensorflow import keras
    from src.models.base import ModelSpec

    keras.backend.clear_session()
    plugin = _load_plugin("patch_hazard")
    plugin.build(ModelSpec(family="patch_hazard", params={
        "seq_len": 16, "patch_len": 8, "d_model": 16, "n_heads": 2, "n_layers": 1,
        "ff_dim": 32, "epochs": 1, "batch_size": 8, "include_quantile": False,
        "dropout": 0.0,
    }), {})
    fold = _hazard_fold(n=32, f=8, seq_len=16)
    fold["feature_cols"] = [f"vw_{i}" for i in range(2)] + [f"of_{i}" for i in range(2)] \
        + [f"rv_{i}" for i in range(2)] + [f"iv_{i}" for i in range(2)]
    fitted = plugin.fit(None, [fold])
    assert fitted.oof_preds is not None and len(fitted.oof_preds) == 32 - 16 + 1
    keras.backend.clear_session()
