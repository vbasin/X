"""Tests: triple-barrier (first-touch) label scheme + scheme plumbing.

Covers:
* order-awareness vs Detection-2 (target-then-retrace, crash-then-rally)
* stop-first rejection; mae boundary semantics (5 pt adverse survives a 6 pt stop)
* scheme dispatch through generate_labels / LabelConfig / objective loader
* fold-cache label params + timestep probe + cache-key separation per scheme
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.config import LabelConfig, _labels_from_objective
from src.training.labels import (
    BIG_DOWN, BIG_UP, encode_dual_binary, generate_labels, generate_labels_tbm,
    on_timestep_grid,
)


def _series(vals):
    idx = pd.date_range("2026-01-05 14:00", periods=len(vals), freq="5s", tz="UTC")
    return pd.Series(np.asarray(vals, dtype=float), index=idx)


def _cfg(scheme="triple_barrier", threshold=50.0, mae=6.0, window=20):
    return LabelConfig(big_move_threshold=threshold, mae_tolerance=mae,
                       forward_windows=[window], scheme=scheme)


# ── Core first-touch semantics ────────────────────────────────────────────────

def test_target_then_full_retrace_is_big_up_under_tbm_but_not_detection2():
    # Bar 0 entry at 100; price rallies +60 by bar 3, then crashes to -100 by bar 8.
    prices = _series([100, 120, 150, 160, 100, 60, 20, 10, 0] + [0] * 15)
    tbm = generate_labels(prices, _cfg())["label"]
    det2 = generate_labels(prices, _cfg(scheme="detection_2"))["label"]
    assert tbm.iloc[0] == 2        # +50 touched (bar 2) before -6 (bar 4) → BIG_UP
    assert det2.iloc[0] == 1       # window peak down_move=100 ≥ mae → rejected

def test_crash_then_rally_is_big_down_under_tbm_but_not_detection2():
    prices = _series([100, 60, 40, 100, 180, 200, 220, 240, 260] + [260] * 15)
    tbm = generate_labels(prices, _cfg())["label"]
    det2 = generate_labels(prices, _cfg(scheme="detection_2"))["label"]
    assert tbm.iloc[0] == 0        # -50 (bar 1) before +6 (bar 3) → BIG_DOWN
    assert det2.iloc[0] == 1

def test_stop_first_rejects_late_target():
    # Dips -7 (stop breach at 6) before rallying +60 → NO_SIGNAL under TBM.
    prices = _series([100, 96, 93, 100, 130, 160, 170, 180] + [180] * 16)
    tbm = generate_labels_tbm(prices, _cfg())["label"]
    assert tbm.iloc[0] == 1


def test_tbbo_target_can_win_before_same_bucket_close_stop():
    close = _series([100, 93] + [93] * 20)
    high = _series([100, 151] + [93] * 20)
    low = _series([100, 93] + [93] * 20)
    cfg = _cfg(threshold=50, mae=6, window=20)
    cfg.target_observation = "tbbo"
    label = generate_labels_tbm(close, cfg, target_high=high, target_low=low)["label"]
    assert label.iloc[0] == BIG_UP


def test_candle_hilo_target_requires_high_low_and_keeps_stop_close_based():
    close = _series([100, 93] + [93] * 20)
    high = _series([100, 149] + [93] * 20)
    low = _series([100, 93] + [93] * 20)
    cfg = _cfg(threshold=50, mae=6, window=20)
    cfg.target_observation = "candle_hilo"
    label = generate_labels_tbm(close, cfg, target_high=high, target_low=low)["label"]
    assert label.iloc[0] == 1
    with pytest.raises(ValueError, match="requires target_high"):
        generate_labels_tbm(close, cfg)


def test_tbbo_target_observation_drives_time_to_target_auxiliary():
    from src.training.aux_labels import compute_aux_labels

    close = _series([100, 100] + [100] * 20)
    high = _series([100, 151] + [100] * 20)
    low = _series([100] * 22)
    aux = compute_aux_labels(
        close, move_threshold=50, horizon_s=100, target_observation="tbbo",
        target_high=high, target_low=low,
    )
    assert aux["Time_to_TGT_long"].iloc[0] == 5

def test_mae_boundary_5pt_adverse_survives_6pt_stop():
    # Adverse excursion of exactly 5 (99.0 → wait: dip to 95 = -5) then +50 rally.
    prices = _series([100, 95, 97, 130, 151, 160, 170] + [170] * 17)
    lbl5 = generate_labels_tbm(prices, _cfg(mae=6.0))["label"]
    assert lbl5.iloc[0] == 2       # -5 does NOT touch the -6 stop → BIG_UP
    lbl_exact = generate_labels_tbm(prices, _cfg(mae=5.0))["label"]
    assert lbl_exact.iloc[0] == 1  # -5 touches a 5 pt stop (<= -mae) → NO_SIGNAL

def test_time_barrier_no_touch_is_no_signal():
    prices = _series([100 + (i % 3) for i in range(30)])
    lbl = generate_labels_tbm(prices, _cfg())["label"]
    assert (lbl == 1).all()

def test_directions_mutually_exclusive_on_random_walk():
    rng = np.random.default_rng(7)
    prices = _series(20000 + np.cumsum(rng.normal(0, 8, size=3000)))
    cfg = _cfg(threshold=50, mae=6, window=120)
    lbl = generate_labels_tbm(prices, cfg)["label"]
    assert set(np.unique(lbl)).issubset({0, 1, 2})
    # TBM (first-touch) admits at least as many directional bars as Detection-2
    det2 = generate_labels(prices, LabelConfig(50, 6, [120], "detection_2"))["label"]
    assert (lbl != 1).sum() >= (det2 != 1).sum()

def test_encode_dual_binary_writes_opposite_zero():
    y = pd.Series([BIG_UP, BIG_DOWN, 1, BIG_UP, 1, BIG_DOWN])
    out = encode_dual_binary(y)
    assert list(out.columns) == ["y_long", "y_short"]
    assert (out["y_long"] == [1, 0, 0, 1, 0, 0]).all()
    assert (out["y_short"] == [0, 1, 0, 0, 0, 1]).all()
    assert not ((out["y_long"] == 1) & (out["y_short"] == 1)).any()
    assert (out.loc[out["y_long"] == 1, "y_short"] == 0).all()
    assert (out.loc[out["y_short"] == 1, "y_long"] == 0).all()


def test_15s_grid_keeps_both_columns_from_5s_tbm():
    prices = _series([100, 120, 150, 160, 100, 60, 20, 10, 0] + [0] * 15)
    y = generate_labels_tbm(prices, _cfg())["label"]
    keep = on_timestep_grid(y.index, 15)
    out = encode_dual_binary(y.loc[keep])
    assert keep[0] and not keep[1] and not keep[2] and keep[3]
    assert out.loc[y.index[0], "y_long"] == 1
    assert out.loc[y.index[0], "y_short"] == 0
    assert set(out.columns) == {"y_long", "y_short"}


def test_default_scheme_unchanged():
    prices = _series([100, 95, 97, 130, 151, 160, 170] + [170] * 17)
    default_cfg = LabelConfig(big_move_threshold=50, mae_tolerance=6, forward_windows=[20])
    explicit = generate_labels(prices, _cfg(scheme="detection_2", mae=6))
    assert (generate_labels(prices, default_cfg)["label"] == explicit["label"]).all()


# ── Plumbing: objective → LabelConfig → fold cache ───────────────────────────

def test_objective_scheme_reaches_label_config():
    cfg = _labels_from_objective("nq-50pt-tbm")
    assert cfg.scheme == "triple_barrier"
    assert cfg.mae_tolerance == 6.0
    base = _labels_from_objective("nq-50pt")
    assert base.scheme == "detection_2"
    assert base.mae_tolerance == 5.0

def test_fold_cache_key_and_timestep_probe(tmp_path):
    from src.search.fold_cache import _cache_key, _label_params, probe_timestep_seconds
    from src.training.label_store import contract_matches, tbm_contract_from_objective

    def _write(pdir, freq):
        pdir.mkdir(parents=True, exist_ok=True)
        idx = pd.date_range("2026-01-05 14:00", periods=100, freq=freq, tz="UTC")
        df = pd.DataFrame({"f_a": np.arange(100.0), "_close": np.arange(100.0)}, index=idx)
        df.to_parquet(pdir / "features_2026-01-05.parquet")

    p5 = tmp_path / "parquet"
    p1 = tmp_path / "parquet_raw1s"
    _write(p5, "5s")
    _write(p1, "1s")
    assert probe_timestep_seconds(p5) == 5
    assert probe_timestep_seconds(p1) == 1

    obj_det2 = {"labels": {"move_threshold": 50, "mae_tolerance": 5},
                "horizon": {"max_trade_horizon": "10m"}}
    obj_tbm = {"labels": {"move_threshold": 50, "mae_tolerance": 6, "scheme": "triple_barrier"},
               "horizon": {"max_trade_horizon": "10m"}}
    assert _label_params(obj_det2, 5) == (50.0, 5.0, 120, "detection_2")
    assert _label_params(obj_tbm, 5) == (50.0, 6.0, 120, "triple_barrier")
    assert _label_params(obj_tbm, 1)[2] == 600          # 1s bars → 600-bar window
    assert _cache_key(p5, obj_det2) != _cache_key(p5, obj_tbm)   # scheme separates keys
    assert _cache_key(p1, obj_tbm) != _cache_key(p5, obj_tbm)    # data dir separates keys
    obj_dual = {"labels": {"move_threshold": 50, "mae_tolerance": 6,
                           "scheme": "triple_barrier", "encoding": "dual_binary"},
                "horizon": {"max_trade_horizon": "10m"}}
    obj_dual_aux = {"labels": dict(obj_dual["labels"], aux=["MFE_long_10m"]),
                    "horizon": obj_dual["horizon"]}
    assert _cache_key(p5, obj_dual) != _cache_key(p5, obj_dual_aux)
    obj_tbbo = {"labels": dict(obj_dual["labels"], target_observation="tbbo"),
                "horizon": obj_dual["horizon"]}
    assert _cache_key(p5, obj_dual) != _cache_key(p5, obj_tbbo)
    assert not contract_matches(tbm_contract_from_objective(obj_dual),
                                tbm_contract_from_objective(obj_tbbo))


def test_contract_addressed_label_set_and_run_snapshot(tmp_path, monkeypatch):
    from src.training.label_store import (
        label_contract_id, label_set_dir, load_labels_5s, tbm_contract_from_objective,
        write_label_contract_snapshot, write_label_parts, write_contract,
    )

    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path / "data"))
    objective = {
        "search": {"input": "native_tf"},
        "labels": {"move_threshold": 50, "mae_tolerance": 6,
                   "scheme": "triple_barrier", "encoding": "dual_binary"},
        "horizon": {"max_trade_horizon": "10m"},
    }
    contract = tbm_contract_from_objective(objective)
    source = label_set_dir("15s", contract)
    write_contract(source, contract)
    frame = pd.DataFrame({
        "ts": pd.date_range("2026-01-05", periods=2, freq="15s", tz="UTC"),
        "y_long": [0, 1], "y_short": [0, 0],
    })
    write_label_parts(source, [frame], resolution="15s")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    snapshot = write_label_contract_snapshot(objective, run_dir)
    assert source.name == label_contract_id(contract)
    assert snapshot is not None and snapshot.is_file()
    source_5s = label_set_dir("5s", contract)
    write_contract(source_5s, contract)
    write_label_parts(source_5s, [frame.assign(label=[0, 1])], resolution="5s")
    loaded_5s = load_labels_5s("2026-01-05", "2026-01-05", contract=contract)
    assert loaded_5s is not None and loaded_5s["label"].tolist() == [0, 1]


def test_fold_cache_builds_tbm_arrays(tmp_path):
    from src.search.fold_cache import build_or_load_fold_arrays

    pdir = tmp_path / "parquet"
    pdir.mkdir()
    idx = pd.date_range("2026-01-05 14:00", periods=400, freq="5s", tz="UTC")
    rng = np.random.default_rng(3)
    close = 20000 + np.cumsum(rng.normal(0, 10, size=400))
    df = pd.DataFrame({"f_a": rng.normal(size=400), "f_b": rng.normal(size=400),
                       "_close": close}, index=idx)
    df.to_parquet(pdir / "features_2026-01-05.parquet")

    obj = {"objective_id": "t", "labels": {"move_threshold": 50, "mae_tolerance": 6,
                                            "scheme": "triple_barrier"},
           "horizon": {"max_trade_horizon": "10m"}}
    import src.search.fold_cache as fc
    old_root = fc._CACHE_ROOT
    fc._CACHE_ROOT = tmp_path / "_fold_cache"
    try:
        arrays = build_or_load_fold_arrays(obj, pdir)
        assert arrays is not None
        X, y, ts, cols = arrays
        assert X.shape == (400, 2) and len(y) == 400
        assert set(np.unique(y)).issubset({0, 1, 2})
        expected = generate_labels_tbm(df["_close"], LabelConfig(50, 6, [120], "triple_barrier"))
        assert (y == expected["label"].to_numpy()).all()
    finally:
        fc._CACHE_ROOT = old_root
