"""ADR-092: native-TF dual-head OOF meta composer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.objectives.loader import load_objective
from src.search.tf_oof_meta import (
    CONTEXT_COLS,
    align_native_oof,
    assert_tf_oof_meta_ready,
    compose,
    infer_native_tf,
    iter_tf_oof_bases,
    load_custom_test_oof,
    match_tf_oof_family,
    missing_tf_oof_stores,
    topology_tf_oof_meta,
)


def test_slim_htf_pack_drops_raw_ohlc_and_pivots():
    assert "tr_ema_stack" in CONTEXT_COLS
    assert "rv_atr_ratio" in CONTEXT_COLS
    assert "sr_prior_high_dist" in CONTEXT_COLS
    assert "cs_atr_14" not in CONTEXT_COLS
    assert "cs_clv" not in CONTEXT_COLS
    assert "sr_pivot_dist" not in CONTEXT_COLS
    assert "ps_trend" not in CONTEXT_COLS


def test_objective_topology_unchanged():
    obj = load_objective("nq-50pt-econ-lg-st")
    cfg = topology_tf_oof_meta(obj)
    assert cfg is not None
    assert [b["native_tf"] for b in cfg["bases"]] == ["15s", "1m", "5m"]
    assert [c["native_tf"] for c in cfg["context"]] == ["1h", "1d"]
    assert cfg["outputs"] == ["p_final_long", "p_final_short"]
    assert cfg["meta"]["scheme"] == "walk_forward_OOF"


def test_joint_search_ready_when_stores_exist():
    obj = load_objective("nq-50pt-econ-lg-st")
    pairs = iter_tf_oof_bases(obj)
    assert [e.name for e, _tf in pairs] == ["lstm_15s", "xgb_1m", "xgb_5m"]
    assert [tf for _e, tf in pairs] == ["15s", "1m", "5m"]
    assert pairs[0][0].family_key == "attention_lstm"
    assert pairs[0][0].params_default.get("head") == "dual_binary"
    assert pairs[0][0].params_default.get("seq_len") == 60
    assert pairs[1][0].params_default.get("n_estimators") == 250
    assert pairs[1][0].params_default.get("max_depth") == 12
    assert pairs[2][0].params_default.get("max_depth") == 8
    missing = missing_tf_oof_stores(obj)
    if missing:
        with pytest.raises(NotImplementedError, match="stores missing"):
            assert_tf_oof_meta_ready(obj)
    else:
        assert_tf_oof_meta_ready(obj)
        assert_tf_oof_meta_ready(obj, experiment_type="model_selection")


def test_match_tf_oof_family_whitelist():
    obj = load_objective("nq-50pt-econ-lg-st")
    pairs = iter_tf_oof_bases(obj)
    lstm = next(e for e, _tf in pairs if e.name == "lstm_15s")
    xgb1 = next(e for e, _tf in pairs if e.name == "xgb_1m")
    assert match_tf_oof_family(lstm, "15s", None) is True
    assert match_tf_oof_family(lstm, "15s", ["lstm_15s"]) is True
    assert match_tf_oof_family(lstm, "15s", ["attention_lstm"]) is True
    assert match_tf_oof_family(lstm, "15s", ["15s"]) is True
    assert match_tf_oof_family(lstm, "15s", ["xgb_1m"]) is False
    assert match_tf_oof_family(xgb1, "1m", ["lstm_15s"]) is False
    assert match_tf_oof_family(xgb1, "1m", ["xgb_1m", "lstm_15s"]) is True


def test_infer_native_tf():
    ts15 = pd.date_range("2026-01-02", periods=20, freq="15s", tz="UTC")
    ts1m = pd.date_range("2026-01-02", periods=20, freq="60s", tz="UTC")
    ts5m = pd.date_range("2026-01-02", periods=20, freq="5min", tz="UTC")
    assert infer_native_tf(pd.DataFrame({"ts": ts15})) == "15s"
    assert infer_native_tf(pd.DataFrame({"ts": ts1m})) == "1m"
    assert infer_native_tf(pd.DataFrame({"ts": ts5m})) == "5m"


def _skilled_oof(ts: pd.DatetimeIndex, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(ts)
    y_long = rng.random(n) < 0.18
    y_short = (~y_long) & (rng.random(n) < 0.18)
    p_long = np.where(y_long, rng.uniform(0.45, 0.95, n), rng.uniform(0.02, 0.25, n))
    p_short = np.where(y_short, rng.uniform(0.45, 0.95, n), rng.uniform(0.02, 0.25, n))
    true = np.where(y_long, "up", np.where(y_short, "down", "flat"))
    return pd.DataFrame({
        "ts": ts, "p_long": p_long, "p_short": p_short,
        "direction_true": true, "direction": "flat", "confidence": np.maximum(p_long, p_short),
    })


def test_align_asof_uses_last_closed_bar():
    grid = pd.date_range("2026-01-02 14:00:00", periods=12, freq="15s", tz="UTC")
    m1 = pd.date_range("2026-01-02 14:00:00", periods=3, freq="60s", tz="UTC")
    # 15s 14:00:45 should see 1m 14:00:00; 14:01:00 sees 14:01:00.
    lstm = pd.DataFrame({
        "ts": grid,
        "p_long": np.linspace(0.1, 0.4, len(grid)),
        "p_short": np.linspace(0.2, 0.3, len(grid)),
        "direction_true": ["flat"] * len(grid),
    })
    xgb = pd.DataFrame({
        "ts": m1,
        "p_long": [0.11, 0.22, 0.33],
        "p_short": [0.01, 0.02, 0.03],
        "direction_true": ["flat"] * 3,
    })
    aligned = align_native_oof(
        {"lstm_15s": lstm, "xgb_1m": xgb},
        [{"id": "lstm_15s", "native_tf": "15s"},
         {"id": "xgb_1m", "native_tf": "1m"}],
        decision_timestep="15s",
    )
    row = aligned.loc[aligned["ts"] == pd.Timestamp("2026-01-02 14:00:45", tz="UTC")].iloc[0]
    assert row["xgb_1m__p_long"] == pytest.approx(0.11)
    row2 = aligned.loc[aligned["ts"] == pd.Timestamp("2026-01-02 14:01:00", tz="UTC")].iloc[0]
    assert row2["xgb_1m__p_long"] == pytest.approx(0.22)
    assert "tod_min" in aligned.columns and "session_min" in aligned.columns
    assert "lstm_15s__spread" in aligned.columns


def test_align_drops_rows_without_slower_base():
    grid = pd.date_range("2026-01-02 14:00:00", periods=8, freq="15s", tz="UTC")
    lstm = pd.DataFrame({
        "ts": grid, "p_long": 0.2, "p_short": 0.1, "direction_true": "flat",
    })
    # 1m only at 14:00:00 — 15s bars after 14:01:00 exceed 60s tolerance.
    xgb = pd.DataFrame({
        "ts": [pd.Timestamp("2026-01-02 14:00:00", tz="UTC")],
        "p_long": [0.5], "p_short": [0.1], "direction_true": ["flat"],
    })
    aligned = align_native_oof(
        {"lstm_15s": lstm, "xgb_1m": xgb},
        [{"id": "lstm_15s", "native_tf": "15s"},
         {"id": "xgb_1m", "native_tf": "1m"}],
    )
    assert aligned["ts"].max() <= pd.Timestamp("2026-01-02 14:01:00", tz="UTC")
    assert aligned["xgb_1m__p_long"].notna().all()


def test_compose_warmup_then_fires():
    ts15 = pd.date_range("2026-01-05 14:00:00", periods=600, freq="15s", tz="UTC")
    ts1m = pd.date_range("2026-01-05 14:00:00", periods=150, freq="60s", tz="UTC")
    lstm = _skilled_oof(ts15, seed=1)
    xgb = _skilled_oof(ts1m, seed=2)
    # Copy 15s labels onto 1m via asof so y is defined; composer uses 15s truth.
    out = compose(
        {"lstm_15s": lstm, "xgb_1m": xgb},
        bases=[{"id": "lstm_15s", "native_tf": "15s"},
               {"id": "xgb_1m", "native_tf": "1m"}],
        n_folds=3, embargo_bars=4, class_balance="sqrt",
        xgb_params={"n_estimators": 20, "max_depth": 2, "n_jobs": 1, "verbosity": 0},
    )
    assert "p_long" in out.columns and "p_final_long" in out.columns
    assert "p_short" in out.columns and "p_final_short" in out.columns
    assert "meta_finite" in out.columns
    b0 = out["meta_block"] == 0
    assert b0.any() and not bool(out.loc[b0, "meta_finite"].any())
    assert (out.loc[b0, "direction"] == "flat").all()
    later = out["meta_finite"]
    assert later.any()
    assert later.sum() < len(out)
    # Skilled synthetic OOF should produce some directional fires on later blocks.
    assert int(out.loc[later, "direction"].isin(["up", "down"]).sum()) >= 1
    names = out.attrs.get("meta_feature_names") or []
    assert any(n.startswith("lstm_15s__") for n in names)


def test_compose_can_drop_grid_base_from_x():
    ts15 = pd.date_range("2026-01-05 14:00:00", periods=240, freq="15s", tz="UTC")
    ts1m = pd.date_range("2026-01-05 14:00:00", periods=60, freq="60s", tz="UTC")
    lstm = _skilled_oof(ts15, seed=3)
    xgb = _skilled_oof(ts1m, seed=4)
    out = compose(
        {"lstm_15s": lstm, "xgb_1m": xgb},
        bases=[{"id": "lstm_15s", "native_tf": "15s"},
               {"id": "xgb_1m", "native_tf": "1m"}],
        n_folds=3, embargo_bars=2, class_balance="sqrt",
        xgb_params={"n_estimators": 10, "max_depth": 2, "n_jobs": 1, "verbosity": 0},
        skip_feature_prefixes=("lstm_15s__",),
    )
    names = out.attrs.get("meta_feature_names") or []
    assert names
    assert all(not n.startswith("lstm_15s__") for n in names)
    assert any(n.startswith("xgb_1m__") for n in names)


def test_load_custom_test_oof_rejects_collision(tmp_path, monkeypatch):
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path))
    run = tmp_path / "research" / "custom_test" / "nq-50pt-econ-lg-st" / "20260828T200039Z"
    for name in ("attention_lstm", "xgboost"):
        d = run / "models" / name
        d.mkdir(parents=True)
        pd.DataFrame({
            "ts": pd.date_range("2026-01-01", periods=3, freq="15s", tz="UTC"),
            "p_long": 0.2, "p_short": 0.1, "direction_true": "flat",
        }).to_parquet(d / "oof.parquet", index=False)
    with pytest.raises(ValueError, match="exactly one model OOF"):
        load_custom_test_oof("20260828T200039Z", objective_id="nq-50pt-econ-lg-st")
