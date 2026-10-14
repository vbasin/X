"""Join specs and the on-demand OOF analysis view. No database."""
import json

import pandas as pd
import pytest

from src.common.causal import CausalityError
from src.experiment.analysis import build_oof_analysis, load_training_frame
from src.experiment.artifacts import compact_oof_columns, validate_model_inputs
from src.experiment.model_identity import write_model_json
from src.common.run_layout import merge_run_json, model_dir


def _point(location: str, columns: list[str]) -> dict:
    return {
        "location": location,
        "columns": columns,
        "availability": "timestamp",
        "time_column": "ts",
    }


def test_compact_oof_drops_feature_columns_and_keeps_predictions():
    kept = compact_oof_columns(
        ["ts", "side", "y_move", "es_a", "p_move", "tau_0.95", "fold"],
        ["es_a"],
    )
    assert kept == ["ts", "side", "y_move", "p_move", "tau_0.95", "fold"]


def test_training_frame_joins_a_feature_file(tmp_path):
    labels = tmp_path / "labels.parquet"
    features = tmp_path / "features.parquet"
    pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"], "side": [1], "y_move": [1]}).to_parquet(labels)
    pd.DataFrame({"ts": ["2026-01-01T00:00:00Z"], "side": [1], "es_a": [0.5]}).to_parquet(features)
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "features": [_point(str(features), ["es_a"])],
        "join_keys": ["ts", "side"],
        "filters": [],
    }, tmp_path)
    assert frame.loc[0, "es_a"] == 0.5
    assert "y_move" in frame.columns
    gated = load_training_frame({
        "labels": {"location": str(labels)},
        "features": [_point(str(features), ["es_a"])],
        "join_keys": ["ts", "side"],
        "filters": [{"op": "eq", "left": "y_move", "right": 0}],
    }, tmp_path)
    assert gated.empty


def test_analysis_view_rejoins_features_without_writing_them_into_the_oof(monkeypatch, tmp_path):
    monkeypatch.setattr("src.common.run_layout.runs_dir", lambda: tmp_path)
    features = tmp_path / "features.parquet"
    pd.DataFrame({
        "ts": ["2026-01-01T00:00:00Z"], "side": [1], "es_a": [0.5],
    }).to_parquet(features, index=False)
    merge_run_json("obj", "t", {
        "pass": 1,
        "role": "directional_screen",
        "purpose": "Score the decision grid.",
        "models": [{"model_id": "abc123abc123", "model_type": "xgboost"}],
    })
    directory = model_dir("obj", "t", 1, "abc123abc123")
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "ts": ["2026-01-01T00:00:00Z"], "side": [1], "y_move": [1], "p_move": [0.4], "fold": [0],
    }).to_parquet(directory / "oof_predictions.parquet", index=False)
    write_model_json(
        directory, model_id="abc123abc123", model_type="xgboost",
        hyperparameters={"max_depth": 6}, target="y_move",
        feature_columns=["es_a"],
        inputs={
            "features": [_point(str(features), ["es_a"])],
            "join_keys": ["ts", "side"],
            "filters": [],
        },
    )
    frame = build_oof_analysis("obj", "t", 1, "abc123abc123")
    assert frame.loc[0, "es_a"] == 0.5
    assert "p_move" in frame.columns
    stored = set(pd.read_parquet(directory / "oof_predictions.parquet").columns)
    assert "es_a" not in stored
    manifest = json.loads((tmp_path / "obj" / "t" / "manifest.json").read_text())
    assert "run.json" not in manifest
    assert manifest["passes"][0]["role"] == "directional_screen"


def _write(path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)


def test_asof_join_keeps_1301_and_drops_130101(tmp_path):
    """13:01:00 may use data through 13:01:00, and must not use 13:01:01."""
    labels = tmp_path / "labels.parquet"
    features = tmp_path / "features.parquet"
    _write(labels, pd.DataFrame({
        "ts": [
            "2026-01-05T13:00:30Z",
            "2026-01-05T13:01:00Z",
            "2026-01-05T13:01:01Z",
            "2026-01-05T12:59:00Z",
        ],
        "side": [1, 1, 1, 1],
        "y_move": [1, 1, 1, 1],
    }))
    _write(features, pd.DataFrame({
        "ts": [
            "2026-01-05T13:00:00Z",
            "2026-01-05T13:01:00Z",
            "2026-01-05T13:01:01Z",
            "2026-01-05T13:01:00Z",
        ],
        "side": [1, 1, 1, -1],
        "es_a": [1.0, 2.0, 9.0, 7.0],
    }))
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "features": [_point(str(features), ["es_a"])],
        "join_keys": ["ts", "side"],
        "filters": [],
    }, tmp_path)
    by_ts = dict(zip(pd.to_datetime(frame["ts"], utc=True), frame["es_a"], strict=True))
    assert pd.isna(by_ts[pd.Timestamp("2026-01-05T12:59:00Z")])
    assert by_ts[pd.Timestamp("2026-01-05T13:00:30Z")] == 1.0
    assert by_ts[pd.Timestamp("2026-01-05T13:01:00Z")] == 2.0
    assert by_ts[pd.Timestamp("2026-01-05T13:01:01Z")] == 9.0
    assert set(frame["y_move"]) == {1}


def test_closed_bar_is_invisible_until_it_closes(tmp_path):
    labels = tmp_path / "labels.parquet"
    bars = tmp_path / "bars.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:00:30Z", "2026-01-05T13:01:00Z", "2026-01-05T13:02:00Z"],
        "side": [1, 1, 1],
        "y_move": [1, 0, 1],
    }))
    _write(bars, pd.DataFrame({
        "ts": ["2026-01-05T13:00:00Z", "2026-01-05T13:01:00Z"],
        "side": [1, 1],
        "htf_close": [1.0, 9.0],
    }))
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "features": [{
            "location": str(bars),
            "columns": ["htf_close"],
            "availability": "bar_close",
            "time_column": "ts",
            "bar_duration": "1min",
        }],
        "join_keys": ["ts", "side"],
        "filters": [],
    }, tmp_path)
    got = dict(zip(pd.to_datetime(frame["ts"], utc=True), frame["htf_close"], strict=True))
    assert pd.isna(got[pd.Timestamp("2026-01-05T13:00:30Z")])
    assert got[pd.Timestamp("2026-01-05T13:01:00Z")] == 1.0
    assert got[pd.Timestamp("2026-01-05T13:02:00Z")] == 9.0


def test_open_session_profile_is_not_joined(tmp_path):
    labels = tmp_path / "labels.parquet"
    profiles = tmp_path / "profiles.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T19:01:00Z", "2026-01-05T21:00:00Z"],
        "session": ["2026-01-05", "2026-01-05"],
        "side": [1, 1],
        "y_move": [1, 1],
    }))
    _write(profiles, pd.DataFrame({
        "session": ["2026-01-05", "2026-01-02"],
        "prior_r": [9.0, 1.0],
    }))
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "context": [{
            "location": str(profiles),
            "columns": ["prior_r"],
            "availability": "session_close",
            "session_column": "session",
            "session_close": "15:00",
            "timezone": "America/Chicago",
        }],
        "join_keys": ["ts", "session", "side"],
        "filters": [],
    }, tmp_path)
    got = dict(zip(pd.to_datetime(frame["ts"], utc=True), frame["prior_r"], strict=True))
    # 19:01Z is 13:01 CT, before the 15:00 CT close. 21:00Z is that close.
    assert got[pd.Timestamp("2026-01-05T19:01:00Z")] == 1.0
    assert got[pd.Timestamp("2026-01-05T21:00:00Z")] == 9.0
    assert str(pd.Timestamp(frame.loc[0, "session"]).date()) == "2026-01-05"


def test_scale_fit_on_later_rows_does_not_join_backward(tmp_path):
    labels = tmp_path / "labels.parquet"
    scale = tmp_path / "scale.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:01:00Z", "2026-01-05T13:05:00Z"],
        "side": [1, 1],
        "y_move": [1, 0],
    }))
    _write(scale, pd.DataFrame({
        "ts": ["2026-01-05T13:05:00Z"],
        "side": [1],
        "train_scale": [1.0],
    }))
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "context": [_point(str(scale), ["train_scale"])],
        "join_keys": ["ts", "side"],
        "filters": [],
    }, tmp_path)
    early = frame.loc[pd.to_datetime(frame["ts"], utc=True) == pd.Timestamp("2026-01-05T13:01:00Z"), "train_scale"]
    late = frame.loc[pd.to_datetime(frame["ts"], utc=True) == pd.Timestamp("2026-01-05T13:05:00Z"), "train_scale"]
    assert pd.isna(early).all()
    assert float(late.iloc[0]) == 1.0


def test_upstream_prediction_after_the_decision_is_not_joined(tmp_path):
    labels = tmp_path / "labels.parquet"
    scores = tmp_path / "oof.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:01:00Z", "2026-01-05T13:02:00Z"],
        "side": [1, 1],
        "y_reach": [1, 0],
    }))
    _write(scores, pd.DataFrame({
        "ts": ["2026-01-05T13:02:00Z"],
        "side": [1],
        "p_move": [0.9],
    }))
    frame = load_training_frame({
        "labels": {"location": str(labels)},
        "predictions": [{"location": str(scores)}],
        "join_keys": ["ts", "side"],
        "filters": [],
    }, tmp_path)
    got = dict(zip(pd.to_datetime(frame["ts"], utc=True), frame["p_move"], strict=True))
    assert pd.isna(got[pd.Timestamp("2026-01-05T13:01:00Z")])
    assert got[pd.Timestamp("2026-01-05T13:02:00Z")] == 0.9


def test_feature_source_must_declare_availability():
    missing = validate_model_inputs({
        "labels": {"location": "labels.parquet"},
        "features": [{"location": "features.parquet", "columns": ["es_a"]}],
        "join_keys": ["ts", "side"],
    })
    assert "features.availability" in missing


def test_left_labeled_bar_file_is_refused(tmp_path):
    labels = tmp_path / "labels.parquet"
    bars = tmp_path / "bars.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:00:30Z"], "side": [1], "y_move": [1],
    }))
    _write(bars, pd.DataFrame({
        "ts": ["2026-01-05T13:00:00Z"],
        "bar_end_ts": ["2026-01-05T13:01:00Z"],
        "side": [1],
        "htf_close": [9.0],
    }))
    with pytest.raises(CausalityError):
        load_training_frame({
            "labels": {"location": str(labels)},
            "features": [_point(str(bars), ["htf_close"])],
            "join_keys": ["ts", "side"],
            "filters": [],
        }, tmp_path)


def test_short_bar_duration_is_refused_when_bar_end_is_later(tmp_path):
    labels = tmp_path / "labels.parquet"
    bars = tmp_path / "bars.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:00:01Z"], "side": [1], "y_move": [1],
    }))
    _write(bars, pd.DataFrame({
        "ts": ["2026-01-05T13:00:00Z"],
        "bar_end_ts": ["2026-01-05T13:01:00Z"],
        "side": [1],
        "htf_close": [9.0],
    }))
    with pytest.raises(CausalityError):
        load_training_frame({
            "labels": {"location": str(labels)},
            "features": [{
                "location": str(bars),
                "columns": ["htf_close"],
                "availability": "bar_close",
                "bar_duration": "1s",
            }],
            "join_keys": ["ts", "side"],
            "filters": [],
        }, tmp_path)


def test_asof_refuses_a_feature_file_without_the_side_key(tmp_path):
    labels = tmp_path / "labels.parquet"
    features = tmp_path / "features.parquet"
    _write(labels, pd.DataFrame({
        "ts": ["2026-01-05T13:01:00Z"], "side": [1], "y_move": [1],
    }))
    _write(features, pd.DataFrame({
        "ts": ["2026-01-05T13:01:00Z"], "es_a": [9.0],
    }))
    with pytest.raises(CausalityError):
        load_training_frame({
            "labels": {"location": str(labels)},
            "features": [_point(str(features), ["es_a"])],
            "join_keys": ["ts", "side"],
            "filters": [],
        }, tmp_path)
