"""Bundle inference policy (ADR-047) applied to Backtest predictions."""

from __future__ import annotations

import json

import pandas as pd

from src.backtest.weights_score import apply_inference_policy_to_preds, overlay_session
from src.search.finalize import _inference_policy_for


POLICY = {
    "tau": 0.65,
    "trading_start": "08:30",
    "trading_end": "10:00",
    "timezone": "America/Chicago",
    "sides": ["up"],
    "cooldown_bars": 120,
}


def _preds(rows):
    """rows: list of (utc_iso, direction, confidence)."""
    ts, d, c = zip(*rows)
    return pd.DataFrame({
        "ts": pd.to_datetime(list(ts), utc=True),
        "direction": list(d),
        "confidence": list(c),
    })


def test_inference_policy_for_unique_objective():
    pol = _inference_policy_for(
        {"session": {"timezone": "America/Chicago"}}, "nq-50pt-xgb-unique")
    assert pol is not None
    assert pol["tau"] == 0.50
    assert pol["sides"] == ["up"]
    assert pol["trading_end"] == "10:00"
    assert pol["cooldown_bars"] == 120
    assert _inference_policy_for({}, "nq-50pt-econ") is None


def test_overlay_session_replaces_trading_window():
    obj = {"session": {"trading_start": "08:30", "trading_end": "14:45",
                       "timezone": "America/Chicago"}}
    out = overlay_session(obj, POLICY)
    assert out["session"]["trading_end"] == "10:00"
    assert obj["session"]["trading_end"] == "14:45"  # original not mutated


def test_policy_flattens_outside_open_and_shorts():
    # 2026-06-02 is CDT (UTC−5): 08:30 CT = 13:30 UTC, 10:00 CT = 15:00 UTC.
    pred = _preds([
        ("2026-06-02T13:20:00Z", "up", 0.9),    # 08:20 CT — too early
        ("2026-06-02T13:35:00Z", "up", 0.9),    # 08:35 CT — keep
        ("2026-06-02T13:36:00Z", "down", 0.9),  # 08:36 CT short — flatten
        ("2026-06-02T15:01:00Z", "up", 0.9),    # 10:01 CT — too late
        ("2026-06-02T13:37:00Z", "up", 0.50),   # 08:37 CT under τ — flatten
    ])
    out = apply_inference_policy_to_preds(pred, tau=0.65, policy=POLICY)
    assert list(out["direction"]) == ["flat", "up", "flat", "flat", "flat"]


def test_policy_cooldown_keeps_first_fire():
    # Two longs 5s apart inside the window; 120-bar cooldown keeps only the first.
    pred = _preds([
        ("2026-06-02T13:35:00Z", "up", 0.9),
        ("2026-06-02T13:35:05Z", "up", 0.9),
        ("2026-06-02T13:45:05Z", "up", 0.9),  # 10m+5s later — keep
    ])
    out = apply_inference_policy_to_preds(pred, tau=0.65, policy=POLICY)
    assert list(out["direction"]) == ["up", "flat", "up"]


def test_policy_confidence_gate_off_keeps_under_tau():
    pred = _preds([
        ("2026-06-02T13:35:00Z", "up", 0.10),   # in window, under τ
        ("2026-06-02T13:20:00Z", "up", 0.10),   # too early — still flattened
    ])
    out = apply_inference_policy_to_preds(
        pred, tau=0.65, policy=POLICY, confidence_gate=False)
    assert list(out["direction"]) == ["up", "flat"]


def test_policy_noop_without_policy():
    pred = _preds([("2026-06-02T13:20:00Z", "down", 0.9)])
    out = apply_inference_policy_to_preds(pred, tau=0.65, policy=None)
    assert list(out["direction"]) == ["down"]


def test_load_serving_bundle_exposes_policy(tmp_path):
    from sklearn.dummy import DummyClassifier
    from src.search.finalize import write_serve_bundle
    from src.serving.bundle import load_serving_bundle

    clf = DummyClassifier(strategy="most_frequent")
    clf.fit([[0.0], [1.0]], [0, 1])
    dest = tmp_path / "bundle"
    write_serve_bundle(
        dest,
        artifact=clf,
        feature_cols=["f1"],
        objective_id="nq-50pt-xgb-unique",
        candidate_id="xgb_causal_topgroups",
        family="xgboost",
        run_ts="20260821T110058Z",
        extra_manifest={"inference_policy": POLICY},
    )
    bundle = load_serving_bundle(dest, validate=False)
    assert bundle["inference_policy"]["tau"] == 0.65
    assert json.loads((dest / "inference_policy.json").read_text())["sides"] == ["up"]
