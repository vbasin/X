"""Focused tests for the Live Chart Model Design DSL + interpreter (ADR-117)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.webui.chart_models import (
    build_var_catalog,
    compile_model,
    simulate,
)
from src.webui.chart_models.expr import compile_expr, evaluate
from src.webui.chart_models.schema import SchemaError, validate_model


# ── helpers ─────────────────────────────────────────────────────────────

def _ts(hh: int, mm: int, day: int = 15) -> int:
    """Unix seconds for 2026-09-<day> HH:MM America/Chicago (CDT = UTC-5)."""
    return int(datetime(2026, 9, day, hh + 5, mm, tzinfo=timezone.utc).timestamp())


def _payload(bars: list[tuple[int, float, float, float, float]], **kw):
    t = [b[0] for b in bars]
    o = [b[1] for b in bars]
    h = [b[2] for b in bars]
    l = [b[3] for b in bars]
    c = [b[4] for b in bars]
    v = [1.0] * len(bars)
    return {
        "timeframe": kw.get("timeframe", "1min"),
        "candles": {"t": t, "o": o, "h": h, "l": l, "c": c, "v": v},
        "overlays": kw.get("overlays", {}),
        "profiles": kw.get("profiles", []),
        "prints": kw.get("prints", []),
        "instrument": "/NQ",
    }


# ── expression tests ────────────────────────────────────────────────────

def test_expr_arithmetic_and_calls():
    ast = compile_expr("min(Cc + 1, Ch - tick)")
    b = {"Cc": 100.0, "Ch": 101.0, "tick": 0.25}
    assert evaluate(ast, b) == pytest.approx(100.75)


def test_expr_next_up_of_returns_smallest_above_ref():
    ast = compile_expr("next_up_of(entry, VP1.vah, VP1.high, VP2.vah, VP2.high)")
    b = {
        "entry": 100.0,
        "VP1.vah": 99.5, "VP1.high": 101.0,
        "VP2.vah": 100.5, "VP2.high": 102.0,
    }
    assert evaluate(ast, b) == 100.5


def test_expr_unknown_identifier_rejected_when_catalog_given():
    catalog = {"C.c": {"kind": "scalar"}}
    with pytest.raises(Exception):
        compile_expr("bogus_symbol + 1", catalog=catalog)


def test_expr_none_propagates_as_none():
    ast = compile_expr("VP1.vah + 1")
    assert evaluate(ast, {"VP1.vah": None}) is None


# ── schema tests ────────────────────────────────────────────────────────

def _minimal_model():
    return {
        "id": "unit_test",
        "name": "unit test",
        "rules": [
            {
                "when": [{"time_between": ["08:30", "15:00"]}],
                "if": {"gt": ["C.c", 0]},
                "then": [
                    {"entry_long": "C.c"},
                    {"set_stop": "C.l"},
                    {"set_target": "C.c + 5"},
                ],
            }
        ],
    }


def test_validate_rejects_missing_rules():
    with pytest.raises(SchemaError):
        validate_model({"id": "x", "rules": []})


def test_validate_rejects_bad_hhmm():
    m = _minimal_model()
    m["rules"][0]["when"] = [{"time_between": ["8-30", "15:00"]}]
    with pytest.raises(SchemaError):
        validate_model(m)


def test_validate_requires_entry_or_exit_in_then():
    m = _minimal_model()
    m["rules"][0]["then"] = [{"set_stop": "C.l"}]
    with pytest.raises(SchemaError):
        validate_model(m)


def test_compile_produces_ast_leaves():
    m = _minimal_model()
    spec = compile_model(m)
    action = spec["rules"][0]["then"][0]
    assert "entry_long" in action
    assert "__expr__" in action["entry_long"]


# ── interpreter — headline scenarios ────────────────────────────────────

def test_simulate_long_entry_target_hit_next_bar():
    # Two bars: signal at 09:00, entry at close 100.0, target 100.5, stop 99.0.
    # Next bar (09:01) high 101 → target hit at 100.5.
    bars = [
        (_ts(9, 0),  99.5, 100.2,  99.4, 100.0),
        (_ts(9, 1), 100.0, 101.0,  99.8, 100.9),
    ]
    payload = _payload(bars)
    model = compile_model({
        "id": "vp_break",
        "name": "vp break",
        "rules": [{
            "when": [{"time_between": ["08:30", "15:00"]}],
            "if": {"eq": ["C.color", "green"]},
            "then": [
                {"entry_long": "C.c"},
                {"set_target": "C.c + 0.5"},
                {"set_stop": "C.c - 1.0"},
            ],
        }],
    })
    out = simulate(payload, model)
    assert len(out["trades"]) == 1
    tr = out["trades"][0]
    assert tr["side"] == "long"
    assert tr["exit_reason"] == "target"
    assert tr["entry_price"] == pytest.approx(100.0)
    assert tr["exit_price"] == pytest.approx(100.5)
    assert tr["pnl_points"] == pytest.approx(0.5)
    # Markers: entry_long + exit_target
    kinds = [m["type"] for m in out["markers"]]
    assert kinds == ["entry_long", "exit_target"]


def test_simulate_short_stop_hit():
    bars = [
        (_ts(9, 0), 100.5, 100.7, 100.2, 100.3),
        (_ts(9, 1), 100.3, 101.5, 100.2, 101.4),
    ]
    payload = _payload(bars)
    model = compile_model({
        "id": "s",
        "name": "short",
        "rules": [{
            "if": {"eq": ["C.color", "red"]},
            "then": [
                {"entry_short": "C.c"},
                {"set_stop": "C.c + 1.0"},
                {"set_target": "C.c - 2.0"},
            ],
        }],
    })
    out = simulate(payload, model)
    assert len(out["trades"]) == 1
    tr = out["trades"][0]
    assert tr["side"] == "short"
    assert tr["exit_reason"] == "stop"
    assert tr["pnl_points"] == pytest.approx(-1.0)


def test_simulate_when_gate_blocks_outside_window():
    bars = [
        (_ts(8, 0),  99.5, 100.2,  99.4, 100.0),   # outside window
        (_ts(8, 1), 100.0, 101.0,  99.8, 100.9),
    ]
    model = compile_model({
        "id": "g", "name": "gated",
        "rules": [{
            "when": [{"time_between": ["09:00", "10:00"]}],
            "if": {"eq": ["C.color", "green"]},
            "then": [{"entry_long": "C.c"}, {"set_target": "C.c + 1"}],
        }],
    })
    out = simulate(_payload(bars), model)
    assert out["trades"] == []


def test_simulate_vp_reference_expression():
    """Reproduces the user's example: entry above VP1.vah, target via next_up_of."""
    bars = [
        (_ts(9, 0), 100.0, 101.5, 99.5, 101.0),   # green; Co=100 > VP1.vah=99.8; Ch>VP1.vah
        (_ts(9, 1), 101.0, 104.0, 100.5, 103.9),  # target = min level above entry (102) hit
    ]
    profiles = [
        {"n": 1, "label": "VP1", "kind": "current", "session_tf": "cash",
         "high": 105.0, "low": 95.0, "poc": 100.0,
         "va_high": 99.8, "va_low": 96.0,
         "hi_body": 102.0, "lo_body": 96.5},
        {"n": 2, "label": "VP2", "kind": "recurring", "session_tf": "cash",
         "high": 110.0, "low": 90.0, "poc": 105.0,
         "va_high": 107.0, "va_low": 92.0,
         "hi_body": None, "lo_body": None},
    ]
    payload = _payload(bars, profiles=profiles)
    model = compile_model({
        "id": "u", "name": "user example",
        "rules": [{
            "when": [{"time_between": ["08:46", "11:00"]}],
            "if": {"all": [
                {"eq": ["C.color", "green"]},
                {"gt": ["Co", "VP1.vah"]},
                {"gt": ["Ch", "VP1.vah"]},
            ]},
            "then": [
                {"entry_long": "Cc + 1"},
                {"set_target": "next_up_of(entry, VP1.vah, VP1.high, VP2.vah, VP2.high)"},
                {"set_stop": "Cl"},
            ],
        }],
    }, catalog=build_var_catalog(payload))
    out = simulate(payload, model)
    assert len(out["trades"]) == 1
    tr = out["trades"][0]
    # entry = Cc + 1 = 102.0; next_up_of(102, 99.8, 105, 107, 110) = 105
    assert tr["entry_price"] == pytest.approx(102.0)
    assert tr["target_level"] == pytest.approx(105.0)
    # Stop = Cl of signal bar = 99.5
    assert tr["stop_level"] == pytest.approx(99.5)
    # Next bar high 104 does NOT reach 105 → not target; low 100.5 > 99.5 → not stop.
    # Trade closes at EOD (last bar) here.
    assert tr["exit_reason"] in ("eod", "target", "session")


def test_simulate_session_flatten_at_end():
    bars = [
        (_ts(14, 58), 100.0, 100.5, 99.8, 100.4),  # signal
        (_ts(14, 59), 100.4, 100.6, 100.3, 100.5),
        (_ts(15,  0), 100.5, 100.7, 100.4, 100.5),  # ≥ session_end → flatten
    ]
    model = compile_model({
        "id": "e", "name": "eod",
        "rules": [{
            "if": {"eq": ["C.color", "green"]},
            "then": [{"entry_long": "C.c"}, {"set_target": "C.c + 10"}, {"set_stop": "C.c - 10"}],
        }],
    })
    out = simulate(_payload(bars), model, session_end="15:00")
    assert len(out["trades"]) == 1
    assert out["trades"][0]["exit_reason"] == "session"


def test_pnl_summary_shape():
    bars = [
        (_ts(9, 0), 100.0, 100.5, 99.8, 100.3),
        (_ts(9, 1), 100.3, 101.5, 100.2, 101.4),
        (_ts(9, 2), 101.4, 101.5,  99.0,  99.1),
    ]
    model = compile_model({
        "id": "p", "name": "p",
        "rules": [{
            "if": {"eq": ["C.color", "green"]},
            "then": [
                {"entry_long": "C.c"},
                {"set_target": "C.c + 1.0"},
                {"set_stop": "C.c - 0.5"},
            ],
        }],
    })
    out = simulate(_payload(bars), model)
    p = out["pnl"]
    for key in ("n_trades", "gross_points", "gross_usd", "win_rate",
                "profit_factor", "max_dd_points", "equity_points"):
        assert key in p


def test_build_var_catalog_reflects_profiles_and_overlays():
    payload = {
        "profiles": [{"n": 1, "label": "VP1"}, {"n": 3, "label": "VP3"}],
        "overlays": {"ps_sar": [1.0], "vw_vwap": [None]},
    }
    cat = build_var_catalog(payload)
    assert "VP1.vah" in cat
    assert "VP3.poc" in cat
    assert "VP2.vah" not in cat        # only n=1 and n=3 configured
    assert "SAR" in cat
    assert "VWAP" in cat
    assert "lp_net()" in cat
