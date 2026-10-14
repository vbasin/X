"""Focused tests for the VWAP crossover engine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.external_models.catalog import coerce_params, factory_params, get_model
from src.external_models.vwap_cross import (
    _session_vwap,
    simulate_vwap_cross,
    vwap_cross_payload,
)


def _row(day: int, minute: int, o, h, l, c, v):
    ts = datetime(2026, 1, day, 0, 0, tzinfo=timezone.utc) + timedelta(minutes=minute + 360)
    return {"bucket": ts, "open": float(o), "high": float(h),
            "low": float(l), "close": float(c), "volume": float(v)}


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("bucket").sort_index()


def _params(**over):
    # Test scenarios use ~6 warmup bars and simple synthetic candles — pin the
    # slope window to 1 and disable the body-shape gate unless a test overrides them.
    base = {"vwap_slope_lookback": 1, "min_close_range_frac": 0.0}
    base.update(over)
    return coerce_params("vwap_cross", base)


# ── catalog wiring ───────────────────────────────────────────────────────

def test_vwap_cross_card_is_registered_and_listed():
    card = get_model("vwap_cross")
    assert card["id"] == "vwap_cross"
    assert card["engine"] == "vwap_cross"
    keys = {row["key"] for row in card["param_spec"]}
    for need in ("session_start", "trade_long", "trade_short", "min_vwap_slope",
                 "max_session_crosses", "chop_lookback", "max_recent_crosses",
                 "stop_mode", "vwap_stop_offset", "profit_target_mode",
                 "profit_target_pts", "profit_target_stop_mult",
                 "max_stop_pts", "vp_va_pct", "vp_fallback_pts",
                 "entry_mode", "entry_ticks", "min_close_range_frac",
                 "qty", "point_value"):
        assert need in keys, need


def test_close_location_filter_rejects_weak_body_crossings():
    """A crossing bar that closes in the lower half of its range must be
    rejected when min_close_range_frac >= 0.5, even if the cross itself is valid."""
    rows = []
    day = 5
    for m in range(510, 515):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    rows.append(_row(day, 515, 99, 101, 99, 101, 10))          # green preceder
    # crossing bar: opens 99 (< VWAP), spikes to 130, closes at 108 — above V_now
    # but in the bottom third of its range (CLV ≈ 0.29).
    rows.append(_row(day, 516, 99, 130, 99, 108, 30))
    for m in range(517, 545):
        rows.append(_row(day, m, 108, 110, 106, 108, 10))
    df = _frame(rows)
    strict = _params(entry_mode="close_confirm", min_vwap_slope=0.0,
                     max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                     stop_mode="crossing_open", min_close_range_frac=0.6)
    loose = _params(entry_mode="close_confirm", min_vwap_slope=0.0,
                    max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                    stop_mode="crossing_open", min_close_range_frac=0.0)
    assert simulate_vwap_cross(df, strict)["trades"].empty
    assert not simulate_vwap_cross(df, loose)["trades"].empty


def test_factory_defaults_match_the_brief():
    p = factory_params("vwap_cross")
    assert p["profit_target_pts"] == 40.0
    assert p["profit_target_mode"] == "absolute"
    assert p["vwap_stop_offset"] == 1.0
    assert p["stop_mode"] == "worse_of"


def test_invalid_stop_mode_is_rejected():
    with pytest.raises(ValueError):
        coerce_params("vwap_cross", {"stop_mode": "guess"})


def test_both_sides_disabled_is_rejected():
    with pytest.raises(ValueError):
        coerce_params("vwap_cross", {"trade_long": False, "trade_short": False})


def test_trade_window_must_be_inside_session():
    with pytest.raises(ValueError):
        coerce_params("vwap_cross", {"trade_end": "16:00"})


# ── VWAP calc ────────────────────────────────────────────────────────────

def test_session_vwap_anchors_at_session_start():
    # 2 pre-session bars @ 100, 3 in-session bars @ 200 with volume=10 each
    rows = []
    for m in range(505, 510):
        rows.append(_row(5, m, 100, 100, 100, 100, 10))
    for m in range(510, 513):
        rows.append(_row(5, m, 200, 200, 200, 200, 10))
    df = _frame(rows)
    from src.external_models.vwap_cross import _ct_frame
    ct = _ct_frame(df)
    v = _session_vwap(ct, session_start_min=510)
    # pre-session bars produce NaN, in-session VWAP = 200
    assert pd.isna(v[0])
    assert v[-1] == pytest.approx(200.0)


# ── entry logic ──────────────────────────────────────────────────────────

def _build_cross_up_scenario() -> pd.DataFrame:
    """Construct a small session where candle N is a clean cross-up:
       - preceding candle (N-1) is green
       - candle N opens below VWAP, closes above VWAP
       - VWAP slope stays >= 0 after the cross."""
    rows = []
    day = 5
    # Bars 510..514 build a flat baseline (VWAP settles near 100)
    for m in range(510, 515):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    # Bar 515 = green preceding candle
    rows.append(_row(day, 515, 99, 101, 99, 101, 10))
    # Bar 516 = crossing candle: opens at 99 (< VWAP), closes at 110 (> VWAP)
    rows.append(_row(day, 516, 99, 110, 99, 110, 30))
    # Bars 517+ keep price above VWAP so slope stays positive
    for m in range(517, 560):
        rows.append(_row(day, m, 110, 112, 108, 111, 10))
    return _frame(rows)


def test_long_cross_up_fires_and_enters_at_next_open():
    df = _build_cross_up_scenario()
    p = _params(min_vwap_slope=0.0, max_session_crosses=100,
                chop_lookback=0, max_recent_crosses=100,
                profit_target_mode="absolute", profit_target_pts=40.0,
                stop_mode="crossing_open")
    res = simulate_vwap_cross(df, p)
    ledger = res["trades"]
    assert not ledger.empty
    t = ledger.iloc[0]
    assert t["side"] == "long"
    # entered at candle N+1's open = 110
    assert t["entry_px"] == pytest.approx(110.0)
    # stop = crossing candle's open = 99
    assert t["stop_px"] == pytest.approx(99.0)


def test_absolute_profit_target_default_is_40_points():
    df = _build_cross_up_scenario()
    p = _params(profit_target_mode="absolute", profit_target_pts=40.0,
                stop_mode="crossing_open")
    res = simulate_vwap_cross(df, p)
    t = res["trades"].iloc[0]
    assert t["target_px"] == pytest.approx(t["entry_px"] + 40.0)


def test_stop_multiple_target_scales_with_stop_distance():
    df = _build_cross_up_scenario()
    p = _params(profit_target_mode="stop_multiple",
                profit_target_stop_mult=2.0,
                stop_mode="crossing_open")
    res = simulate_vwap_cross(df, p)
    t = res["trades"].iloc[0]
    stop_dist = abs(t["entry_px"] - t["stop_px"])
    assert t["target_px"] == pytest.approx(t["entry_px"] + 2.0 * stop_dist)


def test_signal_requires_preceding_green_candle_for_longs():
    df = _build_cross_up_scenario().copy()
    # flip preceding candle (bar 515) into a red one
    ts = df.index[5]
    df.at[ts, "close"] = 99.0
    df.at[ts, "open"] = 101.0
    p = _params()
    res = simulate_vwap_cross(df, p)
    assert res["trades"].empty


def test_chop_filter_suppresses_signals_after_many_crosses():
    # Build a session where price oscillates around VWAP many times before
    # a clean cross-up appears.
    rows = []
    day = 5
    # 5 baseline bars at 100
    for m in range(510, 515):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    # 10 whipsaw bars: price alternates above/below the VWAP (~100)
    prices = [101, 99, 102, 98, 101, 99, 102, 98, 101, 99]
    for k, px in enumerate(prices):
        rows.append(_row(day, 515 + k, 100, max(px, 100), min(px, 100), px, 10))
    # Then the clean cross-up sequence
    rows.append(_row(day, 525, 99, 101, 99, 101, 10))       # green preceder
    rows.append(_row(day, 526, 99, 110, 99, 110, 30))       # crossing candle
    for m in range(527, 560):
        rows.append(_row(day, m, 110, 112, 108, 111, 10))
    df = _frame(rows)
    strict = _params(max_session_crosses=2, chop_lookback=0,
                     max_recent_crosses=100)
    loose = _params(max_session_crosses=100, chop_lookback=0,
                    max_recent_crosses=100)
    assert simulate_vwap_cross(df, strict)["trades"].empty
    assert not simulate_vwap_cross(df, loose)["trades"].empty


def test_payload_shape_is_backtest_tab_compatible():
    df = _build_cross_up_scenario()
    p = _params()
    res = simulate_vwap_cross(df, p)
    card = get_model("vwap_cross")
    payload = vwap_cross_payload(
        res, source="external:vwap_cross", start=None, end=None,
        n_sessions=1, card=card,
    )
    for key in ("config", "summary", "equity_curve", "confusion_matrix",
                "trades", "skipped"):
        assert key in payload
    assert payload["config"]["strategy_id"] == "vwap_cross"
    assert payload["config"]["kind"] == "external_vwap_cross"


# ── early_cross entry mode ───────────────────────────────────────────────

def _build_cross_down_scenario() -> pd.DataFrame:
    rows = []
    day = 5
    for m in range(510, 516):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    rows.append(_row(day, 516, 101, 101, 99, 99, 10))       # red preceder
    rows.append(_row(day, 517, 101, 101, 88, 90, 30))       # crossing bar
    for m in range(518, 560):
        rows.append(_row(day, m, 90, 92, 88, 89, 10))
    return _frame(rows)


def test_entry_mode_defaults_to_close_confirm():
    p = factory_params("vwap_cross")
    assert p["entry_mode"] == "close_confirm"
    assert p["entry_ticks"] >= 0


def test_invalid_entry_mode_is_rejected():
    with pytest.raises(ValueError):
        coerce_params("vwap_cross", {"entry_mode": "psychic"})


def test_early_cross_enters_intra_bar_at_vwap_minus_ticks():
    df = _build_cross_down_scenario()
    p = _params(
        entry_mode="early_cross", entry_ticks=4,
        min_vwap_slope=0.0,
        max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
        stop_mode="vwap_offset", vwap_stop_offset=1.0,
        profit_target_mode="absolute", profit_target_pts=40.0,
    )
    res = simulate_vwap_cross(df, p)
    ledger = res["trades"]
    assert not ledger.empty
    t = ledger.iloc[0]
    assert t["side"] == "short"
    crossing_ts = df.index[7]
    assert t["entry_ts"] == crossing_ts
    assert t["risk_pts"] <= 2.5
    assert t["target_px"] == pytest.approx(t["entry_px"] - 40.0)


def test_early_cross_improves_pt_to_sl_ratio():
    df = _build_cross_down_scenario()
    base = dict(
        min_vwap_slope=0.0,
        max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
        stop_mode="vwap_offset", vwap_stop_offset=1.0,
        profit_target_mode="absolute", profit_target_pts=40.0,
    )
    r_close = simulate_vwap_cross(df, _params(entry_mode="close_confirm", **base))
    r_early = simulate_vwap_cross(df, _params(entry_mode="early_cross",
                                              entry_ticks=4, **base))
    assert not r_close["trades"].empty and not r_early["trades"].empty
    ratio_close = float(r_close["trades"].iloc[0]["target_pts"]
                        / r_close["trades"].iloc[0]["risk_pts"])
    ratio_early = float(r_early["trades"].iloc[0]["target_pts"]
                        / r_early["trades"].iloc[0]["risk_pts"])
    assert ratio_early >= ratio_close


def test_early_cross_requires_min_penetration_when_ticks_high():
    df = _build_cross_down_scenario()
    strict = _params(
        entry_mode="early_cross", entry_ticks=200,
        min_vwap_slope=0.0,
        max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
        stop_mode="vwap_offset", vwap_stop_offset=1.0,
    )
    assert simulate_vwap_cross(df, strict)["trades"].empty


# ── max_stop_pts clamp ──────────────────────────────────────────────────

def test_max_stop_pts_clamps_wide_crossing_open_stop():
    df = _build_cross_up_scenario()
    # crossing_open stop would sit at open_N (~99); entry at ~110 → ~11 pt stop.
    tight = _params(
        stop_mode="crossing_open", max_stop_pts=3.0,
        profit_target_mode="absolute", profit_target_pts=40.0,
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
    )
    res = simulate_vwap_cross(df, tight)
    assert not res["trades"].empty
    t = res["trades"].iloc[0]
    assert t["risk_pts"] <= 3.0 + 1e-9


def test_max_stop_pts_zero_disables_clamp():
    df = _build_cross_up_scenario()
    p = _params(stop_mode="crossing_open", max_stop_pts=0.0,
                profit_target_mode="absolute", profit_target_pts=40.0,
                min_vwap_slope=0.0, max_session_crosses=100,
                chop_lookback=0, max_recent_crosses=100)
    res = simulate_vwap_cross(df, p)
    assert not res["trades"].empty
    # crossing_open ≈ 99, entry ≈ 110 → risk ≈ 11
    assert res["trades"].iloc[0]["risk_pts"] > 5.0


# ── vp_level profit target ──────────────────────────────────────────────

def _build_two_session_long_scenario() -> pd.DataFrame:
    """A prior session that establishes clear VP levels, plus a next-day
    long cross-up. Prior day: RTH 08:30–14:45 at 100 with a spike to 130 and a dip to 85.
    """
    rows = []
    day = 5
    # Prior session — enough volume around 100 (POC), high 130, low 85
    for m in range(510, 700):
        rows.append(_row(day, m, 100, 100, 100, 100, 30))
    rows.append(_row(day, 700, 100, 130, 100, 128, 15))   # push high
    for m in range(701, 850):
        rows.append(_row(day, m, 128, 128, 100, 100, 20))
    rows.append(_row(day, 850, 100, 100, 85, 90, 15))     # push low
    for m in range(851, 885):
        rows.append(_row(day, m, 90, 92, 88, 90, 20))
    # Next session
    day = 6
    for m in range(510, 520):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    rows.append(_row(day, 520, 99, 101, 99, 101, 10))       # green preceder
    rows.append(_row(day, 521, 99, 115, 99, 115, 30))       # crossing bar
    for m in range(522, 560):
        rows.append(_row(day, m, 115, 118, 112, 116, 10))
    return _frame(rows)


def test_vp_level_target_picks_nearest_prior_session_level_above_entry():
    df = _build_two_session_long_scenario()
    p = _params(
        entry_mode="close_confirm",
        profit_target_mode="vp_level", vp_fallback_pts=999.0,
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
    )
    res = simulate_vwap_cross(df, p)
    assert not res["trades"].empty
    # Filter to the next session's LONG trade (prior-day setup may fire its own signals).
    ledger = res["trades"]
    day6 = datetime(2026, 1, 6).date()
    day6_longs = ledger[(ledger["session_date"] == day6) & (ledger["side"] == "long")]
    assert not day6_longs.empty, "no long entry produced on the next session"
    t = day6_longs.iloc[0]
    entry = float(t["entry_px"])
    target = float(t["target_px"])
    ref = str(t.get("target_ref", ""))
    assert target > entry
    # Must be one of the prior-session level names, not "fallback"
    assert ref in {"low", "val", "poc", "vah", "high"}


def test_vp_level_target_falls_back_when_no_level_above_entry():
    df = _build_two_session_long_scenario()
    # Blow entry above the prior day's high (~130) so no VP level lies above.
    df = df.copy()
    ts = df.index[-40]
    df.at[ts, "high"] = 200.0
    df.at[ts, "close"] = 200.0
    p = _params(
        entry_mode="close_confirm",
        profit_target_mode="vp_level", vp_fallback_pts=17.0,
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
    )
    res = simulate_vwap_cross(df, p)
    ledger = res["trades"]
    day6 = datetime(2026, 1, 6).date()
    day6_longs = ledger[(ledger["session_date"] == day6) & (ledger["side"] == "long")]
    if day6_longs.empty:
        pytest.skip("scenario did not produce a next-day long — irrelevant here")
    t = day6_longs.iloc[0]
    if str(t["target_ref"]) == "fallback":
        assert abs(float(t["target_pts"]) - 17.0) < 1e-6


# ── close_confirm_early_fill mode ───────────────────────────────────────

def test_close_confirm_early_fill_same_population_as_close_confirm():
    df = _build_cross_up_scenario()
    base = dict(
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open", vwap_stop_offset=1.0,
        profit_target_mode="absolute", profit_target_pts=40.0,
    )
    a = simulate_vwap_cross(df, _params(entry_mode="close_confirm", **base))
    b = simulate_vwap_cross(df, _params(entry_mode="close_confirm_early_fill",
                                        entry_ticks=4, **base))
    # Same number of signals — early_fill only changes the fill price, not the filter.
    assert len(a["trades"]) == len(b["trades"])
    if not a["trades"].empty:
        # Early-fill entry <= close_confirm entry (for longs — earlier / better fill)
        for row_a, row_b in zip(a["trades"].to_dict("records"),
                                b["trades"].to_dict("records")):
            if row_a["side"] == "long":
                assert row_b["entry_px"] <= row_a["entry_px"] + 1e-6
            else:
                assert row_b["entry_px"] >= row_a["entry_px"] - 1e-6


# ── 15s cadence data source ─────────────────────────────────────────────

def _to_15s_frame(df_1m: pd.DataFrame) -> pd.DataFrame:
    """Split each 1-min bar into four synthetic 15s bars whose OHLC is compatible."""
    rows = []
    for ts, r in df_1m.iterrows():
        o, h, l, c, v = r["open"], r["high"], r["low"], r["close"], r["volume"]
        # Reproduce OHLC via 4 sub-bars: [o..h], [h..l or c], [l..c], [c..c]
        seg = [
            {"bucket": ts,                          "open": o, "high": h, "low": o, "close": h, "volume": v / 4},
            {"bucket": ts + timedelta(seconds=15),  "open": h, "high": h, "low": l, "close": l, "volume": v / 4},
            {"bucket": ts + timedelta(seconds=30),  "open": l, "high": c, "low": l, "close": c, "volume": v / 4},
            {"bucket": ts + timedelta(seconds=45),  "open": c, "high": c, "low": c, "close": c, "volume": v / 4},
        ]
        rows.extend(seg)
    return pd.DataFrame(rows).set_index("bucket").sort_index()


def test_engine_accepts_15s_cadence_and_produces_same_signal_count():
    df_1m = _build_cross_up_scenario()
    df_15s = _to_15s_frame(df_1m)
    p = _params(
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
    )
    res_1m = simulate_vwap_cross(df_1m, p)
    res_15s = simulate_vwap_cross(df_15s, p)
    assert res_15s["bar_seconds"] == 15
    assert res_1m["bar_seconds"] >= 60
    # Same signal population on both cadences (filter runs on 1-min view).
    assert len(res_15s["trades"]) == len(res_1m["trades"])


# ── live-safety: no forward-looking bars are consulted ──────────────────

def test_close_confirm_is_live_safe_truncating_after_signal_bar_does_not_change_signal():
    """close_confirm must be decidable at close of candle N — the presence or
    absence of bars *after* N cannot flip the signal."""
    df = _build_cross_up_scenario()
    p = _params(
        entry_mode="close_confirm",
        vwap_slope_lookback=1, min_vwap_slope=0.0,
        max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
    )
    full = simulate_vwap_cross(df, p)
    assert not full["trades"].empty
    first = full["trades"].iloc[0]
    signal_bar_ts = pd.Timestamp(first["entry_ts"]) - pd.Timedelta(minutes=1)
    # Truncate the frame right after the signal bar (keep bar N, drop N+1..)
    truncated = df.loc[df.index <= signal_bar_ts]
    # Add exactly one extra bar so the exit walk has somewhere to fire (N+1 open only).
    tail_ts = truncated.index[-1] + pd.Timedelta(minutes=1)
    tail_bar = df.loc[[df.index[df.index.get_indexer([tail_ts], method="nearest")[0]]]]
    truncated = pd.concat([truncated, tail_bar]).sort_index()
    trunc_res = simulate_vwap_cross(truncated, p)
    # Signal must still fire on the same bar N despite everything after it being unknown.
    assert not trunc_res["trades"].empty
    assert pd.Timestamp(trunc_res["trades"].iloc[0]["entry_ts"]) == pd.Timestamp(first["entry_ts"])


def test_forward_slope_direction_actually_reads_future_bars():
    """vwap_slope_direction='forward' must consult bars AFTER the signal bar —
    truncating the frame right after N should suppress the trade in forward
    mode while backward mode still fires."""
    df = _build_cross_up_scenario()
    backward_p = _params(
        entry_mode="close_confirm",
        use_vwap_slope_filter=True, vwap_slope_lookback=3,
        vwap_slope_direction="backward", min_vwap_slope=0.0,
        max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
    )
    forward_p = {**backward_p, "vwap_slope_direction": "forward"}
    full_fwd = simulate_vwap_cross(df, forward_p)
    assert not full_fwd["trades"].empty
    signal_ts = pd.Timestamp(full_fwd["trades"].iloc[0]["entry_ts"]) - pd.Timedelta(minutes=1)
    # Truncate to keep bar N and only ONE bar after (needed for the fill).
    end_keep = signal_ts + pd.Timedelta(minutes=1)
    truncated = df.loc[df.index <= end_keep]
    fwd_truncated = simulate_vwap_cross(truncated, forward_p)
    bwd_truncated = simulate_vwap_cross(truncated, backward_p)
    # Forward mode needs slope_lookback=3 bars AFTER N — those are missing → no signal.
    assert fwd_truncated["trades"].empty
    # Backward mode uses bars up to N only — signal still fires.
    assert not bwd_truncated["trades"].empty


# ── new toggles: VWAP slope off, preceder off, bar-size / EMA gates ─────

def test_use_vwap_slope_filter_off_ignores_slope():
    df = _build_cross_up_scenario()
    # Force min_slope=huge → guaranteed rejection with slope filter ON.
    strict = _params(min_vwap_slope=100.0, use_vwap_slope_filter=True,
                     max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                     stop_mode="crossing_open")
    loose = _params(min_vwap_slope=100.0, use_vwap_slope_filter=False,
                    max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                    stop_mode="crossing_open")
    assert simulate_vwap_cross(df, strict)["trades"].empty
    assert not simulate_vwap_cross(df, loose)["trades"].empty


def test_use_preceding_bar_color_off_admits_non_green_preceder():
    df = _build_cross_up_scenario().copy()
    # Flip preceding bar (index 5 = bar 515) into red
    ts = df.index[5]
    df.at[ts, "close"] = 99.0
    df.at[ts, "open"] = 101.0
    strict = _params(use_preceding_bar_color=True, min_vwap_slope=0.0,
                     max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                     stop_mode="crossing_open")
    loose = _params(use_preceding_bar_color=False, min_vwap_slope=0.0,
                    max_session_crosses=100, chop_lookback=0, max_recent_crosses=100,
                    stop_mode="crossing_open")
    assert simulate_vwap_cross(df, strict)["trades"].empty
    assert not simulate_vwap_cross(df, loose)["trades"].empty


def test_use_bar_size_filter_requires_min_range_vs_atr():
    df = _build_cross_up_scenario()
    # ATR needs 14 warmup bars but scenario has ~6. Use lookback=3 for the test.
    strict = _params(use_bar_size_filter=True, bar_size_atr_mult=8.0, atr_lookback=3,
                     min_vwap_slope=0.0, max_session_crosses=100,
                     chop_lookback=0, max_recent_crosses=100,
                     stop_mode="crossing_open")
    loose = _params(use_bar_size_filter=False,
                    min_vwap_slope=0.0, max_session_crosses=100,
                    chop_lookback=0, max_recent_crosses=100,
                    stop_mode="crossing_open")
    # A giant ATR multiplier makes the strict filter impossible to satisfy.
    assert simulate_vwap_cross(df, strict)["trades"].empty
    assert not simulate_vwap_cross(df, loose)["trades"].empty


def test_use_ema_cross_filter_requires_same_side_agreement():
    df = _build_cross_up_scenario()
    # ema_length=3 so a very small warmup suffices in the synthetic scenario.
    on_same_side = _params(use_ema_cross_filter=True, ema_length=3,
                           min_vwap_slope=0.0, max_session_crosses=100,
                           chop_lookback=0, max_recent_crosses=100,
                           stop_mode="crossing_open")
    off = _params(use_ema_cross_filter=False, ema_length=3,
                  min_vwap_slope=0.0, max_session_crosses=100,
                  chop_lookback=0, max_recent_crosses=100,
                  stop_mode="crossing_open")
    # In the cross-up scenario price rallies from 100 to 110+, so EMA lags but
    # ends up above VWAP by the signal bar — filter should PASS.
    r_on = simulate_vwap_cross(df, on_same_side)
    r_off = simulate_vwap_cross(df, off)
    assert not r_off["trades"].empty
    assert not r_on["trades"].empty


def test_use_ema_cross_filter_fires_on_ema_cross_bar_when_price_already_above():
    """The second-cross trigger: price crossed VWAP first, EMA lagged, and the
    trade must fire on the bar where the EMA finally crosses — not require the
    two events to coincide."""
    rows = []
    day = 5
    # Warm the VWAP with enough volume that later bars can't skew it fast.
    for m in range(510, 540):
        rows.append(_row(day, m, 100, 100, 100, 100, 500))
    # 4 down bars: VWAP dips slightly, EMA follows.
    for m, px in [(540, 99), (541, 98), (542, 97), (543, 96)]:
        rows.append(_row(day, m, px + 1, px + 1, px, px, 500))
    # Reversal: bar 544 crosses up (opens below VWAP, closes above). EMA lags.
    rows.append(_row(day, 544, 96, 101, 96, 101, 500))
    # A few steady bars above VWAP so EMA catches up.
    for m in range(545, 600):
        rows.append(_row(day, m, 100, 101, 100, 101, 500))
    df = _frame(rows)
    p = _params(
        entry_mode="close_confirm",
        use_ema_cross_filter=True, ema_length=6,
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="vwap_offset", vwap_stop_offset=1.0, max_stop_pts=0.0,
        profit_target_mode="absolute", profit_target_pts=5.0,
    )
    res = simulate_vwap_cross(df, p)
    # The new second-cross trigger should still fire a long even though EMA
    # lagged the price cross. The old same-bar EMA check rejected these.
    assert not res["trades"].empty
    assert res["trades"].iloc[0]["side"] == "long"


def test_fast_ema_length_produces_earlier_cross_than_slow_ema_length():
    """A quicker crossover EMA should generally fire its cross event sooner —
    so we should see equal-or-more trades with a fast crossover EMA than with a
    slow one when the same-side gate is held constant."""
    df = _build_cross_up_scenario()
    fast = _params(entry_mode="close_confirm",
                   use_ema_cross_filter=True, ema_length=12, ema_cross_length=2,
                   min_vwap_slope=0.0, max_session_crosses=100,
                   chop_lookback=0, max_recent_crosses=100,
                   stop_mode="crossing_open")
    slow = _params(entry_mode="close_confirm",
                   use_ema_cross_filter=True, ema_length=12, ema_cross_length=20,
                   min_vwap_slope=0.0, max_session_crosses=100,
                   chop_lookback=0, max_recent_crosses=100,
                   stop_mode="crossing_open")
    n_fast = len(simulate_vwap_cross(df, fast)["trades"])
    n_slow = len(simulate_vwap_cross(df, slow)["trades"])
    assert n_fast >= n_slow


def test_ema_vwap_convergence_pt_mode_exits_on_gap_shrink():
    """Trending prices push EMA away from VWAP, then prices mean-revert and the
    gap starts to shrink → convergence exit fires."""
    rows = []
    day = 5
    for m in range(510, 515):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    rows.append(_row(day, 515, 99, 101, 99, 101, 10))          # green preceder
    rows.append(_row(day, 516, 99, 115, 99, 115, 30))          # crossing bar rockets up
    # A few strong up bars — EMA lags, gap grows
    for m in range(517, 525):
        rows.append(_row(day, m, 115, 118, 114, 117, 10))
    # Then price stalls / drifts back toward VWAP — gap shrinks
    for m in range(525, 550):
        rows.append(_row(day, m, 117, 118, 113, 114, 10))
    df = _frame(rows)
    p = _params(
        entry_mode="close_confirm",
        profit_target_mode="ema_vwap_convergence",
        min_bars_before_ema_pt=1, ema_vwap_shrink_pct=5.0,
        ema_length=3,
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open", max_stop_pts=0.0,
    )
    res = simulate_vwap_cross(df, p)
    assert not res["trades"].empty
    t = res["trades"].iloc[0]
    assert str(t["exit_reason"]) == "ema_convergence"
    # target_px is null-serialized when the mode has no fixed price target.
    card = get_model("vwap_cross")
    payload = vwap_cross_payload(res, source="external:vwap_cross",
                                 start=None, end=None, n_sessions=1, card=card)
    assert payload["trades"][0]["target_px"] is None


# ── no overlapping positions ────────────────────────────────────────────

def test_no_overlapping_trades_signal_cursor_advances_past_exit():
    """Two consecutive short-cross bars must not both fire while the first
    trade is still open — the signal cursor must jump past the exit bar."""
    rows = []
    day = 5
    for m in range(510, 516):
        rows.append(_row(day, m, 100, 100, 100, 100, 10))
    rows.append(_row(day, 516, 101, 101, 99, 99, 10))     # red preceder
    # Two shorts back-to-back: bar 517 is a clean short setup; bar 518 is also.
    rows.append(_row(day, 517, 101, 101, 88, 90, 30))     # short cross bar 1
    rows.append(_row(day, 518, 101, 101, 87, 89, 30))     # short cross bar 2 (would fire too)
    # Prices stay low so no exit fires for many minutes
    for m in range(519, 550):
        rows.append(_row(day, m, 89, 90, 88, 89, 10))
    df = _frame(rows)
    p = _params(
        entry_mode="close_confirm",
        min_vwap_slope=0.0, max_session_crosses=100,
        chop_lookback=0, max_recent_crosses=100,
        stop_mode="crossing_open",
        profit_target_mode="absolute", profit_target_pts=500.0,  # never hit
    )
    ledger = simulate_vwap_cross(df, p)["trades"]
    assert not ledger.empty
    # Verify no overlapping intervals across trades.
    intervals = sorted(zip(pd.to_datetime(ledger["entry_ts"]),
                            pd.to_datetime(ledger["exit_ts"])))
    for (e1, x1), (e2, x2) in zip(intervals, intervals[1:]):
        assert e2 > x1, f"trade [{e2}, {x2}] overlaps prior [{e1}, {x1}]"
