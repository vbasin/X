"""Build the variable catalog exposed by a specific chart configuration.

The catalog is a flat mapping ``{"name": {"kind": ..., "help": ...}}`` used
for autocomplete on the frontend and for compile-time validation on the
backend. A name in the catalog is guaranteed to have a value in the
per-bar bindings produced by :mod:`.interpreter` — an unknown name is
rejected at compile time so rules cannot silently reference variables
that the chart is not rendering.
"""

from __future__ import annotations

from typing import Any


CANDLE_CORE = [
    ("C.o", "Open of the current candle"),
    ("C.h", "High of the current candle"),
    ("C.l", "Low of the current candle"),
    ("C.c", "Close of the current candle"),
    ("C.color", "'green' if close > open, 'red' if close < open, 'doji' otherwise"),
    ("C.body_hi", "max(open, close)"),
    ("C.body_lo", "min(open, close)"),
    ("C.range", "high − low"),
    ("C.body", "|close − open|"),
    ("Co", "Alias for C.o"),
    ("Ch", "Alias for C.h"),
    ("Cl", "Alias for C.l"),
    ("Cc", "Alias for C.c"),
    ("Po", "Prior bar open"),
    ("Ph", "Prior bar high"),
    ("Pl", "Prior bar low"),
    ("Pc", "Prior bar close"),
]

CONSTANTS = [
    ("tick", "NQ tick size (0.25)"),
    ("time", "'HH:MM' in America/Chicago (compare to string literals)"),
    ("bars_from_open", "Bars since the current cash-session open"),
    ("entry", "Only inside set_target/set_stop expressions — the entry fill price"),
]

VP_FIELDS = [
    ("high",     "Absolute high of the profile window"),
    ("low",      "Absolute low of the profile window"),
    ("poc",      "Point of control"),
    ("vah",      "Value area high"),
    ("val",      "Value area low"),
    ("hi_body",  "max(open, close) of the candle setting the profile high"),
    ("lo_body",  "min(open, close) of the candle setting the profile low"),
]

OVERLAY_LABELS: dict[str, tuple[str, str]] = {
    "ps_sar":              ("SAR",         "Parabolic SAR value at bar close"),
    "vw_vwap":             ("VWAP",        "Session VWAP"),
    "vw_vwap_dist":        ("VWAP_DIST",   "Distance to VWAP"),
    "of_delta":            ("DELTA",       "Signed volume for the bar"),
    "of_cvd":              ("CVD",         "Cumulative volume delta"),
    "of_ofi":              ("OFI",         "Order-flow imbalance for the bar"),
    "of_ofi_cum":          ("OFI_CUM",     "Cumulative order-flow imbalance"),
    "cs_clv":              ("CLV",         "Close location value"),
    "mf_mfi":              ("MFI",         "Money flow index (14)"),
    "ad_line":             ("AD",          "Accumulation/distribution line"),
    "so_fast_d_9_3":       ("SO_FAST_D",   "Stochastic fast %D (9,3)"),
    "so_sync_score_all":   ("SO_SYNC",     "Stochastic sync score (all)"),
}


def build_var_catalog(payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Enumerate the variables the DSL may reference for this chart payload.

    *payload* is the ``dict`` returned by :func:`src.webui.live_chart.series`.
    When ``None`` (autocomplete during editing before a first Run), a
    "canonical" catalog with all core candle fields and the first two VPs is
    still returned so the editor is useful.
    """
    catalog: dict[str, dict[str, Any]] = {}
    for name, help_ in CANDLE_CORE:
        catalog[name] = {"kind": "scalar", "group": "candle", "help": help_}
    for name, help_ in CONSTANTS:
        catalog[name] = {"kind": "scalar", "group": "constant", "help": help_}

    profiles = (payload or {}).get("profiles") or []
    n_seen: set[int] = set()
    if not profiles:
        # Default to VP1/VP2 shells so the editor is not empty.
        for n in (1, 2):
            for field, help_ in VP_FIELDS:
                catalog[f"VP{n}.{field}"] = {
                    "kind": "scalar", "group": f"VP{n}",
                    "help": f"{help_} (VP{n} — not yet configured on the chart)",
                }
        n_seen.update({1, 2})
    else:
        for p in profiles:
            try:
                n = int(p.get("n") or 1)
            except (TypeError, ValueError):
                n = 1
            if n in n_seen:
                continue
            n_seen.add(n)
            label = p.get("label") or f"VP{n}"
            for field, help_ in VP_FIELDS:
                catalog[f"VP{n}.{field}"] = {
                    "kind": "scalar", "group": label, "help": help_,
                }

    overlays = (payload or {}).get("overlays") or {}
    for key, (alias, help_) in OVERLAY_LABELS.items():
        if key in overlays:
            catalog[alias] = {"kind": "scalar", "group": "overlay", "help": help_}

    # Large-prints functions are ALWAYS callable; they just return 0 when
    # no prints series is loaded.
    for name, help_ in [
        ("lp_net",       "lp_net(secs) — net signed size (buys − sells) in the trailing <secs> before this bar"),
        ("lp_buy",       "lp_buy(secs) — total buy-side size in the trailing <secs>"),
        ("lp_sell",      "lp_sell(secs) — total sell-side size in the trailing <secs>"),
        ("lp_count",     "lp_count(secs) — number of large prints in the trailing <secs>"),
        ("lp_max_size",  "lp_max_size(secs) — largest single print size in the trailing <secs>"),
    ]:
        catalog[name + "()"] = {"kind": "function", "group": "large_prints", "help": help_}

    for name, help_ in [
        ("atr",  "atr(n) — Wilder ATR of length n on the current timeframe"),
        ("ema",  "ema(n) — TradingView EMA(n) on close"),
        ("sma",  "sma(n) — simple moving average of close, length n"),
        ("min",  "min(a, b, ...) — smallest value"),
        ("max",  "max(a, b, ...) — largest value"),
        ("abs",  "abs(x) — absolute value"),
        ("next_up_of",   "next_up_of(ref, a, b, ...) — smallest candidate strictly > ref"),
        ("next_down_of", "next_down_of(ref, a, b, ...) — largest candidate strictly < ref"),
    ]:
        catalog[name + "()"] = {"kind": "function", "group": "math", "help": help_}

    return catalog
