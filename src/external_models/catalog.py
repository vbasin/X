"""Factory catalog of external models (repo YAML + in-code param spec).

Per-model knobs live on each card (SAR, 20 EMA, ZigZag High/Low, Timed
PSAR). Session / trade window / ATR gate / PT-SL / costs / sides live on
the shared **book** spec (ADR-062 / ADR-064 / ADR-096) — Backtest
Parameters on the Backtest tab when type is EXT Model — and apply to a
stacked backtest, not to a single card's Parameters form. Each stacked
engine has its own candle timeframe.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from src.external_models.timeframe import CANDLE_TIMEFRAMES, coerce_candle_tf

_REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_DIR = _REPO_ROOT / "config" / "external_models"
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
HARDENED_NAME_RE = MODEL_ID_RE
LEGACY_COMBINED_ID = "psar_ema20"
COMPAT_STACK = ("psar", "ema20")

# ── Shared book (Backtest Parameters, applied to a stack) ─────────────────

BOOK_PARAM_GROUPS: list[tuple[str, str]] = [
    ("session", "Session"),
    ("entries", "Entries"),
    ("exits", "Exits"),
    ("size", "Size & costs"),
]

_UNIT_CHOICES = ["points", "atr"]
_PT_UNIT_CHOICES = ["points", "atr", "stop"]

BOOK_PARAM_SPEC: list[dict[str, Any]] = [
    {
        "key": "session_start", "label": "Session start (CT)", "type": "time",
        "group": "session", "pine": "session 08:30",
        "help": "Trading session begins (America/Chicago). Bars before this are not used. Session end is the last possible exit.",
    },
    {
        "key": "session_end", "label": "Session end (CT)", "type": "time",
        "group": "session", "pine": "flatten 14:45",
        "help": "Last possible exit. Open positions flatten at this America/Chicago clock time.",
    },
    {
        "key": "trade_start", "label": "Trade start (CT)", "type": "time",
        "group": "session",
        "help": "Earliest new-entry time. Must fall inside the session. Open trades may continue until session end.",
    },
    {
        "key": "trade_end", "label": "Trade end (CT)", "type": "time",
        "group": "session",
        "help": "Last new-entry time. After this, no new entries; open trades still manage PT/SL/signal exits until session end.",
    },
    {
        "key": "timeframe", "label": "Candle timeframe", "type": "enum",
        "choices": list(CANDLE_TIMEFRAMES), "group": "session", "pine": "chart TF",
        "help": "Display candles, ATR entry gate, and PT/SL bars (resampled from databento.tbbo_1min). 1hour is clock hours; 1day is the Chicago calendar day labeled 08:30 CT. Each stacked engine has its own candle timeframe.",
    },
    {
        "key": "trade_long", "label": "Longs", "type": "bool",
        "group": "entries",
        "help": "Checked: long trades are allowed. Unchecked: no longs. Default on. Leave Longs and Shorts both checked to trade both sides.",
    },
    {
        "key": "trade_short", "label": "Shorts", "type": "bool",
        "group": "entries",
        "help": "Checked: short trades are allowed. Unchecked: no shorts. Default on. Leave Longs and Shorts both checked to trade both sides.",
    },
    {
        "key": "signal_lock", "label": "Signal entry lock", "type": "bool",
        "group": "entries",
        "help": "After a profit-target or stop-loss exit, do not re-enter on the same direction. Wait for the next opposite-direction consensus signal before a new entry.",
    },
    {
        "key": "atr_period", "label": "ATR bars (wilders)", "type": "int",
        "min": 1, "max": 50, "step": 1, "group": "entries",
        "help": "Wilder ATR length (bars) on the Backtest candle timeframe. Used for the min-ATR entry gate and for PT/SL when those are set to × ATR. Not used in the SAR trail math.",
    },
    {
        "key": "min_atr", "label": "Min ATR entry", "type": "qty_unit",
        "unit_key": "min_atr_unit", "min": 0.0, "max": 200.0, "step": 0.1,
        "group": "entries", "pine": "minATR",
        "help": "Points: skip new entries unless ATR(period) > this many NQ points. × ATR: skip unless this bar's true range is at least this many ATRs (0 = off).",
    },
    {
        "key": "profit_target_pts", "label": "Profit target", "type": "qty_unit",
        "unit_key": "profit_target_unit", "min": 0.0, "max": 2000.0, "step": 0.25,
        "group": "exits", "pine": "profitTargetPts",
        "unit_choices": list(_PT_UNIT_CHOICES),
        "help": "Points: limit from average entry in NQ points. × ATR: that many × ATR(period) at entry (fixed for the trade). × stop: that many × the stop-loss distance at entry. 0 = off.",
    },
    {
        "key": "stop_loss_pts", "label": "Stop loss", "type": "qty_unit",
        "unit_key": "stop_loss_unit", "min": 0.0, "max": 2000.0, "step": 0.25,
        "group": "exits", "pine": "stopLossPts",
        "help": "Points: stop from average entry in NQ points. × ATR: that many × ATR(period) at entry (fixed for the trade). 0 = off.",
    },
    {
        "key": "qty", "label": "Contracts", "type": "int",
        "min": 1, "max": 20, "step": 1, "group": "size", "pine": "default_qty_value",
        "help": "Fixed contract quantity (Pine default_qty_type=strategy.fixed).",
    },
    {
        "key": "cost_per_trade", "label": "Round-trip cost (pts)", "type": "float",
        "min": 0.0, "max": 20.0, "step": 0.25, "group": "size",
        "help": "Commission + fees in NQ points per round trip. Pine script is 0.",
    },
    {
        "key": "slippage_points", "label": "Slippage per side (pts)", "type": "float",
        "min": 0.0, "max": 10.0, "step": 0.25, "group": "size",
        "help": "Added against the trade on entry and on exit.",
    },
]

BOOK_FACTORY_PARAMS: dict[str, Any] = {
    "trade_long": True,
    "trade_short": True,
    "signal_lock": False,
    "atr_period": 8,
    "min_atr": 15.0,
    "min_atr_unit": "points",
    "profit_target_pts": 50.0,
    "profit_target_unit": "points",
    "stop_loss_pts": 10.0,
    "stop_loss_unit": "points",
    "session_start": "08:30",
    "session_end": "14:45",
    "trade_start": "08:30",
    "trade_end": "14:45",
    "timezone": "America/Chicago",
    "qty": 1,
    "cost_per_trade": 0.0,
    "slippage_points": 0.0,
    "timeframe": "1min",
}

PSAR_PARAM_SPEC: list[dict[str, Any]] = [
    {
        "key": "start", "label": "SAR start (AF)", "type": "float",
        "min": 0.001, "max": 1.0, "step": 0.01, "pine": "start",
        "help": "Initial acceleration factor.",
    },
    {
        "key": "increment", "label": "SAR increment", "type": "float",
        "min": 0.001, "max": 1.0, "step": 0.01, "pine": "increment",
        "help": "AF step on a new extreme.",
    },
    {
        "key": "maximum", "label": "SAR maximum AF", "type": "float",
        "min": 0.01, "max": 1.0, "step": 0.01, "pine": "maximum",
        "help": "AF cap (Wilder default 0.2).",
    },
    {
        "key": "exit_at_sar", "label": "Exit at SAR on intersection", "type": "bool",
        "help": "On: when price tags the in-effect SAR trail, fill the exit at that SAR (gap-through at the open) instead of waiting for the intersection bar's close. Off (Pine default): market-close on that bar. The fill is the trail in force at the start of the bar, not the post-flip SAR dot.",
    },
]

PSAR_FACTORY_PARAMS: dict[str, Any] = {
    "start": 0.02,
    "increment": 0.02,
    "maximum": 0.2,
    "exit_at_sar": False,
}

EMA20_PARAM_SPEC: list[dict[str, Any]] = [
    {
        "key": "ema_length", "label": "EMA length", "type": "int",
        "min": 2, "max": 200, "step": 1, "pine": "ta.ema(close, 20)",
        "help": "EMA period on the candle close.",
    },
    {
        "key": "ema20_l_bars", "label": "EMA slope look-back (L)", "type": "int",
        "min": 0, "max": 200, "step": 1, "pine": "ema20Lbars",
        "help": "Established slope: EMA vs SMA of the last L EMA values. 0 = ignore this filter.",
    },
    {
        "key": "ema20_s_bars", "label": "EMA slope look-back (S)", "type": "int",
        "min": 0, "max": 50, "step": 1, "pine": "ema20Sbars",
        "help": "Steep slope: last S EMA steps strictly monotonic. 0 = ignore this filter.",
    },
]

EMA20_FACTORY_PARAMS: dict[str, Any] = {
    "ema_length": 20,
    "ema20_l_bars": 8,
    "ema20_s_bars": 2,
}

ZZHL_PARAM_SPEC: list[dict[str, Any]] = [
    {
        "key": "percentage_reversal", "label": "Percentage reversal (%)", "type": "float",
        "min": 0.0, "max": 50.0, "step": 0.05,
        "help": "TOS percentageReversal. 2.5 means 2.5% of the swing extreme. On 1-minute NQ that is hundreds of points — set 0 to reverse on ATR and/or absolute points only.",
    },
    {
        "key": "absolute_reversal", "label": "Absolute reversal (pts)", "type": "float",
        "min": 0.0, "max": 2000.0, "step": 0.25,
        "help": "TOS absoluteReversal in NQ points, added on top of the percentage/ATR distance. Non-zero overrides tick reversal. 0 = off.",
    },
    {
        "key": "atr_length", "label": "ATR length (Wilder)", "type": "int",
        "min": 1, "max": 100, "step": 1,
        "help": "TOS atrLength. Wilder average of true range used only when ATR reversal > 0. Independent of Backtest Parameters ATR bars.",
    },
    {
        "key": "atr_reversal", "label": "ATR reversal (×)", "type": "float",
        "min": 0.0, "max": 20.0, "step": 0.1,
        "help": "TOS atrReversal. When > 0, hlPivot = %/100 + ATR(length)/close × this. 0 = ignore ATR in the reversal distance.",
    },
    {
        "key": "tick_reversal", "label": "Tick reversal", "type": "int",
        "min": 0, "max": 10000, "step": 1,
        "help": "TOS tickReversal. Used only when absolute reversal is 0: absReversal = ticks × tick size. 0 = off.",
    },
    {
        "key": "tick_size", "label": "Tick size", "type": "float",
        "min": 0.01, "max": 10.0, "step": 0.01,
        "help": "Instrument tick (TOS TickSize). NQ is 0.25. Multiplies tick reversal when absolute reversal is 0.",
    },
]

ZZHL_FACTORY_PARAMS: dict[str, Any] = {
    "percentage_reversal": 2.5,
    "absolute_reversal": 0.0,
    "atr_length": 5,
    "atr_reversal": 1.0,
    "tick_reversal": 0,
    "tick_size": 0.25,
}

TIMED_PSAR_PARAM_SPEC: list[dict[str, Any]] = [
    {
        "key": "entry_mode", "label": "Entry timing", "type": "enum",
        "choices": ["clock", "bars_from_open"],
        "help": "clock: first in-session book bar at or after Entry time (CT). bars_from_open: N bars after the first in-session bar (0 = the open bar).",
    },
    {
        "key": "entry_time", "label": "Entry time (CT)", "type": "time",
        "help": "Used when Entry timing is clock. America/Chicago. Must fall inside the session; the book's trade window still gates the fill.",
    },
    {
        "key": "bars_from_open", "label": "Bars from session open", "type": "int",
        "min": 0, "max": 2000, "step": 1,
        "help": "Used when Entry timing is bars_from_open. 0 = first in-session bar on the book timeframe. Session open is Session start, not Globex 17:00.",
    },
    {
        "key": "direction_tf", "label": "PSAR direction timeframe", "type": "enum",
        "choices": ["1hour", "1day"],
        "help": "Last closed 1-hour or 1-day Wilder SAR sets long vs short at the trigger. 1hour at 08:30 CT is the 07:00–08:00 hour. 1day is yesterday's Chicago calendar day (visible from 08:29 CT).",
    },
    {
        "key": "start", "label": "SAR start (AF)", "type": "float",
        "min": 0.001, "max": 1.0, "step": 0.01,
        "help": "Initial acceleration factor on the direction timeframe.",
    },
    {
        "key": "increment", "label": "SAR increment", "type": "float",
        "min": 0.001, "max": 1.0, "step": 0.01,
        "help": "AF step on a new extreme.",
    },
    {
        "key": "maximum", "label": "SAR maximum AF", "type": "float",
        "min": 0.01, "max": 1.0, "step": 0.01,
        "help": "AF cap (Wilder default 0.2).",
    },
]

TIMED_PSAR_FACTORY_PARAMS: dict[str, Any] = {
    "entry_mode": "clock",
    "entry_time": "08:30",
    "bars_from_open": 0,
    "direction_tf": "1day",
    "start": 0.02,
    "increment": 0.02,
    "maximum": 0.2,
}

# ── Opening Range Breakout (ADR-118) ─────────────────────────────────────
# Self-contained engine: it owns session/risk/exit knobs rather than the
# shared book, because sizing and the drawdown floor are account-level state
# that ``simulate_book`` does not model.

ORB_PARAM_GROUPS: list[tuple[str, str]] = [
    ("range", "Opening range"),
    ("filters", "Quality filters"),
    ("entries", "Setups"),
    ("exits", "Exits"),
    ("risk", "Risk & account"),
]

ORB_PARAM_SPEC: list[dict[str, Any]] = [
    {"key": "or_minutes", "label": "Opening range (min)", "type": "int",
     "min": 1, "max": 120, "step": 1, "group": "range",
     "help": "Length of the opening range measured from Session start. 15 = 08:30–08:45 CT."},
    {"key": "session_start", "label": "Session start (CT)", "type": "time", "group": "range",
     "help": "RTH open. The opening range begins here."},
    {"key": "trade_end", "label": "Last new entry (CT)", "type": "time", "group": "range",
     "help": "No new entries after this clock time; open trades still manage exits."},
    {"key": "session_end", "label": "Flatten (CT)", "type": "time", "group": "range",
     "help": "Hard flatten. Everything is closed at this bar's close."},
    {"key": "va_pct", "label": "Value area %", "type": "float",
     "min": 0.5, "max": 0.95, "step": 0.05, "group": "range",
     "help": "Prior-day value area width used for the VAH/VAL reference levels."},

    {"key": "min_rvol", "label": "Min opening rel. volume", "type": "float",
     "min": 0.0, "max": 10.0, "step": 0.05, "group": "filters",
     "help": "Opening-range volume ÷ its own trailing average. Zarattini/Barbon/Aziz (2024) found ORB expectancy was negative below 100% and rose steadily above it — this is the single highest-value filter in the model."},
    {"key": "rvol_lookback", "label": "Rel. volume lookback (days)", "type": "int",
     "min": 3, "max": 60, "step": 1, "group": "filters",
     "help": "Sessions used for the average opening-range volume baseline."},
    {"key": "min_or_atr", "label": "Min OR ÷ ATR", "type": "float",
     "min": 0.0, "max": 3.0, "step": 0.05, "group": "filters",
     "help": "Skip sessions whose opening range is too narrow to carry a move — these produce whipsaw breakouts."},
    {"key": "max_or_atr", "label": "Max OR ÷ ATR", "type": "float",
     "min": 0.1, "max": 5.0, "step": 0.05, "group": "filters",
     "help": "Skip sessions whose opening range already spans most of a normal day's range — the move has largely happened."},
    {"key": "atr_days", "label": "ATR days", "type": "int",
     "min": 3, "max": 60, "step": 1, "group": "filters",
     "help": "Daily ATR lookback (prior sessions only) used by the range-quality gates."},

    {"key": "setups", "label": "Enabled setups", "type": "text", "group": "entries",
     "help": "Comma list of breakout_retest, failed_break, fade. breakout_retest = continuation entered on the pullback to the broken extreme (stop sits just under it); failed_break = poke through then close back inside, fading trapped entrants; fade = inside-range rotation on balance days."},
    {"key": "entry_start", "label": "First entry (CT)", "type": "time", "group": "entries",
     "help": "Earliest new entry. Clamped to the end of the opening range."},
    {"key": "break_buffer_ticks", "label": "Break buffer (ticks)", "type": "float",
     "min": 0.0, "max": 40.0, "step": 1.0, "group": "entries",
     "help": "Distance beyond the OR extreme required to call it a break, and the pad placed beyond structure when siting a stop."},
    {"key": "min_poke_atr", "label": "Min failed-break poke (× ATR)", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.005, "group": "entries",
     "help": "How far price must have exceeded the OR extreme before a close back inside counts as a failed break. Filters meaningless one-tick pokes."},
    {"key": "rejection_clv", "label": "Rejection close position", "type": "float",
     "min": 0.5, "max": 1.0, "step": 0.05, "group": "entries",
     "help": "Where in its own range the signal bar must close (0.7 = upper 30% for a long). Enforces that the bar actually rejected, rather than drifting."},
    {"key": "retest_band_frac", "label": "Retest band (× OR range)", "type": "float",
     "min": 0.01, "max": 0.5, "step": 0.01, "group": "entries",
     "help": "How close a pullback must come to the broken extreme to arm the retest entry."},
    {"key": "fade_band_frac", "label": "Fade band (× OR range)", "type": "float",
     "min": 0.02, "max": 0.5, "step": 0.01, "group": "entries",
     "help": "How close to an OR extreme a wick must reach to arm the inside-range fade."},
    {"key": "fade_max_or_atr", "label": "Fade max OR ÷ ATR", "type": "float",
     "min": 0.1, "max": 3.0, "step": 0.05, "group": "entries",
     "help": "Only fade inside the range on genuinely balanced sessions — wide ranges tend to trend."},
    {"key": "max_trades_per_day", "label": "Max trades / day", "type": "int",
     "min": 1, "max": 10, "step": 1, "group": "entries",
     "help": "Hard cap on entries per session. Low values are the main defence against revenge-trading drawdown."},

    {"key": "t1_r", "label": "Target 1 (R)", "type": "float",
     "min": 0.25, "max": 10.0, "step": 0.25, "group": "exits",
     "help": "First target as a multiple of the stop distance. Part of the position is banked here."},
    {"key": "t2_r", "label": "Target 2 (R)", "type": "float",
     "min": 0.5, "max": 30.0, "step": 0.25, "group": "exits",
     "help": "Runner target as a multiple of the stop distance."},
    {"key": "scale_out_pct", "label": "Scale out at T1", "type": "float",
     "min": 0.0, "max": 0.9, "step": 0.05, "group": "exits",
     "help": "Fraction of contracts closed at Target 1. Raising this lifts the hit rate and cuts drawdown at the cost of tail upside — the correct trade when a hard loss barrier binds."},
    {"key": "breakeven_after_t1", "label": "Breakeven after T1", "type": "bool", "group": "exits",
     "help": "Move the runner's stop to entry once Target 1 fills."},
    {"key": "be_offset_pts", "label": "Breakeven offset (pts)", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.25, "group": "exits",
     "help": "Push the breakeven stop this far into profit."},
    {"key": "trail_pts", "label": "Runner trail (pts)", "type": "float",
     "min": 0.0, "max": 200.0, "step": 1.0, "group": "exits",
     "help": "Trail the runner this far behind the best price after T1. 0 = no trail."},
    {"key": "max_stop_points", "label": "Max stop (pts)", "type": "float",
     "min": 1.0, "max": 300.0, "step": 1.0, "group": "exits",
     "help": "Structural stops wider than this are truncated (breakout) or the trade is skipped. Caps the worst-case loss so the drawdown floor holds."},

    {"key": "initial_balance", "label": "Starting balance ($)", "type": "float",
     "min": 500.0, "max": 1000000.0, "step": 500.0, "group": "risk",
     "help": "Account equity at the start of the run."},
    {"key": "point_value", "label": "$ per point / contract", "type": "float",
     "min": 0.5, "max": 500.0, "step": 0.5, "group": "risk",
     "help": "20 = /NQ full-size, 2 = /MNQ micro. On a small account with a hard drawdown floor this is the single most important number: /NQ's median 15-minute opening range is ~110 points, so one structurally-valid stop can exceed a $1,000 loss barrier outright."},
    {"key": "max_drawdown", "label": "Max drawdown ($)", "type": "float",
     "min": 100.0, "max": 500000.0, "step": 100.0, "group": "risk",
     "help": "Hard loss barrier measured from the starting balance. Trading halts permanently if equity touches it."},
    {"key": "risk_usd_per_trade", "label": "Risk / trade ($)", "type": "float",
     "min": 10.0, "max": 100000.0, "step": 10.0, "group": "risk",
     "help": "Dollar risk budget at the stop. Contracts are sized from this and the stop distance."},
    {"key": "max_contracts", "label": "Max contracts", "type": "int",
     "min": 1, "max": 50, "step": 1, "group": "risk",
     "help": "Position-size ceiling."},
    {"key": "margin_per_contract", "label": "Margin / contract ($)", "type": "float",
     "min": 0.0, "max": 100000.0, "step": 100.0, "group": "risk",
     "help": "Intraday margin requirement; caps size on a small balance."},
    {"key": "daily_loss_limit", "label": "Daily loss limit ($)", "type": "float",
     "min": 0.0, "max": 100000.0, "step": 50.0, "group": "risk",
     "help": "Stop taking new entries for the session once realised P&L is this far down."},
    {"key": "headroom_frac", "label": "Headroom fraction", "type": "float",
     "min": 0.01, "max": 1.0, "step": 0.01, "group": "risk",
     "help": "Cap risk at this fraction of the remaining distance to the floor. Shrinks size automatically as the account approaches the barrier."},
    {"key": "cost_per_trade", "label": "Round-trip cost (pts)", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.25, "group": "risk",
     "help": "Commission + fees per contract round trip, in NQ points."},
    {"key": "slippage_points", "label": "Slippage (pts)", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.25, "group": "risk",
     "help": "Applied against the trade on entry."},
]

# ── VWAP Crossover (/NQ) ─────────────────────────────────────────────────
# Standalone engine (owns its session, entries, stops, targets and chop
# guards). See src/external_models/vwap_cross.py for the specification.

VWAP_CROSS_PARAM_GROUPS: list[tuple[str, str]] = [
    ("session", "Session"),
    ("entries", "Entries"),
    ("chop", "Chop guard"),
    ("exits", "Exits"),
    ("size", "Size & costs"),
]

_STOP_MODE_CHOICES = ["crossing_open", "vwap_offset", "worse_of"]
_PT_MODE_CHOICES = ["absolute", "stop_multiple", "vp_level", "ema_vwap_convergence"]
_ENTRY_MODE_CHOICES = ["close_confirm", "early_cross", "close_confirm_early_fill"]

VWAP_CROSS_PARAM_SPEC: list[dict[str, Any]] = [
    {"key": "session_start", "label": "Session start (CT)", "type": "time", "group": "session",
     "help": "VWAP anchor and earliest bar considered for the strategy (America/Chicago). "
             "Bars before this each day are ignored and do not update session VWAP."},
    {"key": "session_end", "label": "Session end (CT)", "type": "time", "group": "session",
     "help": "Hard flatten. Any open position is closed at this bar's close."},
    {"key": "trade_start", "label": "Trade start (CT)", "type": "time", "group": "session",
     "help": "Earliest allowed *entry* candle (candle N+1). VWAP still accumulates from Session start."},
    {"key": "trade_end", "label": "Trade end (CT)", "type": "time", "group": "session",
     "help": "Last allowed entry candle. Open trades still manage exits until Session end."},

    {"key": "entry_mode", "label": "Entry mode", "type": "enum",
     "choices": list(_ENTRY_MODE_CHOICES), "group": "entries",
     "help": "close_confirm = original rule (wait for candle N to close, enter at N+1 open). "
             "early_cross = intra-bar entry as soon as price is >= entry_ticks past the VWAP "
             "known at candle N's open (V_prev). Tighter stop and better PT:SL ratio at the "
             "cost of some whipsaw (different signal population — see help on entry_ticks). "
             "close_confirm_early_fill = same signal population as close_confirm (filter on the "
             "confirmed close of N) but the FILL is retro-priced at V_prev ± entry_ticks × TICK "
             "as if the intra-bar cross had been caught. Research / A-B use only — not live-safe."},
    {"key": "entry_ticks", "label": "Early-entry ticks past VWAP", "type": "int",
     "min": 0, "max": 400, "step": 1, "group": "entries",
     "help": "For early_cross and close_confirm_early_fill only. Trigger / fill price = V_prev ± "
             "entry_ticks × TICK (0.25 NQ). 0 fires on the very first tick past VWAP; larger "
             "values require deeper penetration (early_cross) or push the retro-fill farther "
             "past VWAP (close_confirm_early_fill)."},
    {"key": "entry_bar_delay", "label": "Entry bar delay", "type": "int",
     "min": 0, "max": 10, "step": 1, "group": "entries",
     "help": "Close-confirm only. Delay the FILL by this many 1-min bars — the strategy "
             "still fires at close of N and computes stop/target from V[N], but the actual "
             "market order goes in at open[N+1+delay] instead of open[N+1]. Pairs with a "
             "1-bar forward slope filter to reproduce that filter's decision live-safely: "
             "you wait one full bar so V[N+1] is legitimately known when you enter."},
    {"key": "trade_dow_mask", "label": "Trade days (M-T-W-T-F-S-S)", "type": "text",
     "group": "entries",
     "help": "7-char '0'/'1' mask for Mon..Sun. '1' = trade the session, '0' = skip. "
             "Default '1111100' = Mon-Fri. '0001100' = Thu+Fri only. Shared across both "
             "sides unless overridden by long_trade_dow_mask / short_trade_dow_mask."},
    {"key": "long_trade_dow_mask", "label": "Trade days — LONG override", "type": "text",
     "group": "entries",
     "help": "Optional per-side override of trade_dow_mask for LONG entries. Empty string "
             "= inherit the shared mask. Enables asymmetric DoW gates."},
    {"key": "short_trade_dow_mask", "label": "Trade days — SHORT override", "type": "text",
     "group": "entries",
     "help": "Optional per-side override of trade_dow_mask for SHORT entries. Empty string "
             "= inherit the shared mask."},
    {"key": "long_max_session_crosses", "label": "Max crosses/session — LONG", "type": "int",
     "min": 0, "max": 200, "step": 1, "group": "chop",
     "help": "Per-side override of max_session_crosses. Set to the shared default to inherit."},
    {"key": "short_max_session_crosses", "label": "Max crosses/session — SHORT", "type": "int",
     "min": 0, "max": 200, "step": 1, "group": "chop",
     "help": "Per-side override of max_session_crosses."},
    {"key": "long_chop_lookback", "label": "Rolling chop window — LONG", "type": "int",
     "min": 0, "max": 240, "step": 1, "group": "chop",
     "help": "Per-side override of chop_lookback."},
    {"key": "short_chop_lookback", "label": "Rolling chop window — SHORT", "type": "int",
     "min": 0, "max": 240, "step": 1, "group": "chop",
     "help": "Per-side override of chop_lookback."},
    {"key": "long_max_recent_crosses", "label": "Max crosses in window — LONG", "type": "int",
     "min": 0, "max": 100, "step": 1, "group": "chop",
     "help": "Per-side override of max_recent_crosses."},
    {"key": "short_max_recent_crosses", "label": "Max crosses in window — SHORT", "type": "int",
     "min": 0, "max": 100, "step": 1, "group": "chop",
     "help": "Per-side override of max_recent_crosses."},
    {"key": "long_vwap_stop_offset", "label": "VWAP stop offset — LONG (pts)", "type": "float",
     "min": 0.0, "max": 50.0, "step": 0.25, "group": "exits",
     "help": "Per-side override of vwap_stop_offset."},
    {"key": "short_vwap_stop_offset", "label": "VWAP stop offset — SHORT (pts)", "type": "float",
     "min": 0.0, "max": 50.0, "step": 0.25, "group": "exits",
     "help": "Per-side override of vwap_stop_offset."},
    {"key": "long_max_stop_pts", "label": "Max stop distance — LONG (pts)", "type": "float",
     "min": 0.0, "max": 500.0, "step": 0.25, "group": "exits",
     "help": "Per-side override of max_stop_pts (0 = no clamp)."},
    {"key": "short_max_stop_pts", "label": "Max stop distance — SHORT (pts)", "type": "float",
     "min": 0.0, "max": 500.0, "step": 0.25, "group": "exits",
     "help": "Per-side override of max_stop_pts (0 = no clamp)."},

    {"key": "trade_long", "label": "Longs", "type": "bool", "group": "entries",
     "help": "Allow the long cross-up setup."},
    {"key": "trade_short", "label": "Shorts", "type": "bool", "group": "entries",
     "help": "Allow the short cross-down setup."},
    {"key": "use_preceding_bar_color", "label": "Require green/red preceder", "type": "bool",
     "group": "entries",
     "help": "When ON: long requires bar N-1 to be green (close > open); short requires red. "
             "OFF disables the preceding-bar color check entirely."},
    {"key": "use_vwap_slope_filter", "label": "Require VWAP slope", "type": "bool",
     "group": "entries",
     "help": "When ON, apply the backward VWAP slope filter (min_vwap_slope over "
             "vwap_slope_lookback bars). OFF ignores VWAP direction and accepts crossings "
             "regardless of the slope."},
    {"key": "vwap_slope_lookback", "label": "VWAP slope bars", "type": "int",
     "min": 1, "max": 60, "step": 1, "group": "entries",
     "help": "Slope window size in bars. Combined with vwap_slope_direction below to "
             "measure the VWAP change over this many bars."},
    {"key": "vwap_slope_direction", "label": "VWAP slope direction", "type": "enum",
     "choices": ["backward", "forward"], "group": "entries",
     "help": "backward (default, LIVE-SAFE) — use bars ending at N (close_confirm) or "
             "N-1 (early modes). forward — use bars N+1..N+lookback, i.e. bars AFTER "
             "the signal. Forward is the original filter and is generally stronger in "
             "backtests, but it PEEKS AT FUTURE BARS and is therefore research-only — "
             "it cannot be reproduced live because the slope over bars after N is "
             "unknowable at the moment the entry order would be sent."},
    {"key": "min_vwap_slope", "label": "Min VWAP slope (pts/bar)", "type": "float",
     "min": 0.0, "max": 100.0, "step": 0.01, "group": "entries",
     "help": "Long requires slope ≥ this value, short requires slope ≤ −this value. "
             "0 = require non-negative slope for longs, non-positive for shorts."},
    {"key": "use_min_signal_clv", "label": "Require min signal CLV", "type": "bool",
     "group": "entries",
     "help": "When ON, reject the signal bar if its close-in-range fraction is "
             "below Min signal CLV. When OFF, the CLV filter is skipped entirely."},
    {"key": "min_close_range_frac", "label": "Min signal CLV", "type": "float",
     "min": 0.0, "max": 1.0, "step": 0.05, "group": "entries",
     "help": "Body-shape filter on the signal bar (live-safe). Long requires "
             "close ≥ low + fraction × (high − low); short requires close ≤ high − "
             "fraction × (high − low). 0.5 = close in the strong half. 0.7 rejects "
             "'spike-and-fade' bars that top-tick and roll over. Only applied when "
             "'Require min signal CLV' is ON."},
    {"key": "use_bar_size_filter", "label": "Require min bar size", "type": "bool",
     "group": "entries",
     "help": "When ON, require the crossing bar's range (high − low) to be at least "
             "bar_size_atr_mult × Wilder ATR(atr_lookback). Filters small-body signals."},
    {"key": "bar_size_atr_mult", "label": "Bar size × ATR", "type": "float",
     "min": 0.1, "max": 10.0, "step": 0.1, "group": "entries",
     "help": "Crossing bar range must be at least this multiple of the Wilder ATR. "
             "1.5 is a common 'large-range' threshold."},
    {"key": "atr_lookback", "label": "ATR bars (Wilder)", "type": "int",
     "min": 2, "max": 100, "step": 1, "group": "entries",
     "help": "Number of bars for the Wilder-smoothed ATR used by the bar-size filter."},
    {"key": "use_ema_cross_filter", "label": "Require EMA/VWAP agreement", "type": "bool",
     "group": "entries",
     "help": "When ON, the signal bar must be a strict price cross-bar (opens on one "
             "side of VWAP, closes on the other) AND the Crossover EMA must already "
             "be on the SAME side of VWAP as the close. Only the price-VWAP crossing "
             "is an event; the EMA side is a pure state check — it does not need to "
             "cross on the signal bar. Uses ema_cross_length. For early_cross the "
             "EMA state is read on the last-final bar (V[N-1])."},
    {"key": "ema_length", "label": "Convergence EMA bars", "type": "int",
     "min": 2, "max": 200, "step": 1, "group": "entries",
     "help": "Slower EMA period used ONLY by the ema_vwap_convergence profit-target "
             "mode's |EMA − VWAP| gap calculation. Not consulted by any entry filter. "
             "Default 6."},
    {"key": "ema_cross_length", "label": "Crossover EMA bars", "type": "int",
     "min": 2, "max": 200, "step": 1, "group": "entries",
     "help": "Quicker EMA period. At signal time the Crossover EMA must be on the "
             "same side of VWAP as the direction of the price cross (state check — "
             "the EMA does not itself need to cross on the signal bar). Default 3."},

    {"key": "max_session_crosses", "label": "Max crosses / session", "type": "int",
     "min": 0, "max": 200, "step": 1, "group": "chop",
     "help": "Hard cap on close-vs-VWAP sign flips per Chicago session. Once exceeded, no more entries "
             "fire that day. Primary defence against choppy VWAP whipsaw sessions."},
    {"key": "chop_lookback", "label": "Rolling chop window (bars)", "type": "int",
     "min": 0, "max": 240, "step": 1, "group": "chop",
     "help": "Window (in candles ending at N-1) used for the rolling chop test. 0 = disable the rolling test."},
    {"key": "max_recent_crosses", "label": "Max crosses in window", "type": "int",
     "min": 0, "max": 100, "step": 1, "group": "chop",
     "help": "Suppress the signal if this many VWAP crosses already fired inside the rolling window. "
             "Set high (e.g. 100) to disable."},

    {"key": "stop_mode", "label": "Stop-loss mode", "type": "enum",
     "choices": list(_STOP_MODE_CHOICES), "group": "exits",
     "help": "crossing_open = crossing candle's opening price. "
             "vwap_offset = VWAP at candle N ± offset. "
             "worse_of (default) = whichever is farther from entry — both invalidations must agree."},
    {"key": "vwap_stop_offset", "label": "VWAP stop offset (pts)", "type": "float",
     "min": 0.0, "max": 50.0, "step": 0.25, "group": "exits",
     "help": "Points below (long) / above (short) the VWAP line for the vwap_offset and worse_of modes."},
    {"key": "max_stop_pts", "label": "Max stop distance (pts)", "type": "float",
     "min": 0.0, "max": 500.0, "step": 0.25, "group": "exits",
     "help": "Hard cap on stop-loss distance in NQ points. If the resolved stop is farther "
             "than this, the stop is pulled in to entry ± max_stop_pts. 0 = no clamp."},
    {"key": "profit_target_mode", "label": "Profit-target mode", "type": "enum",
     "choices": list(_PT_MODE_CHOICES), "group": "exits",
     "help": "absolute = fixed distance in NQ points. "
             "stop_multiple = distance in multiples of the stop range. "
             "vp_level = nearest prior-session volume-profile level (Lo / VAL / POC / VAH / Hi) "
             "past entry; falls back to vp_fallback_pts if no level exists on that side. "
             "ema_vwap_convergence = adaptive exit — bank the trade as soon as the |EMA − VWAP| "
             "gap starts shrinking, once the trade has held for at least min_bars_before_ema_pt bars. "
             "Uses the CONVERGENCE EMA (ema_length) for the gap."},
    {"key": "profit_target_pts", "label": "Profit target (pts)", "type": "float",
     "min": 0.25, "max": 2000.0, "step": 0.25, "group": "exits",
     "help": "Used when Profit-target mode = absolute. Default 40 NQ points."},
    {"key": "profit_target_stop_mult", "label": "Profit target (× stop)", "type": "float",
     "min": 0.1, "max": 20.0, "step": 0.25, "group": "exits",
     "help": "Used when Profit-target mode = stop_multiple. Target distance = mult × stop distance."},
    {"key": "vp_va_pct", "label": "Volume-profile VA %", "type": "float",
     "min": 0.5, "max": 0.95, "step": 0.05, "group": "exits",
     "help": "Value-area fraction used when computing prior-day VAH / VAL for the vp_level target."},
    {"key": "vp_fallback_pts", "label": "VP-level fallback (pts)", "type": "float",
     "min": 0.25, "max": 2000.0, "step": 0.25, "group": "exits",
     "help": "Used only when profit_target_mode = vp_level AND no prior-session level exists "
             "past the entry price on the required side (e.g. long above prior-day high). "
             "Falls back to this absolute point target so the trade still has an exit."},
    {"key": "min_bars_before_ema_pt", "label": "EMA-PT: min bars in trade", "type": "int",
     "min": 0, "max": 240, "step": 1, "group": "exits",
     "help": "For ema_vwap_convergence PT mode only. Do not evaluate the gap-shrink exit "
             "until the trade has held for at least this many 1-min bars."},
    {"key": "ema_vwap_shrink_pct", "label": "EMA-PT: shrink % from peak", "type": "float",
     "min": 0.0, "max": 100.0, "step": 1.0, "group": "exits",
     "help": "Exit when the |EMA − VWAP| gap has fallen this % below its peak-since-entry. "
             "15 = exit on a 15% pullback of the gap; smaller values exit sooner."},

    {"key": "qty", "label": "Contracts", "type": "int",
     "min": 1, "max": 20, "step": 1, "group": "size",
     "help": "Fixed contract quantity per trade."},
    {"key": "point_value", "label": "$ per point / contract", "type": "float",
     "min": 0.5, "max": 500.0, "step": 0.5, "group": "size",
     "help": "20 = /NQ full-size, 2 = /MNQ micro."},
    {"key": "cost_per_trade", "label": "Round-trip cost (pts)", "type": "float",
     "min": 0.0, "max": 20.0, "step": 0.25, "group": "size",
     "help": "Commission + fees in NQ points per contract round trip."},
    {"key": "slippage_points", "label": "Slippage per side (pts)", "type": "float",
     "min": 0.0, "max": 10.0, "step": 0.25, "group": "size",
     "help": "Applied against the trade on entry and on exit."},
]

VWAP_CROSS_FACTORY_PARAMS: dict[str, Any] = {
    "session_start": "08:30",
    "session_end": "14:45",
    "trade_start": "08:30",
    "trade_end": "14:45",
    "entry_mode": "close_confirm",
    "entry_ticks": 4,
    "entry_bar_delay": 0,
    "trade_dow_mask": "1111100",
    "long_trade_dow_mask": "",
    "short_trade_dow_mask": "",
    "long_max_session_crosses": 4,
    "short_max_session_crosses": 4,
    "long_chop_lookback": 20,
    "short_chop_lookback": 20,
    "long_max_recent_crosses": 2,
    "short_max_recent_crosses": 2,
    "long_vwap_stop_offset": 1.0,
    "short_vwap_stop_offset": 1.0,
    "long_max_stop_pts": 20.0,
    "short_max_stop_pts": 20.0,
    "trade_long": True,
    "trade_short": True,
    "use_preceding_bar_color": True,
    "use_vwap_slope_filter": True,
    "vwap_slope_lookback": 8,
    "vwap_slope_direction": "backward",
    "min_vwap_slope": 0.0,
    "min_close_range_frac": 0.6,
    "use_min_signal_clv": True,
    "use_bar_size_filter": False,
    "bar_size_atr_mult": 1.5,
    "atr_lookback": 14,
    "use_ema_cross_filter": False,
    "ema_length": 6,
    "ema_cross_length": 3,
    "max_session_crosses": 4,
    "chop_lookback": 20,
    "max_recent_crosses": 2,
    "stop_mode": "worse_of",
    "vwap_stop_offset": 1.0,
    "max_stop_pts": 20.0,
    "profit_target_mode": "absolute",
    "profit_target_pts": 40.0,
    "profit_target_stop_mult": 2.0,
    "vp_va_pct": 0.70,
    "vp_fallback_pts": 40.0,
    "min_bars_before_ema_pt": 3,
    "ema_vwap_shrink_pct": 15.0,
    "qty": 1,
    "point_value": 20.0,
    "cost_per_trade": 0.0,
    "slippage_points": 0.0,
}

ORB_FACTORY_PARAMS: dict[str, Any] = {
    "or_minutes": 15,
    "session_start": "08:30",
    "trade_end": "13:00",
    "session_end": "14:45",
    "va_pct": 0.70,
    "min_rvol": 1.3,
    "rvol_lookback": 14,
    "min_or_atr": 0.20,
    "max_or_atr": 1.30,
    "atr_days": 14,
    "setups": "breakout_retest",
    "entry_start": "08:45",
    "break_buffer_ticks": 4.0,
    "min_poke_atr": 0.02,
    "rejection_clv": 0.65,
    "retest_band_frac": 0.10,
    "fade_band_frac": 0.15,
    "fade_max_or_atr": 0.80,
    "max_trades_per_day": 2,
    "t1_r": 1.0,
    "t2_r": 2.0,
    "scale_out_pct": 0.5,
    "breakeven_after_t1": True,
    "be_offset_pts": 1.0,
    "trail_pts": 0.0,
    "max_stop_points": 50.0,
    "initial_balance": 5000.0,
    "point_value": 20.0,
    "max_drawdown": 1000.0,
    "risk_usd_per_trade": 200.0,
    "max_contracts": 3,
    "margin_per_contract": 1000.0,
    "daily_loss_limit": 300.0,
    "headroom_frac": 0.5,
    "cost_per_trade": 0.0,
    "slippage_points": 0.25,
}

_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "psar": {
        "engine": "psar",
        "param_spec": PSAR_PARAM_SPEC,
        "factory": PSAR_FACTORY_PARAMS,
    },
    "ema20": {
        "engine": "ema",
        "param_spec": EMA20_PARAM_SPEC,
        "factory": EMA20_FACTORY_PARAMS,
    },
    "zigzag": {
        "engine": "zigzag",
        "param_spec": ZZHL_PARAM_SPEC,
        "factory": ZZHL_FACTORY_PARAMS,
    },
    "timed_psar": {
        "engine": "timed_psar",
        "param_spec": TIMED_PSAR_PARAM_SPEC,
        "factory": TIMED_PSAR_FACTORY_PARAMS,
    },
    "orb": {
        "engine": "orb",
        "param_spec": ORB_PARAM_SPEC,
        "factory": ORB_FACTORY_PARAMS,
        "groups": ORB_PARAM_GROUPS,
        "standalone": True,
    },
    "vwap_cross": {
        "engine": "vwap_cross",
        "param_spec": VWAP_CROSS_PARAM_SPEC,
        "factory": VWAP_CROSS_FACTORY_PARAMS,
        "groups": VWAP_CROSS_PARAM_GROUPS,
        "standalone": True,
    },
    # Hidden compat card: old combined Pine id → stack of the two engines.
    LEGACY_COMBINED_ID: {
        "engine": "stack",
        "param_spec": PSAR_PARAM_SPEC + EMA20_PARAM_SPEC,
        "factory": {**PSAR_FACTORY_PARAMS, **EMA20_FACTORY_PARAMS},
        "listed": False,
        "compat_stack": list(COMPAT_STACK),
    },
}

# Chart / ML overlay fallback (session + SAR AF). Not a third catalog family.
FACTORY_PARAMS: dict[str, Any] = {
    **BOOK_FACTORY_PARAMS,
    **PSAR_FACTORY_PARAMS,
    **EMA20_FACTORY_PARAMS,
    **ZZHL_FACTORY_PARAMS,
    **TIMED_PSAR_FACTORY_PARAMS,
}
_BOOK_TYPES = {row["key"]: row for row in BOOK_PARAM_SPEC}
for _row in BOOK_PARAM_SPEC:
    _uk = _row.get("unit_key")
    if _uk:
        _BOOK_TYPES[str(_uk)] = {
            "key": str(_uk), "type": "enum",
            "choices": list(_row.get("unit_choices") or _UNIT_CHOICES),
        }
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?$")
_IGNORED_OVERLAY_KEYS = {
    "timezone", "long_only", "short_only", "ema_gate",
    "profit_target", "stop_loss",
}


def _load_yaml_card(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or not MODEL_ID_RE.match(str(raw.get("id") or "")):
        raise ValueError(f"invalid external-model card: {path}")
    return raw


def list_cards(*, include_hidden: bool = False) -> list[dict[str, Any]]:
    """Factory cards from ``config/external_models/*.yaml``."""
    if not CATALOG_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        try:
            card = _load_yaml_card(path)
        except (ValueError, yaml.YAMLError):
            continue
        listed = card.get("listed", True)
        if listed is False and not include_hidden:
            continue
        out.append(card)
    order = {"psar": 0, "ema20": 1, "zigzag": 2, "timed_psar": 3, "orb": 4, "vwap_cross": 5}
    out.sort(key=lambda c: (order.get(c["id"], 50), c["id"]))
    return out


def get_card(model_id: str) -> dict[str, Any]:
    check_model_id(model_id)
    path = CATALOG_DIR / f"{model_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown external model: {model_id}")
    return _load_yaml_card(path)


def check_model_id(model_id: str) -> str:
    if not MODEL_ID_RE.match(model_id or ""):
        raise ValueError("invalid external model id")
    return model_id


def check_hardened_name(name: str) -> str:
    if not HARDENED_NAME_RE.match(name or ""):
        raise ValueError(
            "hardened name must be lowercase alphanumeric / underscore / hyphen, 1–64 chars")
    return name


def model_engine(model_id: str) -> str:
    meta = _MODEL_SPECS.get(model_id)
    if meta and meta.get("engine"):
        return str(meta["engine"])
    card = get_card(model_id)
    return str(card.get("engine") or model_id)


def compat_stack_for(model_id: str) -> list[str] | None:
    if model_id == LEGACY_COMBINED_ID:
        return list(COMPAT_STACK)
    card = get_card(model_id)
    raw = card.get("compat_stack")
    if isinstance(raw, list) and raw:
        return [check_model_id(str(x)) for x in raw]
    meta = _MODEL_SPECS.get(model_id) or {}
    cs = meta.get("compat_stack")
    if isinstance(cs, list) and cs:
        return [str(x) for x in cs]
    return None


def factory_params(model_id: str) -> dict[str, Any]:
    check_model_id(model_id)
    meta = _MODEL_SPECS.get(model_id)
    merged = dict(meta["factory"]) if meta else {}
    try:
        card = get_card(model_id)
    except FileNotFoundError:
        if not merged:
            raise
        return merged
    yaml_params = card.get("params") or {}
    if isinstance(yaml_params, dict):
        allowed = set(merged) if merged else set(yaml_params)
        merged.update({k: yaml_params[k] for k in yaml_params if k in allowed})
    return merged


def factory_book_params() -> dict[str, Any]:
    return dict(BOOK_FACTORY_PARAMS)


def param_spec(model_id: str) -> list[dict[str, Any]]:
    check_model_id(model_id)
    meta = _MODEL_SPECS.get(model_id)
    if meta:
        return list(meta["param_spec"])
    return []


def book_param_spec() -> list[dict[str, Any]]:
    return list(BOOK_PARAM_SPEC)


def all_known_keys() -> set[str]:
    keys = set(BOOK_FACTORY_PARAMS) | _IGNORED_OVERLAY_KEYS
    for meta in _MODEL_SPECS.values():
        keys |= set(meta["factory"])
        keys |= {row["key"] for row in meta["param_spec"]}
    return keys


def partition_params(raw: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Split a mixed overlay into ``({model_id: params}, book_params)``."""
    src = dict(raw or {})
    book = {k: src[k] for k in src if k in BOOK_FACTORY_PARAMS or k in _BOOK_TYPES}
    per_model: dict[str, dict[str, Any]] = {}
    for mid, meta in _MODEL_SPECS.items():
        if mid == LEGACY_COMBINED_ID:
            continue
        allowed = set(meta["factory"])
        chunk = {k: src[k] for k in src if k in allowed}
        if chunk:
            per_model[mid] = chunk
    return per_model, book


def coerce_params(model_id: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate model-specific knobs. Book keys in *raw* are ignored."""
    base = factory_params(model_id)
    if not raw:
        return dict(base)
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    spec = param_spec(model_id)
    types = {row["key"]: row for row in spec}
    known = all_known_keys()
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown param(s): {', '.join(unknown)}")
    out = dict(base)
    for key, value in raw.items():
        spec_row = types.get(key)
        if spec_row is None:
            continue
        out[key] = _coerce_one(spec_row, value)
    if "start" in out and "maximum" in out:
        if float(out["maximum"]) < float(out["start"]):
            raise ValueError("SAR maximum must be ≥ start")
    if "ema_length" in out:
        use_slope = int(out.get("ema20_l_bars") or 0) > 0 or int(out.get("ema20_s_bars") or 0) > 0
        if use_slope and int(out.get("ema_length") or 0) < 2:
            raise ValueError("ema_length must be ≥ 2")
    if "percentage_reversal" in out and "atr_reversal" in out:
        pct = float(out.get("percentage_reversal") or 0)
        abs_rev = float(out.get("absolute_reversal") or 0)
        atr_rev = float(out.get("atr_reversal") or 0)
        tick_rev = int(out.get("tick_reversal") or 0)
        if pct == 0.0 and abs_rev == 0.0 and atr_rev == 0.0 and tick_rev == 0:
            raise ValueError(
                "either percentage, absolute, ATR, or tick reversal must be non-zero")
    if model_id == "orb":
        allowed_setups = {"breakout_retest", "failed_break", "fade"}
        picked = [s.strip() for s in str(out.get("setups") or "").split(",") if s.strip()]
        if not picked:
            raise ValueError("at least one ORB setup must be enabled")
        bad = sorted(set(picked) - allowed_setups)
        if bad:
            raise ValueError(
                f"unknown ORB setup(s): {', '.join(bad)}; allowed: {sorted(allowed_setups)}")
        out["setups"] = ",".join(dict.fromkeys(picked))
        if float(out["max_or_atr"]) < float(out["min_or_atr"]):
            raise ValueError("max OR ÷ ATR must be ≥ min OR ÷ ATR")
        if float(out["t2_r"]) < float(out["t1_r"]):
            raise ValueError("target 2 (R) must be ≥ target 1 (R)")
        if float(out["max_drawdown"]) >= float(out["initial_balance"]):
            raise ValueError("max drawdown must be smaller than the starting balance")
    if model_id == "vwap_cross":
        for k in ("session_start", "session_end", "trade_start", "trade_end"):
            _parse_hhmm(str(out[k]))
        if _hhmm_minutes(str(out["session_end"])) <= _hhmm_minutes(str(out["session_start"])):
            raise ValueError("session_end must be after session_start")
        if _hhmm_minutes(str(out["trade_end"])) <= _hhmm_minutes(str(out["trade_start"])):
            raise ValueError("trade_end must be after trade_start")
        if _hhmm_minutes(str(out["trade_start"])) < _hhmm_minutes(str(out["session_start"])):
            raise ValueError("trade_start must be at or after session_start")
        if _hhmm_minutes(str(out["trade_end"])) > _hhmm_minutes(str(out["session_end"])):
            raise ValueError("trade_end must be at or before session_end")
        if not (bool(out["trade_long"]) or bool(out["trade_short"])):
            raise ValueError("at least one of longs / shorts must be enabled")
        if out["stop_mode"] not in _STOP_MODE_CHOICES:
            raise ValueError(f"stop_mode must be one of {_STOP_MODE_CHOICES}")
        if out["profit_target_mode"] not in _PT_MODE_CHOICES:
            raise ValueError(f"profit_target_mode must be one of {_PT_MODE_CHOICES}")
        if out["entry_mode"] not in _ENTRY_MODE_CHOICES:
            raise ValueError(f"entry_mode must be one of {_ENTRY_MODE_CHOICES}")
        if str(out.get("vwap_slope_direction", "backward")).lower() not in ("backward", "forward"):
            raise ValueError("vwap_slope_direction must be 'backward' or 'forward'")
        dow = str(out.get("trade_dow_mask") or "").strip()
        if dow and (len(dow) != 7 or any(c not in "01" for c in dow) or "1" not in dow):
            raise ValueError("trade_dow_mask must be a 7-char '0'/'1' string with at least one '1'")
        # Per-side DoW overrides: empty string = inherit; non-empty must be valid.
        for k in ("long_trade_dow_mask", "short_trade_dow_mask"):
            v = str(out.get(k) or "").strip()
            if v and (len(v) != 7 or any(c not in "01" for c in v) or "1" not in v):
                raise ValueError(f"{k} must be empty or a 7-char '0'/'1' string with at least one '1'")
    return out


def coerce_book_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Validate shared book knobs. Model-specific keys in *raw* are ignored."""
    base = factory_book_params()
    if not raw:
        return dict(base)
    if not isinstance(raw, dict):
        raise ValueError("params must be an object")
    known = all_known_keys()
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown param(s): {', '.join(unknown)}")
    out = dict(base)
    src = dict(raw)
    if "profit_target" in src and "profit_target_pts" not in src:
        src["profit_target_pts"] = src["profit_target"]
    if "stop_loss" in src and "stop_loss_pts" not in src:
        src["stop_loss_pts"] = src["stop_loss"]
    for key, value in src.items():
        if key == "timezone":
            tz = str(value or "America/Chicago").strip()
            if not tz:
                raise ValueError("timezone is empty")
            out["timezone"] = tz
            continue
        spec = _BOOK_TYPES.get(key)
        if spec is None:
            continue
        if key == "timeframe":
            out[key] = coerce_candle_tf(value)
            continue
        kind = spec["type"]
        if kind == "qty_unit":
            out[key] = _coerce_one({**spec, "type": "float"}, value)
            continue
        out[key] = _coerce_one(spec, value)
    _parse_hhmm(str(out["session_start"]))
    _parse_hhmm(str(out["session_end"]))
    _parse_hhmm(str(out["trade_start"]))
    _parse_hhmm(str(out["trade_end"]))
    if _hhmm_minutes(str(out["session_end"])) <= _hhmm_minutes(str(out["session_start"])):
        raise ValueError("session_end must be after session_start")
    if _hhmm_minutes(str(out["trade_end"])) <= _hhmm_minutes(str(out["trade_start"])):
        raise ValueError("trade_end must be after trade_start")
    if _hhmm_minutes(str(out["trade_start"])) < _hhmm_minutes(str(out["session_start"])):
        raise ValueError("trade_start must be at or after session_start")
    if _hhmm_minutes(str(out["trade_end"])) > _hhmm_minutes(str(out["session_end"])):
        raise ValueError("trade_end must be at or before session_end")
    if "long_only" in src or "short_only" in src:
        lo = _as_bool(src.get("long_only"), False)
        so = _as_bool(src.get("short_only"), False)
        if lo and not so:
            out["trade_long"], out["trade_short"] = True, False
        elif so and not lo:
            out["trade_long"], out["trade_short"] = False, True
    return out


def coerce_chart_params(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Book + SAR AF for candle charts that are not a specific EXT stack."""
    return {**coerce_book_params(raw), **coerce_params("psar", raw)}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        return bool(_coerce_one({"type": "bool", "key": "flag"}, value))
    except ValueError:
        return default


def _coerce_one(spec: dict[str, Any], value: Any) -> Any:
    kind = spec["type"]
    key = spec["key"]
    if kind == "int":
        try:
            n = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be an integer") from exc
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and n < lo:
            raise ValueError(f"{key} must be ≥ {lo}")
        if hi is not None and n > hi:
            raise ValueError(f"{key} must be ≤ {hi}")
        return n
    if kind in ("float", "qty_unit"):
        try:
            x = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a number") from exc
        if not (x == x):  # NaN
            raise ValueError(f"{key} is not a number")
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and x < lo:
            raise ValueError(f"{key} must be ≥ {lo}")
        if hi is not None and x > hi:
            raise ValueError(f"{key} must be ≤ {hi}")
        return x
    if kind == "time":
        s = str(value or "").strip()
        h, m = _parse_hhmm(s)
        return f"{h:02d}:{m:02d}"
    if kind == "enum":
        s = str(value or "").strip()
        choices = spec.get("choices") or []
        if s not in choices:
            raise ValueError(f"{key} must be one of {choices}")
        return s
    if kind == "text":
        return str(value or "").strip()
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} must be true or false")
    return value


def _parse_hhmm(s: str) -> tuple[int, int]:
    m = _TIME_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"time must be HH:MM (got {s!r})")
    return int(m.group(1)), int(m.group(2))


def _hhmm_minutes(s: str) -> int:
    h, m = _parse_hhmm(s)
    return h * 60 + m


def get_model(model_id: str) -> dict[str, Any]:
    """Factory card plus param spec (no operator overlays)."""
    card = get_card(model_id)
    return {
        **card,
        "engine": model_engine(model_id),
        "param_spec": param_spec(model_id),
        "factory_params": factory_params(model_id),
        "compat_stack": compat_stack_for(model_id),
    }


def list_models(*, include_hidden: bool = False) -> list[dict[str, Any]]:
    return [get_model(c["id"]) for c in list_cards(include_hidden=include_hidden)]
