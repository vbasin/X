"""Path-realised portfolio simulator — account, margin and risk constraints on 1s fills.

This module answers a different question from the two evaluators that already exist:

* :func:`src.evaluation._kpi.label_implied_pnl` prices a trade from its **label**
  (correct call ⇒ ``+move_threshold``, wrong call ⇒ ``-mae_tolerance``). It never looks
  at a price path, so it cannot see a stop that was tagged intrabar.
* :mod:`src.backtest.engine` walks the **5-second close** series. Better, but a 6-point
  stop is smaller than a typical 5s NQ bar range during the cash open, so intrabar stop
  touches are still invisible.

Here, first-touch is resolved on the **1-second high/low path**
(``parquet_path1s``, built by ``scripts/data/build_path1s.py``) and every trade is run
through an account: contract sizing, margin capacity, a concurrent-position cap, a
per-session loss halt, commissions and adverse stop slippage. The simulator **never
reads a label** — P&L comes only from the path, so it is the honest counterpart to the
label-implied KPI.

Two deliberate design choices:

*Constraints are enforced, not learned.* Position caps, margin, the hold limit, the loss
halt and the session window are environment rules evaluated here. A model should spend
its capacity on direction and timing, never on rediscovering a rule that can be applied
for free.

*Trade frequency is a dial, not a training target.* :func:`sweep_tau` reports the
``(n_trades, expectancy, drawdown, $/month)`` frontier across confidence thresholds, so
the operating point is *read off* a curve instead of being forced by a quota reward.
A quota makes idling impossible but also makes negative-expectancy trades mandatory —
the frontier keeps the model honest about when there is no edge.

Fidelity knobs that exist so bias can be *measured* rather than assumed:

``tie_policy``
    ``"stop_first"`` (default, conservative) — when one 1s bar straddles both barriers
    the stop is credited; ``"target_first"`` — optimistic upper bound;
    ``"close_only"`` — reproduces the legacy close-series behaviour. Running the same
    signals under ``close_only`` and ``stop_first`` quantifies the labelling bias.

``stop_slippage_ticks``
    A stop is a market order in an impulse: it fills *through* the level.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, replace
from datetime import time as dtime
from typing import TYPE_CHECKING, Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.features.session_path import parse_hhmm as _parse_hhmm
from src.features.session_path import prepare_path, session_grid as _session_grid

if TYPE_CHECKING:  # the economic boundary; imported lazily at call time
    from src.evaluation.economic import EconomicAssumptions

logger = logging.getLogger("vtech.backtest.portfolio_sim")

_CT = ZoneInfo("America/Chicago")

# Exit reason codes (ints inside the vectorised scan, strings in the blotter).
_TARGET, _STOP, _TIME, _FLATTEN, _HALT = 0, 1, 2, 3, 4
_REASON = {_TARGET: "target", _STOP: "stop", _TIME: "time",
           _FLATTEN: "flatten", _HALT: "halt"}

_TIE_POLICIES = ("stop_first", "target_first", "close_only")

# A session needs an entry bar and at least one bar to exit on. Dead-day
# filtering is the path builder's job (``_MIN_ROWS``), not the simulator's.
_MIN_SESSION_BARS = 2


@dataclass(frozen=True)
class SimConfig:
    """Trade geometry, contract economics, account rules and fill conservatism.

    Defaults match the ``nq-50pt-*`` objectives (TBM +50 / −6 / 10m) on NQ
    (``$20``/point, ``0.25`` tick) with the ADR-047 inference policy window.
    """

    # ── trade geometry ────────────────────────────────────────────────────────
    target_points: float = 50.0
    stop_points: float = 6.0
    max_hold_seconds: int = 600

    # ── session (America/Chicago); entries only inside [trade_start, trade_end) ─
    trade_start: str = "08:30"
    trade_end: str = "10:00"
    flatten_at: str = "14:45"

    # ── contract economics ────────────────────────────────────────────────────
    point_value_usd: float = 20.0
    tick_size: float = 0.25
    commission_per_side_usd: float = 2.50
    entry_slippage_ticks: float = 1.0
    stop_slippage_ticks: float = 2.0
    target_slippage_ticks: float = 0.0

    # ── account & risk ────────────────────────────────────────────────────────
    starting_equity_usd: float = 5_000.0
    margin_per_contract_usd: float = 1_000.0
    contracts_per_trade: int = 1
    max_concurrent_positions: int = 5
    daily_loss_limit_frac: float = 0.5      # halt at −50% of session-start equity
    allow_opposing_positions: bool = False  # hedged long+short book

    # ── signal gating ─────────────────────────────────────────────────────────
    tau: float = 0.65
    cooldown_seconds: int = 600
    sides: tuple[str, ...] = ("up", "down")

    # ── fill conservatism ─────────────────────────────────────────────────────
    tie_policy: str = "stop_first"

    def __post_init__(self) -> None:
        if self.tie_policy not in _TIE_POLICIES:
            raise ValueError(f"tie_policy must be one of {_TIE_POLICIES}")
        if self.target_points <= 0 or self.stop_points <= 0:
            raise ValueError("target_points and stop_points must be positive")
        if self.max_hold_seconds < 1:
            raise ValueError("max_hold_seconds must be >= 1")
        if self.contracts_per_trade < 1 or self.max_concurrent_positions < 1:
            raise ValueError("contracts_per_trade and max_concurrent_positions must be >= 1")
        if not 0 < self.daily_loss_limit_frac <= 1:
            raise ValueError("daily_loss_limit_frac must be in (0, 1]")
        if _parse_hhmm(self.trade_end) > _parse_hhmm(self.flatten_at):
            raise ValueError("flatten_at must be at or after trade_end")
        bad = set(self.sides) - {"up", "down"}
        if bad:
            raise ValueError(f"sides must be a subset of ('up','down'); got {sorted(bad)}")

    # Derived economics -------------------------------------------------------
    @property
    def entry_slip(self) -> float:
        return self.entry_slippage_ticks * self.tick_size

    @property
    def stop_slip(self) -> float:
        return self.stop_slippage_ticks * self.tick_size

    @property
    def target_slip(self) -> float:
        return self.target_slippage_ticks * self.tick_size

    @property
    def round_trip_commission_usd(self) -> float:
        return 2.0 * self.commission_per_side_usd * self.contracts_per_trade

    @property
    def margin_per_trade_usd(self) -> float:
        return self.margin_per_contract_usd * self.contracts_per_trade

    # Economic boundary ------------------------------------------------------
    # This simulator owns trade sequencing, path resolution and account state.
    # What a point is worth, what a round trip costs and how a leg slips are
    # the canonical economic layer's to define
    # (src/evaluation/economic.py). These two methods are the whole
    # bridge: nothing else here reads or writes a cost assumption, and there is
    # no second cost model to drift from.
    #
    # The defaults above are NOT replaced by a config profile: this sim's
    # $2.50/side ($5.00 round trip) is a more conservative assumption than
    # config/execution_costs.yaml's NQ profile ($3.36 all-in), and silently
    # adopting the cheaper number would change every historical KPI. A caller
    # that wants the profile passes it explicitly.

    def as_assumptions(self, *, label: str | None = None) -> "EconomicAssumptions":
        """This config's cost assumptions, in the canonical vocabulary.

        Lossless: the per-leg slippage this simulator distinguishes is carried
        on the canonical object's per-leg fields.
        """
        from src.evaluation.economic import EconomicAssumptions

        return EconomicAssumptions(
            contract_multiplier=self.point_value_usd,
            commission_per_trade=2.0 * self.commission_per_side_usd,
            slippage_points_per_side=self.entry_slip,
            entry_slippage_points=self.entry_slip,
            stop_slippage_points=self.stop_slip,
            target_slippage_points=self.target_slip,
            underlying=None,
            fill_model=f"path_first_touch tie_policy={self.tie_policy}",
            label=label or "portfolio_sim",
        )

    @classmethod
    def from_assumptions(
        cls, assumptions: "EconomicAssumptions", *, tick_size: float = 0.25, **overrides: Any
    ) -> "SimConfig":
        """Build a config whose economics come from the canonical layer.

        ``commission_per_trade`` is one all-in round-trip figure, so it halves
        into this simulator's per-side field. Slippage arrives in points and is
        stored in ticks, which is the same quantity in different units.
        """
        tick = float(tick_size)
        if tick <= 0:
            raise ValueError("tick_size must be > 0")
        return cls(
            point_value_usd=float(assumptions.contract_multiplier),
            commission_per_side_usd=float(assumptions.commission_per_trade) / 2.0,
            tick_size=tick,
            entry_slippage_ticks=assumptions.slippage_for("entry") / tick,
            stop_slippage_ticks=assumptions.slippage_for("stop") / tick,
            target_slippage_ticks=assumptions.slippage_for("target") / tick,
            **overrides,
        )


@dataclass
class SimResult:
    """Blotter, per-session ledger, headline KPIs and the config that produced them."""

    trades: pd.DataFrame
    daily: pd.DataFrame
    kpis: dict[str, Any]
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        k = self.kpis
        def f(key: str, fmt: str = "{:.2f}") -> str:
            v = k.get(key)
            return "n/a" if v is None else fmt.format(v)
        return (
            f"n_trades={k.get('n_trades', 0)}  "
            f"hit_rate={f('hit_rate', '{:.3f}')}  "
            f"net_exp_pts={f('net_expectancy_points')}  "
            f"net_usd={f('net_usd', '{:,.0f}')}  "
            f"usd_per_month={f('median_month_net_usd', '{:,.0f}')}  "
            f"mdd_usd={f('max_drawdown_usd', '{:,.0f}')}  "
            f"halt_days={k.get('halt_days', 0)}/{k.get('n_sessions', 0)}  "
            f"ambiguous={f('ambiguous_exit_frac', '{:.3f}')}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Path preparation
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Vectorised first-touch exit resolution
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_exits(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entry_idx: np.ndarray,
    side: int,
    *,
    cfg: SimConfig,
    last_idx: int,
) -> dict[str, np.ndarray]:
    """First-touch outcome for a hypothetical entry at each bar in *entry_idx*.

    Exits are independent of the account (barriers depend only on price), so every
    candidate is priced once here and the account loop simply picks the ones it took.

    ``side`` is ``+1`` long / ``-1`` short. Scanning starts at ``k=1``: the entry fills
    at the *close* of the signal bar, so that bar's own range cannot exit it.
    Returns per-candidate arrays: ``exit_off`` (bars held), ``reason``, ``entry_px``,
    ``exit_px``, ``mae_points``, ``mfe_points``, ``ambiguous``.
    """
    n = len(entry_idx)
    entry_px = close[entry_idx] + side * cfg.entry_slip
    target = entry_px + side * cfg.target_points
    stop = entry_px - side * cfg.stop_points

    close_only = cfg.tie_policy == "close_only"
    # Favourable / adverse extreme of a bar, from this side's point of view.
    fav_series = close if close_only else (high if side > 0 else low)
    adv_series = close if close_only else (low if side > 0 else high)

    # Bars available before the forced session flatten (>=1 guaranteed by the caller).
    limit = np.minimum(cfg.max_hold_seconds, last_idx - entry_idx).astype(np.int64)

    sentinel = np.int64(cfg.max_hold_seconds + 1)
    exit_off = np.full(n, sentinel, dtype=np.int64)
    reason = np.full(n, -1, dtype=np.int8)
    exit_px = np.full(n, np.nan, dtype="float64")
    ambiguous = np.zeros(n, dtype=bool)
    mae = np.zeros(n, dtype="float64")   # adverse excursion, positive points
    mfe = np.zeros(n, dtype="float64")   # favourable excursion, positive points

    for k in range(1, int(cfg.max_hold_seconds) + 1):
        live = (reason < 0) & (limit >= k)
        if not live.any():
            break
        j = entry_idx[live] + k
        fav, adv = fav_series[j], adv_series[j]
        ep = entry_px[live]

        # Excursions over every held bar, including the exit bar. The account loop
        # re-tracks these for the positions it actually opens (a halt can cut a
        # position short of its barrier), so these are the no-halt reference values.
        mae[live] = np.maximum(mae[live], side * (ep - adv))
        mfe[live] = np.maximum(mfe[live], side * (fav - ep))

        t_hit = side * (fav - target[live]) >= 0
        s_hit = side * (adv - stop[live]) <= 0
        both = t_hit & s_hit

        if cfg.tie_policy == "target_first":
            take_t, take_s = t_hit, s_hit & ~t_hit
        else:  # stop_first, and close_only (a close cannot satisfy both barriers)
            take_s, take_t = s_hit, t_hit & ~s_hit

        idx_live = np.flatnonzero(live)
        for mask, code, px in ((take_t, _TARGET, target[live] - side * cfg.target_slip),
                               (take_s, _STOP, stop[live] - side * cfg.stop_slip)):
            if mask.any():
                sel = idx_live[mask]
                exit_off[sel] = k
                reason[sel] = code
                exit_px[sel] = px[mask]
        if both.any():
            ambiguous[idx_live[both]] = True

    # Anything still open exited on time (hold cap) or the forced session flatten.
    open_still = reason < 0
    if open_still.any():
        sel = np.flatnonzero(open_still)
        off = limit[sel]
        exit_off[sel] = off
        exit_px[sel] = close[entry_idx[sel] + off]
        reason[sel] = np.where(off >= cfg.max_hold_seconds, _TIME, _FLATTEN)

    return {"exit_off": exit_off, "reason": reason, "entry_px": entry_px,
            "exit_px": exit_px, "mae_points": mae, "mfe_points": mfe,
            "ambiguous": ambiguous}


# ══════════════════════════════════════════════════════════════════════════════
# Session-level account loop
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_session(
    grid: pd.DataFrame,
    sig_pos: np.ndarray,
    sig_side: np.ndarray,
    equity_open: float,
    cfg: SimConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one session. Returns ``(trades, session_summary)``.

    Ordering inside a second is deliberate and conservative: mark open positions at
    their adverse extreme → resolve barrier exits → test the loss halt → only then
    consider a new entry. So a signal never opens on a second the account was already
    over its limit, and the halt is evaluated on worst-case intrabar equity rather than
    on closes.
    """
    high = grid["high"].to_numpy()
    low = grid["low"].to_numpy()
    close = grid["close"].to_numpy()
    ts = grid.index
    last_idx = len(grid) - 1

    # Pre-price every candidate entry, both sides.
    pre = {s: _resolve_exits(high, low, close, sig_pos, s, cfg=cfg, last_idx=last_idx)
           for s in (1, -1)}

    loss_limit = equity_open * cfg.daily_loss_limit_frac
    comm_pts = cfg.round_trip_commission_usd / (cfg.point_value_usd * cfg.contracts_per_trade)
    usd_per_point = cfg.point_value_usd * cfg.contracts_per_trade

    equity = equity_open              # realised only
    trades: list[dict[str, Any]] = []
    open_pos: list[dict[str, Any]] = []
    halted = False
    halt_ts = None
    last_entry_pos = -(10 ** 9)
    peak_margin = 0.0
    min_equity_mark = equity_open
    rejects = {"tau": 0, "capacity": 0, "margin": 0, "cooldown": 0,
               "halted": 0, "opposing": 0, "eod": 0}

    sig_at = dict(zip(sig_pos.tolist(), sig_side.tolist()))
    cand_of_pos = {int(p): c for c, p in enumerate(sig_pos.tolist())}

    def _close_position(p: dict[str, Any], i: int, px: float, code: int) -> None:
        nonlocal equity
        gross = p["side"] * (px - p["entry_px"])
        net_pts = gross - comm_pts
        net_usd = net_pts * usd_per_point
        equity += net_usd
        trades.append({
            "entry_ts": ts[p["entry_i"]], "exit_ts": ts[i],
            "side": "up" if p["side"] > 0 else "down",
            "contracts": cfg.contracts_per_trade,
            "entry_px": p["entry_px"], "exit_px": px,
            "exit_reason": _REASON[code],
            "hold_seconds": int(i - p["entry_i"]),
            "gross_points": gross, "net_points": net_pts, "net_usd": net_usd,
            "mae_points": p["mae"], "mfe_points": p["mfe"],
            "ambiguous_bar": bool(p["ambiguous"]),
            "equity_after_usd": equity,
        })

    ptr = 0                           # next unconsumed signal in sig_pos (monotonic)
    i = int(sig_pos[0])
    while i <= last_idx:
        if not open_pos:
            # Nothing to monitor — fast-forward to the next signal second. Without an
            # open position the halt cannot trip, so intermediate seconds are inert.
            if halted:
                # Book the rest of the day's signals as halt-refused: an operator has to
                # be able to tell a halted session from a session with no signal.
                rejects["halted"] += int((sig_pos >= i).sum())
                break
            while ptr < len(sig_pos) and sig_pos[ptr] < i:
                ptr += 1
            if ptr >= len(sig_pos):
                break
            i = int(sig_pos[ptr])

        hi, lo = high[i], low[i]

        # 1. mark held positions on this bar's extremes (before any exit is booked)
        for p in open_pos:
            adv = lo if p["side"] > 0 else hi
            fav = hi if p["side"] > 0 else lo
            p["mae"] = max(p["mae"], p["side"] * (p["entry_px"] - adv))
            p["mfe"] = max(p["mfe"], p["side"] * (fav - p["entry_px"]))

        # 2. barrier / time exits scheduled for this second
        if open_pos:
            for p in [q for q in open_pos if q["exit_i"] == i]:
                _close_position(p, i, p["exit_px"], p["reason"])
                open_pos.remove(p)

        # 3. session loss halt, on worst-case equity = realised + adverse unrealised
        #    (recomputed after step 2 so closed positions are not counted twice)
        unreal = 0.0
        for p in open_pos:
            adv = lo if p["side"] > 0 else hi
            unreal += p["side"] * (adv - p["entry_px"]) * usd_per_point - cfg.round_trip_commission_usd
        equity_mark = equity + unreal
        if equity_mark < min_equity_mark:
            min_equity_mark = equity_mark
        if not halted and (equity_open - equity_mark) >= loss_limit:
            halted, halt_ts = True, ts[i]
            for p in list(open_pos):
                adv = lo if p["side"] > 0 else hi
                _close_position(p, i, adv - p["side"] * cfg.stop_slip, _HALT)
                open_pos.remove(p)

        # 4. entry decision for this second
        side = sig_at.get(i)
        if side is not None:
            if halted:
                rejects["halted"] += 1
            elif i - last_entry_pos < cfg.cooldown_seconds:
                rejects["cooldown"] += 1
            elif len(open_pos) >= cfg.max_concurrent_positions:
                rejects["capacity"] += 1
            elif (not cfg.allow_opposing_positions
                  and any(p["side"] != side for p in open_pos)):
                rejects["opposing"] += 1
            elif (equity - len(open_pos) * cfg.margin_per_trade_usd) < cfg.margin_per_trade_usd:
                rejects["margin"] += 1
            elif i >= last_idx:
                rejects["eod"] += 1          # no bar left to exit on
            else:
                c = cand_of_pos[i]
                d = pre[side]
                open_pos.append({
                    "entry_i": i, "side": side,
                    "entry_px": float(d["entry_px"][c]),
                    "exit_i": int(i + d["exit_off"][c]),
                    "exit_px": float(d["exit_px"][c]),
                    "reason": int(d["reason"][c]),
                    "ambiguous": bool(d["ambiguous"][c]),
                    "mae": 0.0, "mfe": 0.0,
                })
                last_entry_pos = i
                peak_margin = max(peak_margin, len(open_pos) * cfg.margin_per_trade_usd)

        i += 1

    # Anything still open at the grid end is flattened on the last close.
    for p in list(open_pos):
        _close_position(p, last_idx, close[last_idx], _FLATTEN)
        open_pos.remove(p)

    summary = {
        "session": ts[0].tz_convert(_CT).date(),
        "equity_open_usd": equity_open,
        "equity_close_usd": equity,
        "net_usd": equity - equity_open,
        "n_trades": len(trades),
        "halted": halted,
        "halt_ts": halt_ts,
        "min_equity_mark_usd": min_equity_mark,
        "intraday_dd_usd": min_equity_mark - equity_open,
        "peak_margin_usd": peak_margin,
        **{f"reject_{k}": v for k, v in rejects.items()},
    }
    return trades, summary


# ══════════════════════════════════════════════════════════════════════════════
# Top level
# ══════════════════════════════════════════════════════════════════════════════

def simulate(
    signals: pd.DataFrame,
    path: pd.DataFrame,
    cfg: SimConfig | None = None,
) -> SimResult:
    """Simulate *signals* against the 1-second *path* under the account rules in *cfg*.

    *signals* is a canonical prediction frame — ``ts``, ``direction``
    (``up``/``down``/``flat``), ``confidence`` — i.e. any ``oof_*.parquet``. Cadence is
    taken from the frame itself, so a 5s model and a 1s model both work. Labels
    (``direction_true``) are ignored if present: P&L comes only from the path.

    Sessions are independent — equity carries forward, positions never do.
    """
    cfg = cfg or SimConfig()
    grid_all = prepare_path(path)
    sig = signals.copy()
    if "ts" in sig.columns:
        sig = sig.set_index("ts")
    for col in ("direction", "confidence"):
        if col not in sig.columns:
            raise ValueError(f"signals frame requires a '{col}' column")
    try:
        sig.index = pd.to_datetime(sig.index, utc=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("signals need a 'ts' column (or index) of timestamps") from exc
    sig = sig[sig.index.notna()]

    gated = sig[sig["direction"].isin(cfg.sides)
                & (sig["confidence"].astype("float64") >= cfg.tau)]
    n_gated_total = len(gated)

    t_start, t_end = _parse_hhmm(cfg.trade_start), _parse_hhmm(cfg.trade_end)
    flat_t = _parse_hhmm(cfg.flatten_at)

    ct_dates = grid_all.index.tz_convert(_CT)
    all_trades: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    equity = cfg.starting_equity_usd

    for day, day_path in grid_all.groupby(ct_dates.date, sort=True):
        if len(day_path) < _MIN_SESSION_BARS:
            continue
        grid = _session_grid(day_path)
        gct = grid.index.tz_convert(_CT)
        # Trim the grid at the forced-flatten time: nothing after it is tradable.
        keep = gct.time <= flat_t
        if keep.sum() < _MIN_SESSION_BARS:
            continue
        grid, gct = grid[keep], gct[keep]

        in_window = (gct.time >= t_start) & (gct.time < t_end)
        day_sig = gated[(gated.index >= grid.index[0]) & (gated.index <= grid.index[-1])]
        if day_sig.empty or not in_window.any():
            sessions.append({"session": day, "equity_open_usd": equity,
                             "equity_close_usd": equity, "net_usd": 0.0, "n_trades": 0,
                             "halted": False, "halt_ts": None,
                             "min_equity_mark_usd": equity, "intraday_dd_usd": 0.0,
                             "peak_margin_usd": 0.0})
            continue

        pos = grid.index.get_indexer(day_sig.index.floor("s"), method="nearest",
                                     tolerance=pd.Timedelta("1s"))
        ok = (pos >= 0) & in_window[np.clip(pos, 0, len(grid) - 1)]
        if not ok.any():
            sessions.append({"session": day, "equity_open_usd": equity,
                             "equity_close_usd": equity, "net_usd": 0.0, "n_trades": 0,
                             "halted": False, "halt_ts": None,
                             "min_equity_mark_usd": equity, "intraday_dd_usd": 0.0,
                             "peak_margin_usd": 0.0})
            continue

        sig_pos = pos[ok].astype(np.int64)
        sig_side = np.where(day_sig["direction"].to_numpy()[ok] == "up", 1, -1).astype(np.int64)
        # One decision per second (last signal wins if a frame double-stamps).
        _, uniq = np.unique(sig_pos[::-1], return_index=True)
        keep_i = np.sort(len(sig_pos) - 1 - uniq)
        sig_pos, sig_side = sig_pos[keep_i], sig_side[keep_i]

        if equity < cfg.margin_per_trade_usd:
            logger.warning("session %s: equity %.0f below margin for one trade — flat",
                           day, equity)
            sessions.append({"session": day, "equity_open_usd": equity,
                             "equity_close_usd": equity, "net_usd": 0.0, "n_trades": 0,
                             "halted": True, "halt_ts": None,
                             "min_equity_mark_usd": equity, "intraday_dd_usd": 0.0,
                             "peak_margin_usd": 0.0})
            continue

        trades, summary = _simulate_session(grid, sig_pos, sig_side, equity, cfg)
        equity = summary["equity_close_usd"]
        all_trades.extend(trades)
        sessions.append(summary)

    trades_df = pd.DataFrame(all_trades)
    daily_df = pd.DataFrame(sessions)
    kpis = _compute_kpis(trades_df, daily_df, cfg, n_gated_signals=n_gated_total)
    return SimResult(trades=trades_df, daily=daily_df, kpis=kpis,
                     config={**asdict(cfg), "sides": list(cfg.sides)})


def _compute_kpis(
    trades: pd.DataFrame,
    daily: pd.DataFrame,
    cfg: SimConfig,
    *,
    n_gated_signals: int,
) -> dict[str, Any]:
    """Headline economics, risk and *frequency* KPIs (frequency is a first-class output).

    Drawdown uses the same definition as :func:`src.evaluation._kpi.max_drawdown` — the
    high-water mark is seeded at the starting equity, so a leading loss counts.
    """
    n_sessions = int(len(daily))
    out: dict[str, Any] = {
        "n_trades": int(len(trades)),
        "n_sessions": n_sessions,
        "n_gated_signals": int(n_gated_signals),
        "halt_days": int(daily["halted"].sum()) if n_sessions else 0,
        "starting_equity_usd": cfg.starting_equity_usd,
        "tau": cfg.tau,
        "tie_policy": cfg.tie_policy,
    }
    if n_sessions:
        out["flat_session_frac"] = float((daily["n_trades"] == 0).mean())
        out["trades_per_session"] = float(daily["n_trades"].mean())
        out["worst_intraday_dd_usd"] = float(daily["intraday_dd_usd"].min())
        out["peak_margin_usd"] = float(daily["peak_margin_usd"].max())
        out["final_equity_usd"] = float(daily["equity_close_usd"].iloc[-1])
    if trades.empty:
        out.update({"hit_rate": None, "net_expectancy_points": None, "net_usd": 0.0,
                    "total_points": 0.0, "max_drawdown_usd": 0.0,
                    "median_month_net_usd": None, "ambiguous_exit_frac": None})
        return out

    net_pts = trades["net_points"].to_numpy(dtype="float64")
    net_usd = trades["net_usd"].to_numpy(dtype="float64")
    equity_curve = cfg.starting_equity_usd + np.cumsum(net_usd)
    peak = np.maximum.accumulate(np.maximum(equity_curve, cfg.starting_equity_usd))

    out.update({
        "hit_rate": float((net_pts > 0).mean()),
        "net_expectancy_points": float(net_pts.mean()),
        "net_expectancy_usd": float(net_usd.mean()),
        "total_points": float(net_pts.sum()),
        "net_usd": float(net_usd.sum()),
        "trade_sharpe": (float(net_pts.mean() / net_pts.std(ddof=1))
                         if len(net_pts) > 1 and net_pts.std(ddof=1) > 0 else None),
        "max_drawdown_usd": float((equity_curve - peak).min()),
        "max_drawdown_frac_of_start": float((equity_curve - peak).min() / cfg.starting_equity_usd),
        "avg_hold_seconds": float(trades["hold_seconds"].mean()),
        "avg_mae_points": float(trades["mae_points"].mean()),
        "avg_mfe_points": float(trades["mfe_points"].mean()),
        "ambiguous_exit_frac": float(trades["ambiguous_bar"].mean()),
        "exit_reason_mix": {k: int(v) for k, v in
                            trades["exit_reason"].value_counts().items()},
        "side_mix": {k: int(v) for k, v in trades["side"].value_counts().items()},
    })

    months = trades.set_index("exit_ts")["net_usd"].groupby(
        lambda t: (t.year, t.month)).sum()
    out.update({
        "n_months": int(len(months)),
        "median_month_net_usd": float(months.median()),
        "mean_month_net_usd": float(months.mean()),
        "worst_month_net_usd": float(months.min()),
        "best_month_net_usd": float(months.max()),
        "profitable_month_frac": float((months > 0).mean()),
    })
    return out


def required_scaling(kpis: dict[str, Any], target_usd_per_month: float,
                     cfg: SimConfig) -> dict[str, Any]:
    """What contract size and account a monthly $ target implies at *this* measured edge.

    Deliberately linear: P&L and drawdown both scale with contracts, and margin is
    ``contracts × margin_per_contract``. Slippage does **not** scale linearly at size —
    treat the answer as a floor, not a forecast.
    """
    per_month = kpis.get("median_month_net_usd")
    if not per_month or per_month <= 0:
        return {"feasible": False, "reason": "measured monthly expectancy is not positive",
                "median_month_net_usd": per_month}
    per_contract = per_month / cfg.contracts_per_trade
    contracts = int(np.ceil(target_usd_per_month / per_contract))
    mdd_scaled = abs(kpis.get("max_drawdown_usd") or 0.0) * contracts / cfg.contracts_per_trade
    margin = contracts * cfg.margin_per_contract_usd
    return {
        "feasible": True,
        "target_usd_per_month": float(target_usd_per_month),
        "usd_per_month_per_contract": float(per_contract),
        "required_contracts": contracts,
        "required_margin_usd": float(margin),
        "scaled_max_drawdown_usd": float(mdd_scaled),
        "required_equity_usd": float(margin + mdd_scaled),
        "note": "linear scaling; ignores market impact and margin-call dynamics",
    }


def sweep_tau(
    signals: pd.DataFrame,
    path: pd.DataFrame,
    cfg: SimConfig | None = None,
    taus: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90),
) -> pd.DataFrame:
    """Frequency/economics frontier across confidence thresholds.

    This is the principled replacement for a "must trade N times" quota: pick the
    operating point off this curve on *validation* folds, then freeze it for test. A
    quota forces trades where expectancy is negative; a threshold sweep lets the model
    say "no edge here" and still lets you choose how often it speaks.
    """
    cfg = cfg or SimConfig()
    prepared = prepare_path(path)
    rows = []
    for t in taus:
        r = simulate(signals, prepared, replace(cfg, tau=float(t)))
        k = r.kpis
        rows.append({
            "tau": float(t),
            "n_trades": k["n_trades"],
            "trades_per_session": k.get("trades_per_session"),
            "flat_session_frac": k.get("flat_session_frac"),
            "hit_rate": k.get("hit_rate"),
            "net_expectancy_points": k.get("net_expectancy_points"),
            "net_usd": k.get("net_usd"),
            "median_month_net_usd": k.get("median_month_net_usd"),
            "max_drawdown_usd": k.get("max_drawdown_usd"),
            "halt_days": k.get("halt_days"),
            "ambiguous_exit_frac": k.get("ambiguous_exit_frac"),
        })
    return pd.DataFrame(rows)


def compare_tie_policies(
    signals: pd.DataFrame,
    path: pd.DataFrame,
    cfg: SimConfig | None = None,
    policies: Iterable[str] = _TIE_POLICIES,
) -> pd.DataFrame:
    """Quantify the close-only labelling bias by re-pricing identical signals.

    ``close_only`` reproduces what ``label_implied_pnl`` and the 5s bar engine can see;
    ``stop_first`` is the honest floor. The gap between them is the optimism baked into
    every KPI currently on the leaderboard.
    """
    cfg = cfg or SimConfig()
    prepared = prepare_path(path)
    rows = []
    for p in policies:
        r = simulate(signals, prepared, replace(cfg, tie_policy=str(p)))
        k = r.kpis
        rows.append({"tie_policy": p, "n_trades": k["n_trades"],
                     "hit_rate": k.get("hit_rate"),
                     "net_expectancy_points": k.get("net_expectancy_points"),
                     "net_usd": k.get("net_usd"),
                     "max_drawdown_usd": k.get("max_drawdown_usd"),
                     "ambiguous_exit_frac": k.get("ambiguous_exit_frac"),
                     "stop_exits": (k.get("exit_reason_mix") or {}).get("stop", 0),
                     "target_exits": (k.get("exit_reason_mix") or {}).get("target", 0)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# The economic boundary: this simulator's book, priced by the canonical layer
# ══════════════════════════════════════════════════════════════════════════════
#
# Division of responsibility, finished here:
#
#   this module              trade sequencing, path first-touch resolution,
#                            which price each leg actually got, position and
#                            margin capacity, the per-session loss halt,
#                            account equity over time
#   canonical/evaluation/    what a point is worth, what a round trip costs,
#   economic.py              how a leg is priced, the trade ledger, PnL
#                            aggregates and cost sensitivity
#
# The legs handed over are LIMIT legs at the prices this simulator already
# resolved. That is deliberate and is not a claim that the orders were limits:
# entry, stop and target slippage were ALREADY applied inside _resolve_exits
# when those prices were formed, and economic.execution_price applies no
# slippage to a LIMIT leg. Handing them over as MARKET legs would apply
# slippage a second time. This is exactly the case economic.py documents for
# LIMIT -- "this evaluator only prices a leg that is already known to have
# filled".
#
# Historical behaviour is preserved: the blotter's own net_usd column and the
# KPI block are unchanged. test_portfolio_sim.py::
# test_canonical_ledger_agrees_with_the_simulator_blotter is the explicit
# comparison that licenses treating the ledger as the authority -- the two
# agree to floating tolerance per trade and in aggregate, because the
# simulator's net_usd is algebraically the canonical formula:
#
#   sim:        (gross_points - round_trip_commission/(mult*qty)) * mult * qty
#   canonical:  gross_points * mult * qty - commission_per_trade * qty
#
# with as_assumptions() mapping commission_per_trade = 2 * per_side.


def _legs_for(row: dict, cfg: "SimConfig", direction: int):
    """This simulator's resolved fills, as legs that keep their real order type.

    The entry is a MARKET order: the sim takes the signal at the decision bar
    and applies ``entry_slippage_ticks`` adversely. The exit's type follows the
    reason it happened:

    ``stop``    a STOP -- the stop level is the trigger, and the recorded price
                is where it actually filled after ``stop_slippage_ticks``.
    ``target``  a LIMIT resting at the target level when no target slippage is
                assumed. With ``target_slippage_ticks > 0`` the sim is modelling
                a fill THROUGH the level, which a limit order cannot do, so the
                leg is MARKET -- the order semantics follow the fill model
                rather than being asserted over it.
    otherwise   a MARKET flatten (hold limit, session end, loss halt).

    Every leg carries ``resolved_fill_price``, so the canonical layer prices it
    verbatim and adds no slippage: this simulator already did.
    """
    from src.evaluation.economic import (
        FILL_SOURCE_PATH_SIM, Leg, LegRole, OrderType,
    )

    entry_px = float(row["entry_px"])
    exit_px = float(row["exit_px"])
    reason = str(row.get("exit_reason") or "")
    entry = Leg(
        order_type=OrderType.MARKET, role=LegRole.ENTRY,
        resolved_fill_price=entry_px, fill_source=FILL_SOURCE_PATH_SIM,
        ts=str(row.get("entry_ts") or ""),
    )
    if reason == "stop":
        exit_leg = Leg(
            order_type=OrderType.STOP, role=LegRole.EXIT_STOP,
            stop_price=entry_px - direction * cfg.stop_points,
            resolved_fill_price=exit_px, fill_source=FILL_SOURCE_PATH_SIM,
            ts=str(row.get("exit_ts") or ""),
        )
    elif reason == "target" and cfg.target_slippage_ticks == 0:
        exit_leg = Leg(
            order_type=OrderType.LIMIT, role=LegRole.EXIT_TARGET,
            limit_price=entry_px + direction * cfg.target_points,
            resolved_fill_price=exit_px, fill_source=FILL_SOURCE_PATH_SIM,
            ts=str(row.get("exit_ts") or ""),
        )
    else:
        exit_leg = Leg(
            order_type=OrderType.MARKET,
            role=LegRole.EXIT_TARGET if reason == "target" else LegRole.EXIT_TIME,
            resolved_fill_price=exit_px, fill_source=FILL_SOURCE_PATH_SIM,
            ts=str(row.get("exit_ts") or ""),
        )
    return entry, exit_leg


def trade_ledger(result: "SimResult", *, assumptions: "EconomicAssumptions | None" = None):
    """This run's book as a canonical ``economic.TradeLedger``.

    ``assumptions`` defaults to the config's own
    :meth:`SimConfig.as_assumptions`, so the ledger states the cost model the
    simulation actually used. Pass another to reprice the same fills under a
    different commission or contract size -- but not a different slippage: that
    changes which fills happen, which is :func:`execution_sensitivity`.
    """
    from src.evaluation.economic import Trade, compute_trade_ledger

    cfg = SimConfig(**result.config) if result.config else SimConfig()
    costs = assumptions or cfg.as_assumptions()
    trades = []
    for i, row in enumerate(result.trades.to_dict("records")):
        direction = 1 if str(row.get("side")) == "up" else -1
        entry, exit_leg = _legs_for(row, cfg, direction)
        trades.append(Trade(
            trade_id=str(row.get("entry_ts", i)),
            direction=direction,
            quantity=float(row.get("contracts") or cfg.contracts_per_trade),
            sequence=float(i),
            entry=entry, exit=exit_leg,
        ))
    return compute_trade_ledger(trades, costs)


def cost_sensitivity(result: "SimResult", scenarios: "Sequence[EconomicAssumptions]"):
    """FIXED-FILL sensitivity: the same fills, repriced.

    Delegates to ``economic.evaluate_cost_sensitivity``. Because this
    simulator's fills are already resolved, a scenario that varies slippage
    cannot take effect and is refused there by name -- use
    :func:`execution_sensitivity` for that question.
    """
    from src.evaluation.economic import evaluate_cost_sensitivity

    ledger = trade_ledger(result)
    return evaluate_cost_sensitivity(ledger_trades(ledger), scenarios)


def execution_sensitivity(
    signals, path, cfg: "SimConfig", scenarios: "Sequence[EconomicAssumptions]",
):
    """EXECUTION sensitivity: re-simulate under each slippage assumption.

    A different slippage assumption changes the entry price, which changes which
    stops and targets are reached, so the book is re-derived rather than
    repriced. Each scenario's per-leg slippage is pushed back into a
    ``SimConfig`` via :meth:`SimConfig.from_assumptions`, the path is
    re-simulated, and the resulting book is priced under that same scenario.

    This is the seam a higher-fidelity execution source replaces: swap the
    re-simulation for a TBBO-quoted fill model and the comparison is unchanged.
    """
    from src.evaluation.economic import evaluate_execution_sensitivity

    geometry = {k: v for k, v in asdict(cfg).items() if k not in (
        "point_value_usd", "commission_per_side_usd", "tick_size",
        "entry_slippage_ticks", "stop_slippage_ticks", "target_slippage_ticks",
    )}

    def _resimulate(scenario):
        scenario_cfg = SimConfig.from_assumptions(
            scenario, tick_size=cfg.tick_size, **geometry)
        run = simulate(signals, path, scenario_cfg)
        return ledger_trades(trade_ledger(run, assumptions=scenario))

    return evaluate_execution_sensitivity(scenarios, resimulate=_resimulate)


def ledger_trades(ledger) -> list:
    """The ``Trade`` objects behind a ledger, for fixed-fill repricing.

    Reconstructed from the priced rows, keeping each leg's real order type,
    role and fill source. The prices come back as ``resolved_fill_price``,
    which is what they are: fills that already happened. Nothing is rewritten
    as a LIMIT to suppress slippage -- a resolved fill suppresses it by being
    resolved.
    """
    from src.evaluation.economic import Leg, LegRole, OrderType, Trade

    def _leg(order_type: str, role: str, price: float, source: str, ts):
        ot = OrderType(order_type)
        kwargs = {}
        if ot is OrderType.LIMIT:
            kwargs["limit_price"] = price
        elif ot is OrderType.STOP:
            kwargs["stop_price"] = price
        return Leg(order_type=ot, role=LegRole(role), resolved_fill_price=price,
                   fill_source=source, ts=ts, **kwargs)

    return [
        Trade(
            trade_id=r.trade_id, direction=r.direction, quantity=r.quantity,
            sequence=r.sequence,
            entry=_leg(r.entry_order_type, LegRole.ENTRY.value, r.entry_price,
                       r.entry_fill_source, r.entry_ts),
            exit=_leg(r.exit_order_type, r.exit_role, r.exit_price,
                      r.exit_fill_source, r.exit_ts),
        )
        for r in ledger.trades
    ]
