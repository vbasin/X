"""Order-flow state: aggression, absorption, acceptance and rejection.

The naive construction — "more buy aggression means price goes up" — is
contradicted by the data. Patzelt & Bouchaud (*Phys. Rev. E* 97, 012304, 2018)
show the OFI-to-return relation is **arc-shaped**: extreme one-sided imbalance
coincides with price being *pinned*, and prices move when local flow is
balanced. That is exactly what a trader calls absorption, and it means aggression
magnitude alone is not a directional feature.

So every measure here is aggression **relative to the price response it
produced**:

* **acceptance** — large signed flow *with* a proportionate move in its direction
* **absorption** — large signed flow *without* the move: someone is filling it
* **rejection** — flow and move in opposite directions, usually into a wick

The quantitative form is a residual against a rolling impact coefficient, which
is Kyle's lambda estimated online: Cont, Kukanov & Stoikov (*J. Financial
Econometrics* 12(1), 2014) establish that price change is linear in OFI with a
slope inversely proportional to depth, so the residual is well-defined and the
raw OFI must be depth-normalised before it is comparable across regimes.

Trade classification uses the TBBO ``side`` field, which is the **aggressor**:
``B`` = buyer lifted the offer (positive delta), ``A`` = seller hit the bid
(negative delta). This matches ``src/features/order_flow.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.tape import print_ofi

logger = logging.getLogger("vtech.features.flow_state")

PREFIX = "fs_"
BUY, SELL = "B", "A"

# Signed in the absolute frame, so a short must see them negated for the model to
# treat both sides as one problem. Magnitude-only columns (``*_per_body``,
# ``*_large_share``) and the absolute-price ``fs_last_reversal_level`` are
# deliberately absent.
DIRECTIONAL_SUFFIXES: tuple[str, ...] = (
    "delta_1m", "delta_5m", "delta_ratio_1m", "delta_ratio_5m",
    "ofi_1m", "ofi_5m", "absorption_1m", "absorption_5m",
    "impact_efficiency_1m", "impact_efficiency_5m",
    "vwap_dist_r", "vwap_velocity_r", "vwap_accel_r",
    "vwap_velocity_5m_r", "vwap_velocity_15m_r",
    "ema6_dist_r", "ema6_slope_r", "vwap_ema_spread_r",
    "ema20_slope_r", "ema60_slope_r",
    "ema20_vwap_diff_r", "ema60_vwap_diff_r",
    "ema6_slope_vwap_diff_r", "ema20_slope_vwap_diff_r", "ema60_slope_vwap_diff_r",
    "ema6_slope_vwap_diff_accel_r", "ema20_slope_vwap_diff_accel_r",
    "ema60_slope_vwap_diff_accel_r",
    "absorb_continue", "accept_continue", "absorb_reversal", "reject_reversal",
    "flow_score", "flow_score_5bar",
)
DIRECTIONAL: tuple[str, ...] = tuple(PREFIX + s for s in DIRECTIONAL_SUFFIXES)


@dataclass(frozen=True)
class FlowConfig:
    """Windows and thresholds. Windows are seconds on the 1s grid."""

    fast_window: int = 60             # "last 1 minute from now"; also the flow_score window
    slow_window: int = 300            # "last 5 minutes from now" = 5 x fast_window
    impact_window: int = 900          # rolling lambda estimation
    large_order_size: int = 20        # contracts in a single TBBO print
    ema_bars_1m: int = 6               # short EMA over 1-minute closes
    ema_bars_20: int = 20              # medium EMA over 1-minute closes
    ema_bars_60: int = 60              # long EMA over 1-minute closes
    vwap_vel_window: int = 60
    vwap_vel_window_5m: int = 300      # 5-bar (5-minute) VWAP slope window
    vwap_vel_window_15m: int = 900     # 15-bar (15-minute) VWAP slope window
    min_impact_obs: int = 120
    max_reversal_scale: float = 10.0  # caps wick/body blowup on near-zero-body bars

    def __post_init__(self) -> None:
        for n in ("fast_window", "slow_window", "impact_window", "vwap_vel_window",
                  "vwap_vel_window_5m", "vwap_vel_window_15m"):
            if getattr(self, n) < 2:
                raise ValueError(f"{n} must be >= 2")
        for n in ("ema_bars_1m", "ema_bars_20", "ema_bars_60"):
            if getattr(self, n) < 1:
                raise ValueError(f"{n} must be >= 1")
        if self.large_order_size < 1:
            raise ValueError("large_order_size must be >= 1")
        if self.max_reversal_scale <= 0:
            raise ValueError("max_reversal_scale must be positive")


# ══════════════════════════════════════════════════════════════════════════════
# Candle location of a print — the basis of the acceptance/absorption taxonomy
# ══════════════════════════════════════════════════════════════════════════════

BODY, UPPER_WICK, LOWER_WICK = 0, 1, 2


def candle_location(price: np.ndarray, o: np.ndarray, h: np.ndarray,
                    l: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Where each print sits in its bar: body, upper wick or lower wick.

    The body spans ``[min(O,C), max(O,C)]`` regardless of direction; anything
    above is upper wick and anything below is lower wick.
    """
    body_lo = np.minimum(o, c)
    body_hi = np.maximum(o, c)
    loc = np.full(len(price), BODY, dtype=np.int8)
    loc[price > body_hi] = UPPER_WICK
    loc[price < body_lo] = LOWER_WICK
    return loc


def classify_large_prints(trades: pd.DataFrame, bars: pd.DataFrame,
                          cfg: FlowConfig) -> pd.DataFrame:
    """Tag every large print with its auction meaning, at print granularity.

    *trades* needs ``price``, ``size``, ``side`` and a UTC index; *bars* needs
    ``open/high/low/close`` indexed by bar-end, and each trade is matched to the
    bar it closed into. Four states follow from aggressor side, where the print
    landed relative to the body's **midpoint** and the body's own edges, and the
    candle's direction:

    ==========================  =========================  ===========================
    aggressor                   location (up candle)       reading
    ==========================  =========================  ===========================
    buy                         body                        continuation: **acceptance**
    sell                        bottom half (low..mid)      continuation: **absorption**
    buy                         upper wick (> body high)    reversal: **absorption**
    sell                        upper wick (> body high)    reversal: **rejection**
    ==========================  =========================  ===========================

    Down candles mirror: aggressor sides swap and "upper" becomes "lower".

    Continuation contributions (``accept_continue``, ``absorb_continue``) are
    scaled by ``body_range / full_range`` — a continuation read on a candle that
    is almost all body means more than one on a candle that is mostly wick.
    Reversal contributions (``absorb_reversal``, ``reject_reversal``) are scaled
    by ``wick_range / body_range``, capped at ``cfg.max_reversal_scale`` to keep a
    near-zero body from producing an unbounded ratio. Every qualifying print's
    contract size is divided by ``cfg.large_order_size`` rather than by traded
    volume, so a large print reads the same whether the bar's volume was thin or
    heavy.
    """
    cols = ["ts", "accept_continue", "absorb_continue", "absorb_reversal",
           "reject_reversal", "reversal_price", "size"]
    if trades.empty or bars.empty:
        return pd.DataFrame(columns=cols)
    big = trades[trades["size"].to_numpy() >= cfg.large_order_size]
    if big.empty:
        return pd.DataFrame(columns=cols)

    # Bars are labelled by their start, covering [start, start+1) (tape_to_path
    # floors to the second); the bar actually containing a print is found by a
    # floor match, not by taking the next label. That bar is not yet complete
    # at its own timestamp, so the print is only visible from the *next* bar's
    # row onward -- never fabricate a still-forming bar's OHLC as known.
    starts = bars.index.to_numpy()
    own_bar = np.searchsorted(starts, big.index.to_numpy(), side="right") - 1
    ok = (own_bar >= 0) & (own_bar + 1 < len(starts))
    if not ok.any():
        return pd.DataFrame(columns=cols)
    b = bars.iloc[own_bar[ok]]
    vis_ts = bars.index[own_bar[ok] + 1]

    price = big["price"].to_numpy(dtype="float64")[ok]
    size = big["size"].to_numpy(dtype="float64")[ok]
    is_buy = (big["side"].to_numpy()[ok] == BUY)
    o = b["open"].to_numpy(dtype="float64")
    h = b["high"].to_numpy(dtype="float64")
    lo = b["low"].to_numpy(dtype="float64")
    c = b["close"].to_numpy(dtype="float64")
    up = c >= o

    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    body_mid = 0.5 * (body_hi + body_lo)
    full_range = h - lo
    body_range = body_hi - body_lo
    upper_wick_range = h - body_hi
    lower_wick_range = body_lo - lo

    in_body = (price >= body_lo) & (price <= body_hi)
    bottom_half = (price >= lo) & (price <= body_mid)      # low .. body midpoint
    top_half = (price >= body_mid) & (price <= h)          # body midpoint .. high
    in_upper_wick = price > body_hi
    in_lower_wick = price < body_lo

    # An up candle reads long; a down candle is the mirror, so fold direction in.
    accept_continue_hit = np.where(up, is_buy & in_body, (~is_buy) & in_body)
    absorb_continue_hit = np.where(up, (~is_buy) & bottom_half, is_buy & top_half)
    absorb_reversal_hit = np.where(up, is_buy & in_upper_wick, (~is_buy) & in_lower_wick)
    reject_reversal_hit = np.where(up, (~is_buy) & in_upper_wick, is_buy & in_lower_wick)
    is_reversal = absorb_reversal_hit | reject_reversal_hit

    cont_scale = np.divide(body_range, full_range,
                           out=np.zeros_like(body_range), where=full_range > 0)
    wick_for_reversal = np.where(up, upper_wick_range, lower_wick_range)
    rev_scale = np.minimum(
        np.divide(wick_for_reversal, body_range,
                 out=np.zeros_like(body_range), where=body_range > 0),
        cfg.max_reversal_scale)

    norm_size = size / cfg.large_order_size
    sign = np.where(up, 1.0, -1.0)

    out = pd.DataFrame({
        # Tag at the bar *after* the one the print closed into, so a consumer
        # joining on the bar grid never sees a print before its bar completed.
        "ts": vis_ts,
        # Continuation follows candle direction; reversal opposes it, matching
        # the sign convention the model already expects (positive = bullish).
        "accept_continue": np.where(accept_continue_hit, sign * norm_size * cont_scale, 0.0),
        "absorb_continue": np.where(absorb_continue_hit, sign * norm_size * cont_scale, 0.0),
        "absorb_reversal": np.where(absorb_reversal_hit, -sign * norm_size * rev_scale, 0.0),
        "reject_reversal": np.where(reject_reversal_hit, -sign * norm_size * rev_scale, 0.0),
        # Raw execution price of reversal prints only, carried for fs_last_reversal_level.
        "reversal_price": np.where(is_reversal, price, np.nan),
        "size": size,
    })
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Bar-level flow statistics
# ══════════════════════════════════════════════════════════════════════════════

def rolling_impact_residual(dprice: pd.Series, ofi: pd.Series,
                            window: int, min_obs: int) -> pd.Series:
    """Absorption measure: flow-implied move minus realised move.

    Fits ``dprice ~ beta * ofi`` on a trailing window (Kyle's lambda estimated
    online, per Cont-Kukanov-Stoikov) and returns ``beta * ofi - dprice``.
    Positive means the move was **smaller** than the flow implied — the flow was
    absorbed. Negative means price ran further than the flow justified.

    Both moments are trailing, so the coefficient at bar *t* uses only bars
    strictly before *t*.
    """
    x, y = ofi.astype("float64"), dprice.astype("float64")
    xy = (x * y).rolling(window, min_periods=min_obs).mean().shift(1)
    xx = (x * x).rolling(window, min_periods=min_obs).mean().shift(1)
    beta = xy / xx.replace(0.0, np.nan)
    return beta * x - y


def build_flow_state(
    bars: pd.DataFrame,
    trades: pd.DataFrame | None,
    cfg: FlowConfig | None = None,
    *,
    r_value: float = 1.0,
    warmup_close_1m: pd.Series | None = None,
) -> pd.DataFrame:
    """``fs_``-prefixed flow state on the *bars* grid.

    *bars* must carry ``open/high/low/close/volume``. *trades* is the raw
    print stream: when supplied, it grounds the causal current price, VWAP
    and signed flow (delta/OFI) directly in individual prints -- each is read
    as of the last print at or before a row's own timestamp -- and drives the
    large-order taxonomy. Omit it and price/VWAP fall back to the bar's own
    OHLC, delta/OFI fall back to ``bars["delta"]``/``bars["ofi"]`` if present
    (both floor-labelled, so diluted-leaky the same way the bar's own close
    is) or zero, and the taxonomy columns come back zero.
    """
    cfg = cfg or FlowConfig()
    if bars.empty:
        return pd.DataFrame()
    idx = bars.index
    o = bars["open"].to_numpy(dtype="float64")
    v = bars["volume"].to_numpy(dtype="float64")
    out = pd.DataFrame(index=idx)

    # ── causal current price and VWAP, from individual prints ───────────────
    # A bar's own close can carry a print up to ~1s after this row's own
    # timestamp (tape_to_path floor-labels bars by their start, so bar T spans
    # [T, T+1)). The causal price is the last print at or before this row's
    # timestamp; VWAP is the true trade-weighted average (cumsum(price*size)
    # over individual prints), not the (h+l+c)/3 bar proxy -- both the proxy
    # and the bar-close comparator bake in that same forward-looking print.
    # Both fall back to the bar's own close/an OHLC proxy -- not causally
    # clean, but the best available -- before this session's first print or
    # when *trades* was not supplied.
    bar_close = bars["close"].to_numpy(dtype="float64")
    if trades is not None and not trades.empty:
        t_ts = trades.index.to_numpy()
        t_px = trades["price"].to_numpy(dtype="float64")
        t_sz = trades["size"].to_numpy(dtype="float64")
        as_of = np.searchsorted(t_ts, idx.to_numpy(), side="right") - 1
        seen = as_of >= 0
        safe_pos = np.maximum(as_of, 0)
        c = np.where(seen, t_px[safe_pos], bar_close)
        cum_pv = np.cumsum(t_px * t_sz)
        cum_v = np.maximum(np.cumsum(t_sz), 1e-9)
        vwap = pd.Series(np.where(seen, cum_pv[safe_pos] / cum_v[safe_pos], np.nan),
                         index=idx)
    else:
        c = bar_close
        tp = (bars["high"].to_numpy(dtype="float64")
             + bars["low"].to_numpy(dtype="float64") + c) / 3.0
        vwap = pd.Series(np.cumsum(tp * v) / np.maximum(np.cumsum(v), 1e-9), index=idx)
    s_close = pd.Series(c, index=idx)

    # ── signed flow, depth-normalised, from individual prints ───────────────
    # fs_delta_*/fs_ofi_*/fs_delta_ratio_*/fs_absorption_*/fs_impact_efficiency_*
    # used to be rolling sums over bars["delta"]/bars["ofi"], which come from
    # the same floor-labelled per-second aggregation as tape_to_path: bar T's
    # delta/ofi can include a print up to ~1s after this row's own timestamp,
    # diluted across the window rather than concentrated in one row the way
    # fs_clv's leak was, but present at every row of every window. Rebuilt
    # here as the window-diff of a running total over individual prints: the
    # total as of the last print at or before this row, minus the total as of
    # the last print at or before (this row - window) -- both ends are
    # therefore bounded by prints strictly at or before their own instant.
    if trades is not None and not trades.empty:
        t_side = np.where(trades["side"].to_numpy() == BUY, 1.0, -1.0)
        signed_sz = t_sz * t_side
        cum_delta = np.cumsum(signed_sz)
        has_book = {"bid_px", "ask_px", "bid_sz", "ask_sz"}.issubset(trades.columns)
        ofi_print = print_ofi(trades) if has_book else signed_sz
        cum_ofi = np.cumsum(np.nan_to_num(ofi_print, nan=0.0))
        cum_vol = np.cumsum(t_sz)

        def _running_total(cum: np.ndarray, at: np.ndarray) -> np.ndarray:
            pos = np.searchsorted(t_ts, at, side="right") - 1
            return np.where(pos >= 0, cum[np.maximum(pos, 0)], 0.0)

        now_ts = idx.to_numpy()
        now_delta, now_ofi, now_vol = (_running_total(cum_delta, now_ts),
                                       _running_total(cum_ofi, now_ts),
                                       _running_total(cum_vol, now_ts))
        for w, tag in ((cfg.fast_window, "1m"), (cfg.slow_window, "5m")):
            back_ts = (idx - pd.Timedelta(seconds=w)).to_numpy()
            back_delta = _running_total(cum_delta, back_ts)
            back_ofi = _running_total(cum_ofi, back_ts)
            back_vol = _running_total(cum_vol, back_ts)
            out[f"fs_delta_{tag}"] = now_delta - back_delta
            out[f"fs_ofi_{tag}"] = now_ofi - back_ofi
            vol_w = pd.Series(now_vol - back_vol, index=idx)
            # Normalising by traded volume makes the ratio comparable across
            # regimes; raw delta is dominated by the activity profile of the
            # session.
            out[f"fs_delta_ratio_{tag}"] = out[f"fs_delta_{tag}"] / vol_w.replace(0, np.nan)
            body = (s_close - pd.Series(o, index=idx).shift(w - 1)).abs()
            out[f"fs_delta_per_body_{tag}"] = (
                out[f"fs_delta_{tag}"].abs() / body.replace(0, np.nan).clip(lower=0.25))
    else:
        delta = bars["delta"].to_numpy(dtype="float64") if "delta" in bars else np.zeros(len(idx))
        ofi = bars["ofi"].to_numpy(dtype="float64") if "ofi" in bars else delta
        s_delta, s_ofi = pd.Series(delta, index=idx), pd.Series(ofi, index=idx)
        for w, tag in ((cfg.fast_window, "1m"), (cfg.slow_window, "5m")):
            vol_w = pd.Series(v, index=idx).rolling(w, min_periods=1).sum()
            out[f"fs_delta_{tag}"] = s_delta.rolling(w, min_periods=1).sum()
            out[f"fs_delta_ratio_{tag}"] = out[f"fs_delta_{tag}"] / vol_w.replace(0, np.nan)
            out[f"fs_ofi_{tag}"] = s_ofi.rolling(w, min_periods=1).sum()
            body = (s_close - pd.Series(o, index=idx).shift(w - 1)).abs()
            out[f"fs_delta_per_body_{tag}"] = (
                out[f"fs_delta_{tag}"].abs() / body.replace(0, np.nan).clip(lower=0.25))

    # ── absorption vs acceptance ─────────────────────────────────────────────
    for w, tag in ((cfg.fast_window, "1m"), (cfg.slow_window, "5m")):
        resid = rolling_impact_residual(s_close.diff(w), out[f"fs_ofi_{tag}"],
                                        cfg.impact_window, cfg.min_impact_obs)
        out[f"fs_absorption_{tag}"] = resid / max(r_value, 1e-9)
        # Arc-shape control (Patzelt & Bouchaud): |flow| alone is not direction.
        # Efficiency separates a move that flow paid for from one it did not.
        out[f"fs_impact_efficiency_{tag}"] = (
            s_close.diff(w) / out[f"fs_ofi_{tag}"].replace(0, np.nan))

    # ── VWAP velocity and acceleration (vwap itself built above) ─────────────
    r_safe = max(r_value, 1e-9)
    out["fs_vwap_dist_r"] = (s_close - vwap) / r_safe
    vel = vwap.diff(cfg.vwap_vel_window)
    out["fs_vwap_velocity_r"] = vel / r_safe
    out["fs_vwap_accel_r"] = vel.diff(cfg.vwap_vel_window) / r_safe
    out["fs_vwap_velocity_5m_r"] = vwap.diff(cfg.vwap_vel_window_5m) / r_safe
    out["fs_vwap_velocity_15m_r"] = vwap.diff(cfg.vwap_vel_window_15m) / r_safe

    # ── EMAs on 1-minute closes, held on the bar grid ────────────────────────
    # ema6 is the legacy short EMA, so its VWAP value-diff keeps the original
    # fs_vwap_ema_spread_r name; ema20/ema60 follow the fs_ema{N}_vwap_diff_r
    # pattern. Each EMA also gets a slope-vs-VWAP-slope diff and that diff's
    # own 60s slope (its acceleration), all against the same 60s VWAP slope.
    m1 = s_close.resample("1min", label="right", closed="left").last().dropna()
    if warmup_close_1m is not None and len(warmup_close_1m):
        warm = pd.to_numeric(warmup_close_1m, errors="coerce").dropna()
        warm = warm[~warm.index.duplicated(keep="last")].sort_index()
        if len(m1):
            warm = warm.loc[warm.index < m1.index.min()]
        if len(warm):
            m1 = pd.concat([warm, m1])
    ema_specs = ((cfg.ema_bars_1m, "6", "fs_vwap_ema_spread_r"),
                 (cfg.ema_bars_20, "20", "fs_ema20_vwap_diff_r"),
                 (cfg.ema_bars_60, "60", "fs_ema60_vwap_diff_r"))
    for span, tag, vwap_diff_col in ema_specs:
        # ewm sees pre-session minutes; values are held only on the cash grid.
        ema = m1.ewm(span=span, adjust=False).mean().reindex(idx, method="ffill")
        ema_slope = ema.diff(cfg.fast_window)
        slope_diff = ema_slope - vel
        if tag == "6":
            out["fs_ema6_dist_r"] = (s_close - ema) / r_safe
        out[f"fs_ema{tag}_slope_r"] = ema_slope / r_safe
        out[vwap_diff_col] = (ema - vwap) / r_safe
        out[f"fs_ema{tag}_slope_vwap_diff_r"] = slope_diff / r_safe
        out[f"fs_ema{tag}_slope_vwap_diff_accel_r"] = slope_diff.diff(cfg.fast_window) / r_safe

    # ── large-print taxonomy: continuation vs reversal classification ────────
    tax_cols = ["accept_continue", "absorb_continue", "absorb_reversal", "reject_reversal"]
    if trades is not None and not trades.empty:
        tax = classify_large_prints(trades, bars, cfg)
    else:
        tax = pd.DataFrame(columns=["ts", *tax_cols, "reversal_price", "size"])

    if not tax.empty:
        # Multiple qualifying prints in the same bar are summed, not overwritten.
        signed = (tax.groupby("ts")[[*tax_cols, "size"]].sum()
                     .reindex(idx).fillna(0.0))
        rev_price = (tax.dropna(subset=["reversal_price"])
                        .groupby("ts")["reversal_price"].last()
                        .reindex(idx))
    else:
        signed = pd.DataFrame(0.0, index=idx, columns=[*tax_cols, "size"])
        rev_price = pd.Series(np.nan, index=idx)

    raw_combined = signed[tax_cols].sum(axis=1)
    for col in tax_cols:
        # Features 1-4: causal 1-minute read of each category, diagnostic only —
        # not fed to the model directly (fs_flow_score/_5bar below are).
        out[f"fs_{col}"] = signed[col].rolling(cfg.fast_window, min_periods=1).sum()
    # Feature 5: net signed flow quality over the trailing minute. Model input.
    out["fs_flow_score"] = raw_combined.rolling(cfg.fast_window, min_periods=1).sum()
    # Feature 6: the same signal over the trailing 5 minutes (slow_window = 5 x
    # fast_window) — the aggregate of the last 5 one-minute fs_flow_score reads.
    # Model input.
    out["fs_flow_score_5bar"] = raw_combined.rolling(cfg.slow_window, min_periods=1).sum()

    for w, tag in ((cfg.fast_window, "1m"), (cfg.slow_window, "5m")):
        vol_w = pd.Series(v, index=idx).rolling(w, min_periods=1).sum().replace(0, np.nan)
        out[f"fs_large_share_{tag}"] = (
            signed["size"].rolling(w, min_periods=1).sum() / vol_w)

    # Feature 7: price of the most recent reversal print, expiring after
    # slow_window seconds (5 minutes / 5 bars). Metadata for future entry/stop
    # logic — not a model input.
    ts_ns = idx.values.astype("int64")
    last_ts = np.where(rev_price.notna().to_numpy(), ts_ns, np.nan)
    last_ts = pd.Series(last_ts, index=idx).ffill().to_numpy()
    age_s = (ts_ns - last_ts) / 1e9
    level = rev_price.ffill().to_numpy()
    out["fs_last_reversal_level"] = np.where(
        np.isnan(last_ts) | (age_s > cfg.slow_window), np.nan, level)

    return out.replace([np.inf, -np.inf], np.nan)
