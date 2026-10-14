"""Candle structure features: body/range ratio, CLV, volatility regime.

v3: Uses bar_windows(). Removed cs_5m multitimeframe (redundant with 128-bar
    sequence). Removed cs_clv_bullish_count (redundant with cs_clv_ma).
v5.4: Restored 5m/10m CLV (SHAP: intermediate CLV horizons carry side);
    added cs_range_ma_6; added 15s/30s CLV from TBBO prints.
v5.15: 5s-native OHLCV (`cs_open`/`cs_high`/`cs_low`/`cs_close`/`cs_volume`)
    plus `cs_range` / `cs_direction` / `cs_return`. `cs_close` is the
    feature-catalog twin of `_close` (ADR-009 / ADR-069).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.bar_windows import bars, bars_m, get_timestep

# 5m/10m emit CLV only (requested). 15m/30m keep body_range + CLV (compat).
_DEFAULT_MULTITF = ("5min", "10min", "15min", "30min")
_MULTITF_BODY_AND_CLV = frozenset({"15min", "30min"})
_DEFAULT_SUBMIN = ("15s", "30s")
# Columns the CS parquet patch writes (v5.4 new cols + 15m/30m rewrite).
CS_PATCH_COLS = (
    "cs_range_ma_6",
    "cs_5m_clv",
    "cs_10m_clv",
    "cs_15m_clv",
    "cs_15m_body_range",
    "cs_30m_clv",
    "cs_30m_body_range",
    "cs_15s_clv",
    "cs_30s_clv",
)

# 5s-native candle columns (not 1-min; do not stale-forward with slow cs_*).
CS_5S_OHLC_COLS: tuple[str, ...] = (
    "cs_open",
    "cs_high",
    "cs_low",
    "cs_close",
    "cs_volume",
    "cs_range",
    "cs_direction",
    "cs_return",
)

# Server-side 5s OHLCV for parquet patch (no raw tick pull).
CS_5S_OHLC_SQL = """
SELECT
    time_bucket(%s::interval, ts_event) AS ts,
    first(price, ts_event) AS open,
    max(price) AS high,
    min(price) AS low,
    last(price, ts_event) AS close,
    COALESCE(sum(size), 0) AS volume
FROM databento.tbbo
WHERE ts_recv >= %s AND ts_recv < %s
  AND ts_event >= %s AND ts_event < %s
  AND symbol = %s
GROUP BY 1
"""


def _as_utc_index(idx) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(idx)
    if out.tz is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def ohlcv_5s_from_tbbo(tbbo: pd.DataFrame, timestep: str) -> pd.DataFrame:
    """Left-labeled OHLCV from TBBO prints (`price` / `size`)."""
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return empty
    ts_col = "ts_event" if "ts_event" in tbbo.columns else (
        "ts_recv" if "ts_recv" in tbbo.columns else None
    )
    if ts_col is None:
        return empty
    ts = pd.to_datetime(tbbo[ts_col], utc=True)
    px = pd.Series(tbbo["price"].to_numpy(dtype="float64"), index=ts, name="price")
    px = px.sort_index()
    if px.empty:
        return empty
    ohlc = px.resample(timestep).ohlc()
    if "size" in tbbo.columns:
        sz = pd.Series(tbbo["size"].to_numpy(dtype="float64"), index=ts).sort_index()
        ohlc["volume"] = sz.resample(timestep).sum()
    else:
        ohlc["volume"] = px.resample(timestep).count().astype("float64")
    return ohlc


def compute_5s_ohlc(
    tbbo: pd.DataFrame | None = None,
    grid: pd.DatetimeIndex | None = None,
    timestep: str | None = None,
    close: pd.Series | None = None,
    ohlcv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """5s candle features on *grid* (parquet timestep).

    ``cs_close`` copies *close* when given (the existing ``_close`` series).
    Bars with no prints are a flat candle at that close, ``cs_volume`` = 0.
    """
    timestep = timestep or get_timestep()
    src = None
    if ohlcv is not None and not ohlcv.empty:
        src = ohlcv.copy()
    else:
        src = ohlcv_5s_from_tbbo(tbbo if tbbo is not None else pd.DataFrame(), timestep)

    if src is None or src.empty:
        if grid is None:
            return pd.DataFrame(columns=list(CS_5S_OHLC_COLS))
        out = pd.DataFrame(
            index=_as_utc_index(grid), columns=list(CS_5S_OHLC_COLS), dtype="float32",
        )
        if close is not None:
            c = pd.Series(close)
            c.index = _as_utc_index(c.index)
            out["cs_close"] = pd.to_numeric(c, errors="coerce").reindex(out.index).astype(np.float32)
        return out

    src.index = _as_utc_index(src.index)
    src = src.loc[~src.index.duplicated(keep="last")].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in src.columns:
            src[col] = np.nan
        src[col] = pd.to_numeric(src[col], errors="coerce")

    grid_i = _as_utc_index(grid) if grid is not None else src.index
    open_ = src["open"].reindex(grid_i)
    high = src["high"].reindex(grid_i)
    low = src["low"].reindex(grid_i)
    vol = src["volume"].reindex(grid_i).fillna(0.0)
    empty = open_.isna()

    if close is not None:
        c = pd.Series(close)
        c.index = _as_utc_index(c.index)
        cs_close = pd.to_numeric(c, errors="coerce").reindex(grid_i)
    else:
        cs_close = src["close"].reindex(grid_i)
    cs_close = cs_close.ffill()

    open_ = open_.where(~empty, cs_close)
    high = high.where(~empty, cs_close)
    low = low.where(~empty, cs_close)
    high = pd.concat({"h": high, "c": cs_close}, axis=1).max(axis=1, skipna=True)
    low = pd.concat({"l": low, "c": cs_close}, axis=1).min(axis=1, skipna=True)

    rng = high - low
    direction = np.sign(cs_close - open_)
    ret = (cs_close - open_) / open_.replace(0, np.nan)

    out = pd.DataFrame(
        {
            "cs_open": open_,
            "cs_high": high,
            "cs_low": low,
            "cs_close": cs_close,
            "cs_volume": vol,
            "cs_range": rng,
            "cs_direction": direction,
            "cs_return": ret,
        },
        index=grid_i,
    )
    return out[list(CS_5S_OHLC_COLS)].astype(np.float32)


def align_to_grid(features: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Stale-forward slower CS values onto a 5s feature index.

    A right-labeled 5m (10m / 15s / 30s) close is held constant on every
    subsequent 5s bar until the next close of that bucket — 60 / 120 / 3 / 6
    five-second bars respectively.
    """
    if features is None or features.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(grid))
    src = features.copy()
    idx = src.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, utc=True)
        src.index = idx
    elif idx.tz is None:
        src.index = idx.tz_localize("UTC")
    else:
        src.index = idx.tz_convert("UTC")
    src = src.loc[~src.index.duplicated(keep="last")]
    grid = pd.DatetimeIndex(grid)
    if grid.tz is None:
        grid = grid.tz_localize("UTC")
    else:
        grid = grid.tz_convert("UTC")
    return src.reindex(grid, method="ffill")


def build_cs_patch_frame(
    candles_1m: pd.DataFrame,
    tbbo: pd.DataFrame,
    grid: pd.DatetimeIndex,
    timestep: str | None = None,
) -> pd.DataFrame:
    """Build the five new CS columns aligned (ffill) to *grid*."""
    timestep = timestep or get_timestep()
    parts: list[pd.DataFrame] = []
    if candles_1m is not None and not candles_1m.empty:
        base = compute(candles_1m)
        if "cs_range_ma_6" in base.columns:
            parts.append(base[["cs_range_ma_6"]])
        mtf = compute_multitimeframe(candles_1m)
        keep = [c for c in CS_PATCH_COLS if c in mtf.columns]
        if keep:
            parts.append(mtf[keep])
    if tbbo is not None and not tbbo.empty:
        sub = compute_subminute_clv(tbbo, timestep=timestep, grid=grid)
        keep = [c for c in ("cs_15s_clv", "cs_30s_clv") if c in sub.columns]
        if keep:
            parts.append(sub[keep])
    if not parts:
        return pd.DataFrame(index=pd.DatetimeIndex(grid), columns=list(CS_PATCH_COLS))
    raw = parts[0]
    for extra in parts[1:]:
        raw = raw.join(extra, how="outer")
    out = align_to_grid(raw, grid)
    # Second ffill after union (1m and 5s indexes merge with gaps).
    out = out.ffill()
    for col in CS_PATCH_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out[list(CS_PATCH_COLS)].astype(np.float32)


def _clv_from_ohlc(ohlc: pd.DataFrame) -> pd.Series:
    """Close location value ∈ [−1, +1]; NaN when high == low."""
    rng = (ohlc["high"] - ohlc["low"]).replace(0, np.nan)
    return (2.0 * ohlc["close"] - ohlc["high"] - ohlc["low"]) / rng


def compute(
    candles: pd.DataFrame,
) -> pd.DataFrame:
    """Compute candle structure features from OHLCV candle data.

    Parameters
    ----------
    candles : DataFrame with columns: open, high, low, close, volume.
    """
    # CLV rolling windows: ~20s, 30s, 50s, 100s in real time
    w20s = bars(20)
    w30s = bars(30)
    w50s = bars(50)
    w100s = bars(100)
    clv_rolling = [w20s, w30s, w50s, w100s]

    w1m  = bars_m(1)
    w5m  = bars_m(5)
    w10m = bars_m(10)

    df = candles.copy()
    features = pd.DataFrame(index=df.index)

    candle_range = df["high"] - df["low"]
    body = (df["close"] - df["open"]).abs()

    features["cs_body_range_ratio"] = body / candle_range.replace(0, np.nan)
    features["cs_body_direction"] = np.sign(df["close"] - df["open"])

    # Close Location Value (CLV): where close sits in the bar's range
    features["cs_clv"] = (2 * df["close"] - df["high"] - df["low"]) / candle_range.replace(0, np.nan)

    # Rolling CLV
    for w in clv_rolling:
        features[f"cs_clv_ma_{w}"] = features["cs_clv"].rolling(w).mean()

    # Upper / lower shadow ratios
    features["cs_upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range.replace(0, np.nan)
    features["cs_lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range.replace(0, np.nan)

    # Volatility regime: rolling range relative to longer-term.
    # Suffix is the rolling window on the 1-min candle series (same as 12/60/120).
    features["cs_range_ma_6"] = candle_range.rolling(6).mean()
    for w in (w1m, w5m, w10m):
        features[f"cs_range_ma_{w}"] = candle_range.rolling(w).mean()
    features["cs_vol_regime"] = candle_range / features[f"cs_range_ma_{w5m}"].replace(0, np.nan)

    # Range expansion/contraction
    features["cs_range_chg"] = candle_range / candle_range.shift(1).replace(0, np.nan) - 1

    return features.astype(np.float32)


def compute_multitimeframe(
    candles_1m: pd.DataFrame,
    timeframes: list[str] | None = None,
) -> pd.DataFrame:
    """Resample 1-min candles and compute CLV (and body/range on 15m/30m).

    Default timeframes: 5m, 10m, 15m, 30m. ``label="right"`` so a still-forming
    interval is not visible until it closes (same contract as 15m/30m).
    """
    if timeframes is None:
        timeframes = list(_DEFAULT_MULTITF)

    features = pd.DataFrame(index=candles_1m.index)

    for tf in timeframes:
        label = tf.replace("min", "m")
        # label="right" stamps each bar at its interval END, so the ffill below
        # only ever exposes a *completed* prior interval to later bars. The pandas
        # default (label="left") stamped the whole window at its start, leaking the
        # still-forming interval's high/low/close into its own opening minutes.
        # Bin membership is unchanged (still left-closed [t, t+tf)); only the label
        # — and therefore the first bar at which each value becomes visible — moves.
        ohlcv = candles_1m.resample(tf, label="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])
        if ohlcv.empty:
            continue

        keep = [f"cs_{label}_clv"]
        ohlcv[f"cs_{label}_clv"] = _clv_from_ohlc(ohlcv)
        if tf in _MULTITF_BODY_AND_CLV:
            rng = (ohlcv["high"] - ohlcv["low"]).replace(0, np.nan)
            ohlcv[f"cs_{label}_body_range"] = (
                (ohlcv["close"] - ohlcv["open"]).abs() / rng
            )
            keep.insert(0, f"cs_{label}_body_range")

        aligned = ohlcv[keep].reindex(candles_1m.index, method="ffill")
        features = features.join(aligned)

    return features.astype(np.float32)


def compute_subminute_clv(
    tbbo: pd.DataFrame,
    timestep: str | None = None,
    timeframes: tuple[str, ...] = _DEFAULT_SUBMIN,
    grid: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """CLV on 15s / 30s OHLC built from TBBO prints, aligned to the 5s grid.

    Print OHLC (first / max / min / last trade in the bucket) matches how 1-min
    candles are synthesized. ``label="right"`` so a forming 15s/30s bar is not
    visible on earlier 5s slots. Values are then stale-forwarded onto *grid*
    (or a TBBO-derived 5s index): 15s holds 3 bars, 30s holds 6 bars.
    """
    timestep = timestep or get_timestep()
    if tbbo is None or tbbo.empty or "price" not in tbbo.columns:
        return pd.DataFrame()

    df = tbbo
    if "ts_event" in df.columns:
        ts = pd.to_datetime(df["ts_event"], utc=True)
        px = pd.Series(df["price"].to_numpy(dtype="float64"), index=ts, name="price")
    elif isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        px = pd.Series(df["price"].to_numpy(dtype="float64"), index=idx, name="price")
    else:
        return pd.DataFrame()
    px = px.sort_index()
    px = px[~px.index.duplicated(keep="last")]
    if px.empty:
        return pd.DataFrame()

    if grid is None:
        out_idx = px.resample(timestep).last().index
    else:
        out_idx = pd.DatetimeIndex(grid)
        if out_idx.tz is None:
            out_idx = out_idx.tz_localize("UTC")
        else:
            out_idx = out_idx.tz_convert("UTC")
    features = pd.DataFrame(index=out_idx)
    for tf in timeframes:
        label = tf.replace("sec", "s")
        ohlc = px.resample(tf, label="right").agg(
            ["first", "max", "min", "last"]
        )
        ohlc.columns = ["open", "high", "low", "close"]
        ohlc = ohlc.dropna(subset=["open"])
        if ohlc.empty:
            features[f"cs_{label}_clv"] = np.nan
            continue
        clv = _clv_from_ohlc(ohlc)
        features[f"cs_{label}_clv"] = clv.reindex(out_idx, method="ffill")

    return features.astype(np.float32)
