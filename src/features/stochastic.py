"""Stochastic oscillator features: %D lines from StochasticFast and StochasticFull.

ThinkorSwim studies (1-minute NQ OHLC, forward-filled to 5s):

  1. StochasticFast(80, 20,  9,  3, HIGH, LOW, CLOSE, WILDERS,     On FastD)
  2. StochasticFast(80, 20, 14,  3, HIGH, LOW, CLOSE, WILDERS,     On FastD)
  3. StochasticFast(80, 20, 40,  4, HIGH, LOW, CLOSE, WILDERS,     On FastD)
  4. StochasticFull(80, 20, 60,  1, HIGH, LOW, CLOSE, 10, EXPONENTIAL, On FullD)

Only the %D plot is emitted. %K is computed internally (FastK / FullK) because
%D is a moving average of that %K.

Column names: ``so_{fast|full}_{d|k}_{kPeriod}_{dPeriod}``. All four studies
use the ``d`` line, so e.g. Fast(9, 3) → ``so_fast_d_9_3``.

TOS difference (Fast vs Full):
  Fast %K  = raw Lane %K (no slowing)
  Fast %D  = MA(Fast %K, dPeriod)                 — here Wilder, α = 1 / dPeriod
  Full %K  = MA(raw %K, slowingPeriod)            — here EMA, slowing = 1
  Full %D  = MA(Full %K, dPeriod)                 — here EMA, dPeriod = 10

TOS: slowing = 1 makes FullK identical to FastK; FullD is then an EMA of
that unsmoothed %K (not a Wilder FastD).

Sync scores ∈ [−1, +1]: after a 20-up / 80-down, each %D keeps an exponential
envelope from the print toward midpoint 50 (soft pre-cross term still applies).
``so_sync_score_all`` uses all four; ``so_sync_score_9_60`` uses Fast(9,3)+Full(60,10).

Wilder's EMA ≡ pandas ewm(com = n - 1, adjust=False).
Standard EMA ≡ pandas ewm(span = n, adjust=False).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# TOS StochasticFull slowing period (EMA of raw %K → FullK). Not in the
# column name; the name carries kPeriod and dPeriod only. slowing=1 ⇒ FullK
# = FastK; FullD is still EMA(FullK, dPeriod).
_FULL_SLOWING = 1

SO_COLS = (
    "so_fast_d_9_3",
    "so_fast_d_14_3",
    "so_fast_d_40_4",
    "so_full_d_60_10",
    "so_sync_score_all",
    "so_sync_score_9_60",
)

# Post-cross envelope: 1 at the 20/80 print, 0 at midpoint 50.
_SYNC_LAMBDA = 1.0
_SYNC_OB = 80.0
_SYNC_OS = 20.0
_SYNC_MID = 50.0
_SYNC_SPAN = 30.0  # |80-50| = |20-50|

# Pre-v5.7 names (K lines + short Fast names). Patch drops these.
SO_RETIRED_COLS = (
    "so_9_k", "so_9_d", "so_14_k", "so_14_d",
    "so_40_k", "so_40_d", "so_60_k", "so_60_d",
    "so_fast9_k", "so_fast9_d", "so_fast14_k", "so_fast14_d",
    "so_fast40_k", "so_fast40_d", "so_full60_k", "so_full60_d",
    "so_sync_score",
)


def _col(kind: str, k_period: int, d_period: int) -> str:
    """``so_{fast|full}_d_{k}_{d}`` — %D line; k/d are the study parameters."""
    return f"so_{kind}_d_{k_period}_{d_period}"


def _raw_k(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Lane Fast %K: 100 × (Close − LowestLow(n)) / (HighestHigh(n) − LowestLow(n))."""
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    denom = (highest - lowest).replace(0.0, np.nan)
    return (100.0 * (close - lowest) / denom).clip(0.0, 100.0)


def _fast_d(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    smooth_d: int,
) -> pd.Series:
    """TOS StochasticFast %D: Wilder MA of unsmoothed Fast %K."""
    k = _raw_k(high, low, close, period)
    return k.ewm(com=smooth_d - 1, adjust=False, min_periods=smooth_d).mean()


def _full_d(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
    slowing: int,
    smooth_d: int,
) -> pd.Series:
    """TOS StochasticFull %D: EMA(EMA(raw %K, slowing), dPeriod)."""
    raw_k = _raw_k(high, low, close, period)
    full_k = raw_k.ewm(span=slowing, adjust=False, min_periods=slowing).mean()
    return full_k.ewm(span=smooth_d, adjust=False, min_periods=max(smooth_d, 1)).mean()


def _decay_to_mid(progress: pd.Series) -> pd.Series:
    """Affine exponential: 1 at progress=0 (20/80 print), 0 at progress=1 (50)."""
    p = progress.astype("float64").clip(0.0, 1.0)
    e_l = np.exp(-_SYNC_LAMBDA)
    return ((np.exp(-_SYNC_LAMBDA * p) - e_l) / (1.0 - e_l)).clip(0.0, 1.0)


def _arm_flag(arm_on: pd.Series, clear: pd.Series) -> pd.Series:
    """Latch arm_on until clear (recross or through midpoint). Clear wins same bar."""
    event = np.where(
        clear.fillna(False).to_numpy(),
        0.0,
        np.where(arm_on.fillna(False).to_numpy(), 1.0, np.nan),
    )
    return pd.Series(event, index=arm_on.index).ffill().fillna(0.0).eq(1.0)


def _dir_ok(delta: pd.Series, *, want_up: bool) -> pd.Series:
    """True unless two consecutive bars move against the signal (1-bar pause)."""
    contrary = (delta < 0) if want_up else (delta > 0)
    return ~(contrary & contrary.shift(1, fill_value=False))


def _votes_for_d(d: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Bull / bear votes in [0, 1] for one %D series.

    Soft: oversold+rising (0–25) or overbought+falling (75–100).
    Armed envelope: after 20-up (resp. 80-down), ``exp`` fade from the
    print toward 50; 0 past the midpoint; 1-bar pause on a contrary tick.
    """
    d = d.astype("float64")
    delta = d.diff()
    prev = d.shift(1)

    soft_bull = ((25.0 - d) / 25.0).clip(0.0, 1.0) * (delta / 3.0).clip(0.0, 1.0)
    soft_bear = ((d - 75.0) / 25.0).clip(0.0, 1.0) * ((-delta) / 3.0).clip(0.0, 1.0)

    cross_up = (prev < _SYNC_OS) & (d >= _SYNC_OS)
    recross_os = (prev >= _SYNC_OS) & (d < _SYNC_OS)
    bull_arm = _arm_flag(cross_up, recross_os | (d > _SYNC_MID))
    bull_p = (d - _SYNC_OS) / _SYNC_SPAN
    env_bull = (
        bull_arm.astype("float64")
        * _dir_ok(delta, want_up=True).astype("float64")
        * _decay_to_mid(bull_p)
    )

    cross_dn = (prev > _SYNC_OB) & (d <= _SYNC_OB)
    recross_ob = (prev <= _SYNC_OB) & (d > _SYNC_OB)
    bear_arm = _arm_flag(cross_dn, recross_ob | (d < _SYNC_MID))
    bear_p = (_SYNC_OB - d) / _SYNC_SPAN
    env_bear = (
        bear_arm.astype("float64")
        * _dir_ok(delta, want_up=False).astype("float64")
        * _decay_to_mid(bear_p)
    )

    bull = np.maximum(soft_bull, env_bull).clip(0.0, 1.0)
    bear = np.maximum(soft_bear, env_bear).clip(0.0, 1.0)
    return bull, bear


def _sync_score(d_frame: pd.DataFrame) -> pd.Series:
    """Signed sync score from N %D series, ∈ [−1, +1].

    +1 = members oversold and turning up (20-up family) in agreement.
    −1 = members overbought and turning down (80-down family) in agreement.
    Magnitude grows when more of the N agree.
    """
    n = float(len(d_frame.columns))
    if n == 0:
        return pd.Series(np.nan, index=d_frame.index)
    parts_bull = []
    parts_bear = []
    for col in d_frame.columns:
        bull, bear = _votes_for_d(d_frame[col])
        parts_bull.append(bull)
        parts_bear.append(bear)
    bull_m = pd.concat(parts_bull, axis=1).mean(axis=1)
    bear_m = pd.concat(parts_bear, axis=1).mean(axis=1)
    n_bull = pd.concat([(p > 0.25).astype("float64") for p in parts_bull], axis=1).sum(axis=1)
    n_bear = pd.concat([(p > 0.25).astype("float64") for p in parts_bear], axis=1).sum(axis=1)
    agree = pd.concat([n_bull, n_bear], axis=1).max(axis=1) / n
    return ((bull_m - bear_m) * (0.5 + 0.5 * agree)).clip(-1.0, 1.0)


def compute(candles: pd.DataFrame) -> pd.DataFrame:
    """Four %D oscillators on 1-min OHLC plus two sync-reversal scores.

    Fast (Wilder %D): (9, 3), (14, 3), (40, 4).
    Full (EMA slowing=1, EMA %D=10): (60, 10).
    ``so_sync_score_all`` ∈ [−1, +1] — all four %Ds.
    ``so_sync_score_9_60`` ∈ [−1, +1] — Fast(9,3) + Full(60,10) only.
    """
    if candles.empty or not {"high", "low", "close"}.issubset(candles.columns):
        return pd.DataFrame(np.nan, index=candles.index, columns=list(SO_COLS)).astype(np.float32)

    high = candles["high"].astype(np.float64)
    low = candles["low"].astype(np.float64)
    close = candles["close"].astype(np.float64)

    d_fast_9 = _fast_d(high, low, close, period=9, smooth_d=3)
    d_fast_14 = _fast_d(high, low, close, period=14, smooth_d=3)
    d_fast_40 = _fast_d(high, low, close, period=40, smooth_d=4)
    d_full_60 = _full_d(
        high, low, close, period=60, slowing=_FULL_SLOWING, smooth_d=10,
    )

    c9 = _col("fast", 9, 3)
    c14 = _col("fast", 14, 3)
    c40 = _col("fast", 40, 4)
    c60 = _col("full", 60, 10)
    out = pd.DataFrame(
        {
            c9: d_fast_9,
            c14: d_fast_14,
            c40: d_fast_40,
            c60: d_full_60,
        },
        index=close.index,
    )
    out["so_sync_score_all"] = _sync_score(out[[c9, c14, c40, c60]])
    out["so_sync_score_9_60"] = _sync_score(out[[c9, c60]])
    return out[list(SO_COLS)].astype(np.float32)
