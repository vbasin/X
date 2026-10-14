"""Label generation — Detection-2 (order-agnostic MAE window) + Triple Barrier Model.

Three integer label values are emitted by both schemes:
    0 = BIG_DOWN  (directional short signal)
    1 = NO_SIGNAL (masked at training time via sample_weight=0)
    2 = BIG_UP    (directional long signal)

``LabelConfig.scheme`` selects the scheme (single dispatch point for every
consumer — trainer, search fold builder, fold cache):

* ``detection_2`` (default) — order-agnostic window filter. BIG_UP when, within
  the forward window: peak_excursion_up >= big_move_threshold AND
  peak_excursion_down < mae_tolerance. Ignores the ORDER of the two excursions.
* ``triple_barrier`` — path-dependent first-touch (López de Prado TBM). BIG_UP
  when the +threshold profit barrier is touched strictly BEFORE the -mae_tolerance
  stop barrier and before the time barrier (window end). Credits a captured winner
  even if price later reverses; rejects a target printed only after the stop was
  already breached. BIG_DOWN symmetric. Stop-first or time-barrier => NO_SIGNAL.

Both schemes label on the concatenated multi-day close series, so forward windows
near the session close bleed into the next session (shared, documented caveat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.config import LabelConfig

BIG_DOWN = 0
NO_SIGNAL = 1
BIG_UP = 2


def _classify_detection_2(
    up_move: pd.Series,
    down_move: pd.Series,
    big_threshold: float,
    mae_tolerance: float,
) -> pd.Series:
    """Apply size + MAE filter to emit {0, 1, 2} labels."""
    label = pd.Series(1, index=up_move.index, dtype=np.int32)  # default NO_SIGNAL
    big_up = (up_move >= big_threshold) & (down_move < mae_tolerance)
    big_down = (down_move >= big_threshold) & (up_move < mae_tolerance)
    label[big_up] = 2
    label[big_down] = 0
    return label


def _classify_triple_barrier(
    prices: np.ndarray,
    window: int,
    big_threshold: float,
    mae_tolerance: float,
    *,
    target_high: np.ndarray | None = None,
    target_low: np.ndarray | None = None,
    target_observation: str = "candle_close",
) -> np.ndarray:
    """First-touch triple-barrier labels for one forward *window* (vectorised).

    For every bar t, finds the first offset k in [1, window] at which each of the
    four barriers is touched (long profit +threshold, long stop -mae, short profit
    -threshold, short stop +mae), then:

        BIG_UP   (2) iff first_touch(+threshold) <  first_touch(-mae)   and within window
        BIG_DOWN (0) iff first_touch(-threshold) <  first_touch(+mae)   and within window
        NO_SIGNAL(1) otherwise (stop first, or time barrier reached)

    The two directional conditions are mutually exclusive (touching +threshold
    implies +mae was touched at or before the same bar). Bars near the end of the
    series have truncated windows: a genuine first-touch within the available bars
    still labels; missing future data can only yield NO_SIGNAL, never a signal.
    O(n·window) with per-offset vectorised updates.
    """
    n = len(prices)
    mode = str(target_observation or "candle_close").lower()
    if mode not in {"candle_close", "candle_hilo", "tbbo"}:
        raise ValueError(f"unsupported target_observation={target_observation!r}")
    if mode == "candle_close":
        high = low = prices
    else:
        if target_high is None or target_low is None:
            raise ValueError(f"target_observation={mode!r} requires target_high and target_low")
        high = np.asarray(target_high, dtype=np.float64)
        low = np.asarray(target_low, dtype=np.float64)
        if high.shape != prices.shape or low.shape != prices.shape:
            raise ValueError("target high/low lengths must match prices")
    win = int(window)
    sentinel = np.int32(win + 1)
    first_up_target = np.full(n, sentinel, dtype=np.int32)   # +threshold touch
    first_dn_target = np.full(n, sentinel, dtype=np.int32)   # -threshold touch
    first_up_stop = np.full(n, sentinel, dtype=np.int32)     # +mae touch (short stop)
    first_dn_stop = np.full(n, sentinel, dtype=np.int32)     # -mae touch (long stop)

    for k in range(1, win + 1):
        if k >= n:
            break
        m = n - k
        close_diff = prices[k:] - prices[:m]
        high_diff = high[k:] - prices[:m]
        low_diff = low[k:] - prices[:m]
        for arr, cond in (
            (first_up_target, high_diff >= big_threshold),
            (first_dn_target, low_diff <= -big_threshold),
            (first_up_stop, close_diff >= mae_tolerance),
            (first_dn_stop, close_diff <= -mae_tolerance),
        ):
            head = arr[:m]                       # view — writes propagate
            hit = (head == sentinel) & cond
            if hit.any():
                head[hit] = k

    label = np.ones(n, dtype=np.int32)           # default NO_SIGNAL
    # High/low and TBBO target hits occur before the bucket's closing stop check.
    target_first = mode != "candle_close"
    up_wins = first_up_target <= first_dn_stop if target_first else first_up_target < first_dn_stop
    down_wins = first_dn_target <= first_up_stop if target_first else first_dn_target < first_up_stop
    label[(first_up_target <= win) & up_wins] = 2
    label[(first_dn_target <= win) & down_wins] = 0
    return label


def generate_labels_tbm(
    prices: pd.Series,
    config: LabelConfig,
    *,
    target_high: pd.Series | np.ndarray | None = None,
    target_low: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute triple-barrier (first-touch) labels for every forward window."""
    threshold = float(config.big_move_threshold)
    mae_tolerance = float(config.mae_tolerance)
    p = prices.to_numpy(dtype=np.float64)
    high = None if target_high is None else np.asarray(target_high, dtype=np.float64)
    low = None if target_low is None else np.asarray(target_low, dtype=np.float64)
    labels = pd.DataFrame(index=prices.index)
    for window in config.forward_windows:
        labels[f"label_{window}"] = _classify_triple_barrier(
            p, window, threshold, mae_tolerance, target_high=high, target_low=low,
            target_observation=config.target_observation,
        )
    labels["label"] = labels[f"label_{config.forward_windows[0]}"]
    return labels


def generate_labels(
    prices: pd.Series,
    config: LabelConfig,
    *,
    target_high: pd.Series | np.ndarray | None = None,
    target_low: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute labels for every forward window in *config*.

    Dispatches on ``config.scheme``: ``"triple_barrier"`` → first-touch TBM
    (:func:`generate_labels_tbm`); anything else → Detection-2 (default), which
    computes the peak upward and downward price excursions over the next *window*
    bars and applies the size + MAE filter from :func:`_classify_detection_2`.
    """
    if getattr(config, "scheme", "detection_2") == "triple_barrier":
        return generate_labels_tbm(
            prices, config, target_high=target_high, target_low=target_low)

    threshold = float(config.big_move_threshold)
    mae_tolerance = float(config.mae_tolerance)
    labels = pd.DataFrame(index=prices.index)

    for window in config.forward_windows:
        max_future = prices.rolling(window, min_periods=1).max().shift(-window)
        min_future = prices.rolling(window, min_periods=1).min().shift(-window)

        up_move = max_future - prices
        down_move = prices - min_future

        labels[f"label_{window}"] = _classify_detection_2(
            up_move, down_move, threshold, mae_tolerance,
        )

    # Primary label = shortest forward window
    labels["label"] = labels[f"label_{config.forward_windows[0]}"]
    return labels


def encode_dual_binary(label: pd.Series | np.ndarray) -> pd.DataFrame:
    """Every timestamp gets both columns.

    BIG_UP   → y_long=1, y_short=0
    BIG_DOWN → y_long=0, y_short=1
    else     → y_long=0, y_short=0
    """
    y = np.asarray(label)
    y_long = np.zeros(len(y), dtype=np.int8)
    y_short = np.zeros(len(y), dtype=np.int8)
    y_long[y == BIG_UP] = 1
    y_short[y == BIG_DOWN] = 1
    idx = label.index if isinstance(label, pd.Series) else None
    return pd.DataFrame({"y_long": y_long, "y_short": y_short}, index=idx)


def on_timestep_grid(index: pd.Index, timestep_s: int) -> np.ndarray:
    """True where *index* unix-seconds land on a ``timestep_s`` boundary."""
    ts = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return (ts.asi8 // 1_000_000_000 % int(timestep_s)) == 0
