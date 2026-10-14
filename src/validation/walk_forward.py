"""Expanding calendar walk-forward — Custom Test designer (ADR-042).

Cuts folds on session-local calendar dates (not equal bar counts). Fold 1
trains ``train`` period from *start*; each later fold's train is all prior
train+test blocks (expanding) until *stop*. Optional val is carved from the
end of train or the start of test by calendar length. Tree families ignore
``X_val``; sequence/DL families use it for early stopping.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.search.walk_forward_calendar")

UNITS = ("days", "weeks", "months")
MAX_FOLDS = 24
DEFAULT_START = "2025-05-01"
DEFAULT_STOP = "2026-06-30"
DEFAULT_PURGE_BARS = 120        # == the 10-minute TBM horizon at 5s bars ONLY
DEFAULT_PURGE_SECONDS = 600     # the horizon as a duration — interval-independent
DEFAULT_WALK_FORWARD: dict[str, Any] = {
    "enabled": True,
    "start": DEFAULT_START,
    "stop": DEFAULT_STOP,
    "train_n": 2,
    "train_unit": "months",
    "test_n": 2,
    "test_unit": "months",
    "val": {
        "enabled": False,
        "n": 10,
        "unit": "days",
        "cut_from": "train",
    },
}


def _parse_day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    ts = pd.Timestamp(str(value).strip()[:10])
    if pd.isna(ts):
        raise ValueError(f"invalid date: {value!r}")
    return ts.date()


def _norm_unit(unit: Any) -> str:
    u = str(unit or "months").strip().lower()
    if u in ("month", "mo", "m") and u not in ("minutes",):
        if u == "m":
            u = "months"
        elif u == "mo":
            u = "months"
        elif u == "month":
            u = "months"
    if u in ("day", "d"):
        u = "days"
    if u in ("week", "wk", "w"):
        u = "weeks"
    if u not in UNITS:
        raise ValueError(f"unit must be days, weeks, or months (got {unit!r})")
    return u


def add_period(day: date, n: int, unit: str) -> date:
    """Return *day* plus *n* periods (*unit* = days/weeks/months)."""
    n = int(n)
    if n < 0:
        raise ValueError("period length must be >= 0")
    unit = _norm_unit(unit)
    ts = pd.Timestamp(day) + pd.DateOffset(**{unit: n})
    return ts.date()


def inclusive_end(exclusive: date) -> date:
    return exclusive - timedelta(days=1)


def period_cfg(n: Any, unit: Any) -> tuple[int, str]:
    nn = int(n)
    if nn < 1:
        raise ValueError("period length must be >= 1")
    return nn, _norm_unit(unit)


def default_walk_forward(objective: dict[str, Any] | None = None) -> dict[str, Any]:
    """Custom Test designer defaults (not the objective YAML equal-count split)."""
    out = {
        "enabled": True,
        "start": DEFAULT_START,
        "stop": DEFAULT_STOP,
        "train_n": 2,
        "train_unit": "months",
        "test_n": 2,
        "test_unit": "months",
        "val": {
            "enabled": False,
            "n": 10,
            "unit": "days",
            "cut_from": "train",
        },
    }
    _ = objective  # designer defaults are fixed; objective yaml is not mutated
    return out


def normalize_board_wf(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a Custom Test board ``walk_forward`` object."""
    base = default_walk_forward()
    if not raw:
        return base
    val_in = raw.get("val") if isinstance(raw.get("val"), dict) else {}
    val = {**base["val"], **(val_in or {})}
    cut = str(val.get("cut_from") or "train").strip().lower()
    if cut not in ("train", "test"):
        raise ValueError("val.cut_from must be 'train' or 'test'")
    val["cut_from"] = cut
    val["enabled"] = bool(val.get("enabled"))
    val["n"] = int(val.get("n") or 10)
    val["unit"] = _norm_unit(val.get("unit") or "days")
    if val["n"] < 1:
        raise ValueError("val length must be >= 1")
    train_n, train_unit = period_cfg(raw.get("train_n", 2), raw.get("train_unit", "months"))
    test_n, test_unit = period_cfg(raw.get("test_n", 2), raw.get("test_unit", "months"))
    start = _parse_day(raw.get("start") or DEFAULT_START).isoformat()
    stop = _parse_day(raw.get("stop") or DEFAULT_STOP).isoformat()
    if start > stop:
        raise ValueError(f"walk-forward start {start} is after stop {stop}")
    return {
        "enabled": bool(raw.get("enabled", True)),
        "start": start,
        "stop": stop,
        "train_n": train_n,
        "train_unit": train_unit,
        "test_n": test_n,
        "test_unit": test_unit,
        "val": val,
    }


def build_expanding_schedule(
    *,
    start: str | date,
    stop: str | date,
    train_n: int = 2,
    train_unit: str = "months",
    test_n: int = 2,
    test_unit: str = "months",
    val_enabled: bool = False,
    val_n: int = 10,
    val_unit: str = "days",
    val_cut_from: str = "train",
    max_folds: int = MAX_FOLDS,
) -> list[dict[str, Any]]:
    """Return fold calendar ranges (inclusive dates). Expanding: train1+test1 = train2."""
    start_d = _parse_day(start)
    stop_d = _parse_day(stop)
    train_n, train_unit = period_cfg(train_n, train_unit)
    test_n, test_unit = period_cfg(test_n, test_unit)
    stop_excl = stop_d + timedelta(days=1)
    train_end_excl = add_period(start_d, train_n, train_unit)
    if train_end_excl > stop_excl:
        raise ValueError(
            f"initial train period ({train_n} {train_unit}) overruns stop {stop_d.isoformat()}"
        )
    out: list[dict[str, Any]] = []
    clipped = False
    while len(out) < int(max_folds):
        test_end_excl = add_period(train_end_excl, test_n, test_unit)
        if train_end_excl >= stop_excl:
            break
        if test_end_excl > stop_excl:
            test_end_excl = stop_excl
            clipped = True
        if test_end_excl <= train_end_excl:
            break
        train_end_incl = inclusive_end(train_end_excl)
        test_start = train_end_excl
        test_end_incl = inclusive_end(test_end_excl)
        row: dict[str, Any] = {
            "fold": len(out),
            "split": "expanding_calendar",
            "train": {
                "start": start_d.isoformat(),
                "end": train_end_incl.isoformat(),
            },
            "oof": {
                "start": test_start.isoformat(),
                "end": test_end_incl.isoformat(),
            },
            "val": None,
            "clipped": clipped,
        }
        if val_enabled:
            row["val"] = _carve_val(
                train_start=start_d,
                train_end_incl=train_end_incl,
                test_start=test_start,
                test_end_incl=test_end_incl,
                val_n=int(val_n),
                val_unit=val_unit,
                cut_from=val_cut_from,
            )
            row["val_cut_from"] = str(val_cut_from)
            # Fit-train shrinks when val is taken from train.
            if str(val_cut_from) == "train" and row["val"] is not None:
                vs = _parse_day(row["val"]["start"])
                fit_end = vs - timedelta(days=1)
                if fit_end < start_d:
                    raise ValueError("val period leaves no training days in a fold")
                row["train"]["end"] = fit_end.isoformat()
            elif str(val_cut_from) == "test" and row["val"] is not None:
                ve = _parse_day(row["val"]["end"])
                oof_start = ve + timedelta(days=1)
                if oof_start > test_end_incl:
                    raise ValueError("val period leaves no OOF days in a fold")
                row["oof"]["start"] = oof_start.isoformat()
        out.append(row)
        train_end_excl = test_end_excl
        if clipped:
            break
    if not out:
        raise ValueError(
            "walk-forward window does not fit one train + test block; "
            "widen the stop date or shrink train/test periods"
        )
    if len(out) >= int(max_folds) and train_end_excl < stop_excl:
        logger.warning("walk_forward capped at %d folds (stop date not reached)", max_folds)
    return out


def _carve_val(
    *,
    train_start: date,
    train_end_incl: date,
    test_start: date,
    test_end_incl: date,
    val_n: int,
    val_unit: str,
    cut_from: str,
) -> dict[str, str]:
    cut = str(cut_from or "train").strip().lower()
    vn, vu = period_cfg(val_n, val_unit)
    if cut == "test":
        val_end_excl = add_period(test_start, vn, vu)
        if val_end_excl > test_end_incl + timedelta(days=1):
            raise ValueError("val period is longer than the test window")
        val_end_incl = inclusive_end(val_end_excl)
        if val_end_incl < test_start:
            raise ValueError("val period is empty")
        if val_end_incl >= test_end_incl:
            raise ValueError("val carved from test leaves no OOF remainder")
        return {"start": test_start.isoformat(), "end": val_end_incl.isoformat()}
    # train tail: last val_n units of [train_start, train_end_incl]
    train_end_excl = train_end_incl + timedelta(days=1)
    val_start = _subtract_period(train_end_excl, vn, vu)
    if val_start <= train_start:
        raise ValueError("val period is longer than the train window")
    return {"start": val_start.isoformat(), "end": train_end_incl.isoformat()}


def _subtract_period(exclusive_end: date, n: int, unit: str) -> date:
    """Start date such that [start, exclusive_end) has length *n* *unit*."""
    unit = _norm_unit(unit)
    ts = pd.Timestamp(exclusive_end) - pd.DateOffset(**{unit: int(n)})
    return ts.date()


def overlay_from_board(board_wf: dict[str, Any]) -> dict[str, Any] | None:
    """Eval overlay written as ``*.walk_forward.yaml`` (None = do not override)."""
    cfg = normalize_board_wf(board_wf)
    if not cfg["enabled"]:
        return None
    schedule = build_expanding_schedule(
        start=cfg["start"], stop=cfg["stop"],
        train_n=cfg["train_n"], train_unit=cfg["train_unit"],
        test_n=cfg["test_n"], test_unit=cfg["test_unit"],
        val_enabled=bool(cfg["val"]["enabled"]),
        val_n=int(cfg["val"]["n"]),
        val_unit=str(cfg["val"]["unit"]),
        val_cut_from=str(cfg["val"]["cut_from"]),
    )
    return {
        "date_range": {"start": cfg["start"], "end": cfg["stop"]},
        "walk_forward_folds": len(schedule),
        "walk_forward": {
            "mode": "expanding_calendar",
            "train": {"n": cfg["train_n"], "unit": cfg["train_unit"]},
            "test": {"n": cfg["test_n"], "unit": cfg["test_unit"]},
            "purge_bars": DEFAULT_PURGE_BARS,
            "purge_seconds": DEFAULT_PURGE_SECONDS,
            "val": {
                "enabled": bool(cfg["val"]["enabled"]),
                "n": int(cfg["val"]["n"]),
                "unit": str(cfg["val"]["unit"]),
                "cut_from": str(cfg["val"]["cut_from"]),
            },
            "schedule": schedule,
        },
    }


def schedule_from_eval(eval_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    wf = eval_cfg.get("walk_forward") or {}
    if wf.get("schedule"):
        return list(wf["schedule"])
    dr = eval_cfg.get("date_range") or {}
    train = wf.get("train") or {}
    test = wf.get("test") or {}
    val = wf.get("val") or {}
    return build_expanding_schedule(
        start=dr.get("start") or DEFAULT_START,
        stop=dr.get("end") or DEFAULT_STOP,
        train_n=int(train.get("n") or 2),
        train_unit=str(train.get("unit") or "months"),
        test_n=int(test.get("n") or 2),
        test_unit=str(test.get("unit") or "months"),
        val_enabled=bool(val.get("enabled")),
        val_n=int(val.get("n") or 10),
        val_unit=str(val.get("unit") or "days"),
        val_cut_from=str(val.get("cut_from") or "train"),
    )


def probe_bar_seconds(ts: np.ndarray, *, default: float = 5.0) -> float:
    """Median spacing of *ts* in seconds (0 / non-finite → *default*)."""
    if ts is None or len(ts) < 2:
        return default
    d = np.diff(pd.to_datetime(ts, utc=True).asi8) / 1e9
    d = d[d > 0]
    if not len(d):
        return default
    med = float(np.median(d))
    return med if med > 0 else default


def resolve_purge_rows(
    ts: np.ndarray,
    *,
    purge_bars: int | None = None,
    purge_seconds: int | None = None,
) -> tuple[int, float]:
    """Rows of train to drop at the OOF boundary, and the probed bar interval.

    ``purge_bars`` is a **row count**, so it only equals the label horizon at the bar
    interval it was written for: the default 120 is 10 minutes at 5s bars and 2 minutes at
    1s bars. Prefer ``purge_seconds`` — it is converted with the interval probed from
    *ts*, so the purged *duration* is correct at any resolution. When only ``purge_bars``
    is given and the implied duration does not match ``DEFAULT_PURGE_SECONDS``, this warns
    rather than silently under-purging.
    """
    bar_s = probe_bar_seconds(ts)
    if purge_seconds is not None:
        rows = int(np.ceil(max(0, int(purge_seconds)) / bar_s))
        return rows, bar_s
    rows = max(0, int(purge_bars or 0))
    implied = rows * bar_s
    if rows and abs(implied - DEFAULT_PURGE_SECONDS) > 1.0:
        logger.warning(
            "purge_bars=%d at a probed %.3gs bar interval purges only %.0fs — the TBM "
            "label horizon is %ds. Train rows inside the horizon overlap the OOF block "
            "(leak). Set purge_seconds=%d (or purge_bars=%d).",
            rows, bar_s, implied, DEFAULT_PURGE_SECONDS, DEFAULT_PURGE_SECONDS,
            int(np.ceil(DEFAULT_PURGE_SECONDS / bar_s)))
    return rows, bar_s


def _session_dates(ts: np.ndarray, tz_name: str) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return np.array(idx.tz_convert(tz_name).date)


def _span_meta(ts_slice: np.ndarray, y_slice: np.ndarray, tz_name: str) -> dict[str, Any]:
    if ts_slice is None or len(ts_slice) == 0:
        return {"start": None, "end": None, "n_bars": 0, "n_days": 0,
                "n_down": 0, "n_flat": 0, "n_up": 0}
    dates = _session_dates(ts_slice, tz_name)
    y = np.asarray(y_slice)
    return {
        "start": pd.Timestamp(pd.to_datetime(ts_slice[0], utc=True)).isoformat(),
        "end": pd.Timestamp(pd.to_datetime(ts_slice[-1], utc=True)).isoformat(),
        "n_bars": int(len(ts_slice)),
        "n_days": int(len(set(dates.tolist()))),
        "n_down": int((y == 0).sum()),
        "n_flat": int((y == 1).sum()),
        "n_up": int((y == 2).sum()),
    }


def apply_schedule_to_arrays(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    schedule: list[dict[str, Any]],
    feature_cols: list[str],
    *,
    tz_name: str = "America/Chicago",
    purge_bars: int = DEFAULT_PURGE_BARS,
    purge_seconds: int | None = None,
    extras: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Slice cached arrays into walk-forward folds using calendar *schedule*.

    ``purge_seconds`` (preferred) is interval-independent; ``purge_bars`` is a raw row
    count kept for backward compatibility. See :func:`resolve_purge_rows`.
    ``extras`` (path-label hazard targets, etc.) are sliced onto each fold as
    ``{name}_train`` / ``{name}_val`` / ``{name}_oof``.
    """
    dates = _session_dates(ts, tz_name)
    pb, bar_s = resolve_purge_rows(ts, purge_bars=purge_bars, purge_seconds=purge_seconds)
    logger.info("calendar folds: purge %d rows (%.0fs at a probed %.3gs bar interval)",
                pb, pb * bar_s, bar_s)
    empty_x = X[:0]
    empty_y = y[:0]
    folds: list[dict[str, Any]] = []

    def _between(start_s: str, end_s: str) -> np.ndarray:
        a, b = _parse_day(start_s), _parse_day(end_s)
        return (dates >= a) & (dates <= b)

    for spec in schedule:
        tr = spec["train"]
        oof = spec["oof"]
        train_mask = _between(tr["start"], tr["end"])
        oof_mask = _between(oof["start"], oof["end"])
        train_idx = np.where(train_mask)[0]
        oof_idx = np.where(oof_mask)[0]
        if pb and len(train_idx) > pb:
            # Drop the last purge_bars of train (label horizon overlap with OOF).
            train_idx = train_idx[:-pb]
        val_idx = np.array([], dtype=int)
        val_spec = spec.get("val")
        if val_spec:
            val_mask = _between(val_spec["start"], val_spec["end"])
            val_idx = np.where(val_mask)[0]
            cut = str(spec.get("val_cut_from") or "train")
            if cut == "train":
                train_idx = train_idx[~np.isin(train_idx, val_idx)]
            else:
                oof_idx = oof_idx[~np.isin(oof_idx, val_idx)]
        if len(train_idx) < 20 or len(oof_idx) < 1:
            logger.warning(
                "calendar fold %s skipped (train_bars=%d oof_bars=%d)",
                spec.get("fold"), len(train_idx), len(oof_idx),
            )
            continue
        if len(val_idx) == 0:
            X_val, y_val = empty_x, empty_y
        else:
            X_val, y_val = X[val_idx], y[val_idx]
        meta = {
            "fold": int(spec.get("fold", len(folds))),
            "split": "expanding_calendar",
            "train": {**tr, **_span_meta(ts[train_idx], y[train_idx], tz_name)},
            "oof": {**oof, **_span_meta(ts[oof_idx], y[oof_idx], tz_name)},
            "val": (
                {**val_spec, **_span_meta(ts[val_idx], y[val_idx], tz_name)}
                if val_spec is not None and len(val_idx) else None
            ),
            "purge_bars": pb,
            "clipped": bool(spec.get("clipped")),
        }
        fold: dict[str, Any] = {
            "X_train": X[train_idx],
            "y_train": y[train_idx],
            "ts_train": ts[train_idx],
            "X_val": X_val,
            "y_val": y_val,
            "ts_val": ts[val_idx] if len(val_idx) else ts[:0],
            "X_oof": X[oof_idx],
            "y_oof": y[oof_idx],
            "ts_oof": ts[oof_idx],
            "feature_cols": feature_cols,
            "fold": int(spec.get("fold", len(folds))),
            "fold_meta": meta,
        }
        if extras:
            for name, arr in extras.items():
                fold[f"{name}_train"] = arr[train_idx]
                fold[f"{name}_oof"] = arr[oof_idx]
                fold[f"{name}_val"] = arr[val_idx] if len(val_idx) else arr[:0]
        folds.append(fold)
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# One entry point, two schedule shapes
# ══════════════════════════════════════════════════════════════════════════════
#
# build_expanding_schedule() above is Track B's calendar schedule: periods in
# days/weeks/months, an optional validation carve-out, purge/embargo resolved
# separately by resolve_purge_rows().
#
# Track C's evaluators used their own inline loop instead: whole calendar months,
# an expanding train set of at least `min_train_months`, and a test block of
# `test_months` that advances by its own width. That is a genuinely different
# schedule, not a worse one -- it is driven by however many months of sessions
# the labels happen to cover, with no fixed start date. Reproducing it here (in
# the SAME fold-row shape) is what lets both tracks share one implementation and
# lets a Track B result be compared with a Track C result.

def build_month_block_schedule(
    sessions: Any,
    *,
    test_months: int = 2,
    min_train_months: int = 4,
    max_folds: int = MAX_FOLDS,
) -> list[dict[str, Any]]:
    """Expanding whole-month folds over the months ``sessions`` actually covers.

    Emits the same row shape as :func:`build_expanding_schedule`
    (``fold``/``split``/``train``/``oof``/``val``/``clipped``) so a caller,
    ``manifest.validation`` and ``fold_metrics.json`` cannot tell which schedule
    produced it. ``train`` always starts at the first observed session: the set
    expands, it does not slide.

    Returns ``[]`` when the sessions do not cover ``min_train_months +
    test_months`` whole months -- the caller reports "not enough months" rather
    than this inventing a shorter fold.
    """
    import pandas as pd

    index = pd.DatetimeIndex(pd.Series(list(sessions)).dropna().unique())
    if index.empty:
        return []
    months = index.to_period("M").unique().sort_values()
    test_n = max(1, int(test_months))
    min_train = max(1, int(min_train_months))
    out: list[dict[str, Any]] = []
    i = min_train
    while i + test_n <= len(months) and len(out) < int(max_folds):
        train_months = months[:i]
        test_block = months[i:i + test_n]
        out.append({
            "fold": len(out),
            "split": "expanding_month_blocks",
            "train": {
                "start": train_months[0].start_time.date().isoformat(),
                "end": train_months[-1].end_time.date().isoformat(),
            },
            "oof": {
                "start": test_block[0].start_time.date().isoformat(),
                "end": test_block[-1].end_time.date().isoformat(),
            },
            "val": None,
            "clipped": False,
        })
        i += test_n
    return out


#: Schedule name -> what it means. The objective's ``eval.split`` picks one.
SPLITS = ("expanding_calendar", "expanding_month_blocks", "last_range")


def build_last_range_schedule(sessions: Any) -> list[dict[str, Any]]:
    """One fold: train on everything before the last month, test on that month.

    Available when an objective sets ``eval.split: last_range``. An HPO study
    records whatever schedule it scored; it does not force this one.
    """
    import pandas as pd

    index = pd.DatetimeIndex(pd.Series(list(sessions)).dropna().unique())
    if index.empty:
        return []
    months = index.to_period("M").unique().sort_values()
    if len(months) < 2:
        return []
    train_months, test_month = months[:-1], months[-1]
    return [{
        "fold": 0,
        "split": "last_range",
        "train": {
            "start": train_months[0].start_time.date().isoformat(),
            "end": train_months[-1].end_time.date().isoformat(),
        },
        "oof": {
            "start": test_month.start_time.date().isoformat(),
            "end": test_month.end_time.date().isoformat(),
        },
        "val": None,
        "clipped": False,
    }]


def folds_from_objective(
    objective: Mapping[str, Any] | None,
    *,
    sessions: Any = None,
    overrides: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """THE fold builder. Both tracks call this; neither carries its own loop.

    The objective's ``eval`` block decides the schedule:

    ``split: expanding_calendar``     (default when ``start``/``stop`` are given)
        Track B's calendar schedule via :func:`build_expanding_schedule`.
    ``split: expanding_month_blocks`` (default when only ``sessions`` are given)
        Track C's whole-month schedule via :func:`build_month_block_schedule`.
    ``split: last_range``
        The single shared HPO window.

    ``overrides`` is the runtime's own last word (a CLI flag), applied over the
    objective so an operator can change the schedule without editing the YAML --
    and so the resolved values are what gets recorded on the manifest.
    """
    cfg: dict[str, Any] = dict((objective or {}).get("eval") or {})
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})

    split = str(cfg.get("split") or "").strip()
    if not split:
        split = "expanding_calendar" if cfg.get("start") and cfg.get("stop") \
            else "expanding_month_blocks"
    if split not in SPLITS:
        raise ValueError(f"unknown eval.split {split!r}; have {list(SPLITS)}")

    if split == "expanding_calendar":
        if not (cfg.get("start") and cfg.get("stop")):
            raise ValueError(
                "eval.split=expanding_calendar needs eval.start and eval.stop; "
                "use expanding_month_blocks to derive the window from the "
                "sessions present in the data"
            )
        return build_expanding_schedule(
            start=cfg["start"], stop=cfg["stop"],
            train_n=cfg.get("train_n", 2), train_unit=cfg.get("train_unit", "months"),
            test_n=cfg.get("test_n", 2), test_unit=cfg.get("test_unit", "months"),
            val_enabled=bool(cfg.get("val_enabled", False)),
            val_n=int(cfg.get("val_n", 10)), val_unit=cfg.get("val_unit", "days"),
            val_cut_from=str(cfg.get("val_cut_from", "train")),
            max_folds=int(cfg.get("max_folds", MAX_FOLDS)),
        )
    if sessions is None:
        raise ValueError(f"eval.split={split} needs the observed sessions")
    if split == "last_range":
        return build_last_range_schedule(sessions)
    return build_month_block_schedule(
        sessions,
        test_months=int(cfg.get("test_months", 2)),
        min_train_months=int(cfg.get("min_train_months", 4)),
        max_folds=int(cfg.get("max_folds", MAX_FOLDS)),
    )
