"""Per-archetype attribution for backtest / live trades.

An *archetype* is a discrete signal bucket the model tends to fire from — the
tuple ``(direction, confidence-bucket, hour-of-day CT)``. There are typically
~30-50 archetypes at τ≥0.8 across a full walk-forward. Attributing each
backtest trade to its archetype exposes:

- whether live PnL is spread across the same archetypes as training,
- whether a small handful of archetypes dominate (concentration risk),
- whether any archetype is hitting materially below its training precision
  (early decay signal).

Training-time stats are computed once per bundle from the search-run OOF
parquet and cached at ``<bundle>/archetype_stats.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.data_paths import data_root

logger = logging.getLogger("vtech.backtest.archetypes")

# 5-percentage-point confidence bins (0.80..0.85, 0.85..0.90, 0.90..0.95, 0.95..1.00 at τ=0.8).
_CONF_BUCKET_STEPS = 20  # bins per unit → 0.05 wide
_SESSION_TZ = "America/Chicago"


def archetype_key(direction: str, confidence: float, ts: Any) -> str:
    """Deterministic archetype key from a signal's (direction, conf, ts)."""
    conf_bin = int(round(float(confidence) * _CONF_BUCKET_STEPS))
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    hod_ct = int(t.tz_convert(_SESSION_TZ).hour)
    return f"{direction}|{conf_bin}|{hod_ct}"


def archetype_label(key: str) -> str:
    """Human-readable label, e.g. ``up · conf≥0.85 · 09:00 CT``."""
    try:
        d, cb, hod = key.split("|")
        lo = int(cb) / _CONF_BUCKET_STEPS
        return f"{d} · conf≥{lo:.2f} · {int(hod):02d}:00 CT"
    except Exception:  # noqa: BLE001
        return key


def _cache_path(bundle_path: str | Path) -> Path:
    return Path(bundle_path) / "archetype_stats.json"


def _oof_path_for_bundle(bundle_path: str | Path) -> Path | None:
    """Locate the OOF parquet from the bundle's finalize manifest."""
    bp = Path(bundle_path)
    mf = bp / "finalize_manifest.json"
    if not mf.exists():
        return None
    manifest = json.loads(mf.read_text())
    sibling = bp / "oof.parquet"
    if sibling.exists():
        return sibling
    candidate = str(manifest.get("candidate_id") or "")
    search_dir = manifest.get("search_run_dir")
    if not search_dir or not candidate:
        return None

    # search_run_dir in the manifest is the *legacy* layout (data/search/...);
    # try both the current post-ADR-029 layout and the legacy fallback.
    rel = Path(str(search_dir)).relative_to("data") if str(search_dir).startswith("data/") \
        else Path(str(search_dir))
    for candidate_root in (data_root() / "research" / rel,
                           data_root() / rel,
                           Path(str(search_dir))):
        for p in (
            candidate_root / "models" / candidate / "oof.parquet",
            candidate_root / f"oof_{candidate}.parquet",
        ):
            if p.exists():
                return p
    return None


def training_archetype_stats(
    bundle_path: str | Path,
    *,
    tau: float = 0.8,
    target_pts: float | None = None,
    stop_pts: float | None = None,
    cost_per_trade: float = 1.0,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compute (or load) per-archetype training-time stats from the bundle OOF.

    Cached to ``<bundle>/archetype_stats.json``. Returns
    ``{archetype: {n, hit_rate, expectancy_pts, share}}``.
    """
    cache = _cache_path(bundle_path)
    if cache.exists() and not force:
        try:
            payload = json.loads(cache.read_text())
            if (
                payload.get("tau") == float(tau)
                and payload.get("target_pts") == target_pts
                and payload.get("stop_pts") == stop_pts
            ):
                return payload.get("archetypes") or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("archetype cache read failed: %s", exc)

    oof_path = _oof_path_for_bundle(bundle_path)
    if oof_path is None:
        logger.info("archetype_stats: no OOF parquet located for %s", bundle_path)
        return {}

    df = pd.read_parquet(oof_path, columns=["ts", "direction", "confidence", "direction_true"])
    gate = df["direction"].isin(["up", "down"]) & (df["confidence"] >= float(tau))
    sub = df.loc[gate].copy()
    if sub.empty:
        return {}
    sub["archetype"] = [
        archetype_key(str(d), float(c), t)
        for d, c, t in zip(sub["direction"], sub["confidence"], sub["ts"])
    ]
    sub["hit"] = (sub["direction_true"].astype(str) == sub["direction"].astype(str)).astype("int64")

    total_n = int(len(sub))
    grouped = sub.groupby("archetype", sort=False).agg(
        n=("hit", "size"),
        hits=("hit", "sum"),
    )
    grouped["hit_rate"] = grouped["hits"] / grouped["n"]
    if target_pts is not None and stop_pts is not None:
        grouped["expectancy_pts"] = (
            grouped["hit_rate"] * float(target_pts)
            - (1.0 - grouped["hit_rate"]) * float(stop_pts)
            - float(cost_per_trade)
        )
    else:
        grouped["expectancy_pts"] = None
    grouped["share"] = grouped["n"] / total_n

    out: dict[str, dict[str, Any]] = {}
    for key, row in grouped.iterrows():
        exp_val = row["expectancy_pts"]
        out[str(key)] = {
            "n": int(row["n"]),
            "hit_rate": float(row["hit_rate"]),
            "expectancy_pts": (None if exp_val is None or pd.isna(exp_val)
                               else float(exp_val)),
            "share": float(row["share"]),
        }

    payload = {
        "tau": float(tau),
        "target_pts": target_pts,
        "stop_pts": stop_pts,
        "cost_per_trade": float(cost_per_trade),
        "n_total": total_n,
        "archetypes": out,
    }
    try:
        cache.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.warning("archetype cache write failed: %s", exc)
    return out


def _side_to_direction(side: str) -> str:
    return "up" if str(side).lower() == "long" else "down"


def attribute_trades(
    trades: list[dict[str, Any]],
    training_stats: dict[str, dict[str, Any]],
    *,
    entry_ts_key: str = "entry_ts",
    side_key: str = "side",
    confidence_key: str = "confidence",
    pnl_key: str = "pnl_points",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach ``archetype`` (+ training reference) to each trade; return
    ``(trades_out, aggregate_rows)``.

    Trades missing confidence are keyed with ``conf_bin = -1`` so they still
    group together (rare, only when the caller could not surface it).
    """
    if not trades:
        return trades, []

    n_live_total = len(trades)
    tagged: list[dict[str, Any]] = []
    per_arch: dict[str, dict[str, Any]] = {}

    for t in trades:
        direction = _side_to_direction(t.get(side_key, ""))
        conf = t.get(confidence_key)
        conf_f = float(conf) if conf is not None else -1.0
        key = archetype_key(direction, conf_f, t.get(entry_ts_key))
        pnl = float(t.get(pnl_key, 0.0) or 0.0)
        win = 1 if pnl > 0 else 0

        row = dict(t)
        row["archetype"] = key
        row["archetype_label"] = archetype_label(key)
        train = training_stats.get(key)
        row["archetype_train_hit_rate"] = (
            float(train["hit_rate"]) if train and train.get("hit_rate") is not None else None
        )
        row["archetype_train_n"] = int(train["n"]) if train else 0
        tagged.append(row)

        bucket = per_arch.setdefault(key, {"n": 0, "wins": 0, "pnl_pts": 0.0})
        bucket["n"] += 1
        bucket["wins"] += win
        bucket["pnl_pts"] += pnl

    agg: list[dict[str, Any]] = []
    for key, b in per_arch.items():
        n = int(b["n"])
        wins = int(b["wins"])
        pnl_pts = float(b["pnl_pts"])
        train = training_stats.get(key)
        row = {
            "archetype": key,
            "label": archetype_label(key),
            "n_live": n,
            "share_live": n / n_live_total,
            "hit_rate_live": (wins / n) if n else None,
            "pnl_pts_live": pnl_pts,
            "expectancy_live": pnl_pts / n if n else None,
            "n_train": int(train["n"]) if train else 0,
            "share_train": float(train["share"]) if train else None,
            "hit_rate_train": float(train["hit_rate"]) if train else None,
            "expectancy_train": (float(train["expectancy_pts"])
                                 if train and train.get("expectancy_pts") is not None
                                 else None),
        }
        hr_l = row["hit_rate_live"]
        hr_t = row["hit_rate_train"]
        row["hit_rate_delta"] = (
            (hr_l - hr_t) if hr_l is not None and hr_t is not None else None
        )
        agg.append(row)
    agg.sort(key=lambda r: r["pnl_pts_live"], reverse=True)
    return tagged, agg
