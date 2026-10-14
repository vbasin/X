"""Feature-store registry (ADR-076 / ADR-081): hive monthly parts per timeframe.

A *store* is a timeframe product (catalog + on-disk layout + loader), not a
single parquet file. Custom Test and the fold builder resolve features through
this module. Every hive timeframe (including 5s) uses
``timeframe={tf}/symbol=NQ/year=/month=/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.data_paths import (
    NATIVE_TF_TIMEFRAMES,
    native_tf_dir,
)

_REPO = Path(__file__).resolve().parents[2]
_CATALOG_DIR = _REPO / "docs" / "reference"

STORE_IDS: tuple[str, ...] = ("5s",) + NATIVE_TF_TIMEFRAMES

STORE_VIEWS: dict[str, dict[str, list[str]]] = {
    "15s": {
        "native_15s": ["of_", "cs_", "tl_", "ss_", "sd_", "lg_", "st_"],
    },
    "1m": {
        "native_1m": ["so_", "cs_", "ps_", "vw_", "rv_", "vp_", "ms_"],
    },
    "5m": {
        "native_5m": ["cs_", "vp_", "vw_", "ps_", "zz_", "vf_", "rv_", "svp_"],
    },
    "1h": {
        "native_1h": ["cs_", "tr_", "ps_", "zz_", "rv_", "sr_"],
    },
    "1d": {
        "native_1d": ["cs_", "tr_", "ps_", "zz_", "rv_", "sr_"],
    },
}

STORE_VIEW_INFO: dict[str, str] = {
    "native_15s": "23 native 15s columns: 14 1-bar + 5 RTH/5-bar flow (ADR-083) + 4 per-head target-geometry (ADR-075 / ADR-082).",
    "native_1m": "71 native 1m columns: so_ / cs_ / ps_ / vw_ / rv_ / vp_ / ms_ (ADR-085; options IV later).",
    "native_5m": "53 hive 5m columns + 11 fold-cache svp_ (Globex 17:00 session POC/VA, ADR-099).",
    "native_1h": "45 native 1h HTF-context columns: cs_ / tr_ / ps_ / zz_ / rv_ / sr_ (ADR-091; trend/vol regime + S/R from tbbo_1h).",
    "native_1d": "45 native 1d HTF-context columns: cs_ / tr_ / ps_ / zz_ / rv_ / sr_ (ADR-091; trend/vol regime + S/R from Globex tbbo_1d).",
}


@dataclass(frozen=True)
class FeatureStore:
    id: str
    catalog_md: Path
    available: bool
    note: str = ""


def _catalog_path(store_id: str) -> Path:
    if store_id == "5s":
        return _CATALOG_DIR / "FEATURE_CATALOG.md"
    if store_id == "15s":
        return _CATALOG_DIR / "FEATURE_CATALOG_15s.md"
    return _CATALOG_DIR / f"FEATURE_CATALOG_{store_id}.md"


def _hive_has_parts(store_id: str) -> bool:
    try:
        root = native_tf_dir(store_id)
    except ValueError:
        return False
    if not root.is_dir():
        return False
    return next(root.glob("year=*/month=*/features_*.parquet"), None) is not None


def store_available(store_id: str) -> bool:
    sid = str(store_id).strip()
    if sid in STORE_IDS:
        return _hive_has_parts(sid)
    return False


def list_feature_stores() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sid in STORE_IDS:
        try:
            avail = store_available(sid)
        except Exception:
            avail = False
        note = ""
        if sid == "5s":
            note = "5s monthly parts (08:00–11:00 CT; ADR-079 / ADR-081)."
        elif sid == "15s":
            note = "Native 15s 1-bar + RTH CVD/cum OFI + target-geometry hive, full Globex (ADR-075 / ADR-082 / ADR-083)."
        elif sid == "1m":
            note = "Native 1m so_/cs_/ps_/vw_/rv_/vp_/ms_ hive, full Globex (ADR-085; no options IV)."
        elif sid == "5m":
            note = "Native 5m cs_/vp_/vw_/ps_/zz_/vf_/rv_ hive, full Globex (ADR-090; VWAP ±2σ; ATR(5) on the 5m bar)."
        elif sid == "1h":
            note = "Native 1h HTF context cs_/tr_/ps_/zz_/rv_/sr_ from tbbo_1h (ADR-091; trend/vol regime + S/R)."
        elif sid == "1d":
            note = "Native 1d HTF context cs_/tr_/ps_/zz_/rv_/sr_ from Globex tbbo_1d (ADR-091; trend/vol regime + S/R)."
        else:
            note = "Hive dirs exist; parquet files not written yet."
        if not avail:
            note = note + " Unavailable until files exist."
        out.append({
            "id": sid,
            "available": avail,
            "catalog": str(_catalog_path(sid).name),
            "note": note,
        })
    return out


def require_store(store_id: str) -> str:
    sid = str(store_id or "5s").strip()
    if sid not in STORE_IDS:
        raise ValueError(f"unknown feature store {store_id!r}; expected one of {STORE_IDS}")
    return sid


def feature_store_id(objective: dict[str, Any] | None) -> str:
    """Board overlay `topology.feature_store`, else native_tf decision_timestep, else 5s."""
    obj = objective or {}
    search = obj.get("search") or {}
    topo = search.get("topology") or {}
    raw = str(topo.get("feature_store") or "").strip()
    if raw:
        return require_store(raw)
    if str(search.get("input") or "") == "native_tf":
        return require_store(str(search.get("decision_timestep") or "15s"))
    return "5s"


def default_head(objective: dict[str, Any] | None) -> str:
    enc = str(((objective or {}).get("labels") or {}).get("encoding") or "").strip()
    if enc == "dual_binary":
        return "dual_binary"
    return "multiclass"


def parquet_dir_for_store(store_id: str) -> Path:
    return native_tf_dir(require_store(store_id))


def default_parquet_dir(objective: dict[str, Any] | None = None) -> Path:
    """Training/Custom Test/GPU default: hive for the objective's store (5s if unset)."""
    return parquet_dir_for_store(feature_store_id(objective))


def iter_store_parquets(store_id: str) -> list[Path]:
    root = native_tf_dir(require_store(store_id))
    return sorted(root.glob("year=*/month=*/features_*.parquet"))


def catalog_path(store_id: str) -> Path:
    return _catalog_path(require_store(store_id))


def views_for_store(store_id: str) -> dict[str, list[str]]:
    sid = require_store(store_id)
    if sid == "5s":
        from src.models.feature_views import FEATURE_VIEWS
        return dict(FEATURE_VIEWS)
    return dict(STORE_VIEWS.get(sid) or {})


def load_feature_frame(
    store_id: str,
    start: str | date | None = None,
    end: str | date | None = None,
) -> pd.DataFrame:
    """Load a store into a UTC DatetimeIndex frame.

    Monthly parts under ``timeframe={store}/``. Index name is ``ts``
    (from ``bar_end_ts``). *start*/*end* are inclusive calendar dates
    (UTC date of the index).
    """
    return _load_native_tf(require_store(store_id), start, end)


def _clip_dates(idx: pd.DatetimeIndex, start, end) -> pd.Series:
    ts = pd.DatetimeIndex(idx)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    mask = pd.Series(True, index=ts)
    if start is not None:
        d0 = pd.Timestamp(start).tz_localize("UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        mask &= ts >= d0.normalize()
    if end is not None:
        d1 = pd.Timestamp(end)
        if d1.tzinfo is None:
            d1 = d1.tz_localize("UTC")
        mask &= ts < (d1.normalize() + pd.Timedelta(days=1))
    return mask


def _load_native_tf(store_id: str, start, end) -> pd.DataFrame:
    from src.features.native_tf import load_hive_frame

    df = load_hive_frame(store_id, start, end)
    if store_id == "15s":
        if "of_delta" in df.columns:
            from src.features.bar_features import attach_rth_session_features
            df = attach_rth_session_features(df, overwrite=False)
        from src.features.target_geometry import attach_target_geometry
        df = attach_target_geometry(df)
    if start is None and end is None:
        return df
    mask = _clip_dates(df.index, start, end)
    return df.loc[mask.to_numpy()]
