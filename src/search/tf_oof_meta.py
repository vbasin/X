"""Native-timeframe dual-head OOF meta (ADR-066 / ADR-092 / ADR-093).

Joint search trains each topology base on its own hive, then this composer
asof-aligns OOF onto the 15s decision grid, joins 1h/1d context, and fits two
independent XGBs for ``p_final_long`` / ``p_final_short``. OOF-only recomposes
from named Custom Test runs without retraining bases.

    python -m src.search.tf_oof_meta \\
        --objective nq-50pt-econ-lg-st \\
        --base lstm_15s=20260828T175240Z \\
        --base xgb_1m=20260828T200724Z \\
        --base xgb_5m=20260828T192637Z
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.htf_context import (
    CONTEXT_COLS,
    asof_attach_context,
    load_htf_frames,
)
from src.models.dual_binary import DUAL_BINARY_FIRE, dual_binary_fire, dual_binary_to_canonical

logger = logging.getLogger("vtech.search.tf_oof_meta")

_PREDICTIVE_TF = frozenset({"15s", "1m", "5m"})
_CONTEXT_TF = frozenset({"1h", "1d"})

_TF_TOLERANCE = {
    "15s": pd.Timedelta("15s"),
    "1m": pd.Timedelta("60s"),
    "5m": pd.Timedelta("5min"),
    "1h": pd.Timedelta("65min"),
    "1d": pd.Timedelta("26h"),
}

# Slim HTF pack lives in src.features.htf_context (ADR-095). Re-exported.

_DEFAULT_XGB: dict[str, Any] = {
    "n_estimators": 80,
    "max_depth": 3,
    "learning_rate": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 4,
    "verbosity": 0,
    "tree_method": "hist",
}


def topology_tf_oof_meta(objective: dict[str, Any]) -> dict[str, Any] | None:
    """Return topology payload, or None if mode is not ``tf_oof_meta``."""
    topo = (objective.get("search") or {}).get("topology") or {}
    if topo.get("mode") != "tf_oof_meta":
        return None
    bases: list[dict[str, Any]] = []
    for b in list(topo.get("bases") or []):
        if isinstance(b, str) and b.strip():
            bases.append({"id": b.strip(), "family": b.strip()})
            continue
        if not isinstance(b, dict):
            continue
        bid = str(b.get("id") or "").strip()
        family = str(b.get("family") or "").strip()
        native_tf = str(b.get("native_tf") or "").strip()
        if not bid or not family or native_tf not in _PREDICTIVE_TF:
            continue
        row = {
            "id": bid,
            "family": family,
            "native_tf": native_tf,
            "head": str(b.get("head") or "dual_binary"),
            "role": str(b.get("role") or "predictive"),
        }
        raw_p = b.get("params")
        if isinstance(raw_p, dict) and raw_p:
            row["params"] = {str(k): v for k, v in raw_p.items() if v is not None}
        bases.append(row)
    if not bases:
        return None
    context = []
    for c in list(topo.get("context") or []):
        if not isinstance(c, dict):
            continue
        native_tf = str(c.get("native_tf") or "").strip()
        if native_tf not in _CONTEXT_TF:
            continue
        context.append({
            "native_tf": native_tf,
            "role": str(c.get("role") or "context"),
        })
    wf_folds = int((objective.get("eval") or {}).get("walk_forward_folds") or 3)
    meta = {
        "family": "xgboost",
        "scheme": "walk_forward_OOF",
        "n_folds": wf_folds,
        "embargo_bars": 0,
        "outputs": ["p_final_long", "p_final_short"],
        **(topo.get("meta") or {}),
    }
    return {
        "bases": bases,
        "context": context,
        "meta": meta,
        "outputs": list(topo.get("outputs") or meta.get("outputs") or [
            "p_final_long", "p_final_short",
        ]),
        "decision_timestep": str(
            (objective.get("search") or {}).get("decision_timestep") or "15s"
        ),
    }


def required_tf_oof_stores(objective: dict[str, Any]) -> list[str]:
    """Predictive + context timeframes the joint search needs on disk."""
    cfg = topology_tf_oof_meta(objective)
    if cfg is None:
        return []
    stores = [str(b.get("native_tf") or "") for b in cfg["bases"]]
    stores.extend(str(c.get("native_tf") or "") for c in cfg["context"])
    return [s for s in stores if s]


def missing_tf_oof_stores(objective: dict[str, Any]) -> list[str]:
    from src.features.stores import store_available
    return [s for s in required_tf_oof_stores(objective) if not store_available(s)]


def objective_for_store(objective: dict[str, Any], store: str) -> dict[str, Any]:
    """Copy of *objective* with ``topology.feature_store`` set (fold cache key)."""
    out = dict(objective)
    search = dict(out.get("search") or {})
    topo = dict(search.get("topology") or {})
    topo["feature_store"] = str(store)
    search["topology"] = topo
    out["search"] = search
    return out


def iter_tf_oof_bases(objective: dict[str, Any]) -> list[tuple[Any, str]]:
    """``(CatalogEntry, native_tf)`` for each predictive topology base."""
    from dataclasses import replace
    from src.models.catalog import get_entry

    cfg = topology_tf_oof_meta(objective)
    if cfg is None:
        return []
    out: list[tuple[Any, str]] = []
    for b in cfg["bases"]:
        family = str(b.get("family") or "")
        bid = str(b.get("id") or family)
        tf = str(b.get("native_tf") or "")
        base = get_entry(family)
        params = dict(base.params_default)
        for k, v in (b.get("params") or {}).items():
            if v is not None:
                params[str(k)] = v
        params["head"] = str(b.get("head") or params.get("head") or "dual_binary")
        entry = replace(
            base,
            name=bid,
            plugin_family=base.family_key,
            params_default=params,
            search_max_rows=None,
            search_space={},
            notes=f"tf_oof_meta base native_tf={tf}",
        )
        out.append((entry, tf))
    return out


def match_tf_oof_family(entry: Any, store: str, families: list[str] | None) -> bool:
    """True when *entry* is in a ``--families`` whitelist (id, family key, or store)."""
    if not families:
        return True
    want = {str(f).strip().lower() for f in families if str(f).strip()}
    if not want:
        return True
    names = {
        str(getattr(entry, "name", "") or "").lower(),
        str(getattr(entry, "family_key", "") or "").lower(),
        str(store or "").lower(),
    }
    return bool(want & names)


def assert_tf_oof_meta_ready(
    objective: dict[str, Any],
    *,
    experiment_type: str | None = None,
) -> None:
    """Raise if native-TF dual-head stack cannot run.

    Custom Test with meta off and an available feature store is allowed
    (ADR-076). Joint search (ADR-093) is allowed when every topology store
    is on disk. OOF-only compose remains ``python -m src.search.tf_oof_meta``.
    """
    search = objective.get("search") or {}
    inp = str(search.get("input") or "")
    topo = search.get("topology") or {}
    mode = str(topo.get("mode") or "")
    if inp != "native_tf" and topology_tf_oof_meta(objective) is None:
        return
    if str(experiment_type or "") == "custom_test" and mode in ("none", "off", ""):
        from src.features.stores import feature_store_id, store_available
        sid = feature_store_id(objective)
        if store_available(sid):
            return
        raise NotImplementedError(
            f"{objective.get('objective_id', '?')}: Custom Test feature store "
            f"{sid!r} is not available (no parquet files)."
        )
    oid = objective.get("objective_id", "?")
    cfg = topology_tf_oof_meta(objective)
    if cfg is not None:
        missing = missing_tf_oof_stores(objective)
        if not missing:
            return
        raise NotImplementedError(
            f"{oid}: tf_oof_meta stores missing: {missing}. "
            "Need native 15s/1m/5m bases and 1h/1d context hive files."
        )
    raise NotImplementedError(
        f"{oid}: search.input=native_tf requires topology.mode=tf_oof_meta "
        "or Custom Test mode none with an available store (ADR-066 / ADR-076 / "
        "ADR-093). Live pin unchanged."
    )


# ── Alignment ─────────────────────────────────────────────────────────────────

def infer_native_tf(df: pd.DataFrame) -> str:
    """Guess bar width from median ``ts`` spacing."""
    ts = pd.to_datetime(df["ts"], utc=True).sort_values()
    dt = ts.diff().median()
    if pd.isna(dt):
        return "15s"
    sec = float(dt.total_seconds())
    if sec <= 20:
        return "15s"
    if sec <= 90:
        return "1m"
    if sec <= 400:
        return "5m"
    if sec <= 4000:
        return "1h"
    return "1d"


def _prep_ts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if "ts" not in out.columns:
                out = out.rename(columns={out.columns[0]: "ts"})
        else:
            raise ValueError("OOF frame has no ts")
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = out.dropna(subset=["ts"]).sort_values("ts")
    out = out.drop_duplicates(subset=["ts"], keep="last")
    return out.reset_index(drop=True)


def _asof_right(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    cols: list[str],
    tolerance: pd.Timedelta,
    prefix: str,
) -> pd.DataFrame:
    piece = right[["ts", *cols]].copy()
    rename = {c: f"{prefix}{c}" for c in cols}
    piece = piece.rename(columns=rename)
    return pd.merge_asof(
        left, piece, on="ts", direction="backward", tolerance=tolerance,
    )


def align_native_oof(
    base_oofs: dict[str, pd.DataFrame],
    bases: list[dict[str, Any]],
    *,
    decision_timestep: str = "15s",
    context: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Causal asof-join of native-TF OOF onto the decision grid.

    Predictive bases must all be present. A row is kept only when every
    predictive base has a last-closed bar within that TF's tolerance.
    1h/1d context is optional; missing context is filled with 0.
    """
    if not bases:
        raise ValueError("tf_oof_meta: no bases")
    specs: list[dict[str, Any]] = []
    for b in bases:
        if isinstance(b, str):
            bid = b.strip()
            if bid not in base_oofs:
                raise ValueError(f"tf_oof_meta: missing OOF for base '{bid}'")
            specs.append({
                "id": bid,
                "native_tf": infer_native_tf(base_oofs[bid]),
            })
            continue
        bid = str(b.get("id") or "").strip()
        if not bid:
            continue
        if bid not in base_oofs:
            raise ValueError(f"tf_oof_meta: missing OOF for base '{bid}'")
        tf = str(b.get("native_tf") or "").strip() or infer_native_tf(base_oofs[bid])
        specs.append({**b, "id": bid, "native_tf": tf})
    if not specs:
        raise ValueError("tf_oof_meta: no usable bases")

    grid_id = None
    for s in specs:
        if s["native_tf"] == decision_timestep:
            grid_id = s["id"]
            break
    if grid_id is None:
        grid_id = min(specs, key=lambda s: _TF_TOLERANCE[s["native_tf"]])["id"]

    grid_raw = _prep_ts(base_oofs[grid_id])
    if "direction_true" not in grid_raw.columns:
        grid_raw = grid_raw.copy()
        grid_raw["direction_true"] = "flat"
    keep = ["ts"]
    for c in ("p_long", "p_short", "direction_true"):
        if c in grid_raw.columns:
            keep.append(c)
    if "p_long" not in keep or "p_short" not in keep:
        raise ValueError(f"tf_oof_meta: base '{grid_id}' OOF missing p_long/p_short")
    aligned = grid_raw[keep].rename(columns={
        "p_long": f"{grid_id}__p_long",
        "p_short": f"{grid_id}__p_short",
    })

    for s in specs:
        bid = s["id"]
        if bid == grid_id:
            continue
        raw = _prep_ts(base_oofs[bid])
        if "p_long" not in raw.columns or "p_short" not in raw.columns:
            raise ValueError(f"tf_oof_meta: base '{bid}' OOF missing p_long/p_short")
        tf = s["native_tf"]
        tol = _TF_TOLERANCE.get(tf, pd.Timedelta("60s"))
        aligned = _asof_right(
            aligned, raw, cols=["p_long", "p_short"],
            tolerance=tol, prefix=f"{bid}__",
        )

    pred_cols = []
    for s in specs:
        bid = s["id"]
        for c in (f"{bid}__p_long", f"{bid}__p_short"):
            if c not in aligned.columns:
                raise ValueError(f"tf_oof_meta: alignment lost {c}")
            pred_cols.append(c)
    aligned = aligned.dropna(subset=pred_cols).reset_index(drop=True)

    for s in specs:
        bid = s["id"]
        pl = aligned[f"{bid}__p_long"].to_numpy(dtype="float64")
        ps = aligned[f"{bid}__p_short"].to_numpy(dtype="float64")
        aligned[f"{bid}__spread"] = pl - ps
        aligned[f"{bid}__p_max"] = np.maximum(pl, ps)

    aligned = _attach_time_features(aligned)
    if context:
        aligned = asof_attach_context(aligned, context)

    extra = [c for c in aligned.columns
             if c not in ("ts", "direction_true")
             and pd.api.types.is_numeric_dtype(aligned[c])]
    aligned[extra] = aligned[extra].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return aligned.sort_values("ts").reset_index(drop=True)


def _attach_time_features(aligned: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(aligned["ts"])
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    ct = ts.tz_convert("America/Chicago")
    midnight = ct.normalize()
    session_open = midnight + pd.Timedelta(hours=8, minutes=30)
    out = aligned.copy()
    out["tod_min"] = (ct - midnight).total_seconds() / 60.0
    out["session_min"] = (ct - session_open).total_seconds() / 60.0
    out["dow"] = ct.dayofweek.astype("float64")
    return out


def _attach_context(
    aligned: pd.DataFrame,
    context: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    return asof_attach_context(aligned, context)


def feature_matrix(
    aligned: pd.DataFrame,
    *,
    skip_feature_prefixes: tuple[str, ...] | list[str] = (),
) -> tuple[np.ndarray, list[str]]:
    """Numeric meta features; drops ``ts`` / labels."""
    skip = {"ts", "direction_true", "direction", "confidence",
            "p_long", "p_short", "p_final_long", "p_final_short",
            "meta_finite", "meta_block"}
    prefixes = tuple(p for p in skip_feature_prefixes if p)
    names: list[str] = []
    blocks: list[np.ndarray] = []
    for c in aligned.columns:
        if c in skip:
            continue
        if prefixes and any(c.startswith(p) for p in prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(aligned[c]):
            continue
        arr = aligned[c].to_numpy(dtype="float64")
        arr = np.where(np.isfinite(arr), arr, 0.0)
        blocks.append(arr)
        names.append(c)
    if not blocks:
        return np.zeros((len(aligned), 0)), []
    return np.column_stack(blocks), names


def select_meta_features(aligned: pd.DataFrame, names: list[str]) -> np.ndarray:
    """Build X in *names* order; missing / non-finite columns are 0."""
    n = len(aligned)
    if not names:
        return np.zeros((n, 0), dtype="float64")
    cols: list[np.ndarray] = []
    for name in names:
        if name in aligned.columns and pd.api.types.is_numeric_dtype(aligned[name]):
            arr = aligned[name].to_numpy(dtype="float64")
        else:
            arr = np.zeros(n, dtype="float64")
        cols.append(np.where(np.isfinite(arr), arr, 0.0))
    return np.column_stack(cols)


def load_store_features_for_serve(
    store: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Hive frame for one native store, with HTF / session VP attached as trained.

    Pads two calendar days so 5m Globex-17:00 SVP and 1h asof have lookback.
    """
    from src.features.stores import load_feature_frame

    t0 = pd.Timestamp(start)
    t1 = pd.Timestamp(end)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    else:
        t1 = t1.tz_convert("UTC")
    df = load_feature_frame(store, (t0 - pd.Timedelta(days=2)).date(), t1.date())
    if df is None or df.empty:
        raise FileNotFoundError(
            f"tf_oof_meta serve: no {store} hive rows {t0.date()}→{t1.date()}")
    cols = [str(c) for c in (feature_cols or [])]
    if any(c.startswith("svp_") for c in cols):
        from src.features.session_vp import attach_session_vp
        df = attach_session_vp(df)
    if any(c.startswith(("h1_", "d1_")) for c in cols):
        from src.features.htf_context import attach_htf_to_frame
        df = attach_htf_to_frame(df)
    return df


# ── Walk-forward dual XGB ─────────────────────────────────────────────────────

def _xgb_params(overrides: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(_DEFAULT_XGB)
    if overrides:
        p.update({k: v for k, v in overrides.items() if v is not None})
    return p


def _fit_dual_xgb(
    X: np.ndarray,
    y_long: np.ndarray,
    y_short: np.ndarray,
    *,
    params: dict[str, Any],
    class_balance: str,
) -> Any:
    from xgboost import XGBClassifier
    from src.models.tabular_plugins import DualBinaryPair, _make_sample_weight

    y_l = np.asarray(y_long).astype("int32").reshape(-1)
    y_s = np.asarray(y_short).astype("int32").reshape(-1)
    long_clf = XGBClassifier(**params)
    short_clf = XGBClassifier(**params)
    sw_l = _make_sample_weight(y_l, class_balance)
    sw_s = _make_sample_weight(y_s, class_balance)
    if sw_l is not None:
        long_clf.fit(X, y_l, sample_weight=sw_l)
    else:
        long_clf.fit(X, y_l)
    if sw_s is not None:
        short_clf.fit(X, y_s, sample_weight=sw_s)
    else:
        short_clf.fit(X, y_s)
    return DualBinaryPair(long_clf, short_clf)


def _block_ids(
    ts: pd.Series,
    *,
    n_folds: int,
    fold_starts: list[pd.Timestamp] | None,
) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True)
    n = len(t)
    if fold_starts:
        edges = [pd.Timestamp(s) for s in fold_starts]
        edges = [
            e.tz_localize("UTC") if e.tzinfo is None else e.tz_convert("UTC")
            for e in edges
        ]
        block = np.full(n, -1, dtype=int)
        for i, t0 in enumerate(edges):
            t1 = edges[i + 1] if i + 1 < len(edges) else None
            m = t >= t0
            if t1 is not None:
                m &= t < t1
            block[m.to_numpy()] = i
        return block
    n_folds = max(2, int(n_folds))
    order = np.arange(n)
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    block = np.full(n, -1, dtype=int)
    for k in range(n_folds):
        block[order[edges[k]:edges[k + 1]]] = k
    return block


def compose(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[dict[str, Any]] | list[str],
    decision_timestep: str = "15s",
    context: dict[str, pd.DataFrame] | None = None,
    n_folds: int = 3,
    embargo_bars: int = 40,
    class_balance: str = "sqrt",
    xgb_params: dict[str, Any] | None = None,
    fold_starts: list[pd.Timestamp] | None = None,
    fire: float = DUAL_BINARY_FIRE,
    skip_feature_prefixes: tuple[str, ...] | list[str] = (),
) -> pd.DataFrame:
    """Walk-forward dual-binary XGB on aligned native-TF OOF.

    Block 0 is warmup (no meta fires). Block k is scored by a model fit on
    blocks ``0..k-1`` with ``embargo_bars`` dropped from the train tail.
    Returns a canonical dual-binary OOF plus ``p_final_*``, ``meta_finite``,
    ``meta_block``. Warmup / skipped blocks have ``p=0`` and stay flat.
    """
    specs = list(bases)
    aligned = align_native_oof(
        base_oofs, specs,
        decision_timestep=decision_timestep, context=context,
    )
    if aligned.empty:
        raise ValueError("tf_oof_meta: no overlapping ts across bases")
    X, feat_names = feature_matrix(
        aligned, skip_feature_prefixes=skip_feature_prefixes)
    truth = aligned["direction_true"].astype(str).to_numpy()
    y_long = (truth == "up").astype("int32")
    y_short = (truth == "down").astype("int32")
    block = _block_ids(aligned["ts"], n_folds=n_folds, fold_starts=fold_starts)
    n_folds_eff = int(max(block.max() + 1, 2)) if (block >= 0).any() else 2
    params = _xgb_params(xgb_params)
    p_long = np.full(len(aligned), np.nan, dtype="float64")
    p_short = np.full(len(aligned), np.nan, dtype="float64")
    embargo = max(0, int(embargo_bars))

    for k in range(1, n_folds_eff):
        te = np.where(block == k)[0]
        tr = np.where((block >= 0) & (block < k))[0]
        if embargo and len(tr):
            tr = tr[: max(0, len(tr) - embargo)]
        if len(te) == 0 or len(tr) < 20:
            logger.warning("tf_oof_meta block=%d skipped: train=%d test=%d",
                           k, len(tr), len(te))
            continue
        if len(np.unique(y_long[tr])) < 2 or len(np.unique(y_short[tr])) < 2:
            logger.warning("tf_oof_meta block=%d skipped: single-class train", k)
            continue
        clf = _fit_dual_xgb(
            X[tr], y_long[tr], y_short[tr],
            params=params, class_balance=class_balance,
        )
        pl, ps = clf.predict_pair(X[te])
        p_long[te] = pl
        p_short[te] = ps

    finite = np.isfinite(p_long) & np.isfinite(p_short)
    pl_out = np.where(finite, p_long, 0.0)
    ps_out = np.where(finite, p_short, 0.0)
    out = dual_binary_to_canonical(
        pl_out, ps_out, ts=aligned["ts"].to_numpy(),
        y_true=np.column_stack([y_long, y_short]), fire=fire,
    )
    out.loc[~finite, "direction"] = "flat"
    out.loc[~finite, "confidence"] = 0.0
    out["p_final_long"] = pl_out
    out["p_final_short"] = ps_out
    out["meta_finite"] = finite
    out["meta_block"] = block
    out["direction_true"] = truth
    out.attrs["meta_feature_names"] = feat_names
    out.attrs["n_folds"] = n_folds_eff
    logger.info(
        "tf_oof_meta n=%d finite=%d fired=%d n_folds=%d embargo=%d",
        len(out), int(finite.sum()),
        int(out.loc[finite, "direction"].isin(["up", "down"]).sum()),
        n_folds_eff, embargo,
    )
    return out


def fit_final(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[dict[str, Any]] | list[str],
    decision_timestep: str = "15s",
    context: dict[str, pd.DataFrame] | None = None,
    class_balance: str = "sqrt",
    xgb_params: dict[str, Any] | None = None,
    exclude_blocks: list[int] | None = None,
    n_folds: int = 3,
    fold_starts: list[pd.Timestamp] | None = None,
    skip_feature_prefixes: tuple[str, ...] | list[str] = (),
) -> dict[str, Any]:
    """Fit one dual XGB on aligned OOF (optionally dropping holdout blocks)."""
    aligned = align_native_oof(
        base_oofs, list(bases),
        decision_timestep=decision_timestep, context=context,
    )
    X, feat_names = feature_matrix(
        aligned, skip_feature_prefixes=skip_feature_prefixes)
    truth = aligned["direction_true"].astype(str).to_numpy()
    y_long = (truth == "up").astype("int32")
    y_short = (truth == "down").astype("int32")
    mask = np.ones(len(aligned), dtype=bool)
    if exclude_blocks:
        block = _block_ids(aligned["ts"], n_folds=n_folds, fold_starts=fold_starts)
        mask &= ~np.isin(block, list(exclude_blocks))
    if int(mask.sum()) < 20:
        raise ValueError("tf_oof_meta finalize: not enough rows")
    clf = _fit_dual_xgb(
        X[mask], y_long[mask], y_short[mask],
        params=_xgb_params(xgb_params), class_balance=class_balance,
    )
    return {
        "estimator": clf,
        "feature_names": feat_names,
        "family": "xgboost",
        "bases": [b["id"] if isinstance(b, dict) else str(b) for b in bases],
        "n_train": int(mask.sum()),
    }


# ── Custom Test run I/O ───────────────────────────────────────────────────────

def fold_starts_from_walk_forward(path: Path | str) -> list[pd.Timestamp]:
    import yaml
    raw = yaml.safe_load(Path(path).read_text()) or {}
    sched = ((raw.get("walk_forward") or {}).get("schedule") or raw.get("schedule") or [])
    out: list[pd.Timestamp] = []
    for row in sched:
        oof = (row or {}).get("oof") or {}
        start = oof.get("start")
        if start:
            ts = pd.Timestamp(start)
            out.append(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))
    return out


def load_custom_test_oof(
    run_ts: str,
    *,
    objective_id: str,
    candidate: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one base run's OOF. Refuses collided multi-model dirs.

    Resolves research/runs/ first, then the historical roots, so a meta layer
    can still read a base run that predates the cutover.
    """
    from src.common.run_layout import resolve_run_dir

    run_dir = resolve_run_dir(objective_id, run_ts)
    models = run_dir / "models"
    if not models.is_dir():
        raise FileNotFoundError(f"no models/ under {run_dir}")
    cands = [
        p for p in models.iterdir()
        if p.is_dir() and (p / "oof.parquet").is_file()
    ]
    if candidate:
        match = [p for p in cands if p.name == candidate]
        if not match:
            raise FileNotFoundError(
                f"{run_ts}: candidate {candidate!r} not in {[p.name for p in cands]}")
        cands = match
    if len(cands) != 1:
        names = [p.name for p in cands]
        raise ValueError(
            f"{run_ts}: expected exactly one model OOF, found {names}. "
            "Pass candidate= or skip collided dirs (e.g. 20260828T200039Z)."
        )
    cand = cands[0]
    oof = pd.read_parquet(cand / "oof.parquet")
    meta: dict[str, Any] = {
        "run_ts": run_ts,
        "run_dir": str(run_dir),
        "candidate": cand.name,
        "oof_path": str(cand / "oof.parquet"),
    }
    topo_p = run_dir / "topology.yaml"
    if topo_p.exists():
        import yaml
        topo = yaml.safe_load(topo_p.read_text()) or {}
        meta["feature_store"] = topo.get("feature_store")
        meta["topology"] = topo
    wf_p = run_dir / "walk_forward.yaml"
    if wf_p.exists():
        meta["walk_forward_path"] = str(wf_p)
        meta["fold_starts"] = [str(t) for t in fold_starts_from_walk_forward(wf_p)]
    return oof, meta


def load_htf_context(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Load 1h/1d hive frames clipped to [*start* − 5d, *end*]."""
    return load_htf_frames(start, end)


def write_tf_oof_meta_serving_bundle(
    dest: str | Path,
    *,
    objective: dict[str, Any],
    candidate_id: str,
    run_ts: str,
    search_run_dir: str | Path,
    bases: list[dict[str, Any]],
    base_bundle_dirs: dict[str, Path],
    fitted: dict[str, Any],
    composed_oof: pd.DataFrame | None = None,
    search_score: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    spec_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Self-contained Backtest serving folder (ADR-100).

    Copies last-fold bases into ``bases/<id>/``, writes ``meta.pkl`` /
    ``meta.json`` / ``feature_schema.json`` so ``is_serving_bundle`` is true.
    Live pin unchanged — this is a Custom Test lab bundle, not registry.
    """
    import joblib
    import shutil

    from src.models.dual_binary import dual_binary_fire
    from src.search.controller import _git_sha
    from src.search.finalize import _copy_lab_bases

    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    ids = [str(b.get("id") or "").strip() for b in bases if str(b.get("id") or "").strip()]
    missing = [i for i in ids if i not in base_bundle_dirs]
    if missing:
        raise FileNotFoundError(f"tf_oof_meta serving: missing base dirs {missing}")
    saved, _all_cols = _copy_lab_bases(dest_p, base_bundle_dirs, ids)
    by_id = {str(b.get("id") or ""): b for b in bases}
    for spec in saved:
        bid = str(spec.get("family") or "")
        src = Path(base_bundle_dirs[bid])
        bdir = dest_p / "bases" / bid
        man_src = src / "finalize_manifest.json"
        if man_src.exists():
            shutil.copy2(man_src, bdir / "finalize_manifest.json")
        topo = by_id.get(bid) or {}
        spec["id"] = bid
        spec["native_tf"] = str(topo.get("native_tf") or spec.get("native_tf") or "")
        spec["family"] = str(topo.get("family") or spec.get("family") or bid)
        src_man: dict[str, Any] = {}
        if (bdir / "finalize_manifest.json").exists():
            try:
                src_man = json.loads((bdir / "finalize_manifest.json").read_text())
            except json.JSONDecodeError:
                src_man = {}
        if src_man.get("family"):
            spec["plugin_family"] = src_man["family"]
        spec["kind"] = spec.get("kind") or src_man.get("kind") or (
            "sequence" if (bdir / "model.keras").exists() else "tabular")
        spec["seq_len"] = spec.get("seq_len") or src_man.get("seq_len") or (
            (src_man.get("spec_params") or {}).get("seq_len"))

    est = fitted.get("estimator") if isinstance(fitted, dict) else fitted
    names = list(fitted.get("feature_names") or []) if isinstance(fitted, dict) else []
    joblib.dump(est, dest_p / "meta.pkl")
    joblib.dump(
        {
            "estimator": est,
            "feature_names": names,
            "family": (fitted.get("family") if isinstance(fitted, dict) else "xgboost"),
            "bases": ids,
            "n_train": (fitted.get("n_train") if isinstance(fitted, dict) else None),
        },
        dest_p / "meta_estimator.pkl",
    )

    sess = objective.get("session") or {}
    fire = dual_binary_fire(objective)
    decision_tf = str(
        (objective.get("search") or {}).get("decision_timestep") or "15s")
    policy = {
        "tau": fire,
        "trading_start": str(sess.get("trading_start") or "08:30"),
        "trading_end": str(sess.get("trading_end") or "11:00"),
        "timezone": str(sess.get("timezone") or "America/Chicago"),
        "sides": ["up", "down"],
        "note": "ADR-100 tf_oof_meta last-fold asof; fire=labels.fire. Not Live.",
    }
    (dest_p / "inference_policy.json").write_text(
        json.dumps(policy, indent=2, default=str))

    meta_json = {
        "topology": "tf_oof_meta",
        "bases": list(bases),
        "feature_names": names,
        "family": "xgboost",
        "fire": fire,
        "decision_timestep": decision_tf,
        "n_train": fitted.get("n_train") if isinstance(fitted, dict) else None,
        "scheme_at_search_time": "walk_forward_OOF",
        "fire_rule": "argmax AND max(p_long,p_short) >= labels.fire",
        "weights_source": "last_walk_forward_fold",
        "serving": True,
    }
    (dest_p / "meta.json").write_text(json.dumps(meta_json, indent=2, default=str))

    grid_cols: list[str] = []
    for spec in saved:
        if str(spec.get("native_tf") or "") == decision_tf:
            grid_cols = list(spec.get("feature_cols") or [])
            break
    if not grid_cols:
        grid_cols = list(names)
    (dest_p / "feature_schema.json").write_text(json.dumps({
        "feature_cols": grid_cols,
        "n_features": len(grid_cols),
        "meta_feature_names": names,
        "decision_timestep": decision_tf,
        "bases": saved,
    }, indent=2, default=str))

    if composed_oof is not None:
        composed_oof.to_parquet(dest_p / "oof.parquet", index=False)
    if diagnostics:
        (dest_p / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str))

    compose_doc = {
        "topology": "tf_oof_meta",
        "bases": bases,
        "serving": True,
        "n_train": fitted.get("n_train") if isinstance(fitted, dict) else None,
        "feature_names": names,
    }
    (dest_p / "compose.json").write_text(json.dumps(compose_doc, indent=2, default=str))

    oid = str(objective.get("objective_id") or "")
    version = f"{candidate_id}_{run_ts}"
    manifest: dict[str, Any] = {
        "objective_id": oid,
        "candidate_id": candidate_id,
        "family": "tf_oof_meta",
        "kind": "tf_oof_meta",
        "version": version,
        "search_run_ts": run_ts,
        "search_run_dir": str(search_run_dir),
        "search_score": search_score,
        "spec_params": spec_params or {"bases": bases},
        "bases": saved,
        "meta": meta_json,
        "serving": True,
        "decision_timestep": decision_tf,
        "inference_policy": policy,
        "weights_source": "last_walk_forward_fold",
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(dest_p),
        "note": "ADR-100 Custom Test serving asof. Not promoted. Live pin unchanged.",
    }
    (dest_p / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    return manifest


def pack_lab_tf_oof_meta(
    run_dir: str | Path,
    *,
    objective: dict[str, Any] | None = None,
    candidate_id: str | None = None,
) -> Path:
    """Turn an existing joint-search meta folder into a serving bundle."""
    import joblib

    from src.objectives.loader import load_objective
    from src.search.finalize import lab_model_dir

    run_p = Path(run_dir)
    models = run_p / "models"
    if not models.is_dir():
        raise FileNotFoundError(f"no models/ under {run_p}")
    dest = None
    if candidate_id:
        cand = models / candidate_id
        if cand.is_dir():
            dest = cand
    if dest is None:
        hits = sorted(
            p for p in models.iterdir()
            if p.is_dir() and p.name.startswith("tf_oof_meta")
        )
        if not hits:
            raise FileNotFoundError(f"no tf_oof_meta_* under {models}")
        dest = hits[0]
    compose = {}
    if (dest / "compose.json").exists():
        compose = json.loads((dest / "compose.json").read_text())
    oid = str((objective or {}).get("objective_id") or "")
    if not oid and (run_p / "run.json").exists():
        oid = str(json.loads((run_p / "run.json").read_text()).get("objective_id") or "")
    if objective is None:
        if not oid:
            oid = run_p.parent.name
        objective = load_objective(oid)
    bases = list(compose.get("bases") or (objective.get("search") or {}).get("topology", {}).get("bases") or [])
    if not bases:
        raise ValueError(f"tf_oof_meta pack: no bases in {dest / 'compose.json'}")
    fitted_path = dest / "meta_estimator.pkl"
    if not fitted_path.exists():
        raise FileNotFoundError(fitted_path)
    raw = joblib.load(fitted_path)
    if isinstance(raw, dict) and raw.get("estimator") is not None:
        fitted = raw
    else:
        names = list(compose.get("feature_names") or [])
        fitted = {"estimator": raw, "feature_names": names, "n_train": compose.get("n_train")}
    if not fitted.get("feature_names"):
        fitted["feature_names"] = list(compose.get("feature_names") or [])
    ids = [str(b.get("id") or "").strip() for b in bases]
    base_dirs = {i: lab_model_dir(run_p, i) for i in ids}
    missing = [i for i, p in base_dirs.items() if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"tf_oof_meta pack: missing sibling bases {missing}")
    oof = None
    oof_p = dest / "oof.parquet"
    if oof_p.exists():
        oof = pd.read_parquet(oof_p)
    diag = None
    if (dest / "diagnostics.json").exists():
        diag = json.loads((dest / "diagnostics.json").read_text())
    score = None
    sr = run_p / "search_result.json"
    if sr.exists():
        try:
            payload = json.loads(sr.read_text())
            for c in payload.get("candidates") or []:
                if c.get("candidate_id") == dest.name:
                    score = c.get("score")
                    break
        except json.JSONDecodeError:
            score = None
    write_tf_oof_meta_serving_bundle(
        dest,
        objective=objective,
        candidate_id=dest.name,
        run_ts=run_p.name,
        search_run_dir=run_p,
        bases=bases,
        base_bundle_dirs=base_dirs,
        fitted=fitted,
        composed_oof=oof,
        search_score=score,
        diagnostics=diag,
        spec_params={"bases": bases, "meta": ((objective.get("search") or {}).get("topology") or {}).get("meta")},
    )
    return dest


def persist_compose_run(
    composed: pd.DataFrame,
    *,
    objective: dict[str, Any],
    bases: list[dict[str, Any]],
    base_meta: dict[str, dict[str, Any]],
    fold_starts: list[pd.Timestamp] | None,
    fitted: dict[str, Any] | None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a Custom Test-style report dir (not a serving bundle)."""
    from src.common.run_layout import new_run_dir
    from src.evaluation._kpi import canonical_trade_kpi, compute_diagnostics
    from src.search.controller import custom_test_run_doc, _git_sha
    import joblib
    import yaml

    oid = str(objective.get("objective_id") or "nq-50pt-econ-lg-st")
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = new_run_dir(oid, run_ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "models" / "tf_oof_meta"
    dest.mkdir(parents=True, exist_ok=True)

    scored = composed[composed["meta_finite"]].copy() if "meta_finite" in composed.columns else composed
    composed.to_parquet(out_dir / "oof_tf_oof_meta.parquet", index=False)
    composed.to_parquet(dest / "oof.parquet", index=False)

    gt = pd.DataFrame({"direction_true": scored["direction_true"]}) if len(scored) else None
    diag = None
    score: dict[str, Any] = {}
    if gt is not None and len(scored):
        diag = compute_diagnostics(scored, gt, objective, gate_tau=None)
        score = canonical_trade_kpi(scored, gt, objective, tau=None)
        (out_dir / "diagnostics_tf_oof_meta.json").write_text(
            json.dumps(diag, indent=2, default=str))
        (dest / "diagnostics.json").write_text(
            json.dumps(diag, indent=2, default=str))

    holdout_diag = None
    holdout_score: dict[str, Any] = {}
    if "meta_block" in composed.columns and composed["meta_finite"].any():
        last = int(composed.loc[composed["meta_finite"], "meta_block"].max())
        ho = composed[(composed["meta_block"] == last) & composed["meta_finite"]]
        if len(ho):
            hgt = pd.DataFrame({"direction_true": ho["direction_true"]})
            holdout_diag = compute_diagnostics(ho, hgt, objective, gate_tau=None)
            holdout_score = canonical_trade_kpi(ho, hgt, objective, tau=None)
            (out_dir / "diagnostics_tf_oof_meta_holdout.json").write_text(
                json.dumps(holdout_diag, indent=2, default=str))

    if fitted is not None:
        joblib.dump(
            {
                "estimator": fitted["estimator"],
                "feature_names": fitted["feature_names"],
                "family": fitted.get("family"),
                "bases": fitted.get("bases"),
                "n_train": fitted.get("n_train"),
            },
            dest / "meta_estimator.pkl",
        )

    compose_doc = {
        "topology": "tf_oof_meta",
        "objective_id": oid,
        "run_ts": run_ts,
        "bases": bases,
        "base_runs": {
            b["id"]: {
                "run_ts": (base_meta.get(b["id"]) or {}).get("run_ts"),
                "candidate": (base_meta.get(b["id"]) or {}).get("candidate"),
                "feature_store": (base_meta.get(b["id"]) or {}).get("feature_store"),
            }
            for b in bases
        },
        "fold_starts": [str(t) for t in (fold_starts or [])],
        "n_rows": int(len(composed)),
        "n_finite": int(composed["meta_finite"].sum()) if "meta_finite" in composed.columns else int(len(composed)),
        "feature_names": list(composed.attrs.get("meta_feature_names") or []),
        "serving": False,
        "note": (
            "OOF-only meta (ADR-092). Not a Live/Backtest serving bundle "
            "(no feature_schema.json). Live pin unchanged."
        ),
        "git_sha": _git_sha(),
        **(extra or {}),
    }
    (out_dir / "compose.json").write_text(json.dumps(compose_doc, indent=2, default=str))
    (dest / "compose.json").write_text(json.dumps(compose_doc, indent=2, default=str))
    (out_dir / "topology.yaml").write_text(yaml.safe_dump({
        "mode": "tf_oof_meta",
        "bases": bases,
        "meta": (objective.get("search") or {}).get("topology", {}).get("meta") or {},
        "outputs": ["p_final_long", "p_final_short"],
    }, sort_keys=False))
    (out_dir / "run.json").write_text(json.dumps(
        custom_test_run_doc(
            objective_id=oid, run_ts=run_ts, out_dir=out_dir,
            n_candidates=1, n_ok=1, n_failed=0,
        ), indent=2))
    (out_dir / "search_result.json").write_text(json.dumps({
        "candidate_id": "tf_oof_meta",
        "family": "tf_oof_meta",
        "status": "ok",
        "score": score,
        "holdout_score": holdout_score,
    }, indent=2, default=str))
    return out_dir


def _print_kpi_table(title: str, diag: dict[str, Any] | None, score: dict[str, Any]) -> None:
    print(f"\n{title}")
    heads = ((diag or {}).get("heads") or {}) if diag else {}
    print("| Head | AUC | AP | p_max | n_fire@0.5 | Prec@0.5 |")
    print("|------|-----|----|-------|------------|----------|")
    for side in ("long", "short"):
        h = heads.get(side) or {}
        auc = h.get("roc_auc")
        ap = h.get("average_precision")
        pmax = h.get("p_max")
        npos = h.get("n_pred_pos")
        # precision at 0.5 from confusion
        cm = ((h.get("confusion_at_fire") or {}).get("rows_true_cols_pred") or None)
        prec = None
        if cm and len(cm) == 2 and (cm[0][1] + cm[1][1]):
            prec = cm[1][1] / (cm[0][1] + cm[1][1])
        def _f(x):
            return "—" if x is None else f"{float(x):.3f}"
        print(f"| {side:5} | {_f(auc)} | {_f(ap)} | {_f(pmax)} | {npos} | {_f(prec)} |")
    ne = score.get("net_expectancy")
    n = score.get("n_signals")
    print(f"canonical n_signals={n}  NE={ne}  n_up={score.get('n_up')} n_down={score.get('n_down')}")


def _parse_base_arg(raw: str) -> tuple[str, str, str | None]:
    """``id=RUN_TS`` or ``id=RUN_TS:candidate``."""
    if "=" not in raw:
        raise ValueError(f"--base must be id=RUN_TS[:candidate], got {raw!r}")
    bid, rest = raw.split("=", 1)
    if ":" in rest:
        run_ts, cand = rest.split(":", 1)
        return bid.strip(), run_ts.strip(), cand.strip() or None
    return bid.strip(), rest.strip(), None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OOF-only native-TF dual-head meta (ADR-092)")
    parser.add_argument("--objective", default="nq-50pt-econ-lg-st")
    parser.add_argument(
        "--base", action="append", required=True,
        help="id=RUN_TS[:candidate]  (repeat). Decision grid = 15s id.",
    )
    parser.add_argument("--decision-base", default=None,
                        help="Base id for the 15s grid (default: first 15s store).")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--class-balance", default="sqrt")
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument(
        "--skip-features", action="append", default=[],
        help="Drop columns whose name starts with this prefix from X "
             "(repeat). Grid labels still come from the 15s base. "
             "Example: --skip-features lstm_15s__",
    )
    args = parser.parse_args(argv)

    from src.objectives.loader import load_objective
    from src.evaluation._kpi import canonical_trade_kpi, compute_diagnostics

    objective = load_objective(args.objective)
    oid = str(objective.get("objective_id"))
    cfg = topology_tf_oof_meta(objective) or {}
    meta_cfg = cfg.get("meta") or {}
    embargo = int(meta_cfg.get("embargo_bars") or 40)
    n_folds = int(meta_cfg.get("n_folds") or 3)

    parsed = [_parse_base_arg(b) for b in args.base]
    base_oofs: dict[str, pd.DataFrame] = {}
    base_meta: dict[str, dict[str, Any]] = {}
    bases: list[dict[str, Any]] = []
    t0 = time.time()
    print("⏳ Loading base OOF…")
    for bid, run_ts, cand in parsed:
        oof, info = load_custom_test_oof(run_ts, objective_id=oid, candidate=cand)
        tf = str(info.get("feature_store") or infer_native_tf(oof))
        base_oofs[bid] = oof
        base_meta[bid] = info
        bases.append({
            "id": bid,
            "family": info.get("candidate") or bid,
            "native_tf": tf,
            "head": "dual_binary",
            "role": "predictive",
            "run_ts": run_ts,
        })
        print(f"     • {bid:<12} {run_ts}  {info.get('candidate')}  "
              f"tf={tf}  n={len(oof):,}")

    decision = args.decision_base
    if not decision:
        decision = next((b["id"] for b in bases if b["native_tf"] == "15s"), bases[0]["id"])
    fold_starts = fold_starts_from_walk_forward(
        base_meta[decision]["walk_forward_path"]
    ) if base_meta.get(decision, {}).get("walk_forward_path") else None

    context = None
    if not args.no_context:
        print("⏳ Loading 1h / 1d HTF context…")
        grid = _prep_ts(base_oofs[decision])
        context = load_htf_context(grid["ts"].min(), grid["ts"].max())
        for tf, frame in context.items():
            print(f"     • {tf}  n={len(frame):,}")

    xgb_params = {
        "n_estimators": int(args.n_estimators),
        "max_depth": int(args.max_depth),
        "learning_rate": float(args.learning_rate),
    }
    print("⏳ Walk-forward dual-binary XGB on aligned OOF…")
    skip_prefixes = tuple(args.skip_features or ())
    fire = dual_binary_fire(objective)
    composed = compose(
        base_oofs, bases=bases, decision_timestep="15s",
        context=context, n_folds=n_folds, embargo_bars=embargo,
        class_balance=args.class_balance, xgb_params=xgb_params,
        fold_starts=fold_starts, skip_feature_prefixes=skip_prefixes,
        fire=fire,
    )
    print("⏳ Fit final estimator (exclude last block)…")
    last_block = int(composed.loc[composed["meta_finite"], "meta_block"].max()) \
        if composed["meta_finite"].any() else None
    exclude = [last_block] if last_block is not None else None
    fitted = fit_final(
        base_oofs, bases=bases, decision_timestep="15s", context=context,
        class_balance=args.class_balance, xgb_params=xgb_params,
        exclude_blocks=exclude, n_folds=n_folds, fold_starts=fold_starts,
        skip_feature_prefixes=skip_prefixes,
    )
    out_dir = persist_compose_run(
        composed, objective=objective, bases=bases, base_meta=base_meta,
        fold_starts=fold_starts, fitted=fitted,
        extra={"xgb_params": xgb_params, "class_balance": args.class_balance,
               "embargo_bars": embargo, "skip_feature_prefixes": list(skip_prefixes),
               "elapsed_s": time.time() - t0},
    )
    scored = composed[composed["meta_finite"]]
    diag = compute_diagnostics(
        scored, pd.DataFrame({"direction_true": scored["direction_true"]}),
        objective, gate_tau=None,
    )
    score = canonical_trade_kpi(
        scored, pd.DataFrame({"direction_true": scored["direction_true"]}),
        objective, tau=None,
    )
    _print_kpi_table("## OOF (finite walk-forward blocks)", diag, score)
    if last_block is not None:
        ho = composed[(composed["meta_block"] == last_block) & composed["meta_finite"]]
        if len(ho):
            hdiag = compute_diagnostics(
                ho, pd.DataFrame({"direction_true": ho["direction_true"]}),
                objective, gate_tau=None,
            )
            hscore = canonical_trade_kpi(
                ho, pd.DataFrame({"direction_true": ho["direction_true"]}),
                objective, tau=None,
            )
            _print_kpi_table(f"## Untouched test (block {last_block})", hdiag, hscore)
    elapsed = time.time() - t0
    print(f"\n✅ Done in {elapsed:.0f}s → {out_dir}")
    print("   Live pin unchanged. Bundle is report-only (not serving).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
