"""N-base output-meta: binary trade/no-trade gate on upstream prediction vectors.

Generic over ``topology.bases`` — no catalog family names in this module.
Each base OOF must carry ``ts`` plus either full ``p_down/p_flat/p_up`` or
``direction`` + ``confidence``. Meta never sees in-sample base homework: callers
pass walk-forward OOF frames. Meta itself is cross-fit with time blocks + embargo.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.search.output_meta")

_DIRS = ("down", "flat", "up")


def topology_output_meta(objective: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``{bases, meta}`` or None if mode is not ``output_meta``."""
    topo = (objective.get("search") or {}).get("topology") or {}
    if topo.get("mode") != "output_meta":
        return None
    bases = [str(b) for b in (topo.get("bases") or []) if str(b).strip()]
    if not bases:
        return None
    meta = {
        "family": "logistic_regression",
        "approve_threshold": 0.55,
        "n_folds": 5,
        "embargo_bars": 0,
        "context_view": None,
        **(topo.get("meta") or {}),
    }
    return {"bases": bases, "meta": meta}


def compose(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[str],
    approve_threshold: float = 0.55,
    n_folds: int = 5,
    embargo_bars: int = 0,
    context: pd.DataFrame | None = None,
    meta_family: str = "logistic_regression",
) -> pd.DataFrame:
    """Align base OOF on ts, cross-fit a binary gate, return composed canonical OOF."""
    aligned = align_on_ts(base_oofs, bases)
    if aligned.empty:
        raise ValueError("output_meta: no overlapping ts across bases")
    if context is not None and len(context):
        aligned = _join_context(aligned, context)
    side = proposed_side(aligned, bases)
    aligned["proposed_side"] = side
    y = _meta_target(side, aligned.get("direction_true"))
    X, feat_names = build_X_meta(aligned, bases)
    train_mask = np.array([s in ("up", "down") for s in side], dtype=bool)
    meta_prob = np.zeros(len(aligned), dtype="float64")
    if int(train_mask.sum()) < max(20, n_folds * 4):
        logger.warning("output_meta: too few proposed sides (%d) — all-flat gate",
                       int(train_mask.sum()))
    else:
        idx = np.where(train_mask)[0]
        X_tr = X[train_mask]
        y_tr = y[train_mask]
        folds = _embargo_time_folds(len(idx), n_folds=n_folds, embargo=int(embargo_bars))
        oof = np.full(len(idx), np.nan, dtype="float64")
        for te, tr in folds:
            if len(tr) == 0 or len(te) == 0 or len(np.unique(y_tr[tr])) < 2:
                continue
            clf = fit_meta_estimator(X_tr[tr], y_tr[tr], family=meta_family)
            oof[te] = _predict_meta_proba(clf, X_tr[te])
        # leftover rows (empty fold / single class) — fit on all other train rows
        miss = np.isnan(oof)
        if miss.any() and (~miss).sum() >= 4 and len(np.unique(y_tr[~miss])) >= 2:
            clf = fit_meta_estimator(X_tr[~miss], y_tr[~miss], family=meta_family)
            oof[miss] = _predict_meta_proba(clf, X_tr[miss])
        oof = np.nan_to_num(oof, nan=0.0)
        meta_prob[idx] = oof

    approved = train_mask & (meta_prob >= float(approve_threshold))
    out = pd.DataFrame({
        "ts": aligned["ts"].to_numpy(),
        "direction": np.where(approved, side, "flat"),
        "confidence": np.where(approved, meta_prob, 0.0),
        "proposed_side": side,
        "meta_prob": meta_prob,
        "direction_true": aligned["direction_true"].to_numpy()
        if "direction_true" in aligned.columns else None,
    })
    out.attrs["meta_feature_names"] = feat_names
    logger.info(
        "output_meta: n=%d proposed=%d approved=%d thr=%.2f",
        len(out), int(train_mask.sum()), int(approved.sum()), float(approve_threshold),
    )
    return out


def fit_final_meta(
    base_oofs: dict[str, pd.DataFrame],
    *,
    bases: list[str],
    meta_family: str = "logistic_regression",
    context: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Fit one gate on all proposed-side rows (finalize). Returns estimator + names."""
    aligned = align_on_ts(base_oofs, bases)
    if context is not None and len(context):
        aligned = _join_context(aligned, context)
    side = proposed_side(aligned, bases)
    y = _meta_target(side, aligned.get("direction_true"))
    X, feat_names = build_X_meta(aligned, bases)
    train_mask = np.array([s in ("up", "down") for s in side], dtype=bool)
    if int(train_mask.sum()) < 8 or len(np.unique(y[train_mask])) < 2:
        raise ValueError("output_meta: not enough proposed-side rows to fit final gate")
    clf = fit_meta_estimator(X[train_mask], y[train_mask], family=meta_family)
    return {"estimator": clf, "feature_names": feat_names, "family": meta_family,
            "bases": list(bases)}


def align_on_ts(base_oofs: dict[str, pd.DataFrame], bases: list[str]) -> pd.DataFrame:
    """Inner-join base OOF frames on ``ts``. Prefixes per-base columns with ``{name}__``."""
    frames: list[pd.DataFrame] = []
    true_col = None
    for name in bases:
        if name not in base_oofs:
            raise ValueError(f"output_meta: missing OOF for base '{name}'")
        raw = base_oofs[name].copy()
        if "ts" not in raw.columns:
            raise ValueError(f"output_meta: base '{name}' OOF has no ts")
        raw["ts"] = pd.to_datetime(raw["ts"], utc=True)
        raw = raw.dropna(subset=["ts"]).drop_duplicates(subset=["ts"], keep="last")
        keep = ["ts"]
        for c in ("direction", "confidence", "p_down", "p_flat", "p_up"):
            if c in raw.columns:
                keep.append(c)
        piece = raw[keep].rename(columns={c: f"{name}__{c}" for c in keep if c != "ts"})
        if "direction_true" in raw.columns and true_col is None:
            true_col = raw[["ts", "direction_true"]]
        frames.append(piece)
    out = frames[0]
    for nxt in frames[1:]:
        out = out.merge(nxt, on="ts", how="inner")
    if true_col is not None:
        out = out.merge(true_col.drop_duplicates("ts"), on="ts", how="left")
    return out.sort_values("ts").reset_index(drop=True)


def proposed_side(aligned: pd.DataFrame, bases: list[str]) -> np.ndarray:
    """Resolve a side from base ``direction`` columns (no market features)."""
    n = len(aligned)
    dirs = np.stack(
        [aligned[f"{b}__direction"].astype(str).to_numpy() for b in bases], axis=1,
    )
    confs = []
    for b in bases:
        col = f"{b}__confidence"
        if col in aligned.columns:
            confs.append(aligned[col].to_numpy(dtype="float64"))
        else:
            confs.append(np.zeros(n, dtype="float64"))
    confs_a = np.stack(confs, axis=1)
    out = np.empty(n, dtype=object)
    for i in range(n):
        row = dirs[i]
        directional = [(j, row[j]) for j in range(len(bases)) if row[j] in ("up", "down")]
        if not directional:
            out[i] = None
            continue
        sides = {d for _j, d in directional}
        if len(sides) == 1:
            out[i] = next(iter(sides))
            continue
        # conflict → no proposed side
        out[i] = None
    return out


def build_X_meta(aligned: pd.DataFrame, bases: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-base ``[p_down, p_flat, p_up, conf, 1{up}, 1{down}]`` plus agree / n_dir."""
    blocks: list[np.ndarray] = []
    names: list[str] = []
    dir_cols = []
    for b in bases:
        p_down, p_flat, p_up, conf, is_up, is_dn = _base_vector(aligned, b)
        blocks.extend([p_down, p_flat, p_up, conf, is_up, is_dn])
        names.extend([
            f"{b}__p_down", f"{b}__p_flat", f"{b}__p_up",
            f"{b}__confidence", f"{b}__is_up", f"{b}__is_down",
        ])
        dir_cols.append(aligned[f"{b}__direction"].astype(str).to_numpy())
    dirs = np.stack(dir_cols, axis=1)
    n_dir = np.sum(np.isin(dirs, ("up", "down")), axis=1).astype("float64")
    # agree_flag: all directional votes share one side
    agree = np.zeros(len(aligned), dtype="float64")
    for i in range(len(aligned)):
        sides = {d for d in dirs[i] if d in ("up", "down")}
        if len(sides) == 1 and n_dir[i] >= 2:
            agree[i] = 1.0
    blocks.extend([n_dir, agree])
    names.extend(["n_directional", "agree_side"])
    extra = [c for c in aligned.columns
             if not c.startswith(tuple(f"{b}__" for b in bases))
             and c not in ("ts", "direction_true", "proposed_side")]
    for c in extra:
        if pd.api.types.is_numeric_dtype(aligned[c]):
            arr = aligned[c].to_numpy(dtype="float64")
            arr = np.where(np.isfinite(arr), arr, 0.0)
            blocks.append(arr)
            names.append(c)
    X = np.column_stack(blocks) if blocks else np.zeros((len(aligned), 0))
    return X, names


def fit_meta_estimator(X: np.ndarray, y: np.ndarray, *, family: str) -> Any:
    fam = (family or "logistic_regression").lower()
    if fam in ("xgboost", "xgb"):
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            n_jobs=1, verbosity=0,
        )
        clf.fit(X, y)
        return clf
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X, y)
    return clf


def _predict_meta_proba(clf: Any, X: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(X)
    classes = list(getattr(clf, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def _base_vector(aligned: pd.DataFrame, name: str) -> tuple[np.ndarray, ...]:
    n = len(aligned)
    direction = aligned[f"{name}__direction"].astype(str).to_numpy()
    conf_col = f"{name}__confidence"
    conf = (aligned[conf_col].to_numpy(dtype="float64") if conf_col in aligned.columns
            else np.zeros(n))
    if all(f"{name}__{c}" in aligned.columns for c in ("p_down", "p_flat", "p_up")):
        p_down = aligned[f"{name}__p_down"].to_numpy(dtype="float64")
        p_flat = aligned[f"{name}__p_flat"].to_numpy(dtype="float64")
        p_up = aligned[f"{name}__p_up"].to_numpy(dtype="float64")
    else:
        p_down = np.where(direction == "down", conf, 0.0)
        p_flat = np.where(direction == "flat", conf, 0.0)
        p_up = np.where(direction == "up", conf, 0.0)
    is_up = (direction == "up").astype("float64")
    is_dn = (direction == "down").astype("float64")
    return p_down, p_flat, p_up, conf, is_up, is_dn


def _meta_target(side: np.ndarray, direction_true: Any) -> np.ndarray:
    y = np.zeros(len(side), dtype=np.int8)
    if direction_true is None:
        return y
    truth = np.asarray(direction_true, dtype=object)
    for i, s in enumerate(side):
        if s in ("up", "down") and str(truth[i]) == s:
            y[i] = 1
    return y


def _join_context(aligned: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    ctx = context.copy()
    if "ts" in ctx.columns:
        ctx["ts"] = pd.to_datetime(ctx["ts"], utc=True)
        ctx = ctx.set_index("ts")
    if not isinstance(ctx.index, pd.DatetimeIndex):
        raise ValueError("context must be ts-indexed or have a ts column")
    if ctx.index.tz is None:
        ctx.index = ctx.index.tz_localize("UTC")
    else:
        ctx.index = ctx.index.tz_convert("UTC")
    num = ctx.select_dtypes(include=[np.number])
    joined = num.reindex(aligned["ts"])
    extra = joined.reset_index(drop=True)
    return pd.concat([aligned.reset_index(drop=True), extra], axis=1)


def _embargo_time_folds(
    n: int, *, n_folds: int, embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_folds = max(2, min(int(n_folds), n // 4 if n >= 8 else 2))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    embargo = max(0, int(embargo))
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        te = np.arange(edges[k], edges[k + 1])
        lo = max(0, edges[k] - embargo)
        hi = min(n, edges[k + 1] + embargo)
        tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        if len(te) and len(tr):
            out.append((te, tr))
    return out
