"""Canonical direction/confidence inference from a loaded serving bundle.

Single rule used by Live (snapshot + blotter) and Backtest (weights path)::

    proba = booster.predict_proba(X)   # columns [down, flat, up]
    if p_up >= dir_threshold and p_up >= p_down → direction=up, conf=p_up
    elif p_down >= dir_threshold and p_down > p_up → direction=down, conf=p_down
    else → direction=flat, conf=p_flat
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def direction_from_proba(
    p_down: float | np.ndarray,
    p_flat: float | np.ndarray,
    p_up: float | np.ndarray,
    dir_thr: float,
) -> tuple[Any, Any]:
    """Map class probs → (direction, confidence).

    Accepts scalars or 1-d arrays; returns matching shapes (str / object array
    for direction, float / float array for confidence).
    """
    thr = float(dir_thr)
    pu = np.asarray(p_up, dtype="float64")
    pdn = np.asarray(p_down, dtype="float64")
    pf = np.asarray(p_flat, dtype="float64")
    scalar = pu.ndim == 0
    if scalar:
        pu, pdn, pf = pu.reshape(1), pdn.reshape(1), pf.reshape(1)

    n = pu.shape[0]
    direction = np.empty(n, dtype=object)
    confidence = np.empty(n, dtype="float64")
    for i in range(n):
        u, d, f = float(pu[i]), float(pdn[i]), float(pf[i])
        if u >= thr and u >= d:
            direction[i], confidence[i] = "up", u
        elif d >= thr and d > u:
            direction[i], confidence[i] = "down", d
        else:
            direction[i], confidence[i] = "flat", f

    if scalar:
        return direction[0], float(confidence[0])
    return direction, confidence


def predict_canonical(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score *df* (feature matrix, DatetimeIndex) → canonical prediction frame.

    Returns columns ``ts, direction, confidence, p_down, p_flat, p_up`` aligned
    1:1 with *df*'s index. Feature columns come from ``bundle['feature_cols']``;
    missing columns become NaN (same as Live ``reindex``).
    """
    if bundle.get("tf_oof_meta"):
        return _predict_tf_oof_meta(df, bundle)
    if bundle.get("primary_meta"):
        return _predict_primary_meta(df, bundle)
    if bundle.get("meta_stack"):
        return _predict_meta_stack(df, bundle)
    if bundle.get("output_meta"):
        return _predict_output_meta(df, bundle)
    if bundle.get("kind") == "sequence" or bundle.get("family") in (
            "attention_lstm", "tcn", "transformer", "cnn"):
        return _predict_sequence(df, bundle)
    cols = bundle["feature_cols"]
    X = df.reindex(columns=cols).to_numpy(dtype="float64")
    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    booster = bundle["booster"]
    fire = bundle.get("dual_binary_fire")
    if hasattr(booster, "predict_pair"):
        p_long, p_short = booster.predict_pair(X)
        return _frame_from_dual(
            p_long, p_short, pd.to_datetime(df.index, utc=True), fire=fire)

    if hasattr(booster, "predict_proba"):
        proba = booster.predict_proba(X)
    else:
        proba = booster.predict(X, verbose=0)
    from src.models.dual_binary import stack_head_predict
    proba = stack_head_predict(proba)
    if proba.ndim == 1:
        proba = np.stack([1.0 - proba, np.zeros_like(proba), proba], axis=1)
    if proba.shape[1] == 2:
        return _frame_from_dual(
            proba[:, 0], proba[:, 1],
            pd.to_datetime(df.index, utc=True), fire=fire)
    if proba.shape[1] < 3:
        raise ValueError(
            f"expected 3-class proba [down,flat,up]; got shape {proba.shape}")
    p_down = proba[:, 0]
    p_flat = proba[:, 1]
    p_up = proba[:, 2]
    direction, confidence = direction_from_proba(
        p_down, p_flat, p_up, float(bundle.get("dir_thr", 0.5)))
    return pd.DataFrame({
        "ts": pd.to_datetime(df.index, utc=True),
        "direction": direction,
        "confidence": confidence,
        "p_down": p_down,
        "p_flat": p_flat,
        "p_up": p_up,
    })


def _predict_sequence(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Window 2-D features and score a sequence model; pad warmup bars as flat."""
    spec = {
        "family": bundle.get("family"),
        "kind": "sequence",
        "artifact": bundle["booster"],
        "scaler": bundle.get("scaler"),
        "feature_cols": bundle["feature_cols"],
        "seq_len": int(bundle.get("seq_len") or 128),
    }
    scored = _score_base(df, spec)
    idx = pd.to_datetime(df.index, utc=True)
    full = pd.DataFrame({
        "ts": idx,
        "direction": "flat",
        "confidence": 0.0,
        "p_down": 0.0,
        "p_flat": 1.0,
        "p_up": 0.0,
    }, index=idx)
    if scored is None or scored.empty:
        return full.reset_index(drop=True)
    n_full, n_scored = len(full), len(scored)
    if n_scored > n_full:
        raise ValueError(
            f"sequence score has {n_scored} rows but features have {n_full}")
    # Positional suffix: make_windows/trim_warmup drop the first seq_len-1 rows.
    # Do not loc-by-timestamp — duplicate ts (live next-session spill) makes
    # pandas raise "Must have equal len keys and value when setting with an iterable".
    start = n_full - n_scored
    for col in ("direction", "confidence", "p_down", "p_flat", "p_up",
                "p_long", "p_short"):
        if col in scored.columns:
            if col not in full.columns:
                full[col] = 0.0
            full.iloc[start:, full.columns.get_loc(col)] = scored[col].to_numpy()
    return full.reset_index(drop=True)


def _scored_to_oof(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    if "ts" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            if out.columns[0] != "ts":
                out = out.rename(columns={out.columns[0]: "ts"})
        else:
            raise ValueError("base score has no ts")
    if "p_long" not in out.columns and "p_up" in out.columns:
        out["p_long"] = out["p_up"]
    if "p_short" not in out.columns and "p_down" in out.columns:
        out["p_short"] = out["p_down"]
    if "p_long" not in out.columns or "p_short" not in out.columns:
        raise ValueError("base score missing p_long/p_short")
    out["direction_true"] = "flat"
    return out


def _predict_tf_oof_meta(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score 15s/1m/5m last-fold bases, asof, dual-binary meta (ADR-100)."""
    from src.models.dual_binary import dual_binary_to_canonical
    from src.search.tf_oof_meta import (
        align_native_oof, load_htf_context, load_store_features_for_serve,
        select_meta_features,
    )

    idx = pd.to_datetime(df.index, utc=True)
    full = pd.DataFrame({
        "ts": idx,
        "direction": "flat",
        "confidence": 0.0,
        "p_down": 0.0,
        "p_flat": 1.0,
        "p_up": 0.0,
        "p_long": 0.0,
        "p_short": 0.0,
    }, index=idx)
    specs = list(bundle.get("bases") or [])
    if not specs:
        raise ValueError("tf_oof_meta bundle has no bases")
    meta = bundle.get("meta") or {}
    decision_tf = str(meta.get("decision_timestep") or bundle.get("native_tf") or "15s")
    fire = bundle.get("dual_binary_fire")
    if fire is None:
        fire = meta.get("fire")
    if fire is None:
        fire = bundle.get("dir_thr")
    fire = float(fire if fire is not None else 0.40)
    names = list(meta.get("feature_names") or [])
    start, end = idx.min(), idx.max()

    oofs: dict[str, pd.DataFrame] = {}
    topo: list[dict[str, Any]] = []
    for spec in specs:
        bid = str(spec.get("id") or spec.get("family") or "")
        tf = str(spec.get("native_tf") or "").strip() or decision_tf
        feat_df = df
        if tf != decision_tf:
            feat_df = load_store_features_for_serve(
                tf, start, end, feature_cols=list(spec.get("feature_cols") or []))
        scored = _score_base(feat_df, spec)
        oofs[bid] = _scored_to_oof(scored)
        topo.append({"id": bid, "native_tf": tf, "family": spec.get("family")})

    context = None
    if any(str(n).startswith(("h1_", "d1_")) for n in names):
        try:
            context = load_htf_context(start, end)
        except Exception:  # noqa: BLE001
            context = None
    aligned = align_native_oof(
        oofs, topo, decision_timestep=decision_tf, context=context)
    if aligned.empty:
        return full.reset_index(drop=True)
    X = select_meta_features(aligned, names)
    est = bundle["meta_estimator"]
    if hasattr(est, "predict_pair"):
        p_long, p_short = est.predict_pair(X)
    else:
        proba = np.asarray(est.predict_proba(X), dtype="float64")
        if proba.ndim == 2 and proba.shape[1] >= 2:
            p_long, p_short = proba[:, 0], proba[:, 1]
        else:
            raise ValueError("tf_oof_meta estimator has no predict_pair")
    canon = dual_binary_to_canonical(
        p_long, p_short, ts=aligned["ts"].to_numpy(), fire=fire)
    pl = canon["p_long"].to_numpy(dtype="float64")
    ps = canon["p_short"].to_numpy(dtype="float64")
    out = pd.DataFrame({
        "ts": pd.to_datetime(canon["ts"], utc=True),
        "direction": canon["direction"].to_numpy(),
        "confidence": canon["confidence"].to_numpy(dtype="float64"),
        "p_long": pl,
        "p_short": ps,
        "p_up": pl,
        "p_down": ps,
        "p_flat": np.clip(1.0 - np.maximum(pl, ps), 0.0, 1.0),
    }).set_index(pd.DatetimeIndex(pd.to_datetime(canon["ts"], utc=True)))
    common = full.index.intersection(out.index)
    for col in ("direction", "confidence", "p_down", "p_flat", "p_up", "p_long", "p_short"):
        full.loc[common, col] = out.loc[common, col].to_numpy()
    return full.reset_index(drop=True)


def _predict_primary_meta(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Primary 3-class booster + binary P(win) gate (ADR-027). Keep primary confidence."""
    from src.search.meta_compose import meta_design_from_scored, predict_meta_p_win

    idx = pd.to_datetime(df.index, utc=True)
    specs = bundle.get("bases") or []
    if not specs:
        raise ValueError("primary_meta bundle has no bases")
    primary = _score_base(df, specs[0])
    primary = primary.set_index(pd.DatetimeIndex(primary["ts"]))
    meta = bundle.get("meta") or {}
    names = list(meta.get("feature_names") or ["primary_confidence", "is_up"])
    ctx_cols = [n for n in names if n not in ("primary_confidence", "is_up")]
    context = df.reindex(columns=ctx_cols) if ctx_cols else None
    if context is not None:
        context = context.copy()
        context.index = pd.DatetimeIndex(df.index)
    X = meta_design_from_scored(
        direction=primary["direction"].to_numpy(),
        confidence=primary["confidence"].to_numpy(),
        ts=pd.DatetimeIndex(primary.index),
        context=context,
        feature_names=names,
        impute_values=meta.get("impute_values") or {},
    )
    p_win = predict_meta_p_win(bundle["meta_estimator"], X)
    side = primary["direction"].astype(str).to_numpy()
    conf = primary["confidence"].to_numpy(dtype=float)
    min_c = float(meta.get("min_primary_confidence", 0.50))
    thr = float(meta.get("approve_threshold", 0.55))
    eligible = np.isin(side, ("up", "down")) & (conf >= min_c) & np.isfinite(conf)
    approved = eligible & (p_win >= thr)
    direction = np.where(approved, side, "flat")
    confidence = np.where(approved, conf, 0.0)
    p_down = primary["p_down"].to_numpy() if "p_down" in primary.columns else np.zeros(len(primary))
    p_flat = primary["p_flat"].to_numpy() if "p_flat" in primary.columns else np.ones(len(primary))
    p_up = primary["p_up"].to_numpy() if "p_up" in primary.columns else np.zeros(len(primary))
    out = pd.DataFrame({
        "ts": primary.index,
        "direction": direction,
        "confidence": confidence,
        "p_down": np.where(approved & (side == "down"), p_down, 0.0),
        "p_flat": np.where(approved, np.where(side == "flat", p_flat, 0.0), 1.0),
        "p_up": np.where(approved & (side == "up"), p_up, 0.0),
    }).set_index(pd.DatetimeIndex(primary.index))
    full = pd.DataFrame({
        "ts": idx,
        "direction": "flat",
        "confidence": 0.0,
        "p_down": 0.0,
        "p_flat": 1.0,
        "p_up": 0.0,
    }, index=idx)
    common = full.index.intersection(out.index)
    full.loc[common, "direction"] = out.loc[common, "direction"].to_numpy()
    full.loc[common, "confidence"] = out.loc[common, "confidence"].to_numpy()
    full.loc[common, "p_down"] = out.loc[common, "p_down"].to_numpy()
    full.loc[common, "p_flat"] = out.loc[common, "p_flat"].to_numpy()
    full.loc[common, "p_up"] = out.loc[common, "p_up"].to_numpy()
    return full.reset_index(drop=True)


def _predict_meta_stack(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score last-fold bases, then the signed-evidence LR fire rule (ADR-035)."""
    from src.search.meta_stack import align_on_ts, signed_features

    idx = pd.to_datetime(df.index, utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for spec in bundle.get("bases") or []:
        frames[spec["family"]] = _score_base(df, spec)
    bases = [s["family"] for s in (bundle.get("bases") or [])]
    aligned = align_on_ts(frames, bases)
    feature = str((bundle.get("meta") or {}).get("feature") or "linear_weights")
    X, names = signed_features(aligned, bases, feature)
    expected = list((bundle.get("meta") or {}).get("feature_names") or names)
    if expected and list(names) != expected:
        colmap = {n: X[:, i] for i, n in enumerate(names)}
        X = np.column_stack([colmap.get(n, np.zeros(len(aligned))) for n in expected])
    est = bundle["meta_estimator"]
    proba = est.predict_proba(X)
    classes = list(getattr(est, "classes_", [0, 1]))
    p_up = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
    thr = float((bundle.get("meta") or {}).get("approve_threshold", 0.55))
    finite = np.isfinite(p_up)
    approved_up = finite & (p_up >= thr)
    approved_dn = finite & (p_up <= 1.0 - thr)
    direction = np.where(approved_up, "up",
                np.where(approved_dn, "down", "flat"))
    confidence = np.where(approved_up, p_up,
                 np.where(approved_dn, 1.0 - p_up, 0.0))
    out = pd.DataFrame({
        "ts": aligned["ts"].to_numpy(),
        "direction": direction,
        "confidence": confidence,
        "p_down": np.where(approved_dn, confidence, 0.0),
        "p_flat": np.where(~approved_up & ~approved_dn, 1.0, 0.0),
        "p_up": np.where(approved_up, confidence, 0.0),
    }).set_index(pd.DatetimeIndex(aligned["ts"]))
    full = pd.DataFrame({
        "ts": idx,
        "direction": "flat",
        "confidence": 0.0,
        "p_down": 0.0,
        "p_flat": 1.0,
        "p_up": 0.0,
    }, index=idx)
    full.loc[out.index, "direction"] = out["direction"].to_numpy()
    full.loc[out.index, "confidence"] = out["confidence"].to_numpy()
    full.loc[out.index, "p_down"] = out["p_down"].to_numpy()
    full.loc[out.index, "p_flat"] = out["p_flat"].to_numpy()
    full.loc[out.index, "p_up"] = out["p_up"].to_numpy()
    return full.reset_index(drop=True)


def _predict_output_meta(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Run each base, stack prediction vectors, apply the binary gate."""
    from src.search.output_meta import (
        align_on_ts, build_X_meta, proposed_side, _predict_meta_proba,
    )

    idx = pd.to_datetime(df.index, utc=True)
    frames: dict[str, pd.DataFrame] = {}
    for spec in bundle.get("bases") or []:
        frames[spec["family"]] = _score_base(df, spec)
    bases = [s["family"] for s in (bundle.get("bases") or [])]
    aligned = align_on_ts(frames, bases)
    side = proposed_side(aligned, bases)
    X, names = build_X_meta(aligned, bases)
    expected = list((bundle.get("meta") or {}).get("feature_names") or names)
    if expected and list(names) != expected:
        # pad / reorder if extra context cols missing at serve
        colmap = {n: X[:, i] for i, n in enumerate(names)}
        X = np.column_stack([colmap.get(n, np.zeros(len(aligned))) for n in expected])
    p_trade = _predict_meta_proba(bundle["meta_estimator"], X)
    thr = float((bundle.get("meta") or {}).get("approve_threshold", 0.55))
    approved = np.array([s in ("up", "down") for s in side]) & (p_trade >= thr)
    direction = np.where(approved, side, "flat")
    # Reindex to the original df length (warmup / non-overlap → flat)
    out = pd.DataFrame({
        "ts": aligned["ts"].to_numpy(),
        "direction": direction,
        "confidence": np.where(approved, p_trade, 0.0),
        "proposed_side": side,
        "meta_prob": p_trade,
    }).set_index(pd.DatetimeIndex(aligned["ts"]))
    full = pd.DataFrame({
        "ts": idx,
        "direction": "flat",
        "confidence": 0.0,
        "p_down": 0.0,
        "p_flat": 1.0,
        "p_up": 0.0,
    }, index=idx)
    full.loc[out.index, "direction"] = out["direction"].to_numpy()
    full.loc[out.index, "confidence"] = out["confidence"].to_numpy()
    return full.reset_index(drop=True)


def _score_base(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    cols = spec.get("feature_cols") or []
    X = df.reindex(columns=cols).to_numpy(dtype="float64")
    scaler = spec.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    art = spec["artifact"]
    ts = pd.DatetimeIndex(df.index)
    if spec.get("kind") == "sequence":
        from src.models.dl_memory import predict_chunked
        from src.models.dual_binary import window_pack_for_model
        from src.models.sequence_windows import trim_warmup
        seq_len = int(spec.get("seq_len") or 128)
        Xw = window_pack_for_model(
            X, seq_len, spec.get("feature_cols") or [], art)
        if hasattr(art, "predict_proba"):
            proba = art.predict_proba(Xw)
        else:
            proba = predict_chunked(art, Xw)
        ts_w = trim_warmup(ts.to_numpy(), seq_len)
        return _frame_from_proba(proba, pd.to_datetime(ts_w, utc=True))
    if hasattr(art, "predict_pair"):
        p_long, p_short = art.predict_pair(X)
        return _frame_from_dual(p_long, p_short, ts)
    proba = art.predict_proba(X)
    return _frame_from_proba(proba, ts)


def _frame_from_dual(
    p_long: np.ndarray,
    p_short: np.ndarray,
    ts,
    *,
    fire: float | None = None,
) -> pd.DataFrame:
    from src.models.dual_binary import dual_binary_to_canonical
    kwargs: dict[str, Any] = {}
    if fire is not None:
        kwargs["fire"] = fire
    df = dual_binary_to_canonical(p_long, p_short, ts=ts, **kwargs)
    pl = df["p_long"].to_numpy(dtype="float64")
    ps = df["p_short"].to_numpy(dtype="float64")
    df["p_up"] = pl
    df["p_down"] = ps
    df["p_flat"] = np.clip(1.0 - np.maximum(pl, ps), 0.0, 1.0)
    return df


def _frame_from_proba(proba: Any, ts) -> pd.DataFrame:
    from src.models.dual_binary import stack_head_predict
    arr = stack_head_predict(proba)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 2:
        return _frame_from_dual(arr[:, 0], arr[:, 1], ts)
    if arr.shape[1] < 3:
        raise ValueError(
            f"expected 3-class proba [down,flat,up]; got shape {arr.shape}")
    p_down, p_flat, p_up = arr[:, 0], arr[:, 1], arr[:, 2]
    direction, conf = direction_from_proba(p_down, p_flat, p_up, 0.5)
    return pd.DataFrame({
        "ts": ts,
        "direction": direction,
        "confidence": conf,
        "p_down": p_down, "p_flat": p_flat, "p_up": p_up,
    })
