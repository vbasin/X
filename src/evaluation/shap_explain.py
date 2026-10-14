"""TreeSHAP on a serving-bundle folder (XGBoost / tree boosters).

3-class boosters go through ``shap.TreeExplainer`` directly. Dual-binary
(``DualBinaryPair``) and OvR (``OvrDirectional``) wrappers are two independent
binary trees — TreeExplainer cannot consume the pair — so each head is explained
and mapped onto ``down`` / ``up`` (flat unused).

Writes ``shap_importance.csv``, ``shap_groups.csv``, ``shap_meta.json``,
``SHAP_SUMMARY.md``, bar/beeswarm PNGs into a ``SHAP/`` subdirectory of the
search / Custom Test run (or of the staging/registry bundle). Pass ``--out``
to override.

Usage::

    python -m src.evaluation.shap_explain --model /path/to/bundle
    make shap MODEL=/path/to/bundle
    make shap MODEL=/path/to/bundle SHAP_ARGS='--sample 20000 --tau 0.8'
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

logger = logging.getLogger("vtech.evaluation.shap_explain")

_CLASS = ("down", "flat", "up")
_SHAP_DIRNAME = "SHAP"


def _group(col: str) -> str:
    return f"{col.split('_', 1)[0]}_" if "_" in col else col


def _catalog_desc() -> dict[str, str]:
    from src.search.custom_test import parse_feature_catalog
    try:
        parsed = parse_feature_catalog()
    except Exception:  # noqa: BLE001
        return {}
    return {f["name"]: f.get("description") or "" for f in parsed.get("features") or []}


def _available_dates(pdir: Path) -> list[str]:
    from src.backtest.runner import _available_dates
    return _available_dates(pdir)


def _as_utc_index(idx) -> pd.DatetimeIndex:
    """Force a tz-aware UTC DatetimeIndex (mixed naive/aware parquet days)."""
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True))


def _load_matrix(
    bundle: dict[str, Any],
    *,
    start: str | None,
    end: str | None,
    max_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw feature frame, canonical predictions) aligned on ts."""
    from src.features.stores import (
        feature_store_id, load_feature_frame, parquet_dir_for_store,
    )
    from src.serving.predict import predict_canonical

    store = "5s"
    oid = bundle.get("objective_id")
    if oid:
        try:
            from src.objectives.loader import load_objective
            store = feature_store_id(load_objective(str(oid)))
        except Exception:
            store = "5s"
    pdir = parquet_dir_for_store(store)
    days = _available_dates(pdir)
    if not days:
        raise FileNotFoundError(f"no feature parquets in {pdir}")
    lo = start or days[0]
    hi = end or days[-1]
    use = [d for d in days if lo <= d <= hi]
    if max_days and len(use) > max_days:
        use = use[-max_days:]
    if not use:
        raise FileNotFoundError(f"no feature days in {lo}→{hi}")
    cols = list(bundle["feature_cols"])
    df = load_feature_frame(store, use[0], use[-1])
    df.index = _as_utc_index(df.index)
    X = df.reindex(columns=cols).sort_index()
    if X.empty or X.isna().all().all():
        raise FileNotFoundError(f"no feature days in {lo}→{hi}")
    X.index = _as_utc_index(X.index)
    X = X.replace([np.inf, -np.inf], np.nan)
    preds = predict_canonical(X.fillna(0.0), bundle)
    # predict_canonical returns a RangeIndex + ts column; SHAP samples on ts.
    if "ts" in preds.columns:
        preds = preds.set_index("ts")
    if len(preds) == len(X):
        preds.index = X.index
    else:
        preds.index = _as_utc_index(preds.index)
        preds = preds.reindex(X.index)
    return X, preds


def _sample_rows(
    X: pd.DataFrame,
    preds: pd.DataFrame,
    *,
    sample: int,
    tau: float,
    seed: int,
) -> pd.Index:
    rng = np.random.default_rng(seed)
    sig = preds.index[
        preds["direction"].isin(["up", "down"]) & (preds["confidence"] >= tau)
    ]
    rest = preds.index.difference(sig)
    n_sig = min(len(sig), sample)
    n_bg = min(len(rest), max(0, sample - n_sig))
    pick: list[pd.Index] = []
    if n_sig:
        pick.append(sig[rng.choice(len(sig), size=n_sig, replace=False)])
    if n_bg:
        pick.append(rest[rng.choice(len(rest), size=n_bg, replace=False)])
    if not pick:
        raise ValueError("no rows to explain")
    out = pick[0]
    for extra in pick[1:]:
        out = out.union(extra)
    return out


def _is_dual_binary_pair(model: Any) -> bool:
    """True for ``DualBinaryPair`` (two independent binary tree heads)."""
    return (
        hasattr(model, "predict_pair")
        and getattr(model, "long_clf", None) is not None
        and getattr(model, "short_clf", None) is not None
    )


def _is_ovr_directional(model: Any) -> bool:
    """True for ``OvrDirectional`` (up-vs-rest + down-vs-rest)."""
    return (
        getattr(model, "up", None) is not None
        and getattr(model, "down", None) is not None
        and not _is_dual_binary_pair(model)
    )


def _shap_values_3class(explainer: Any, Xs: np.ndarray) -> np.ndarray:
    """Return array shape (n, n_features, 3) regardless of shap version."""
    raw = explainer.shap_values(Xs)
    if isinstance(raw, list):
        arrs = [np.asarray(a) for a in raw]
        if len(arrs) == 3:
            return np.stack(arrs, axis=-1)
        return np.stack(arrs, axis=-1)
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            return arr
        if arr.shape[1] == 3:
            return np.transpose(arr, (0, 2, 1))
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    raise ValueError(f"unrecognized shap_values shape {arr.shape}")


def _positive_class_shap(explainer: Any, X: np.ndarray) -> np.ndarray:
    """Binary TreeSHAP → ``(n, n_features)`` for the positive class (log-odds)."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        arrs = [np.asarray(a) for a in raw]
        arr = arrs[-1] if len(arrs) >= 2 else arrs[0]
        if arr.ndim != 2:
            raise ValueError(f"unrecognized binary shap list shape {arr.shape}")
        return arr
    arr = np.asarray(raw)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[1] == 2:
            return arr[:, 1, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"unrecognized binary shap_values shape {arr.shape}")


def _scatter_shap(
    sv: np.ndarray, n_features: int, idx: list[int] | None,
) -> np.ndarray:
    """Pad a head-private SHAP matrix onto the full feature axis."""
    n, k = sv.shape
    if idx is None:
        if k != n_features:
            raise ValueError(
                f"shap width {k} != n_features {n_features} (no column index)")
        return sv
    out = np.zeros((n, n_features), dtype="float64")
    take = np.asarray(idx, dtype=int)
    if k != len(take):
        raise ValueError(f"shap width {k} != index length {len(take)}")
    out[:, take] = sv
    return out


def _expected_scalar(explainer: Any) -> float:
    ev = getattr(explainer, "expected_value", None)
    if ev is None:
        return 0.0
    arr = np.atleast_1d(np.asarray(ev, dtype="float64"))
    return float(arr[-1])


def _tree_explainer(booster: Any) -> Any:
    import shap
    return shap.TreeExplainer(booster)


def shap_values_for_booster(
    booster: Any, Xs: np.ndarray,
) -> tuple[np.ndarray, list[float], str]:
    """TreeSHAP as ``(n, F, 3)`` down/flat/up plus expected_value and head kind.

    Dual-binary / OvR wrappers are two independent binary boosters; TreeExplainer
    cannot consume the pair. Each head is explained on its own columns, then
    scattered onto the full feature axis. ``flat`` is unused (zeros) — there is
    no flat-class model. ``down`` = short / down-vs-rest, ``up`` = long / up-vs-rest.
    """
    n, n_feat = int(Xs.shape[0]), int(Xs.shape[1])
    if _is_dual_binary_pair(booster):
        expl_l = _tree_explainer(booster.long_clf)
        expl_s = _tree_explainer(booster.short_clf)
        li = booster.long_idx
        si = booster.short_idx
        Xl = Xs[:, li] if li is not None else Xs
        Xsh = Xs[:, si] if si is not None else Xs
        sv_l = _scatter_shap(_positive_class_shap(expl_l, Xl), n_feat, li)
        sv_s = _scatter_shap(_positive_class_shap(expl_s, Xsh), n_feat, si)
        sv = np.stack([sv_s, np.zeros((n, n_feat), dtype="float64"), sv_l], axis=-1)
        ev = [_expected_scalar(expl_s), 0.0, _expected_scalar(expl_l)]
        return sv, ev, "dual_binary"
    if _is_ovr_directional(booster):
        expl_u = _tree_explainer(booster.up)
        expl_d = _tree_explainer(booster.down)
        sv_u = _positive_class_shap(expl_u, Xs)
        sv_d = _positive_class_shap(expl_d, Xs)
        sv = np.stack([sv_d, np.zeros((n, n_feat), dtype="float64"), sv_u], axis=-1)
        ev = [_expected_scalar(expl_d), 0.0, _expected_scalar(expl_u)]
        return sv, ev, "ovr"
    explainer = _tree_explainer(booster)
    sv = _shap_values_3class(explainer, Xs)
    ev_raw = getattr(explainer, "expected_value", None)
    ev = [float(x) for x in np.atleast_1d(ev_raw)] if ev_raw is not None else []
    return sv, ev, "multiclass"


def _score_dict(clf: Any, names: list[str], kind: str) -> dict[str, float]:
    get = getattr(clf, "get_booster", None)
    b = get() if callable(get) else clf
    try:
        raw = b.get_score(importance_type=kind)
    except Exception:  # noqa: BLE001
        return {n: 0.0 for n in names}
    return {
        n: float(raw.get(n, raw.get(f"f{i}", 0.0)))
        for i, n in enumerate(names)
    }


def _head_feature_names(cols: list[str], idx: list[int] | None) -> list[str]:
    if idx is None:
        return list(cols)
    return [cols[i] for i in idx]


def _xgb_importances(booster: Any, cols: list[str]) -> pd.DataFrame:
    heads: list[tuple[str, Any, list[int] | None]]
    if _is_dual_binary_pair(booster):
        heads = [
            ("long", booster.long_clf, booster.long_idx),
            ("short", booster.short_clf, booster.short_idx),
        ]
    elif _is_ovr_directional(booster):
        heads = [("up", booster.up, None), ("down", booster.down, None)]
    else:
        heads = [("main", booster, None)]

    gain = {c: 0.0 for c in cols}
    cover = {c: 0.0 for c in cols}
    weight = {c: 0.0 for c in cols}
    per_head: dict[str, dict[str, float]] = {}
    for label, clf, idx in heads:
        names = _head_feature_names(cols, idx)
        g = _score_dict(clf, names, "gain")
        cvr = _score_dict(clf, names, "cover")
        w = _score_dict(clf, names, "weight")
        per_head[label] = g
        for n in names:
            gain[n] = gain.get(n, 0.0) + g.get(n, 0.0)
            cover[n] = cover.get(n, 0.0) + cvr.get(n, 0.0)
            weight[n] = weight.get(n, 0.0) + w.get(n, 0.0)

    rows = {
        "feature": cols,
        "xgb_gain": [gain[c] for c in cols],
        "xgb_cover": [cover[c] for c in cols],
        "xgb_weight": [weight[c] for c in cols],
    }
    if "long" in per_head and "short" in per_head:
        rows["xgb_gain_long"] = [per_head["long"].get(c, 0.0) for c in cols]
        rows["xgb_gain_short"] = [per_head["short"].get(c, 0.0) for c in cols]
    return pd.DataFrame(rows)


def _write_plots(
    out: Path,
    importance: pd.DataFrame,
    sv: np.ndarray,
    Xs: np.ndarray,
    cols: list[str],
    signal_mask: np.ndarray,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.sort_values("mean_abs_shap", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#37b6ff")
    ax.set_xlabel("mean |SHAP| (all bars)")
    ax.set_title("Global SHAP")
    fig.tight_layout()
    fig.savefig(out / "shap_bar_global.png", dpi=120)
    plt.close(fig)

    top_s = importance.sort_values("mean_abs_shap_signal", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_s["feature"][::-1], top_s["mean_abs_shap_signal"][::-1], color="#7ee0a3")
    ax.set_xlabel("mean |SHAP| (τ fires)")
    ax.set_title("Signal SHAP")
    fig.tight_layout()
    fig.savefig(out / "shap_bar_signals.png", dpi=120)
    plt.close(fig)

    try:
        import shap
        for i, name in enumerate(_CLASS):
            plane = sv[:, :, i]
            if not np.isfinite(plane).any() or np.allclose(plane, 0.0):
                continue
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(plane, Xs, feature_names=cols, show=False, max_display=18)
            fig.tight_layout()
            fig.savefig(out / f"shap_beeswarm_{name}.png", dpi=110, bbox_inches="tight")
            plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        logger.warning("beeswarm_skip: %s", exc)


def resolve_user_model_path(raw: str | Path) -> Path:
    """Resolve operator paths like ``/data/research/search/...`` or a relative run dir."""
    from src.common.data_paths import data_root

    text = str(raw).strip()
    p = Path(text)
    tried: list[Path] = []

    def _add(c: Path) -> None:
        tried.append(c)

    _add(p)
    if text.startswith("/data/") or text.startswith("data/"):
        rest = text[len("/data/"):] if text.startswith("/data/") else text[len("data/"):]
        _add(data_root() / rest)
        _add(Path("/opt/vtech/data") / rest)
    if text.startswith("research/"):
        _add(data_root() / text)
        _add(Path("/opt/vtech/data") / text)
    if not p.is_absolute():
        _add(Path.cwd() / p)
        _add(data_root() / p)
        _add(Path("/opt/vtech") / p)
        _add(Path("/opt/vtech/data") / p)
    if p.suffix == ".parquet" and p.name.startswith("oof_"):
        _add(p.parent)

    for c in tried:
        try:
            if c.exists():
                return c.resolve()
        except OSError:
            continue
    raise FileNotFoundError(
        f"model path not found: {raw}\n  tried: " + ", ".join(str(t) for t in tried[:8])
    )


def is_search_run(path: Path) -> bool:
    d = path if path.is_dir() else path.parent
    return d.is_dir() and (d / "search_result.json").exists()


def experiment_run_dir(path: Path) -> Path | None:
    """Nearest ancestor (or *path*) that is a search / Custom Test run folder."""
    d = path if path.is_dir() else path.parent
    try:
        d = d.resolve()
    except OSError:
        pass
    for p in [d, *d.parents]:
        if ((p / "manifest.json").is_file()
                or (p / "search_result.json").is_file()
                or (p / "run.json").is_file()):
            return p
    return None


def default_shap_out_dir(model_path: Path) -> Path:
    """``<model>/report`` inside a canonical run, else ``<run>/SHAP``, else ``<bundle>/SHAP``.

    Keeps CSVs/PNGs/npz out of the run root and out of the serving-bundle folder.
    """
    # A canonical model folder keeps its own report/ (STORAGE_CONTRACT 21), so a
    # multi-model run does not collapse every model's SHAP into one run-level
    # directory the way the legacy layout did.
    holder = model_path if model_path.is_dir() else model_path.parent
    if (holder / "model.json").is_file() and holder.name.startswith("model_"):
        return holder / "report"
    run = experiment_run_dir(model_path)
    if run is not None:
        return run / _SHAP_DIRNAME
    root = model_path if model_path.is_dir() else model_path.parent
    return root / _SHAP_DIRNAME


def _pick_tree_candidate(summary: dict[str, Any], candidate_id: str | None) -> dict[str, Any]:
    from src.search.finalize import _pick_candidate
    trees = {"xgboost", "lightgbm", "random_forest"}
    if candidate_id:
        return _pick_candidate(summary, candidate_id)
    for c in summary.get("candidates") or []:
        fam = str(c.get("family") or "")
        cid = str(c.get("candidate_id") or "")
        if c.get("status") != "ok":
            continue
        if fam in trees and not cid.startswith("meta"):
            return c
    raise ValueError(
        "no tree candidate (xgboost/lightgbm/random_forest) in this search run — "
        "pass --candidate or finalize a serving bundle first")


def ensure_search_shap_bundle(
    run_dir: Path, *, candidate_id: str | None = None,
) -> Path:
    """Use ``<run>/<candidate_id>/`` if present; else refit into that folder."""
    from src.search.finalize import finalize_champion
    from src.serving.bundle import is_serving_bundle

    summary = json.loads((run_dir / "search_result.json").read_text())
    cand = _pick_tree_candidate(summary, candidate_id)
    cid = str(cand["candidate_id"])
    dest = run_dir / "models" / cid
    if is_serving_bundle(dest):
        return dest
    dest = run_dir / cid
    if is_serving_bundle(dest):
        return dest
    shap_dest = run_dir / "shap_bundle"
    if is_serving_bundle(shap_dest):
        man = shap_dest / "finalize_manifest.json"
        if man.exists():
            try:
                got = json.loads(man.read_text()).get("candidate_id")
                if got == cand["candidate_id"]:
                    return shap_dest
            except json.JSONDecodeError:
                pass
        else:
            return shap_dest
    oid = str(summary.get("objective_id") or run_dir.parent.name)
    dest = run_dir / "models" / cid
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"⏳ Search run has no booster — refitting {cand['candidate_id']} "
          f"({cand['family']}) into {dest} for TreeSHAP…")
    finalize_champion(
        oid, run_dir=run_dir, candidate_id=cand["candidate_id"], dest=dest,
    )
    return dest


def explain_bundle(
    model_dir: str | Path,
    *,
    sample: int = 8000,
    tau: float = 0.8,
    seed: int = 42,
    start: str | None = None,
    end: str | None = None,
    max_days: int = 0,
    out_dir: str | Path | None = None,
    candidate: str | None = None,
) -> dict[str, Any]:
    from src.serving.bundle import is_serving_bundle, load_serving_bundle, validate_bundle_path

    resolved = resolve_user_model_path(model_dir)
    if resolved.is_file() and resolved.name.startswith("oof_"):
        resolved = resolved.parent

    if is_search_run(resolved):
        run_dir = resolved if resolved.is_dir() else resolved.parent
        bundle_path = ensure_search_shap_bundle(run_dir, candidate_id=candidate or None)
        bundle = load_serving_bundle(bundle_path, validate=False)
    else:
        bundle_path = validate_bundle_path(resolved)
        bundle = load_serving_bundle(bundle_path)

    dest = Path(out_dir) if out_dir else default_shap_out_dir(resolved)
    dest.mkdir(parents=True, exist_ok=True)
    if bundle.get("output_meta"):
        raise ValueError(
            "output_meta / meta_stack bundles are not TreeSHAP-able; "
            "run SHAP on a base (xgboost) serving folder")
    family = str(bundle.get("family") or "")
    booster = bundle["booster"]
    tree_pair = _is_dual_binary_pair(booster) or _is_ovr_directional(booster)
    if (
        family not in ("xgboost", "lightgbm", "random_forest")
        and not hasattr(booster, "get_booster")
        and not tree_pair
    ):
        raise ValueError(f"TreeSHAP supports tree boosters; family={family!r}")

    t0 = time.time()
    X_raw, preds = _load_matrix(bundle, start=start, end=end, max_days=max_days)
    idx = _sample_rows(X_raw, preds, sample=sample, tau=tau, seed=seed)
    X_s = X_raw.loc[idx]
    nan_rate = X_s.isna().mean()
    X_filled = X_s.fillna(0.0)
    scaler = bundle.get("scaler")
    Xs = X_filled.to_numpy(dtype="float64")
    if scaler is not None:
        Xs = scaler.transform(Xs)

    sv, expected_value, head_kind = shap_values_for_booster(booster, Xs)
    pred_s = preds.loc[idx]
    sig_mask = (
        pred_s["direction"].isin(["up", "down"]).to_numpy()
        & (pred_s["confidence"].to_numpy() >= tau)
    )

    mean_abs = np.abs(sv).mean(axis=0)              # (F, 3)
    # Dual-binary / OvR have no flat-class model — do not dilute by averaging zeros.
    if head_kind in ("dual_binary", "ovr"):
        mean_abs_all = 0.5 * (mean_abs[:, 0] + mean_abs[:, 2])
        mean_abs_sig = (
            0.5 * (np.abs(sv[sig_mask][:, :, 0]).mean(axis=0)
                   + np.abs(sv[sig_mask][:, :, 2]).mean(axis=0))
            if sig_mask.any() else np.zeros(len(X_s.columns))
        )
    else:
        mean_abs_all = mean_abs.mean(axis=1)
        mean_abs_sig = (np.abs(sv[sig_mask]).mean(axis=0).mean(axis=1)
                        if sig_mask.any() else np.zeros(len(X_s.columns)))
    tot_all = float(mean_abs_all.sum()) or 1.0
    tot_sig = float(mean_abs_sig.sum()) or 1.0
    signed = sv.mean(axis=0)
    desc = _catalog_desc()
    cols = list(X_s.columns)
    xgb_imp = _xgb_importances(booster, cols)

    rows = []
    for i, c in enumerate(cols):
        rows.append({
            "feature": c,
            "group": _group(c),
            "description": desc.get(c, ""),
            "mean_abs_shap": float(mean_abs_all[i]),
            "share_global": float(mean_abs_all[i] / tot_all),
            "mean_abs_shap_signal": float(mean_abs_sig[i]),
            "share_signal": float(mean_abs_sig[i] / tot_sig),
            "mean_abs_down": float(mean_abs[i, 0]),
            "mean_abs_flat": float(mean_abs[i, 1]),
            "mean_abs_up": float(mean_abs[i, 2]),
            "mean_signed_down": float(signed[i, 0]),
            "mean_signed_flat": float(signed[i, 1]),
            "mean_signed_up": float(signed[i, 2]),
            "nan_rate": float(nan_rate.get(c, 0.0)),
        })
    importance = pd.DataFrame(rows).merge(xgb_imp, on="feature", how="left")
    importance = importance.sort_values("share_signal", ascending=False)
    importance.to_csv(dest / "shap_importance.csv", index=False)

    grp = (importance.groupby("group", as_index=False)
           .agg(n_features=("feature", "size"),
                share_global=("share_global", "sum"),
                share_signal=("share_signal", "sum"),
                mean_abs_shap=("mean_abs_shap", "sum"),
                mean_abs_shap_signal=("mean_abs_shap_signal", "sum"),
                xgb_gain=("xgb_gain", "sum")))
    grp = grp.sort_values("share_signal", ascending=False)
    grp.to_csv(dest / "shap_groups.csv", index=False)

    n_sig_loaded = int((
        preds["direction"].isin(["up", "down"]) & (preds["confidence"] >= tau)
    ).sum())
    counts = pred_s["direction"].value_counts().to_dict()
    shap_unit = (
        "binary log-odds per head (down=short, up=long; flat unused)"
        if head_kind == "dual_binary"
        else "binary log-odds per OvR head (down / up; flat unused)"
        if head_kind == "ovr"
        else "pre-softmax class margin"
    )
    meta = {
        "bundle": str(bundle_path),
        "version": bundle.get("version") or bundle_path.name,
        "objective_id": bundle.get("objective_id"),
        "candidate_id": bundle.get("candidate_id"),
        "family": bundle.get("family"),
        "head": head_kind,
        "n_features": len(cols),
        "feature_cols": cols,
        "date_start": str(X_raw.index.min().date()) if len(X_raw) else None,
        "date_end": str(X_raw.index.max().date()) if len(X_raw) else None,
        "n_rows_loaded": int(len(X_raw)),
        "n_rows_explained": int(len(idx)),
        "n_signal_loaded": n_sig_loaded,
        "n_signal_explained": int(sig_mask.sum()),
        "tau": tau,
        "dir_threshold": float(bundle.get("dir_thr", 0.5)),
        "sample_size": sample,
        "seed": seed,
        "fillna": 0.0,
        "scaled": scaler is not None,
        "shap_unit": shap_unit,
        "class_names": list(_CLASS),
        "class_map": (
            {"down": "short", "flat": None, "up": "long"}
            if head_kind == "dual_binary" else None
        ),
        "expected_value": expected_value,
        "pred_counts_sample": {str(k): int(v) for k, v in counts.items()},
        "elapsed_s": round(time.time() - t0, 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(dest),
    }
    (dest / "shap_meta.json").write_text(json.dumps(meta, indent=2))

    top5 = importance.head(5)
    lines = [
        f"# SHAP summary — `{meta['version']}`",
        "",
        f"Generated `{meta['generated_at']}` from `{bundle_path}`.",
        "",
        "## Headline",
        "",
        f"Explained **{meta['n_rows_explained']:,}** bars "
        f"({meta['n_signal_explained']:,} fires @ τ={tau}) "
        f"across **{meta['n_features']}** features.",
        "",
    ]
    if head_kind == "dual_binary":
        lines += [
            "Head is **dual-binary**: two independent XGBs (`p_long` / `p_short`). "
            "`mean_abs_up` is the long head; `mean_abs_down` is the short head; "
            "flat is unused (no third class).",
            "",
        ]
    elif head_kind == "ovr":
        lines += [
            "Head is **OvR**: up-vs-rest and down-vs-rest. "
            "Flat is unused (no third class).",
            "",
        ]
    lines += [
        "Top signal-share features:",
        "",
    ]
    for _, r in top5.iterrows():
        lines.append(f"- `{r['feature']}` — {r['share_signal']*100:.1f}% of signal |SHAP|")
    lines += [
        "",
        "## Groups",
        "",
        "| group | n | share_signal | share_global |",
        "|-------|---|--------------|--------------|",
    ]
    for _, r in grp.iterrows():
        lines.append(
            f"| `{r['group']}` | {int(r['n_features'])} | "
            f"{r['share_signal']*100:.1f}% | {r['share_global']*100:.1f}% |")
    lines += [
        "",
        "High SHAP means *this booster uses the feature*. It is not a claim that "
        "the feature causes NQ to move.",
        "",
    ]
    (dest / "SHAP_SUMMARY.md").write_text("\n".join(lines))

    rng = np.random.default_rng(seed)
    keep = min(4000, len(idx))
    take = rng.choice(len(idx), size=keep, replace=False)
    np.savez_compressed(
        dest / "shap_values_sample.npz",
        shap=sv[take], X=Xs[take], columns=np.array(cols),
        ts=idx[take].astype(str).to_numpy(),
    )
    _write_plots(dest, importance, sv, Xs, cols, sig_mask)
    logger.info("shap_wrote dest=%s elapsed=%.1fs", dest, time.time() - t0)
    return meta


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="TreeSHAP a serving bundle or a search-run folder")
    p.add_argument("--model", "--bundle", dest="model", required=True,
                   help="Serving bundle OR research/search/<obj>/<run_ts> "
                        "(aliases: /data/research/search/… → $VTECH_DATA_DIR/research/search/…)")
    p.add_argument("--candidate", default="",
                   help="Search-run candidate id (default: first ok xgboost/lightgbm/rf)")
    p.add_argument("--sample", type=int, default=8000)
    p.add_argument("--tau", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", default="")
    p.add_argument("--end", default="")
    p.add_argument("--max-days", type=int, default=0,
                   help="If set, only the last N feature days are loaded")
    p.add_argument("--out", default="",
                   help="Write artifacts here (default: <run>/SHAP or <bundle>/SHAP)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    meta = explain_bundle(
        args.model,
        sample=args.sample,
        tau=args.tau,
        seed=args.seed,
        start=args.start or None,
        end=args.end or None,
        max_days=args.max_days,
        out_dir=args.out or None,
        candidate=args.candidate or None,
    )
    summary = Path(args.out) if args.out else Path(meta.get("artifact_dir") or meta["bundle"])
    print(f"✅ SHAP wrote {meta['n_rows_explained']} rows → {summary}")
    print(f"   summary: {summary / 'SHAP_SUMMARY.md'}")


if __name__ == "__main__":
    _cli()
