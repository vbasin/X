"""Shared feature-matrix + per-objective label cache for the fold builder.

The parquet feature matrix is **shared across objectives**; labels are derived **per
objective** from ``(move_threshold, mae_tolerance, horizon)``. Building both — loading the
full parquet set and regenerating labels — is the slow part of every fold build, and the
search currently runs it **once per parallel worker** (and again per HPO trial and per
objective). This module caches the final *aligned* arrays to disk keyed by
``(parquet fingerprint + label params)`` so repeated builds load in seconds (memory-mapped)
instead of reloading parquet + regenerating labels.

Cache entry: ``data/research/search/_fold_cache/<key>/{X.npy, y.npy, ts.npy, meta.json}``
Concurrency-safe via an ``flock`` on ``<key>.lock``; written atomically via temp-dir rename,
so the 4+ parallel search workers cooperate (first builds, the rest load).
"""

from __future__ import annotations

from src.common.data_paths import data_root, parquet_pathlabels_dir, search_dir

import fcntl
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("vtech.search.fold_cache")

_DATA_ROOT = data_root()
_CACHE_ROOT = search_dir() / "_fold_cache"
_CACHE_VERSION = "v1"
_FEATURE_TIMESTEP_SECONDS = 5
PATH_SCHEME = "path_vol_normalized"
_PATH_EXTRAS = ("y_up", "y_down", "y_quantile", "r_value")


def _parse_duration_to_seconds(s: str) -> int:
    s = str(s).strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("s"):
        return int(s[:-1])
    try:
        return int(s)
    except ValueError:
        return 600


def _feature_parquets(pdir: Path) -> list[Path]:
    p = Path(pdir)
    files = sorted(p.glob("features_*.parquet"))
    if not files:
        files = sorted(p.glob("year=*/month=*/features_*.parquet"))
    return files


def probe_timestep_seconds(pdir: Path) -> int:
    """Infer the bar interval (seconds) from the first parquet's timestamps.

    Prefers ``bar_end_ts`` (hive parts) then the DatetimeIndex (day files).
    Falls back to the standard 5s feature timestep on any failure.
    """
    files = _feature_parquets(pdir)
    if not files:
        return _FEATURE_TIMESTEP_SECONDS
    try:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(files[0]).schema_arrow.names)
        if "bar_end_ts" in names:
            sample = pd.read_parquet(files[0], columns=["bar_end_ts"])
            idx = pd.to_datetime(sample["bar_end_ts"], utc=True)
        else:
            idx = pd.read_parquet(files[0], columns=[]).index
        if len(idx) >= 3:
            di = pd.DatetimeIndex(idx[:500])
            if di.tz is None:
                di = di.tz_localize("UTC")
            else:
                di = di.tz_convert("UTC")
            step = int(round(float(np.median(np.diff(di.asi8))) / 1e9))
            if step >= 1:
                return step
    except Exception as exc:  # noqa: BLE001 — probing must never break the build
        logger.warning("timestep probe failed (%s) — assuming %ds",
                       exc, _FEATURE_TIMESTEP_SECONDS)
    return _FEATURE_TIMESTEP_SECONDS


def _label_params(
    objective: dict[str, Any], timestep_s: int = _FEATURE_TIMESTEP_SECONDS,
) -> tuple[float, float, int, str]:
    """(move_threshold, mae_tolerance, horizon_window_bars, scheme) from the objective."""
    lbl = objective.get("labels") or {}
    hor = objective.get("horizon") or {}
    move_threshold = float(lbl.get("move_threshold", 50.0))
    mae_tolerance = float(lbl.get("mae_tolerance", 5.0))
    scheme = str(lbl.get("scheme", "detection_2"))
    horizon_secs = _parse_duration_to_seconds(hor.get("max_trade_horizon", "10m"))
    primary_window = max(1, horizon_secs // max(1, timestep_s))
    return move_threshold, mae_tolerance, primary_window, scheme


def _parquet_fingerprint(pdir: Path) -> str:
    """Fingerprint the parquet feature set (name+size+mtime of every features_*.parquet)."""
    files = _feature_parquets(pdir)
    parts = [f"{f.relative_to(pdir) if pdir in f.parents else f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}"
             for f in files]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _session_clock_mask(index, objective: dict[str, Any]) -> pd.Series | None:
    """True on bars in ``session.training_start``–``training_end`` (Chicago clock)."""
    sess = objective.get("session") or {}
    start = str(sess.get("training_start") or "").strip()
    end = str(sess.get("training_end") or "").strip()
    if not start or not end:
        return None
    try:
        sh, sm = (int(x) for x in start.split(":")[:2])
        eh, em = (int(x) for x in end.split(":")[:2])
    except ValueError:
        return None
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    ct = idx.tz_convert("America/Chicago")
    minutes = ct.hour.to_numpy() * 60 + ct.minute.to_numpy()
    lo = sh * 60 + sm
    hi = eh * 60 + em
    return pd.Series((minutes >= lo) & (minutes < hi), index=idx)


def _date_range(objective: dict[str, Any]) -> tuple[str | None, str | None]:
    """Optional inclusive [start, end] calendar clip from ``objective.eval.date_range``.

    ``None`` for either bound means "use the parquet dir's natural extent". Values
    must parse as ``YYYY-MM-DD``; anything else is treated as unset (with a warning).
    """
    dr = ((objective.get("eval") or {}).get("date_range") or {})
    if not isinstance(dr, dict):
        return None, None
    def _norm(v):
        if v is None:
            return None
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except Exception:
            logger.warning("fold_cache: eval.date_range value %r is not a date — ignored", v)
            return None
    return _norm(dr.get("start")), _norm(dr.get("end"))


def _cache_key(pdir: Path, objective: dict[str, Any]) -> str:
    step = probe_timestep_seconds(pdir)
    mt, mae, win, scheme = _label_params(objective, step)
    target_observation = str((objective.get("labels") or {}).get(
        "target_observation") or "candle_close")
    fp = _parquet_fingerprint(pdir)
    raw = f"{_CACHE_VERSION}|{fp}|mt={mt}|mae={mae}|win={win}"
    # Back-compat: the canonical detection_2 @ 5s key text is unchanged so existing
    # cache entries stay valid; other schemes/timesteps get their own key space.
    if scheme != "detection_2":
        raw += f"|scheme={scheme}"
    if target_observation != "candle_close":
        raw += f"|target_observation={target_observation}"
    if step != _FEATURE_TIMESTEP_SECONDS:
        raw += f"|step={step}"
    try:
        from src.features.stores import feature_store_id
        raw += f"|store={feature_store_id(objective)}"
    except Exception:
        pass
    enc = str((objective.get("labels") or {}).get("encoding") or "")
    if enc:
        raw += f"|enc={enc}"
    if enc == "dual_binary":
        raw += "|tg=lgst_v1"
        try:
            from src.training.aux_labels import aux_from_objective
            req = aux_from_objective(objective)
            if req:
                raw += "|aux=" + ",".join(req)
            raw += "|ttt=nan_timeout"
        except Exception:
            pass
    if scheme == "triple_barrier_atr":
        lbl = objective.get("labels") or {}
        raw += (
            f"|atr={lbl.get('atr_feature', 'cs_atr_10m')}"
            f"|t={lbl.get('move_threshold')}|s={lbl.get('mae_tolerance')}"
            f"|onset={int(bool(lbl.get('onset_only')))}|au=v1"
        )
    try:
        from src.features.htf_context import HTF_PACK_ID, wants_htf_on_store
        from src.features.stores import feature_store_id
        if wants_htf_on_store(objective, feature_store_id(objective)):
            raw += f"|htf={HTF_PACK_ID}"
    except Exception:
        pass
    try:
        from src.features.session_vp import session_origin_hhmm, wants_session_vp
        from src.features.stores import feature_store_id as _fs
        if wants_session_vp(objective, _fs(objective)):
            raw += f"|svp={session_origin_hhmm(objective)}"
    except Exception:
        pass
    sess = objective.get("session") or {}
    ts0 = str(sess.get("training_start") or "")
    ts1 = str(sess.get("training_end") or "")
    if ts0 or ts1:
        raw += f"|sess={ts0}-{ts1}"
        try:
            from src.features.stores import feature_store_id as _sid
            if _sid(objective) == "15s":
                raw += "|tape=globex"
        except Exception:
            pass
    dr_start, dr_end = _date_range(objective)
    if dr_start or dr_end:
        raw += f"|range={dr_start or ''}:{dr_end or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def describe_walk_forward_folds(
    ts: np.ndarray,
    y: np.ndarray,
    n_folds: int,
) -> list[dict[str, Any]]:
    """Calendar ranges implied by the equal-count expanding split in ``_build_folds``.

    The splitter is index arithmetic (``fold_size = n // (n_folds + 1)``), not
    calendar quarters. Dates are a *consequence* of bar counts. The remainder
    after ``(n_folds + 1) * fold_size`` bars is unused.
    """
    ts = pd.to_datetime(ts, utc=True)
    y = np.asarray(y)
    n = len(y)
    n_folds = int(n_folds)
    fold_size = n // (n_folds + 1)
    out: list[dict[str, Any]] = []
    if fold_size < 1:
        return out

    def _span(a: pd.DatetimeIndex | pd.Series) -> dict[str, Any]:
        if len(a) == 0:
            return {"start": None, "end": None, "n_bars": 0, "n_days": 0}
        days = pd.Series(pd.DatetimeIndex(a).tz_convert("UTC").date).nunique()
        return {
            "start": pd.Timestamp(a.min()).isoformat(),
            "end": pd.Timestamp(a.max()).isoformat(),
            "n_bars": int(len(a)),
            "n_days": int(days),
        }

    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_end = min(train_end + fold_size, n)
        y_tr, y_te = y[:train_end], y[train_end:test_end]
        out.append({
            "fold": i,
            "split": "expanding_equal_count",
            "train": {**_span(ts[:train_end]),
                      "n_down": int((y_tr == 0).sum()),
                      "n_flat": int((y_tr == 1).sum()),
                      "n_up": int((y_tr == 2).sum())},
            "oof": {**_span(ts[train_end:test_end]),
                    "n_down": int((y_te == 0).sum()),
                    "n_flat": int((y_te == 1).sum()),
                    "n_up": int((y_te == 2).sum())},
        })
    return out


def _load_entry(d: Path):
    meta = json.loads((d / "meta.json").read_text())
    X = np.load(d / "X.npy", mmap_mode="r")                 # shared OS page cache across workers
    y = np.load(d / "y.npy")
    ts = np.load(d / "ts.npy").view("datetime64[ns]")
    return X, y, ts, meta["feature_cols"]


def build_or_load_fold_arrays(objective: dict[str, Any], pdir, *, force: bool = False):
    """Return ``(X_full float32, y_full int32, ts_full datetime64[ns], feature_cols)`` or None.

    Cached to disk keyed by ``(parquet fingerprint + label params)``. Concurrency-safe: the
    first caller builds under an exclusive lock, others block then load the memmap.
    """
    pdir = Path(pdir)
    key = _cache_key(pdir, objective)
    entry = _CACHE_ROOT / key
    if not force and (entry / "meta.json").exists():
        try:
            return _load_entry(entry)
        except Exception as exc:  # corrupt/partial → rebuild under lock
            logger.warning("fold_cache load failed (%s) — rebuilding", exc)

    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_ROOT / f"{key}.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if not force and (entry / "meta.json").exists():   # another worker built it while we waited
            try:
                return _load_entry(entry)
            except Exception:
                pass
        arrays = _build_arrays(objective, pdir)
        if arrays is None:
            return None
        X, y, ts, cols, extras = arrays
        _write_entry(entry, X, y, ts, cols, objective, pdir, extras=extras)
    return _load_entry(entry)


def load_fold_extras(objective: dict[str, Any], pdir) -> dict[str, np.ndarray]:
    """Hazard extras (`y_up`, `y_down`, `y_quantile`, `r_value`) for a cache entry.

    Empty dict when the scheme is not :data:`PATH_SCHEME` or the sidecars are
    missing (legacy TBM caches).
    """
    pdir = Path(pdir)
    entry = _CACHE_ROOT / _cache_key(pdir, objective)
    return _load_extras(entry)


def _load_extras(entry: Path) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    names = list(_PATH_EXTRAS) + ["close"]
    meta_p = entry / "meta.json"
    if meta_p.exists():
        try:
            extra_names = json.loads(meta_p.read_text()).get("extras") or []
            names = list(dict.fromkeys(list(names) + list(extra_names)))
        except Exception:
            pass
    for name in names:
        p = entry / f"{name}.npy"
        if p.exists():
            out[name] = np.load(p)
    return out


def _write_entry(entry: Path, X, y, ts, cols, objective, pdir: Path,
                 extras: dict[str, np.ndarray] | None = None) -> None:
    tmp = entry.parent / f".{entry.name}.tmp.{os.getpid()}"
    if tmp.exists():
        for p in tmp.iterdir():
            p.unlink()
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "X.npy", np.ascontiguousarray(X))
    np.save(tmp / "y.npy", y)
    np.save(tmp / "ts.npy", ts.astype("datetime64[ns]").view("int64"))
    if extras:
        for name, arr in extras.items():
            np.save(tmp / f"{name}.npy", np.ascontiguousarray(arr))
    step = probe_timestep_seconds(pdir)
    mt, mae, win, scheme = _label_params(objective, step)
    (tmp / "meta.json").write_text(json.dumps({
        "feature_cols": list(cols), "n_rows": int(len(y)),
        "move_threshold": mt, "mae_tolerance": mae, "primary_window": win,
        "scheme": scheme, "timestep_s": step,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "extras": sorted(extras or {}),
    }))
    os.replace(tmp, entry)  # atomic rename (same parent dir)


def _build_arrays(objective: dict[str, Any], pdir: Path):
    """Load feature store + join persisted labels (ADR-076). Fallback: generate_labels."""
    try:
        from src.features.stores import feature_store_id, load_feature_frame
        from src.training.config import LabelConfig
        from src.training.label_store import (
            contract_matches, label_set_dir, load_labels_15s, load_labels_5s,
            tbm_contract_from_objective, _read_contract,
        )
        from src.training.labels import generate_labels
    except ImportError:
        return None

    store = feature_store_id(objective)
    dr_start, dr_end = _date_range(objective)
    day_files = [p for p in _feature_parquets(pdir) if "year=" not in str(p)]
    try:
        if day_files:
            from src.training.data_pipeline import ensure_utc_index
            df = ensure_utc_index(pd.concat(
                [pd.read_parquet(p) for p in day_files], sort=False,
            ))
            df = df[~df.index.duplicated(keep="last")].sort_index()
        else:
            df = load_feature_frame(store, dr_start, dr_end)
    except Exception as exc:
        logger.warning("fold_cache build: load_feature_frame store=%s failed: %s", store, exc)
        return None
    if df is None or df.empty:
        return None

    try:
        from src.features.session_vp import attach_session_vp, wants_session_vp
        if wants_session_vp(objective, store):
            df = attach_session_vp(df)
    except Exception as exc:
        logger.warning("fold_cache: session VP attach store=%s failed: %s", store, exc)

    try:
        from src.features.htf_context import attach_htf_to_frame, wants_htf_on_store
        if wants_htf_on_store(objective, store):
            df = attach_htf_to_frame(df)
    except Exception as exc:
        logger.warning("fold_cache: HTF attach store=%s failed: %s", store, exc)

    session_w: np.ndarray | None = None
    if str((objective.get("search") or {}).get("input") or "") == "native_tf":
        clk = _session_clock_mask(df.index, objective)
        if clk is not None:
            # 15s LSTM needs Globex continuity for seq_len lookback.
            # Clipping to 08:30–11:00 concatenates mornings so 11:00→next
            # 08:30 looks like one 15s step (~21.5 h) and the encoder goes
            # cold (p_max 0.51 → 0.38, 0 fires at 0.40). Trees have no
            # lookback: they still train only on session rows.
            if store in ("1m", "5m"):
                df = df.iloc[clk.to_numpy()]
                if df.empty:
                    logger.warning("fold_cache: session clock filter emptied store=%s", store)
                    return None
            else:
                session_w = clk.to_numpy(dtype="float32")
                logger.info(
                    "fold_cache: store=%s keeping %d bars (session mask %.1f%%)",
                    store, len(df), 100.0 * float(np.asarray(session_w).mean()),
                )

    scheme_early = str((objective.get("labels") or {}).get("scheme") or "")
    if scheme_early == "triple_barrier_atr":
        try:
            from src.features.auction_structure import attach_auction_structure
            df = attach_auction_structure(df)
        except Exception as exc:
            logger.warning("fold_cache: auction attach failed: %s", exc)

    step = 15 if store == "15s" else probe_timestep_seconds(pdir)
    mt, mae, win, scheme = _label_params(objective, step)
    from src.training.aux_labels import AUX_LABEL_COLS as _AUX_SKIP
    skip = {"label", "y_long", "y_short", "bar_end_ts", *_AUX_SKIP}
    feature_cols = [
        c for c in df.columns
        if not str(c).startswith("_") and not str(c).startswith("label") and c not in skip
    ]
    if scheme == PATH_SCHEME:
        return _build_pathlabel_arrays(df, feature_cols)

    encoding = str((objective.get("labels") or {}).get("encoding") or "").strip()
    wanted = tbm_contract_from_objective(objective)
    close_col = "_close" if "_close" in df.columns else (
        "cs_close" if "cs_close" in df.columns else None
    )
    from src.training.aux_labels import aux_from_objective, aux_matrix

    if store != "5s" and scheme != "triple_barrier_atr":
        label_dir = label_set_dir("15s", wanted)
        stored_c = _read_contract(label_dir)
        if not contract_matches(stored_c, wanted):
            logger.warning("fold_cache: matching 15s label set is missing at %s", label_dir)
            return None
        lab = load_labels_15s(dr_start, dr_end, contract=wanted)
        if lab is None or lab.empty or "y_long" not in lab.columns:
            logger.warning("fold_cache: native-TF store %s needs parquet_labels_15s", store)
            return None
        lab = lab.reindex(df.index)
        combined = pd.concat([df[feature_cols], lab[["y_long", "y_short"]]], axis=1)
        combined = combined.dropna(subset=["y_long", "y_short"])
        combined[feature_cols] = combined[feature_cols].fillna(0.0)
        X = combined[feature_cols].to_numpy(dtype="float32")
        y = combined[["y_long", "y_short"]].to_numpy(dtype="float32")
        ts = combined.index.values
        close = (
            df[close_col].reindex(combined.index).to_numpy(dtype="float32")
            if close_col else np.full(len(combined), np.nan, dtype="float32")
        )
        extras: dict[str, np.ndarray] = {"close": close}
        if aux_from_objective(objective) is not None:
            y_aux = aux_matrix(lab.reindex(combined.index))
            if y_aux is not None:
                extras["y_aux"] = y_aux
        if session_w is not None:
            sw = pd.Series(session_w, index=df.index).reindex(combined.index).fillna(0.0)
            extras["w_session"] = sw.to_numpy(dtype="float32")
        return X, y, ts, feature_cols, extras

    lab = None
    if scheme == "triple_barrier_atr":
        if close_col is None:
            logger.warning("fold_cache: triple_barrier_atr needs cs_close/_close")
            return None
        from src.training.atr_labels import atr_label_params, generate_labels_tbm_atr
        ap = atr_label_params(objective)
        atr_col = ap["atr_feature"]
        if atr_col not in df.columns:
            logger.warning("fold_cache: missing ATR column %s", atr_col)
            return None
        labels_df = generate_labels_tbm_atr(
            df[close_col], df[atr_col],
            target_mult=ap["target_mult"], stop_mult=ap["stop_mult"],
            window=win, min_target_pts=ap["min_target_pts"],
            min_stop_pts=ap["min_stop_pts"], onset_only=ap["onset_only"],
        )[["label"]]
    elif not day_files:
        label_dir = label_set_dir("5s", wanted)
        stored_c = _read_contract(label_dir)
        if contract_matches(stored_c, wanted):
            lab = load_labels_5s(dr_start, dr_end, contract=wanted)
        if lab is not None and "label" in lab.columns:
            lab = lab.reindex(df.index)
            labels_df = lab[["label"]]
        else:
            lab = None
            if close_col is None:
                logger.warning("fold_cache build: no _close/cs_close and no persisted 5s labels")
                return None
            label_cfg = LabelConfig(
                big_move_threshold=mt, mae_tolerance=mae,
                forward_windows=[win], scheme=scheme,
            )
            labels_df = generate_labels(df[close_col], label_cfg)[["label"]]
    else:
        if close_col is None:
            logger.warning("fold_cache build: no _close/cs_close and no persisted 5s labels")
            return None
        label_cfg = LabelConfig(
            big_move_threshold=mt, mae_tolerance=mae,
            forward_windows=[win], scheme=scheme,
        )
        labels_df = generate_labels(df[close_col], label_cfg)[["label"]]

    combined = pd.concat([df[feature_cols], labels_df], axis=1).dropna(subset=["label"])
    combined[feature_cols] = combined[feature_cols].fillna(0.0)
    X = combined[feature_cols].to_numpy(dtype="float32")
    y = combined["label"].to_numpy(dtype="int32")
    if encoding == "dual_binary":
        from src.models.dual_binary import as_dual_binary_y
        y = as_dual_binary_y(y)
    ts = combined.index.values
    close = (
        df[close_col].reindex(combined.index).to_numpy(dtype="float32")
        if close_col else np.full(len(y), np.nan, dtype="float32")
    )
    extras = {"close": close}
    if lab is not None and aux_from_objective(objective) is not None:
        y_aux = aux_matrix(lab.reindex(combined.index))
        if y_aux is not None:
            extras["y_aux"] = y_aux
    return X, y, ts, feature_cols, extras


def _build_pathlabel_arrays(df: pd.DataFrame, feature_cols: list[str]):
    """Inner-join 5s (or 1s) feature rows onto persisted path-label parquets.

    Keeps ``usable=True`` rows only (ADR-053 supervision gate). Sparse ``label``
    stays the 3-class KPI target; hazard extras are encoded alongside.
    """
    from src.models.heads import encode_pathlabel_frame

    labels = _load_pathlabels_aligned(df.index)
    if labels is None or labels.empty:
        logger.warning("fold_cache pathlabels: no usable rows joined onto the feature index")
        return None
    feat = df[feature_cols].copy()
    feat.index = pd.to_datetime(feat.index, utc=True)
    combined = feat.join(labels, how="inner")
    combined = combined.dropna(subset=["label"])
    if combined.empty:
        logger.warning("fold_cache pathlabels: inner join produced 0 rows")
        return None
    combined[feature_cols] = combined[feature_cols].fillna(0.0)
    packed = encode_pathlabel_frame(combined.reset_index(drop=True))
    extras = {
        "y_up": packed["y_up"].astype("int32"),
        "y_down": packed["y_down"].astype("int32"),
    }
    if "y_quantile" in packed:
        extras["y_quantile"] = packed["y_quantile"].astype("float32")
    if "r_value" in combined.columns:
        extras["r_value"] = combined["r_value"].to_numpy(dtype="float64")
    X = combined[feature_cols].to_numpy(dtype="float32")
    y = combined["label"].to_numpy(dtype="int32")
    ts = combined.index.values
    logger.info("fold_cache pathlabels: joined %d rows (%d feature cols)", len(y), len(feature_cols))
    return X, y, ts, feature_cols, extras


def _load_pathlabels_aligned(index: pd.Index) -> pd.DataFrame | None:
    """Load per-session path-label parquets covering *index* (UTC timestamps)."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    ct_dates = pd.DatetimeIndex(idx).tz_convert("America/Chicago")
    days = sorted({d.strftime("%Y-%m-%d") for d in ct_dates.date})
    lab_dir = parquet_pathlabels_dir()
    frames: list[pd.DataFrame] = []
    for day in days:
        p = lab_dir / f"labels_{day}.parquet"
        if not p.exists():
            continue
        lab = pd.read_parquet(p)
        ts = pd.to_datetime(lab["ts"], utc=True)
        lab = lab.set_index(ts)
        if "usable" in lab.columns:
            lab = lab.loc[lab["usable"].to_numpy()]
        if lab.empty:
            continue
        frames.append(lab)
    if not frames:
        return None
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
