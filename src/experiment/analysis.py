"""Join declared model inputs, and rebuild an OOF analysis frame on demand.

Training reads labels, features, and optional context through the model's
``inputs`` spec. Features, context, and upstream predictions are attached
as-of: a joined value must have been knowable at or before that row's
decision ``ts``. Labels stay on the decision row and may use the forward
price path. ``training_matrix.parquet`` is consulted only when a pass
materialized that population once. ``build_oof_analysis`` starts from the
compact OOF and joins the same sources back, for inspection or SHAP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.common.causal import (
    CausalityError,
    as_of_index,
    assert_known_at_or_before,
)
from src.experiment.artifacts import (
    ANALYSIS_VIEW,
    AVAILABILITY_BAR_CLOSE,
    AVAILABILITY_SESSION_CLOSE,
    AVAILABILITY_TIMESTAMP,
    MODEL_JSON,
    OOF_PREDICTIONS,
    validate_model_inputs,
)

_NAT = np.iinfo(np.int64).min


def resolve_input_path(location: str, run_directory: Path) -> Path:
    """Resolve a run-relative or data-root-relative input location."""
    raw = Path(location)
    if raw.is_absolute():
        return raw
    from src.common.data_paths import data_root
    for candidate in (run_directory / location, data_root() / location):
        if candidate.exists():
            return candidate
    return run_directory / location


def _schema_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq
    names: list[str] = []
    for file in _parquet_files(path):
        for name in pq.read_schema(file).names:
            if name not in names:
                names.append(name)
    return names


def _already_on_frame(frame: pd.DataFrame, path: Path, wanted: Sequence[str], reserved: set[str]) -> bool:
    """True when this file would not add a column. Avoids re-reading a population cache."""
    if wanted:
        return all(column in frame.columns for column in wanted)
    try:
        names = _schema_names(path)
    except FileNotFoundError:
        return False
    return bool(names) and all(name in frame.columns or name in reserved for name in names)


def _parquet_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.glob("*.parquet") if p.is_file() and not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"no parquet under {path}")
    return files


def read_source_frame(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read one parquet file or a directory of parts."""
    files = _parquet_files(path)
    use: list[str] | None = None
    if columns:
        import pyarrow.parquet as pq
        have = set(pq.read_schema(files[0]).names)
        use = [c for c in columns if c in have]
        if not use:
            use = None
    frames = [pd.read_parquet(f, columns=use) for f in files]
    frame = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    if "session" in frame.columns:
        frame["session"] = pd.to_datetime(frame["session"])
    return frame


def _source_list(inputs: Mapping[str, Any], kind: str) -> list[Mapping[str, Any]]:
    block = inputs.get(kind) or []
    if isinstance(block, Mapping):
        return [block]
    return [item for item in block if isinstance(item, Mapping)]


def _utc_ns(values: Any) -> np.ndarray:
    """UTC timestamps as int64 nanoseconds. NaT is ``np.iinfo(np.int64).min``."""
    converted = pd.to_datetime(values, utc=True)
    if not isinstance(converted, pd.Series):
        converted = pd.Series(converted)
    arr = converted.to_numpy(dtype="datetime64[ns]")
    return np.ascontiguousarray(arr).view(np.int64)


def _availability_mode(source: Mapping[str, Any], kind: str) -> str:
    raw = source.get("availability")
    if kind == "predictions" and not raw:
        raw = AVAILABILITY_TIMESTAMP
    if kind == "predictions" and raw != AVAILABILITY_TIMESTAMP:
        raise CausalityError(
            "upstream predictions are decision-time scores and use availability 'timestamp'"
        )
    if raw not in {AVAILABILITY_TIMESTAMP, AVAILABILITY_BAR_CLOSE, AVAILABILITY_SESSION_CLOSE}:
        raise CausalityError(
            f"{kind} source must declare availability as timestamp, bar_close, or session_close"
        )
    return str(raw)


def _parse_clock(value: Any) -> tuple[int, int, int]:
    parts = str(value).split(":")
    if len(parts) not in (2, 3):
        raise CausalityError(f"session close {value!r} is not HH:MM")
    try:
        hour, minute = int(parts[0]), int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise CausalityError(f"session close {value!r} is not HH:MM") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise CausalityError(f"session close {value!r} is not a time of day")
    return hour, minute, second


def _session_close_ns(sessions: pd.Series, close_clock: Any, timezone: str) -> np.ndarray:
    """Close instant of the session each row describes, as UTC nanoseconds.

    A naive session value is a calendar date in ``timezone``, not a UTC
    midnight. An aware timestamp uses its local date in ``timezone``.
    """
    hour, minute, second = _parse_clock(close_clock)
    raw = pd.to_datetime(sessions, utc=False)
    try:
        if getattr(raw.dt, "tz", None) is not None:
            local = raw.dt.tz_convert(timezone).dt.normalize()
        else:
            local = raw.dt.normalize().dt.tz_localize(timezone)
        close = local + pd.Timedelta(hours=hour, minutes=minute, seconds=second)
        utc = close.dt.tz_convert("UTC")
    except (TypeError, ValueError) as exc:
        raise CausalityError(f"cannot place session close in {timezone}") from exc
    return _utc_ns(utc)


def _availability_ns(source: Mapping[str, Any], frame: pd.DataFrame, mode: str, path: Path) -> np.ndarray:
    time_column = str(source.get("time_column") or "ts")
    if mode == AVAILABILITY_SESSION_CLOSE:
        if not source.get("session_close") or not source.get("timezone"):
            raise CausalityError(f"{path} session_close source needs session_close and timezone")
        session_column = str(source.get("session_column") or "session")
        if session_column not in frame.columns:
            raise CausalityError(f"{path} has no session column {session_column!r}")
        return _session_close_ns(frame[session_column], source["session_close"], str(source["timezone"]))
    if time_column not in frame.columns:
        raise CausalityError(f"{path} has no time column {time_column!r}")
    opened = _utc_ns(frame[time_column])
    if mode == AVAILABILITY_TIMESTAMP:
        return opened
    try:
        duration = pd.Timedelta(source.get("bar_duration"))
    except (TypeError, ValueError) as exc:
        raise CausalityError(f"{path} bar_duration {source.get('bar_duration')!r} is not a duration") from exc
    if pd.isna(duration) or duration <= pd.Timedelta(0):
        raise CausalityError(f"{path} bar_duration must be positive")
    available = opened + np.int64(duration.value)
    if np.any(available < opened):
        raise CausalityError(f"{path} bar close timestamp overflowed")
    return available


def _reject_unclosed_bars(
    frame: pd.DataFrame, mode: str, time_column: str, available_ns: np.ndarray, path: Path,
) -> None:
    """A file that records ``bar_end_ts`` must not be visible before that close."""
    if "bar_end_ts" not in frame.columns:
        return
    end = _utc_ns(frame["bar_end_ts"])
    finite = end != _NAT
    if not finite.any():
        return
    if mode == AVAILABILITY_TIMESTAMP and time_column == "bar_end_ts":
        return
    if mode == AVAILABILITY_TIMESTAMP and time_column in frame.columns:
        start = _utc_ns(frame[time_column])
        if np.any((end > start) & finite & (start != _NAT)):
            raise CausalityError(
                f"{path} has bar_end_ts after {time_column}; join on bar_end_ts "
                "or declare availability bar_close"
            )
        return
    if mode == AVAILABILITY_BAR_CLOSE and np.any((available_ns < end) & finite & (available_ns != _NAT)):
        raise CausalityError(f"{path} bar_duration makes a bar visible before bar_end_ts")


def _group_positions(frame: pd.DataFrame, keys: list[str]) -> dict[tuple[Any, ...], np.ndarray]:
    if not keys:
        return {(): np.arange(len(frame), dtype=np.int64)}
    tagged = frame.loc[:, keys].copy()
    tagged["_row"] = np.arange(len(frame), dtype=np.int64)
    groups: dict[tuple[Any, ...], np.ndarray] = {}
    for key, group in tagged.groupby(keys, sort=False, dropna=False):
        boxed = key if isinstance(key, tuple) else (key,)
        groups[boxed] = group["_row"].to_numpy(dtype=np.int64)
    return groups


def _blank(n: int, series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return np.full(n, np.nan, dtype="float64")
    buf = np.empty(n, dtype=object)
    buf[:] = pd.NA
    return buf


def _values(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return series.to_numpy()


def _asof_take(
    decision_ns: np.ndarray,
    right_positions: np.ndarray,
    available_ns: np.ndarray,
) -> np.ndarray:
    """Positions into the right frame for each decision, or -1 when none are known yet.

    ``right_positions`` indexes ``available_ns``. Duplicate availability times
    keep the last row in file order. A chosen row's availability is checked
    against the decision time before the value is read.
    """
    if right_positions.size == 0 or decision_ns.size == 0:
        return np.full(decision_ns.size, -1, dtype=np.int64)
    avail = available_ns[right_positions]
    order = np.argsort(avail, kind="mergesort")
    avail_sorted = avail[order]
    if order.size > 1:
        keep = np.empty(order.size, dtype=bool)
        keep[-1] = True
        keep[:-1] = avail_sorted[:-1] != avail_sorted[1:]
        order = order[keep]
        avail_sorted = avail[order]
    chosen = right_positions[order]
    pos = as_of_index(avail_sorted, decision_ns)
    seen = pos >= 0
    if seen.any():
        assert_known_at_or_before(avail_sorted[pos[seen]], decision_ns[seen])
    taken = np.full(decision_ns.size, -1, dtype=np.int64)
    taken[seen] = chosen[pos[seen]]
    return taken


def _columns_to_attach(
    left: pd.DataFrame,
    right: pd.DataFrame,
    wanted: Sequence[str],
    reserved: set[str],
) -> list[str]:
    if wanted:
        names = [c for c in wanted if c not in left.columns and c in right.columns and c not in reserved]
    else:
        names = [c for c in right.columns if c not in left.columns and c not in reserved]
    return list(dict.fromkeys(names))


def _attach_asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    decision_ns: np.ndarray,
    available_ns: np.ndarray,
    by: list[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    if not columns:
        return left
    out = left.copy()
    buffers = [_blank(len(out), right[col]) for col in columns]
    stored = [_values(right[col]) for col in columns]
    left_groups = _group_positions(out, by)
    right_groups = _group_positions(right, by)
    for key, left_pos in left_groups.items():
        right_pos = right_groups.get(key)
        if right_pos is None or right_pos.size == 0:
            continue
        taken = _asof_take(decision_ns[left_pos], right_pos, available_ns)
        hit = taken >= 0
        if not hit.any():
            continue
        dest = left_pos[hit]
        src = taken[hit]
        for buf, values in zip(buffers, stored):
            buf[dest] = values[src]
    for name, buf in zip(columns, buffers):
        out[name] = buf
    return out


def _join_sources(
    frame: pd.DataFrame,
    inputs: Mapping[str, Any],
    run_directory: Path,
    kinds: Sequence[str],
) -> pd.DataFrame:
    """Attach feature, context, and prediction columns as-of the decision ``ts``.

    Within the other join keys, each decision receives the latest source row
    whose availability is at or before ``ts``, and never a later row. A closed
    bar is available at its close. A session summary is available at that
    session's close, so the session being decided does not contribute its own
    completed profile. Labels are not joined here.
    """
    if "ts" not in frame.columns:
        raise CausalityError("decision frame has no ts; features cannot be joined")
    keys = [str(k) for k in (inputs.get("join_keys") or [])]
    if "ts" not in keys:
        raise CausalityError("join_keys must include ts")
    decision_ns = _utc_ns(frame["ts"])
    if decision_ns.size and np.any(decision_ns == _NAT):
        raise CausalityError("decision timestamp is missing; cannot join features")
    for kind in kinds:
        for source in _source_list(inputs, kind):
            frame = _join_one(frame, decision_ns, source, kind, keys, run_directory)
    return frame


def _join_one(
    frame: pd.DataFrame,
    decision_ns: np.ndarray,
    source: Mapping[str, Any],
    kind: str,
    join_keys: Sequence[str],
    run_directory: Path,
) -> pd.DataFrame:
    location = source.get("location")
    if not location:
        return frame
    path = resolve_input_path(str(location), run_directory)
    if not path.exists():
        return frame
    mode = _availability_mode(source, kind)
    time_column = str(source.get("time_column") or "ts")
    session_column = str(source.get("session_column") or "session")
    wanted = [str(c) for c in (source.get("columns") or [])]
    ignore = {"ts"}
    if mode == AVAILABILITY_SESSION_CLOSE:
        ignore.add(session_column)
    by = [k for k in join_keys if k not in ignore]
    if mode != AVAILABILITY_SESSION_CLOSE:
        missing_left = [k for k in by if k not in frame.columns]
        if missing_left:
            raise CausalityError(
                f"decision frame is missing join key(s) {missing_left}"
            )
    by = [k for k in by if k in frame.columns]
    if mode != AVAILABILITY_SESSION_CLOSE:
        read_keys = [*by, time_column, "bar_end_ts"]
    else:
        read_keys = [*by, session_column, "bar_end_ts"]
    if _already_on_frame(frame, path, wanted, set(read_keys)):
        return frame
    if wanted:
        other = read_source_frame(path, list(dict.fromkeys([*read_keys, *wanted])))
    else:
        other = read_source_frame(path)
    if mode == AVAILABILITY_SESSION_CLOSE:
        # A session summary is not matched on the session it describes. Other
        # keys (symbol, side) group only when the file actually carries them.
        by = [k for k in by if k in other.columns]
    else:
        missing_keys = [k for k in by if k not in other.columns]
        if missing_keys:
            raise CausalityError(
                f"{path} is missing join key(s) {missing_keys}; refusing an ungrouped as-of"
            )
    if other.empty:
        return frame
    available = _availability_ns(source, other, mode, path)
    if np.any(available == _NAT):
        raise CausalityError(f"{path} has a row with no availability timestamp")
    _reject_unclosed_bars(other, mode, time_column, available, path)
    reserved = set(by) | {"ts", time_column, "bar_end_ts"}
    if mode == AVAILABILITY_SESSION_CLOSE:
        reserved.add(session_column)
    attach = _columns_to_attach(frame, other, wanted, reserved)
    if not attach:
        return frame
    return _attach_asof(
        frame, other,
        decision_ns=decision_ns,
        available_ns=available,
        by=by,
        columns=attach,
    )


def apply_filters(frame: pd.DataFrame, filters: Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    """Apply ``[{op, left, right}]`` filters. ``right`` may be a column name."""
    out = frame
    for spec in filters or []:
        op = str(spec.get("op") or "")
        left_name = str(spec["left"])
        if left_name not in out.columns:
            raise KeyError(f"filter column {left_name!r} is not in the frame")
        left = out[left_name]
        right = spec.get("right")
        if isinstance(right, str) and right in out.columns:
            right_value: Any = out[right]
        else:
            right_value = right
        if op == "gte":
            mask = left >= right_value
        elif op == "gt":
            mask = left > right_value
        elif op == "eq":
            mask = left == right_value
        elif op == "in":
            mask = left.isin(right_value)
        else:
            raise ValueError(f"unsupported filter op {op!r}")
        out = out.loc[mask]
    return out


def load_training_frame(inputs: Mapping[str, Any], run_directory: Path) -> pd.DataFrame:
    """Join the model's sources. Uses the population cache when that file exists.

    The cache is one materialization of a row population. Feature subsets are
    a column selection on this frame, not a second file. Columns read from
    another dataset are attached as-of the decision timestamp. Outcome columns
    already on the label row are left as they are.
    """
    missing = validate_model_inputs(inputs)
    if missing:
        raise ValueError(f"model inputs are incomplete: {missing}")
    run_directory = Path(run_directory)
    frame: pd.DataFrame | None = None
    cache = inputs.get("population_cache")
    if cache:
        path = resolve_input_path(str(cache), run_directory)
        if path.exists():
            frame = read_source_frame(path)
    if frame is None:
        labels = inputs.get("labels") if isinstance(inputs.get("labels"), Mapping) else {}
        location = labels.get("location") if isinstance(labels, Mapping) else None
        if not location:
            raise FileNotFoundError("model inputs have no readable population cache or label dataset")
        frame = read_source_frame(resolve_input_path(str(location), run_directory))
    frame = _join_sources(frame, inputs, run_directory, ("features", "context", "predictions"))
    return apply_filters(frame, inputs.get("filters") or [])


def load_run_frame(
    objective_id: str,
    run_ts: str,
    pass_n: int,
    model_id: str,
    *,
    explicit: Path | None = None,
) -> pd.DataFrame:
    """Training frame for one model: an explicit directory, else its join spec."""
    if explicit is not None:
        return read_source_frame(Path(explicit))
    from src.common.run_layout import model_dir, resolve_population_cache, run_dir
    directory = run_dir(objective_id, run_ts)
    meta_path = model_dir(objective_id, run_ts, pass_n, model_id) / MODEL_JSON
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        inputs = meta.get("inputs")
        if isinstance(inputs, Mapping) and not validate_model_inputs(inputs):
            return load_training_frame(inputs, directory)
    cache = resolve_population_cache(objective_id, run_ts, pass_n)
    if cache is None:
        raise FileNotFoundError(
            f"no input spec or population cache for {objective_id}/{run_ts} pass {pass_n}"
        )
    return read_source_frame(cache)


def build_oof_analysis(
    objective_id: str,
    run_ts: str,
    pass_n: int,
    model_id: str,
    *,
    materialize: Path | None = None,
) -> pd.DataFrame:
    """Compact OOF joined back to the feature columns that model trained on.

    Reads ``manifest.json`` (or the legacy run files) and ``model.json``.
    ``materialize`` writes a temporary ``analysis_view.parquet`` for Data
    Wrangler. That file is not a run artifact.
    """
    from src.common.run_layout import model_dir, read_run, resolve_population_cache, run_dir
    directory = run_dir(objective_id, run_ts)
    meta_path = model_dir(objective_id, run_ts, pass_n, model_id) / MODEL_JSON
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text())
    oof_path = model_dir(objective_id, run_ts, pass_n, model_id) / OOF_PREDICTIONS
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)
    frame = read_source_frame(oof_path)
    record = read_run(objective_id, run_ts)
    inputs = meta.get("inputs") if isinstance(meta.get("inputs"), Mapping) else None
    if not inputs:
        for entry in record.get("passes") or []:
            if int(entry.get("pass", -1)) == int(pass_n) and isinstance(entry.get("inputs"), Mapping):
                inputs = entry["inputs"]
                break
    feature_columns = [str(c) for c in (meta.get("feature_columns") or [])]
    if inputs:
        scoped = dict(inputs)
        if feature_columns:
            features = _source_list(scoped, "features") + _source_list(scoped, "context")
            if features:
                narrowed = []
                for source in features:
                    item = dict(source)
                    if not item.get("columns"):
                        item["columns"] = feature_columns
                    narrowed.append(item)
                scoped = dict(scoped)
                scoped["features"] = narrowed
                scoped["context"] = []
        frame = _join_sources(frame, scoped, directory, ("features", "context"))
    elif feature_columns:
        cache = resolve_population_cache(objective_id, run_ts, pass_n)
        if cache is not None:
            frame = _join_sources(
                frame,
                {"join_keys": ["ts", "side"], "features": [{
                    "location": str(cache),
                    "columns": feature_columns,
                    "availability": AVAILABILITY_TIMESTAMP,
                    "time_column": "ts",
                }]},
                directory,
                ("features",),
            )
    if materialize is not None:
        dest = Path(materialize)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dest, index=False, compression="zstd")
    return frame


def default_analysis_view_path(objective_id: str, run_ts: str, pass_n: int, model_id: str) -> Path:
    from src.common.run_layout import model_dir
    return model_dir(objective_id, run_ts, pass_n, model_id) / ANALYSIS_VIEW


def write_shap_report(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target: str,
    out_dir: Path,
    *,
    max_depth: int = 6,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    seed: int = 42,
    title: str = "shap",
) -> pd.DataFrame | None:
    """SHAP for one finished OOF, using that model's feature columns.

    Fits one tree model on the rows before the last 60 sessions and explains
    a sample of the holdout. Called after the OOF file is written. Returns
    the top-20 table, or None when the frame has no usable features.
    """
    features = [c for c in feature_columns if c in frame.columns and c != target]
    if not features or target not in frame.columns or frame.empty:
        return None
    import numpy as np
    work = frame.dropna(subset=[target]).copy()
    if "session" in work.columns:
        cutoff = pd.to_datetime(work["session"]).max() - pd.Timedelta(days=60)
        train = work[pd.to_datetime(work["session"]) < cutoff]
        hold = work[pd.to_datetime(work["session"]) >= cutoff]
        if train.empty or hold.empty:
            train, hold = work, work
    else:
        train, hold = work, work
    sample = hold.sample(min(20_000, len(hold)), random_state=seed)
    import xgboost as xgb
    y_train = train[target].to_numpy()
    binary = set(pd.unique(y_train)).issubset({0, 1, 0.0, 1.0})
    if binary:
        pos = float(np.mean(y_train))
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=0,
            eval_metric="logloss",
            scale_pos_weight=(1 - pos) / max(pos, 1e-9),
        )
        model.fit(train[features].to_numpy("float32"), y_train.astype("int8"))
    else:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=0,
        )
        model.fit(train[features].to_numpy("float32"), y_train.astype("float32"))
    booster = model.get_booster()
    matrix = xgb.DMatrix(sample[features].to_numpy("float32"), feature_names=features)
    values = booster.predict(matrix, pred_contribs=True)[:, :-1]
    mean_abs = np.abs(values).mean(axis=0)
    ranked = (pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True))
    top = ranked.head(20).copy()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(out_dir / "shap_importance.csv", index=False)
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    top_features = top["feature"].tolist()
    top_idx = [features.index(name) for name in top_features]
    order = np.arange(len(top_features))[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(order, top["mean_abs_shap"].to_numpy(), color="#2f6f8f")
    ax.set_yticks(order, top_features)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_bar.png", dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(seed)
    for row, feature_idx in enumerate(top_idx):
        y = np.full(len(values), row, dtype="float64")
        y += rng.uniform(-0.22, 0.22, len(values))
        colour = sample[features[feature_idx]].to_numpy(dtype="float64")
        scale = np.nanpercentile(np.abs(colour), 99)
        colour = np.clip(colour / max(scale, 1e-9), -1.0, 1.0)
        ax.scatter(values[:, feature_idx], y, c=colour, cmap="coolwarm",
                   vmin=-1.0, vmax=1.0, s=3, alpha=0.18, linewidths=0, rasterized=True)
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_yticks(np.arange(len(top_features)), top_features)
    ax.set_xlabel("SHAP value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_dir / "shap_beeswarm.png", dpi=120)
    plt.close(fig)
    return top
