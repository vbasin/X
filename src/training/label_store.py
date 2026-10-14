"""Persisted TBM label stores (ADR-076).

5s 3-class files sit beside the existing 15s dual-binary projection.
Fold cache joins on ``ts`` when the sidecar contract matches.
"""

from __future__ import annotations

import json
import hashlib
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.data_paths import labels_dir, outcome_dataset_dir
from src.training.aux_labels import AUX_LABEL_COLS, attach_aux_labels
from src.training.labels import generate_labels
from src.training.config import LabelConfig
from src.training.dual_binary_15s import label_params_from_objective
CONTRACT_NAME = "_contract.json"
#: Realization/build-provenance sidecar (tenth pass) -- distinct from
#: CONTRACT_NAME's semantic identity, see write_build_sidecar().
BUILD_NAME = "_build.json"
LABEL_PART_MAX_BYTES = 100 * 1024 * 1024
#: _contract.json's own shape version (a sidecar field, never hashed into
#: contract_id -- see normalize_contract()). Bump only if the sidecar's
#: OWN fields change shape; a semantic contract field change is a new
#: contract_id, not a schema_version bump.
CONTRACT_SCHEMA_VERSION = 1


def normalize_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Universal semantic-contract normalization, shared by every outcome family.

    Strips bookkeeping keys that describe the SIDECAR FILE, not the outcome's
    meaning (``contract_id`` is derived FROM the hash, ``created_at_utc``/
    ``schema_version`` are provenance about the write, not the definition) --
    popping all three keeps :func:`label_contract_id`/:func:`outcome_contract_id`
    stable across a read-modify-write round trip through :func:`write_contract`.
    Applies NO family-specific defaults: see :func:`normalize_label_contract`
    for Track B's TBM-specific default, which does not apply to other
    outcome families (Track C's move/target contracts have no
    ``target_observation`` concept at all).
    """
    out = dict(contract)
    # applies_to / dataset_dir tie a contract file to a run. They are not
    # part of what the labels mean, so they must not change contract_id.
    for key in ("contract_id", "created_at_utc", "schema_version",
                "applies_to", "dataset_dir"):
        out.pop(key, None)
    return out


def label_contract_id(contract: dict[str, Any]) -> str:
    """Stable identifier for a normalized label contract, not an objective name."""
    raw = json.dumps(normalize_label_contract(contract), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def normalize_label_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Track B TBM semantic fields: universal normalization plus the TBM default."""
    out = normalize_contract(contract)
    out.setdefault("target_observation", "candle_close")
    return out


def outcome_contract_id(contract: dict[str, Any]) -> str:
    """Track-neutral semantic-contract hash (docs/STORAGE_CONTRACT.md §17.1).

    Same mechanism :func:`label_contract_id` uses (sha256 of the normalized
    contract dict, same JSON serialization) but via :func:`normalize_contract`
    rather than :func:`normalize_label_contract` -- no Track-B-specific
    ``target_observation`` default is forced onto a contract from a
    different outcome family. Use this for any NEW outcome family (Track C's
    move/target contracts, or a future one); Track B's own
    :func:`label_contract_id`/:func:`label_set_dir` are unchanged and remain
    the compatibility path for its existing on-disk directories.
    """
    raw = json.dumps(normalize_contract(contract), sort_keys=True,
                     separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def label_set_dir(resolution: str, contract: dict[str, Any]) -> Path:
    """Contract-addressed label directory for one decision resolution."""
    return labels_dir() / str(resolution) / label_contract_id(contract)


def outcome_set_dir(outcome_family: str, resolution: str, contract: dict[str, Any]) -> Path:
    """Canonical NEW-write directory for any outcome dataset (§17.1).

    Track-neutral generalization of :func:`label_set_dir`: same contract-hash
    mechanism (:func:`outcome_contract_id`), addressed by
    ``<outcome_family>/<resolution>`` rather than Track B's fixed
    ``<resolution>`` alone. Pure path construction plus one hash -- does not
    read, write or resolve any existing compatibility path.
    """
    return outcome_dataset_dir(outcome_family, resolution, outcome_contract_id(contract))


def outcome_part_name(part: int | date) -> str:
    """Canonical outcome-dataset part filename (docs/STORAGE_CONTRACT.md §18).

    An integer part number for size-split multi-session parts -- Track B's
    existing convention, generalized and zero-padded (``part001.parquet``,
    correct sort order past 9 parts, unlike the un-padded
    ``labels_15s_part10.parquet`` compatibility naming). A session ``date``
    for producers whose natural grain is one file per session (Track C's
    move/target builders -- ``part_2026-01-01.parquet``). Both match the
    same ``part*.parquet`` glob and carry no family/track/stage vocabulary:
    the directory already encodes that (§17.1), so the filename does not
    need to repeat it.
    """
    if isinstance(part, (date, datetime)):
        d = part.date() if isinstance(part, datetime) else part
        return f"part_{d.isoformat()}.parquet"
    n = int(part)
    if n < 1:
        raise ValueError(f"part must be >= 1, got {part!r}")
    return f"part{n:03d}.parquet"


def move_outcome_contract(
    *, decision_grid_seconds: int, horizon_seconds: int, move_r: float,
    adverse_r: float, r_lookback_sessions: int,
) -> dict[str, Any]:
    """Semantic contract for Track C's directional first-passage (move) outcome.

    Only fields that change what ``y_move`` MEANS -- see
    docs/STORAGE_CONTRACT.md §19 for the full classification of
    ``build_move_matrix.py``'s CLI into semantic / build-population /
    execution-only. ``r_lookback_sessions`` is included as semantic (not
    merely population config): it defines how the R normalization unit
    itself is computed, which changes what "big enough" means for the
    same raw point distance.
    """
    return {
        "family": "directional_first_passage",
        "decision_grid_seconds": int(decision_grid_seconds),
        "horizon_seconds": int(horizon_seconds),
        "move_r": float(move_r),
        "adverse_r": float(adverse_r),
        "r_lookback_sessions": int(r_lookback_sessions),
    }


#: Key and outcome columns of a directional-screen label row. Feature
#: columns are everything else on that row's training matrix.
MOVE_LABEL_COLUMNS: tuple[str, ...] = (
    "ts", "session", "side", "y_move", "reached", "t_move",
    "mfe_pts", "mae_pts", "price", "r_value",
)


def target_outcome_contract(
    *, horizon_seconds: int, max_targets: int, stop_ratio: float, max_stop_r: float,
    resting_entry: bool, max_fill_wait: int, lookback_sessions: int,
    r_lookback_sessions: int,
) -> dict[str, Any]:
    """Semantic contract for Track C's candidate-level reachability (target) outcome.

    See docs/STORAGE_CONTRACT.md §19 for the full field classification.
    ``lookback_sessions`` is semantic here (unlike in
    :func:`move_outcome_contract`): it determines which structural bounds
    are even considered candidate targets, not merely a feature-history
    depth, so it changes what ``y_reach`` can be true OF.
    """
    return {
        "family": "candidate_level_reachability",
        "horizon_seconds": int(horizon_seconds),
        "max_targets": int(max_targets),
        "stop_ratio": float(stop_ratio),
        "max_stop_r": float(max_stop_r),
        "resting_entry": bool(resting_entry),
        "max_fill_wait": int(max_fill_wait),
        "lookback_sessions": int(lookback_sessions),
        "r_lookback_sessions": int(r_lookback_sessions),
    }


#: Key and outcome columns of a target-reach label row. ``p_move`` is an
#: upstream prediction, not part of this label.
TARGET_LABEL_COLUMNS: tuple[str, ...] = (
    "ts", "session", "side", "price", "decision_price", "t_fill", "stop_px",
    "tgt_level", "y_reach", "t_reach", "t_stop", "tgt_rank", "tgt_dist_r",
    "stp_dist_r", "tgt_strength", "tgt_n_levels", "tgt_width_r",
    "tgt_newest_age", "tgt_frac_value_edges", "tgt_frac_poc", "n_bounds",
    "inside_bound", "r_value",
)


def write_build_sidecar(
    d: Path,
    *,
    population_config: dict[str, Any],
    upstream: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Realization/provenance sidecar, distinct from ``_contract.json``'s
    semantic identity (docs/STORAGE_CONTRACT.md §5a, §18, §19).

    Records ONLY genuine build evidence: the current git commit + dirty flag
    (``src.common.code_identity``, shared with ``RunManifest.code``), the
    caller's build/population configuration (values that change which
    rows/columns are realized WITHOUT redefining the outcome -- date range,
    an upstream gate quantile, a context-column mode; never the semantic
    contract fields themselves), and upstream lineage (``DatasetRef``-shaped
    dicts, e.g. a target dataset's move-dataset and move-score-predictions
    inputs). Writes nothing and returns ``None`` when git identity is
    unavailable -- a fabricated ``build_id`` is worse than an absent one.
    """
    from src.common.code_identity import git_identity
    got = git_identity()
    if got is None:
        return None
    sha, dirty = got
    payload: dict[str, Any] = {
        "code_sha": sha,
        "code_dirty": dirty,
        "population_config": population_config,
        "upstream": upstream or [],
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    build_id = hashlib.sha256(blob.encode()).hexdigest()[:16]
    d.mkdir(parents=True, exist_ok=True)
    out = d / BUILD_NAME
    out.write_text(json.dumps({**payload, "build_id": build_id}, indent=2, sort_keys=True) + "\n")
    return out


def tbm_contract_from_objective(objective: dict[str, Any]) -> dict[str, Any]:
    p = label_params_from_objective(objective)
    enc = str((objective.get("labels") or {}).get("encoding") or "3class").strip()
    if enc not in ("dual_binary", "3class"):
        enc = "3class" if enc not in ("dual_binary",) else enc
    if not (objective.get("labels") or {}).get("encoding"):
        enc = "3class"
    return {
        "scheme": p["scheme"],
        "move_threshold": p["move_threshold"],
        "mae_tolerance": p["mae_tolerance"],
        "horizon_s": p["horizon_s"],
        "target_observation": p["target_observation"],
        "encoding": enc,
        "aux_labels": list(AUX_LABEL_COLS),
        "time_to_tgt": "nan_on_timeout",
    }


def _read_contract(d: Path) -> dict[str, Any] | None:
    p = d / CONTRACT_NAME
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def write_contract(d: Path, contract: dict[str, Any]) -> None:
    """Write ``_contract.json``. Track-neutral: normalizes via whichever of
    :func:`normalize_label_contract`/:func:`normalize_contract` the caller's
    *contract* dict was already built with -- this function itself only
    needs :func:`label_contract_id`'s TBM-aware hash for Track B's existing
    callers; Track C's canonical-write path calls :func:`outcome_contract_id`
    directly to get its own ``contract_id`` and passes it through unchanged
    (see ``outcome_set_dir`` callers). ``schema_version`` is the sidecar
    file's own shape version -- added after hashing, excluded from it by
    :func:`normalize_contract`, exactly like ``created_at_utc``.
    """
    d.mkdir(parents=True, exist_ok=True)
    prior = _read_contract(d) or {}
    normalized = normalize_label_contract(contract)
    payload = {
        **normalized,
        "contract_id": label_contract_id(normalized),
        "created_at_utc": prior.get("created_at_utc")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": prior.get("schema_version") or CONTRACT_SCHEMA_VERSION,
    }
    (d / CONTRACT_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_outcome_contract(d: Path, contract: dict[str, Any]) -> str:
    """Track-neutral ``_contract.json`` writer -- companion to
    :func:`write_contract` (Track B's TBM-specific writer) for any other
    outcome family (Track C's move/target contracts, or a future one).
    Uses :func:`normalize_contract`/:func:`outcome_contract_id` throughout,
    so the ``contract_id`` recorded in the sidecar always matches
    :func:`outcome_set_dir`'s own directory name for the same contract dict
    -- using :func:`write_contract` here would silently inject Track B's
    ``target_observation`` default and hash mismatch the directory. Returns
    the ``contract_id`` written.
    """
    d.mkdir(parents=True, exist_ok=True)
    prior = _read_contract(d) or {}
    normalized = normalize_contract(contract)
    contract_id = outcome_contract_id(normalized)
    payload = {
        **normalized,
        "contract_id": contract_id,
        "created_at_utc": prior.get("created_at_utc")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": prior.get("schema_version") or CONTRACT_SCHEMA_VERSION,
    }
    if "applies_to" in contract:
        payload["applies_to"] = contract["applies_to"]
    elif prior.get("applies_to"):
        payload["applies_to"] = prior["applies_to"]
    if "dataset_dir" in contract:
        payload["dataset_dir"] = contract["dataset_dir"]
    elif prior.get("dataset_dir"):
        payload["dataset_dir"] = prior["dataset_dir"]
    (d / CONTRACT_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return contract_id


def tie_outcome_contract(
    d: Path,
    *,
    objective_id: str,
    run_ts: str,
    pass_n: int,
    model_id: str,
    dataset_dir: str | None = None,
) -> str:
    """Legacy: append a consuming run onto the label contract's ``applies_to``.

    New objective runs do not call this. A label contract is a reusable
    definition. The run's manifest references ``contract_id``; the contract
    file does not gain a list of runs.
    """
    raw = _read_contract(d)
    if not raw or "contract_id" not in raw:
        raise FileNotFoundError(f"no {CONTRACT_NAME} under {d}")
    link = {
        "objective_id": objective_id,
        "run_ts": run_ts,
        "pass": int(pass_n),
        "model_id": model_id,
    }
    applies = [a for a in (raw.get("applies_to") or []) if a != link]
    applies.append(link)
    body = dict(raw)
    body["applies_to"] = applies
    if dataset_dir is not None:
        body["dataset_dir"] = dataset_dir
    return write_outcome_contract(d, body)


def contract_matches(stored: dict[str, Any] | None, wanted: dict[str, Any]) -> bool:
    if not stored:
        return False
    stored = normalize_label_contract(stored)
    wanted = normalize_label_contract(wanted)
    return label_contract_id(stored) == label_contract_id(wanted)


def write_label_parts(
    dest: Path,
    frames: list[pd.DataFrame],
    *,
    resolution: str,
    max_bytes: int = LABEL_PART_MAX_BYTES,
    force: bool = False,
) -> list[Path]:
    """Write ordered label frames as compressed resolution-specific parts."""
    if not frames:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"labels_{resolution}_part"
    existing = sorted(dest.glob(f"{prefix}*.parquet"))
    if existing and not force:
        return existing
    if force:
        for path in existing:
            path.unlink()
    full = pd.concat(frames, ignore_index=True).sort_values("ts").drop_duplicates("ts", keep="last")
    if full.empty:
        return []
    probe = full.iloc[:min(len(full), 50_000)]
    probe_path = dest / ".label_probe.parquet"
    probe.to_parquet(probe_path, index=False, compression="zstd")
    bytes_per_row = max(1.0, probe_path.stat().st_size / max(len(probe), 1))
    probe_path.unlink()
    rows_per_part = max(1, int(max_bytes / bytes_per_row * 0.92))
    paths: list[Path] = []
    for part, start in enumerate(range(0, len(full), rows_per_part), start=1):
        path = dest / f"{prefix}{part}.parquet"
        full.iloc[start:start + rows_per_part].to_parquet(path, index=False, compression="zstd")
        paths.append(path)
    # Producer-side build record, for FUTURE writes only. Nothing is backfilled:
    # a dataset built before this existed has no build evidence and must keep
    # saying so. With it, a run consuming this dataset can record a genuine
    # build_id and reach CANONICAL_COMPLETE on dataset evidence instead of being
    # capped at contract identity forever. Writes nothing when git is
    # unavailable -- a fabricated build_id is worse than an absent one.
    write_build_sidecar(dest, population_config={
        "resolution": str(resolution),
        "n_rows": int(len(full)),
        "n_parts": len(paths),
        "ts_min": str(full["ts"].iloc[0]),
        "ts_max": str(full["ts"].iloc[-1]),
        "max_part_bytes": int(max_bytes),
    })
    return paths


def write_label_contract_snapshot(objective: dict[str, Any], run_dir: Path) -> Path | None:
    """Record the exact selected 15s label set in a native-TF research run."""
    if str((objective.get("search") or {}).get("input") or "") != "native_tf":
        return None
    contract = tbm_contract_from_objective(objective)
    source = label_set_dir("15s", contract)
    stored = _read_contract(source)
    if not contract_matches(stored, contract):
        raise FileNotFoundError(
            f"no matching 15s label set for contract {label_contract_id(contract)} at {source}")
    parts = sorted(source.glob("labels_15s_part*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no label parts under {source}")
    out = run_dir / "label_contract.json"
    out.write_text(json.dumps({
        "contract": stored,
        "contract_id": label_contract_id(contract),
        "source_dir": str(source),
        "contract_sha256": hashlib.sha256((source / CONTRACT_NAME).read_bytes()).hexdigest(),
        "parts": [{"name": p.name, "bytes": p.stat().st_size} for p in parts],
    }, indent=2, sort_keys=True) + "\n")
    return out


def labels_3class_from_close(
    close: pd.Series,
    *,
    move_threshold: float = 50.0,
    mae_tolerance: float = 6.0,
    horizon_s: int = 600,
    scheme: str = "triple_barrier",
    src_step_s: int = 5,
    target_observation: str = "candle_close",
    target_high: pd.Series | None = None,
    target_low: pd.Series | None = None,
) -> pd.DataFrame:
    close = pd.Series(close, copy=False)
    close.index = pd.to_datetime(close.index, utc=True)
    window = max(1, int(horizon_s) // max(1, int(src_step_s)))
    cfg = LabelConfig(
        big_move_threshold=float(move_threshold),
        mae_tolerance=float(mae_tolerance),
        forward_windows=[window],
        scheme=scheme,
        target_observation=target_observation,
    )
    lab = generate_labels(
        close, cfg, target_high=target_high, target_low=target_low)["label"].astype("int8")
    out = pd.DataFrame({
        "label": lab,
        "close": pd.to_numeric(close, errors="coerce").astype("float32"),
    }, index=lab.index)
    out = attach_aux_labels(
        out, close,
        move_threshold=float(move_threshold),
        horizon_s=int(horizon_s),
        src_step_s=int(src_step_s),
        target_observation=target_observation,
        target_high=target_high,
        target_low=target_low,
    )
    out.index.name = "ts"
    return out


def _as_utc_index(df: pd.DataFrame) -> pd.DataFrame | None:
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            df = df.copy()
            df.index = idx.tz_localize("UTC")
        else:
            df = df.copy()
            df.index = idx.tz_convert("UTC")
        df.index.name = "ts"
        return df
    if "ts" in df.columns:
        out = df.copy()
        out["ts"] = pd.to_datetime(out["ts"], utc=True)
        out = out.set_index("ts")
        return out
    return None


def _load_day_labels(d: Path, start, end, *, part_glob: str = "labels_*.parquet") -> pd.DataFrame | None:
    if not d.is_dir():
        return None
    files = sorted(d.glob(part_glob))
    if not files:
        return None
    frames: list[pd.DataFrame] = []
    for f in files:
        got = _as_utc_index(pd.read_parquet(f))
        if got is not None and not got.empty:
            frames.append(got)
    if not frames:
        return None
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out.loc[out.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    from src.training.aux_labels import sanitize_time_to_tgt
    return sanitize_time_to_tgt(out)


def load_labels_5s(
    start: str | date | None,
    end: str | date | None,
    *,
    contract: dict[str, Any],
) -> pd.DataFrame | None:
    return _load_day_labels(
        label_set_dir("5s", contract), start, end, part_glob="labels_5s_part*.parquet")


def load_labels_15s(
    start: str | date | None,
    end: str | date | None,
    *,
    contract: dict[str, Any],
) -> pd.DataFrame | None:
    return _load_day_labels(
        label_set_dir("15s", contract), start, end, part_glob="labels_15s_part*.parquet")
