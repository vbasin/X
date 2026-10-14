"""Emit a Track B run in the canonical run layout.

``src/search/run_manifest.py`` builds the ``RunManifest`` OBJECT for a Track B
run directory. This module writes the whole canonical run TREE for it:
``pass_<n>/model_<model_id>/`` folders, ``model.json``, a compact OOF, metrics,
the report set, and ``manifest.json`` via ``merge_pass``.

It is an adapter, not a second architecture: every filename, path, identity and
join rule comes from ``src/experiment/artifacts.py``,
``model_identity.py``, ``manifest.py`` and ``src/common/run_layout.py``. Nothing
here defines a name or a layout of its own.

Two callers, one code path:

* ``src/search/controller.py`` calls :func:`emit_canonical_run` when a search
  run finishes, so a FUTURE Track B run lands in the canonical layout as part
  of the run rather than waiting for a later migration. Weight files are
  hard-linked, so the canonical tree is complete without storing a second copy
  of the bytes.
* ``scripts/research/migrate_custom_test.py`` calls it for an EXISTING run,
  copying instead of linking.

Copy-first either way: nothing under ``research/custom_test/`` is moved,
renamed or deleted, and ``run.json`` / ``search_result.json`` / ``folds.json``
keep being written by the controller for their current readers.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.economic import assumptions_record
from src.experiment.artifacts import (
    AVAILABILITY_BAR_CLOSE,
    AVAILABILITY_TIMESTAMP,
    MODEL_ARTIFACT_FILENAMES,
    compact_oof_columns,
)
from src.experiment.manifest import CodeIdentity, read_run_record
from src.experiment.model_identity import (
    allocate_model_id,
    validate_model_id,
    write_json,
    write_model_json,
)
from src.common.data_paths import custom_test_dir
from src.common.run_layout import merge_run_json, model_dir, report_dir

#: docs/new_file_map.md's pass rule. A meta candidate consumes other
#: candidates' out-of-fold predictions, so it is pass 2; everything else in the
#: run is pass 1. Matched on the candidate id, which is how these runs name
#: their topologies.
_META_PREFIXES = ("meta__", "meta_stack__", "tf_oof_meta")

#: Files copied verbatim into the model folder when the bundle has them.
#: `realises` is recorded on model.json rather than the file being renamed --
#: nothing here is retired until src/serving/bundle.py reads model.json.
_BUNDLE_FILES = (
    "scaler.pkl",            # train-fitted; replay applies it before the model
    "feature_schema.json",   # serving feature list; same columns on model.json
    "finalize_manifest.json",  # bundle provenance the serving loader reads
    "compose.json",          # which upstream OOF columns a meta model consumes
    "inference_policy.json",  # tau and session window used at inference
    "meta.json",             # a live meta bundle is meta.json + meta.pkl
    "archetype_stats.json",
    "diagnostics.json",
)

#: SHAP report files, copied under their own names into the model's report/.
#: They are NOT renamed to shap_importance.csv -- a top-20 extract is not the
#: full importance table (docs/new_file_map.md).
_SHAP_FILES = (
    "SHAP_SUMMARY.md", "shap_bar_global.png", "shap_bar_signals.png",
    "shap_beeswarm_up.png", "shap_beeswarm_down.png", "shap_beeswarm_flat.png",
    "shap_groups.csv", "shap_importance.csv", "shap_meta.json",
    "shap_values_sample.npz",
)


def _objective_document(objective_id: str) -> dict[str, Any]:
    """The objective YAML, read for its cost declaration only.

    Deliberately NOT used as ``config_snapshot``: this is today's file, and a
    run that captured no snapshot must not be given one retroactively.
    """
    try:
        from src.objectives.loader import load_objective
        return dict(load_objective(objective_id))
    except Exception:
        return {}


_SHA_RE = __import__("re").compile(r"^[0-9a-f]{40}$")


def _source_code_identity(
    src: Path, candidates: list[dict[str, Any]],
) -> tuple[CodeIdentity | None, str | None]:
    """``(code identity, why not)`` for the commit the RUN was fitted at.

    Read from each bundle's ``finalize_manifest.json``. Stamping a migrated run
    with today's ``HEAD`` -- which is what ``merge_pass`` does by default --
    would claim the run was produced by code it never saw, so the source's own
    record wins whenever it is usable.

    Only a real 40-hex SHA counts. One bundle in the audited run stored the
    literal string ``"HEAD"`` (a failed capture), and that is not an identity.
    ``dirty`` is recorded ``True``: these runs had no dirty check at all, and
    ``CodeIdentity`` deliberately refuses to default the question -- an
    unqualified SHA from a tree that was never inspected is the weaker claim,
    so it is recorded as the weaker claim.
    """
    found: dict[str, list[str]] = {}
    unusable: list[str] = []
    for cand in candidates:
        cid = str(cand.get("candidate_id"))
        sha = (_read_json(src / "models" / cid / "finalize_manifest.json")
               .get("git_sha"))
        text = str(sha or "").strip().lower()
        if _SHA_RE.match(text):
            found.setdefault(text, []).append(cid)
        elif text:
            unusable.append(f"{cid}={sha!r}")
    if len(found) == 1:
        sha = next(iter(found))
        why = (f"unusable git_sha on {', '.join(unusable)}" if unusable else None)
        return CodeIdentity(git_sha=sha, dirty=True), why
    if not found:
        return None, ("no usable git_sha in any finalize_manifest.json"
                      + (f" ({', '.join(unusable)})" if unusable else ""))
    return None, (
        "the fits in this run record different commits "
        + "; ".join(f"{sha[:12]}: {', '.join(ids)}" for sha, ids in sorted(found.items()))
        + " -- no single run-level code identity is true"
    )


def _applied_costs(candidates: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    """The cost a Track B run demonstrably applied, from its own pnl block.

    ``cost_per_trade`` is echoed per candidate. Slippage is reported as 0.0
    only because these scores are computed from label-implied points with no
    slippage term at all -- that is an applied value, not an assumed one.
    """
    for cand in candidates:
        pnl = (cand.get("score") or {}).get("pnl") or (cand.get("score") or {}).get("pnl_raw")
        if isinstance(pnl, dict) and pnl.get("cost_per_trade") is not None:
            return float(pnl["cost_per_trade"]), 0.0
    return None, None


def _recorded_seeds(src: Path, candidates: list[dict[str, Any]]) -> dict[str, int]:
    """Seeds the source run genuinely recorded for its FITS.

    Returns ``{}`` when none exists. A SHAP sampler seed is not a fit seed and
    is deliberately not promoted into this field.
    """
    seeds: dict[str, int] = {}
    for cand in candidates:
        cid = str(cand.get("candidate_id"))
        params = (_read_json(src / "models" / cid / "finalize_manifest.json")
                  .get("spec_params") or {})
        for key in ("seed", "random_state", "random_seed"):
            if isinstance(params.get(key), int):
                seeds[cid] = int(params[key])
                break
    return seeds


def _resolve_cross_run_model(
    objective_id: str, base_run_ts: str, base: Mapping[str, Any],
) -> str | None:
    """The model id a meta's base has in ANOTHER canonical run.

    Two translations are needed, in order:

    1. **id**: a base run whose candidate was named after its algorithm
       ("attention_lstm") could not keep that as a model id, so it carries a
       configuration hash and records the historical name as
       ``source_candidate_id``.
    2. **label**: a meta's ``compose.json`` names its bases by its OWN labels
       ("lstm_15s"), which need not match the base run's candidate at all. The
       only correspondence both sides actually record is family plus native
       timeframe, so that is what is matched -- and only when it is unique
       within that run. Two models of the same family and timeframe in one run
       are genuinely ambiguous, and guessing between them would fabricate
       lineage.

    Returns ``None`` when the base run is not migrated or the match is not
    unique; the caller then records the pointer as unresolved rather than
    inventing one.
    """
    from src.common.run_layout import find_run_dir

    run = find_run_dir(objective_id, base_run_ts)
    if run is None:
        return None
    base_id = str(base.get("id") or "")
    docs = [(_read_json(mj)) for mj in sorted(run.glob("pass_*/model_*/model.json"))]
    for doc in docs:
        if base_id and base_id in (doc.get("model_id"), doc.get("source_candidate_id")):
            return str(doc.get("model_id"))
    family = str(base.get("family") or "").strip().lower()
    native_tf = str(base.get("native_tf") or "").strip()
    if not family:
        return None
    matches = [doc for doc in docs
               if str(doc.get("model_type") or "").strip().lower() == family]
    if len(matches) > 1 and native_tf:
        # native_tf DISAMBIGUATES; it does not qualify. A target run that never
        # recorded its timeframe must not be excluded by that absence.
        narrowed = [doc for doc in matches
                    if str((doc.get("feature_spec") or {}).get("native_tf") or "")
                    == native_tf]
        if len(narrowed) == 1:
            return str(narrowed[0].get("model_id"))
    if len(matches) == 1:
        return str(matches[0].get("model_id"))
    return None


def _candidates_from_bundles(src: Path) -> list[dict[str, Any]]:
    """Candidates inferred from ``models/<candidate>/`` when no list was recorded.

    A directory counts only when it holds a fitted artifact -- a weight file or
    an OOF -- so an empty or abandoned folder is not promoted into a model.
    ``family`` comes from that bundle's own ``finalize_manifest.json``, never
    guessed from the folder name.
    """
    models = src / "models"
    if not models.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for bundle in sorted(p for p in models.iterdir() if p.is_dir()):
        has_fit = (bundle / "oof.parquet").is_file() or any(
            (bundle / w).is_file() for w in MODEL_ARTIFACT_FILENAMES)
        if not has_fit:
            continue
        finalize = _read_json(bundle / "finalize_manifest.json")
        meta_doc = _read_json(bundle / "meta.json")
        out.append({
            "candidate_id": bundle.name,
            "family": finalize.get("family") or meta_doc.get("family") or "unknown",
            "status": "ok",
            "score": {},
            "params": finalize.get("spec_params") or {},
        })
    return out


def _pass_of(candidate_id: str) -> int:
    cid = str(candidate_id)
    return 2 if cid.startswith(_META_PREFIXES) else 1


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _label_contract(src: Path) -> tuple[str | None, str | None, dict[str, Any]]:
    """``(contract_id, dataset location, contract body)`` for this run.

    A ``nq-50pt-econ-lg-st`` run snapshots the label contract into the run
    directory. The BYTES stay in ``sources/labels/``, so the migrated run
    references the contract id and points at that directory -- it does not
    claim the snapshot as the dataset.
    """
    doc = _read_json(src / "label_contract.json")
    contract_id = doc.get("contract_id")
    if not contract_id:
        return None, None, {}
    contract = doc.get("contract") or {}
    resolution = contract.get("resolution") or contract.get("timeframe")
    if resolution:
        location = f"sources/labels/{resolution}/{contract_id}"
    else:
        location = doc.get("source_dir")
        if location and Path(str(location)).is_absolute():
            try:
                location = str(Path(str(location)).relative_to(Path.cwd()))
            except ValueError:
                location = None
    return str(contract_id), (str(location) if location else None), contract


def _metrics_payload(score: dict[str, Any], n_samples: int) -> dict[str, Any]:
    """``schemas.RunMetrics`` shape from a Track B candidate score.

    A transform, not a reinterpretation: the scalar fields the leaderboard
    already ranks on are lifted into ``metrics`` and everything else is kept
    verbatim under ``source_score`` so nothing is lost. The raw
    ``diagnostics_*.json`` stays in place as the legacy artifact.
    """
    flat: dict[str, float] = {}
    for key in ("precision_at_tau", "directional_f1", "f1_up", "f1_down",
                "precision_up", "precision_down", "net_expectancy", "hit_rate",
                "sharpe", "max_drawdown", "n_signals", "n_up", "n_down"):
        value = score.get(key)
        if isinstance(value, (int, float)) and value == value:  # drop NaN
            flat[key] = float(value)
    return {
        "n_samples": int(n_samples),
        "metrics": flat,
        "notes": "transformed from search_result.json candidate score; "
                 "raw diagnostics_<id>.json kept in the source run",
        "source_score": score,
    }


def _fold_metrics_payload(folds_doc: dict[str, Any], src: Path) -> dict[str, Any]:
    """``schemas.FoldMetrics`` records from ``folds.json`` plus ``folds/fold_NN.json``."""
    out: list[dict[str, Any]] = []
    for entry in folds_doc.get("folds") or []:
        fold_id = int(entry.get("fold", len(out)))
        train = entry.get("train") or {}
        oof = entry.get("oof") or {}
        detail = _read_json(src / "folds" / f"fold_{fold_id:02d}.json")
        record: dict[str, Any] = {
            "fold_id": str(fold_id),
            "train_start": str(train.get("start") or ""),
            "train_end": str(train.get("end") or ""),
            "test_start": str(oof.get("start") or ""),
            "test_end": str(oof.get("end") or ""),
            "n_test": int(detail.get("n") or 0),
            "metrics": {k: float(v) for k, v in detail.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)},
        }
        out.append(record)
    return {"folds": out, "split": folds_doc.get("split"),
            "n_folds": folds_doc.get("n_folds") or len(out)}


def _copy(src_file: Path, dest_file: Path, *, force: bool, link: bool = False) -> bool:
    """Place one file in the canonical tree. Returns whether it was written.

    ``link`` hard-links instead of copying, so the canonical layout is complete
    without a second copy of a 9 MB booster or a Keras weight file. Both paths
    then name the same inode: either can be read, and removing one leaves the
    other intact. Falls back to a copy when linking is not possible (a
    different filesystem, or a platform that refuses it).
    """
    if not src_file.is_file():
        return False
    if dest_file.exists():
        if not force:
            return False
        dest_file.unlink()
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src_file, dest_file)
            return True
        except OSError:
            pass  # cross-device or unsupported: a copy is still correct
    shutil.copy2(src_file, dest_file)
    return True


def _write_compact_oof(src_oof: Path, dest: Path, feature_columns: list[str],
                       *, force: bool) -> tuple[int, list[str]]:
    """Project the candidate OOF to the compact column set and copy it.

    A Track B OOF is already compact -- the walk-forward writer never appended
    feature columns -- so this is normally an identity projection. It is still
    applied, because "already compact" must be verified per run rather than
    assumed: the same call is what keeps a future wide OOF from being copied.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    columns = list(pq.read_schema(src_oof).names)
    keep = compact_oof_columns(columns, feature_columns)
    if dest.exists() and not force:
        return -1, list(pq.read_schema(dest).names)
    frame = pd.read_parquet(src_oof, columns=keep)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(dest)
    return len(frame), keep


#: Higher-timeframe context prefixes a meta model joins alongside the base
#: predictions, and the bar whose close makes each one knowable. A daily row is
#: recorded as a closed 1d bar rather than a session_close: bar_close needs only
#: the bar's duration, whereas session_close would require asserting a clock
#: time this run never recorded.
_CONTEXT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("h1_", "1h"),
    ("d1_", "1d"),
)

#: Calendar columns derived from the decision timestamp itself, so they are
#: knowable AT that timestamp and join with availability `timestamp`.
_CALENDAR_COLUMNS = frozenset({"tod_min", "session_min", "dow"})


def _meta_column_groups(
    feature_columns: list[str], base_ids: list[str],
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Split a meta model's fitted columns into upstream predictions and context.

    A ``tf_oof_meta`` model is NOT purely a function of its bases: alongside
    ``<base>__p_long`` style columns it joins higher-timeframe context and
    calendar columns of its own. Recording it as predictions-only would
    understate its inputs and hide a real causality claim, so each group is
    given its own source with its own availability.
    """
    per_base: dict[str, list[str]] = {bid: [] for bid in base_ids}
    leftover: list[str] = []
    for name in feature_columns:
        owner = next((bid for bid in base_ids if name.startswith(f"{bid}__")), None)
        if owner is not None:
            per_base[owner].append(name)
        else:
            leftover.append(name)

    context: list[dict[str, Any]] = []
    for prefix, duration in _CONTEXT_PREFIXES:
        cols = [n for n in leftover if n.startswith(prefix)]
        if cols:
            context.append({
                "semantic_type": "context",
                "logical_id": f"context_{prefix.rstrip('_')}",
                "columns": cols,
                "availability": AVAILABILITY_BAR_CLOSE,
                "bar_duration": duration,
                "time_column": "ts",
                "location": f"sources/features/timeframe={duration}",
                "store_id": f"native_tf:timeframe={duration}",
            })
    calendar = [n for n in leftover if n in _CALENDAR_COLUMNS]
    if calendar:
        context.append({
            "semantic_type": "context",
            "logical_id": "context_calendar",
            "columns": calendar,
            "availability": AVAILABILITY_TIMESTAMP,
            "time_column": "ts",
            "note": "derived from the decision timestamp itself",
        })
    classified = {c for src in context for c in src["columns"]}
    unclassified = [n for n in leftover if n not in classified]
    if unclassified and len(base_ids) == 1:
        # A single-base meta names columns derived from that base's own
        # prediction without a <base>__ prefix (primary_confidence, is_up).
        # With exactly one base the attribution is unambiguous, so they are
        # recorded as that base's prediction columns rather than as
        # unclassified context -- the upstream identity is the point.
        per_base[base_ids[0]].extend(unclassified)
        unclassified = []
    if unclassified:
        context.append({
            "semantic_type": "context",
            "logical_id": "context_unclassified",
            "columns": unclassified,
            "availability": AVAILABILITY_BAR_CLOSE,
            "bar_duration": "15s",
            "time_column": "ts",
            "note": "prefix not recognised by the migration; availability is the "
                    "decision grid's own bar, which is the most conservative "
                    "claim that is still true of a closed bar",
        })
    return per_base, context


def _feature_source(native_tf: str | None, feature_columns: list[str]) -> dict[str, Any]:
    """The native-timeframe feature store this fit joined, with its availability.

    A closed bar is knowable at its close, so the availability is
    ``bar_close`` with the bar's own duration. That is the causality claim the
    canonical join enforces; a Track B run had no other record of it.
    """
    tf = str(native_tf or "").strip()
    source: dict[str, Any] = {
        "semantic_type": "features",
        "logical_id": f"native_tf_{tf}" if tf else "native_tf",
        "columns": list(feature_columns),
        "availability": AVAILABILITY_BAR_CLOSE,
        "time_column": "ts",
    }
    if tf:
        source["location"] = f"sources/features/timeframe={tf}"
        source["bar_duration"] = tf
        source["store_id"] = f"native_tf:timeframe={tf}"
    else:
        # No declared timeframe: the source is named but its bar length is not
        # known, and bar_close without a duration is not a usable claim.
        source["availability"] = AVAILABILITY_TIMESTAMP
    return source


def emit_canonical_run(
    src: Path,
    *,
    objective_id: str | None = None,
    run_ts: str | None = None,
    dest_run_ts: str | None = None,
    target: str = "direction",
    model_id_override: str | None = None,
    force: bool = False,
    link_weights: bool = False,
    log: Any = None,
) -> Path | None:
    """Write ``src`` (a Track B run directory) as a canonical run.

    Returns the canonical run directory, or ``None`` when ``src`` holds no
    fitted candidate. ``link_weights`` hard-links the weight and bundle files
    instead of copying them -- what the controller uses, so emitting the
    canonical layout for a live run costs no extra bytes.
    """
    src = Path(src)
    say = (log.info if log is not None else
           (lambda fmt, *a: print(fmt % a if a else fmt, flush=True)))
    if not src.is_dir():
        say("canonical_run: source run not found: %s", src)
        return None
    record = read_run_record(src)          # the unified reader
    objective_id = str(objective_id or record.get("objective_id") or "")
    run_ts = str(dest_run_ts or run_ts or src.name)
    if not objective_id:
        say("canonical_run: %s names no objective_id", src)
        return None
    candidates = [c for c in (record.get("candidates") or [])
                  if str(c.get("status", "ok")) == "ok"]
    if not candidates:
        # Some fitted runs have no search_result.json at all -- a tf_oof_meta
        # report dir writes run.json only, and a parked run may have lost it.
        # The bundles themselves are the ground truth for "what did this fit",
        # so discover the candidates from models/ rather than declaring a run
        # with real weights on disk unfitted.
        candidates = _candidates_from_bundles(src)
        if candidates:
            say("canonical_run: %s records no candidate list; discovered %d from "
                "models/", src, len(candidates))
    if not candidates:
        say("canonical_run: no fitted candidates in %s", src)
        return None

    contract_id, contract_location, _contract = _label_contract(src)
    folds_doc = _read_json(src / "folds.json")
    fold_metrics = _fold_metrics_payload(folds_doc, src)
    validation = {
        "split": folds_doc.get("split") or "unknown",
        "n_folds": int(folds_doc.get("n_folds") or len(fold_metrics["folds"]) or 1),
        "fold_windows": folds_doc.get("folds") or [],
    }
    shap_owner = str(_read_json(src / "SHAP" / "shap_meta.json").get("candidate_id") or "")

    # ── evidence, recorded only where it is genuinely known ─────────────────
    # A Track B run's pnl block echoes the cost it actually applied. The
    # objective may DECLARE another (nq-50pt-* declares slippage_model
    # fixed_1pt, which src/backtest/engine.py never translated -- ADR-136), so
    # both are recorded and the divergence is named rather than flattened.
    applied_commission, applied_slippage = _applied_costs(candidates)
    costs = assumptions_record(
        _objective_document(objective_id),
        applied_commission_per_trade=applied_commission,
        applied_slippage_points_per_side=applied_slippage,
        pnl_unit="points",
    )
    # Seeds are NOT invented. This run recorded no fit seed anywhere; the only
    # seed on disk belongs to the SHAP sampler, which is a different thing.
    seeds = _recorded_seeds(src, candidates)
    code, code_why = _source_code_identity(src, candidates)
    evidence_notes: dict[str, Any] = {"migrated_from": str(src)}
    if code is not None:
        evidence_notes["code"] = (
            f"git_sha {code.git_sha} read from the source run's "
            "finalize_manifest.json, not from the migrating machine's HEAD. "
            "dirty=True because these runs performed no dirty-tree check."
            + (f" Note: {code_why}." if code_why else "")
        )
    else:
        evidence_notes["code"] = f"absent: {code_why}"
    if not seeds:
        evidence_notes["seeds"] = (
            "absent: the source run recorded no per-fit seed. Not reconstructed "
            "-- a fabricated seed would claim reproducibility this run cannot "
            "support. SHAP's own sampler seed is not the fit seed."
        )
    # config_snapshot counts as evidence only when the caller CLAIMS it is the
    # fully resolved configuration. A Track B run evaluates several candidates,
    # each with its own resolved hyperparameters, so no single run-level mapping
    # is the resolved config and the claim stays false -- which means the field
    # is reported missing whether the overlay YAMLs were captured or not. Both
    # cases are recorded, because they are different facts.
    if not (record.get("config_snapshot") or {}):
        evidence_notes["config_snapshot"] = (
            "absent: this run copied no spec.manual.yaml / topology.yaml into "
            "its directory, so no run-time configuration capture exists. The "
            "current config/objectives/ document is NOT substituted: it may "
            "have changed since the fit, and config_snapshot is part of the "
            "scientific fingerprint."
        )
    else:
        evidence_notes["config_snapshot"] = (
            "recorded from this run's own spec.manual.yaml / topology.yaml "
            "overlays, but NOT claimed resolved: a Track B run fits several "
            "candidates, each with its own resolved hyperparameters, so no "
            "single run-level mapping is the resolved configuration. Each "
            "model's resolved fit is in its own model.json."
        )
    if not (folds_doc.get("folds") or []):
        evidence_notes["validation"] = (
            "absent: this run recorded no folds.json, so its fold design was "
            "never persisted. Not reconstructed -- inventing a split would "
            "misstate how the run was validated, and validation is part of the "
            "scientific fingerprint."
        )
    evidence_notes["datasets"] = (
        "the label dataset is contract-identified only. No _build.json exists "
        "for it, and neither a part inventory nor the contract's own sha256 is "
        "build or content identity (docs/STORAGE_CONTRACT.md 5a)."
    )

    by_pass: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    for cand in candidates:
        by_pass[_pass_of(cand["candidate_id"])].append(cand)
    # candidate_id -> the model id actually used, filled as each pass is written
    # so a pass-2 compose.json base resolves to the right folder even when the
    # base's own id had to be reallocated.
    ids_by_candidate: dict[str, str] = {}

    for pass_n in (1, 2):
        entries = by_pass[pass_n]
        if not entries:
            continue
        models: list[dict[str, Any]] = []
        upstream: list[dict[str, Any]] = []
        if contract_id:
            upstream.append({
                "kind": "labels", "label_contract_id": contract_id,
                "logical_id": f"labels_{contract_id}",
                "path": contract_location or "",
            })

        for cand in entries:
            cid = str(cand["candidate_id"])
            bundle = src / "models" / cid
            finalize = _read_json(bundle / "finalize_manifest.json")
            schema = _read_json(bundle / "feature_schema.json")
            compose = _read_json(bundle / "compose.json")
            if not compose.get("bases"):
                # A primary_meta candidate records its lineage in meta.json
                # instead, with bases as plain id strings. Same fact, different
                # sidecar -- read both rather than treating one topology's file
                # as the only source of upstream identity.
                meta_doc = _read_json(bundle / "meta.json")
                if meta_doc.get("bases"):
                    compose = {
                        "bases": [{"id": b} if isinstance(b, str) else b
                                  for b in meta_doc["bases"]],
                        "feature_names": meta_doc.get("feature_names") or [],
                        "head": meta_doc.get("head"),
                    }
            family = str(cand.get("family") or finalize.get("family") or "unknown")
            # The historical candidate_id is normally the model id, which keeps
            # leaderboard rows and compose.json base pointers resolving. Some
            # single-candidate runs named the candidate after its algorithm
            # ("xgboost"), which is exactly what a model id must not be: it
            # cannot distinguish two fits of that family. Those fall back to the
            # configuration hash, and the historical name is preserved as
            # source_candidate_id so the old pointer is still resolvable.
            try:
                model_id = validate_model_id(model_id_override or cid, family)
                id_source = "candidate_id"
            except ValueError:
                model_id = allocate_model_id(
                    objective_id, run_ts, pass_n, family,
                    dict(finalize.get("spec_params") or cand.get("params") or {}),
                    feature_spec=_read_json(bundle / "feature_schema.json")
                    .get("feature_cols") or None,
                    target=target)
                id_source = "allocate_model_id"
                say("canonical_run: candidate_id %r is its own model type; using "
                    "the configuration hash %s instead (source_candidate_id kept)",
                    cid, model_id)
            dest = model_dir(objective_id, run_ts, pass_n, model_id)
            dest.mkdir(parents=True, exist_ok=True)

            # A meta model's fitted columns are its compose feature_names --
            # base prediction columns plus its own context -- NOT the serving
            # bundle's feature_cols, which lists what the BASES need in order
            # to be computed at inference.
            if compose.get("bases"):
                feature_columns = list(
                    compose.get("feature_names")
                    or schema.get("meta_feature_names") or [])
            else:
                feature_columns = list(schema.get("feature_cols") or [])
            hyper = dict(finalize.get("spec_params") or cand.get("params") or {})
            native_tf = finalize.get("native_tf") or compose.get("native_tf")

            # ── weights and the serving bundle ──────────────────────────────
            copied: list[str] = []
            # Hard-link only the large, genuinely immutable fitted artifacts.
            # A hard link on a small JSON saves nothing and makes the two paths
            # the same mutable file, so an edit to one silently changes the
            # other -- provenance and config files are copied.
            for name in MODEL_ARTIFACT_FILENAMES:
                if _copy(bundle / name, dest / name, force=force, link=link_weights):
                    copied.append(name)
            for name in _BUNDLE_FILES:
                link = link_weights and name.endswith(".pkl")
                if _copy(bundle / name, dest / name, force=force, link=link):
                    copied.append(name)
            # A run-level diagnostics copy exists for candidates whose bundle
            # has none; both copies are byte-identical on the runs checked.
            if "diagnostics.json" not in copied:
                _copy(src / f"diagnostics_{cid}.json", dest / "diagnostics.json",
                      force=force)

            # ── compact OOF ─────────────────────────────────────────────────
            src_oof = bundle / "oof.parquet"
            if not src_oof.is_file():
                src_oof = src / f"oof_{cid}.parquet"
            rows, oof_columns = (-1, [])
            if src_oof.is_file():
                rows, oof_columns = _write_compact_oof(
                    src_oof, dest / "oof_predictions.parquet", feature_columns,
                    force=force)

            # ── inputs: the join this fit actually performed ────────────────
            inputs: dict[str, Any] = {"join_keys": ["ts"], "filters": []}
            if contract_id:
                inputs["labels"] = {
                    "semantic_type": "outcomes",
                    "logical_id": f"labels_{contract_id}",
                    "contract_id": contract_id,
                    "location": contract_location,
                }
            bases = list(compose.get("bases") or [])
            if bases:
                base_ids = [ids_by_candidate.get(str(b.get("id")), str(b.get("id")))
                            for b in bases]
                per_base, context = _meta_column_groups(feature_columns, base_ids)
                predictions = []
                for base, base_id in zip(bases, base_ids):
                    # A meta layer's base may live in ANOTHER run: compose.json
                    # records that run_ts, and a cross-run dependency is the
                    # lineage fact, not an inconvenience. Same-run bases keep a
                    # run-relative location; cross-run bases are addressed by
                    # objective/run_ts so the pointer resolves either way.
                    base_run_ts = str(base.get("run_ts") or "").strip()
                    if base_run_ts and base_run_ts != run_ts:
                        resolved = _resolve_cross_run_model(
                            objective_id, base_run_ts, base)
                        upstream_run = f"{objective_id}/{base_run_ts}"
                        base_label = base_id
                        if resolved:
                            base_id = resolved
                            location = (f"{objective_id}/{base_run_ts}/pass_1/"
                                        f"model_{base_id}/oof_predictions.parquet")
                            unresolved = False
                        else:
                            # The upstream RUN is known; which model folder in it
                            # is not. Recorded as unresolved rather than pointed
                            # at a path that does not exist.
                            location = f"{objective_id}/{base_run_ts}"
                            unresolved = True
                    else:
                        base_label = base_id
                        location = f"pass_1/model_{base_id}/oof_predictions.parquet"
                        upstream_run = None
                        unresolved = False
                    ref = {
                        "semantic_type": "predictions",
                        "logical_id": f"pass_1_oof_{base_label}",
                        "location": location,
                        "model_id": base_id,
                        "pass": 1,
                        "build_id": base_id,
                        "columns": per_base.get(base_id) or [],
                        "availability": AVAILABILITY_TIMESTAMP,
                        "time_column": "ts",
                    }
                    if upstream_run:
                        ref["upstream_run_id"] = upstream_run
                    if base_label != base_id:
                        ref["upstream_candidate_label"] = base_label
                    if unresolved:
                        ref["upstream_resolved"] = False
                    predictions.append(ref)
                inputs["predictions"] = predictions
                if context:
                    inputs["context"] = context
                upstream.extend({
                    "kind": "oof_core", "pass": 1, "model_id": p["model_id"],
                    "logical_id": p["logical_id"], "build_id": p["build_id"],
                    "path": p["location"], "columns": "oof_core_columns",
                } for p in predictions)
            elif feature_columns:
                inputs["features"] = [_feature_source(native_tf, feature_columns)]

            # ── model.json ──────────────────────────────────────────────────
            write_model_json(
                dest,
                model_id=model_id, model_type=family, hyperparameters=hyper,
                target=target,
                feature_spec={"feature_selection": schema.get("feature_selection"),
                              "n_features": schema.get("n_features"),
                              "native_tf": native_tf,
                              "head": compose.get("head") or finalize.get("head")},
                feature_columns=feature_columns,
                inputs=inputs,
                oof_columns=oof_columns or None,
                label_contract_id=contract_id,
            )
            if id_source != "candidate_id" or cid != model_id:
                # Resolvable alias for anything still naming the old candidate.
                meta_path = dest / "model.json"
                doc = _read_json(meta_path)
                doc["source_candidate_id"] = cid
                doc["model_id_source"] = id_source
                write_json(meta_path, doc)
            # Provenance the source run genuinely recorded. git_sha is real
            # code identity; it is copied, never invented.
            extra: dict[str, Any] = {}
            for key in ("git_sha", "finalized_at", "version", "weights_source",
                        "staged_to", "search_run_dir", "search_run_ts"):
                if finalize.get(key) is not None:
                    extra[key] = finalize[key]
            if extra:
                write_json(dest / "migration_provenance.json", {
                    "source_run": str(src),
                    "source_candidate_id": cid,
                    # Per-model detail. The run-level commit goes on
                    # manifest.code; this keeps the per-bundle record, which is
                    # the only place a disagreement between fits is visible.
                    "finalize_manifest": extra,
                })

            # ── metrics / fold metrics ──────────────────────────────────────
            n_samples = rows if rows > 0 else int(
                sum(f.get("n_test") or 0 for f in fold_metrics["folds"]) or 1)
            write_json(dest / "metrics.json",
                       _metrics_payload(dict(cand.get("score") or {}), n_samples))
            write_json(dest / "fold_metrics.json", fold_metrics)

            # ── report/ ─────────────────────────────────────────────────────
            if shap_owner == cid:
                rep = report_dir(objective_id, run_ts, pass_n, model_id)
                for name in _SHAP_FILES:
                    _copy(src / "SHAP" / name, rep / name, force=force,
                          link=link_weights and name.endswith(".npz"))

            entry_models = {
                "model_id": model_id,
                "source_candidate_id": cid,
                "model_type": family,
                "model_json": f"pass_{pass_n}/model_{model_id}/model.json",
                "metrics": f"pass_{pass_n}/model_{model_id}/metrics.json",
                "fold_metrics": f"pass_{pass_n}/model_{model_id}/fold_metrics.json",
            }
            if oof_columns:
                entry_models["oof_predictions"] = (
                    f"pass_{pass_n}/model_{model_id}/oof_predictions.parquet")
            ids_by_candidate[cid] = model_id
            models.append(entry_models)
            say("canonical_run: pass %d model_%s: %s, %d features, "
                "oof %d cols, files %s", pass_n, model_id, family,
                len(feature_columns), len(oof_columns), sorted(copied))

        role = ("directional_screen" if pass_n == 1
                else "meta_confirmation")
        purpose = (
            "Score each decision row for direction from this timeframe's own "
            "features. One model per base timeframe; they do not read each "
            "other." if pass_n == 1 else
            "Confirm or block the previous pass's directional calls from those "
            "models' out-of-fold prediction columns, joined as-of the decision "
            "timestamp together with this pass's own higher-timeframe context. "
            "It does not reuse a base model's feature columns as its matrix."
        )
        merge_run_json(
            objective_id, run_ts,
            {
                "pass": pass_n, "role": role, "purpose": purpose,
                "label_contract_id": contract_id,
                "inputs": {"join_keys": ["ts"]},
                "upstream": upstream,
                "models": models,
            },
            config_snapshot=(record.get("config_snapshot") or None) if pass_n == 1 else None,
            # A MIGRATED run's snapshot is whatever it captured at fit time. The
            # overlay YAMLs are its real run-level inputs, but they are not the
            # post-override resolved document the live controller now records, so
            # this stays an unresolved claim for historical runs. New runs get
            # their resolution claim from src/search/run_manifest.py instead.
            config_snapshot_resolved=False,
            validation=validation if pass_n == 1 else None,
            seeds=seeds or None,
            execution_assumptions=costs,
            extra={"evidence": evidence_notes},
            code=code,
        )

    from src.common.run_layout import run_dir
    dest_run = run_dir(objective_id, run_ts)
    say("canonical_run: %s -> %s", src, dest_run)
    return dest_run
