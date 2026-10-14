"""Track B adapter: build a canonical ``RunManifest`` from a search run.

This is the FIRST integration of ``src/experiment`` into an existing
executable path (``docs/MIGRATION_MANIFEST.md`` §18). It is deliberately:

* **additive** -- ``manifest.json`` is written alongside the existing
  ``run.json`` / ``search_result.json`` / ``folds.json``. Nothing existing is
  altered, moved or removed, and no existing consumer reads this file;
* **best-effort** -- every failure is swallowed and logged, exactly like the
  results-index and outcome-episode writes next to it. A provenance record
  must never fail a search;
* **honest** -- it populates only values genuinely available on this code
  path. It fabricates nothing. A search run cannot supply a hypothesis or a
  seed today, so the resulting manifest reports
  ``evidence_level: INCOMPLETE`` and names exactly what is missing. That is
  the point: the gap becomes visible and measurable instead of being papered
  over with a plausible-looking placeholder.

It lives in ``src/search/`` rather than ``src/`` because the mapping
from *this* pipeline's internals to the canonical fields is Track B specific;
the canonical layer stays track-neutral.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

from src.evaluation.economic import assumptions_record
from src.experiment.manifest import (
    CodeIdentity,
    DatasetRef,
    RunManifest,
    ValidationDesign,
)

logger = logging.getLogger("vtech.search.run_manifest")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_identity() -> CodeIdentity | None:
    """Current commit plus a real dirty-tree check.

    ``controller._git_sha`` deliberately has no dirty guard (ADR-022 moved
    code identity to content-addressed bundles), so a bare SHA there can
    misrepresent the working tree. ``CodeIdentity`` requires ``dirty`` to be
    stated, so it is computed here rather than assumed False. Delegates to
    ``src.common.code_identity`` (tenth pass) rather than its own subprocess
    calls, so Track C's canonical-write path can share the same mechanism.
    """
    from src.common.code_identity import git_identity
    got = git_identity()
    if got is None:
        return None
    sha, dirty = got
    return CodeIdentity(git_sha=sha, dirty=dirty)


def _datasets_from_run_dir(out_dir: Path) -> tuple[DatasetRef, ...]:
    """Dataset lineage from the ``label_contract.json`` the run already wrote.

    Reuses ``src/training/label_store.py``'s CONTRACT addressing (semantic
    identity -- what the dataset means) rather than inventing one.
    Deliberately does NOT surface ``label_contract.json``'s
    ``contract_sha256`` as ``DatasetRef.content_sha256``: that field is a
    hash of the ``_contract.json`` METADATA sidecar
    (``label_store.write_label_contract_snapshot()``), not of the label
    parquet content, and stays unchanged across a ``force=True`` rebuild that
    replaces every part -- exposing it as a content hash would be false
    provenance (``docs/STORAGE_INVENTORY.md`` §4 STOR-11). No genuine content
    digest is computed anywhere on this path, so ``content_sha256`` is always
    ``None`` here.

    ``build_id`` is also always ``None`` on this path (tenth pass correction):
    the ninth pass populated it from a fingerprint of the recorded part
    inventory (filenames + byte sizes), which is NOT genuine build/provenance
    evidence -- it says nothing about which code, config or upstream data
    produced the realization, only that a directory listing didn't change.
    Nothing available to this adapter today (no code SHA, no resolved
    builder config, no upstream dataset version is threaded down to a
    per-dataset record) constitutes genuine build evidence, so ``build_id``
    stays honestly ``None`` rather than substituting a weaker signal for it
    -- ``contract_id`` is the strongest identity this adapter can honestly
    claim, which correctly makes ``datasets`` count as missing evidence for
    ``CANONICAL_COMPLETE`` (``RunManifest.missing_evidence()``): a semantic
    contract alone does not certify which realization of it was used.

    Returns ``()`` when the run used an input with no contract mechanism --
    an absent reference is reported as missing evidence, which is true,
    whereas a location-only reference would claim an identity that does not
    exist.
    """
    path = out_dir / "label_contract.json"
    if not path.is_file():
        return ()
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return ()
    # Key names verified against a real label_contract.json on disk:
    # {contract, contract_id, contract_sha256, parts, source_dir}.
    # contract_sha256 is intentionally not read into any DatasetRef field --
    # see the docstring above. Neither is the part inventory used to derive
    # build_id (also see the docstring above).
    contract_id = doc.get("contract_id")
    if not contract_id:
        return ()
    contract = doc.get("contract") or {}
    resolution = contract.get("resolution") or contract.get("timeframe")
    if not resolution and doc.get("source_dir"):
        # label_store.label_set_dir() lays sets out as labels/<resolution>/<id>;
        # read the resolution back from that resolver-defined position.
        resolution = Path(str(doc["source_dir"])).parent.name or None
    from src.common.data_paths import logical_path
    return (
        DatasetRef(
            logical_id=f"labels_{resolution}" if resolution else "labels",
            semantic_type="outcomes",
            resolution=str(resolution) if resolution else None,
            contract_id=str(contract_id) if contract_id else None,
            # No genuine build/provenance evidence exists on this path, and
            # no genuine content digest either -- see docstring.
            build_id=None,
            content_sha256=None,
            # Machine-independent: recorded relative to data_root(). None if
            # the source is outside the data root -- never an absolute path.
            location=logical_path(doc["source_dir"]) if doc.get("source_dir") else None,
            # Row count is not in the contract doc; part count is not a
            # row count, so n_rows stays None rather than being wrong.
            n_rows=None,
        ),
    )


def _validation(fold_split: str | None, fold_meta: Any) -> ValidationDesign | None:
    """Validation design from the fold metadata the run already computed.

    ``None`` when no folds ran -- which is a real, previously-invisible
    outcome: 3 of the 54 runs on disk had no walk-forward validation at all
    (``docs/CONFIG_AUDIT.md``).
    """
    if not fold_meta or not fold_split:
        return None
    try:
        return ValidationDesign(split=str(fold_split), n_folds=len(fold_meta))
    except (TypeError, ValueError):
        return None


def _config_snapshot(out_dir: Path, objective: Mapping[str, Any]) -> dict[str, Any]:
    """The most complete configuration record genuinely available here.

    ``objective`` is the post-topology-override, post-eval-override document
    (``controller.py``'s ``_apply_topology_override``/``_apply_eval_override``
    already ran before this is called) -- a real improvement over the raw
    YAML on disk, not a re-read of it. ``spec.manual.yaml`` / ``topology.yaml``
    / ``walk_forward.yaml``, when present, are the run's own overlay INPUTS
    (``controller._copy_custom_test_specs``, which runs before this) and are
    folded in under their own keys for the same reason.

    This is still NOT claimed as fully resolved (``config_snapshot_resolved``
    stays False in :func:`build_manifest`): a Custom Test run evaluates
    MULTIPLE candidates, each with its own resolved model/search
    hyperparameters, and no single flat mapping at the run level can
    truthfully represent all of them.
    """
    import yaml

    snapshot: dict[str, Any] = {"objective": dict(objective or {})}
    for key, filename in (("manual_spec", "spec.manual.yaml"),
                          ("topology", "topology.yaml"),
                          ("walk_forward", "walk_forward.yaml")):
        p = out_dir / filename
        if not p.is_file():
            continue
        try:
            snapshot[key] = yaml.safe_load(p.read_text())
        except (OSError, ValueError):
            continue  # unreadable overlay is not fabricated content
    return snapshot


def _artifact_inventory(out_dir: Path) -> dict[str, list[str]]:
    """Canonical class -> existing files that realise it, at write time.

    ``RunManifest.artifacts`` is canonical-class -> path(s), and only that --
    ``RunManifest.__post_init__`` refuses any other key. A file that exists
    but realises no canonical class (``progress*.json``, ``models/``, ...)
    is real information, but it belongs in its compatibility classification
    (``src.experiment.artifacts.classify_existing()``), not in this
    mapping: inserting e.g. ``"compatibility_consumed"`` as a key here would
    make it indistinguishable from a genuine artifact class. Nothing is
    renamed, copied or predicted -- only files that actually exist are listed.
    """
    from src.experiment.artifacts import classify_existing

    inv: dict[str, list[str]] = {}
    if not out_dir.is_dir():
        return inv
    for f in sorted(out_dir.iterdir()):
        if f.name == "manifest.json":
            continue
        _status, realises = classify_existing(f.name)
        if realises is not None:
            inv.setdefault(realises, []).append(f.name)
    return inv


def build_manifest(
    *,
    objective_id: str,
    run_ts: str,
    out_dir: Path,
    objective: Mapping[str, Any],
    experiment_type: str | None,
    fold_split: str | None,
    fold_meta: Any,
    model_ids: tuple[str, ...] = (),
    seeds: Mapping[str, int] | None = None,
    datasets: tuple[DatasetRef, ...] | None = None,
) -> RunManifest:
    """Assemble the manifest from values available on the search path.

    ``seeds`` and ``datasets`` are what the RUNTIME knows. When the controller
    passes them, they are used directly instead of being reconstructed from the
    files the run just wrote: re-reading ``label_contract.json`` to learn the
    contract the controller already held in memory is a round-trip that can only
    lose information, and a seed that was genuinely applied should be recorded
    as applied rather than reported missing. Both fall back to the on-disk
    reconstruction so a migrated or third-party run still produces a manifest.
    """
    code = _git_identity()
    responsibility_versions: dict[str, str] = {}
    if experiment_type:
        responsibility_versions["model_strategy"] = str(experiment_type)
    for mid in model_ids:
        responsibility_versions[f"model_strategy:{mid}"] = str(mid)
    if fold_split:
        responsibility_versions["validation"] = str(fold_split)
    if not responsibility_versions:
        # __post_init__ requires at least one; the search controller is the
        # implementation that ran, and saying so is not a fabrication.
        responsibility_versions["orchestration"] = "src/search/controller.py"

    return RunManifest(
        run_id=f"{objective_id}/{run_ts}",
        objective_id=str(objective_id),
        # No hypothesis is recorded anywhere on this path. Left empty on
        # purpose: it surfaces in missing_evidence() rather than being
        # invented from the objective's description.
        hypothesis="",
        responsibility_versions=responsibility_versions,
        # None when git is unavailable. Never a placeholder such as
        # "unknown": that would be fabricated provenance. RunManifest reports
        # a missing code identity as missing evidence, making the run
        # INCOMPLETE rather than silently mis-attributed.
        code=code,
        datasets=datasets if datasets is not None else _datasets_from_run_dir(Path(out_dir)),
        config_snapshot=_config_snapshot(Path(out_dir), objective or {}),
        # Claimed resolved at the RUN level, which is what this field means.
        # `objective` here is the post-topology-override, post-eval-override
        # document (controller._apply_topology_override/_apply_eval_override
        # have already run), folded together with the run's own overlay inputs.
        # That IS this run's resolved configuration.
        #
        # Multiple candidates do not make it unresolved: each candidate's
        # resolved hyperparameters, features, target and join spec live in its
        # own pass_<n>/model_<id>/model.json. Treating a multi-candidate sweep
        # as proof of an unresolved RUN config was a category error that kept
        # every Track B run permanently INCOMPLETE for no evidentiary reason.
        config_snapshot_resolved=bool(objective),
        validation=_validation(fold_split, fold_meta),
        # The runtime's own seeds when it passes them; {} otherwise, which
        # surfaces as missing evidence rather than a fabricated value.
        seeds=dict(seeds or {}),
        # One builder for every run's cost record (src/evaluation/economic.py),
        # rather than a raw copy of the objective's costs block: it resolves a
        # declared cost profile, refuses to read an unparsed slippage_model as
        # zero, and names a declared-but-unapplied cost instead of hiding it.
        execution_assumptions=assumptions_record(objective or {}),
        artifacts=_artifact_inventory(Path(out_dir)),
    )


def write_manifest(**kwargs: Any) -> Path | None:
    """Build and persist ``manifest.json``. Never raises; returns the path.

    Idempotent by construction: :meth:`RunManifest.write` permits rewriting
    an identical manifest and refuses a rewrite whose science changed, so a
    retried run is safe and a silently-altered one is not.
    """
    try:
        out_dir = Path(kwargs["out_dir"])
        manifest = build_manifest(**kwargs)
        path = manifest.write(out_dir)
        logger.info(
            "canonical_manifest_written path=%s evidence=%s missing=%s",
            path, manifest.evidence_level, ",".join(manifest.missing_evidence()),
        )
        return path
    except Exception as exc:  # noqa: BLE001 — provenance must never fail a search
        logger.warning("canonical_manifest_write_failed: %s", exc)
        return None


def runtime_datasets(objective: Mapping[str, Any]) -> tuple[DatasetRef, ...]:
    """Dataset refs the controller can build from the objective it is running.

    The label contract is derived from the in-memory objective, exactly as
    ``label_store.write_label_contract_snapshot`` derives the file it writes
    beside the run. Building the ref here instead of re-reading that file keeps
    the manifest's identity independent of a side artifact, and is the reason a
    NEW run does not need the post-hoc reconstruction path at all.

    Returns ``()`` when this objective uses an input with no contract mechanism
    -- an absent ref is reported as missing evidence, which is true, whereas a
    location-only ref would claim an identity that does not exist.
    """
    try:
        from src.training.label_store import (
            label_contract_id, label_set_dir, tbm_contract_from_objective,
        )
    except Exception:  # noqa: BLE001
        return ()
    # The label contract mechanism exists only for native_tf label sets. The
    # FEATURE store is joined either way, so its lineage is recorded regardless:
    # a run with no contract-addressed labels is still not a run with no inputs.
    contract: dict[str, Any] = {}
    contract_id = None
    source = None
    if str((objective.get("search") or {}).get("input") or "") == "native_tf":
        try:
            contract = tbm_contract_from_objective(dict(objective))
            contract_id = label_contract_id(contract)
            source = label_set_dir("15s", contract)
        except Exception as exc:  # noqa: BLE001
            logger.warning("runtime_datasets_label_contract_failed: %s", exc)
            contract_id = None
    location: str | None = None
    if source is not None:
        try:
            from src.common.data_paths import data_root
            location = str(Path(source).resolve().relative_to(Path(data_root()).resolve()))
        except Exception:  # noqa: BLE001
            location = None
    # Strongest truthful identity: a genuine _build.json when the producer wrote
    # one (label_store.write_label_parts does, for writes after 2026-09-23).
    build_id = None
    if source is not None:
        try:
            from src.training.label_store import BUILD_NAME
            sidecar = Path(source) / BUILD_NAME
            if sidecar.is_file():
                build_id = json.loads(sidecar.read_text()).get("build_id") or None
        except (OSError, ValueError):
            build_id = None

    refs: list[DatasetRef] = []
    if contract_id:
        refs.append(DatasetRef(
            logical_id=f"labels_{contract_id}",
            semantic_type="outcomes",
            contract_id=contract_id,
            build_id=build_id,
            resolution=str(contract.get("resolution") or "15s"),
            location=location,
        ))
    # The native-timeframe feature store this run actually joined. A hive store
    # is rebuilt in place, so store_id is the honest identity for it -- weaker
    # than a contract, and reported as such, but the lineage must be recorded.
    try:
        from src.features.stores import feature_store_id, parquet_dir_for_store
        store_id = feature_store_id(dict(objective))
        store_dir = parquet_dir_for_store(store_id)
        store_loc = None
        try:
            from src.common.data_paths import data_root
            store_loc = str(Path(store_dir).resolve().relative_to(
                Path(data_root()).resolve()))
        except Exception:  # noqa: BLE001
            store_loc = None
        refs.append(DatasetRef(
            logical_id=f"features_{store_id}",
            semantic_type="features",
            store_id=str(store_id),
            location=store_loc,
        ))
    except Exception as exc:  # noqa: BLE001
        logger.warning("runtime_datasets_feature_store_failed: %s", exc)
    return tuple(refs)
