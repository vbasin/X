"""RunManifest: the canonical identity of one experiment.

See ``docs/EXPERIMENTS.md`` §1 for the field-by-field rationale.

Derived from existing behaviour, not invented
---------------------------------------------
The field set below was derived by auditing the seven identity mechanisms that
already exist in this repository (``docs/FUNCTIONAL_COVERAGE.md`` §11) and
keeping what works:

  * ``objective_id`` is first-class because the objective concept is
    demonstrably durable and topology-independent: it keys 9 objective docs,
    the run directories (``research/runs/<objective_id>/``; historical
    runs also lived under ``research/custom_test/``), the
    leaderboards (``models/leaders/<objective_id>/``), the model registry and
    the frozen evaluation contracts. It survives every change of model
    topology, which is the test for first-class status.
  * ``DatasetRef`` mirrors ``src/training/label_store.py``'s contract
    addressing (sha256 of a *normalised semantic contract*, self-describing
    sidecar, part inventory) -- the strongest artifact in the repo. It is
    reused rather than re-invented.
  * ``CodeIdentity`` carries ``dirty`` because ``src/search/controller.py``'s
    ``_git_sha`` is best-effort with no dirty-tree guard (ADR-022 removed it),
    so a recorded SHA can silently misrepresent the tree.
  * ``ValidationDesign`` exists because ``folds.json`` already persists
    ``split``/``n_folds``, and because two engines with incompatible config
    contracts coexist (``docs/CONFIG_AUDIT.md``) -- which engine ran is part
    of a run's scientific identity.
  * ``schema_version`` exists because ``run.json`` has *already* drifted into
    two incompatible shapes on disk with no way to tell them apart.

Scientific immutability
-----------------------
A ``run_id`` is a claim about a specific experiment. Rewriting the same
``run_id`` with changed scientific content silently rewrites history, so
:meth:`RunManifest.write` refuses it: an idempotent rewrite succeeds, a
semantically changed one raises. See :meth:`scientific_fingerprint`.

This module IS the run identity. Track B's ``src/search/controller.py`` writes
it for every Custom Test search run via ``src/search/run_manifest.py``.
Objective runs write it as the only run-level file: ``passes`` holds what
``run.json`` used to hold, ``config_snapshot`` holds the resolved objective,
and ``validation`` holds the fold design. ``run.json``, ``contract.json`` and
``folds.json`` remain readable for runs that already have them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 4
# v1 -> v2: ``research_objective_id`` renamed to the repository's existing
# term ``objective_id``; ``DatasetRef`` gained semantic_type/store_id/
# resolution; ``artifacts`` became a canonical-class -> relative-paths map.
# v2 -> v3 (tenth pass, Storage Implementation Tranche 1): ``DatasetRef.build_id``
# changed MEANING, not just presence. Under v2 it was (incorrectly) populated from
# a fingerprint of the output part inventory (filenames + byte sizes) -- not
# genuine build/provenance evidence, since it says nothing about which code,
# config or upstream data produced the realization. Under v3 it is populated only
# from genuine code/config/upstream-version evidence (or left ``None``), and
# ``missing_evidence()``'s ``datasets`` check no longer treats a contract-only
# ``DatasetRef`` as sufficient realization evidence (only ``build`` or ``content``
# now satisfy it -- see ``identity_strength``). v1 and v2 files remain readable
# and are normalised on read (see ``read``); a v2 file's ``build_id`` value, if
# any, is read back as-is (old semantics, honestly a weaker claim than a v3
# ``build_id`` would be, but not rewritten or discarded).
# v3 -> v4: ``passes`` records each pass's role, input spec, and model
# pointers (what ``run.json`` held). ``validation`` gained the objective-run
# fold-design fields. Older files omit ``passes`` and still read.


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


#: Semantic types of an experiment input (docs/STORAGE_CONTRACT.md §2.6).
DATASET_SEMANTIC_TYPES = (
    "raw_source",      # vendor/normalised market data
    "features",        # decision-time state representation
    "outcomes",        # labels / outcome datasets
    "predictions",      # a trained model's own OOF/scored output, consumed
                        # as input by a downstream stage (tenth pass) -- e.g.
                        # build_target_matrix.py's --scores. NOT an outcome:
                        # an outcome is ground truth: what actually happened;
                        # a prediction is a model's claim about it. Conflating
                        # the two would hide prediction lineage inside a
                        # dataset that is not the model realization it came
                        # from -- see docs/STORAGE_CONTRACT.md §19.
    "context",         # reference/context data (e.g. session levels)
    "objective",       # config/objectives/<objective_id>.yaml
)


@dataclass(frozen=True)
class DatasetRef:
    """One experiment input, identified by a recorded identity mechanism -- never by path alone.

    This is the manifest's RECORD of an input. It does not resolve anything:
    resolution stays with the existing registries (``src/common/data_paths``,
    ``src/features/stores.py``, ``src/training/label_store.py``).

    Criteria (docs/STORAGE_CONTRACT.md §2.6):

    1. ``semantic_type`` -- what the input is (``DATASET_SEMANTIC_TYPES``).
    2. ``logical_id`` -- stable name, independent of any machine path. Not a
       version-suffixed directory name (``move_matrix_v4`` is a location).
    3. ``location`` -- data-root-RELATIVE path as produced by
       ``data_paths.logical_path()``. An absolute machine path is refused.
       Traceability only: stripped from the scientific fingerprint.
    4. Version identity, strongest available, at least one of the next four
       required. These are three genuinely DIFFERENT questions, not degrees
       of the same one -- conflating them is a documented failure mode (a
       hash of contract METADATA was once recorded as ``content_sha256``;
       see ``docs/STORAGE_INVENTORY.md`` §4 STOR-11):

       * ``contract_id`` -- SEMANTIC identity: what the dataset is supposed
         to MEAN (label rule, horizon, stop/target semantics, resolution).
         ``label_store.label_contract_id()`` hashes the normalized semantic
         contract dict, never the dataset's bytes. Two realizations with the
         same ``contract_id`` agree on meaning; they do not thereby agree on
         content -- ``label_store.write_label_parts(..., force=True)`` can
         replace every part beneath an unchanged ``contract_id`` directory,
         so a directory keyed this way is idempotent-by-default, **not
         immutable-by-content**.
       * ``build_id`` -- REALIZATION identity: which deterministic build
         (code revision, effective builder config, upstream input version)
         produced this specific on-disk snapshot. Distinguishes two runs
         that share a ``contract_id`` but were built at different times
         against different upstream data. **Not a content hash** -- it does
         not verify the bytes, only that the same recorded build-time
         evidence was reused. **Must be derived from genuine build-process
         evidence -- never from the OUTPUT's own filenames or byte sizes.**
         A fingerprint of the part inventory (which files, how large) was
         tried in the ninth pass and retired in the tenth: it says nothing
         about which code, config or upstream data produced the bytes, only
         that a directory listing didn't change -- a much weaker and
         differently-shaped claim than "this build". If no genuine
         code/config/upstream evidence exists, leave ``build_id=None``
         rather than substitute one.
       * ``content_sha256`` -- CONTENT identity: a digest of the actually
         materialized dataset bytes. The strongest claim; also the one this
         codebase can least often back up, since hashing an existing
         multi-GB dataset is deliberately not done as a side effect of
         writing a manifest. Only set this field from a genuine content
         digest -- never from a hash of a metadata sidecar, a contract, or
         a build spec, even to make evidence look stronger.
       * ``store_id`` (a ``stores.py`` registry id) is *logical* identity
         only -- it names WHICH dataset, not WHICH version: several stores
         are rebuilt in place (``native_tf.py:161-162``), so the same
         ``store_id`` can silently refer to different content over time.

       Ranked by ``identity_strength``: ``content`` > ``build`` > ``contract``
       > ``store``. A store-only *or contract-only* ``DatasetRef`` is still
       truthful and constructible, but ``RunManifest.missing_evidence()``
       will not count either as satisfying ``datasets`` (tenth pass): a
       ``contract_id`` states what a dataset is supposed to MEAN, not WHICH
       realization was used, and the same contract_id can silently name
       different bytes over time (``write_label_parts(..., force=True)``) --
       so it is real, useful evidence, just not strong enough on its own to
       certify a controlled comparison. Only ``build`` (realization) or
       ``content`` identity satisfies ``datasets`` for
       ``CANONICAL_COMPLETE``. If no identity mechanism exists at all, omit
       the ``DatasetRef`` entirely -- an absent input is reported as missing
       evidence, which is true.
    5. ``resolution`` -- timeframe/resolution (``"15s"``, ``"5s"``) where it
       materially changes the dataset.
    """

    logical_id: str
    semantic_type: str | None = None
    contract_id: str | None = None
    build_id: str | None = None
    content_sha256: str | None = None
    store_id: str | None = None
    resolution: str | None = None
    location: str | None = None
    n_rows: int | None = None

    def __post_init__(self) -> None:
        if not self.logical_id:
            raise ValueError("DatasetRef.logical_id must be non-empty")
        if self.semantic_type is not None and self.semantic_type not in DATASET_SEMANTIC_TYPES:
            raise ValueError(
                f"DatasetRef.semantic_type={self.semantic_type!r} not in "
                f"{DATASET_SEMANTIC_TYPES}"
            )
        if self.location is not None and Path(self.location).is_absolute():
            raise ValueError(
                f"DatasetRef.location={self.location!r} is an absolute machine "
                f"path. Record the data-root-relative form from "
                f"src.common.data_paths.logical_path()."
            )
        if not (self.content_sha256 or self.build_id or self.contract_id or self.store_id):
            raise ValueError(
                f"DatasetRef({self.logical_id!r}) has no scientific identity: "
                f"one of content_sha256, build_id, contract_id or store_id is "
                f"required. A physical location is traceability, not identity. "
                f"If none can be produced, omit the DatasetRef entirely: an "
                f"absent reference is reported as missing evidence, whereas a "
                f"location-only one silently claims an identity it does not have."
            )

    @property
    def identity_strength(self) -> str:
        """``content`` > ``build`` > ``contract`` > ``store`` -- the strongest recorded."""
        if self.content_sha256:
            return "content"
        if self.build_id:
            return "build"
        if self.contract_id:
            return "contract"
        return "store"


@dataclass(frozen=True)
class CodeIdentity:
    """Which code produced this run.

    Attributes:
        git_sha: Commit SHA at run time.
        dirty: Whether the working tree had uncommitted changes. **Required
            and not defaulted to False**: the repo's existing ``_git_sha``
            helper has no dirty check, so an unqualified SHA is an
            unverifiable claim. Recording ``dirty=True`` is honest; omitting
            the question is not.
        bundle_sha256: Content hash of the dispatched code bundle
            (``src/common/code_bundle.py``) when the run executed remotely.
    """

    git_sha: str
    dirty: bool
    bundle_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.git_sha:
            raise ValueError("CodeIdentity.git_sha must be non-empty")


@dataclass(frozen=True)
class ValidationDesign:
    """How the run was validated. Part of scientific identity.

    ``split`` matters as much as ``n_folds``: ``expanding_equal_count`` and
    ``expanding_calendar`` are different engines with different leakage
    profiles, and only the former reads ``eval.walk_forward_folds``
    (``docs/CONFIG_AUDIT.md``).
    """

    split: str
    n_folds: int
    purge_bars: int | None = None
    embargo_bars: int | None = None
    weighting: str | None = None
    test_months: int | None = None
    min_train_months: int | None = None
    purge_sessions: int | None = None
    #: Realized windows and hold-out dates are the design applied to one
    #: population. They are stripped from the scientific fingerprint so
    #: recording them after the design itself does not look like a new run.
    hold_out_dates: tuple[str, ...] | None = None
    fold_windows: tuple[Any, ...] | None = None

    def __post_init__(self) -> None:
        if not self.split:
            raise ValueError("ValidationDesign.split must be non-empty")
        if self.n_folds < 1:
            raise ValueError("ValidationDesign.n_folds must be >= 1")
        if isinstance(self.hold_out_dates, list):
            object.__setattr__(self, "hold_out_dates", tuple(self.hold_out_dates))
        if isinstance(self.fold_windows, list):
            object.__setattr__(self, "fold_windows", tuple(self.fold_windows))


@dataclass(frozen=True)
class RunManifest:
    """Identity + provenance of one experiment run.

    Attributes:
        run_id: Globally unique LOGICAL identifier for one execution. For
            the Track B convention ``run_id = "<objective_id>/<run_ts>"``.
            ``run_ts`` alone is not unique (it is only the physical
            directory leaf and collides across objectives); a
            version-suffixed name like ``move_matrix_v4`` is not an id.
            ``run_id`` is identity; the directory
            ``research/<workflow>/<objective_id>/<run_ts>/`` is location.
            Related, not interchangeable -- no third term (e.g. "run_key")
            exists for either.
        objective_id: Stable, topology-independent identifier for the
            trading/research problem a run serves -- the repository's
            existing term, matching ``config/objectives/<objective_id>.yaml``.
            Fields tagged ``x-objective-role: identity`` in
            ``objective.schema.json`` define what the id MEANS.
        hypothesis: One sentence stating what this run tests. Evidence,
            not a construction gate (see ``REQUIRED_FOR_CANONICAL_COMPLETE``).
        responsibility_versions: Canonical responsibility -> implementation
            identifier for every responsibility this run exercised.
        code: :class:`CodeIdentity`.
        datasets: Ordered :class:`DatasetRef` inputs.
        config_snapshot: The configuration actually used, post-defaults --
            not a reference to ``config.yaml``, because declared and
            effective values are confirmed to diverge (``docs/CONFIG_AUDIT.md``).
            A non-empty mapping here is NOT by itself proof of full
            resolution: see ``config_snapshot_resolved``.
        config_snapshot_resolved: Whether ``config_snapshot`` genuinely IS
            the fully resolved **run-level** configuration (post-default,
            post-override), rather than a raw or partial source document.

            Run-level and model-level configuration are different objects and
            are resolved separately:

            * ``manifest.config_snapshot`` -- the resolved RUN: objective,
              runtime overrides, validation design, shared dataset definitions,
              search/topology configuration, execution assumptions, shared
              seeds.
            * ``pass_<n>/model_<id>/model.json`` -- the resolved MODEL: type,
              that candidate's hyperparameters, its exact feature columns, its
              target, its join spec, its upstream model dependencies.

            A multi-candidate run therefore CAN have a fully resolved run-level
            config: differing candidate hyperparameters live in each
            ``model.json`` and say nothing about whether the run's own
            configuration was resolved. Treating "several candidates" as proof
            of an unresolved run config was a category error that made every
            Track B run permanently INCOMPLETE.

            Defaults to ``False`` -- a populated dict proves nothing about
            its own completeness, and a caller that cannot support the claim
            must not set this. Excluded from the scientific fingerprint (it
            describes evidence QUALITY, not the experiment itself) but
            consulted by ``missing_evidence()``: an unresolved snapshot
            makes ``config_snapshot`` count as missing evidence even when
            non-empty.
        validation: :class:`ValidationDesign`, when the run validated anything.
        experiment_family: Optional grouping for runs exploring one question.
        parent_run_id: Run this one is derived FROM (e.g. a meta layer's base
            run, as ``compose.json`` already records).
        baseline_run_id: Run this one should be COMPARED AGAINST. Distinct
            from ``parent_run_id`` -- derivation and comparison are different
            relationships, and nothing in the repo currently records the
            latter.
        seeds: Seeds actually applied, per library.
        execution_assumptions: Cost/fill assumptions, shaped like
            ``src/evaluation/economic.py::EconomicAssumptions``.
            Must be recorded even when zero -- an absent cost record is
            indistinguishable from a deliberate zero-cost choice, which is
            the confirmed ``slippage_model`` failure mode.
        decision_policy / calibration: Resolved descriptions when applicable.
        artifacts: Canonical artifact class -> run-dir-relative paths that
            realise it (``src/experiment/artifacts.py``). A mapping
            rather than a list so an existing file (e.g. ``oof_xgboost.parquet``)
            can realise a canonical class without being renamed or copied.
        schema_version / created_at / extra: Bookkeeping. ``extra`` is for
            genuinely run-specific annotation only; it is excluded from the
            scientific fingerprint and must never carry identity.
    """

    run_id: str
    objective_id: str
    hypothesis: str
    responsibility_versions: Mapping[str, str]
    code: CodeIdentity | None
    datasets: Sequence[DatasetRef]
    config_snapshot: Mapping[str, Any]
    config_snapshot_resolved: bool = False
    validation: ValidationDesign | None = None
    experiment_family: str | None = None
    parent_run_id: str | None = None
    baseline_run_id: str | None = None
    seeds: Mapping[str, int] = field(default_factory=dict)
    execution_assumptions: Mapping[str, Any] = field(default_factory=dict)
    decision_policy: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Sequence[str]] = field(default_factory=dict)
    #: Pass roles, input specs, and model pointers. Accumulates as each pass
    #: is trained, so it is not part of the scientific fingerprint. The
    #: resolved objective and the fold design are.
    passes: tuple[Any, ...] = ()
    schema_version: int = SCHEMA_VERSION
    created_at: str = field(default_factory=_utc_now_iso)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        if not self.objective_id:
            raise ValueError(
                "objective_id must be non-empty -- a run that does not "
                "state which research objective it serves cannot be placed, "
                "compared or promoted"
            )
        if not self.responsibility_versions:
            raise ValueError(
                "responsibility_versions must record at least one "
                "responsibility this run exercised"
            )
        if self.artifacts:
            from src.experiment.artifacts import BY_CLASS
            unknown = set(self.artifacts) - set(BY_CLASS)
            if unknown:
                raise ValueError(
                    f"artifacts keys {sorted(unknown)} are not canonical artifact "
                    f"classes (known: {sorted(BY_CLASS)}). ``artifacts`` maps "
                    f"canonical class -> realising file(s); a file that realises "
                    f"no canonical class (e.g. progress*.json, models/) is "
                    f"classified by src.experiment.artifacts."
                    f"classify_existing() and must not be inserted here under "
                    f"its compatibility status."
                )

    # --- evidence completeness ----------------------------------------------

    #: Provenance a run must carry to be CANONICAL_COMPLETE -- i.e. to be
    #: usable as evidence in a controlled comparison or a promotion decision.
    #:
    #: These are NOT construction-time requirements. Existing Track B/C
    #: workflows genuinely cannot supply several of them today (no run records
    #: a seed; most record no cost assumptions). Refusing to construct a
    #: manifest for such a run would mean either fabricating the missing
    #: values or recording nothing at all, and both are worse than recording
    #: the truth: this run's evidence is incomplete, and here is exactly what
    #: is missing.
    #: ``hypothesis`` is deliberately NOT here. It is comparison/context prose,
    #: excluded from ``scientific_fingerprint`` for exactly that reason, so a
    #: run's reproducibility cannot depend on it: every field below is something
    #: you need in order to RE-RUN the experiment and get the same answer, and a
    #: sentence of English is not. Requiring it made every run INCOMPLETE for a
    #: reason that has no bearing on whether the evidence is sound, which
    #: devalued the signal the level exists to give. It remains a first-class
    #: field and is still reported as absent by ``missing_evidence`` callers that
    #: ask for it.
    REQUIRED_FOR_CANONICAL_COMPLETE = (
        "objective_id",   # objective identity
        "responsibility_versions", # implementation identity
        "code",                    # code identity
        "datasets",                # dataset identity/lineage
        "config_snapshot",         # resolved config
        "seeds",                   # reproducibility
        "execution_assumptions",   # cost/fill assumptions
        "validation",              # validation design
    )

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if hasattr(value, "__len__"):
            return len(value) == 0
        return False

    def missing_evidence(self) -> tuple[str, ...]:
        """Required provenance this run did not record. Empty == complete."""
        missing = []
        for name in self.REQUIRED_FOR_CANONICAL_COMPLETE:
            value = getattr(self, name)
            if name == "config_snapshot":
                # A non-empty mapping proves a config was RECORDED, not that
                # it was fully RESOLVED -- e.g. a raw objective dict handed
                # straight through satisfies "non-empty" while omitting
                # every per-candidate override. config_snapshot_resolved is
                # the caller's explicit, unfabricated claim of the latter.
                if self._is_empty(value) or not self.config_snapshot_resolved:
                    missing.append(name)
                continue
            if name == "datasets":
                # A DatasetRef backed only by store_id names a mutable store
                # (several are rebuilt in place, e.g. native_tf.py:161-162)
                # -- it identifies WHAT the dataset is, not WHICH version was
                # used. A DatasetRef backed only by contract_id has the same
                # gap one tier up: it states WHAT the dataset is supposed to
                # MEAN, not WHICH realization was used -- the same
                # contract_id can silently name different bytes over time
                # (label_store.write_label_parts(..., force=True); tenth
                # pass, docs/STORAGE_CONTRACT.md §5a). Both are real,
                # truthful evidence (identity_strength "store"/"contract"),
                # just not strong enough to certify a controlled comparison
                # -- only "build" or "content" identity satisfies this.
                if self._is_empty(value) or any(
                    d.identity_strength in ("store", "contract") for d in value
                ):
                    missing.append(name)
                continue
            if self._is_empty(value):
                missing.append(name)
        return tuple(missing)

    @property
    def evidence_level(self) -> str:
        """``CANONICAL_COMPLETE`` or ``INCOMPLETE``.

        An INCOMPLETE manifest is a valid, useful record -- it is simply not
        admissible on its own as evidence for a controlled comparison. The
        distinction is stated rather than implied so that a run cannot be
        silently treated as better-evidenced than it is.
        """
        return "CANONICAL_COMPLETE" if not self.missing_evidence() else "INCOMPLETE"

    # --- scientific identity -------------------------------------------------

    #: Fields that carry scientific meaning. Changing any of them changes what
    #: the run *is*, so they determine the fingerprint. Everything else
    #: (created_at, artifacts, extra) is bookkeeping and may be rewritten.
    SCIENTIFIC_FIELDS = (
        "run_id", "objective_id",
        "responsibility_versions", "code", "datasets", "config_snapshot",
        "validation", "parent_run_id",
        "seeds", "execution_assumptions", "decision_policy", "calibration",
    )

    #: Fields deliberately EXCLUDED from the fingerprint, with the reason.
    #: Recorded here because three of them were previously included, which
    #: made a prose edit or a re-labelling trip the immutability guard as
    #: though the science had changed.
    #:
    #:   hypothesis        COMPARISON CONTEXT. Two runs with identical setup
    #:                     and differently-worded hypotheses ran the same
    #:                     experiment. Fixing a typo must not fork identity.
    #:   experiment_family COMPARISON CONTEXT. A grouping label applied to a
    #:                     run; regrouping does not change what was run.
    #:   baseline_run_id   COMPARISON CONTEXT. What you choose to compare a
    #:                     run against does not change what the run did.
    #:   created_at        BOOKKEEPING.
    #:   artifacts         BOOKKEEPING (an inventory of outputs, not inputs).
    #:   extra             BOOKKEEPING. Excluded by design, which is also why
    #:                     it must never carry identity.
    #:   passes            OUTPUT INVENTORY. Roles and input specs are recorded
    #:                     here, and model pointers are added as each pass
    #:                     finishes. Changing the objective snapshot or the
    #:                     fold design is still a new run. A pass's resolved
    #:                     feature list lives in that model's ``model.json``.
    #:
    #: ``parent_run_id`` is NOT here: derivation is real lineage. A meta layer
    #: trained on a different base run IS a different experiment.
    NON_FINGERPRINTED_FIELDS = (
        "hypothesis", "experiment_family", "baseline_run_id",
        "created_at", "artifacts", "extra", "schema_version",
        "config_snapshot_resolved", "passes",
    )

    #: Leaf keys stripped from the fingerprint payload wherever they occur.
    #: ``DatasetRef.location`` documents itself as "for human traceability
    #: only. Never the identity" -- so it must not enter the hash that DEFINES
    #: identity. Leaving it in meant a dataset moved between runs (a store
    #: migration, or a write-to-temp-then-swap) changed the fingerprint without
    #: changing the science, and ``write()`` would then refuse a legitimate
    #: idempotent retry -- the exact failure it exists to prevent.
    #: ``n_rows`` is deliberately NOT stripped: a different row count is a
    #: different dataset.
    NON_SCIENTIFIC_LEAVES = ("location",)

    @classmethod
    def _strip_non_scientific(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: cls._strip_non_scientific(v)
                for k, v in value.items()
                if k not in cls.NON_SCIENTIFIC_LEAVES
            }
        if isinstance(value, (list, tuple)):
            return [cls._strip_non_scientific(v) for v in value]
        return value

    def _science_dict(self) -> dict[str, Any]:
        """Fingerprinted fields. Realized fold windows are not part of the design."""
        raw = asdict(self)
        payload: dict[str, Any] = {}
        for key in self.SCIENTIFIC_FIELDS:
            value = self._strip_non_scientific(raw.get(key))
            if key == "validation" and isinstance(value, dict):
                value = {
                    ik: iv for ik, iv in value.items()
                    if ik not in {"fold_windows", "hold_out_dates"}
                }
            payload[key] = value
        return payload

    def scientific_fingerprint(self) -> str:
        """Stable hash over the scientifically meaningful fields only."""
        blob = json.dumps(self._science_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _fills_empty_science(self, existing: "RunManifest") -> bool:
        """True when every scientific difference is an empty field being filled.

        Recording the fold design on a manifest the builder opened before
        validation ran is that case. Changing a design that was already
        recorded is not.
        """
        old = existing._science_dict()
        new = self._science_dict()
        for key, new_value in new.items():
            old_value = old.get(key)
            if old_value == new_value:
                continue
            if key == "datasets" and self._only_adds_datasets(old_value, new_value):
                continue
            if not self._is_empty(old_value):
                return False
        return True

    @staticmethod
    def _only_adds_datasets(old_value: Any, new_value: Any) -> bool:
        """True when the dataset inventory only GREW.

        A multi-pass run is assembled one pass at a time, and pass 2 names
        inputs pass 1 never saw -- its own outcome contract and the upstream
        predictions it gates on. Appending those is not rewriting history, so
        refusing it would force every pass into its own ``run_id`` and destroy
        the run as the unit that holds a cascade together. Changing or dropping
        a ref that is already recorded IS rewriting history, and still fails:
        every existing entry must survive byte-identically.
        """
        if not isinstance(old_value, (list, tuple)) or not isinstance(new_value, (list, tuple)):
            return False
        remaining = list(new_value)
        for entry in old_value:
            if entry not in remaining:
                return False
            remaining.remove(entry)
        return True

    def to_json(self) -> str:
        data = asdict(self)
        # Always persisted in the CURRENT format, even for a manifest that was
        # read from a v1 file: its fields were normalised on read.
        data["schema_version"] = SCHEMA_VERSION
        data["scientific_fingerprint"] = self.scientific_fingerprint()
        data["evidence_level"] = self.evidence_level
        data["missing_evidence"] = list(self.missing_evidence())
        return json.dumps(data, indent=2, sort_keys=True, default=str)

    def write(self, run_dir: Path) -> Path:
        """Write ``manifest.json`` under ``run_dir``.

        Enforces scientific immutability:

          * no existing manifest -> write;
          * existing manifest with an identical scientific fingerprint ->
            rewrite (idempotent, e.g. a retried job);
          * existing manifest with a different ``run_id`` **or** a different
            fingerprint -> ``FileExistsError``.

        The second case is what makes a ``run_id`` trustworthy: a reader can
        rely on a given id always having meant the same experiment.
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / "manifest.json"
        if target.exists():
            existing = RunManifest.read(run_dir)
            if existing.run_id != self.run_id:
                raise FileExistsError(
                    f"{target} already holds manifest for run_id="
                    f"{existing.run_id!r}, refusing to overwrite with "
                    f"run_id={self.run_id!r}"
                )
            same = existing.scientific_fingerprint() == self.scientific_fingerprint()
            if not same and not self._fills_empty_science(existing):
                raise FileExistsError(
                    f"{target} already holds run_id={self.run_id!r} with a "
                    f"different scientific fingerprint "
                    f"({existing.scientific_fingerprint()} != "
                    f"{self.scientific_fingerprint()}). A run_id is immutable: "
                    f"changing what an experiment was while keeping its id "
                    f"silently rewrites history. Use a new run_id."
                )
        target.write_text(self.to_json())
        return target

    @staticmethod
    def read(run_dir: Path) -> "RunManifest":
        """Read ``manifest.json`` from ``run_dir`` back into a RunManifest."""
        data = json.loads((Path(run_dir) / "manifest.json").read_text())
        # Derived, not stored state: recomputed from the fields on every read
        # so a hand-edited file cannot assert a fingerprint or an evidence
        # level its own contents do not support.
        for derived in ("scientific_fingerprint", "evidence_level", "missing_evidence"):
            data.pop(derived, None)
        # v1 -> v2 normalisation. The old key and the new one are the SAME
        # concept; both present is contradictory and refused rather than
        # silently resolved.
        if "research_objective_id" in data:
            if "objective_id" in data and data["objective_id"] != data["research_objective_id"]:
                raise ValueError(
                    "manifest carries both objective_id and legacy "
                    "research_objective_id with different values"
                )
            data["objective_id"] = data.pop("research_objective_id")
        if isinstance(data.get("artifacts"), list):  # v1: flat list of paths
            # Re-derive canonical-class buckets the same way the write path
            # does now (artifacts.classify_existing), rather than inventing
            # an "unclassified" bucket that would itself violate the
            # canonical-classes-only contract enforced in __post_init__.
            # A legacy path with no canonical realisation is dropped here
            # exactly as one is dropped from a freshly-built inventory.
            from src.experiment.artifacts import classify_existing
            grouped: dict[str, list[str]] = {}
            for relpath in data["artifacts"]:
                _status, realises = classify_existing(str(relpath))
                if realises is not None:
                    grouped.setdefault(realises, []).append(str(relpath))
            data["artifacts"] = grouped
        version = data.get("schema_version", 0)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version={version} is newer than this code "
                f"supports ({SCHEMA_VERSION}); upgrade rather than "
                f"silently misreading it"
            )
        if "code" in data and isinstance(data["code"], dict):
            data["code"] = CodeIdentity(**data["code"])
        if data.get("validation") is not None and isinstance(data["validation"], dict):
            data["validation"] = ValidationDesign(**{
                k: v for k, v in data["validation"].items()
                if k in {f.name for f in ValidationDesign.__dataclass_fields__.values()}
            })
        passes = data.get("passes") or ()
        data["passes"] = tuple(passes)
        if "datasets" in data:
            data["datasets"] = tuple(
                DatasetRef(**d) if isinstance(d, dict) else d for d in data["datasets"]
            )
        return RunManifest(**data)


def validation_from_mapping(data: Mapping[str, Any] | None) -> ValidationDesign | None:
    """Build a :class:`ValidationDesign` from a fold-design dict.

    Accepts the legacy ``folds.json`` shape (``folds`` instead of
    ``fold_windows``, ``n_folds`` omitted) and ignores keys this dataclass
    does not declare.
    """
    if not isinstance(data, Mapping) or not data.get("split"):
        return None
    windows = data.get("fold_windows", data.get("folds"))
    n_folds = data.get("n_folds")
    if not n_folds:
        n_folds = len(windows) if windows else 0
    if int(n_folds) < 1:
        return None
    payload = dict(data)
    payload["n_folds"] = int(n_folds)
    if windows is not None:
        payload["fold_windows"] = windows
    known = {f.name for f in ValidationDesign.__dataclass_fields__.values()}
    return ValidationDesign(**{k: payload[k] for k in known if k in payload})


def _merge_pass_list(passes: list[dict[str, Any]], incoming: dict[str, Any]) -> list[dict[str, Any]]:
    found = next((i for i, p in enumerate(passes) if p.get("pass") == incoming.get("pass")), None)
    if found is None:
        passes.append(dict(incoming))
    else:
        current = dict(passes[found])
        models_in = incoming.get("models")
        if models_in is not None:
            models = list(current.get("models") or [])
            for item in models_in:
                models = [m for m in models if m.get("model_id") != item.get("model_id")]
                models.append(dict(item))
            current["models"] = models
        for key, value in incoming.items():
            if key == "models":
                continue
            if key == "fold_windows" and value is None:
                continue
            current[key] = value
        passes[found] = current
    passes.sort(key=lambda p: int(p["pass"]))
    return passes


def _collect_artifacts(passes: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}

    def add(kind: str, rel: str | None) -> None:
        if not rel:
            return
        bucket = found.setdefault(kind, [])
        if rel not in bucket:
            bucket.append(str(rel))

    for entry in passes:
        for model in entry.get("models") or []:
            mid = model.get("model_id")
            add("oof_predictions", model.get("oof_predictions"))
            add("metrics", model.get("metrics"))
            add("fold_metrics", model.get("fold_metrics"))
            add("model", model.get("model_json"))
            if mid and model.get("model_json") is None and entry.get("pass") is not None:
                add("model", f"pass_{int(entry['pass'])}/model_{mid}/model.json")
    return found


def _union_artifacts(
    existing: Mapping[str, Sequence[str]],
    extra: Mapping[str, Sequence[str]],
) -> dict[str, list[str]]:
    merged = {key: list(value) for key, value in existing.items()}
    for key, paths in extra.items():
        bucket = merged.setdefault(key, [])
        for rel in paths:
            if rel not in bucket:
                bucket.append(rel)
    return merged


_UPSTREAM_SEMANTIC = {
    "labels": "outcomes",
    "oof": "predictions",
    "oof_core": "predictions",
    "predictions": "predictions",
    "features": "features",
    "context": "context",
    "price_tape": "raw_source",
}


def _datasets_from_passes(passes: Sequence[Mapping[str, Any]]) -> tuple[DatasetRef, ...]:
    """Label and feature refs named by a pass. A path alone is not an identity."""
    refs: list[DatasetRef] = []
    seen: set[tuple[str, str | None]] = set()
    for entry in passes:
        sources: list[Mapping[str, Any]] = list(entry.get("upstream") or [])
        inputs = entry.get("inputs") or {}
        labels = inputs.get("labels")
        if isinstance(labels, Mapping):
            sources.append(labels)
        for key in ("features", "context", "predictions"):
            block = inputs.get(key) or []
            if isinstance(block, Mapping):
                block = [block]
            sources.extend(item for item in block if isinstance(item, Mapping))
        for item in sources:
            contract_id = item.get("label_contract_id") or item.get("contract_id")
            store_id = item.get("store_id")
            if not (contract_id or store_id or item.get("build_id") or item.get("content_sha256")):
                continue
            location = item.get("location") or item.get("path")
            if location is not None and Path(str(location)).is_absolute():
                location = None
            logical = str(item.get("logical_id") or contract_id or store_id)
            marker = (logical, str(location) if location else None)
            if marker in seen:
                continue
            seen.add(marker)
            semantic = item.get("semantic_type") or _UPSTREAM_SEMANTIC.get(str(item.get("kind") or ""))
            if semantic not in DATASET_SEMANTIC_TYPES:
                semantic = "outcomes" if contract_id else None
            refs.append(DatasetRef(
                logical_id=logical,
                semantic_type=semantic,
                contract_id=str(contract_id) if contract_id else None,
                build_id=item.get("build_id"),
                content_sha256=item.get("content_sha256"),
                store_id=str(store_id) if store_id else None,
                location=str(location) if location else None,
            ))
    return tuple(refs)


def _code_identity() -> CodeIdentity | None:
    try:
        from src.common.code_identity import git_identity
    except Exception:
        return None
    got = git_identity()
    if not got:
        return None
    sha, dirty = got
    return CodeIdentity(git_sha=sha, dirty=dirty)


def _manifest_from_legacy(run_dir: Path, objective_id: str, run_ts: str) -> RunManifest | None:
    """Read a pre-manifest objective run. Does not write or delete its files."""
    run_path = run_dir / "run.json"
    if not run_path.is_file():
        return None
    doc = json.loads(run_path.read_text())
    snapshot: Mapping[str, Any] = {}
    resolved = False
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        raw = json.loads(contract_path.read_text())
        if isinstance(raw.get("objective_snapshot"), dict):
            snapshot = raw["objective_snapshot"]
            resolved = True
        elif isinstance(raw, dict):
            snapshot = raw
    validation = None
    folds_path = run_dir / "folds.json"
    if folds_path.is_file():
        validation = validation_from_mapping(json.loads(folds_path.read_text()))
    passes = tuple(doc.get("passes") or [])
    role = "model_strategy"
    if passes and isinstance(passes[0], Mapping) and passes[0].get("role"):
        role = str(passes[0]["role"])
    return RunManifest(
        run_id=f"{objective_id}/{run_ts}",
        objective_id=str(doc.get("objective_id") or objective_id),
        hypothesis=str(doc.get("hypothesis") or ""),
        responsibility_versions={role: "legacy-run-json"},
        code=_code_identity(),
        datasets=_datasets_from_passes(passes),
        config_snapshot=dict(snapshot),
        config_snapshot_resolved=resolved,
        validation=validation,
        passes=passes,
        artifacts=_collect_artifacts(passes),
    )


def read_run_record(run_dir: Path) -> dict[str, Any]:
    """Run metadata from ``manifest.json``, or from the legacy trio.

    The returned dict always has ``passes``. ``config_snapshot`` and
    ``validation`` are present when that file recorded them. Legacy
    ``run.json`` / ``contract.json`` / ``folds.json`` are not rewritten.
    """
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text())
        run_id = str(data.get("run_id") or "")
        run_ts = run_id.split("/", 1)[1] if "/" in run_id else run_dir.name
        doc = {
            "objective_id": data.get("objective_id"),
            "run_ts": run_ts,
            "run_id": run_id,
            "passes": list(data.get("passes") or []),
            "config_snapshot": data.get("config_snapshot") or {},
            "config_snapshot_resolved": bool(data.get("config_snapshot_resolved")),
            "validation": data.get("validation"),
            "datasets": list(data.get("datasets") or []),
            "candidates": [],
        }
        # A run that has BOTH a manifest and a search_result (the canonical
        # layout written in place by the controller) must still report its
        # candidate list: a reader asking "what did this run fit" should get an
        # answer regardless of which file happens to hold it.
        legacy_result = run_dir / "search_result.json"
        if legacy_result.is_file():
            try:
                result = json.loads(legacy_result.read_text())
            except (OSError, ValueError):
                result = {}
            if isinstance(result, Mapping):
                doc["candidates"] = list(result.get("candidates") or [])
                doc["search_result"] = dict(result)
        return doc
    legacy = run_dir / "run.json"
    search_result_path = run_dir / "search_result.json"
    if not legacy.is_file() and not search_result_path.is_file():
        raise FileNotFoundError(manifest_path)
    doc = json.loads(legacy.read_text()) if legacy.is_file() else {}
    doc.setdefault("passes", [])
    doc.setdefault("run_ts", run_dir.name)
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        raw = json.loads(contract_path.read_text())
        doc["config_snapshot"] = raw.get("objective_snapshot") or raw
    # A Track B search / custom_test run has no contract.json. Its run-level
    # record is run.json plus search_result.json, so this reader covers both
    # trees and callers do not each reimplement the fallback. The candidate
    # list is surfaced under its own key -- it is NOT promoted to ``passes``,
    # because deciding which candidate is pass 1 and which is a meta layer is a
    # migration decision (docs/new_file_map.md's pass rule), not a read.
    if search_result_path.is_file():
        try:
            result = json.loads(search_result_path.read_text())
        except (OSError, ValueError):
            result = {}
        if isinstance(result, Mapping):
            doc.setdefault("objective_id", result.get("objective_id"))
            doc["search_result"] = dict(result)
            doc["candidates"] = list(result.get("candidates") or [])
            if "config_snapshot" not in doc:
                doc["config_snapshot"] = {}
    # A Track B run's configuration is its overlay YAMLs, copied into the run
    # by controller._copy_custom_test_specs BEFORE the fit. They are the run's
    # own inputs, so they belong in config_snapshot rather than being re-read
    # from config/ at migration time (where they may since have changed).
    if not doc.get("config_snapshot"):
        overlay: dict[str, Any] = {}
        for key, filename in (("manual_spec", "spec.manual.yaml"),
                              ("topology", "topology.yaml"),
                              ("walk_forward", "walk_forward.yaml")):
            path = run_dir / filename
            if not path.is_file():
                continue
            try:
                import yaml
                overlay[key] = yaml.safe_load(path.read_text())
            except (OSError, ValueError, ImportError):
                continue  # an unreadable overlay is not fabricated content
        if overlay:
            doc["config_snapshot"] = overlay
    folds_path = run_dir / "folds.json"
    if folds_path.is_file():
        doc["validation"] = json.loads(folds_path.read_text())
    elif isinstance(doc.get("search_result"), Mapping):
        folds = doc["search_result"].get("folds")
        if folds:
            doc["validation"] = {"folds": folds}
    return doc


def merge_pass(
    run_dir: Path,
    *,
    objective_id: str,
    run_ts: str,
    pass_entry: Mapping[str, Any],
    config_snapshot: Mapping[str, Any] | None = None,
    config_snapshot_resolved: bool | None = None,
    validation: Mapping[str, Any] | None = None,
    hypothesis: str | None = None,
    responsibility_versions: Mapping[str, str] | None = None,
    seeds: Mapping[str, int] | None = None,
    execution_assumptions: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    code: "CodeIdentity | None" = None,
) -> Path:
    """Insert or update one pass inside ``manifest.json``.

    Creates the manifest when the run does not have one. An existing
    scientific field is left unchanged; an empty one may be filled. Model
    pointers and pass windows are inventory and update in place. Legacy
    ``run.json``, ``contract.json`` and ``folds.json`` are not written.
    The label contract file is not opened.
    """
    run_dir = Path(run_dir)
    existing: RunManifest | None = None
    if (run_dir / "manifest.json").is_file():
        existing = RunManifest.read(run_dir)
    else:
        existing = _manifest_from_legacy(run_dir, objective_id, run_ts)

    if existing is None:
        passes = _merge_pass_list([], dict(pass_entry))
        datasets = _datasets_from_passes(passes)
        role = str(pass_entry.get("role") or "model_strategy")
        versions = dict(responsibility_versions or {role: "objective-run"})
        manifest = RunManifest(
            run_id=f"{objective_id}/{run_ts}",
            objective_id=objective_id,
            hypothesis=hypothesis or str(pass_entry.get("purpose") or ""),
            responsibility_versions=versions,
            # A migrated historical run must NOT be stamped with today's
            # commit: the caller passes the identity the run itself recorded.
            code=code or _code_identity(),
            datasets=datasets,
            config_snapshot=dict(config_snapshot or {}),
            config_snapshot_resolved=bool(config_snapshot_resolved),
            validation=validation_from_mapping(validation),
            seeds=dict(seeds or {}),
            execution_assumptions=dict(execution_assumptions or {}),
            extra=dict(extra or {}),
            passes=tuple(passes),
            artifacts=_collect_artifacts(passes),
        )
        return manifest.write(run_dir)

    passes = _merge_pass_list([dict(p) for p in existing.passes], dict(pass_entry))
    artifacts = _union_artifacts(existing.artifacts, _collect_artifacts(passes))
    snapshot = existing.config_snapshot
    resolved = existing.config_snapshot_resolved
    if config_snapshot is not None and RunManifest._is_empty(existing.config_snapshot):
        snapshot = dict(config_snapshot)
        resolved = bool(config_snapshot_resolved)
    design = existing.validation
    if validation is not None and existing.validation is None:
        design = validation_from_mapping(validation)
    # Datasets are INVENTORY derived from the pass list, not a scientific field
    # that the first writer gets to freeze: merging a second pass must add that
    # pass's inputs. Refs already on the manifest that no pass names (a caller
    # set them directly) are kept; a ref the passes still name is refreshed.
    inferred = _datasets_from_passes(passes)
    if inferred:
        by_key = {(d.logical_id, d.location): d for d in existing.datasets}
        by_key.update({(d.logical_id, d.location): d for d in inferred})
        datasets = tuple(by_key.values())
    else:
        datasets = existing.datasets
    # Seeds and cost assumptions are scientific fields: an empty one may be
    # filled by the step that learned it, a recorded one is never restated.
    applied_seeds = existing.seeds
    if seeds and RunManifest._is_empty(existing.seeds):
        applied_seeds = dict(seeds)
    costs = existing.execution_assumptions
    if execution_assumptions and RunManifest._is_empty(existing.execution_assumptions):
        costs = dict(execution_assumptions)
    # `extra` is bookkeeping, excluded from the fingerprint: new keys merge in.
    bookkeeping = dict(existing.extra or {})
    if extra:
        bookkeeping.update({k: v for k, v in extra.items() if k not in bookkeeping})
    text = existing.hypothesis
    if hypothesis and RunManifest._is_empty(existing.hypothesis):
        text = hypothesis
    versions = existing.responsibility_versions
    if responsibility_versions and RunManifest._is_empty(existing.responsibility_versions):
        versions = dict(responsibility_versions)
    updated = RunManifest(
        run_id=existing.run_id,
        objective_id=existing.objective_id,
        hypothesis=text,
        responsibility_versions=versions,
        code=existing.code or code,
        datasets=datasets,
        config_snapshot=snapshot,
        config_snapshot_resolved=resolved,
        validation=design,
        experiment_family=existing.experiment_family,
        parent_run_id=existing.parent_run_id,
        baseline_run_id=existing.baseline_run_id,
        seeds=applied_seeds,
        execution_assumptions=costs,
        decision_policy=existing.decision_policy,
        calibration=existing.calibration,
        artifacts=artifacts,
        passes=tuple(passes),
        schema_version=SCHEMA_VERSION,
        created_at=existing.created_at,
        extra=bookkeeping,
    )
    return updated.write(run_dir)
