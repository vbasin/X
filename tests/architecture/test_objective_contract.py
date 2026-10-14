"""Objective, run-identity, input-path and output-artifact contracts. DB-free."""
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from src.experiment.artifacts import (
    ARTIFACTS,
    applicable,
    classify_existing,
)
from src.experiment.manifest import (
    CodeIdentity,
    DatasetRef,
    RunManifest,
)
from src.experiment.objective import (
    field_roles,
    objective_identity,
    objective_identity_sha256,
)

REPO = Path(__file__).resolve().parents[2]
TRACK_C = REPO / "config" / "objectives" / "nq-Rscaled-mkt.yaml"
FIRSTPASS = REPO / "config" / "objectives" / "mnq-auction-firstpass.yaml"


def _m(**kw):
    base = dict(run_id="nq-50pt-econ/20260922T145336Z", objective_id="nq-50pt-econ",
                hypothesis="h", responsibility_versions={"model_strategy": "x"},
                code=CodeIdentity(git_sha="abc", dirty=False),
                datasets=(), config_snapshot={"a": 1})
    base.update(kw)
    return RunManifest(**base)


# --- 1. objective_id is the one term -----------------------------------------

def test_new_manifests_write_objective_id_only(tmp_path):
    _m().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["objective_id"] == "nq-50pt-econ"
    assert "research_objective_id" not in data
    assert data["schema_version"] == 4


def test_legacy_v1_manifest_is_readable_and_normalised(tmp_path):
    _m().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["research_objective_id"] = data.pop("objective_id")
    data["schema_version"] = 1
    # v1 flat list: one path with a canonical realisation, one without --
    # the latter must be dropped, never bucketed under an invented
    # "unclassified" class (that would itself violate the canonical-classes-
    # only contract enforced by __post_init__).
    data["artifacts"] = ["oof_xgboost.parquet", "run.json"]
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    back = RunManifest.read(tmp_path)
    assert back.objective_id == "nq-50pt-econ"
    assert back.artifacts == {"oof_predictions": ["oof_xgboost.parquet"]}
    assert set(back.artifacts) <= set(a.artifact_class for a in ARTIFACTS)
    # same science -> same fingerprint -> the legacy file can be rewritten
    # idempotently (and is then persisted in the current schema)
    assert back.scientific_fingerprint() == _m().scientific_fingerprint()
    _m().write(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 4


def test_v2_manifest_is_readable_and_new_writes_are_current(tmp_path):
    """v2 -> v3 (tenth pass): DatasetRef.build_id's MEANING changed (no
    longer part-inventory-derived), not just its presence -- a v2 file's
    build_id value is read back as-is (old, weaker semantics), never
    rewritten or discarded, but any NEW write uses the current schema."""
    m = _m(datasets=(DatasetRef(logical_id="labels_15s", contract_id="c1",
                                build_id="old-part-inventory-fingerprint"),))
    m.write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["schema_version"] == 4
    data["schema_version"] = 2
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    back = RunManifest.read(tmp_path)
    assert back.datasets[0].build_id == "old-part-inventory-fingerprint"
    assert back.datasets[0].identity_strength == "build"

    back.write(tmp_path)  # idempotent rewrite -> persisted in v3
    assert json.loads((tmp_path / "manifest.json").read_text())["schema_version"] == 4


def test_future_schema_version_is_still_rejected(tmp_path):
    _m().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["schema_version"] = 5
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="newer than this code supports"):
        RunManifest.read(tmp_path)


def test_legacy_dataset_ref_without_build_id_still_reads(tmp_path):
    """A manifest.json written before build_id existed (ninth pass) has no
    'build_id' key in its datasets dicts at all -- simulated here by
    deleting it, since asdict() always serialises every field including
    None defaults. DatasetRef(**d) must still reconstruct it -- the
    dataclass default (None) fills the true gap, not an error. Backward
    compatibility for the field addition in §5a."""
    m = _m(datasets=(DatasetRef(logical_id="labels_15s", contract_id="c1"),))
    m.write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    assert data["datasets"][0]["build_id"] is None
    del data["datasets"][0]["build_id"]
    (tmp_path / "manifest.json").write_text(json.dumps(data))

    back = RunManifest.read(tmp_path)
    assert back.datasets[0].contract_id == "c1"
    assert back.datasets[0].build_id is None
    assert back.datasets[0].identity_strength == "contract"


def test_contradictory_legacy_and_new_keys_are_refused(tmp_path):
    _m().write(tmp_path)
    data = json.loads((tmp_path / "manifest.json").read_text())
    data["research_objective_id"] = "something-else"
    (tmp_path / "manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="both objective_id and legacy"):
        RunManifest.read(tmp_path)


def test_research_objective_id_is_gone_from_canonical_code():
    """No field or second alias by that name; only manifest.py's legacy
    reader may mention it, to normalise v1 files."""
    from dataclasses import fields
    assert "research_objective_id" not in {f.name for f in fields(RunManifest)}
    for f in (REPO / "src" / "canonical").rglob("*.py"):
        if f.name != "manifest.py":
            assert "research_objective_id" not in f.read_text(), f


# --- 2/4. run_id vs run_ts ----------------------------------------------------

def test_track_b_run_id_is_objective_id_slash_run_ts(tmp_path):
    from src.search.run_manifest import build_manifest
    m = build_manifest(objective_id="nq-50pt-econ", run_ts="20260922T145336Z",
                       out_dir=tmp_path, objective={}, experiment_type="custom_test",
                       fold_split=None, fold_meta=None)
    assert m.run_id == "nq-50pt-econ/20260922T145336Z"
    objective_id, run_ts = m.run_id.split("/")
    assert (objective_id, run_ts) == (m.objective_id, "20260922T145336Z")


def test_no_run_key_terminology_in_canonical_code_or_docs():
    paths = list((REPO / "src" / "canonical").rglob("*.py")) + [
        REPO / "src" / "search" / "run_manifest.py",
        REPO / "docs" / "EXPERIMENTS.md", REPO / "docs" / "STORAGE_CONTRACT.md",
    ]
    for p in paths:
        assert not re.search(r"\brun_key\b", p.read_text().replace('"run_key"', "")
                             .replace("(e.g. \"run_key\")", "")), p


# --- 2. objective identity vs defaults ---------------------------------------

def test_every_schema_property_has_a_role():
    schema = json.loads((REPO / "config/objectives/objective.schema.json").read_text())
    for key, spec in schema["properties"].items():
        assert spec.get("x-objective-role") in {"identity", "default", "meta", "mixed"}, key


def test_defaults_do_not_change_objective_identity_but_identity_fields_do():
    obj = yaml.safe_load((REPO / "config/objectives/nq-50pt-econ.yaml").read_text())
    base = objective_identity_sha256(obj)
    changed_default = dict(obj, eval={"walk_forward_folds": 9}, search={"tune_trials": 99})
    assert objective_identity_sha256(changed_default) == base
    changed_identity = dict(obj, labels=dict(obj["labels"], move_threshold=75))
    assert objective_identity_sha256(changed_identity) != base


def test_training_window_is_default_trading_window_is_identity():
    roles = field_roles()
    assert roles["session.training_start"] == "default"
    assert roles["session.trading_start"] == "identity"
    assert roles["labels.encoding"] == "default"
    assert roles["labels.scheme"] == "identity"


def test_all_existing_objectives_still_validate():
    from src.objectives.loader import load_objective
    for f in sorted((REPO / "config" / "objectives").glob("*.yaml")):
        load_objective(str(f))


# --- 3. Track C objective -----------------------------------------------------

def test_track_c_objective_validates_through_the_existing_loader():
    from src.objectives.loader import load_objective
    obj = load_objective("mnq-auction-firstpass")
    assert obj["objective_id"] == "mnq-auction-firstpass"


def test_track_c_identity_matches_cascade_policy_source_of_truth():
    from src.training import cascade_policy as policy
    from src.training.move_label import SIDES, MoveConfig
    obj = yaml.safe_load(TRACK_C.read_text())
    assert obj["labels"]["move_threshold"] == policy.MIN_TGT_R
    assert obj["labels"]["stop_rule"] == {"ratio_to_target": policy.STOP_RATIO,
                                          "max": policy.MAX_STOP_R}
    assert obj["horizon"]["max_trade_horizon"] == f"{policy.HORIZON_SECONDS // 60}m"
    assert obj["session"]["decision_grid_seconds"] == policy.DECISION_GRID_SECONDS
    cfg = MoveConfig()
    assert (obj["session"]["trading_start"], obj["session"]["trading_end"],
            obj["session"]["flatten"]) == (cfg.entry_start, cfg.entry_end, cfg.flatten)
    assert len(obj["directions"]) == len(SIDES)


def test_track_c_objective_name_is_not_topology_or_version():
    oid = yaml.safe_load(FIRSTPASS.read_text())["objective_id"]
    assert oid == "mnq-auction-firstpass"
    assert not re.search(r"stage|cascade|xgb|xgboost|move_matrix|target_matrix|v\d", oid)


def test_track_c_underlying_is_nq_and_mnq_equivalent_is_sizing_only():
    """The cascade's default economic SIZING is MNQ_equivalent, but the
    underlying market/data stays NQ. That claim lives on nq-Rscaled-mkt,
    not on the superseded first-passage objective."""
    from src.evaluation.economic import load_assumptions
    obj = yaml.safe_load(TRACK_C.read_text())
    primary = [p for p in obj["traded products"] if p["role"] == "primary"]
    assert [p["symbol"] for p in primary] == ["NQ"]
    # cost profile is a REFERENCE, not duplicated numbers
    assert obj["costs"] == {"execution_cost_profile": "MNQ_equivalent"}
    costs = REPO / "config" / "execution_costs.yaml"
    mnq = load_assumptions(costs, "MNQ_equivalent")
    assert mnq.underlying == "NQ"
    assert mnq.economic_contract_unit == "MNQ_equivalent"
    assert "MNQ" not in [p["symbol"] for p in obj["traded products"]]


def test_firstpass_objective_matches_the_historical_label_and_book():
    """mnq-auction-firstpass.yaml states ADR-123 as amended by ADR-126.

    The label builder's --candidates default is still the superseded
    profile path. The document describes --candidates bounds.
    """
    import importlib.util
    import inspect

    from src.features.cash_session_profile import LevelConfig, bound_candidates
    from src.training.first_passage import FirstPassageConfig
    from src.training.path_labels import PathLabelConfig

    obj = yaml.safe_load(FIRSTPASS.read_text())
    cfg = FirstPassageConfig()
    levels = LevelConfig()
    rcfg = PathLabelConfig()
    sig = inspect.signature(bound_candidates)
    labels = obj["labels"]

    assert obj["horizon"]["max_trade_horizon"] == "session_close"
    assert "move_threshold" not in labels and "stop_rule" not in labels
    assert (obj["session"]["trading_start"], obj["session"]["trading_end"],
            obj["session"]["flatten"], obj["session"]["timezone"]) == (
                cfg.entry_start, cfg.entry_end, cfg.flatten, cfg.timezone)
    assert obj["session"]["decision_grid_seconds"] == cfg.decision_grid_seconds == 60

    assert labels["scheme"] == "candidate_exit_surface"
    assert labels["target_observation"] == "path_1s_hilo"
    assert labels["entry"] == "limit"
    assert labels["outcome"] == {
        "task": "regression",
        "target": "pnl_r",
        "tie_policy": cfg.tie_policy,
        "time_barrier": "session_flatten",
        "censored_exit": "flatten_close",
        "unfilled_emits_row": False,
    }
    assert labels["entry_rule"] == {
        "inside_bound": "far_edge",
        "between_bounds_offset_pts": levels.entry_offset_pts,
        "max_fill_wait_seconds": cfg.max_fill_wait_seconds,
        "race_starts_at": "fill",
    }
    assert labels["candidates"]["level_lookback_sessions"] == levels.lookback_sessions
    assert labels["candidates"]["sigma_pts"] == levels.sigma_pts
    assert labels["candidates"]["min_dist_r"] == sig.parameters["min_dist_r"].default
    assert labels["candidates"]["max_dist_r"] == sig.parameters["max_dist_r"].default
    assert labels["candidates"]["fill_min_dist_r"] == cfg.min_dist_r
    assert labels["candidates"]["max_per_side_per_kind"] == sig.parameters["max_per_side"].default
    assert labels["candidates"]["ladder_r"] == list(sig.parameters["ladder_r"].default)
    assert labels["candidates"]["source"] == "kde_bounds_plus_ladder"
    assert labels["candidates"]["budgets"] == "separate_structural_and_ladder"

    unit = labels["unit_definition"]
    assert unit["lookback_sessions"] == rcfg.vol_lookback_sessions
    assert unit["per_session_block_seconds"] == rcfg.vol_window_seconds
    assert unit["per_session_metric"] == "median_1s_block_range"
    assert unit["excludes_current_session"] is True
    assert unit["window_end"] == cfg.flatten

    spec = importlib.util.spec_from_file_location(
        "eval_first_passage_model",
        REPO / "scripts" / "features" / "eval_first_passage_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sel = labels["selection"]
    assert sel["quantiles"] == [float(q) for q in mod.TAU_QUANTILES]
    assert sel["point_value_dollars"] == mod.MNQ_POINT_VALUE
    assert sel["costs_applied"] is False
    assert sel["per_bar"] == "argmax_predicted_target"
    assert sel["busy_seconds"] == "t_fill_plus_t_event"
    assert sel["operating_point"] == "max_t_stat"
    assert sel["min_trades"] == 150
    assert obj["kpi"]["primary"] == "t_stat"
    assert obj["eval"]["bar_stride"] == 15
    assert "costs" not in obj

    builder = (REPO / "scripts" / "features" / "build_first_passage.py").read_text()
    assert 'choices=["profile", "bounds"], default="profile"' in builder
    assert "PathLabelConfig(entry_start=args.entry_start, entry_end=args.flatten" in builder
    primary = [p for p in obj["traded products"] if p["role"] == "primary"]
    assert [p["symbol"] for p in primary] == ["NQ"]


def test_economic_contract_unit_replaces_instrument():
    from dataclasses import fields
    from src.evaluation.economic import EconomicAssumptions
    names = {f.name for f in fields(EconomicAssumptions)}
    assert "economic_contract_unit" in names and "underlying" in names
    assert "instrument" not in names


# --- 5. input path contract ---------------------------------------------------

def test_dataset_ref_refuses_absolute_machine_paths():
    with pytest.raises(ValueError, match="absolute machine path"):
        DatasetRef(logical_id="l", contract_id="c", location="/opt/vtech/data/x")


def test_dataset_ref_identity_strength_order():
    """content > build > contract > store -- three distinct questions, not
    degrees of one. See docs/STORAGE_CONTRACT.md §5a (ninth pass)."""
    assert DatasetRef("l", content_sha256="h", build_id="b",
                      contract_id="c").identity_strength == "content"
    assert DatasetRef("l", build_id="b", contract_id="c").identity_strength == "build"
    assert DatasetRef("l", contract_id="c").identity_strength == "contract"
    assert DatasetRef("l", store_id="native_tf_15s").identity_strength == "store"


def test_dataset_ref_build_id_alone_is_sufficient_identity():
    """build_id, like contract_id and content_sha256, satisfies the
    constructor's scientific-identity requirement on its own."""
    ref = DatasetRef("l", build_id="b")
    assert ref.identity_strength == "build"
    with pytest.raises(ValueError, match="scientific identity"):
        DatasetRef("l")  # none of content_sha256/build_id/contract_id/store_id


def test_contract_only_dataset_evidence_is_incomplete_but_build_or_content_satisfies_it():
    """Tenth-pass correction: a contract_id-only DatasetRef states what a
    dataset MEANS, not WHICH realization was used -- a dataset can be
    rebuilt under the same semantic contract (label_store.write_label_parts
    (..., force=True)). Only build (realization) or content identity is
    strong enough for RunManifest.missing_evidence() to count `datasets` as
    resolved. See docs/STORAGE_CONTRACT.md §5a."""
    contract_only = _m(datasets=(DatasetRef("labels_15s", contract_id="c1"),))
    assert "datasets" in contract_only.missing_evidence()

    with_build = _m(datasets=(DatasetRef("labels_15s", contract_id="c1", build_id="b1"),))
    assert "datasets" not in with_build.missing_evidence()

    with_content = _m(datasets=(DatasetRef("labels_15s", contract_id="c1",
                                           content_sha256="deadbeef"),))
    assert "datasets" not in with_content.missing_evidence()

    store_only = _m(datasets=(DatasetRef("labels_15s", store_id="native_tf_15s"),))
    assert "datasets" in store_only.missing_evidence()


def test_predictions_is_a_distinct_semantic_type_from_outcomes():
    """A model's OOF/scored output (e.g. build_target_matrix.py's --scores)
    is a CLAIM about the future, not ground truth -- it must be recordable
    with its own semantic_type, never folded into 'outcomes'. Tenth pass,
    docs/STORAGE_CONTRACT.md §19."""
    from src.experiment.manifest import DATASET_SEMANTIC_TYPES
    assert "predictions" in DATASET_SEMANTIC_TYPES
    assert "outcomes" in DATASET_SEMANTIC_TYPES
    ref = DatasetRef("move_scores", semantic_type="predictions", contract_id="c1")
    assert ref.semantic_type == "predictions"


def test_canonical_outcome_path_pattern_is_topology_neutral():
    """docs/STORAGE_CONTRACT.md §17.1's canonical NEW-write pattern must not
    encode a stage number, model family, version suffix, or either of Track
    C's own implementation nicknames ("move"/"target") -- those belong in
    Track C's EXISTING compatibility paths (discussed elsewhere in §17),
    never in the canonical pattern itself. Tenth-pass tightening: the ninth
    pass's pattern only forbade compound tokens (move_matrix/target_matrix);
    this also forbids the bare words, since <outcome_family> must now be a
    genuine semantic name, not a Track C jargon abbreviation."""
    text = (REPO / "docs" / "STORAGE_CONTRACT.md").read_text()
    start = text.index("### 17.1 The pattern")
    block_start = text.index("```", start) + 3
    block_end = text.index("```", block_start)
    pattern_block = text[block_start:block_end]

    assert "<outcome_family>" in pattern_block
    assert "<resolution>" in pattern_block
    assert "<contract_id>" in pattern_block
    forbidden = ("stage1", "stage_1", "stage 1", "stage3", "stage_3", "stage 3",
                 "xgboost", "lstm", "_v2", "_v3", "_v4",
                 "move_matrix", "target_matrix", "move", "target", "cascade", "gate")
    lowered = pattern_block.lower()
    hits = [tok for tok in forbidden if tok in lowered]
    assert not hits, f"canonical pattern encodes topology-specific tokens: {hits}"


def test_live_path_status_is_internally_consistent_in_docs():
    """docs/STORAGE_CONTRACT.md must not assert two canonical locations for
    live_dir() at once: §8's tree table and §8a's resolution must agree that
    top-level live/ is CURRENT and ops/live/ is the TARGET, not both current."""
    text = (REPO / "docs" / "STORAGE_CONTRACT.md").read_text()
    assert "## 8a." in text, "the live/ops contradiction section must exist"
    section_8a = text[text.index("## 8a."):text.index("## 9.")]
    assert "ops/live" in section_8a
    assert "live_dir()" in section_8a
    # §8's own tree table must mark live_dir() as an explicit exception, not
    # silently fold it into the generic ops/<subsystem>/ pattern row (that
    # row lists the OTHER operational-state accessors, deliberately without
    # live_dir(), to avoid re-asserting the contradiction this test guards).
    section_8 = text[text.index("## 8."):text.index("## 8a.")]
    exception_lines = [ln for ln in section_8.splitlines() if "live_dir()" in ln]
    assert exception_lines, "live_dir() must appear in §8's table"
    assert any("exception" in ln.lower() for ln in exception_lines)
    generic_ops_lines = [ln for ln in section_8.splitlines()
                         if "ops/<subsystem>" in ln]
    assert generic_ops_lines and all("live_dir()" not in ln for ln in generic_ops_lines)


def test_dataset_ref_rejects_unknown_semantic_type():
    with pytest.raises(ValueError):
        DatasetRef("l", contract_id="c", semantic_type="stuff")


def test_label_store_path_goes_through_the_existing_resolver():
    from src.common.data_paths import data_root, labels_dir, layout, logical_path
    from src.training.label_store import label_contract_id, label_set_dir
    c = {"scheme": "triple_barrier"}
    assert labels_dir() == layout("labels") == data_root() / "sources" / "labels"
    assert label_set_dir("15s", c) == labels_dir() / "15s" / label_contract_id(c)
    assert logical_path(label_set_dir("15s", c)) == f"sources/labels/15s/{label_contract_id(c)}"
    assert logical_path("/somewhere/else") is None


def test_adapter_records_relative_location_and_semantic_type(tmp_path, monkeypatch):
    from src.search.run_manifest import _datasets_from_run_dir
    monkeypatch.setenv("VTECH_DATA_DIR", str(tmp_path / "data"))
    src = tmp_path / "data" / "sources" / "labels" / "15s" / "1aa87aaf223d0f7a"
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "label_contract.json").write_text(json.dumps({
        "contract": {}, "contract_id": "1aa87aaf223d0f7a",
        "contract_sha256": "e2e7", "source_dir": str(src)}))
    (ref,) = _datasets_from_run_dir(tmp_path / "run")
    assert ref.location == "sources/labels/15s/1aa87aaf223d0f7a"
    assert ref.semantic_type == "outcomes" and ref.resolution == "15s"
    assert ref.logical_id == "labels_15s"


# --- 6. output / artifact contract -------------------------------------------

def test_one_canonical_filename_per_artifact_class():
    names = [a.filename for a in ARTIFACTS]
    classes = [a.artifact_class for a in ARTIFACTS]
    assert len(set(names)) == len(names) and len(set(classes)) == len(classes)


def test_canonical_filenames_carry_no_topology_model_or_version():
    banned = re.compile(r"stage\d|v\d|xgb|xgboost|lstm|tcn|meta|cascade|move_|target_matrix")
    for a in ARTIFACTS:
        assert not banned.search(a.filename), a.filename


def test_applicability_is_derived_from_the_manifest():
    from src.experiment.manifest import ValidationDesign
    bare = applicable(_m())
    assert bare["manifest"] and bare["metrics"]
    assert not bare["fold_metrics"] and not bare["trade_ledger"]
    assert not bare["hpo_study"] and not bare["trials_summary"]
    study = applicable(_m(config_snapshot={"study": {"kind": "hpo"}}))
    assert study["hpo_study"] and study["trials_summary"]
    assert not bare["effective_sample_size"]            # DEFERRED
    assert applicable(_m(validation=ValidationDesign("expanding_equal_count", 3)))["fold_metrics"]
    econ = applicable(_m(responsibility_versions={"economic_evaluation": "canonical@1"}))
    assert econ["trade_ledger"] and econ["economic_metrics"] and econ["cost_sensitivity"]


def test_cost_assumptions_alone_do_not_make_economic_artifacts_required():
    """A custom_test search records its objective's costs as configuration
    but performs no economic evaluation -- economics must not be demanded."""
    m = _m(execution_assumptions={"execution_cost_profile": "NQ"})
    a = applicable(m)
    assert not a["economic_metrics"] and not a["trade_ledger"]


def test_uncalibrated_run_is_not_missing_a_calibration_artifact():
    assert not applicable(_m(calibration={"method": "none"}))["calibration"]
    assert applicable(_m(calibration={"method": "isotonic"}))["calibration"]


def test_existing_track_b_files_are_classified_not_renamed():
    assert classify_existing("oof_xgboost.parquet") == ("compatibility_consumed", "oof_predictions")
    assert classify_existing("run.json")[0] == "compatibility_consumed"
    assert classify_existing("trades.parquet") == ("superseded_migration_candidate", "trade_ledger")
    assert classify_existing("metrics.json") == ("canonical", "metrics")
    assert classify_existing("whatever.bin") == ("unclassified", None)


#: Every filename docs/new_file_map.md's "Standard files" table names. A run
#: file the layout documents must never come back "unclassified" -- that is how
#: a Keras weight file or a serving scaler becomes an unknown byte blob at
#: migration time.
_DOCUMENTED_RUN_FILES: tuple[str, ...] = (
    "manifest.json", "training_matrix.parquet", "oof_predictions.parquet",
    "model.json", "model.ubj", "model.keras", "model.pkl",
    "meta.pkl", "meta_estimator.pkl", "meta.json", "scaler.pkl",
    "feature_schema.json", "finalize_manifest.json", "compose.json",
    "inference_policy.json", "metrics.json", "fold_metrics.json",
    "selected_population.json", "trade_ledger.parquet",
    "economic_metrics.json", "cost_sensitivity.json",
    "hpo.json", "trials_summary.parquet", "trial.json",
    "run.json", "contract.json", "folds.json",
    "report/summary.md", "report/shap_importance.csv", "report/shap_bar.png",
    "report/shap_beeswarm.png", "report/shap_bar_global.png",
    "report/shap_bar_signals.png", "report/shap_beeswarm_up.png",
    "report/shap_groups.csv", "report/shap_meta.json",
    "report/SHAP_SUMMARY.md", "report/shap_values_sample.npz",
)


def test_every_documented_run_file_is_classified():
    unknown = [f for f in _DOCUMENTED_RUN_FILES
               if classify_existing(f) == ("unclassified", None)]
    assert unknown == [], f"documented run files fell through classification: {unknown}"


def test_fitted_weight_files_are_one_class_in_a_bundle_or_beside_model_json():
    """A Track B bundle path and its migrated location classify identically,
    and every weight format is the same semantic class."""
    for name in ("model.ubj", "model.keras", "model.pkl",
                 "meta.pkl", "meta_estimator.pkl"):
        assert classify_existing(name) == ("canonical", "model_artifact")
        assert classify_existing(f"models/lstm_15s/{name}") == ("canonical", "model_artifact")
        assert classify_existing(f"pass_1/model_abc123/{name}") == ("canonical", "model_artifact")
    assert classify_existing("models/lstm_15s/scaler.pkl") == ("canonical", None)


def test_adapter_inventory_maps_existing_files_without_copying(tmp_path):
    """manifest.artifacts is canonical-class -> file(s) ONLY. A file that
    realises no canonical class (run.json: 'compatibility_consumed', not a
    class) must not appear as a key in this mapping -- see
    test_manifest_artifact_keys_are_always_canonical_classes below."""
    from src.search.run_manifest import build_manifest
    for f in ("run.json", "folds.json", "oof_xgboost.parquet", "label_contract.json"):
        (tmp_path / f).write_text("{}")
    m = build_manifest(objective_id="o", run_ts="t", out_dir=tmp_path, objective={},
                       experiment_type="custom_test", fold_split=None, fold_meta=None)
    assert m.artifacts["oof_predictions"] == ["oof_xgboost.parquet"]
    assert "compatibility_consumed" not in m.artifacts
    assert not any("run.json" in files for files in m.artifacts.values())
    assert not (tmp_path / "oof_predictions.parquet").exists()


def test_manifest_artifact_keys_are_always_canonical_classes():
    """Direct construction, not just the adapter: a compatibility-status
    name (e.g. from classify_existing()) must never enter `artifacts`."""
    with pytest.raises(ValueError, match="not canonical artifact classes"):
        _m(artifacts={"compatibility_consumed": ["run.json"]})
    with pytest.raises(ValueError, match="not canonical artifact classes"):
        _m(artifacts={"implementation_supplemental": ["progress.json"]})
    _m(artifacts={"metrics": ["metrics.json"]})  # a real class is fine


def test_existing_file_with_no_canonical_realisation_is_dropped_not_relabelled(tmp_path):
    """progress*.json is real, classified ('implementation_supplemental'),
    but realises no canonical class -- it must be absent from `artifacts`
    entirely, not smuggled in under its compatibility status."""
    from src.search.run_manifest import build_manifest
    (tmp_path / "progress.json").write_text("{}")
    m = build_manifest(objective_id="o", run_ts="t", out_dir=tmp_path, objective={},
                       experiment_type="custom_test", fold_split=None, fold_meta=None)
    assert not any("progress.json" in files for files in m.artifacts.values())


def _storage_contract_section(title: str) -> str:
    """Text of one `## N. ...` section of docs/STORAGE_CONTRACT.md."""
    text = (REPO / "docs" / "STORAGE_CONTRACT.md").read_text()
    start = text.index(title)
    nxt = text.find("\n## ", start + len(title))
    return text[start:nxt if nxt != -1 else len(text)]


def test_section_20_covers_every_storage_class_section_21_introduces():
    """§20 calls itself authoritative and sufficient on its own. A class that
    §21 introduces but §20 never names makes that claim false -- the failure
    mode is a fresh agent reading §20, not finding price tape or predictions,
    and inventing a path."""
    authoritative = _storage_contract_section("## 20. Final canonical storage syntax")
    run_layout = _storage_contract_section("## 21. Objective run layout")

    for token in ("sources/price_tape", "oof_predictions.parquet",
                  "research/runs", "sources/labels"):
        assert token in run_layout, f"§21 no longer mentions {token}; update this test"
        assert token in authoritative, (
            f"§21 introduces {token} but §20 -- the authoritative table -- never names it")


def test_section_20_gives_each_class_exactly_one_canonical_filename_form():
    """Price tape is session-grained and outcome datasets are not, so they have
    different filename forms on purpose. The contradiction to guard against is
    ONE class being given two names in the same document."""
    authoritative = _storage_contract_section("## 20. Final canonical storage syntax")
    tape_row = next(line for line in authoritative.splitlines()
                    if "sources/price_tape" in line)
    assert "tape_<YYYY-MM-DD>.parquet" in tape_row
    # partNNN may appear only as an explicitly deferred option, never as the name.
    if "partNNN" in tape_row:
        assert "deferred" in tape_row.lower()


def test_labels_are_never_relocated_into_a_run_cache():
    """The population cache is a cache of a join. STORAGE_CONTRACT must say so
    once, so new_file_map's label -> training_matrix rows cannot be read as a
    relocation of the contract-addressed outcome store."""
    authoritative = _storage_contract_section("## 20. Final canonical storage syntax")
    assert "training_matrix.parquet" in authoritative
    assert "cache of a" in authoritative
    assert "sources/labels/<outcome_family>/<resolution>/<contract_id>/" in authoritative


def test_new_file_map_does_not_target_the_legacy_per_model_matrix_path():
    """run_layout.training_matrix_path is documented as legacy. A migration map
    that points new copies at it contradicts the pass-level population cache."""
    text = (REPO / "docs" / "new_file_map.md").read_text()
    offenders = [line for line in text.splitlines()
                 if "model_" in line and "/training_matrix.parquet" in line
                 and "legacy" not in line.lower()]
    assert offenders == [], f"file map targets the legacy per-model matrix: {offenders}"


def test_a_second_pass_may_add_its_own_datasets_to_one_run():
    """A cascade is assembled one pass at a time. Pass 2 names inputs pass 1
    never saw -- its own outcome contract and the upstream predictions it gates
    on. Appending those must not trip scientific immutability, or every pass is
    forced into its own run_id and the run stops holding the cascade together.
    Changing or dropping an already-recorded ref must still fail."""
    import tempfile
    from src.experiment.manifest import merge_pass

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        merge_pass(run, objective_id="obj", run_ts="t", pass_entry={
            "pass": 1, "role": "directional_screen",
            "upstream": [{"kind": "labels", "label_contract_id": "aaa",
                          "logical_id": "labels_a", "path": "sources/labels/a"}],
            "models": [{"model_id": "m1m1m1m1m1m1", "model_type": "xgboost"}],
        })
        merge_pass(run, objective_id="obj", run_ts="t", pass_entry={
            "pass": 2, "role": "target_reach",
            "upstream": [
                {"kind": "labels", "label_contract_id": "bbb",
                 "logical_id": "labels_b", "path": "sources/labels/b"},
                {"kind": "oof_core", "logical_id": "pass_1_oof",
                 "build_id": "m1m1m1m1m1m1", "pass": 1,
                 "path": "pass_1/model_m1m1m1m1m1m1/oof_predictions.parquet"},
            ],
            "models": [{"model_id": "m2m2m2m2m2m2", "model_type": "xgboost"}],
        })
        m = RunManifest.read(run)
        by_id = {d.logical_id: d for d in m.datasets}
        assert set(by_id) == {"labels_a", "labels_b", "pass_1_oof"}
        assert by_id["labels_b"].contract_id == "bbb"
        # The upstream OOF is its own dataset, typed as predictions, carrying the
        # producing model's configuration hash as its build identity.
        assert by_id["pass_1_oof"].semantic_type == "predictions"
        assert by_id["pass_1_oof"].build_id == "m1m1m1m1m1m1"
        assert by_id["pass_1_oof"].identity_strength == "build"
        assert not (run / "run.json").exists()


def test_rewriting_a_recorded_dataset_still_fails():
    """The additive allowance must not become a blanket exemption."""
    import tempfile
    from src.experiment.manifest import merge_pass

    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        merge_pass(run, objective_id="obj", run_ts="t", pass_entry={
            "pass": 1, "role": "directional_screen",
            "upstream": [{"kind": "labels", "label_contract_id": "aaa",
                          "logical_id": "labels_a", "path": "sources/labels/a"}],
            "models": [{"model_id": "m1m1m1m1m1m1", "model_type": "xgboost"}],
        })
        existing = RunManifest.read(run)
        # Same logical dataset, different contract: the outcome was redefined.
        changed = replace(existing, datasets=(
            DatasetRef(logical_id="labels_a", semantic_type="outcomes",
                       contract_id="ZZZ", location="sources/labels/a"),
        ))
        with pytest.raises(FileExistsError):
            changed.write(run)


def test_compact_oof_drops_population_features_and_keeps_scores():
    """The projection copy_pass applies. A historical scores file carrying the
    feature columns it was scored on must come back as keys + fold + outcomes +
    predictions + thresholds only."""
    from src.experiment.artifacts import compact_oof_columns

    population = ["ts", "session", "side", "y_move", "es_a", "es_b", "st_c", "r_value"]
    scores = ["ts", "session", "side", "y_move", "es_a", "es_b", "st_c",
              "r_value", "weight", "p_move", "fold", "tau_0.95"]
    kept = compact_oof_columns(scores, population)
    assert "es_a" not in kept and "es_b" not in kept and "st_c" not in kept
    for name in ("ts", "session", "side", "p_move", "fold", "tau_0.95", "weight", "r_value"):
        assert name in kept, name
    features = [c for c in population if c not in set(kept)]
    assert features == ["es_a", "es_b", "st_c"]


def test_one_reader_covers_every_run_shape(tmp_path):
    """Step 7. `read_run_record` is the single run-level reader: a new objective
    run (manifest.json), an older objective run (run.json + contract.json +
    folds.json), and a Track B search / custom_test run (run.json +
    search_result.json). Track B's candidate list is surfaced under its own key
    and deliberately NOT promoted into `passes` -- deciding which candidate is
    pass 1 and which is a meta layer is a migration decision, not a read."""
    from src.experiment.manifest import merge_pass, read_run_record

    # 1. new objective run
    new_run = tmp_path / "new"
    merge_pass(new_run, objective_id="obj", run_ts="t1", pass_entry={
        "pass": 1, "role": "directional_screen",
        "models": [{"model_id": "m1m1m1m1m1m1", "model_type": "xgboost"}],
    })
    got = read_run_record(new_run)
    assert got["objective_id"] == "obj" and len(got["passes"]) == 1

    # 2. older objective run: the legacy trio
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "run.json").write_text(json.dumps(
        {"objective_id": "obj", "run_ts": "t2", "passes": [{"pass": 1}]}))
    (legacy / "contract.json").write_text(json.dumps({"objective_snapshot": {"a": 1}}))
    (legacy / "folds.json").write_text(json.dumps({"split": "expanding", "n_folds": 3}))
    got = read_run_record(legacy)
    assert got["config_snapshot"] == {"a": 1}
    assert got["validation"]["n_folds"] == 3

    # 3. Track B search / custom_test run
    track_b = tmp_path / "20260821T072412Z"
    track_b.mkdir()
    (track_b / "run.json").write_text(json.dumps(
        {"objective_id": "nq-50pt-tbm", "run_ts": "20260821T072412Z",
         "kind": "custom_test", "n_candidates": 1}))
    (track_b / "search_result.json").write_text(json.dumps(
        {"objective_id": "nq-50pt-tbm", "candidates": [{"candidate_id": "xgboost"}],
         "folds": [{"fold": 0}]}))
    got = read_run_record(track_b)
    assert got["objective_id"] == "nq-50pt-tbm"
    assert [c["candidate_id"] for c in got["candidates"]] == ["xgboost"]
    assert got["passes"] == []          # not inferred from candidates
    assert got["validation"]["folds"] == [{"fold": 0}]

    # 4. a directory that is not a run at all still raises
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_run_record(empty)
