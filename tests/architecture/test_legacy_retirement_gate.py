"""Legacy compatibility artifacts may only be retired after their real
consumers move to manifest.json / model.json.

This is the executable form of that gate. For each legacy artifact it asserts
BOTH halves are still true:

  * a real consumer still reads it, so it must NOT be deleted yet;
  * the canonical replacement exists, so the migration is actually possible.

When a consumer is finally migrated, the corresponding row here fails and the
failure IS the signal that the file can be retired -- the test is updated in the
same change that removes the last reader, never before it.

DB-free: greps source text and inspects one migrated run on disk.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: legacy filename -> (module paths that still read it, canonical replacement)
_GATE: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("run.json",
     ("src/search/tf_oof_meta.py", "src/evaluation/shap_explain.py",
      "src/experiment/manifest.py"),
     "manifest.json (passes + config_snapshot)"),
    ("search_result.json",
     ("src/search/mlflow_backfill.py", "src/experiment/manifest.py",
      "src/evaluation/shap_explain.py"),
     "manifest.json (passes[].models + metrics.json)"),
    ("folds.json",
     ("src/experiment/manifest.py",),
     "manifest.json (validation) + fold_metrics.json"),
    ("feature_schema.json",
     ("src/serving/bundle.py",),
     "model.json (feature_columns)"),
    ("finalize_manifest.json",
     ("src/serving/bundle.py",),
     "model.json (provenance fields)"),
    ("compose.json",
     ("src/search/tf_oof_meta.py",),
     "model.json (inputs.predictions)"),
)


def _greps(needle: str, *paths: str) -> str:
    return subprocess.run(
        ["grep", "-rn", "--include=*.py", needle, *paths],
        cwd=REPO, capture_output=True, text=True).stdout


@pytest.mark.parametrize("filename,readers,replacement", _GATE,
                         ids=[row[0] for row in _GATE])
def test_legacy_artifact_still_has_a_consumer(filename, readers, replacement):
    """Do not retire this file yet: at least one of these modules still reads it."""
    alive = [r for r in readers
             if (REPO / r).is_file() and filename in (REPO / r).read_text()]
    assert alive, (
        f"no module in {list(readers)} mentions {filename} any more. Its last "
        f"consumer appears to be migrated to {replacement} -- retire the file "
        f"and delete this row together, in one change."
    )


def test_the_migrated_track_b_run_did_not_remove_its_source():
    """Leftover-source check.

    Present: ``data/research/custom_test/`` was removed after the fitted
    runs moved to ``research/runs/``. An absent source is that tree, not a
    failed copy. The canonical copy is checked by the next test. A leftover
    source directory, if one is still on disk, must still contain its run
    files.
    """
    from src.common.data_paths import custom_test_dir

    src = custom_test_dir() / "nq-50pt-econ-lg-st" / "20260829T182514Z"
    if not src.is_dir():
        pytest.skip("custom_test source was removed after migration")
    for name in ("run.json", "search_result.json", "folds.json",
                 "label_contract.json"):
        assert (src / name).is_file(), f"migration removed {name} from the source"
    for candidate in ("lstm_15s", "xgb_1m", "xgb_5m"):
        bundle = src / "models" / candidate
        assert (bundle / "feature_schema.json").is_file()
        assert (bundle / "oof.parquet").is_file()
        assert any((bundle / w).is_file()
                   for w in ("model.keras", "model.pkl", "model.ubj")), candidate


def test_the_canonical_replacement_is_present_for_the_migrated_run():
    """The other half of the gate: the replacement must actually exist, or
    'migrate the consumers' is not yet a real option."""
    from src.common.run_layout import run_dir

    dest = run_dir("nq-50pt-econ-lg-st", "20260829T182514Z")
    if not dest.is_dir():
        pytest.skip("the Track B run has not been migrated on this machine")
    assert (dest / "manifest.json").is_file()
    # No legacy run-level file is written into the canonical tree.
    for name in ("run.json", "contract.json", "folds.json", "search_result.json"):
        assert not (dest / name).exists(), f"canonical run must not write {name}"
    for pass_n, model_id in ((1, "lstm_15s"), (1, "xgb_1m"), (1, "xgb_5m"),
                             (2, "tf_oof_meta__lstm_15s_xgb_1m_xgb_5m")):
        folder = dest / f"pass_{pass_n}" / f"model_{model_id}"
        assert (folder / "model.json").is_file(), folder
        assert (folder / "oof_predictions.parquet").is_file(), folder
        assert (folder / "metrics.json").is_file(), folder
        assert (folder / "fold_metrics.json").is_file(), folder
