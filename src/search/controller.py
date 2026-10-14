"""Search controller — Phase 23 W3: full-fidelity model-selection run.

Orchestrates the complete walk-forward evaluation of every catalogued model
family (no multi-fidelity pruning — signals are rare and capacity only shows
over the full window), then runs the exhaustive ensemble layer and — when
``kpi.champion_gate: true`` — applies the champion gate (ADR-071). Custom
Test never gates; it reports every fitted candidate.

Architecture
------------
* **CPU families** run on local OCPU workers via ``concurrent.futures``.
* **GPU families** are dispatched to the GPU VM via the managed **OCI Queue** (jobs +
  results channels) and executed there from a content-addressed **code bundle** — see
  ``src/gpu_runner/dispatch.py`` and ADR-022. The CPU only ``ensure_running``’s the VM; the
  VM powers itself off when idle.
* Each base run persists its **out-of-fold prediction matrix** (OOF) to disk so
  the ensemble layer can combine cached predictions without re-training.

Usage (CLI)::

    python -m src.search.controller --objective nq-50pt [--families xgboost,lightgbm]

    # USER-OVERRIDE mode — skip automated selection, fit exactly the listed models:
    python -m src.search.controller --objective nq-50pt \
        --manual-spec config/manual_candidates.example.yaml

Usage (Python)::

    from src.search import run_search
    result = run_search("nq-50pt")
    result = run_search("nq-50pt", manual_spec="config/manual_candidates.example.yaml")
"""

from __future__ import annotations

from src.common.data_paths import (
    data_root, search_dir,
)

import argparse
import json
import logging
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation import EvalContract, record_result
from src.evaluation._kpi import _LOOKUP_TAU, canonical_trade_kpi, compute_diagnostics
from src.evaluation.kpi_criteria import fire_tau
from src.models.base import FittedModel, ModelSpec
from src.models.catalog import CatalogEntry, get_catalog
from src.objectives.loader import load_objective
from src.search.gate import resolve_champion_gate

logger = logging.getLogger("vtech.search.controller")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = data_root()
_SEARCH_ROOT = search_dir()


# ── Run result ────────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    candidate_id:  str
    family:        str
    spec_params:   dict[str, Any]
    status:        str            # "ok" | "failed" | "skipped"
    score:         dict[str, Any] = field(default_factory=dict)
    oof_path:      str | None = None
    error:         str | None = None
    compute_class: str = "cpu"
    elapsed_s:     float = 0.0
    meets_target:  bool = False
    # Explicit feature selection the candidate ran with (None = full matrix / named
    # view). Persisted so the champion-finalize bridge can reproduce the winner.
    feature_selection: dict[str, Any] | None = None
    # Promote-ready serving folder written next to OOF (last walk-forward fold).
    bundle_path:   str | None = None
    # MLflow run id for this candidate (set by the controller's MLflow logger).
    mlflow_run_id: str | None = None
    # Full diagnostics dict (confusion matrices, tau sweep, per-class report).
    # Consumed in-process by the champion gate; persisted separately as
    # diagnostics_<name>.json, so it is stripped from search_result.json.
    diagnostics:   dict[str, Any] | None = field(default=None, repr=False)


# ── Experiment-type routing (ADR-036) ─────────────────────────────────────────

def _path_is_webui_custom_test(value: str | Path | dict[str, Any] | None) -> bool:
    """True when *value* is a path under ``ops/webui/custom_tests/``."""
    if not isinstance(value, (str, Path)):
        return False
    parts = Path(os.fspath(value)).parts
    try:
        i = parts.index("custom_tests")
    except ValueError:
        return False
    return i > 0 and parts[i - 1] == "webui"


def resolve_experiment_type(
    experiment_type: str | None,
    manual_spec: str | Path | dict[str, Any] | None = None,
    topology_override: str | Path | dict[str, Any] | None = None,
) -> str:
    """Pick the artifact tree. Explicit ``--experiment-type`` wins.

    WebUI Custom Test specs live under ``ops/webui/custom_tests/``. If the
    flag is omitted (stale console argv), those paths still route to
    ``custom_test`` so last-fold bundles land under ``research/custom_test/``.
    """
    et = (experiment_type or "").strip()
    if et:
        return et
    if _path_is_webui_custom_test(manual_spec) or _path_is_webui_custom_test(topology_override):
        return "custom_test"
    if manual_spec is not None:
        return "manual_override"
    return "model_selection"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_search(
    objective_id: str,
    *,
    families: list[str] | None = None,
    parquet_dir: str | Path | None = None,
    dry_run: bool = False,
    max_cpu_workers: int | None = None,
    tune_trials: int = 0,
    ensembles: bool = True,
    manual_spec: str | Path | dict[str, Any] | None = None,
    topology_override: str | Path | dict[str, Any] | None = None,
    walk_forward_override: str | Path | dict[str, Any] | None = None,
    experiment_type: str | None = None,
) -> dict[str, Any]:
    """Run the full-fidelity model-selection search for *objective_id*.

    Parameters
    ----------
    objective_id:   Trade objective id (must exist in config/objectives/).
    families:       Optional whitelist of family names; None = all catalogued.
    parquet_dir:    Override for the parquet feature cache directory.
    dry_run:        Skip actual training; emit placeholder results.
    max_cpu_workers: Override the OCPU worker count (default: from vm_config).
    ensembles:      If False, skip the Stage-2 ensemble search (base families only).
    manual_spec:    USER-OVERRIDE mode. Path/dict of a manual candidates spec
                    (config/manual_candidates.schema.json). When given, the automated
                    catalog/Model-Scout candidate set is skipped entirely and ONLY the
                    user's listed models (each with its own params + feature selection)
                    are fit and scored against the objective. Mutually exclusive with
                    ``families``. The objective still owns labels/KPI/targets. In this
                    mode the ensemble layer runs only if BOTH ``ensembles`` and the
                    spec's own ``ensembles`` flag are true.
    topology_override:
                    Optional path/dict replacing ``objective["search"]["topology"]``
                    for this run only (WebUI Custom Test). ``mode: none`` disables
                    the objective's meta/output_meta/primary_meta layer. The YAML
                    on disk is never written.
    walk_forward_override:
                    Optional path/dict replacing ``objective["eval"]`` date_range
                    + expanding calendar walk-forward for this run only (WebUI
                    Custom Test designer, ADR-042). The YAML on disk is never written.
    experiment_type:
                    ``custom_test`` | ``manual_override`` | ``model_selection``.
                    ``custom_test`` writes last-fold bundles under
                    ``research/custom_test/<obj>/<run_ts>/models/<cand>/``.

    Returns
    -------
    dict with keys: ``candidates``, ``champion``, ``gate_result``, ``run_ts``.
    """
    from src.common.vm_config import get_vm_config

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    objective = load_objective(objective_id)
    if topology_override is not None:
        objective = _apply_topology_override(objective, topology_override)
    if walk_forward_override is not None:
        objective = _apply_eval_override(objective, walk_forward_override)
    requested = (experiment_type or "").strip() or None
    experiment_type = resolve_experiment_type(
        experiment_type, manual_spec, topology_override)
    if experiment_type == "custom_test" and not requested:
        logger.info("experiment_type inferred custom_test from WebUI spec path")
    from src.search.tf_oof_meta import assert_tf_oof_meta_ready
    assert_tf_oof_meta_ready(objective, experiment_type=experiment_type)
    if parquet_dir:
        pdir = Path(parquet_dir)
    else:
        from src.features.stores import feature_store_id, parquet_dir_for_store
        pdir = parquet_dir_for_store(feature_store_id(objective))
    persist_bundle = experiment_type == "custom_test"
    # research/runs/ is the primary output for every NEW run, whatever the
    # experiment type. research/custom_test/ and research/search/ are read-only
    # history resolved through run_layout.resolve_run_dir(); nothing writes to
    # them any more. The bundle layout inside the run is unchanged, so
    # src/serving/bundle.py, finalize.py and the leaderboard keep working -- only
    # the root moved.
    from src.common.run_layout import new_run_dir
    out_dir = new_run_dir(objective_id, run_ts)
    out_dir.mkdir(parents=True, exist_ok=True)
    from src.training.label_store import write_label_contract_snapshot
    write_label_contract_snapshot(objective, out_dir)
    _copy_custom_test_specs(
        out_dir, manual_spec, topology_override, persist_bundle,
        walk_forward_override=walk_forward_override,
    )

    # W8: tuning is off unless the caller or the objective's search.tune_trials enables it.
    tune_trials = tune_trials or int((objective.get("search") or {}).get("tune_trials", 0))

    # Load or find the latest evaluation contract
    contract = EvalContract.load_latest(objective_id)
    if contract is None:
        logger.warning("No evaluation contract found for %s — scores will not be pinned to a hold-out. "
                       "Run build_contract() first (W2 setup).", objective_id)

    cfg = get_vm_config()
    n_cpu = max_cpu_workers or cfg.workers_for(mem_per_worker_gb=2.0)

    # ── Candidate set: USER-OVERRIDE (manual spec) or automated catalog ──────────
    from src.search.tf_oof_meta import topology_tf_oof_meta
    tf_cfg = topology_tf_oof_meta(objective)

    if manual_spec is not None:
        if families:
            raise ValueError(
                "`families` whitelist and `manual_spec` are mutually exclusive — "
                "manual mode already names the exact models to fit")
        from src.search.manual_spec import load_manual_spec
        ms = load_manual_spec(manual_spec, objective_id=objective_id)
        catalog = ms.entries
        # In manual mode the ensemble layer runs only if the caller AND the spec allow it.
        ensembles = ensembles and ms.ensembles
        print(f"\n🧑‍🔬 USER-OVERRIDE mode — fitting {len(catalog)} user-specified "
              f"candidate(s), skipping automated model selection:")
        for e in catalog:
            feats = e.feature_selection or (e.feature_view and {"view": e.feature_view}) or "full-matrix"
            print(f"     • {e.name:<24} family={e.family_key:<18} features={feats}")
        logger.info("manual_override_active n=%d source=%s ensembles=%s",
                    len(catalog), ms.source, ensembles)
    elif tf_cfg:
        # ADR-093: topology.bases are the candidate set (per native_tf store).
        catalog = []
        ensembles = False
        from src.search.tf_oof_meta import iter_tf_oof_bases, match_tf_oof_family
        print("\n🧭 tf_oof_meta — topology bases (one store each), then OOF meta:")
        n_tf = 0
        for e, store in iter_tf_oof_bases(objective):
            if not match_tf_oof_family(e, store, families):
                print(f"     · skip {e.name:<16} (not in --families)")
                continue
            print(f"     • {e.name:<16} family={e.family_key:<18} native_tf={store}")
            n_tf += 1
        if families and n_tf == 0:
            raise ValueError(
                f"--families {families} matched no tf_oof_meta bases "
                f"(id, family key, or native_tf store)"
            )
    else:
        catalog = get_catalog()
        if families:
            catalog = [e for e in catalog if e.name in families]
        # Objective-level eligibility (names and/or family_group tags).
        elig = objective.get("eligible_model_families") or []
        if elig:
            elig_set = set(elig)
            catalog = [
                e for e in catalog
                if e.name in elig_set or e.family_group in elig_set
            ]
            if not catalog:
                raise ValueError(
                    f"eligible_model_families={elig} matched no catalog entries")

    # ── Execution host (ADR-015) ─────────────────────────────────────────────
    # The GPU VM is the single test-execution host: whenever it is configured, EVERY
    # candidate is dispatched to the Object-Storage job queue and trained on the GPU VM
    # (CPU/tree/statistical families on its cores, DL families on cuda). The CPU VM never
    # trains in-process. In-process execution survives ONLY as a no-GPU dev/dry-run
    # fallback (dry_run, or GPU_INSTANCE_OCID unset).
    gpu_ocid = os.environ.get("GPU_INSTANCE_OCID", "")
    dispatch_all_to_gpu = bool(gpu_ocid) and not dry_run

    if dispatch_all_to_gpu:
        cpu_entries: list[CatalogEntry] = []
        gpu_entries = list(catalog)                 # everything runs on the GPU VM
    else:
        cpu_entries = [e for e in catalog if e.compute_class in ("cpu", "either")]
        gpu_entries = [e for e in catalog if e.compute_class == "gpu"]
        # output_meta must not silently skip DL bases when the GPU OCID is unset.
        from src.search.output_meta import topology_output_meta
        from src.search.meta_stack import topology_meta_stack
        if (topology_output_meta(objective) or topology_meta_stack(objective)) and gpu_entries and not gpu_ocid:
            logger.warning(
                "output_meta/meta_stack: GPU_INSTANCE_OCID unset — running all bases in-process "
                "(set the OCID to dispatch the wave to the GPU VM)")
            cpu_entries = list(catalog)
            gpu_entries = []

    logger.info(
        "search_start",
        extra={"objective_id": objective_id,
               "execution": "gpu_vm" if dispatch_all_to_gpu else "local",
               "n_cpu": len(cpu_entries), "n_gpu": len(gpu_entries),
               "cpu_workers": n_cpu, "dry_run": dry_run},
    )

    # Warm the shared fold cache once so the parallel LOCAL workers memory-map it instead
    # of each reloading parquet + regenerating labels. Skipped when dispatching to the GPU
    # VM — the GPU runner builds its own fold arrays from synced parquet.
    if not dry_run and not dispatch_all_to_gpu:
        import time as _t
        try:
            from src.search.fold_cache import build_or_load_fold_arrays
            if tf_cfg:
                from src.features.stores import parquet_dir_for_store
                from src.search.tf_oof_meta import iter_tf_oof_bases, objective_for_store
                for _e, store in iter_tf_oof_bases(objective):
                    print(f"⏳ Warming fold cache store={store}…")
                    _t0 = _t.time()
                    _arr = build_or_load_fold_arrays(
                        objective_for_store(objective, store),
                        parquet_dir_for_store(store),
                    )
                    if _arr is not None:
                        print(f"✅ Fold cache {store} in {_t.time()-_t0:.0f}s — "
                              f"{len(_arr[1]):,} rows, {len(_arr[3])} features")
            else:
                print("⏳ Warming shared fold cache (features + labels)…")
                _t0 = _t.time()
                _arr = build_or_load_fold_arrays(objective, pdir)
                if _arr is not None:
                    print(f"✅ Fold cache ready in {_t.time()-_t0:.0f}s — {len(_arr[1]):,} rows, "
                          f"{len(_arr[3])} features")
        except Exception as exc:
            logger.warning("fold_cache warm failed: %s", exc)

    all_results: list[CandidateResult] = []

    if tf_cfg:
        print(f"\n⏳ Stage 1 — tf_oof_meta bases ({len(tf_cfg['bases'])} native stores)…")
        all_results.extend(_run_tf_oof_base_layer(
            objective, out_dir, dry_run=dry_run, tune_trials=tune_trials,
            persist_bundle=persist_bundle, search_run_ts=run_ts,
            families=families,
        ))
        cpu_entries = []
        gpu_entries = []

    # ── Stage 1a: CPU families (parallel OCPU) ───────────────────────────────
    if cpu_entries:
        print(f"\n⏳ Stage 1a — evaluating {len(cpu_entries)} CPU families "
              f"(up to {n_cpu} parallel workers)…")
        cpu_results = _run_cpu_batch(
            cpu_entries, objective, pdir, out_dir,
            n_workers=n_cpu, dry_run=dry_run, tune_trials=tune_trials,
            persist_bundle=persist_bundle, search_run_ts=run_ts,
        )
        all_results.extend(cpu_results)

    # ── Stage 1b: GPU VM execution plane (Object-Storage job queue) ──────────
    if gpu_entries:
        _where = "the GPU VM" if dispatch_all_to_gpu else "GPU queue"
        print(f"\n⏳ Stage 1b — dispatching {len(gpu_entries)} families to {_where}…")
        gpu_results = _run_gpu_queue(
            gpu_entries, objective, pdir, out_dir, dry_run=dry_run, tune_trials=tune_trials,
            persist_bundle=persist_bundle, search_run_ts=run_ts,
        )
        all_results.extend(gpu_results)

    # ── Stage 2: Ensemble layer (optional) ────────────────────────────────────
    if ensembles and not dry_run and len(all_results) >= 2:
        print(f"\n⏳ Stage 2 — exhaustive ensemble search over {len(all_results)} base candidates…")
        ens_results = _run_ensemble_layer(
            all_results, objective, out_dir, persist_bundle=persist_bundle)
        all_results.extend(ens_results)
    elif not ensembles:
        print("\n⏭  Stage 2 — ensemble search skipped (--no-ensemble)")

    # ── Stage 2b: Primary + meta topology (ADR-027 / W3) ──────────────────────
    # Cross-fitted trade/no-trade gate on each base OOF when the objective asks for it.
    if not dry_run:
        try:
            from src.search.meta_compose import topology_cfg
            if topology_cfg(objective):
                print("\n⏳ Stage 2b — primary+meta composition (cross-fitted meta gate)…")
                meta_results = _run_meta_layer(
                    all_results, objective, out_dir, pdir, persist_bundle=persist_bundle)
                all_results.extend(meta_results)
        except Exception as exc:  # noqa: BLE001 — meta must never kill the search
            logger.warning("meta_layer_failed: %s", exc)

    # ── Stage 2c: N-base output_meta (ADR-033) ────────────────────────────────
    if not dry_run:
        try:
            from src.search.output_meta import topology_output_meta
            if topology_output_meta(objective):
                print("\n⏳ Stage 2c — output_meta (binary gate on upstream vectors)…")
                om_results = _run_output_meta_layer(
                    all_results, objective, out_dir, pdir, persist_bundle=persist_bundle)
                all_results.extend(om_results)
        except Exception as exc:  # noqa: BLE001 — gate must never kill the search
            logger.warning("output_meta_layer_failed: %s", exc)

    # ── Stage 2d: meta_stack (binary LR on signed base evidence) ─────────────
    if not dry_run:
        try:
            from src.search.meta_stack import topology_meta_stack
            if topology_meta_stack(objective):
                print("\n⏳ Stage 2d — meta_stack (signed-evidence LR, learned per-base weights)…")
                ms_results = _run_meta_stack_layer(
                    all_results, objective, out_dir, persist_bundle=persist_bundle)
                all_results.extend(ms_results)
        except Exception as exc:  # noqa: BLE001 — gate must never kill the search
            logger.warning("meta_stack_layer_failed: %s", exc)

    # ── Stage 2e: native-TF dual-head OOF meta (ADR-066 / ADR-093) ────────────
    if not dry_run:
        try:
            if topology_tf_oof_meta(objective):
                print("\n⏳ Stage 2e — tf_oof_meta (dual-binary XGB on aligned OOF)…")
                tm_results = _run_tf_oof_meta_layer(
                    all_results, objective, out_dir, persist_bundle=persist_bundle)
                all_results.extend(tm_results)
        except Exception as exc:  # noqa: BLE001 — compose must never kill the search
            logger.warning("tf_oof_meta_layer_failed: %s", exc)

    # ── MLflow: one parent run per wave, one nested run per candidate ─────────
    # Sets CandidateResult.mlflow_run_id so the leaderboard rows carry lineage.
    if not dry_run:
        _log_wave_to_mlflow(all_results, objective, run_ts,
                            experiment_type=experiment_type)
    # ── Score + leaderboard ───────────────────────────────────────────────────
    _record_all(all_results, objective, contract, run_ts)

    # ── Champion gate (opt-in YAML; Custom Test never applies it — ADR-071)
    ok = [r for r in all_results if r.status == "ok" and r.score]
    n_failed = sum(1 for r in all_results if r.status == "failed")
    gate_result = resolve_champion_gate(experiment_type, ok, objective)

    fold_meta: list[dict[str, Any]] = []
    fold_split = "expanding_equal_count"
    fold_note = (
        "Walk-forward is equal bar-count, not calendar. "
        "fold_size = n // (n_folds+1); train expands by one block per fold; "
        "remainder after (n_folds+1)*fold_size is unused."
    )
    try:
        ev = objective.get("eval") or {}
        wf = ev.get("walk_forward") or {}
        if str(wf.get("mode") or "") == "expanding_calendar":
            from src.validation.walk_forward import schedule_from_eval
            fold_meta = schedule_from_eval(ev)
            fold_split = "expanding_calendar"
            fold_note = (
                "Walk-forward is expanding calendar windows (ADR-042). "
                "Train1+test1 = train2 until stop; last 120 train bars purged "
                "for the TBM label horizon. Dates after stop are unused (backtest)."
            )
        else:
            from src.search.fold_cache import build_or_load_fold_arrays, describe_walk_forward_folds
            arrays = build_or_load_fold_arrays(objective, pdir)
            if arrays is not None:
                _X, y_full, ts_full, _cols = arrays
                n_folds = int(ev.get("walk_forward_folds", 3))
                fold_meta = describe_walk_forward_folds(ts_full, y_full, n_folds)
    except Exception as exc:  # noqa: BLE001 — metadata must never fail the search
        logger.warning("fold_meta_failed: %s", exc)

    summary: dict[str, Any] = {
        "objective_id": objective_id,
        "run_ts":        run_ts,
        "experiment_type": experiment_type,
        "n_candidates":  len(all_results),
        "n_ok":          len(ok),
        "n_failed":      n_failed,
        "folds":         fold_meta,
        "candidates":    [_public_candidate(r) for r in all_results],
    }
    if gate_result is None:
        summary["status"] = "report"
    else:
        summary["status"] = gate_result.status
        summary["champion"] = gate_result.champion_id
        summary["gate_result"] = asdict(gate_result)
    (out_dir / "search_result.json").write_text(json.dumps(summary, indent=2, default=str))
    if persist_bundle:
        (out_dir / "run.json").write_text(json.dumps(
            custom_test_run_doc(
                objective_id=objective_id, run_ts=run_ts, out_dir=out_dir,
                n_candidates=len(all_results), n_ok=len(ok), n_failed=n_failed,
            ), indent=2, default=str))
    if fold_meta:
        (out_dir / "folds.json").write_text(json.dumps({
            "split": fold_split,
            "n_folds": len(fold_meta),
            "note": fold_note,
            "folds": fold_meta,
        }, indent=2))

    # ── Canonical run manifest (ADR-136) — purely additive ──────────────────
    # Writes manifest.json alongside the artifacts above. Nothing reads it
    # yet and nothing above depends on it; it records only provenance this
    # path genuinely has, and reports evidence_level=INCOMPLETE naming what
    # it does not (no hypothesis, no seed). Best-effort like the writes that
    # follow — a provenance record must never fail a search.
    try:
        from src.search.run_manifest import runtime_datasets, write_manifest
        write_manifest(
            objective_id=objective_id, run_ts=run_ts, out_dir=out_dir,
            objective=objective, experiment_type=experiment_type,
            fold_split=fold_split, fold_meta=fold_meta,
            model_ids=tuple(
                str(r.get("model_id")) for r in summary["candidates"]
                if isinstance(r, dict) and r.get("model_id")
            ),
            # Evidence this runtime genuinely holds, passed in rather than
            # reconstructed from the files just written. _subsample_folds'
            # seed is applied on every search path, so it is recorded as
            # applied; the label contract comes from the in-memory objective,
            # not from re-reading the snapshot this run wrote.
            seeds={"fold_subsample": 42},
            datasets=runtime_datasets(objective),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("canonical_manifest_failed: %s", exc)

    # ── Canonical run layout — future runs land there as part of the run ────
    # Emits research/runs/<objective_id>/<run_ts>/ with one
    # pass_<n>/model_<model_id>/ per fitted candidate: model.json, a compact
    # OOF, metrics, fold metrics and the report set, plus manifest.json. Weight
    # and bundle files are HARD-LINKED, so the canonical tree is complete
    # without a second copy of a Keras weight or a 9 MB booster.
    #
    # The run directory is already research/runs/. emit_canonical_run adds
    # manifest.json and the pass_<n>/model_<id>/ view beside the trainer's
    # models/<candidate>/ bundles. research/custom_test/ is not a write root.
    # Best-effort, like the manifest above: a layout record must never fail
    # a search.
    if not dry_run:
        try:
            from src.search.canonical_run import emit_canonical_run
            # src == dest: the run already lives under research/runs/, so this
            # adds manifest.json and the pass_<n>/model_<id>/ views beside the
            # trainer's own models/<candidate>/ bundles rather than copying a
            # tree. Weights are hard-linked, so the two views share bytes.
            emit_canonical_run(
                out_dir, objective_id=objective_id, run_ts=run_ts,
                link_weights=True, log=logger,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("canonical_run_layout_failed: %s", exc)

    # ── Results index — persist warm-start priors keyed by objective traits (W4) ──
    if not dry_run and gate_result is not None:
        try:
            from src.search.results_index import ResultsIndex
            key = ResultsIndex().record_run(
                objective, gate_result, all_results, run_ts=run_ts, git_sha=_git_sha())
            logger.info("results_index_updated key=%s", key)
        except Exception as exc:  # best-effort — never fail the search on index write
            logger.warning("results_index_record_failed: %s", exc)

    # ── Outcome episodes + pattern reports (success/failure scenario metadata) ──
    # Recorded per candidate so the Research Director's two agents can mine the
    # conditions that separate wins from losses/misses. Best-effort — never fails
    # the search.
    if not dry_run:
        try:
            n_ep = _record_outcome_episodes(
                all_results, objective, pdir, run_ts,
                champion_id=gate_result.champion_id if gate_result else None)
            if n_ep:
                print(f"📊 Outcome episodes + pattern reports recorded for {n_ep} candidate(s)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("outcome_episode_record_failed: %s", exc)

    _print_summary(summary)
    logger.info("search_complete", extra={
        "champion": gate_result.champion_id if gate_result else None,
        "status": summary.get("status"),
        "out_dir": str(out_dir),
    })
    return summary


# ── CPU batch (ProcessPoolExecutor) ──────────────────────────────────────────

def _run_cpu_batch(
    entries: list[CatalogEntry],
    objective: dict[str, Any],
    pdir: Path,
    out_dir: Path,
    n_workers: int,
    dry_run: bool,
    tune_trials: int = 0,
    persist_bundle: bool = False,
    search_run_ts: str | None = None,
) -> list[CandidateResult]:
    results = []
    with ProcessPoolExecutor(max_workers=min(n_workers, len(entries))) as pool:
        futures = {
            pool.submit(
                _run_one, e, objective, pdir, out_dir, dry_run, tune_trials,
                persist_bundle, search_run_ts, str(out_dir),
            ): e
            for e in entries
        }
        for fut in as_completed(futures):
            e = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = CandidateResult(
                    candidate_id=e.name, family=e.name, spec_params={},
                    status="failed", error=traceback.format_exc()[:500],
                    compute_class=e.compute_class,
                )
                logger.error("cpu_candidate_failed name=%s err=%s", e.name, exc)
            results.append(res)
            _progress_line(res)
    return results


# ── GPU queue — dispatch to the GPU VM via Object Storage (Phase 23 W9 / ADR-015) ──

_GPU_WAVE_TIMEOUT_S = int(os.environ.get("VTECH_GPU_WAVE_TIMEOUT_S", str(24 * 3600)))  # wave wall-time cap (s)
_GPU_POLL_S         = int(os.environ.get("VTECH_GPU_POLL_S", "30"))                    # done-marker poll interval


def _run_gpu_queue(
    entries: list[CatalogEntry],
    objective: dict[str, Any],
    pdir: Path,
    out_dir: Path,
    dry_run: bool,
    tune_trials: int = 0,
    persist_bundle: bool = False,
    search_run_ts: str | None = None,
) -> list[CandidateResult]:
    """Run GPU-family candidates on the GPU VM (not in-process on the CPU VM).

    Publishes the content-addressed code bundle, enqueues each candidate to the OCI Queue jobs
    channel, ``ensure_running``’s the GPU VM (START iff STOPPED), long-polls the results channel
    for completion, and downloads OOF + diagnostics from Object Storage. The GPU VM powers itself
    off when idle (ADR-022) — the CPU never issues STOP. dry-run emits placeholders; if no GPU VM
    is configured the candidates are skipped (never trained in-process — pointless on the CPU VM).
    """
    if dry_run:
        return [_run_one(e, objective, pdir, out_dir, True, 0,
                         persist_bundle, search_run_ts or out_dir.name, str(out_dir))
                for e in entries]

    ocid = os.environ.get("GPU_INSTANCE_OCID", "")
    if not ocid:
        results = [CandidateResult(
            candidate_id=e.name, family=e.name, spec_params={}, status="skipped",
            error="GPU VM not configured (GPU_INSTANCE_OCID unset)",
            compute_class=e.compute_class) for e in entries]
        for r in results:
            _progress_line(r)
        return results

    return _dispatch_gpu_jobs(
        entries, objective, pdir, out_dir, ocid, tune_trials,
        persist_bundle=persist_bundle, search_run_ts=search_run_ts,
    )


def _dispatch_gpu_jobs(
    entries: list[CatalogEntry],
    objective: dict[str, Any],
    pdir: Path,
    out_dir: Path,
    ocid: str,
    tune_trials: int = 0,
    persist_bundle: bool = False,
    search_run_ts: str | None = None,
) -> list[CandidateResult]:
    import time
    from src.common.code_bundle import build_and_publish
    from src.gpu_runner.dispatch import enqueue_job, make_plane, results_channel
    from src.gpu_runner.fallback import gpu_ready_or_cpu

    objective_id = objective["objective_id"]

    queue, store = make_plane()
    # Content-addressed code identity (ADR-022): publish the src/config bundle once (skipped if
    # already present) and pin every job to its digest — no git checkout on the GPU VM.
    code_digest = build_and_publish(store, _REPO_ROOT)

    from src.gpu_runner.parquet_sync import push_job_training_data
    try:
        data_payload = push_job_training_data(objective, pdir, store=store)
        s3_prefix = data_payload["data_s3_prefix"]
        logger.info(
            "parquet_push prefix=%s dest=%s labels=%s",
            s3_prefix, data_payload.get("data_s3_dest"),
            data_payload.get("labels_s3_prefix"),
        )
        print(
            f"  ☁ parquet → object store: prefix={s3_prefix} "
            f"dest={data_payload.get('data_s3_dest')}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("parquet_push_failed pdir=%s", pdir)
        err = f"GPU data push failed: {exc}"
        results = [CandidateResult(
            candidate_id=e.name, family=e.family_key,
            spec_params=dict(e.params_default), status="failed", error=err,
            compute_class=e.compute_class,
        ) for e in entries]
        for r in results:
            _progress_line(r)
        return results

    path1s_prefix = pathlabels_prefix = None
    if str((objective.get("search") or {}).get("input") or "") == "path1s":
        from src.common.data_paths import parquet_path1s_dir, parquet_pathlabels_dir
        from src.gpu_runner.parquet_sync import push_features as _push
        path1s_prefix, pathlabels_prefix = "parquet_path1s/", "parquet_pathlabels/"
        try:
            u, s = _push(path1s_prefix, parquet_path1s_dir(), store=store, glob="path_*.parquet")
            print(f"  ☁ path1s → object store: uploaded={u} skipped={s}")
            u, s = _push(pathlabels_prefix, parquet_pathlabels_dir(), store=store,
                         glob="labels_*.parquet")
            print(f"  ☁ pathlabels → object store: uploaded={u} skipped={s}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("path1s_push_failed err=%s", exc)

    # Start the GPU only AFTER data is in object storage. Uploading 1s path
    # caches takes many minutes; the runner's idle-exit (default 300s) would
    # SOFTSTOP the VM before enqueue if we started it first.
    if not gpu_ready_or_cpu(ocid):
        print("⚠️  GPU VM did not start — running these families on the CPU VM in-process "
              "(may be slow for DL/GPU families)…")
        results = [_run_one(
            e, objective, pdir, out_dir, False, tune_trials,
            persist_bundle, search_run_ts or out_dir.name, str(out_dir),
        ) for e in entries]
        for r in results:
            _progress_line(r)
        return results

    pending: dict[str, CatalogEntry] = {}
    for e in entries:
        run_id = f"{objective_id}-{e.name}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        payload: dict[str, Any] = {
            "objective_id": objective_id, "run_id": run_id, "code_digest": code_digest,
            "family": e.family_key, "spec_params": dict(e.params_default),
            "parquet_dir": str(pdir), "tune_trials": tune_trials,
            "persist_bundle": persist_bundle,
            "search_run_ts": search_run_ts or out_dir.name,
            "search_run_dir": str(out_dir),
        }
        # Non-default data dirs (e.g. data/parquet_raw1s) are mirrored in Object Storage
        # under a same-named prefix; the GPU-side job_exec self-syncs the dir when absent.
        # Hive (5s and native TF): prefix is timeframe={tf}/, dest is the hive root
        # (not symbol=NQ).
        payload.update(data_payload)
        if path1s_prefix:
            payload["path1s_s3_prefix"] = path1s_prefix
            payload["pathlabels_s3_prefix"] = pathlabels_prefix
        # USER-OVERRIDE candidates carry a custom id + explicit feature selection + row cap the
        # GPU runner must reconstruct (plugin_family set => manual candidate).
        if e.plugin_family is not None:
            payload["candidate_id"]      = e.name
            payload["feature_selection"] = e.feature_selection
            payload["search_max_rows"]   = e.search_max_rows
            if e.search_space:
                payload["search_space"] = dict(e.search_space)
        ev = objective.get("eval") or {}
        if str((ev.get("walk_forward") or {}).get("mode") or "") == "expanding_calendar":
            payload["eval_overlay"] = {
                "date_range": ev.get("date_range"),
                "walk_forward_folds": ev.get("walk_forward_folds"),
                "walk_forward": ev.get("walk_forward"),
            }
        # Custom Test board store lives on topology.feature_store (even mode: none).
        # job_exec reloads the YAML; without this overlay it falls back to
        # decision_timestep (15s) and intersects 1m names down to a handful of cs_*.
        topo = (objective.get("search") or {}).get("topology")
        if topo:
            payload["topology_overlay"] = dict(topo)
        enqueue_job(queue, payload)
        pending[run_id] = e
    logger.info("gpu_dispatch enqueued=%d digest=%s backend=%s (GPU VM confirmed RUNNING)",
                len(pending), code_digest[:12], queue.backend_kind)

    results: list[CandidateResult] = []
    rchan = results_channel(objective_id)
    deadline = time.time() + _GPU_WAVE_TIMEOUT_S
    while pending and time.time() < deadline:
        msgs = queue.get(channel=rchan, visibility_s=120, max_msgs=10, wait_s=_GPU_POLL_S)
        progressed = False
        for msg in msgs:
            rid = str(msg.body.get("run_id") or "")
            entry = pending.pop(rid, None)
            if entry is None:
                continue                 # another wave's result — leave it for its owner
            res = _collect_gpu_result(store, msg.body, entry, out_dir)
            results.append(res)
            _progress_line(res)
            queue.delete(msg)
            progressed = True
        if pending and not progressed:
            time.sleep(min(_GPU_POLL_S, 2))
    for run_id, e in pending.items():
        r = CandidateResult(candidate_id=e.name, family=e.name, spec_params={},
                            status="failed", error="gpu job timed out",
                            compute_class=e.compute_class)
        results.append(r)
        _progress_line(r)
    return results


def _collect_gpu_result(
    store: Any, body: dict[str, Any], entry: CatalogEntry, out_dir: Path,
) -> CandidateResult:
    from src.gpu_runner.dispatch import collect_result
    req = str(body.get("code_digest") or "")
    ran = str(body.get("ran_digest") or "")
    if req and ran and req != ran:
        logger.warning(
            "DIGEST MISMATCH run_id=%s requested=%s ran=%s — the GPU runner executed a "
            "different code bundle than dispatched", body.get("run_id"), req[:12], ran[:12])
    oof_path, diag = collect_result(store, body, out_dir)
    if body.get("status") != "ok":
        logger.warning("gpu_job_result cid=%s status=%s err=%s",
                       entry.name, body.get("status"), (body.get("error") or "")[:800])
    from src.search.finalize import lab_model_dir, rewrite_lab_manifest
    from src.serving.bundle import is_serving_bundle
    cand_dir = lab_model_dir(out_dir, entry.name)
    nested_oof = cand_dir / "oof.parquet"
    root_oof = out_dir / f"oof_{entry.name}.parquet"
    if nested_oof.exists():
        oof_path = str(nested_oof)
    elif root_oof.exists() and oof_path is None:
        oof_path = str(root_oof)
    if root_oof.exists() and cand_dir.is_dir() and not nested_oof.exists():
        shutil.copy2(root_oof, nested_oof)
        oof_path = str(nested_oof)
    nested_diag = cand_dir / "diagnostics.json"
    if nested_diag.exists() and diag is None:
        try:
            diag = json.loads(nested_diag.read_text())
        except Exception:
            pass
    bundle_path: str | None = None
    if is_serving_bundle(cand_dir):
        rewrite_lab_manifest(cand_dir, search_run_dir=out_dir)
        bundle_path = str(cand_dir)
    # Prefer the EXECUTED config echoed by the job (tuned params) over catalog defaults.
    executed_params = body.get("spec_params") or dict(entry.params_default)
    return CandidateResult(
        candidate_id=entry.name, family=entry.family_key,
        spec_params=executed_params,
        status=body.get("status", "failed"), score=body.get("score") or {},
        oof_path=oof_path, error=body.get("error"), diagnostics=diag,
        compute_class=entry.compute_class,
        elapsed_s=float(body.get("elapsed_s") or 0.0),
        meets_target=bool(body.get("meets_target")),
        feature_selection=body.get("feature_selection")
        if body.get("feature_selection") is not None else entry.feature_selection,
        bundle_path=bundle_path,
    )


def _git_sha() -> str:
    """Resolve the current repo commit for lineage/audit only.

    ADR-022 replaced the git-sha code-identity contract with content-addressed bundles, so there
    is no dispatch-time push/dirty guard anymore — this is a best-effort commit stamp for the
    results index.
    """
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "HEAD"


# ── Single-candidate walk-forward run ────────────────────────────────────────

def _run_one(
    entry: CatalogEntry,
    objective: dict[str, Any],
    pdir: Path,
    out_dir: Path,
    dry_run: bool,
    tune_trials: int = 0,
    persist_bundle: bool = False,
    search_run_ts: str | None = None,
    search_run_dir: str | None = None,
) -> CandidateResult:
    import time
    t0 = time.time()
    spec = ModelSpec(family=entry.family_key, params=dict(entry.params_default))
    from src.features.stores import default_head
    _head = default_head(objective)
    if _head and _head != "multiclass":
        spec.params["head"] = _head

    if dry_run:
        return CandidateResult(
            candidate_id=entry.name, family=entry.family_key,
            spec_params=spec.params, status="ok",
            score={"precision_at_tau": None, "directional_f1": None,
                   "n_signals": 0, "note": "dry_run"},
            compute_class=entry.compute_class, elapsed_s=0.0,
        )

    try:
        plugin = _load_plugin(entry.family_key)
    except ImportError as e:
        return CandidateResult(
            candidate_id=entry.name, family=entry.family_key, spec_params=spec.params,
            status="skipped", error=f"Plugin not yet implemented: {e}",
            compute_class=entry.compute_class, elapsed_s=time.time() - t0,
        )

    # Build walk-forward folds from parquets
    from src.training.job_progress import current, note_status, start_session
    if current() is None:
        # CPU in-process: per-candidate files so ProcessPool workers do not clobber
        # a shared wave progress.json. GPU job_exec already started a session.
        start_session(
            run_id=str(search_run_ts or out_dir.name),
            family=entry.family_key,
            objective_id=str(objective.get("objective_id") or ""),
            paths=[out_dir / f"progress_{entry.name}.json"],
            fold_dirs=[out_dir / "folds" / entry.name],
        )
    note_status("building_folds")
    folds = _build_folds(objective, pdir)
    if not folds:
        note_status("failed", error="No walk-forward folds built from parquet data.")
        return CandidateResult(
            candidate_id=entry.name, family=entry.family_key, spec_params=spec.params,
            status="failed", error="No walk-forward folds built from parquet data.",
            compute_class=entry.compute_class, elapsed_s=time.time() - t0,
        )
    note_status("fitting", n_folds=len(folds), family=entry.family_key)

    # ── Feature selection (W4 / ADR-027): explicit → objective family view →
    # catalog view → objective default_feature_view → full matrix.
    sel  = getattr(entry, "feature_selection", None)
    search_cfg = objective.get("search") or {}
    view = (
        (search_cfg.get("feature_views") or {}).get(entry.family_key)
        or getattr(entry, "feature_view", None)
        or search_cfg.get("default_feature_view")
    )
    try:
        if str(search_cfg.get("input") or "") == "path1s":
            pass  # 1s ATR features; 5s views do not apply
        elif sel:
            from src.models.feature_views import apply_feature_selection
            folds = apply_feature_selection(folds, sel)
        elif view:
            from src.models.feature_views import apply_feature_view
            folds = apply_feature_view(folds, view)
    except Exception as exc:  # noqa: BLE001 — feature selection must not kill the candidate
        logger.warning("feature_selection_failed cid=%s sel=%s view=%s err=%s — using full matrix",
                       entry.name, sel, view, exc)

    u_bars = (objective.get("eval") or {}).get("train_uniqueness_bars")
    if u_bars:
        from src.search.uniqueness import apply_train_uniqueness
        folds = apply_train_uniqueness(folds, int(u_bars))
    if (objective.get("eval") or {}).get("train_onset_uniqueness"):
        # Sequence models keep overlapping 15s TBM labels on the tape
        # (ADR-104). Zeroing onsets starved the LSTM event heads
        # (session p_max 0.39, 0 RTH fires at book 0.40).
        if not bool(getattr(entry, "needs_sequence", False)):
            from src.search.uniqueness import apply_train_onset_uniqueness
            from src.training.dual_binary_15s import label_params_from_objective
            hp = label_params_from_objective(objective)
            folds = apply_train_onset_uniqueness(
                folds,
                horizon_s=float(hp.get("horizon_s") or 600),
                drop_rows=True,
            )

    # ── Search-mode row cap (catalog search_max_rows) ──────────────────────────
    # Subsample train folds so a family can finish in minutes. OOF is never
    # subsampled. Multi-base output_meta / meta_stack still drop the cap so
    # every base trains on the same rows; a single-base topology keeps it
    # (ADR-043 — Custom Test LSTM-only was hitting the full 1.2M-bar path).
    max_rows = _effective_search_max_rows(entry, objective)
    if max_rows and folds and "sessions_train" not in folds[0]:
        folds = _subsample_folds(
            folds, int(max_rows), seed=42,
            contiguous=bool(getattr(entry, "needs_sequence", False)),
        )
        logger.info("search_max_rows cid=%s cap=%d contiguous=%s",
                    entry.name, max_rows, bool(getattr(entry, "needs_sequence", False)))

    # ── Optional W8 inner loop: bounded per-candidate tuning before the gate ──
    tuned_info: dict[str, Any] | None = None
    if tune_trials and getattr(entry, "search_space", None):
        try:
            from src.search.tuning import tune_candidate
            tuned = tune_candidate(entry, objective, folds, n_trials=tune_trials)
            if tuned and tuned.get("best_params"):
                spec = ModelSpec(
                    family=entry.family_key,
                    params={**entry.params_default, **tuned["best_params"]})
                tuned_info = {"tune_trials": tune_trials,
                              "best_value": tuned["best_value"],
                              "best_mode": tuned.get("best_mode")}
                logger.info("tuned cid=%s trials=%d best=%.4f mode=%s", entry.name,
                            tune_trials, tuned["best_value"], tuned.get("best_mode"))
        except Exception as exc:  # noqa: BLE001 — tuning must never fail the candidate
            logger.warning("tuning_failed cid=%s err=%s — using catalog defaults", entry.name, exc)

    model   = plugin.build(spec, objective)
    fitted  = plugin.fit(model, folds, objective=objective)

    # Collect OOF predictions across all folds
    oof_frames = [f.oof_preds for f in [fitted] if f.oof_preds is not None]
    oof_all    = pd.concat(oof_frames) if oof_frames else None

    # Score on the test fold (last fold's oof)
    score: dict[str, Any] = {}
    if oof_all is not None and "direction_true" in oof_all.columns:
        gt = pd.DataFrame({"direction_true": oof_all["direction_true"]})
        cut = fire_tau(objective, report_only=persist_bundle)
        score = canonical_trade_kpi(oof_all, gt, objective, tau=cut)
    if tuned_info:
        score["tuned"] = tuned_info

    # Persist OOF (and last-fold weights for Custom Test) to disk.
    # Root ``oof_<id>.parquet`` / ``diagnostics_<id>.json`` are the GPU handoff
    # glob a stale runner still publishes. Lab bundles under ``models/<id>/``
    # match staging (model + schema + manifest + scaler) and are uploaded by
    # job_exec from the code bundle (ADR-036).
    oof_path: str | None = None
    diag: dict[str, Any] | None = None
    bundle_path: str | None = None
    lab_dir: Path | None = None
    if persist_bundle:
        from src.search.finalize import lab_model_dir, write_serve_bundle
        lab_dir = lab_model_dir(out_dir, entry.name)
        cols = list((folds[-1] or {}).get("feature_cols") or [])
        if fitted.artifact is not None and cols:
            try:
                extra: dict[str, Any] = {"weights_source": "last_walk_forward_fold"}
                for tok in ("15s", "5s", "5m", "1m", "1h", "1d"):
                    if tok in str(entry.name).replace("-", "_").split("_"):
                        extra["native_tf"] = tok
                        break
                if getattr(entry, "needs_sequence", False):
                    extra["kind"] = "sequence"
                    extra["seq_len"] = int((spec.params or {}).get("seq_len") or 128)
                if str((objective.get("labels") or {}).get("encoding") or "") == "dual_binary":
                    sess = objective.get("session") or {}
                    extra["inference_policy"] = {
                        "tau": fire_tau(objective, report_only=True),
                        "trading_start": sess.get("trading_start"),
                        "trading_end": sess.get("trading_end"),
                        "timezone": sess.get("timezone") or "America/Chicago",
                        "sides": ["up", "down"],
                        "note": "dual-binary event cut is labels.fire (0.40 on lg-st), not 3-class 80%.",
                    }
                oid = str(objective.get("objective_id") or out_dir.parent.name)
                write_serve_bundle(
                    lab_dir,
                    artifact=fitted.artifact,
                    feature_cols=cols,
                    objective_id=oid,
                    candidate_id=entry.name,
                    family=entry.family_key,
                    run_ts=str(search_run_ts or out_dir.name),
                    search_run_dir=search_run_dir or out_dir,
                    search_score=score,
                    spec_params=spec.params,
                    feature_selection=getattr(entry, "feature_selection", None),
                    scaler=(fitted.meta or {}).get("scaler"),
                    extra_manifest=extra,
                )
                bundle_path = str(lab_dir)
                logger.info("search_bundle_wrote cid=%s dest=%s", entry.name, lab_dir)
            except Exception as exc:  # noqa: BLE001 — bundle persist must not fail the candidate
                logger.warning("search_bundle_write_failed cid=%s err=%s", entry.name, exc)
                lab_dir.mkdir(parents=True, exist_ok=True)
        else:
            lab_dir.mkdir(parents=True, exist_ok=True)
    if oof_all is not None:
        p_root = out_dir / f"oof_{entry.name}.parquet"
        oof_all.to_parquet(p_root, index=False)
        oof_path = str(p_root)
        diag = _write_diagnostics(
            oof_all, objective, out_dir, entry.name,
            gate_tau=fire_tau(objective, report_only=persist_bundle),
        )
        if lab_dir is not None:
            lab_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p_root, lab_dir / "oof.parquet")
            if diag is not None:
                (lab_dir / "diagnostics.json").write_text(
                    json.dumps(diag, indent=2, default=str))
            oof_path = str(lab_dir / "oof.parquet")

    from src.training.job_progress import finish_session as _finish_progress
    _finish_progress(status="ok")
    return CandidateResult(
        candidate_id=entry.name, family=entry.family_key,
        spec_params=spec.params, status="ok",
        score=score, oof_path=oof_path, diagnostics=diag,
        compute_class=entry.compute_class,
        elapsed_s=time.time() - t0,
        feature_selection=getattr(entry, "feature_selection", None),
        bundle_path=bundle_path,
    )


# ── Meta topology layer (ADR-027) ─────────────────────────────────────────────

def _run_meta_layer(
    base_results: list[CandidateResult],
    objective: dict[str, Any],
    out_dir: Path,
    pdir: Path,
    persist_bundle: bool = False,
) -> list[CandidateResult]:
    """Compose primary+meta candidates from each successful base OOF.

    Context features for the meta model are loaded once from the feature parquet
    (named view from ``search.topology.meta.context_view``) when available; if the
    load fails, the meta gate trains on confidence + side only.
    """
    import time
    from src.search.meta_compose import compose_primary_meta, topology_cfg

    cfg = topology_cfg(objective)
    if not cfg:
        return []

    context = _load_meta_context(pdir, objective, cfg.get("context_view"))
    tau = fire_tau(objective, report_only=persist_bundle)
    out: list[CandidateResult] = []

    for r in base_results:
        # Only compose on *base* families (not ensembles / already-meta).
        if r.status != "ok" or not r.oof_path or not Path(r.oof_path).exists():
            continue
        if r.family.startswith("ensemble_") or r.family.startswith("meta_"):
            continue
        cid = f"meta__{r.candidate_id}"
        t0 = time.time()
        try:
            primary = pd.read_parquet(r.oof_path)
            composed = compose_primary_meta(
                primary,
                context=context,
                approve_threshold=float(cfg.get("approve_threshold", 0.55)),
                min_primary_confidence=float(cfg.get("min_primary_confidence", 0.50)),
                n_folds=int(cfg.get("n_folds", 5)),
            )
            gt = pd.DataFrame({"direction_true": composed["direction_true"]})
            score = canonical_trade_kpi(composed, gt, objective, tau=tau)
            oof_path = out_dir / f"oof_{cid}.parquet"
            # Keep meta_prob / primary_* for analysis — extras beyond canonical are fine.
            composed.to_parquet(oof_path, index=False)
            # All-flat after a hard meta gate has no directional rows — skip brittle sklearn diags.
            n_dir = int(composed["direction"].isin(["up", "down"]).sum()) if len(composed) else 0
            diag = (_write_diagnostics(composed, objective, out_dir, cid, gate_tau=tau)
                    if n_dir > 0 else {"note": "meta_all_flat", "trades_at_gate": {"taken": 0}})
            bundle_path: str | None = None
            if persist_bundle:
                from src.search.finalize import lab_model_dir, write_primary_meta_lab_bundle
                from src.search.meta_compose import fit_final_primary_meta
                lab = lab_model_dir(out_dir, cid)
                primary_dir = Path(r.bundle_path) if r.bundle_path else lab_model_dir(
                    out_dir, r.candidate_id)
                try:
                    fitted = fit_final_primary_meta(
                        primary, context=context,
                        min_primary_confidence=float(cfg.get("min_primary_confidence", 0.50)),
                    )
                    write_primary_meta_lab_bundle(
                        lab,
                        primary_bundle_dir=primary_dir,
                        primary_id=str(r.candidate_id),
                        fitted_meta=fitted,
                        composed_oof=composed,
                        objective_id=str(objective.get("objective_id") or out_dir.parent.name),
                        candidate_id=cid,
                        run_ts=out_dir.name,
                        search_run_dir=out_dir,
                        search_score=score,
                        spec_params={
                            "primary": r.candidate_id,
                            "bases": [r.candidate_id],
                            "meta": cfg,
                            "primary_params": r.spec_params,
                        },
                        diagnostics=diag if isinstance(diag, dict) else None,
                        context_view=cfg.get("context_view"),
                    )
                    bundle_path = str(lab)
                    oof_path = lab / "oof.parquet"
                except Exception as exc:  # noqa: BLE001
                    logger.warning("primary_meta_bundle_write_failed primary=%s err=%s",
                                   r.candidate_id, exc)
                    lab.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out_dir / f"oof_{cid}.parquet", lab / "oof.parquet")
                    if diag is not None:
                        (lab / "diagnostics.json").write_text(
                            json.dumps(diag, indent=2, default=str))
            res = CandidateResult(
                candidate_id=cid,
                family=f"meta_{cfg.get('family', 'xgboost')}",
                spec_params={
                    "primary": r.candidate_id,
                    "meta": cfg,
                    "primary_params": r.spec_params,
                },
                status="ok", score=score, oof_path=str(oof_path),
                diagnostics=diag, compute_class="cpu",
                elapsed_s=time.time() - t0,
                feature_selection=r.feature_selection,
                bundle_path=bundle_path,
            )
        except Exception as exc:
            res = CandidateResult(
                candidate_id=cid,
                family=f"meta_{cfg.get('family', 'xgboost')}",
                spec_params={"primary": r.candidate_id, "meta": cfg},
                status="failed", error=traceback.format_exc()[:500],
                compute_class="cpu", elapsed_s=time.time() - t0,
            )
            logger.warning("meta_compose_failed primary=%s err=%s", r.candidate_id, exc)
        out.append(res)
        _progress_line(res)

    return out


def _run_tf_oof_base_layer(
    objective: dict[str, Any],
    out_dir: Path,
    *,
    dry_run: bool,
    tune_trials: int,
    persist_bundle: bool,
    search_run_ts: str | None,
    families: list[str] | None = None,
) -> list[CandidateResult]:
    """Fit each ``tf_oof_meta`` base on its native-TF store (ADR-093 / ADR-094).

    Sequence/GPU families (15s LSTM) dispatch to the A10. Tree bases stay
    local so 1m/5m XGB does not take the GPU round-trip.
    ``families`` restricts by candidate id, family key, or native_tf store.
    """
    from src.features.stores import parquet_dir_for_store
    from src.search.tf_oof_meta import (
        iter_tf_oof_bases, match_tf_oof_family, objective_for_store,
    )

    pairs = [
        (e, s) for e, s in iter_tf_oof_bases(objective)
        if match_tf_oof_family(e, s, families)
    ]
    if families and not pairs:
        raise ValueError(
            f"--families {families} matched no tf_oof_meta bases"
        )
    gpu_jobs = [(e, s) for e, s in pairs if e.compute_class == "gpu"]
    cpu_jobs = [(e, s) for e, s in pairs if e.compute_class != "gpu"]
    results: list[CandidateResult] = []

    gpu_ocid = os.environ.get("GPU_INSTANCE_OCID", "")
    for entry, store in gpu_jobs:
        obj_b = objective_for_store(objective, store)
        pdir_b = parquet_dir_for_store(store)
        if dry_run or not gpu_ocid:
            host = "cpu" if not gpu_ocid else "dry"
            if not gpu_ocid and not dry_run:
                print("⚠️  GPU_INSTANCE_OCID unset — 15s LSTM in-process (slow)")
            print(f"   → {entry.name}  store={store}  family={entry.family_key}  host={host}")
            res = _run_one(
                entry, obj_b, pdir_b, out_dir, dry_run, tune_trials,
                persist_bundle, search_run_ts, str(out_dir),
            )
            _progress_line(res)
            results.append(res)
        else:
            print(f"   → {entry.name}  store={store}  family={entry.family_key}  host=gpu")
            got = _run_gpu_queue(
                [entry], obj_b, pdir_b, out_dir, dry_run, tune_trials,
                persist_bundle=persist_bundle, search_run_ts=search_run_ts,
            )
            results.extend(got)

    for entry, store in cpu_jobs:
        obj_b = objective_for_store(objective, store)
        pdir_b = parquet_dir_for_store(store)
        print(f"   → {entry.name}  store={store}  family={entry.family_key}  host=cpu")
        res = _run_one(
            entry, obj_b, pdir_b, out_dir, dry_run, tune_trials,
            persist_bundle, search_run_ts, str(out_dir),
        )
        _progress_line(res)
        results.append(res)
    return results


def _run_tf_oof_meta_layer(
    base_results: list[CandidateResult],
    objective: dict[str, Any],
    out_dir: Path,
    persist_bundle: bool = False,
) -> list[CandidateResult]:
    """Dual-binary XGB meta on aligned native-TF OOF (ADR-092 / ADR-093)."""
    import time
    from src.evaluation._kpi import canonical_trade_kpi
    from src.evaluation.kpi_criteria import fire_tau
    from src.search.tf_oof_meta import (
        compose, fit_final, load_htf_context, topology_tf_oof_meta,
        write_tf_oof_meta_serving_bundle,
    )
    from src.validation.walk_forward import schedule_from_eval

    cfg = topology_tf_oof_meta(objective)
    if not cfg:
        return []
    bases = list(cfg["bases"])
    ids = [str(b["id"]) for b in bases]
    by_id: dict[str, CandidateResult] = {}
    for r in base_results:
        if r.status != "ok" or not r.oof_path:
            continue
        if r.candidate_id in ids:
            by_id[r.candidate_id] = r
    missing = [i for i in ids if i not in by_id]
    if missing:
        logger.warning("tf_oof_meta skipped — missing base OOF: %s", missing)
        return []

    t0 = time.time()
    cid = "tf_oof_meta__" + "_".join(ids)
    meta_cfg = cfg["meta"]
    try:
        oofs = {i: pd.read_parquet(by_id[i].oof_path) for i in ids}
        grid = next((b["id"] for b in bases if b.get("native_tf") == cfg.get("decision_timestep")), ids[0])
        ts0 = pd.to_datetime(oofs[grid]["ts"], utc=True)
        context = load_htf_context(ts0.min(), ts0.max())
        fold_starts = None
        ev = objective.get("eval") or {}
        if str((ev.get("walk_forward") or {}).get("mode") or "") == "expanding_calendar":
            starts = []
            for row in schedule_from_eval(ev):
                oof = (row or {}).get("oof") or {}
                if oof.get("start"):
                    ts = pd.Timestamp(oof["start"])
                    starts.append(ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC"))
            fold_starts = starts or None
        from src.models.dual_binary import dual_binary_fire
        composed = compose(
            oofs, bases=bases,
            decision_timestep=str(cfg.get("decision_timestep") or "15s"),
            context=context,
            n_folds=int(meta_cfg.get("n_folds") or 3),
            embargo_bars=int(meta_cfg.get("embargo_bars") or 40),
            fold_starts=fold_starts,
            fire=dual_binary_fire(objective),
        )
        scored = composed[composed["meta_finite"]] if "meta_finite" in composed.columns else composed
        tau = fire_tau(objective, report_only=persist_bundle)
        score = canonical_trade_kpi(
            scored, pd.DataFrame({"direction_true": scored["direction_true"]}),
            objective, tau=tau,
        ) if len(scored) else {}
        oof_path = out_dir / f"oof_{cid}.parquet"
        composed.to_parquet(oof_path, index=False)
        n_dir = int(scored["direction"].isin(["up", "down"]).sum()) if len(scored) else 0
        diag = (_write_diagnostics(scored, objective, out_dir, cid, gate_tau=tau)
                if n_dir > 0 else {"note": "tf_oof_meta_all_flat", "trades_at_gate": {"taken": 0}})
        bundle_path = None
        if persist_bundle:
            from src.search.finalize import lab_model_dir
            dest = lab_model_dir(out_dir, cid)
            last_block = None
            if "meta_finite" in composed.columns and bool(composed["meta_finite"].any()):
                last_block = int(composed.loc[composed["meta_finite"], "meta_block"].max())
            fitted = fit_final(
                oofs, bases=bases,
                decision_timestep=str(cfg.get("decision_timestep") or "15s"),
                context=context,
                exclude_blocks=[last_block] if last_block is not None else None,
                n_folds=int(meta_cfg.get("n_folds") or 3),
                fold_starts=fold_starts,
            )
            base_dirs = {i: lab_model_dir(out_dir, i) for i in ids}
            write_tf_oof_meta_serving_bundle(
                dest,
                objective=objective,
                candidate_id=cid,
                run_ts=out_dir.name,
                search_run_dir=out_dir,
                bases=bases,
                base_bundle_dirs=base_dirs,
                fitted=fitted,
                composed_oof=composed,
                search_score=score,
                diagnostics=diag if isinstance(diag, dict) else None,
                spec_params={"bases": bases, "meta": meta_cfg,
                             "base_specs": {i: by_id[i].spec_params for i in ids}},
            )
            bundle_path = str(dest)
            oof_path = dest / "oof.parquet"
        res = CandidateResult(
            candidate_id=cid,
            family="tf_oof_meta",
            spec_params={"bases": bases, "meta": meta_cfg,
                         "base_specs": {i: by_id[i].spec_params for i in ids}},
            status="ok", score=score, oof_path=str(oof_path),
            diagnostics=diag, compute_class="cpu",
            elapsed_s=time.time() - t0, bundle_path=bundle_path,
        )
    except Exception as exc:
        logger.warning("tf_oof_meta_failed: %s", exc)
        res = CandidateResult(
            candidate_id=cid, family="tf_oof_meta",
            spec_params={"bases": bases, "meta": meta_cfg},
            status="failed", error=str(exc)[:500],
            compute_class="cpu", elapsed_s=time.time() - t0,
        )
    _progress_line(res)
    return [res]


def _run_output_meta_layer(
    base_results: list[CandidateResult],
    objective: dict[str, Any],
    out_dir: Path,
    pdir: Path,
    persist_bundle: bool = False,
) -> list[CandidateResult]:
    """Binary trade/no-trade gate over topology.bases OOF (ADR-033)."""
    import time
    from src.evaluation._kpi import canonical_trade_kpi
    from src.evaluation.kpi_criteria import fire_tau
    from src.search.output_meta import compose, topology_output_meta

    cfg = topology_output_meta(objective)
    if not cfg:
        return []
    bases = cfg["bases"]
    meta_cfg = cfg["meta"]
    by_family: dict[str, CandidateResult] = {}
    for r in base_results:
        if r.status != "ok" or not r.oof_path:
            continue
        if r.family.startswith("ensemble_") or r.family.startswith("meta_") or r.family == "output_meta":
            continue
        by_family.setdefault(r.family, r)
    missing = [b for b in bases if b not in by_family]
    if missing:
        logger.warning("output_meta skipped — missing base OOF: %s", missing)
        return []

    t0 = time.time()
    cid = "meta__" + "_".join(bases)
    try:
        oofs = {b: pd.read_parquet(by_family[b].oof_path) for b in bases}
        context = _load_meta_context(pdir, objective, meta_cfg.get("context_view"))
        composed = compose(
            oofs, bases=bases,
            approve_threshold=float(meta_cfg.get("approve_threshold", 0.55)),
            n_folds=int(meta_cfg.get("n_folds", 5)),
            embargo_bars=int(meta_cfg.get("embargo_bars", 0)),
            context=context,
            meta_family=str(meta_cfg.get("family") or "logistic_regression"),
        )
        gt = pd.DataFrame({"direction_true": composed["direction_true"]})
        tau = fire_tau(objective, report_only=persist_bundle)
        score = canonical_trade_kpi(composed, gt, objective, tau=tau)
        oof_path = out_dir / f"oof_{cid}.parquet"
        composed.to_parquet(oof_path, index=False)
        n_dir = int(composed["direction"].isin(["up", "down"]).sum()) if len(composed) else 0
        diag = (_write_diagnostics(composed, objective, out_dir, cid, gate_tau=tau)
                if n_dir > 0 else {"note": "meta_all_flat", "trades_at_gate": {"taken": 0}})
        res = CandidateResult(
            candidate_id=cid,
            family="output_meta",
            spec_params={
                "bases": bases,
                "meta": meta_cfg,
                "base_specs": {b: by_family[b].spec_params for b in bases},
                "base_feature_selection": {b: by_family[b].feature_selection for b in bases},
            },
            status="ok", score=score, oof_path=str(oof_path),
            diagnostics=diag, compute_class="cpu",
            elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        logger.warning("output_meta_failed: %s", exc)
        res = CandidateResult(
            candidate_id=cid, family="output_meta",
            spec_params={"bases": bases, "meta": meta_cfg},
            status="failed", error=str(exc)[:500],
            compute_class="cpu", elapsed_s=time.time() - t0,
        )
    _progress_line(res)
    return [res]


def _run_meta_stack_layer(
    base_results: list[CandidateResult],
    objective: dict[str, Any],
    out_dir: Path,
    persist_bundle: bool = False,
) -> list[CandidateResult]:
    """Binary LR on signed per-base evidence (meta_stack topology).

    Emits a single candidate ``meta_stack__<bases>`` scored on the same canonical
    KPI contract as the base learners.
    """
    import time
    from src.evaluation._kpi import canonical_trade_kpi
    from src.evaluation.kpi_criteria import fire_tau
    from src.search.meta_stack import compose, topology_meta_stack

    cfg = topology_meta_stack(objective)
    if not cfg:
        return []
    bases = cfg["bases"]
    meta_cfg = cfg["meta"]

    by_family: dict[str, CandidateResult] = {}
    for r in base_results:
        if r.status != "ok" or not r.oof_path:
            continue
        if r.family.startswith("ensemble_") or r.family.startswith("meta_") or r.family in (
            "output_meta", "meta_stack",
        ):
            continue
        by_family.setdefault(r.family, r)
    missing = [b for b in bases if b not in by_family]
    if missing:
        logger.warning("meta_stack skipped — missing base OOF: %s", missing)
        return []

    t0 = time.time()
    cid = "meta_stack__" + "_".join(bases)
    try:
        oofs = {b: pd.read_parquet(by_family[b].oof_path) for b in bases}
        composed = compose(
            oofs, bases=bases,
            scheme=str(meta_cfg.get("scheme", "walk_forward_OOF")),
            feature=str(meta_cfg.get("feature", "linear_weights")),
            approve_threshold=float(meta_cfg.get("approve_threshold", 0.55)),
            embargo_bars=int(meta_cfg.get("embargo_bars", 0)),
            n_folds=int(meta_cfg.get("n_folds", 5)),
        )
        gt = pd.DataFrame({"direction_true": composed["direction_true"]})
        tau = fire_tau(objective, report_only=persist_bundle)
        score = canonical_trade_kpi(composed, gt, objective, tau=tau)
        spec_params = {
            "bases": bases,
            "meta": meta_cfg,
            "base_specs": {b: by_family[b].spec_params for b in bases},
            "base_feature_selection": {b: by_family[b].feature_selection for b in bases},
        }
        n_dir = int(composed["direction"].isin(["up", "down"]).sum()) if len(composed) else 0
        bundle_path: str | None = None
        if persist_bundle:
            from src.search.finalize import lab_model_dir, write_meta_stack_lab_bundle
            dest = lab_model_dir(out_dir, cid)
            diag: dict[str, Any] | None
            if n_dir > 0:
                try:
                    diag = compute_diagnostics(
                        composed, pd.DataFrame({"direction_true": composed["direction_true"]}),
                        objective, gate_tau=tau)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("meta_stack_diag_failed: %s", exc)
                    diag = None
            else:
                diag = {"note": "meta_stack_all_flat", "trades_at_gate": {"taken": 0}}
            base_dirs: dict[str, Path] = {}
            for b in bases:
                bp = by_family[b].bundle_path
                if bp and Path(bp).is_dir():
                    base_dirs[b] = Path(bp)
                else:
                    cand = lab_model_dir(out_dir, by_family[b].candidate_id)
                    if cand.is_dir():
                        base_dirs[b] = cand
            try:
                write_meta_stack_lab_bundle(
                    dest,
                    base_bundle_dirs=base_dirs,
                    composed_oof=composed,
                    objective_id=str(objective.get("objective_id") or out_dir.parent.name),
                    candidate_id=cid,
                    run_ts=out_dir.name,
                    search_run_dir=out_dir,
                    search_score=score,
                    spec_params=spec_params,
                    diagnostics=diag if isinstance(diag, dict) else None,
                )
                bundle_path = str(dest)
                oof_path = dest / "oof.parquet"
            except Exception as exc:  # noqa: BLE001
                logger.warning("meta_stack_bundle_write_failed: %s", exc)
                dest.mkdir(parents=True, exist_ok=True)
                oof_path = dest / "oof.parquet"
                composed.to_parquet(oof_path, index=False)
        else:
            oof_path = out_dir / f"oof_{cid}.parquet"
            composed.to_parquet(oof_path, index=False)
            diag = (_write_diagnostics(composed, objective, out_dir, cid, gate_tau=tau)
                    if n_dir > 0 else {"note": "meta_stack_all_flat", "trades_at_gate": {"taken": 0}})
        res = CandidateResult(
            candidate_id=cid,
            family="meta_stack",
            spec_params=spec_params,
            status="ok", score=score, oof_path=str(oof_path),
            diagnostics=diag, compute_class="cpu",
            elapsed_s=time.time() - t0,
            bundle_path=bundle_path,
        )
    except Exception as exc:  # noqa: BLE001 — meta must never kill the search
        logger.warning("meta_stack_failed: %s", exc)
        res = CandidateResult(
            candidate_id=cid, family="meta_stack",
            spec_params={"bases": bases, "meta": meta_cfg},
            status="failed", error=str(exc)[:500],
            compute_class="cpu", elapsed_s=time.time() - t0,
        )
    _progress_line(res)
    return [res]


def _load_meta_context(
    pdir: Path,
    objective: dict[str, Any],
    context_view: str | None,
) -> pd.DataFrame | None:
    """Load a sparse context feature frame (ts-indexed) for the meta gate.

    Best-effort: returns None if parquets are missing or the view selects nothing.
    Caps to a recent window of days to bound memory.
    """
    if not context_view:
        return None
    try:
        from src.models.feature_views import FEATURE_VIEWS
    except Exception:
        return None
    if context_view not in FEATURE_VIEWS:
        logger.warning("meta context_view '%s' unknown — confidence-only meta", context_view)
        return None

    files = sorted(pdir.glob("features_*.parquet"))
    if not files:
        # Program-layout fallback used by some caches.
        files = sorted(pdir.glob("**/features_*.parquet"))
    if not files:
        return None
    # Cap: last ~90 feature days is enough for meta context join.
    files = files[-90:]
    prefixes = FEATURE_VIEWS[context_view]
    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if "ts" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            # Some parquets use the index as ts.
            df = df.reset_index()
            if "ts" not in df.columns and "index" in df.columns:
                df = df.rename(columns={"index": "ts"})
        if "ts" not in df.columns:
            continue
        cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
        if not cols:
            continue
        sub = df[["ts"] + cols].copy()
        sub["ts"] = pd.to_datetime(sub["ts"], utc=True)
        frames.append(sub)
    if not frames:
        return None
    ctx = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts"], keep="last")
    ctx = ctx.set_index("ts").sort_index()
    logger.info("meta_context loaded view=%s rows=%d cols=%d",
                context_view, len(ctx), ctx.shape[1])
    return ctx


# ── Ensemble layer ────────────────────────────────────────────────────────────

def _run_ensemble_layer(
    base_results: list[CandidateResult],
    objective: dict[str, Any],
    out_dir: Path,
    persist_bundle: bool = False,
) -> list[CandidateResult]:
    """Combine base OOF predictions via the W6 ensemble grid (consensus / stacking / vote).

    Bounded by ``objective.ensemble.max_base`` (top-k base learners by primary precision),
    enumerated over all pairs + triples + all-N, each crossed with the configured methods.
    Reuses cached OOF only — no base model is retrained. Every ensemble is scored on the same
    frozen contract as the base models.
    """
    import time
    from src.search import ensembles as ens

    ens_results: list[CandidateResult] = []

    # Load available OOF frames + their base primary precision (for top-k selection).
    oof_map: dict[str, pd.DataFrame] = {}
    prec_map: dict[str, float] = {}
    for r in base_results:
        if r.status == "ok" and r.oof_path and Path(r.oof_path).exists():
            try:
                oof_map[r.candidate_id] = pd.read_parquet(r.oof_path)
                prec_map[r.candidate_id] = float((r.score or {}).get("precision_at_tau") or -1.0)
            except Exception:
                pass

    if len(oof_map) < 2:
        logger.info("ensemble_skipped: fewer than 2 OOF frames available")
        return ens_results

    ecfg = objective.get("ensemble") or {}
    max_base = int(ecfg.get("max_base", ens.DEFAULT_MAX_BASE))
    methods = tuple(ecfg.get("methods", ens.DEFAULT_METHODS))
    min_agree = ecfg.get("min_agree")               # None → simple majority

    plans = ens.plan_ensembles(sorted(oof_map), prec_map, max_base=max_base, methods=methods)
    logger.info("ensemble_plan n=%d max_base=%d methods=%s", len(plans), max_base, methods)
    tau = fire_tau(objective, report_only=persist_bundle)

    for plan in plans:
        method, members = plan["method"], plan["members"]
        cid = f"{method}__" + "__".join(members)
        t0 = time.time()
        try:
            frames = [oof_map[m] for m in members]
            merged = ens.combine(method, frames, min_agree=min_agree)
            gt = pd.DataFrame({"direction_true": merged["direction_true"]})
            score = canonical_trade_kpi(merged, gt, objective, tau=tau)
            oof_path = out_dir / f"oof_{cid}.parquet"
            merged.to_parquet(oof_path, index=False)
            diag = _write_diagnostics(merged, objective, out_dir, cid, gate_tau=tau)
            res = CandidateResult(
                candidate_id=cid, family=f"ensemble_{method}",
                spec_params={"members": members, "method": method},
                status="ok", score=score, oof_path=str(oof_path),
                diagnostics=diag, compute_class="cpu", elapsed_s=time.time() - t0,
            )
        except Exception as exc:
            res = CandidateResult(
                candidate_id=cid, family=f"ensemble_{method}",
                spec_params={"members": members, "method": method},
                status="failed", error=traceback.format_exc()[:500],
                compute_class="cpu", elapsed_s=time.time() - t0,
            )
            logger.warning("ensemble_failed cid=%s err=%s", cid, exc)
        ens_results.append(res)
        _progress_line(res)

    return ens_results


# ── Helpers ───────────────────────────────────────────────────────────────────

_MAX_EPISODE_CANDIDATES = 16     # bound the per-run episode/pattern writes


def _record_outcome_episodes(
    results: list[CandidateResult],
    objective: dict[str, Any],
    pdir: Path,
    run_ts: str,
    *,
    champion_id: str | None = None,
) -> int:
    """Build + persist success/failure episodes and mined outcome reports.

    The champion (if any) is always included; remaining ok candidates are added in
    leaderboard order up to ``_MAX_EPISODE_CANDIDATES``. Context features come from the
    shared fold cache (memory-mapped — cheap). Returns the number of candidates recorded.
    """
    from src.evaluation.episodes import build_episodes, record_episodes
    from src.evaluation.patterns import analyze_and_write
    from src.search.fold_cache import build_or_load_fold_arrays

    ok = [r for r in results
          if r.status == "ok" and r.oof_path and Path(r.oof_path).exists()]
    if not ok:
        return 0
    ok.sort(key=lambda r: (r.candidate_id != champion_id,
                           -float((r.score or {}).get("precision_at_tau") or -1.0)))
    ok = ok[:_MAX_EPISODE_CANDIDATES]

    arrays = build_or_load_fold_arrays(objective, pdir)
    X, ts_arr, feature_cols = (None, None, None)
    if arrays is not None:
        X, _, ts_arr, feature_cols = arrays

    objective_id = objective["objective_id"]
    n_recorded = 0
    for r in ok:
        try:
            oof = pd.read_parquet(r.oof_path)
            episodes = build_episodes(
                oof, X=X, ts=ts_arr, feature_cols=feature_cols, objective=objective)
            if episodes is None or episodes.empty:
                continue
            record_episodes(objective_id, r.candidate_id, run_ts, episodes, kind="model")
            analyze_and_write(
                objective_id, r.candidate_id, run_ts, episodes,
                meta={"kind": "model", "family": r.family,
                      "is_champion": r.candidate_id == champion_id,
                      "score": {k: v for k, v in (r.score or {}).items()
                                if isinstance(v, (int, float, str, bool))}})
            n_recorded += 1
        except Exception as exc:  # noqa: BLE001 — per-candidate best-effort
            logger.warning("episode_build_failed cid=%s err=%s", r.candidate_id, exc)
    return n_recorded


def _write_diagnostics(
    oof: pd.DataFrame,
    objective: dict[str, Any],
    out_dir: Path,
    name: str,
    dest: Path | None = None,
    gate_tau: float | None | object = _LOOKUP_TAU,
) -> dict[str, Any] | None:
    """Persist a full diagnostics report (confusion matrices, per-class F1,
    tau sweep, type-1 breakdown) next to the candidate's OOF parquet and
    return the diagnostics dict (consumed in-process by the champion gate)."""
    if oof is None or "direction_true" not in oof.columns:
        return None
    try:
        diag = compute_diagnostics(
            oof, pd.DataFrame({"direction_true": oof["direction_true"]}), objective,
            gate_tau=gate_tau,
        )
        path = dest if dest is not None else out_dir / f"diagnostics_{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diag, indent=2, default=str))
        return diag
    except Exception as exc:
        logger.warning("diagnostics_write_failed name=%s err=%s", name, exc)
        return None


def _copy_custom_test_specs(
    out_dir: Path,
    manual_spec: str | Path | dict[str, Any] | None,
    topology_override: str | Path | dict[str, Any] | None,
    persist: bool,
    walk_forward_override: str | Path | dict[str, Any] | None = None,
) -> None:
    """Copy the compiled board spec into a Custom Test run dir."""
    if not persist:
        return
    try:
        if isinstance(manual_spec, (str, Path)) and Path(manual_spec).is_file():
            shutil.copy2(manual_spec, out_dir / "spec.manual.yaml")
        elif isinstance(manual_spec, dict):
            (out_dir / "spec.manual.yaml").write_text(
                yaml.safe_dump(manual_spec, sort_keys=False))
        if isinstance(topology_override, (str, Path)) and Path(topology_override).is_file():
            shutil.copy2(topology_override, out_dir / "topology.yaml")
        elif isinstance(topology_override, dict):
            (out_dir / "topology.yaml").write_text(
                yaml.safe_dump(topology_override, sort_keys=False))
        if isinstance(walk_forward_override, (str, Path)) and Path(walk_forward_override).is_file():
            shutil.copy2(walk_forward_override, out_dir / "walk_forward.yaml")
        elif isinstance(walk_forward_override, dict):
            (out_dir / "walk_forward.yaml").write_text(
                yaml.safe_dump(walk_forward_override, sort_keys=False))
    except Exception as exc:  # noqa: BLE001
        logger.warning("custom_test_spec_copy_failed: %s", exc)


def _public_candidate(r: CandidateResult) -> dict[str, Any]:
    """Serialize a CandidateResult without the heavy diagnostics blob
    (persisted separately as diagnostics_<name>.json)."""
    d = asdict(r)
    d.pop("diagnostics", None)
    return d


def _load_plugin(name: str):
    """Import and instantiate the plugin for *name*.  Raises ImportError if not yet built."""
    registry = {
        # GPU — sequence DL
        "attention_lstm":    ("src.models.attention_lstm_plugin", "AttentionLSTMPlugin"),
        "tcn":               ("src.models.tcn_plugin",            "TCNPlugin"),
        "tcn_close":         ("src.models.tcn_plugin",            "TCNPlugin"),
        "tcn_hazard":        ("src.models.tcn_hazard_plugin",     "TCNHazardPlugin"),
        "path_tcn":          ("src.models.path_tcn_plugin",       "PathTCNPlugin"),
        "patch_hazard":      ("src.models.patch_hazard_plugin",   "PatchHazardPlugin"),
        "transformer":       ("src.models.transformer_plugin",    "TransformerPlugin"),
        "cnn":               ("src.models.cnn_plugin",            "CNNPlugin"),
        "rl":                ("src.models.rl_plugin",             "RLPlugin"),
        # CPU — tree ensembles + linear
        "xgboost":           ("src.models.tabular_plugins",       "XGBoostPlugin"),
        "lightgbm":          ("src.models.tabular_plugins",       "LightGBMPlugin"),
        "random_forest":     ("src.models.tabular_plugins",       "RandomForestPlugin"),
        "logistic_regression": ("src.models.tabular_plugins",     "LogisticRegressionPlugin"),
        "svm":               ("src.models.svm_plugin",            "SVMPlugin"),
        # CPU — unsupervised regime
        "hmm":               ("src.models.regime_plugins",        "HMMPlugin"),
        "gmm":               ("src.models.regime_plugins",        "GMMPlugin"),
        "kmeans":            ("src.models.clustering_plugins",    "KMeansPlugin"),
        "dbscan":            ("src.models.clustering_plugins",    "DBSCANPlugin"),
        # GPU — reinforcement learning
        "rl":                ("src.models.rl_plugin",             "RLPlugin"),
    }
    if name not in registry:
        raise ImportError(f"No plugin registered for family '{name}'")
    module_path, class_name = registry[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


# Feature bar resolution for the parquet cache (5-second bars).
_FEATURE_TIMESTEP_SECONDS = 5


def _parse_duration_to_seconds(s: str) -> int:
    """Parse a human duration string (``'10m'``, ``'5s'``, ``'2h'``) to seconds."""
    s = str(s).strip().lower()
    if s.endswith("m"):
        return int(s[:-1]) * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    if s.endswith("s"):
        return int(s[:-1])
    try:
        return int(s)          # assume seconds if bare integer
    except ValueError:
        return 600             # fallback: 10 minutes


def _topology_n_bases(objective: dict[str, Any]) -> int:
    """How many topology bases the objective names (0 if none / not a join mode)."""
    from src.search.meta_stack import topology_meta_stack
    from src.search.output_meta import topology_output_meta
    om = topology_output_meta(objective)
    if om:
        return len(om.get("bases") or [])
    ms = topology_meta_stack(objective)
    if ms:
        return len(ms.get("bases") or [])
    return 0


def _effective_search_max_rows(entry: CatalogEntry, objective: dict[str, Any]) -> int | None:
    """Honor ``search_max_rows`` unless multiple topology bases must share train rows."""
    override = (objective.get("search") or {}).get("max_train_rows")
    if override == 0:
        return None
    if override is not None:
        return int(override)
    max_rows = getattr(entry, "search_max_rows", None)
    if not max_rows:
        return None
    if _topology_n_bases(objective) > 1:
        return None
    return int(max_rows)


def _subsample_folds(
    folds: list[dict[str, Any]],
    max_rows: int,
    seed: int = 42,
    *,
    contiguous: bool = False,
) -> list[dict[str, Any]]:
    """Cap each fold's training set to at most *max_rows*.

    The OOF (test) window is **not** subsampled so scoring metrics remain
    comparable across families. Default is a reproducible random subsample
    (seed=42). Sequence families pass ``contiguous=True`` and take the **last**
    *max_rows* train bars so LSTM/TCN windows stay consecutive 5s bars.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for f in folds:
        g = dict(f)
        n_train = len(f["X_train"])
        if n_train > max_rows:
            if contiguous:
                idx = np.arange(n_train - max_rows, n_train)
            else:
                idx = rng.choice(n_train, size=max_rows, replace=False)
                idx.sort()
            from src.search.uniqueness import slice_train_rows
            g = slice_train_rows(f, idx)
        out.append(g)
    return out


def _build_folds(
    objective: dict[str, Any],
    pdir: Path,
) -> list[dict[str, Any]]:
    """Build walk-forward folds with real objective labels.

    The feature matrix + labels come from the shared fold cache
    (``src.search.fold_cache.build_or_load_fold_arrays``), which applies the
    objective's ``move_threshold`` / ``mae_tolerance`` / ``scheme``
    (``detection_2`` order-agnostic filter, or ``triple_barrier`` first-touch)
    to the ``_close`` passthrough column at the parquet dir's actual bar
    timestep (probed — 5s feature parquets and 1s raw parquets both work).
    Label semantics match ``src.training.labels.generate_labels``:

    * ``0`` = BIG_DOWN  (bearish directional signal)
    * ``1`` = NO_SIGNAL (masked during training; used as "flat" for unsupervised/RL)
    * ``2`` = BIG_UP    (bullish directional signal)

    Each fold dict keys:
        X_train, y_train, X_val, y_val, X_oof, y_oof  — numpy arrays
        ts_oof   — numpy datetime64 array aligned with X_oof / y_oof
    """
    if str((objective.get("search") or {}).get("input") or "") == "path1s":
        from src.search.path1s_folds import build_path1s_folds
        try:
            return build_path1s_folds(objective)
        except Exception as exc:  # noqa: BLE001
            logger.exception("path1s_fold_build_failed")
            raise RuntimeError(f"path1s fold build failed: {exc}") from exc

    try:
        from src.search.fold_cache import (
            build_or_load_fold_arrays, describe_walk_forward_folds, load_fold_extras,
        )
        arrays = build_or_load_fold_arrays(objective, pdir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fold_build_failed: %s", exc)
        return []
    if arrays is None:
        return []
    X_full, y_full, ts_full, feature_cols = arrays
    try:
        extras = load_fold_extras(objective, pdir)
    except Exception:
        extras = {}

    eval_cfg = objective.get("eval") or {}
    wf = eval_cfg.get("walk_forward") or {}
    if str(wf.get("mode") or "") == "expanding_calendar":
        from src.validation.walk_forward import (
            DEFAULT_PURGE_BARS, apply_schedule_to_arrays, schedule_from_eval,
        )
        tz_name = str((objective.get("session") or {}).get("timezone") or "America/Chicago")
        try:
            schedule = schedule_from_eval(eval_cfg)
            folds = apply_schedule_to_arrays(
                X_full, y_full, ts_full, schedule, feature_cols,
                tz_name=tz_name,
                purge_bars=int(wf.get("purge_bars") or DEFAULT_PURGE_BARS),
                # Preferred over purge_bars: interval-independent, so a 1s dataset
                # purges the full label horizon instead of 120 rows (ADR-052).
                purge_seconds=(int(wf["purge_seconds"])
                               if wf.get("purge_seconds") is not None else None),
                extras=extras or None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("calendar_fold_build_failed: %s", exc)
            return []
        if not folds:
            logger.warning("calendar_fold_build_failed: no non-empty folds")
        return folds

    n_folds  = int(eval_cfg.get("walk_forward_folds", 3))

    n = len(y_full)
    fold_size = n // (n_folds + 1)
    if fold_size < 20:
        logger.warning(
            "fold_build_failed: only %d rows available for %d folds (fold_size=%d)",
            n, n_folds, fold_size,
        )
        return []

    fold_meta = describe_walk_forward_folds(ts_full, y_full, n_folds)
    folds: list[dict[str, Any]] = []
    for i in range(n_folds):
        train_end = (i + 1) * fold_size
        test_end  = min(train_end + fold_size, n)
        fold = {
            # Expanding train window
            "X_train": X_full[:train_end],
            "y_train": y_full[:train_end],
            "ts_train": ts_full[:train_end],
            # Held-out OOF window (val == oof for walk-forward)
            "X_val":   X_full[train_end:test_end],
            "y_val":   y_full[train_end:test_end],
            "ts_val":  ts_full[train_end:test_end],
            "X_oof":   X_full[train_end:test_end],
            "y_oof":   y_full[train_end:test_end],
            # Timestamp array aligned with X_oof (plugins may use for canonical ts)
            "ts_oof":  ts_full[train_end:test_end],
            # Column names for X arrays — required by per-family feature views
            "feature_cols": feature_cols,
            "fold": i,
            "fold_meta": fold_meta[i] if i < len(fold_meta) else None,
        }
        if extras:
            for name, arr in extras.items():
                fold[f"{name}_train"] = arr[:train_end]
                fold[f"{name}_oof"] = arr[train_end:test_end]
                fold[f"{name}_val"] = arr[train_end:test_end]
        folds.append(fold)
    return folds


def _record_all(
    results: list[CandidateResult],
    objective: dict[str, Any],
    contract: EvalContract | None,
    run_ts: str,
) -> None:
    """Write each ok result to the leaderboard (with MLflow lineage id when logged)."""
    if contract is None:
        return
    for r in results:
        if r.status == "ok" and r.score:
            try:
                record_result(
                    objective_id=objective["objective_id"],
                    contract_version=contract.contract_version,
                    candidate_id=r.candidate_id,
                    model_family=r.family,
                    score_result={
                        "primary_precision": r.score.get("precision_at_tau"),
                        "primary_dir_f1":    r.score.get("directional_f1"),
                        "primary_tau":       fire_tau(objective),
                        "per_tau":           [r.score],
                    },
                    spec_params=r.spec_params,
                    mlflow_run_id=r.mlflow_run_id,
                )
            except Exception as exc:
                logger.warning("leaderboard_write_failed cid=%s err=%s", r.candidate_id, exc)


def _mlflow_reachable(uri: str, timeout_s: float = 1.5) -> bool:
    """True if *uri* is a local file store or its HTTP host accepts a TCP connect."""
    if not uri or uri.startswith("file:"):
        return True
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        return True
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def _log_wave_to_mlflow(
    results: list[CandidateResult],
    objective: dict[str, Any],
    run_ts: str,
    *,
    experiment_type: str = "model_selection",
) -> None:
    """Log a search wave to MLflow: one parent run per wave, one nested run per candidate.

    Experiment name = ``search/<objective_id>`` so every sweep over one objective lands
    in a single MLflow experiment; the UI's parent/child view shows wave → candidates.
    Sets ``CandidateResult.mlflow_run_id`` in place (consumed by the leaderboard writer).
    Best-effort: any MLflow failure logs a warning and never breaks the search.
    """
    try:
        import mlflow
        from src.training.config import MLflowConfig

        cfg = MLflowConfig()
        uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking_uri)
        if not _mlflow_reachable(uri):
            logger.warning("mlflow_wave_log_skipped: tracking server not reachable at %s", uri)
            return
        # If the probe raced a dying server, do not sit through urllib3's default
        # 7 connect retries (tens of seconds of Connection refused noise).
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        mlflow.set_tracking_uri(uri)
        objective_id = objective["objective_id"]
        mlflow.set_experiment(f"search/{objective_id}")
        ok = [r for r in results if r.status == "ok"]
        with mlflow.start_run(run_name=f"wave_{run_ts}") as parent:
            mlflow.set_tags({
                "objective_id": objective_id, "run_ts": run_ts,
                "experiment_type": experiment_type, "git_sha": _git_sha()[:12],
                "n_candidates": str(len(results)), "n_ok": str(len(ok)),
            })
            for r in results:
                if r.status != "ok":
                    continue
                with mlflow.start_run(run_name=r.candidate_id, nested=True) as child:
                    mlflow.set_tags({
                        "objective_id": objective_id, "run_ts": run_ts,
                        "experiment_type": experiment_type,
                        "candidate_id": r.candidate_id, "family": r.family,
                        "compute_class": r.compute_class,
                    })
                    params = {f"p_{k}": str(v)[:250] for k, v in (r.spec_params or {}).items()}
                    if r.feature_selection:
                        params["feature_selection"] = json.dumps(r.feature_selection)[:250]
                    if params:
                        mlflow.log_params(params)
                    metrics = {k: float(v) for k, v in (r.score or {}).items()
                               if isinstance(v, (int, float)) and v == v}   # drop NaN/dicts
                    metrics["elapsed_s"] = float(r.elapsed_s)
                    mlflow.log_metrics(metrics)
                    r.mlflow_run_id = child.info.run_id
            logger.info("mlflow_wave_logged experiment=search/%s parent=%s candidates=%d",
                        objective_id, parent.info.run_id[:12], len(ok))
    except Exception as exc:  # noqa: BLE001 — observability must never fail the search
        logger.warning("mlflow_wave_log_failed: %s", exc)


def _apply_topology_override(
    objective: dict[str, Any],
    override: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Return a shallow copy of *objective* with ``search.topology`` replaced.

    Used by WebUI Custom Test so a run can enable/disable/reconfigure meta
    without mutating the immutable objective YAML.
    """
    if isinstance(override, (str, Path)):
        p = Path(override)
        raw = yaml.safe_load(p.read_text()) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"topology overlay must be a mapping: {p}")
        overlay = raw
    else:
        overlay = dict(override)
    search = dict(objective.get("search") or {})
    mode = str(overlay.get("mode") or "none")
    if mode in ("none", "off", ""):
        search["topology"] = {"mode": "none"}
        fs = overlay.get("feature_store")
        if fs:
            search["topology"]["feature_store"] = fs
    else:
        search["topology"] = overlay
    out = dict(objective)
    out["search"] = search
    logger.info("topology_override mode=%s bases=%s", mode, overlay.get("bases"))
    return out


def _apply_eval_override(
    objective: dict[str, Any],
    override: str | Path | dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of *objective* with ``eval`` date_range / walk_forward replaced.

    Used by WebUI Custom Test so a run can set expanding calendar folds without
    mutating the immutable objective YAML.
    """
    if isinstance(override, (str, Path)):
        p = Path(override)
        raw = yaml.safe_load(p.read_text()) if p.suffix in {".yaml", ".yml"} else json.loads(p.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"walk-forward overlay must be a mapping: {p}")
        overlay = raw
    else:
        overlay = dict(override)
    if not overlay or overlay.get("enabled") is False:
        return objective
    ev = dict(objective.get("eval") or {})
    if overlay.get("date_range"):
        ev["date_range"] = overlay["date_range"]
    if overlay.get("walk_forward_folds") is not None:
        ev["walk_forward_folds"] = int(overlay["walk_forward_folds"])
    if overlay.get("walk_forward"):
        ev["walk_forward"] = overlay["walk_forward"]
    out = dict(objective)
    out["eval"] = ev
    logger.info(
        "eval_override folds=%s range=%s mode=%s",
        ev.get("walk_forward_folds"),
        ev.get("date_range"),
        (ev.get("walk_forward") or {}).get("mode"),
    )
    return out


def _progress_line(r: CandidateResult) -> None:
    icon = "✅" if r.status == "ok" else ("⚠️ " if r.status == "skipped" else "❌")
    prec = r.score.get("precision_at_tau", "—") if r.score else "—"
    f1   = r.score.get("directional_f1", "—") if r.score else "—"
    print(f"  {icon} {r.candidate_id:<30}  precision={prec!s:<8}  dir_f1={f1!s:<6}  "
          f"[{r.compute_class}  {r.elapsed_s:.1f}s]")


def custom_test_run_doc(
    *,
    objective_id: str,
    run_ts: str,
    out_dir: Path,
    n_candidates: int,
    n_ok: int,
    n_failed: int,
) -> dict[str, Any]:
    """Custom Test run.json: report-only, no champion / gate_result (ADR-071)."""
    return {
        "kind": "custom_test",
        "status": "report",
        "objective_id": objective_id,
        "run_ts": run_ts,
        "out_dir": str(out_dir),
        "n_candidates": n_candidates,
        "n_ok": n_ok,
        "n_failed": n_failed,
    }


def _print_summary(s: dict[str, Any]) -> None:
    gr = s.get("gate_result")
    kind = s.get("experiment_type") or "search"
    print(f"\n{'─'*60}")
    print(f"  {'Custom Test' if kind == 'custom_test' else 'Search'} complete  {s['run_ts']}")
    print(f"  Candidates : {s['n_candidates']}  ok={s['n_ok']}  failed={s['n_failed']}")
    if gr is None:
        print("  Status     : report (no champion gate)")
        if kind == "custom_test":
            print("  Artifacts  : research/custom_test/<obj>/<run_ts>/models/<cand>/")
    else:
        passed = gr.get("passed_ids") or []
        targets = gr.get("target_ids") or []
        print(f"  Gate       : {gr.get('status', 'n/a')}  "
              f"(requirement met: {len(passed)})")
        print(f"  Champion   : {s.get('champion') or '(none — requirement not met)'}"
              + ("  ✅ meets target" if gr.get("champion_meets_target") else ""))
        if targets:
            print(f"  🎯 Target  : {len(targets)} candidate(s) reached the objective target — "
                  + ", ".join(targets))
    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Phase 23 W3 — model selection search")
    p.add_argument("--objective",  required=True, help="Objective id, e.g. nq-50pt")
    p.add_argument("--families",   default="",    help="Comma-separated family whitelist (empty=all). For tf_oof_meta, matches candidate id (lstm_15s), family key, or native_tf store.")
    p.add_argument("--parquet-dir",default="",    help="Override parquet directory")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--workers",    type=int, default=None)
    p.add_argument("--tune-trials", type=int, default=0,
                   help="W8 per-candidate HPO trials before the gate (0 = off, catalog defaults)")
    p.add_argument("--no-ensemble", action="store_true",
                   help="Skip the Stage-2 ensemble search (evaluate base families only)")
    p.add_argument("--manual-spec", default="",
                   help="USER-OVERRIDE mode: path to a manual candidates spec "
                        "(config/manual_candidates.schema.json). Skips automated model "
                        "selection and fits ONLY the listed models. Excludes --families.")
    p.add_argument("--topology-overlay", default="",
                   help="Optional YAML/JSON replacing search.topology for this run only "
                        "(WebUI Custom Test). mode=none disables the objective meta layer.")
    p.add_argument("--walk-forward-overlay", default="",
                   help="Optional YAML/JSON replacing eval.date_range + expanding calendar "
                        "walk-forward for this run only (WebUI Custom Test, ADR-042).")
    p.add_argument("--experiment-type", default="",
                   help="custom_test | manual_override | model_selection. "
                        "custom_test writes last-fold bundles under research/custom_test/.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    run_search(
        args.objective,
        families=[f.strip() for f in args.families.split(",") if f.strip()] or None,
        parquet_dir=args.parquet_dir or None,
        dry_run=args.dry_run,
        max_cpu_workers=args.workers,
        tune_trials=args.tune_trials,
        ensembles=not args.no_ensemble,
        manual_spec=args.manual_spec or None,
        topology_override=args.topology_overlay or None,
        walk_forward_override=args.walk_forward_overlay or None,
        experiment_type=args.experiment_type or None,
    )


if __name__ == "__main__":
    _cli()
