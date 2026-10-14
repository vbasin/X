"""Champion-finalize bridge — search verdict → serve bundle in staging (ADR-024).

Search keeps OOF only; this module refits the champion once into::

    models/model_staging/<objective_id>/<candidate>_<run_ts>/

Promote to production via :mod:`src.serving.promote` →
``model_registry/<objective_id>/LATEST.json`` (RD baseline). No dual-write to
``training_runs/``.

Usage::

    python -m src.search.finalize --objective nq-50pt-tbm
"""

from __future__ import annotations

from src.common.data_paths import data_root, model_staging_dir, search_dir

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("vtech.search.finalize")

_DATA_ROOT = data_root()
_VAL_TAIL_FRACTION = 0.1          # held-out tail used only for the refit sanity score


# ── run-dir / candidate resolution ────────────────────────────────────────────

def _latest_run_dir(objective_id: str) -> Path | None:
    # a finalize target may be a historical run, so resolve across roots
    from src.common.run_layout import run_roots
    root = next((r / objective_id for r in run_roots()
                 if (r / objective_id).is_dir()), search_dir() / objective_id)
    runs = sorted(
        (d for d in root.glob("*Z") if (d / "search_result.json").exists()),
        key=lambda d: d.name,
    )
    return runs[-1] if runs else None


def _pick_candidate(summary: dict[str, Any], candidate_id: str | None) -> dict[str, Any]:
    cands = {c["candidate_id"]: c for c in summary.get("candidates", [])}
    cid = candidate_id or summary.get("champion")
    if not cid:
        raise SystemExit(
            "no champion in search_result.json and no --candidate given "
            f"(gate: {(summary.get('gate_result') or {}).get('status')})")
    if cid not in cands:
        raise SystemExit(f"candidate '{cid}' not in run (have: {sorted(cands)})")
    c = cands[cid]
    if c.get("status") != "ok":
        raise SystemExit(f"candidate '{cid}' status={c.get('status')} — cannot finalize")
    if c["family"].startswith("ensemble_"):
        raise SystemExit(
            f"'{cid}' is an ensemble of cached OOF — finalize its base members instead")
    return c


# ── model persistence (per-framework) ─────────────────────────────────────────

def _save_model(artifact: Any, out_dir: Path) -> str:
    """Persist *artifact* with the right serializer; returns the model filename."""
    cls = type(artifact).__module__ + "." + type(artifact).__name__
    if hasattr(artifact, "save") and "keras" in cls.lower():
        name = "model.keras"
        artifact.save(out_dir / name)
    elif hasattr(artifact, "save_model") and "xgboost" in cls.lower():
        name = "model.ubj"
        artifact.save_model(str(out_dir / name))
    else:                                   # sklearn / hmmlearn / regime dicts …
        import joblib
        name = "model.pkl"
        joblib.dump(artifact, out_dir / name)
    return name


def write_serve_bundle(
    dest: str | Path,
    *,
    artifact: Any,
    feature_cols: list[str],
    objective_id: str,
    candidate_id: str,
    family: str,
    run_ts: str,
    search_run_dir: str | Path | None = None,
    search_score: dict[str, Any] | None = None,
    spec_params: dict[str, Any] | None = None,
    feature_selection: dict[str, Any] | None = None,
    scaler: Any | None = None,
    extra_manifest: dict[str, Any] | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Write a promote-ready serving folder (schema + model + optional scaler + manifest)."""
    from src.search.controller import _git_sha

    dest_p = Path(dest)
    if dest_p.exists() and replace:
        shutil.rmtree(dest_p)
    dest_p.mkdir(parents=True, exist_ok=True)
    model_file = _save_model(artifact, dest_p)
    if scaler is not None:
        import joblib
        joblib.dump(scaler, dest_p / "scaler.pkl")
    cols = list(feature_cols)
    (dest_p / "feature_schema.json").write_text(json.dumps({
        "feature_cols": cols,
        "n_features": len(cols),
        "feature_selection": feature_selection,
    }, indent=2))
    version = f"{candidate_id}_{run_ts}"
    manifest: dict[str, Any] = {
        "objective_id": objective_id,
        "candidate_id": candidate_id,
        "family": family,
        "version": version,
        "search_run_ts": run_ts,
        "search_run_dir": str(search_run_dir) if search_run_dir else None,
        "search_score": search_score,
        "spec_params": spec_params or {},
        "feature_selection": feature_selection,
        "model_file": model_file,
        "has_scaler": scaler is not None,
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(dest_p),
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    pol = manifest.get("inference_policy")
    if isinstance(pol, dict) and pol:
        (dest_p / "inference_policy.json").write_text(
            json.dumps(pol, indent=2, default=str))
    (dest_p / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    return manifest


_LAB_BASE_FILES = (
    "model.ubj", "model.keras", "model.pkl", "model.joblib",
    "scaler.pkl", "feature_schema.json",
)


def _copy_lab_bases(
    dest_p: Path, base_bundle_dirs: dict[str, Path], bases: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Copy last-fold base serving files into ``dest/bases/<family>/``."""
    saved: list[dict[str, Any]] = []
    all_cols: list[str] = []
    for family in bases:
        src = Path(base_bundle_dirs[family])
        bdir = dest_p / "bases" / family
        bdir.mkdir(parents=True, exist_ok=True)
        model_file = None
        for name in _LAB_BASE_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, bdir / name)
                if name.startswith("model"):
                    model_file = name
        schema: dict[str, Any] = {}
        if (bdir / "feature_schema.json").exists():
            schema = json.loads((bdir / "feature_schema.json").read_text())
        cols = list(schema.get("feature_cols") or [])
        for c in cols:
            if c not in all_cols:
                all_cols.append(c)
        src_man: dict[str, Any] = {}
        if (src / "finalize_manifest.json").exists():
            try:
                src_man = json.loads((src / "finalize_manifest.json").read_text())
            except json.JSONDecodeError:
                src_man = {}
        kind = src_man.get("kind") or (
            "sequence" if (bdir / "model.keras").exists() else "tabular")
        saved.append({
            "family": family,
            "kind": kind,
            "model_file": model_file,
            "feature_cols": cols,
            "seq_len": src_man.get("seq_len") or (src_man.get("spec_params") or {}).get("seq_len"),
            "has_scaler": (bdir / "scaler.pkl").exists(),
        })
    return saved, all_cols


def lab_model_dir(run_dir: str | Path, candidate_id: str) -> Path:
    """``<run>/models/<candidate_id>/`` — Custom Test last-fold (or meta) bundle."""
    return Path(run_dir) / "models" / candidate_id


def rewrite_lab_manifest(
    dest: str | Path, *, search_run_dir: str | Path | None = None,
) -> None:
    """Point ``staged_to`` / ``search_run_dir`` at the CPU run after GPU collect."""
    man_p = Path(dest) / "finalize_manifest.json"
    if not man_p.exists():
        return
    try:
        man = json.loads(man_p.read_text())
    except json.JSONDecodeError:
        return
    man["staged_to"] = str(Path(dest))
    if search_run_dir is not None:
        man["search_run_dir"] = str(search_run_dir)
    if not man.get("native_tf"):
        cid = str(man.get("candidate_id") or Path(dest).name)
        for tok in ("15s", "5s", "5m", "1m", "1h", "1d"):
            if tok in cid.replace("-", "_").split("_"):
                man["native_tf"] = tok
                break
    man_p.write_text(json.dumps(man, indent=2, default=str))


def write_meta_stack_lab_bundle(
    dest: str | Path,
    *,
    base_bundle_dirs: dict[str, Path],
    composed_oof: Any,
    objective_id: str,
    candidate_id: str,
    run_ts: str,
    search_run_dir: str | Path,
    search_score: dict[str, Any] | None,
    spec_params: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Self-contained meta_stack serving folder from last-fold bases + combo OOF.

    Copies each base's last-fold weights into ``bases/<family>/``, fits one LR on
    the walk-forward combo OOF (``fit_final``), writes ``meta.pkl`` / ``meta.json``.
    """
    import joblib
    import pandas as pd
    from src.search.meta_stack import fit_final

    dest_p = Path(dest)
    if dest_p.exists():
        shutil.rmtree(dest_p)
    dest_p.mkdir(parents=True, exist_ok=True)

    bases = list(spec_params.get("bases") or list(base_bundle_dirs))
    meta_cfg = dict(spec_params.get("meta") or {})
    saved, all_cols = _copy_lab_bases(dest_p, base_bundle_dirs, bases)
    oofs: dict[str, Any] = {}
    for family in bases:
        src = Path(base_bundle_dirs[family])
        oof_p = src / "oof.parquet"
        if not oof_p.exists():
            oof_p = src.parent.parent / f"oof_{family}.parquet"
        if oof_p.exists():
            oofs[family] = pd.read_parquet(oof_p)

    if composed_oof is not None:
        composed_oof.to_parquet(dest_p / "oof.parquet", index=False)
        if "direction_true" in composed_oof.columns and len(oofs) < len(bases):
            # bases without sibling OOF still participate via composed frame only
            pass

    missing = [b for b in bases if b not in oofs]
    if missing:
        raise ValueError(f"meta_stack lab bundle: missing base OOF for {missing}")

    fitted_meta = fit_final(
        oofs, bases=bases,
        feature=str(meta_cfg.get("feature", "linear_weights")),
    )
    joblib.dump(fitted_meta["estimator"], dest_p / "meta.pkl")
    meta_json = {
        "topology": "meta_stack",
        "bases": bases,
        "family": fitted_meta["family"],
        "feature": fitted_meta["feature"],
        "feature_names": fitted_meta["feature_names"],
        "learned_weights": fitted_meta["weights"],
        "learned_bias": fitted_meta["bias"],
        "approve_threshold": float(meta_cfg.get("approve_threshold", 0.55)),
        "scheme_at_search_time": str(meta_cfg.get("scheme", "walk_forward_OOF")),
        "fire_rule": "symmetric: up if p_up>=thr, down if p_up<=1-thr, else flat",
        "weights_source": "last_walk_forward_fold",
    }
    (dest_p / "meta.json").write_text(json.dumps(meta_json, indent=2))
    (dest_p / "feature_schema.json").write_text(json.dumps({
        "feature_cols": all_cols, "n_features": len(all_cols),
        "bases": saved,
    }, indent=2))
    if diagnostics:
        (dest_p / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str))

    from src.search.controller import _git_sha
    version = f"{candidate_id}_{run_ts}"
    manifest: dict[str, Any] = {
        "objective_id": objective_id,
        "candidate_id": candidate_id,
        "family": "meta_stack",
        "version": version,
        "search_run_ts": run_ts,
        "search_run_dir": str(search_run_dir),
        "search_score": search_score,
        "spec_params": spec_params,
        "bases": saved,
        "meta": meta_json,
        "weights_source": "last_walk_forward_fold",
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(dest_p),
    }
    (dest_p / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    return manifest


def write_primary_meta_lab_bundle(
    dest: str | Path,
    *,
    primary_bundle_dir: str | Path,
    primary_id: str,
    fitted_meta: dict[str, Any],
    composed_oof: Any,
    objective_id: str,
    candidate_id: str,
    run_ts: str,
    search_run_dir: str | Path,
    search_score: dict[str, Any] | None,
    spec_params: dict[str, Any],
    diagnostics: dict[str, Any] | None = None,
    context_view: str | None = None,
) -> dict[str, Any]:
    """Self-contained primary_meta serving folder (binary gate + copied primary).

    Layout::

        meta.pkl / meta.json
        bases/<primary>/   last-fold directional bundle
        feature_schema.json, finalize_manifest.json, oof.parquet
    """
    import joblib

    dest_p = Path(dest)
    if dest_p.exists():
        shutil.rmtree(dest_p)
    dest_p.mkdir(parents=True, exist_ok=True)

    family = str(primary_id)
    saved, all_cols = _copy_lab_bases(
        dest_p, {family: Path(primary_bundle_dir)}, [family])
    meta_cfg = dict(spec_params.get("meta") or {})
    names = list(fitted_meta.get("feature_names") or ["primary_confidence", "is_up"])
    for n in names:
        if n not in all_cols and n not in ("primary_confidence", "is_up"):
            all_cols.append(n)
    joblib.dump({
        "kind": fitted_meta.get("kind", "constant"),
        "estimator": fitted_meta.get("estimator"),
    }, dest_p / "meta.pkl")
    meta_json = {
        "topology": "primary_meta",
        "bases": [family],
        "family": str(meta_cfg.get("family") or "xgboost"),
        "feature_names": names,
        "impute_values": fitted_meta.get("impute_values") or {},
        "approve_threshold": float(meta_cfg.get("approve_threshold", 0.55)),
        "min_primary_confidence": float(meta_cfg.get("min_primary_confidence", 0.50)),
        "context_view": context_view or meta_cfg.get("context_view"),
        "n_train": int(fitted_meta.get("n_train") or 0),
        "scheme_at_search_time": str(meta_cfg.get("scheme", "walk_forward_OOF")),
        "fire_rule": (
            "primary side+confidence if directional and conf>=min_primary_confidence "
            "and P(win)>=approve_threshold; else flat"
        ),
        "weights_source": "full_eligible_refit",
    }
    (dest_p / "meta.json").write_text(json.dumps(meta_json, indent=2, default=str))
    (dest_p / "feature_schema.json").write_text(json.dumps({
        "feature_cols": all_cols, "n_features": len(all_cols),
        "bases": saved,
        "meta_feature_names": names,
    }, indent=2))
    if composed_oof is not None:
        composed_oof.to_parquet(dest_p / "oof.parquet", index=False)
    if diagnostics:
        (dest_p / "diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, default=str))

    from src.search.controller import _git_sha
    version = f"{candidate_id}_{run_ts}"
    manifest: dict[str, Any] = {
        "objective_id": objective_id,
        "candidate_id": candidate_id,
        "family": "primary_meta",
        "version": version,
        "search_run_ts": run_ts,
        "search_run_dir": str(search_run_dir),
        "search_score": search_score,
        "spec_params": spec_params,
        "bases": saved,
        "meta": meta_json,
        "weights_source": "full_eligible_refit",
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(dest_p),
    }
    (dest_p / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    return manifest


# ── inference policy attached at finalize (ADR-047) ───────────────────────────

def _inference_policy_for(
    objective: dict[str, Any], objective_id: str,
) -> dict[str, Any] | None:
    """Serving-time book for objectives that are not “score every 5s bar”.

    Training ``session.trading_*`` stays the data window. This policy is what
    Backtest applies when the staged bundle is loaded (open + uniqueness +
    long-only for the causal 50/6 timer). Not a promotion.
    """
    if objective_id != "nq-50pt-xgb-unique":
        return None
    tz = str((objective.get("session") or {}).get("timezone") or "America/Chicago")
    return {
        "tau": 0.50,
        "trading_start": "08:30",
        "trading_end": "10:00",
        "timezone": tz,
        "sides": ["up"],
        "cooldown_bars": 120,
        "adr": "ADR-047",
        "note": (
            "Causal long timer: 08:30–10:00 CT, long-only, 10-minute uniqueness. "
            "Last-fold serving τ=0.50 (this booster rarely reaches 0.65). "
            "Stacked walk-forward OOF used τ=0.65 (ADR-047). Staging only — not "
            "promoted. Live pin unchanged."
        ),
    }


# ── main entry point ──────────────────────────────────────────────────────────

def finalize_champion(
    objective_id: str,
    run_dir: str | Path | None = None,
    candidate_id: str | None = None,
    *,
    parquet_dir: str | Path | None = None,
    stage: bool = True,
    dest: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Refit the search champion on the full matrix and persist a serve bundle in staging.

    Returns the finalize manifest dict (also written under model_staging).
    """
    from src.objectives.loader import load_objective
    from src.search.controller import _build_folds, _load_plugin, _git_sha
    from src.models.base import ModelSpec
    from src.models.feature_views import apply_feature_selection, apply_feature_view

    objective = load_objective(objective_id)
    rdir = Path(run_dir) if run_dir else _latest_run_dir(objective_id)
    if rdir is None or not (rdir / "search_result.json").exists():
        raise SystemExit(f"no search run with search_result.json found for {objective_id}")
    summary = json.loads((rdir / "search_result.json").read_text())
    cand = _pick_candidate(summary, candidate_id)
    cid, family = cand["candidate_id"], cand["family"]
    spec_params = dict(cand.get("spec_params") or {})
    feature_selection = cand.get("feature_selection")

    print(f"⏳ Finalizing '{cid}' (family={family}) from {rdir.name} "
          f"{'[DRY RUN]' if dry_run else ''}")
    if dry_run:
        return {"candidate_id": cid, "family": family, "spec_params": spec_params,
                "feature_selection": feature_selection, "dry_run": True}

    if family == "output_meta" or cid.startswith("meta__"):
        return _finalize_output_meta(
            objective, objective_id, rdir, summary, cand, parquet_dir)

    if family == "meta_stack" or cid.startswith("meta_stack__"):
        return _finalize_meta_stack(
            objective, objective_id, rdir, summary, cand, parquet_dir)

    # ── Full-matrix refit fold (expanding final window + validation tail) ────
    from src.features.stores import default_parquet_dir

    pdir = Path(parquet_dir) if parquet_dir else default_parquet_dir(objective)
    folds = _build_folds(objective, pdir)
    if not folds:
        raise SystemExit(f"no folds could be built from {pdir}")
    u_bars = (objective.get("eval") or {}).get("train_uniqueness_bars")
    if u_bars:
        from src.search.uniqueness import apply_train_uniqueness
        folds = apply_train_uniqueness(folds, int(u_bars))
    if (objective.get("eval") or {}).get("train_onset_uniqueness"):
        from src.search.uniqueness import apply_train_onset_uniqueness
        from src.training.dual_binary_15s import label_params_from_objective
        hp = label_params_from_objective(objective)
        folds = apply_train_onset_uniqueness(
            folds, horizon_s=float(hp.get("horizon_s") or 600), drop_rows=True,
        )
    last = folds[-1]
    # Uniqueness-thinned objectives must NOT concat dense OOF into train.
    # Last-fold OOF is every 5s bar (mostly flat); mixing it in after a 2-minute
    # train stride drowns the directional rows and the refit becomes a flat
    # classifier (observed on nq-50pt-xgb-unique). Use last expanding-window
    # train as search did; score the fold OOF as the val/sanity split.
    if u_bars:
        refit_fold = {
            "X_train": last["X_train"], "y_train": last["y_train"],
            "X_val": last["X_oof"], "y_val": last["y_oof"],
            "X_oof": last["X_oof"], "y_oof": last["y_oof"],
            "ts_oof": last.get("ts_oof"),
            "feature_cols": last["feature_cols"],
        }
        split = int(len(last["X_train"]))
        n_all = split + int(len(last["X_oof"]))
    else:
        X_all = np.concatenate([last["X_train"], last["X_oof"]])
        y_all = np.concatenate([last["y_train"], last["y_oof"]])
        split = max(1, int(len(X_all) * (1 - _VAL_TAIL_FRACTION)))
        n_all = len(X_all)
        refit_fold = {
            "X_train": X_all[:split], "y_train": y_all[:split],
            "X_val": X_all[split:], "y_val": y_all[split:],
            "X_oof": X_all[split:], "y_oof": y_all[split:],
            "ts_oof": None,
            "feature_cols": last["feature_cols"],
        }
    refit_folds = [refit_fold]
    if feature_selection:
        refit_folds = apply_feature_selection(refit_folds, feature_selection)

    t0 = time.time()
    plugin = _load_plugin(family)
    model = plugin.build(ModelSpec(family=family, params=spec_params), objective)
    fitted = plugin.fit(model, refit_folds, objective=objective)
    elapsed = time.time() - t0

    # Sanity score on the held-out tail (not the frozen contract — just a health check)
    val_score: dict[str, Any] = {}
    try:
        from src.evaluation._kpi import canonical_trade_kpi
        from src.evaluation.kpi_criteria import primary_tau
        import pandas as pd
        if fitted.oof_preds is not None and "direction_true" in fitted.oof_preds.columns:
            gt = pd.DataFrame({"direction_true": fitted.oof_preds["direction_true"]})
            val_score = {k: v for k, v in canonical_trade_kpi(
                fitted.oof_preds, gt, objective, tau=primary_tau(objective)).items()
                if isinstance(v, (int, float, str))}
    except Exception as exc:  # noqa: BLE001
        logger.warning("refit_val_score_failed: %s", exc)

    # ── Persist serve bundle under staging only (no training_runs dual-write) ─
    version = f"{cid}_{summary['run_ts']}"
    if dest is not None:
        stage_dir = Path(dest)
    elif not stage:
        raise SystemExit("finalize requires staging (training_runs dual-write removed)")
    else:
        stage_dir = model_staging_dir() / objective_id / version
    scaler = (fitted.meta or {}).get("scaler")
    extra: dict[str, Any] = {
        "refit_val_score": val_score,
        "n_train_rows": int(split),
        "n_val_rows": int(n_all - split),
        "refit_elapsed_s": round(elapsed, 1),
        "weights_source": "last_expanding_window" if u_bars else "full_matrix_refit",
        "mlflow_run_id": None,
    }
    pol = _inference_policy_for(objective, objective_id)
    if pol:
        extra["inference_policy"] = pol
    manifest = write_serve_bundle(
        stage_dir,
        artifact=fitted.artifact,
        feature_cols=list(refit_folds[0]["feature_cols"]),
        objective_id=objective_id,
        candidate_id=cid,
        family=family,
        run_ts=str(summary["run_ts"]),
        search_run_dir=rdir,
        search_score=cand.get("score"),
        spec_params=spec_params,
        feature_selection=feature_selection,
        scaler=scaler,
        extra_manifest=extra,
    )

    # ── MLflow: log the finalized model next to its search wave ──────────────
    try:
        import mlflow
        from src.training.config import MLflowConfig
        cfg = MLflowConfig()
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg.tracking_uri))
        mlflow.set_experiment(f"search/{objective_id}")
        with mlflow.start_run(run_name=f"finalize_{version}") as run:
            mlflow.set_tags({
                "objective_id": objective_id, "experiment_type": "finalize",
                "candidate_id": cid, "family": family,
                "search_run_ts": summary["run_ts"], "git_sha": manifest["git_sha"][:12],
            })
            mlflow.log_params({f"p_{k}": str(v)[:250] for k, v in spec_params.items()})
            mlflow.log_metrics({k: float(v) for k, v in val_score.items()
                                if isinstance(v, (int, float)) and v == v})
            mlflow.log_artifacts(str(stage_dir))
            manifest["mlflow_run_id"] = run.info.run_id
    except Exception as exc:  # noqa: BLE001 — observability must never fail finalize
        logger.warning("mlflow_finalize_log_failed: %s", exc)

    (stage_dir / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))

    print(f"✅ Finalized in {elapsed:.0f}s → staged {stage_dir}"
          + f"\n   refit sanity: { {k: round(v,4) if isinstance(v,float) else v for k,v in val_score.items() if k in ('precision_at_tau','directional_f1','n_signals')} }"
          + f"\n   mlflow_run_id: {manifest['mlflow_run_id']}"
          + "\n   promote: python -m src.serving.promote --staging-dir <path>")
    return manifest


def _finalize_output_meta(
    objective: dict[str, Any],
    objective_id: str,
    rdir: Path,
    summary: dict[str, Any],
    cand: dict[str, Any],
    parquet_dir: str | Path | None,
) -> dict[str, Any]:
    """Refit every topology base + the binary gate; write a multi-artifact bundle."""
    import joblib
    from src.models.base import ModelSpec
    from src.models.catalog import get_entry
    from src.models.feature_views import apply_feature_selection, apply_feature_view
    from src.search.controller import _build_folds, _load_plugin, _git_sha
    from src.search.output_meta import fit_final_meta

    spec_params = dict(cand.get("spec_params") or {})
    bases = list(spec_params.get("bases") or [])
    meta_cfg = dict(spec_params.get("meta") or {})
    base_specs = spec_params.get("base_specs") or {}
    base_sels = spec_params.get("base_feature_selection") or {}
    if not bases:
        raise SystemExit("output_meta finalize: spec_params.bases is empty")

    from src.features.stores import default_parquet_dir

    pdir = Path(parquet_dir) if parquet_dir else default_parquet_dir(objective)
    folds = _build_folds(objective, pdir)
    if not folds:
        raise SystemExit(f"no folds could be built from {pdir}")

    cid = cand["candidate_id"]
    version = f"{cid}_{summary['run_ts']}"
    stage_dir = model_staging_dir() / objective_id / version
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    search_cfg = objective.get("search") or {}
    views = search_cfg.get("feature_views") or {}
    default_view = search_cfg.get("default_feature_view")
    t0 = time.time()
    saved: list[dict[str, Any]] = []

    last = folds[-1]
    X_all = np.concatenate([last["X_train"], last["X_oof"]])
    y_all = np.concatenate([last["y_train"], last["y_oof"]])
    split = max(1, int(len(X_all) * (1 - _VAL_TAIL_FRACTION)))
    feat_cols = list(last["feature_cols"])

    for family in bases:
        entry = get_entry(family)
        sel = base_sels.get(family)
        view = views.get(family) or getattr(entry, "feature_view", None) or default_view
        refit = {
            "X_train": X_all[:split], "y_train": y_all[:split],
            "X_val": X_all[split:], "y_val": y_all[split:],
            "X_oof": X_all[split:], "y_oof": y_all[split:],
            "ts_oof": None,
            "feature_cols": feat_cols,
        }
        refit_folds = [refit]
        if sel:
            refit_folds = apply_feature_selection(refit_folds, sel)
        elif view:
            refit_folds = apply_feature_view(refit_folds, view)
        params = dict(getattr(entry, "params_default", {}) or {})
        params.update(base_specs.get(family) or {})
        plugin = _load_plugin(family)
        model = plugin.build(ModelSpec(family=family, params=params), objective)
        fitted = plugin.fit(model, refit_folds, objective=objective)
        bdir = stage_dir / "bases" / family
        bdir.mkdir(parents=True, exist_ok=True)
        model_file = _save_model(fitted.artifact, bdir)
        scaler = (fitted.meta or {}).get("scaler")
        if scaler is not None:
            joblib.dump(scaler, bdir / "scaler.pkl")
        cols = list(refit_folds[0]["feature_cols"])
        (bdir / "feature_schema.json").write_text(json.dumps({
            "feature_cols": cols, "n_features": len(cols),
        }, indent=2))
        kind = "sequence" if getattr(entry, "needs_sequence", False) else "tabular"
        saved.append({
            "family": family, "kind": kind, "model_file": model_file,
            "feature_cols": cols,
            "seq_len": (fitted.meta or {}).get("seq_len") or params.get("seq_len"),
            "has_scaler": scaler is not None,
        })

    oofs: dict[str, Any] = {}
    for family in bases:
        src = rdir / f"oof_{family}.parquet"
        if not src.exists():
            # manual-spec ids may differ from family name
            matches = list(rdir.glob(f"oof_*{family}*.parquet"))
            src = matches[0] if matches else src
        if not src.exists():
            raise SystemExit(f"output_meta finalize: missing OOF for base {family}: {src}")
        import pandas as pd
        oofs[family] = pd.read_parquet(src)
    fitted_meta = fit_final_meta(
        oofs, bases=bases,
        meta_family=str(meta_cfg.get("family") or "logistic_regression"),
    )
    joblib.dump(fitted_meta["estimator"], stage_dir / "meta.pkl")
    meta_json = {
        "bases": bases,
        "family": fitted_meta["family"],
        "feature_names": fitted_meta["feature_names"],
        "approve_threshold": float(meta_cfg.get("approve_threshold", 0.55)),
        "side_rule": "agree_or_single_directional",
    }
    (stage_dir / "meta.json").write_text(json.dumps(meta_json, indent=2))
    (stage_dir / "feature_schema.json").write_text(json.dumps({
        "feature_cols": feat_cols, "n_features": len(feat_cols),
        "bases": saved,
    }, indent=2))

    elapsed = time.time() - t0
    manifest = {
        "objective_id": objective_id,
        "candidate_id": cid,
        "family": "output_meta",
        "version": version,
        "search_run_ts": summary["run_ts"],
        "search_run_dir": str(rdir),
        "search_score": cand.get("score"),
        "spec_params": spec_params,
        "bases": saved,
        "meta": meta_json,
        "refit_elapsed_s": round(elapsed, 1),
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(stage_dir),
    }
    (stage_dir / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f"✅ Finalized output_meta in {elapsed:.0f}s → staged {stage_dir}")
    return manifest


def _finalize_meta_stack(
    objective: dict[str, Any],
    objective_id: str,
    rdir: Path,
    summary: dict[str, Any],
    cand: dict[str, Any],
    parquet_dir: str | Path | None,
) -> dict[str, Any]:
    """Refit every topology base + the signed-evidence LR; write a multi-artifact bundle."""
    import joblib
    from src.models.base import ModelSpec
    from src.models.catalog import get_entry
    from src.models.feature_views import apply_feature_selection, apply_feature_view
    from src.search.controller import _build_folds, _load_plugin, _git_sha
    from src.search.meta_stack import fit_final

    spec_params = dict(cand.get("spec_params") or {})
    bases = list(spec_params.get("bases") or [])
    meta_cfg = dict(spec_params.get("meta") or {})
    base_specs = spec_params.get("base_specs") or {}
    base_sels = spec_params.get("base_feature_selection") or {}
    if not bases:
        raise SystemExit("meta_stack finalize: spec_params.bases is empty")

    from src.features.stores import default_parquet_dir

    pdir = Path(parquet_dir) if parquet_dir else default_parquet_dir(objective)
    folds = _build_folds(objective, pdir)
    if not folds:
        raise SystemExit(f"no folds could be built from {pdir}")

    cid = cand["candidate_id"]
    version = f"{cid}_{summary['run_ts']}"
    stage_dir = model_staging_dir() / objective_id / version
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    search_cfg = objective.get("search") or {}
    views = search_cfg.get("feature_views") or {}
    default_view = search_cfg.get("default_feature_view")
    t0 = time.time()
    saved: list[dict[str, Any]] = []

    last = folds[-1]
    X_all = np.concatenate([last["X_train"], last["X_oof"]])
    y_all = np.concatenate([last["y_train"], last["y_oof"]])
    split = max(1, int(len(X_all) * (1 - _VAL_TAIL_FRACTION)))
    feat_cols = list(last["feature_cols"])

    for family in bases:
        entry = get_entry(family)
        sel = base_sels.get(family)
        view = views.get(family) or getattr(entry, "feature_view", None) or default_view
        refit = {
            "X_train": X_all[:split], "y_train": y_all[:split],
            "X_val": X_all[split:], "y_val": y_all[split:],
            "X_oof": X_all[split:], "y_oof": y_all[split:],
            "ts_oof": None,
            "feature_cols": feat_cols,
        }
        refit_folds = [refit]
        if sel:
            refit_folds = apply_feature_selection(refit_folds, sel)
        elif view:
            refit_folds = apply_feature_view(refit_folds, view)
        params = dict(getattr(entry, "params_default", {}) or {})
        params.update(base_specs.get(family) or {})
        plugin = _load_plugin(family)
        model = plugin.build(ModelSpec(family=family, params=params), objective)
        fitted = plugin.fit(model, refit_folds, objective=objective)
        bdir = stage_dir / "bases" / family
        bdir.mkdir(parents=True, exist_ok=True)
        model_file = _save_model(fitted.artifact, bdir)
        scaler = (fitted.meta or {}).get("scaler")
        if scaler is not None:
            joblib.dump(scaler, bdir / "scaler.pkl")
        cols = list(refit_folds[0]["feature_cols"])
        (bdir / "feature_schema.json").write_text(json.dumps({
            "feature_cols": cols, "n_features": len(cols),
        }, indent=2))
        kind = "sequence" if getattr(entry, "needs_sequence", False) else "tabular"
        saved.append({
            "family": family, "kind": kind, "model_file": model_file,
            "feature_cols": cols,
            "seq_len": (fitted.meta or {}).get("seq_len") or params.get("seq_len"),
            "has_scaler": scaler is not None,
        })

    # Load walk-forward base OOFs from the search dir; fit a single LR on all
    # directional labels for the served bundle (the wave already emitted the
    # OOF-schedule-fit LR predictions for KPI scoring).
    oofs: dict[str, Any] = {}
    for family in bases:
        src = rdir / f"oof_{family}.parquet"
        if not src.exists():
            matches = list(rdir.glob(f"oof_*{family}*.parquet"))
            src = matches[0] if matches else src
        if not src.exists():
            raise SystemExit(f"meta_stack finalize: missing OOF for base {family}: {src}")
        import pandas as pd
        oofs[family] = pd.read_parquet(src)

    fitted_meta = fit_final(
        oofs, bases=bases,
        feature=str(meta_cfg.get("feature", "linear_weights")),
    )
    joblib.dump(fitted_meta["estimator"], stage_dir / "meta.pkl")
    meta_json = {
        "topology": "meta_stack",
        "bases": bases,
        "family": fitted_meta["family"],
        "feature": fitted_meta["feature"],
        "feature_names": fitted_meta["feature_names"],
        "learned_weights": fitted_meta["weights"],
        "learned_bias": fitted_meta["bias"],
        "approve_threshold": float(meta_cfg.get("approve_threshold", 0.55)),
        "scheme_at_search_time": str(meta_cfg.get("scheme", "walk_forward_OOF")),
        "fire_rule": "symmetric: up if p_up>=thr, down if p_up<=1-thr, else flat",
    }
    (stage_dir / "meta.json").write_text(json.dumps(meta_json, indent=2))
    (stage_dir / "feature_schema.json").write_text(json.dumps({
        "feature_cols": feat_cols, "n_features": len(feat_cols),
        "bases": saved,
    }, indent=2))

    elapsed = time.time() - t0
    manifest = {
        "objective_id": objective_id,
        "candidate_id": cid,
        "family": "meta_stack",
        "version": version,
        "search_run_ts": summary["run_ts"],
        "search_run_dir": str(rdir),
        "search_score": cand.get("score"),
        "spec_params": spec_params,
        "bases": saved,
        "meta": meta_json,
        "refit_elapsed_s": round(elapsed, 1),
        "git_sha": _git_sha(),
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "staged_to": str(stage_dir),
    }
    (stage_dir / "finalize_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str))
    print(f"✅ Finalized meta_stack in {elapsed:.0f}s → staged {stage_dir}")
    return manifest


def _cli() -> None:
    p = argparse.ArgumentParser(description="ADR-024 — finalize a search champion into a promotable bundle")
    p.add_argument("--objective", required=True)
    p.add_argument("--run", default="", help="Search run dir (default: latest with a result)")
    p.add_argument("--candidate", default="", help="Candidate id (default: the run's champion)")
    p.add_argument("--parquet-dir", default="", help="Feature dir override (e.g. data/parquet_raw1s)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    finalize_champion(
        args.objective,
        run_dir=args.run or None,
        candidate_id=args.candidate or None,
        parquet_dir=args.parquet_dir or None,
        stage=True,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _cli()
