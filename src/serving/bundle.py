"""Serving-bundle discovery + load (staged / registry model dirs).

Contract on disk (finalize/promote output)::

    feature_schema.json          # feature_cols (or columns)
    model.ubj | model.pkl | model.joblib
    scaler.pkl                   # optional
    finalize_manifest.json       # optional — dir_threshold, candidate_id, …
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Live data root (honours ``VTECH_DATA_DIR`` on every call — tests can setenv)."""
    from src.common.data_paths import data_root as _dr
    return _dr()


def is_serving_bundle(d: Path | str) -> bool:
    """True if *d* is a finalize/promote dir Live/Backtest can score."""
    p = Path(d)
    if not p.is_dir() or not (p / "feature_schema.json").exists():
        return False
    if (p / "meta.json").exists() and (p / "meta.pkl").exists():
        return True
    return any((p / n).exists() for n in (
        "model.ubj", "model.pkl", "model.joblib", "model.keras"))


def allowed_model_roots() -> list[Path]:
    from src.common.data_paths import model_artifact_roots
    return model_artifact_roots()


def validate_bundle_path(path: str | Path) -> Path:
    """Resolve *path* and require a serving bundle under staging or registry."""
    p = Path(path)
    if not p.is_absolute():
        p = _REPO / p
    p = p.resolve()
    if not is_serving_bundle(p):
        raise ValueError(
            f"not a serving bundle (need feature_schema.json + model or meta.pkl): {p}")
    allowed = allowed_model_roots()
    if not allowed:
        raise ValueError(
            "no model_staging/ or model_registry/ under data root "
            f"({data_root()})")
    if not any(root in p.parents or p == root for root in allowed):
        raise ValueError(
            f"model path must be under model_staging/, model_registry/, "
        f"research/runs/, research/search/, or research/custom_test/: {p}")
    return p


def load_serving_bundle(
    model_dir: str | Path,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load booster + scaler + schema + dir_threshold from a serving dir.

    Returns a dict with keys::

        path, feature_cols, booster, scaler, dir_thr,
        candidate_id, family, version, objective_id, manifest

    When *validate* is True (default), *model_dir* must resolve under
    ``model_staging/`` or ``model_registry/``. Pass ``validate=False`` only
    for trusted internal paths already checked (e.g. Live env override under
    a nonstandard location — rare).
    """
    import joblib

    if validate:
        model_dir = validate_bundle_path(model_dir)
    else:
        model_dir = Path(model_dir).resolve()
        if not is_serving_bundle(model_dir):
            raise ValueError(
                f"not a serving bundle: {model_dir}")

    schema = json.loads((model_dir / "feature_schema.json").read_text())
    manifest: dict[str, Any] = {}
    if (model_dir / "finalize_manifest.json").exists():
        manifest = json.loads((model_dir / "finalize_manifest.json").read_text())

    if (model_dir / "meta.json").exists() and (model_dir / "meta.pkl").exists():
        return _load_output_meta_bundle(model_dir, schema, manifest)

    if (model_dir / "model.ubj").exists():
        import xgboost as xgb
        booster = xgb.XGBClassifier()
        booster.load_model(str(model_dir / "model.ubj"))
        family = "xgboost"
    elif (model_dir / "model.keras").exists():
        from src.models.keras_layers import load_keras_model
        booster = load_keras_model(model_dir / "model.keras")
        family = str(manifest.get("family") or "keras")
    elif (model_dir / "model.pkl").exists() or (model_dir / "model.joblib").exists():
        p = model_dir / "model.pkl"
        if not p.exists():
            p = model_dir / "model.joblib"
        booster = joblib.load(p)
        family = str(manifest.get("family") or "sklearn")
    else:
        raise FileNotFoundError(f"no model.ubj/keras/pkl/joblib in {model_dir}")

    scaler = (joblib.load(model_dir / "scaler.pkl")
              if (model_dir / "scaler.pkl").exists() else None)
    dir_thr = float(manifest.get("spec_params", {}).get("dir_threshold", 0.5))
    feat_cols = list(schema.get("feature_cols") or schema.get("columns") or [])
    if not feat_cols:
        raise ValueError(f"feature_schema.json has no feature_cols: {model_dir}")

    family_out = manifest.get("family") or family
    kind = manifest.get("kind") or (
        "sequence" if family_out in ("attention_lstm", "tcn", "transformer", "cnn")
        else "tabular")
    return {
        "path": str(model_dir),
        "feature_cols": feat_cols,
        "booster": booster,
        "scaler": scaler,
        "dir_thr": dir_thr,
        "candidate_id": manifest.get("candidate_id") or model_dir.name,
        "family": family_out,
        "kind": kind,
        "seq_len": manifest.get("seq_len") or (manifest.get("spec_params") or {}).get("seq_len"),
        "version": manifest.get("version") or model_dir.name,
        "objective_id": manifest.get("objective_id"),
        "manifest": manifest,
        "inference_policy": _bundle_inference_policy(model_dir, manifest),
    }


def _load_output_meta_bundle(
    model_dir: Path, schema: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    """Load N base artifacts + binary gate (ADR-033)."""
    import joblib

    meta_cfg = json.loads((model_dir / "meta.json").read_text())
    if str(meta_cfg.get("topology") or "") == "tf_oof_meta":
        return _load_tf_oof_meta_bundle(model_dir, schema, manifest, meta_cfg)
    meta_est = joblib.load(model_dir / "meta.pkl")
    bases: list[dict[str, Any]] = []
    for spec in (schema.get("bases") or manifest.get("bases") or []):
        family = spec["family"]
        bdir = model_dir / "bases" / family
        artifact = None
        if (bdir / "model.ubj").exists():
            import xgboost as xgb
            artifact = xgb.XGBClassifier()
            artifact.load_model(str(bdir / "model.ubj"))
        elif (bdir / "model.keras").exists():
            from src.models.keras_layers import load_keras_model
            artifact = load_keras_model(bdir / "model.keras")
        else:
            pkl = bdir / "model.pkl"
            if not pkl.exists():
                pkl = bdir / "model.joblib"
            if pkl.exists():
                artifact = joblib.load(pkl)
        if artifact is None:
            raise FileNotFoundError(f"no model artifact in {bdir}")
        scaler = (joblib.load(bdir / "scaler.pkl")
                  if (bdir / "scaler.pkl").exists() else None)
        cols = list(spec.get("feature_cols") or [])
        if not cols and (bdir / "feature_schema.json").exists():
            cols = json.loads((bdir / "feature_schema.json").read_text()).get("feature_cols") or []
        bases.append({
            "family": family,
            "kind": spec.get("kind") or "tabular",
            "artifact": artifact,
            "scaler": scaler,
            "feature_cols": cols,
            "seq_len": spec.get("seq_len"),
        })
    feat_cols = list(schema.get("feature_cols") or [])
    topo = str(meta_cfg.get("topology") or "")
    fam = str(manifest.get("family") or "")
    is_stack = topo == "meta_stack" or fam == "meta_stack"
    is_primary = topo == "primary_meta" or fam == "primary_meta"
    family_out = (
        "meta_stack" if is_stack else
        "primary_meta" if is_primary else
        "output_meta"
    )
    return {
        "path": str(model_dir),
        "feature_cols": feat_cols,
        "booster": None,
        "scaler": None,
        "dir_thr": 0.5,
        "output_meta": True,
        "meta_stack": is_stack,
        "primary_meta": is_primary,
        "bases": bases,
        "meta_estimator": meta_est,
        "meta": meta_cfg,
        "candidate_id": manifest.get("candidate_id") or model_dir.name,
        "family": family_out,
        "version": manifest.get("version") or model_dir.name,
        "objective_id": manifest.get("objective_id"),
        "manifest": manifest,
        "inference_policy": _bundle_inference_policy(model_dir, manifest),
    }


def _load_base_artifact(bdir: Path) -> Any:
    import joblib

    if (bdir / "model.ubj").exists():
        import xgboost as xgb
        artifact = xgb.XGBClassifier()
        artifact.load_model(str(bdir / "model.ubj"))
        return artifact
    if (bdir / "model.keras").exists():
        from src.models.keras_layers import load_keras_model
        return load_keras_model(bdir / "model.keras")
    pkl = bdir / "model.pkl"
    if not pkl.exists():
        pkl = bdir / "model.joblib"
    if pkl.exists():
        return joblib.load(pkl)
    raise FileNotFoundError(f"no model artifact in {bdir}")


def _load_tf_oof_meta_bundle(
    model_dir: Path,
    schema: dict[str, Any],
    manifest: dict[str, Any],
    meta_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Last-fold 15s/1m/5m bases + dual-binary meta XGB (ADR-100)."""
    import joblib

    raw = joblib.load(model_dir / "meta.pkl")
    if isinstance(raw, dict) and raw.get("estimator") is not None:
        meta_est = raw["estimator"]
        names = list(raw.get("feature_names") or meta_cfg.get("feature_names") or [])
    else:
        meta_est = raw
        names = list(meta_cfg.get("feature_names") or schema.get("meta_feature_names") or [])
    saved = list(schema.get("bases") or manifest.get("bases") or [])
    topo_bases = list(meta_cfg.get("bases") or [])
    by_id = {str(b.get("id") or b.get("family") or ""): b for b in topo_bases}
    bases: list[dict[str, Any]] = []
    for spec in saved:
        bid = str(spec.get("id") or spec.get("family") or "")
        bdir = model_dir / "bases" / bid
        if not bdir.is_dir():
            raise FileNotFoundError(f"tf_oof_meta base missing: {bdir}")
        scaler = (joblib.load(bdir / "scaler.pkl")
                  if (bdir / "scaler.pkl").exists() else None)
        cols = list(spec.get("feature_cols") or [])
        if not cols and (bdir / "feature_schema.json").exists():
            cols = json.loads((bdir / "feature_schema.json").read_text()).get(
                "feature_cols") or []
        topo = by_id.get(bid) or {}
        fam = str(spec.get("plugin_family") or spec.get("family") or
                  topo.get("family") or bid)
        kind = str(spec.get("kind") or (
            "sequence" if fam in ("attention_lstm", "tcn", "transformer", "cnn")
            else "tabular"))
        params = topo.get("params") if isinstance(topo.get("params"), dict) else {}
        bases.append({
            "id": bid,
            "family": fam,
            "native_tf": str(spec.get("native_tf") or topo.get("native_tf") or ""),
            "kind": kind,
            "artifact": _load_base_artifact(bdir),
            "scaler": scaler,
            "feature_cols": cols,
            "seq_len": spec.get("seq_len") or params.get("seq_len"),
        })
    fire = float(meta_cfg.get("fire") if meta_cfg.get("fire") is not None else 0.40)
    decision_tf = str(meta_cfg.get("decision_timestep")
                      or schema.get("decision_timestep") or "15s")
    feat_cols = list(schema.get("feature_cols") or [])
    policy = _bundle_inference_policy(model_dir, manifest)
    return {
        "path": str(model_dir),
        "feature_cols": feat_cols,
        "booster": None,
        "scaler": None,
        "dir_thr": fire,
        "tf_oof_meta": True,
        "output_meta": False,
        "meta_stack": False,
        "primary_meta": False,
        "bases": bases,
        "meta_estimator": meta_est,
        "meta": {
            **meta_cfg,
            "feature_names": names,
            "fire": fire,
            "decision_timestep": decision_tf,
        },
        "candidate_id": manifest.get("candidate_id") or model_dir.name,
        "family": "tf_oof_meta",
        "kind": "tf_oof_meta",
        "version": manifest.get("version") or model_dir.name,
        "objective_id": manifest.get("objective_id"),
        "native_tf": decision_tf,
        "manifest": manifest,
        "inference_policy": policy,
    }


def _bundle_inference_policy(
    model_dir: Path, manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """ADR-047 (or later) serving book: open window, τ, sides, cooldown."""
    pol = manifest.get("inference_policy")
    if isinstance(pol, dict) and pol:
        return pol
    p = Path(model_dir) / "inference_policy.json"
    if p.exists():
        try:
            loaded = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(loaded, dict) and loaded:
            return loaded
    return None
