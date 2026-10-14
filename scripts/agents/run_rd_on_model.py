"""Run the Research Director model-improvement loop on ONE search-produced model.

The scheduled RD (`python -m src.research_director.run`) assembles its Context Pack from the
model-registry LATEST_STABLE pointer. This operator entrypoint instead targets a specific
search/staging model by building the Context Pack from ITS artifacts (finalize_manifest search
score → confusion matrix + per-class precision; diagnostics → raw/gated matrices, pred/truth
distributions, per-class F1, precision@τ sweep; the mined outcome reports) plus an operator
FOCUS directive, then runs the full loop (diagnose → critic → cross-review → synthesis →
auto-validate → handoff).

Usage:
  python scripts/run_rd_on_model.py --objective nq-50pt-tbm \
      --model-dir data/models/model_staging/nq-50pt-tbm/xgboost_20260711T033354Z
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

REPO = Path("/opt/vtech")
log = logging.getLogger("rd.run_on_model")

FOCUS = """\
OPERATOR OPTIMIZATION FOCUS — prioritize these above all when diagnosing and prescribing:
1. MINIMIZE FALSE POSITIVES: the model predicting UP or DOWN when the true label is FLAT. In the
   raw confusion matrix (rows=true down/flat/up, cols=pred) the true-FLAT row's UP and DOWN columns
   are these costly errors. Every prescription must argue its effect on this specific error.
2. MAXIMIZE DIRECTIONAL PRECISION: precision_at_tau on the UP and DOWN signals at the gate
   (objective target: precision_at_tau up/down >= 0.95 @tau 0.9). Do NOT trade precision for a
   flood of low-quality directional signals.
3. FEATURE EFFECTIVENESS + ABLATION: identify which features most drive CORRECT directional calls
   and which are likely false-signal sources. Propose concrete ablation experiments (an
   executable_candidate with a REDUCED feature_selection) that can be trained+scored to test
   whether removing them improves held-out directional precision.
4. FALSE-POSITIVE PATTERNS: from the confusion matrices, pred/truth distributions, precision@tau
   sweep, and the mined outcome patterns (win/loss/missed, direction asymmetry, calibration by
   confidence bucket, session/day effects), diagnose the CONDITIONS under which the large
   flat->directional false positives occur (regime, session, volatility, feature bucket) and aim
   the highest-leverage change at that observed condition.
"""


def _load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def build_pack(objective_id: str, program_id: str, model_dir: Path):
    from src.research_director import tools
    from src.research_director.context_assembler import ContextPack
    from src.objectives.loader import load_objective

    manifest = _load(model_dir / "finalize_manifest.json")
    score = dict(manifest.get("search_score") or {})
    feat_schema = _load(model_dir / "feature_schema.json")
    run_dir = REPO / str(manifest.get("search_run_dir") or "")
    cand = manifest.get("candidate_id") or manifest.get("family") or "model"
    diag = _load(run_dir / f"diagnostics_{cand}.json")

    # training_result: the whitelist keys the Context Pack render + Evaluator understand.
    training_result = {
        "precision_at_tau": score.get("precision_at_tau"),
        "directional_f1": score.get("directional_f1"),
        "confusion_matrix": score.get("confusion_matrix"),
        "test_f1_per_class": diag.get("f1_per_class") or {
            "up": score.get("f1_up"), "down": score.get("f1_down")},
        "n_signals": {"up": score.get("n_up"), "down": score.get("n_down")},
        "precision_up": score.get("precision_up"), "precision_down": score.get("precision_down"),
    }

    # Rich diagnostics that _slim_training_result would drop go into an extras section (rendered
    # verbatim), so the agents SEE the full false-positive evidence.
    diag_block = json.dumps({
        "confusion_matrix_raw": diag.get("confusion_matrix_raw"),
        "confusion_matrix_gated": diag.get("confusion_matrix_gated"),
        "pred_distribution": diag.get("pred_distribution"),
        "truth_distribution": diag.get("truth_distribution"),
        "classification_report": diag.get("classification_report"),
        "precision_at_tau_sweep": diag.get("precision_at_tau_sweep"),
        "trades_at_gate": diag.get("trades_at_gate"),
        "search_score_summary": {k: score.get(k) for k in
                                 ("precision_at_tau", "directional_f1", "n_up", "n_down",
                                  "precision_up", "precision_down", "n_signals")},
    }, indent=1, default=str)
    feat_block = json.dumps(feat_schema, indent=1, default=str)

    try:
        outcome = tools_outcome(objective_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("outcome reports unavailable: %s", exc)
        outcome = []

    try:
        data_cap = tools.data_capability_catalog()
    except Exception:  # noqa: BLE001
        data_cap = None

    return ContextPack(
        program_id=program_id,
        program_objectives=load_objective(objective_id),
        baseline={"version": manifest.get("version"),
                  "config_snapshot": {k: v for k, v in manifest.items() if k != "search_score"},
                  "features": feat_schema, "training_result": training_result},
        recent_journal=_safe(lambda: tools.load_recent_journal(objective_id), []),
        feature_catalog=_safe(lambda: tools.read_reference("docs/reference/FEATURE_CATALOG.md"), ""),
        backlog=_safe(lambda: tools.read_reference("docs/roadmap/BACKLOG.md"), ""),
        data_capability=data_cap,
        outcome_patterns=outcome,
        extras={
            "OPTIMIZATION FOCUS (operator directive)": FOCUS,
            "MODEL DIAGNOSTICS (confusion matrices, pred/truth dist, class report, precision@τ sweep)":
                diag_block,
            "MODEL FEATURE SCHEMA (candidates for ablation)": feat_block,
        },
    )


def tools_outcome(objective_id: str) -> list:
    from src.evaluation.episodes import latest_outcome_reports
    return latest_outcome_reports(objective_id, limit=3)


def _safe(fn, default):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Research Director on one search/staging model")
    ap.add_argument("--objective", required=True)
    ap.add_argument("--program", default=None, help="program_id (defaults to --objective)")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--self-critique", type=int, default=1)
    ap.add_argument("--critic-rounds", type=int, default=2)
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the auto-validation training run (diagnosis + plan only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    program_id = args.program or args.objective
    from src.research_director.orchestrator import load_rd_cfg, run_rd_loop

    cfg = load_rd_cfg()
    cfg["caps"]["self_critique"] = args.self_critique
    cfg["caps"]["critic_rounds"] = args.critic_rounds
    cfg["registry_dir"] = "data/models/model_registry"
    cfg["model_name"] = args.objective
    cfg["objective_id"] = args.objective
    if args.no_validate:
        cfg["skip_validate"] = True   # honored by the orchestrator's validate stage guard

    pack = build_pack(args.objective, program_id, REPO / args.model_dir)
    state = run_rd_loop(program_id, args.objective, pack, cfg=cfg)
    run_ts = state["run_ts"]
    out = REPO / "data" / "research_director" / program_id / run_ts
    print(f"\nRD run complete → {out}")
    print(f"  plan   : {out / 'model_improvement_plan.json'}")
    print(f"  report : {out / 'improvement_report.md'}")
    print(f"  verdict: {(state.get('validation') or {}).get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
